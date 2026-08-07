"""Small Hugging Face config compatibility helpers.

Keep this module free of Megatron and torch imports: rollout workers need to
register model config aliases before SGLang constructs ``ServerArgs``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


_GEMMA4_METADATA_RENAMES = {
    "gemma4_unified": "gemma4",
    "gemma4_unified_text": "gemma4_text",
    "gemma4_unified_audio": "gemma4_audio",
    "gemma4_unified_vision": "gemma4_vision",
    "Gemma4UnifiedForConditionalGeneration": "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedProcessor": "Gemma4Processor",
    "Gemma4UnifiedAudioFeatureExtractor": "Gemma4AudioFeatureExtractor",
    "Gemma4UnifiedImageProcessor": "Gemma4ImageProcessor",
    "Gemma4UnifiedVideoProcessor": "Gemma4VideoProcessor",
}


def _normalize_gemma4_metadata(value):
    if isinstance(value, dict):
        return {key: _normalize_gemma4_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_gemma4_metadata(item) for item in value]
    if isinstance(value, str):
        return _GEMMA4_METADATA_RENAMES.get(value, value)
    return value


def prepare_gemma4_checkpoint_compat(source: str | Path, target: str | Path) -> Path:
    """Create a native Gemma-4 metadata view backed by the original weights."""

    source = Path(source).resolve()
    target = Path(target).resolve()
    with (source / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if config.get("model_type") != "gemma4_unified":
        return source

    target.mkdir(parents=True, exist_ok=True)
    rewritten_files = {"config.json", "processor_config.json", "tokenizer_config.json"}
    for entry in source.iterdir():
        if entry.name in rewritten_files:
            continue
        link = target / entry.name
        if link.is_symlink() and link.resolve() == entry.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Gemma-4 compatibility path already exists with a different target: {link}")
        link.symlink_to(entry, target_is_directory=entry.is_dir())

    for filename in rewritten_files:
        source_file = source / filename
        if not source_file.exists():
            continue
        with source_file.open(encoding="utf-8") as metadata_file:
            metadata = _normalize_gemma4_metadata(json.load(metadata_file))
        if filename == "config.json" and metadata.get("eoa_token_id") is None:
            metadata["eoa_token_id"] = metadata.get("eoa_token_index", 258883)
        output_file = target / filename
        temporary_file = output_file.with_suffix(f"{output_file.suffix}.tmp.{os.getpid()}")
        with temporary_file.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=True, indent=2)
            metadata_file.write("\n")
        temporary_file.replace(output_file)

    return target


class _Gemma4AliasConfigMixin:
    """Normalize transitional Gemma-4 config names during deserialization."""

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # Transformers validates the serialized ``model_type`` in the base
        # implementation before returning the config.  Normalize transitional
        # Gemma-4 names first so that this compatibility alias does not emit a
        # misleading "model type ... to instantiate ..." warning (and nested
        # text configs do not retain the unsupported type).
        config_dict = dict(config_dict)
        config_dict["model_type"] = cls.model_type
        architectures = config_dict.get("architectures")
        if architectures:
            config_dict["architectures"] = [
                architecture.replace("Gemma4Unified", "Gemma4")
                for architecture in architectures
            ]
        text_config = config_dict.get("text_config")
        if isinstance(text_config, dict):
            text_config = dict(text_config)
            if text_config.get("model_type") == "gemma4_unified_text":
                text_config["model_type"] = "gemma4_text"
            architectures = text_config.get("architectures")
            if architectures:
                text_config["architectures"] = [
                    architecture.replace("Gemma4Unified", "Gemma4")
                    for architecture in architectures
                ]
            config_dict["text_config"] = text_config

        config = super().from_dict(config_dict, **kwargs)
        config.model_type = "gemma4"
        architectures = getattr(config, "architectures", None)
        if architectures:
            config.architectures = [
                architecture.replace("Gemma4Unified", "Gemma4") for architecture in architectures
            ]
        text_config = getattr(config, "text_config", None)
        if getattr(text_config, "model_type", None) == "gemma4_unified_text":
            text_config.model_type = "gemma4_text"
        return config


_GEMMA4_ALIAS_TYPES = {
    "Gemma4UnifiedConfig": "gemma4_unified",
    "Gemma4UnifiedTextConfig": "gemma4_unified_text",
}


def __getattr__(name: str):
    """Lazily expose aliases while a spawned process unpickles a config."""
    model_type = _GEMMA4_ALIAS_TYPES.get(name)
    if model_type is None:
        raise AttributeError(name)
    register_gemma4_config_aliases()
    try:
        return globals()[name]
    except KeyError as exc:
        raise AttributeError(name) from exc


def register_gemma4_config_aliases() -> None:
    """Make unified Gemma 4 checkpoints load with native Transformers/SGLang.

    Some Gemma 4 checkpoints use the transitional ``gemma4_unified`` model
    type, while the released Transformers implementation registers the same
    architecture as ``gemma4``.  Register subclasses so AutoConfig accepts
    the checkpoint type while SGLang still receives a real ``Gemma4Config``.
    """

    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4Config,
            Gemma4TextConfig,
        )
    except (ImportError, ModuleNotFoundError):
        # Older Transformers versions cannot serve Gemma 4 in SGLang anyway.
        return

    for alias_name, model_type, config_base in (
        ("Gemma4UnifiedConfig", "gemma4_unified", Gemma4Config),
        ("Gemma4UnifiedTextConfig", "gemma4_unified_text", Gemma4TextConfig),
    ):
        if model_type in CONFIG_MAPPING:
            alias_config = CONFIG_MAPPING[model_type]
            if alias_config.__name__ == alias_name and alias_config.__module__ == __name__:
                globals()[alias_name] = alias_config
            continue

        # Register directly in the lazy mapping because AutoConfig.register
        # requires ``config.model_type == model_type``.  The native SGLang
        # implementation recognizes ``gemma4`` (and the config subclasses are
        # otherwise fully compatible), so normalize the alias on the object.
        alias_config = type(
            alias_name,
            (_Gemma4AliasConfigMixin, config_base),
            # Keep the serialized alias on the class so Transformers' own
            # ``from_pretrained`` model-type check accepts the checkpoint.
            # ``from_dict`` above canonicalizes the resulting instance for
            # SGLang and the rest of Slime.
            {"__module__": __name__, "model_type": model_type},
        )
        # Expose the generated class under its qualified name so
        # multiprocessing/pickle can resolve it on spawn.
        globals()[alias_config.__name__] = alias_config
        CONFIG_MAPPING.register(model_type, alias_config)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Hugging Face checkpoint compatibility metadata.")
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    print(prepare_gemma4_checkpoint_compat(args.source, args.target))


if __name__ == "__main__":
    _main()
