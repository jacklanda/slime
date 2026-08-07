import importlib
import json
import multiprocessing
import pickle
import warnings

from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from slime.utils.hf_config import prepare_gemma4_checkpoint_compat, register_gemma4_config_aliases


def _check_configs_in_spawn(configs, result_queue):
    result_queue.put([(type(config).__name__, config.model_type) for config in configs])


def test_gemma4_alias_configs_are_picklable_for_spawn():
    register_gemma4_config_aliases()
    configs = []

    for model_type in ("gemma4_unified", "gemma4_unified_text"):
        config_type = CONFIG_MAPPING[model_type]
        module = importlib.import_module(config_type.__module__)
        assert getattr(module, config_type.__name__) is config_type
        assert pickle.loads(pickle.dumps(config_type)) is config_type
        config = config_type()
        assert config.model_type == model_type
        config_with_architecture = config_type.from_dict(
            {"model_type": model_type, "architectures": ["Gemma4UnifiedForConditionalGeneration"]}
        )
        assert config_with_architecture.architectures == ["Gemma4ForConditionalGeneration"]
        if model_type == "gemma4_unified":
            assert config_with_architecture.text_config.model_type == "gemma4_text"
        restored = pickle.loads(pickle.dumps(config_with_architecture))
        assert type(restored) is config_type
        assert restored.model_type == "gemma4"
        configs.append(config_with_architecture)

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_check_configs_in_spawn, args=(configs, result_queue))
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 0
    assert result_queue.get(timeout=5) == [
        ("Gemma4UnifiedConfig", "gemma4"),
        ("Gemma4UnifiedTextConfig", "gemma4"),
    ]


def test_gemma4_checkpoint_alias_does_not_emit_model_type_warning(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model_type": "gemma4_unified", '
        '"architectures": ["Gemma4UnifiedForConditionalGeneration"], '
        '"text_config": {"model_type": "gemma4_unified_text"}}'
    )

    from transformers import AutoConfig

    register_gemma4_config_aliases()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = AutoConfig.from_pretrained(tmp_path)

    assert config.model_type == "gemma4"
    assert config.text_config.model_type == "gemma4_text"
    assert not any("You are using a model of type" in str(w.message) for w in caught)


def test_prepare_gemma4_checkpoint_compat_rewrites_metadata_and_links_weights(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_unified",
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "eoa_token_id": None,
                "text_config": {"model_type": "gemma4_unified_text"},
                "vision_config": {"model_type": "gemma4_unified_vision"},
                "audio_config": {"model_type": "gemma4_unified_audio"},
            }
        )
    )
    (source / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gemma4UnifiedProcessor",
                "feature_extractor": {"feature_extractor_type": "Gemma4UnifiedAudioFeatureExtractor"},
                "image_processor": {"image_processor_type": "Gemma4UnifiedImageProcessor"},
                "video_processor": {"video_processor_type": "Gemma4UnifiedVideoProcessor"},
            }
        )
    )
    (source / "tokenizer_config.json").write_text(
        json.dumps({"processor_class": "Gemma4UnifiedProcessor"})
    )
    weights = source / "model.safetensors"
    weights.write_bytes(b"weights")

    assert prepare_gemma4_checkpoint_compat(source, target) == target.resolve()

    config = json.loads((target / "config.json").read_text())
    processor_config = json.loads((target / "processor_config.json").read_text())
    tokenizer_config = json.loads((target / "tokenizer_config.json").read_text())
    assert config["model_type"] == "gemma4"
    assert config["architectures"] == ["Gemma4ForConditionalGeneration"]
    assert config["text_config"]["model_type"] == "gemma4_text"
    assert config["vision_config"]["model_type"] == "gemma4_vision"
    assert config["audio_config"]["model_type"] == "gemma4_audio"
    assert config["eoa_token_id"] == 258883
    assert processor_config["processor_class"] == "Gemma4Processor"
    assert processor_config["feature_extractor"]["feature_extractor_type"] == "Gemma4AudioFeatureExtractor"
    assert processor_config["image_processor"]["image_processor_type"] == "Gemma4ImageProcessor"
    assert processor_config["video_processor"]["video_processor_type"] == "Gemma4VideoProcessor"
    assert tokenizer_config["processor_class"] == "Gemma4Processor"
    assert (target / "model.safetensors").is_symlink()
    assert (target / "model.safetensors").resolve() == weights.resolve()
