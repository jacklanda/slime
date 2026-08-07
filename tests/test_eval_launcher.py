from pathlib import Path

import pytest


@pytest.mark.unit
def test_qwen3_long_context_eval_enables_yarn_override():
    launcher = (Path(__file__).resolve().parents[1] / "experiments" / "evals.sh").read_text(encoding="utf-8")

    assert 'ENABLE_YARN="${ENABLE_YARN:-auto}"' in launcher
    assert '[ "${MODEL_SERIES}" = "qwen3" ] && [ "${EVAL_MAX_CONTEXT_LEN}" -gt 40960 ]' in launcher
    assert '--use-yarn-rope' in launcher
    assert '--yarn-rope-scaling-factor "${YARN_FACTOR}"' in launcher
    assert '--yarn-original-max-position-embeddings "${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}"' in launcher


@pytest.mark.unit
def test_gemma4_eval_accepts_series_alias_and_e4b_config():
    launcher = (Path(__file__).resolve().parents[1] / "experiments" / "evals.sh").read_text(encoding="utf-8")

    assert "qwen3|qwen3.5|gemma4|gemma-4|gemma-4-*" in launcher
    assert 'gemma-4|gemma-4-*) MODEL_SERIES="gemma4"' in launcher
    assert 'gemma-4-e4b|gemma4-e4b) MODEL_CONFIG="gemma4-E4B"' in launcher
    assert 'normalized_type == "gemma4"' in launcher
