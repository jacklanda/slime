from pathlib import Path

import pytest


NUM_GPUS = 0


@pytest.mark.unit
def test_qwen3_sync_launcher_defaults_to_colocated_trainer_offload():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_qwen3_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert 'COLOCATE="${COLOCATE:-true}"' in launcher
    assert 'RELEASE_TRAIN="${RELEASE_TRAIN:-false}"' in launcher
    assert 'OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-${COLOCATE}}"' in launcher
    assert 'CLUSTER_ARGS+=(--colocate)' in launcher
    assert '--release-train' in launcher
    assert '--update-weight-transport disk' in launcher
    assert '--save-interval "${SAVE_INTERVAL:-20}"' in launcher


@pytest.mark.unit
def test_qwen35_sync_launcher_does_not_reset_unsynced_visual_weights_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_qwen3.5_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert 'if is_truthy "${CHECK_WEIGHT_UPDATE_EQUAL:-0}"; then' in launcher
    assert "MISC_ARGS+=(--check-weight-update-equal)" in launcher


@pytest.mark.unit
def test_qwen3_sync_launcher_resolves_and_validates_grm_credentials():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_qwen3_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert '--grm-openrouter-api-key)' in launcher
    assert 'export OPENROUTER_API_KEY="${2:?Missing value for --grm-openrouter-api-key}"' in launcher
    assert '[ -n "${GRM_BASE_URL:-${OPENAI_BASE_URL:-}}" ]' in launcher
    assert 'export OPENROUTER_API_KEY="${OPENAI_API_KEY}"' in launcher
    assert 'GRM evals require OPENROUTER_API_KEY' in launcher
    runtime_env_offset = launcher.index("RUNTIME_ENV_JSON=$(python3")
    ray_submit_offset = launcher.index('ray job submit --address="${RAY_DASHBOARD_ADDRESS}"')
    assert launcher.rfind("set +x", 0, runtime_env_offset) > launcher.index("export SLIME_FUSED_QUOTA")
    assert launcher.index("set -x", ray_submit_offset) > ray_submit_offset


@pytest.mark.unit
def test_qwen3_sync_launcher_applies_yarn_to_actor_and_rollout():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_qwen3_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert 'ENABLE_YARN="${ENABLE_YARN:-true}"' in launcher
    assert 'MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"' in launcher
    assert '--use-yarn-rope' in launcher
    assert '--yarn-rope-scaling-factor "${YARN_FACTOR}"' in launcher
    assert '--yarn-original-max-position-embeddings "${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}"' in launcher
    assert "YARN_MODEL_OVERRIDE" not in launcher
