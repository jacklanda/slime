from pathlib import Path

import pytest


NUM_GPUS = 0


@pytest.mark.unit
def test_gemma4_e4b_sync_launcher_uses_e4b_model_shape():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_gemma4_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert "--num-layers 42" in launcher
    assert "--hidden-size 2560" in launcher
    assert "--ffn-hidden-size 10240" in launcher
    assert "--num-query-groups 2" in launcher
    assert "--moe-token-dispatcher-type alltoall" in launcher
    assert "--no-pin-cpu-params" in launcher
    assert "--no-pin-cpu-grads" in launcher
    assert "--overlap-cpu-optimizer-d2h-h2d" not in launcher
    assert 'CHECK_WEIGHT_UPDATE_EQUAL:-0' in launcher
    assert 'SLIME_TENSOR_BACKUP_PIN_MEMORY:-0' in launcher
    assert 'os.path.join(sys.prefix, "lib"), os.environ.get("LD_LIBRARY_PATH")' in launcher
    assert 'TOP_P="${TOP_P:-1.0}"' in launcher
    assert 'TOP_K="${TOP_K:-64}"' in launcher
    assert '--rollout-top-p "${TOP_P}"' in launcher
    assert '--rollout-top-k "${TOP_K}"' in launcher
    assert 'SGLANG_DETERMINISTIC_INFERENCE="${SGLANG_DETERMINISTIC_INFERENCE:-true}"' in launcher
    assert 'if is_truthy "${SGLANG_DETERMINISTIC_INFERENCE}"; then' in launcher
    assert 'CP_SIZE="${CP_SIZE:-2}"' in launcher
    assert 'MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-20480}"' in launcher
    assert 'LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-20480}"' in launcher
    assert 'LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-4096}"' in launcher
    assert 'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"' in launcher
    assert "--accumulate-allreduce-grads-in-fp32" in launcher
    assert "--grad-reduce-in-bf16" not in launcher
    assert 'export FUSED_MODEL_SERIES="gemma4"' in launcher
    assert "export SLIME_FUSED_STRICT_TITO=1" in launcher
    # Gemma4 rollout uses SGLang batch-invariant kernels under deterministic
    # inference; the trainer receives the matching Gemma4-only runtime flag.
    assert "export SLIME_GEMMA4_BATCH_INVARIANT=1" in launcher
    assert '"SLIME_GEMMA4_BATCH_INVARIANT"' in launcher
    assert "export SLIME_GEMMA4_LOGPROB_BF16=1" in launcher
    # Keep SGLang's Gemma4 batch-invariant GEMM on the same persistent Triton
    # implementation used by Megatron; DeepGEMM has a different reduction path.
    assert "export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0" in launcher
    assert '"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"' in launcher
    assert 'export FUSED_WEBQA_REWARD_MATCH_MODE="normalized_target_span"' in launcher


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
    assert 'CP_SIZE="${CP_SIZE:-2}"' in launcher
    assert 'MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-20480}"' in launcher
    assert 'LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-20480}"' in launcher
    assert 'LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"' in launcher


@pytest.mark.unit
@pytest.mark.parametrize(
    "launcher_name",
    [
        "train_qwen3_fused_agent_sync.sh",
        "train_qwen3.5_fused_agent_sync.sh",
    ],
)
def test_qwen_sync_launchers_do_not_enable_reference_model_by_default(launcher_name):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / f"experiments/{launcher_name}").read_text(encoding="utf-8")

    assert 'KL_COEF="${KL_COEF:-0.0}"' in launcher
    assert 'KL_LOSS_COEF="${KL_LOSS_COEF:-0.00}"' in launcher
    assert 'USE_KL_LOSS="${USE_KL_LOSS:-0}"' in launcher
    assert 'if is_truthy "${USE_KL_LOSS}"; then' in launcher
    assert 'USE_KL_LOSS:-1' not in launcher


@pytest.mark.unit
def test_qwen35_sync_launcher_does_not_reset_unsynced_visual_weights_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_qwen3.5_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert 'if is_truthy "${CHECK_WEIGHT_UPDATE_EQUAL:-0}"; then' in launcher
    assert "MISC_ARGS+=(--check-weight-update-equal)" in launcher


@pytest.mark.unit
def test_qwen35_sync_launcher_avoids_train_memory_saver_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_qwen3.5_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert 'COLOCATE="${COLOCATE:-true}"' in launcher
    assert 'RELEASE_TRAIN="${RELEASE_TRAIN:-true}"' in launcher
    assert 'OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-${COLOCATE}}"' in launcher
    assert 'OFFLOAD_TRAIN=false' in launcher
    assert '--update-weight-mode full' in launcher
    assert '--update-weight-transport disk' in launcher
    assert 'SGLANG_MEM_FRACTION_STATIC=0.6' in launcher
    assert '--rollout-top-p "${TOP_P:-1.0}"' in launcher
    assert '--rollout-presence-penalty "${PRESENCE_PENALTY:-0.0}"' in launcher
    assert 'ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"' in launcher
    assert 'ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"' in launcher


@pytest.mark.unit
def test_qwen35_sync_launcher_defaults_to_cp2_memory_budget():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_qwen3.5_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert 'CP_SIZE="${CP_SIZE:-2}"' in launcher
    assert 'MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-20480}"' in launcher
    assert 'LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-20480}"' in launcher
    assert 'LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"' in launcher


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
