import argparse
from pathlib import Path

import pytest

from slime.backends.sglang_utils.arguments import add_sglang_arguments


NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_patterns", "expected_tp"),
    [
        ("qwen3-4b|qwen3-4b-*|qwen3-8b|qwen3-8b-*", 1),
        ("qwen3-14b|qwen3-14b-*", 2),
        ("qwen3-30b-a3b|qwen3-30b-a3b-*|qwen3-32b|qwen3-32b-*", 4),
    ],
)
@pytest.mark.parametrize(
    "launcher_name",
    ["train_qwen3_fused_agent_sync.sh", "run_qwen3_rejection_sampling.sh"],
)
def test_qwen3_launchers_choose_rollout_tp_by_model(launcher_name, model_patterns, expected_tp):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments" / launcher_name).read_text(encoding="utf-8")

    assert f"{model_patterns})" in launcher
    assert f"DEFAULT_ROLLOUT_NUM_GPUS_PER_ENGINE={expected_tp}" in launcher
    assert (
        'ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-'
        '${DEFAULT_ROLLOUT_NUM_GPUS_PER_ENGINE}}"'
    ) in launcher


@pytest.mark.unit
def test_qwen3_rejection_sampling_defaults_to_adaptive_fully_async_task_filling():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/run_qwen3_rejection_sampling.sh").read_text(encoding="utf-8")

    assert (
        'ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-'
        'slime.rollout.fully_async_rollout.generate_rollout_fully_async}"'
    ) in launcher
    assert 'FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY:-true}"' in launcher
    assert 'FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY="${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY:-$((ROLLOUT_ENGINE_COUNT * 4))}"' in launcher
    assert 'FULLY_ASYNC_MAX_GROUP_CONCURRENCY="${FULLY_ASYNC_MAX_GROUP_CONCURRENCY:-$((ROLLOUT_ENGINE_COUNT * 8))}"' in launcher
    assert 'export SLIME_FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY}"' in launcher
    assert 'ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2048}"' in launcher
    assert 'SAMPLE_N="${SAMPLE_N:-32}"' in launcher
    assert 'export SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS="${SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS:-true}"' in launcher


@pytest.mark.unit
def test_qwen3_rejection_sampling_uses_inference_only_fast_path():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/run_qwen3_rejection_sampling.sh").read_text(encoding="utf-8")

    assert "--debug-rollout-only" in launcher
    assert "--rollout-only-inference-fast-path" in launcher
    assert "--rollout-only-skip-episode-dump" in launcher
    assert "--async-save-debug-rollout-data" in launcher
    assert "--sglang-enable-deterministic-inference" not in launcher
    assert "--sglang-disable-piecewise-cuda-graph" not in launcher
    assert "--log-probs-max-tokens-per-gpu" not in launcher
    assert "--global-batch-size" not in launcher
    assert "--num-steps-per-rollout" not in launcher
    assert "--actor-num-nodes" not in launcher
    assert "--actor-num-gpus-per-node" not in launcher
    assert "--update-weights-interval" not in launcher
    assert "--micro-batch-size" not in launcher

    for training_launcher in (
        "train_odyssey_qwen3_multinode_sync.sh",
        "train_gemma4_fused_agent_async.sh",
    ):
        training = (repo_root / "experiments" / training_launcher).read_text(encoding="utf-8")
        assert "--rollout-only-inference-fast-path" not in training
        assert "--rollout-only-skip-episode-dump" not in training
        assert "--async-save-debug-rollout-data" not in training


@pytest.mark.unit
@pytest.mark.parametrize("launcher_name", ["train_qwen3_fused_agent_sync.sh", "train_odyssey_qwen3_multinode_sync.sh"])
def test_qwen3_training_launchers_default_to_adaptive_fully_async(launcher_name):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments" / launcher_name).read_text(encoding="utf-8")
    assert (
        'ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-'
        'slime.rollout.fully_async_rollout.generate_rollout_fully_async}"'
    ) in launcher
    assert 'FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY:-true}"' in launcher
    assert 'export SLIME_FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY}"' in launcher


@pytest.mark.unit
def test_qwen3_rejection_sampling_uses_final_search_and_mcp_training_data():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/run_qwen3_rejection_sampling.sh").read_text(encoding="utf-8")

    assert '"${SCRIPT_DIR}/artifacts/search_data_final/train.parquet"' in launcher
    assert '"${SCRIPT_DIR}/artifacts/mcp_data_final/train.parquet"' in launcher
    assert "fused_mcp_search_train_shuffled.parquet" not in launcher


@pytest.mark.unit
def test_sglang_accepts_megatron_on_policy_target():
    parser = argparse.ArgumentParser()
    add_sglang_arguments(parser)

    args = parser.parse_args(["--sglang-rl-on-policy-target", "megatron"])

    assert args.sglang_rl_on_policy_target == "megatron"


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
    assert "export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=0" in launcher
    assert '"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT"' in launcher
    assert "--sglang-rl-on-policy-target megatron" in launcher
    assert 'export FUSED_WEBQA_REWARD_MATCH_MODE="normalized_target_span"' in launcher


@pytest.mark.unit
def test_gemma4_launcher_uses_trajectory_mean_grpo_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_gemma4_fused_agent_sync.sh").read_text(encoding="utf-8")

    # Token-sum reduction makes a sequence-level GRPO advantage proportional to
    # response length. Gemma4 must opt into that legacy behavior explicitly.
    assert 'CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-false}"' in launcher
    perf_start = launcher.index("PERF_ARGS=(")
    perf_end = launcher.index("if [ \"${USE_DYNAMIC_BATCH_SIZE:-1}\"", perf_start)
    perf_args = launcher[perf_start:perf_end]
    assert "\n   --calculate-per-token-loss\n" not in perf_args
    assert 'if is_truthy "${CALCULATE_PER_TOKEN_LOSS}"; then' in launcher
    assert "PERF_ARGS+=(--calculate-per-token-loss)" in launcher
    assert "default false = trajectory/sample mean" in launcher


@pytest.mark.unit
def test_gemma4_launcher_defaults_to_even_mcp_webqa_groups():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_gemma4_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert 'ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"' in launcher
    assert 'ROLLOUT_TASK_FAMILY_QUOTAS="${ROLLOUT_TASK_FAMILY_QUOTAS:-mcp=0.5,webqa=0.5}"' in launcher
    assert 'ROLLOUT_ARGS+=(--rollout-task-family-quotas "${ROLLOUT_TASK_FAMILY_QUOTAS}")' in launcher


@pytest.mark.unit
def test_gemma4_launcher_starts_a_new_wandb_run_for_each_attempt_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_gemma4_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert 'WANDB_RESUME_SAME_RUN="${WANDB_RESUME_SAME_RUN:-0}"' in launcher
    assert '[ -n "${WANDB_RUN_ID:-}" ] && is_truthy "${WANDB_RESUME_SAME_RUN}"' in launcher
    assert "unset WANDB_RUN_ID" in launcher


@pytest.mark.unit
def test_gemma4_launcher_avoids_train_memory_saver_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_gemma4_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert 'COLOCATE="${COLOCATE:-true}"' in launcher
    assert 'RELEASE_TRAIN="${RELEASE_TRAIN:-true}"' in launcher
    assert 'OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-${COLOCATE}}"' in launcher
    assert "OFFLOAD_TRAIN=false" in launcher
    assert "--update-weight-mode full" in launcher
    assert "--update-weight-transport disk" in launcher


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
def test_odyssey_launcher_enforces_strict_dynamic_sampling():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_odyssey_qwen3_multinode_sync.sh").read_text(encoding="utf-8")

    assert (
        'DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"'
        in launcher
    )
    assert "FULLY_ASYNC_FILTER_RELAX_AFTER_GROUPS=0" in launcher
    assert launcher.count("--fully-async-filter-relax-after-groups") == 1
    assert "--fully-async-filter-relax-after-groups 0" in launcher
    assert "This launcher requires strict dynamic sampling" in launcher
    assert '--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}"' in launcher


@pytest.mark.unit
def test_launchers_provision_per_trajectory_mcp_workspaces():
    repo_root = Path(__file__).resolve().parents[1]
    rejection = (repo_root / "experiments/run_qwen3_rejection_sampling.sh").read_text(encoding="utf-8")
    odyssey = (repo_root / "experiments/train_odyssey_qwen3_multinode_sync.sh").read_text(encoding="utf-8")

    assert 'MCP_ENV_ROOT="${MCP_ENV_ROOT:-${RUN_ROOT}/cache/mcp_envs}"' in rejection
    assert 'export SLIME_MCP_ENV_ROOT="${MCP_ENV_ROOT}"' in rejection
    assert 'MCP_ENV_ROOT="${MCP_ENV_ROOT:-${RUN_ROOT}/cache/mcp_envs}"' in odyssey
    assert 'export SLIME_MCP_ENV_ROOT="${MCP_ENV_ROOT}"' in odyssey
    assert 'SLIME_MCP_ENV_COPY_CONCURRENCY' in rejection
    assert 'SLIME_MCP_ENV_COPY_CONCURRENCY' in odyssey


@pytest.mark.unit
@pytest.mark.parametrize(
    "launcher_name",
    [
        "train_gemma4_fused_agent_async.sh",
        "train_gemma4_fused_agent_sync.sh",
        "train_odyssey_qwen3_multinode_sync.sh",
        "train_qwen3.5_fused_agent_async.sh",
        "train_qwen3.5_fused_agent_sync.sh",
        "train_qwen3_fused_agent_async.sh",
        "train_qwen3_fused_agent_gspo_async.sh",
        "train_qwen3_fused_agent_sync.sh",
        "train_qwen3_swe_agent_sync.sh",
    ],
)
def test_training_launchers_enable_and_propagate_local_mcp_process_isolation(launcher_name):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments" / launcher_name).read_text(encoding="utf-8")

    assert 'MCP_ENV_ROOT="${MCP_ENV_ROOT:-${RUN_ROOT}/cache/mcp_envs}"' in launcher
    assert 'SLIME_LOCAL_MCP_PROCESS_ISOLATION="${SLIME_LOCAL_MCP_PROCESS_ISOLATION:-true}"' in launcher
    assert 'SLIME_LOCAL_MCP_PROCESS_START_METHOD="${SLIME_LOCAL_MCP_PROCESS_START_METHOD:-forkserver}"' in launcher
    for key in (
        "SLIME_MCP_ENV_ROOT",
        "SLIME_MCP_ENV_COPY_CONCURRENCY",
        "SLIME_LOCAL_MCP_PROCESS_ISOLATION",
        "SLIME_LOCAL_MCP_PROCESS_WORKERS",
        "SLIME_LOCAL_MCP_PROCESS_START_METHOD",
        "SLIME_LOCAL_MCP_PROCESS_TIMEOUT",
        "SLIME_LOCAL_MCP_PROCESS_WARM_TIMEOUT",
    ):
        assert launcher.count(key) >= 2, f"{key} is not propagated through Ray runtime env"


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
    assert "--sglang-rl-on-policy-target megatron" not in launcher
    assert "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT" not in launcher


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
