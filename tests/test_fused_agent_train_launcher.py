import argparse
import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

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
    assert 'ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4096}"' in launcher
    assert 'SAMPLE_N="${SAMPLE_N:-32}"' in launcher
    assert 'export SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS="${SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS:-true}"' in launcher
    assert 'export SLIME_FULLY_ASYNC_CROSS_SHARD_PREFETCH="${SLIME_FULLY_ASYNC_CROSS_SHARD_PREFETCH:-true}"' in launcher
    assert "export SLIME_FULLY_ASYNC_VALID_GROUPS_PER_SHARD" not in launcher
    assert "export SLIME_FULLY_ASYNC_MAX_CANDIDATE_GROUPS_PER_SHARD" not in launcher


@pytest.mark.unit
def test_qwen3_rejection_sampling_defers_episode_shards_until_final_merge():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/run_qwen3_rejection_sampling.sh").read_text(encoding="utf-8")

    assert "start_episode_watcher" not in launcher
    assert "EPISODE_WATCHER_PID" not in launcher
    assert 'rollout_path = rollout_dir / f"{next_rollout_id}.pt"' not in launcher
    assert 'episode_path = episodes_dir / f"global_steps_{rollout_id}.json"' in launcher
    assert 'export SLIME_TOOL_PARSER_ERROR_LOG_ENABLED="${SLIME_TOOL_PARSER_ERROR_LOG_ENABLED:-false}"' in launcher


@pytest.mark.unit
def test_qwen3_rejection_sampling_caches_prepared_parquet_and_keeps_checkpoint_bounded():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/run_qwen3_rejection_sampling.sh").read_text(encoding="utf-8")

    assert 'fingerprint_path = output.with_suffix(output.suffix + ".fingerprint.json")' in launcher
    assert 'print(f"Reusing cached fused train parquet: {output}")' in launcher
    assert '"completed_shards": batch_summaries' in launcher
    assert '"completed_rollout_ids": completed' not in launcher
    assert '"accepted_problem_ids": sorted(accepted_problem_ids)' not in launcher
    assert '"rejected_problem_ids": sorted(rejected_problem_ids)' not in launcher


def _rs_episode(problem_id, trajectory_id, reward, num_steps):
    timing = {"total_time_s": 0.1}
    return {
        "id": f"{problem_id}:{trajectory_id}",
        "session_id": f"session-{problem_id}-{trajectory_id}",
        "task": {"question": f"question-{problem_id}", "data_source": "test"},
        "termination_reason": "env_done",
        "metrics": {"test/pass@1": float(reward > 0), "traj/steps": float(num_steps)},
        "metadata": {"timing": timing},
        "info": {"timing": timing},
        "trajectories": [
            {
                "name": "test_0",
                "uid": f"uid-{problem_id}-{trajectory_id}",
                "reward": reward,
                "info": {"timing": timing},
                "steps": [
                    {
                        "observation": f"observation-{step}",
                        "thought": "",
                        "action": f"action-{step}",
                        "reward": reward if step == num_steps - 1 else 0.0,
                        "done": step == num_steps - 1,
                        "model_response": f"response-{step}",
                        "info": {"timing": timing},
                    }
                    for step in range(num_steps)
                ],
            }
        ],
    }


def _rs_sample(problem_id, trajectory_id, group_index, index, reward, num_steps):
    return {
        "group_index": group_index,
        "index": index,
        "rollout_id": 0,
        "prompt": f"prompt-{problem_id}",
        "response": f"response-{problem_id}-{trajectory_id}",
        "reward": reward,
        "status": "completed",
        "metadata": {
            "instance_id": problem_id,
            "fused_task_type": "mcp",
            "fused_traj_steps": num_steps,
            "fused_termination": "env_done",
            "fused_profile": {"llm_time_s": 0.05},
            "rllm_episode": _rs_episode(problem_id, trajectory_id, reward, num_steps),
        },
    }


def _prepare_rs_workflow_fixture(tmp_path):
    train_a = tmp_path / "train-a.parquet"
    train_b = tmp_path / "train-b.parquet"
    pq.write_table(pa.table({"prompt": ["input-a"], "source": ["a"]}), train_a)
    pq.write_table(pa.table({"prompt": ["input-b"], "difficulty": [2]}), train_b)

    output_dir = tmp_path / "output"
    rollout_dir = output_dir / "debug" / "rollout_data"
    rollout_dir.mkdir(parents=True)
    samples = [
        _rs_sample("mixed", "bad", 0, 0, 0.0, 3),
        _rs_sample("mixed", "good", 0, 1, 1.0, 3),
        _rs_sample("certain", "good-a", 1, 2, 1.0, 3),
        _rs_sample("certain", "good-b", 1, 3, 1.0, 3),
    ]
    torch.save({"rollout_id": 0, "num_samples": len(samples), "samples": samples}, rollout_dir / "0.pt")

    model_dir = tmp_path / "model"
    megatron_dir = tmp_path / "Megatron-LM"
    model_dir.mkdir()
    megatron_dir.mkdir()
    return output_dir, (train_a, train_b), model_dir, megatron_dir


def _run_rs_workflow(output_dir, train_files, model_dir, megatron_dir, results_mode):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in (
        "MODEL_DIR",
        "MEGATRON_LM_PATH",
        "NUM_ROLLOUT",
        "OUTPUT_DIR",
        "PROMPT_DATA",
        "RUN_ROOT",
    ):
        env.pop(key, None)
    env.update(
        {
            "MEGATRON_LM_PATH": str(megatron_dir),
            "RUNS_ROOT": str(output_dir.parent / "runs"),
            "RUN_ROOT": str(output_dir.parent / "run"),
            "MCP_ENV_ROOT": str(output_dir.parent / "mcp-envs"),
            "SKIP_RAY_ROLLOUT": "1",
            "TIMESTAMP": "20260815000000",
        }
    )
    command = [
        "bash",
        str(repo_root / "experiments/run_qwen3_rejection_sampling.sh"),
        "--train-files",
        ",".join(str(path) for path in train_files),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model_dir),
        "--rollout-batch-size",
        "2",
        "--sample-n",
        "2",
        "--max-batches",
        "1",
        "--reward-threshold",
        "0.6",
        "--min-steps",
        "2",
        "--min-sample-trial",
        "2",
        "--max-trajectory-per-problem",
        "1",
        "--certainty-filter",
        "True",
        "--results-mode",
        results_mode,
        "--checkpointing",
        "True",
        "--resume",
        "True",
    ]
    return subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.integration
def test_qwen3_rejection_sampling_manifest_workflow_filters_caches_and_resumes(tmp_path):
    output_dir, train_files, model_dir, megatron_dir = _prepare_rs_workflow_fixture(tmp_path)

    first = _run_rs_workflow(output_dir, train_files, model_dir, megatron_dir, "manifest")

    assert "All rollout batches already present; skipping Ray rollout" in first.stdout
    prepared_path = output_dir / "fused_train.parquet"
    fingerprint_path = prepared_path.with_suffix(".parquet.fingerprint.json")
    episode_path = output_dir / "episodes/global_steps_0.json"
    accepted_path = output_dir / "accepted_episodes/global_steps_0.json"
    checkpoint_path = output_dir / "latest_checkpoint.json"
    assert pq.ParquetFile(prepared_path).metadata.num_rows == 2
    assert json.loads(fingerprint_path.read_text())["inputs"][0]["size"] == train_files[0].stat().st_size

    summary = json.loads((output_dir / "rejection_sampling_summary.json").read_text())
    assert summary["results_mode"] == "manifest"
    assert summary["num_samples"] == 4
    assert summary["num_groups"] == 2
    assert summary["num_accepted"] == 1
    assert summary["num_rejected"] == 3
    assert summary["num_accepted_problems"] == 1

    sample_index = json.loads((output_dir / "rejection_sampling_sample_index.json").read_text())
    assert len(sample_index) == 4
    assert [item["sample_key"] for item in sample_index if item["is_accepted"]] == ["mixed:good"]
    results = json.loads((output_dir / "rejection_sampling_results.json").read_text())
    group_results = {item["problem_id"]: item for item in results["groups"]["items"]}
    assert group_results["mixed"]["pass_rate"] == 0.5
    assert group_results["mixed"]["accepted"] == 1
    assert group_results["certain"]["pass_rate"] == 1.0
    assert group_results["certain"]["accepted"] == 0

    accepted = json.loads(accepted_path.read_text())
    assert accepted["accepted_sample_keys"] == ["mixed:good"]
    assert [row["episode_id"] for row in accepted["trajectories"]] == ["mixed:good"]
    assert json.loads(episode_path.read_text())["num_episodes"] == 4

    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["next_rollout_id"] == 1
    assert checkpoint["num_samples"] == 4
    assert checkpoint["num_accepted"] == 1
    assert len(checkpoint["completed_shards"]) == 1
    assert "accepted_sample_keys" not in checkpoint
    assert "accepted_problem_ids" not in checkpoint
    assert "rejected_problem_ids" not in checkpoint
    assert not list(output_dir.rglob("*.tmp"))

    prepared_mtime = prepared_path.stat().st_mtime_ns
    episode_mtime = episode_path.stat().st_mtime_ns
    accepted_mtime = accepted_path.stat().st_mtime_ns
    second = _run_rs_workflow(output_dir, train_files, model_dir, megatron_dir, "manifest")
    assert "Reusing cached fused train parquet" in second.stdout
    assert prepared_path.stat().st_mtime_ns == prepared_mtime
    assert episode_path.stat().st_mtime_ns == episode_mtime
    assert accepted_path.stat().st_mtime_ns == accepted_mtime

    legacy_accepted = json.loads(accepted_path.read_text())
    legacy_accepted.pop("accepted_sample_keys")
    accepted_path.write_text(json.dumps(legacy_accepted, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    legacy_mtime = accepted_path.stat().st_mtime_ns
    legacy_resume = _run_rs_workflow(output_dir, train_files, model_dir, megatron_dir, "manifest")
    assert "Reusing cached fused train parquet" in legacy_resume.stdout
    assert accepted_path.stat().st_mtime_ns == legacy_mtime
    assert json.loads((output_dir / "rejection_sampling_summary.json").read_text())["num_accepted"] == 1

    accepted_path.write_text("{invalid", encoding="utf-8")
    rebuilt = _run_rs_workflow(output_dir, train_files, model_dir, megatron_dir, "manifest")
    assert "Reusing cached fused train parquet" in rebuilt.stdout
    assert json.loads(accepted_path.read_text())["accepted_sample_keys"] == ["mixed:good"]
    assert not list(output_dir.rglob("*.tmp"))

    old_fingerprint = json.loads(fingerprint_path.read_text())["digest"]
    pq.write_table(
        pa.table({"prompt": ["input-a", "input-a-updated"], "source": ["a", "a-updated"]}),
        train_files[0],
    )
    invalidated = _run_rs_workflow(output_dir, train_files, model_dir, megatron_dir, "manifest")
    assert "Wrote fused train parquet" in invalidated.stdout
    assert pq.ParquetFile(prepared_path).metadata.num_rows == 3
    assert json.loads(fingerprint_path.read_text())["digest"] != old_fingerprint
    assert not list(output_dir.rglob("*.tmp"))


@pytest.mark.integration
def test_qwen3_rejection_sampling_full_workflow_preserves_complete_results(tmp_path):
    output_dir, train_files, model_dir, megatron_dir = _prepare_rs_workflow_fixture(tmp_path)

    _run_rs_workflow(output_dir, train_files, model_dir, megatron_dir, "full")

    results = json.loads((output_dir / "rejection_sampling_results.json").read_text())
    assert results["summary"]["num_samples"] == 4
    assert results["summary"]["num_accepted"] == 1
    assert len(results["samples"]["items"]) == 4
    assert [item["metadata"]["rllm_episode"]["id"] for item in results["samples"]["items"] if item["is_accepted"]] == [
        "mixed:good"
    ]
    assert len(results["episodes"]["items"]) == 4
    assert results["global_steps"]["num_shards"] == 1
    assert results["global_steps"]["num_episodes"] == 4

    episode = json.loads((output_dir / "episodes/global_steps_0.json").read_text())
    assert episode["accepted_sample_keys"] == ["mixed:good"]
    checkpoint = json.loads((output_dir / "latest_checkpoint.json").read_text())
    assert checkpoint["results_mode"] == "full"
    assert checkpoint["next_rollout_id"] == 1
    assert len(checkpoint["completed_shards"]) == 1
    assert "accepted_sample_keys" not in checkpoint
    assert "rejected_sample_keys" not in checkpoint
    assert not list(output_dir.rglob("*.tmp"))


@pytest.mark.integration
def test_qwen3_rejection_sampling_resume_requires_atomic_rollout_pt_shard(tmp_path):
    output_dir, train_files, model_dir, megatron_dir = _prepare_rs_workflow_fixture(tmp_path)
    _run_rs_workflow(output_dir, train_files, model_dir, megatron_dir, "manifest")
    rollout_path = output_dir / "debug/rollout_data/0.pt"
    rollout_path.rename(rollout_path.with_suffix(".pt.saved"))

    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_rs_workflow(output_dir, train_files, model_dir, megatron_dir, "manifest")

    assert "Resume start: rollout batch 0/1" in error.value.stdout
    assert "No non-empty rejection-sampling shards were completed" in error.value.stderr


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
        "train_odyssey_gemma4_async.sh",
    ):
        training = (repo_root / "experiments" / training_launcher).read_text(encoding="utf-8")
        assert "--rollout-only-inference-fast-path" not in training
        assert "--rollout-only-skip-episode-dump" not in training
        assert "--async-save-debug-rollout-data" not in training


@pytest.mark.unit
def test_qwen3_training_launcher_defaults_to_adaptive_fully_async():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_qwen3_fused_agent_sync.sh").read_text(encoding="utf-8")
    assert (
        'ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-'
        'slime.rollout.fully_async_rollout.generate_rollout_fully_async}"'
    ) in launcher
    assert 'FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY:-true}"' in launcher
    assert 'export SLIME_FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY}"' in launcher


@pytest.mark.unit
def test_odyssey_sync_launcher_drains_rollouts_before_weight_updates():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_odyssey_qwen3_multinode_sync.sh").read_text(encoding="utf-8")

    assert (
        'ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-'
        'slime.rollout.sglang_rollout.generate_rollout}"'
    ) in launcher
    assert 'export SLIME_FUSED_REQUIRE_WEIGHT_VERSION="${SLIME_FUSED_REQUIRE_WEIGHT_VERSION:-1}"' in launcher
    assert "--sglang-rl-on-policy-target megatron" in launcher
    assert "--fp32-residual-connection" in launcher
    assert "--batch-invariant-mode" in launcher
    assert "export SLIME_SGLANG_BATCH_INVARIANT_LOGPROB=1" in launcher
    assert "export SLIME_SGLANG_EXACT_RMSNORM=1" in launcher
    assert '"SLIME_FUSED_REQUIRE_WEIGHT_VERSION", "SLIME_SGLANG_BATCH_INVARIANT_LOGPROB"' in launcher
    assert '"SLIME_SGLANG_EXACT_RMSNORM"' in launcher


@pytest.mark.unit
def test_odyssey_sync_launcher_disables_grm_training_rewards_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_odyssey_qwen3_multinode_sync.sh").read_text(encoding="utf-8")

    assert 'ENABLE_USE_GRM_TRAIN="${ENABLE_USE_GRM_TRAIN:-${enable_use_grm_train:-false}}"' in launcher
    assert '--enable_use_grm_train|--enable-use-grm-train)' in launcher
    assert 'ROLLOUT_ARGS+=(--enable-use-grm-train)' in launcher
    assert 'GRM_MODE="${GRM_MODE:-score}"' in launcher
    assert (
        'OpenRouter GRM: train=${ENABLE_USE_GRM_TRAIN}, train_model=${TRAIN_GRM_MODEL}, '
        'evals=${ENABLE_USE_GRM_EVALS}, eval_model=${EVAL_GRM_MODEL}'
    ) in launcher
    assert 'TRAIN_GRM_MODEL="${TRAIN_GRM_MODEL:-deepseek/deepseek-v4-flash-0731}"' in launcher
    assert 'EVAL_GRM_MODEL="${EVAL_GRM_MODEL:-google/gemini-3-flash-preview}"' in launcher
    assert '--train-grm-model "${TRAIN_GRM_MODEL}"' in launcher
    assert '--eval-grm-model "${EVAL_GRM_MODEL}"' in launcher


@pytest.mark.unit
def test_all_training_launchers_use_separate_train_and_eval_grm_models():
    repo_root = Path(__file__).resolve().parents[1]
    launchers = sorted((repo_root / "experiments").glob("train_*.sh"))

    assert launchers
    for launcher_path in launchers:
        launcher = launcher_path.read_text(encoding="utf-8")
        assert 'TRAIN_GRM_MODEL="${TRAIN_GRM_MODEL:-deepseek/deepseek-v4-flash-0731}"' in launcher
        assert 'EVAL_GRM_MODEL="${EVAL_GRM_MODEL:-google/gemini-3.7-flash}"' in launcher
        assert '--train-grm-model "${TRAIN_GRM_MODEL}"' in launcher
        assert '--eval-grm-model "${EVAL_GRM_MODEL}"' in launcher
        assert 'GRM_MODEL="${GRM_MODEL:-' not in launcher
        assert '--grm-model "${GRM_MODEL}"' not in launcher


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
def test_all_training_launchers_use_dapo_token_level_policy_gradient_loss():
    repo_root = Path(__file__).resolve().parents[1]
    launchers = sorted((repo_root / "experiments").glob("train_*.sh"))

    assert launchers
    missing = []
    for launcher in launchers:
        text = launcher.read_text(encoding="utf-8")
        perf_start = text.index("PERF_ARGS=(")
        perf_end = text.index("\n)", perf_start)
        if "--calculate-per-token-loss" not in text[perf_start:perf_end]:
            missing.append(launcher.name)
    assert not missing, f"training launchers missing --calculate-per-token-loss: {missing}"


@pytest.mark.unit
def test_gemma4_launcher_defaults_to_even_mcp_webqa_groups():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_gemma4_fused_agent_sync.sh").read_text(encoding="utf-8")

    assert 'ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"' in launcher
    assert 'ROLLOUT_TASK_FAMILY_QUOTAS="${ROLLOUT_TASK_FAMILY_QUOTAS:-mcp=0.5,webqa=0.5}"' in launcher
    assert 'ROLLOUT_ARGS+=(--rollout-task-family-quotas "${ROLLOUT_TASK_FAMILY_QUOTAS}")' in launcher


@pytest.mark.unit
def test_gemma4_launcher_strictly_resumes_an_explicit_wandb_run_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_odyssey_gemma4_sync.sh").read_text(encoding="utf-8")

    assert 'WANDB_RESUME_SAME_RUN="${WANDB_RESUME_SAME_RUN:-1}"' in launcher
    assert '[ -n "${WANDB_RUN_ID:-}" ] && is_truthy "${WANDB_RESUME_SAME_RUN}"' in launcher
    assert 'WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")' in launcher


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
def test_odyssey_gemma4_sync_launcher_avoids_train_memory_saver_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_odyssey_gemma4_sync.sh").read_text(encoding="utf-8")

    assert 'RELEASE_TRAIN="${RELEASE_TRAIN:-true}"' in launcher
    assert 'OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-${COLOCATE}}"' in launcher
    assert "OFFLOAD_TRAIN=false" in launcher
    assert "--update-weight-mode full" in launcher
    assert "--update-weight-transport disk" in launcher
    assert 'if is_truthy "${COLOCATE}"; then' in launcher
    assert 'SGLANG_MEM_FRACTION_STATIC=0.6' in launcher
    assert (
        'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,max_split_size_mb:256"'
        in launcher
    )


@pytest.mark.unit
def test_odyssey_gemma4_sync_validates_model_and_resume_checkpoint_shapes():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_odyssey_gemma4_sync.sh").read_text(encoding="utf-8")

    assert 'MODEL_CONFIG="${MODEL_CONFIG:-}"' in launcher
    assert 'gemma4-e2b|gemma-4-e2b)' in launcher
    assert 'gemma4-e4b|gemma-4-e4b)' in launcher
    assert '"hidden_size": int(sys.argv[9])' in launcher
    assert '"ffn_hidden_size": int(sys.argv[10])' in launcher
    assert "Refusing incompatible resume" in launcher


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
def test_odyssey_sync_launcher_avoids_train_memory_saver_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_odyssey_qwen3_multinode_sync.sh").read_text(encoding="utf-8")

    assert 'RELEASE_TRAIN="${RELEASE_TRAIN:-true}"' in launcher
    assert 'OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-${COLOCATE}}"' in launcher
    assert '--train-env-vars \'{"TMS_INIT_ENABLE_CPU_BACKUP":"1"}\'' in launcher
    assert 'TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-536870912}"' in launcher
    assert 'LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-256}"' in launcher
    assert '--train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"' in launcher
    assert "export SLIME_TILED_POLICY_LOSS=1" in launcher
    assert "export SLIME_TILED_POLICY_LOSS_CLEAR_CACHE_BEFORE_BACKWARD=1" in launcher
    assert '"SLIME_SGLANG_EXACT_RMSNORM", "SLIME_TILED_POLICY_LOSS",' in launcher
    assert (
        'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False,'
        'max_split_size_mb:256}"' in launcher
    )
    assert "--update-weight-mode full" in launcher
    assert "--update-weight-transport disk" in launcher


@pytest.mark.unit
def test_odyssey_sync_launcher_enables_host_hicache_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_odyssey_qwen3_multinode_sync.sh").read_text(encoding="utf-8")

    assert 'SGLANG_ENABLE_HIERARCHICAL_CACHE="${SGLANG_ENABLE_HIERARCHICAL_CACHE:-true}"' in launcher
    assert 'SGLANG_HICACHE_SIZE="${SGLANG_HICACHE_SIZE:-32}"' in launcher
    assert 'SGLANG_HICACHE_WRITE_POLICY="${SGLANG_HICACHE_WRITE_POLICY:-write_through}"' in launcher
    assert 'SGLANG_HICACHE_IO_BACKEND="${SGLANG_HICACHE_IO_BACKEND:-kernel}"' in launcher
    assert 'SGLANG_HICACHE_MEM_LAYOUT="${SGLANG_HICACHE_MEM_LAYOUT:-layer_first}"' in launcher
    assert 'SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-96}"' in launcher
    assert 'SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"' in launcher
    assert 'OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-128}"' in launcher
    assert "--sglang-enable-hierarchical-cache" in launcher
    assert '--sglang-hicache-size "${SGLANG_HICACHE_SIZE}"' in launcher
    assert '--sglang-hicache-write-policy "${SGLANG_HICACHE_WRITE_POLICY}"' in launcher
    assert '--sglang-hicache-io-backend "${SGLANG_HICACHE_IO_BACKEND}"' in launcher
    assert '--sglang-hicache-mem-layout "${SGLANG_HICACHE_MEM_LAYOUT}"' in launcher


@pytest.mark.unit
def test_single_node_odyssey_launcher_avoids_train_memory_saver_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments" / "train_odyssey_qwen3_sync.sh").read_text(encoding="utf-8")

    assert 'COLOCATE="${COLOCATE:-true}"' in launcher
    assert 'RELEASE_TRAIN="${RELEASE_TRAIN:-true}"' in launcher
    assert 'OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-${COLOCATE}}"' in launcher
    assert "OFFLOAD_TRAIN=false" in launcher
    assert "--update-weight-mode full" in launcher
    assert "--update-weight-transport disk" in launcher


@pytest.mark.unit
def test_odyssey_qwen3_launchers_use_pageable_tensor_backups():
    repo_root = Path(__file__).resolve().parents[1]

    for launcher_name in (
        "train_odyssey_qwen3_multinode_sync.sh",
        "train_odyssey_qwen3_sync.sh",
    ):
        launcher = (repo_root / "experiments" / launcher_name).read_text(encoding="utf-8")

        assert 'export SLIME_TENSOR_BACKUP_PIN_MEMORY="${SLIME_TENSOR_BACKUP_PIN_MEMORY:-0}"' in launcher
        assert '"SLIME_FUSED_REQUIRE_WEIGHT_VERSION", "SLIME_TENSOR_BACKUP_PIN_MEMORY",' in launcher


@pytest.mark.unit
def test_srppo_launcher_releases_actor_but_keeps_custom_advantage_isolated():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments" / "train_srppo_qwen3_sync.sh").read_text(encoding="utf-8")

    assert 'RELEASE_TRAIN="${RELEASE_TRAIN:-true}"' in launcher
    assert '--custom-advantage-function-path "slime.algorithms.srppo.custom_advantage_fn"' in launcher
    assert '--advantage-estimator "ppo"' in launcher


@pytest.mark.unit
def test_single_node_odyssey_launcher_uses_cuda12_compatible_flashinfer_norm():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments" / "train_odyssey_qwen3_sync.sh").read_text(encoding="utf-8")

    assert 'FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-1}"' in launcher
    assert "export FLASHINFER_USE_CUDA_NORM" in launcher
    assert 'export FLASHINFER_USE_TORCH_NORM="${FLASHINFER_USE_TORCH_NORM:-1}"' in launcher
    assert 'os.path.join(sys.prefix, "lib")' in launcher
    assert '"FLASHINFER_USE_CUDA_NORM", "FLASHINFER_USE_TORCH_NORM",' in launcher


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
    assert 'ROLLOUT_INFRA_RETRY_TIMES="${ROLLOUT_INFRA_RETRY_TIMES:-2}"' in launcher
    assert '--rollout-infra-retry-times "${ROLLOUT_INFRA_RETRY_TIMES}"' in launcher
    assert 'SLIME_LOCAL_MCP_LEASE_TIMEOUT="${SLIME_LOCAL_MCP_LEASE_TIMEOUT:-120}"' in launcher
    assert launcher.count("SLIME_LOCAL_MCP_LEASE_TIMEOUT") >= 2


@pytest.mark.unit
def test_odyssey_sync_launcher_keeps_aggressive_pending_group_reservoir():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_odyssey_qwen3_multinode_sync.sh").read_text(encoding="utf-8")

    assert 'OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-128}"' in launcher
    assert "SYNC_MIN_PENDING_GROUPS=96" in launcher
    assert 'export SLIME_SYNC_MIN_PENDING_GROUPS="${SYNC_MIN_PENDING_GROUPS}"' in launcher
    assert 'SYNC_WEBQA_MIN_PENDING_GROUPS="${SYNC_WEBQA_MIN_PENDING_GROUPS:-32}"' in launcher
    assert 'SYNC_MCP_MIN_PENDING_GROUPS="${SYNC_MCP_MIN_PENDING_GROUPS:-32}"' in launcher
    assert 'export SLIME_SYNC_WEBQA_MIN_PENDING_GROUPS="${SYNC_WEBQA_MIN_PENDING_GROUPS}"' in launcher
    assert 'export SLIME_SYNC_MCP_MIN_PENDING_GROUPS="${SYNC_MCP_MIN_PENDING_GROUPS}"' in launcher
    assert "MCP_ENV_COPY_CONCURRENCY=32" in launcher
    assert "export SLIME_LOCAL_MCP_PROCESS_WORKERS=32" in launcher
    assert 'ROUTER_POLICY="${ROUTER_POLICY:-consistent_hashing}"' in launcher
    assert 'SLIME_LOCAL_MCP_TOOLSET_CACHE_SIZE="${SLIME_LOCAL_MCP_TOOLSET_CACHE_SIZE:-64}"' in launcher
    assert 'SLIME_LOCAL_MCP_PROCESS_EAGER_WARM="${SLIME_LOCAL_MCP_PROCESS_EAGER_WARM:-true}"' in launcher
    assert launcher.count("SLIME_LOCAL_MCP_TOOLSET_CACHE_SIZE") >= 2
    assert launcher.count("SLIME_LOCAL_MCP_PROCESS_EAGER_WARM") >= 2
    assert '"SLIME_SYNC_MIN_PENDING_GROUPS"' in launcher


@pytest.mark.unit
def test_single_node_odyssey_sync_launcher_bounds_pending_groups_by_engine_count():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/train_odyssey_qwen3_sync.sh").read_text(encoding="utf-8")

    assert 'OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_ENGINE_COUNT * 2))}"' in launcher
    assert 'SYNC_MIN_PENDING_GROUPS="${SYNC_MIN_PENDING_GROUPS:-$((ROLLOUT_ENGINE_COUNT * 2))}"' in launcher
    assert 'SYNC_WEBQA_MIN_PENDING_GROUPS="${SYNC_WEBQA_MIN_PENDING_GROUPS:-${ROLLOUT_ENGINE_COUNT}}"' in launcher
    assert 'SYNC_MCP_MIN_PENDING_GROUPS="${SYNC_MCP_MIN_PENDING_GROUPS:-${ROLLOUT_ENGINE_COUNT}}"' in launcher


@pytest.mark.unit
def test_odyssey_launchers_balance_webqa_and_mcp_training_groups():
    repo_root = Path(__file__).resolve().parents[1]
    launchers = [
        (repo_root / "experiments/train_odyssey_qwen3_multinode_sync.sh").read_text(encoding="utf-8"),
        (repo_root / "experiments/train_odyssey_qwen3_sync.sh").read_text(encoding="utf-8"),
    ]

    for launcher in launchers:
        assert 'ROLLOUT_TASK_FAMILY_QUOTAS="${ROLLOUT_TASK_FAMILY_QUOTAS:-webqa=0.5,mcp=0.5}"' in launcher
        assert '--rollout-task-family-quotas "${ROLLOUT_TASK_FAMILY_QUOTAS}"' in launcher
        assert "--rollout-task-family-admission-only" not in launcher
        assert "ROLLOUT_BATCH_SIZE must be even for the 50/50 webqa/mcp training mix" in launcher
        assert 'export FUSED_WEBQA_REWARD_MATCH_MODE="normalized_target_span"' in launcher
        assert '"FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "FUSED_WEBQA_REWARD_MATCH_MODE",' in launcher
        assert "FUSED_WEBQA_ALIAS_REGISTRY_PATH" in launcher
        assert "SLIME_ROLLOUT_PREFILTER_AUDIT_DIR" in launcher
        assert "webqa_alias_registry.json" in launcher
        assert "prefilter_audit" in launcher
    multinode_launcher = launchers[0]
    assert 'ROLLOUT_TASK_FAMILY_TOP_MEAN_STEPS="${ROLLOUT_TASK_FAMILY_TOP_MEAN_STEPS:-true}"' in multinode_launcher
    assert '--rollout-task-family-top-mean-steps) ROLLOUT_TASK_FAMILY_TOP_MEAN_STEPS=' in multinode_launcher
    assert 'if is_truthy "${ROLLOUT_TASK_FAMILY_TOP_MEAN_STEPS}"; then' in multinode_launcher
    assert 'ROLLOUT_ARGS+=(--rollout-task-family-top-mean-steps)' in multinode_launcher
    assert 'USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-true}"' in multinode_launcher
    assert 'MISC_ARGS+=(--use-fault-tolerance)' in multinode_launcher
    assert '--replay-rollout-id) REPLAY_ROLLOUT_ID=' in multinode_launcher
    assert 'SLIME_DIAGNOSTIC_ROLLOUT_DATA="${DUMP_DETAILS}/rollout_data/${REPLAY_ROLLOUT_ID}.pt"' in multinode_launcher
    assert 'NUM_ROLLOUT=$((REPLAY_ROLLOUT_ID + 1))' in multinode_launcher
    assert 'MISC_ARGS+=(--debug-train-only)' in multinode_launcher


@pytest.mark.unit
def test_launchers_provision_per_trajectory_mcp_workspaces():
    repo_root = Path(__file__).resolve().parents[1]
    rejection = (repo_root / "experiments/run_qwen3_rejection_sampling.sh").read_text(encoding="utf-8")
    odyssey = (repo_root / "experiments/train_odyssey_qwen3_multinode_sync.sh").read_text(encoding="utf-8")

    assert 'MCP_ENV_ROOT="${MCP_ENV_ROOT:-${RUN_ROOT}/cache/mcp_envs}"' in rejection
    assert 'export SLIME_MCP_ENV_ROOT="${MCP_ENV_ROOT}"' in rejection
    assert 'MCP_ENV_ROOT="${MCP_ENV_ROOT:-${RUN_ROOT}/cache/mcp_envs}"' in odyssey
    assert 'export SLIME_MCP_ENV_ROOT="${MCP_ENV_ROOT}"' in odyssey
    assert 'export SLIME_MCP_WORKSPACE_SCOPE="${SLIME_MCP_WORKSPACE_SCOPE:-task}"' in odyssey
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
