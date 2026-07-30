from pathlib import Path

import pytest

NUM_GPUS = 0


@pytest.mark.unit
def test_eval_launcher_routes_acebench_to_isolated_official_pipeline():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/evals.sh").read_text(encoding="utf-8")
    acebench_launcher = (
        repo_root / "experiments/artifacts/benchmarks/ACEBench/run_eval.sh"
    ).read_text(encoding="utf-8")

    assert 'normalized_include="$(printf' in launcher
    assert 'ACEBENCH_ROOT="${BENCHMARKS_ROOT}/ACEBench"' in launcher
    assert '--sglang-model-path "${ACEBENCH_MODEL_DIR}"' in launcher
    assert '--sglang-context-length "${EVAL_MAX_CONTEXT_LEN}"' in launcher
    assert 'ACEBENCH_SGLANG_PREFIX="$("${ACEBENCH_SGLANG_PYTHON_BIN}"' in launcher
    assert 'export LD_PRELOAD="${ACEBENCH_SGLANG_LIBSTDCXX}' in launcher
    assert '--max-samples-per-task "${LIMIT_PER_BENCHMARK}"' in launcher
    assert '--protocol-mode "${ACEBENCH_PROTOCOL}"' in launcher
    assert '--top-k "${TOP_K}"' in launcher
    assert '--seed "${ROLLOUT_SEED}"' in launcher
    assert '--slime-repo-root "${REPO_ROOT}"' in launcher
    assert 'ACEBENCH_PROTOCOL=slime_fused_gem' in launcher
    assert 'ACEBENCH_AGENT_BACKEND=rllm_tool_agent' in launcher
    assert 'ACEBENCH_NUM_THREADS="${ACEBENCH_NUM_THREADS:-auto}"' in launcher
    assert 'ACEBENCH_NUM_THREADS=$((ACEBENCH_DP_SIZE * 8))' in launcher
    assert 'ACEBENCH_OVERWRITE="${ACEBENCH_OVERWRITE:-false}"' in launcher
    assert 'is_truthy "${ACEBENCH_OVERWRITE}"' in launcher
    assert '--max-tokens "${EVAL_MAX_RESPONSE_LEN}"' not in launcher
    assert 'ACEBENCH_MAX_TOKENS="${ACEBENCH_MAX_TOKENS:-}"' in launcher
    assert 'ACEBENCH_CMD+=(--max-tokens "${ACEBENCH_MAX_TOKENS}")' in launcher
    assert 'ACEBENCH_VENV_DIR="${ACEBENCH_ROOT}/.venv-slime-evals"' in launcher
    assert 'python3 -m venv "${ACEBENCH_VENV_DIR}"' in launcher
    assert 'tomllib.load(file)["project"]["dependencies"]' in launcher
    assert 'export OPENROUTER_API_KEY="${ACEBENCH_USER_API_KEY}"' in launcher
    assert '--user-api-key "${ACEBENCH_USER_API_KEY}"' not in launcher
    assert 'exec "${ACEBENCH_CMD[@]}"' in launcher
    acebench_branch = launcher.split('if is_truthy "${ACEBENCH_SELECTED}"; then', 1)[1].split(
        'case "${ROUTER_POLICY}"', 1
    )[0]
    assert "ray stop --force" not in acebench_branch
    assert 'is_truthy "${CLEANUP}"' not in acebench_branch
    assert 'export PYTHONPATH="${REPO_ROOT}' in acebench_branch
    assert "--no-overwrite" in acebench_launcher
    assert '--no-overwrite)\n      OVERWRITE="0"' in acebench_launcher
    assert "'<redacted>'" in acebench_launcher
    assert 'generate_cmd+=(--user-api-key "$USER_API_KEY")' not in acebench_launcher
    assert 'endpoint_serves_expected_model' in acebench_launcher
    assert 'run_config.json' in acebench_launcher
    assert 'flock -n 9' in acebench_launcher


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
