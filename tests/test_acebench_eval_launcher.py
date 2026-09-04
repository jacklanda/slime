import subprocess
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
    assert 'ACEBENCH_OVERWRITE="${ACEBENCH_OVERWRITE:-true}"' in launcher
    assert 'is_truthy "${ACEBENCH_OVERWRITE}"' in launcher
    acebench_branch = launcher.split('if is_truthy "${ACEBENCH_SELECTED}"; then', 1)[1].split(
        'if is_truthy "${TAU2_SELECTED}"; then', 1
    )[0]
    assert 'ACEBENCH_OVERWRITE=true' in acebench_branch
    assert '--max-tokens "${EVAL_MAX_RESPONSE_LEN}"' not in acebench_branch
    assert 'ACEBENCH_LANGUAGE="${ACEBENCH_LANGUAGE:-en}"' in launcher
    assert 'ACEBENCH_USER_MODEL="${ACEBENCH_USER_MODEL:-${OPENROUTER_MODEL:-openai/gpt-4o}}"' in launcher
    assert 'ACEBENCH_USER_BASE_URL="${ACEBENCH_USER_BASE_URL:-${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}}"' in launcher
    assert 'ACEBENCH_MAX_TOKENS="${ACEBENCH_MAX_TOKENS:-16384}"' in launcher
    assert 'ACEBENCH_MAX_TOKENS=16384' in acebench_branch
    assert 'ACEBENCH_CMD+=(--max-tokens "${ACEBENCH_MAX_TOKENS}")' in launcher
    assert 'ACEBENCH_TEMPERATURE="${ACEBENCH_TEMPERATURE:-0.2}"' in launcher
    assert 'ACEBENCH_TOP_P="${ACEBENCH_TOP_P:-0.95}"' in launcher
    assert 'EVAL_MAX_CONTEXT_LEN=40960' in launcher
    assert 'EVAL_MAX_PROMPT_LEN=8192' in launcher
    assert 'EVAL_MAX_RESPONSE_LEN=8192' in launcher
    assert 'PER_STEP_MAX_TOKENS=8192' in launcher
    assert 'SGLANG_MAX_RUNNING_REQUESTS=64' in launcher
    assert 'LIMIT_PER_BENCHMARK=99999' in launcher
    assert 'N_SAMPLES_PER_PROMPT=1' in launcher
    assert '--temperature "${ACEBENCH_TEMPERATURE}"' in acebench_branch
    assert '--top-p "${ACEBENCH_TOP_P}"' in acebench_branch
    assert 'ACEBENCH_VENV_DIR="${ACEBENCH_ROOT}/.venv-slime-evals"' in launcher
    assert 'python3 -m venv "${ACEBENCH_VENV_DIR}"' in launcher
    assert 'tomllib.load(file)["project"]["dependencies"]' in launcher
    assert 'export OPENROUTER_API_KEY="${ACEBENCH_USER_API_KEY}"' in launcher
    assert '--user-api-key "${ACEBENCH_USER_API_KEY}"' not in launcher
    assert 'exec "${ACEBENCH_CMD[@]}"' in launcher
    assert 'LANGUAGE="en"' in acebench_launcher
    assert 'NUM_THREADS="16"' in acebench_launcher
    assert 'TEMPERATURE="0.2"' in acebench_launcher
    assert 'MAX_SAMPLES_PER_TASK="99999"' in acebench_launcher
    assert 'MAX_TOKENS="16384"' in acebench_launcher
    assert 'ENABLE_THINKING="true"' in acebench_launcher
    assert 'SGLANG_CONTEXT_LENGTH="40960"' in acebench_launcher
    assert 'USER_MODEL="${OPENROUTER_MODEL:-openai/gpt-4o}"' in acebench_launcher
    assert 'USER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"' in acebench_launcher
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


@pytest.mark.unit
def test_eval_launcher_accepts_acebench_runner_option_aliases():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "bash",
            "experiments/evals.sh",
            "--protocol-mode",
            "slime_fused_gem",
            "--agent-backend",
            "rllm_tool_agent",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Alias: --protocol-mode" in result.stdout
    assert "Alias: --agent-backend" in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
