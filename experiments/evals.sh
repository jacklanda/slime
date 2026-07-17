#!/bin/bash
# Comprehensive fused-agent benchmark eval launcher.
#
# The script keeps evaluation logic in experiments/:
# - discovers benchmarks under experiments/artifacts/benchmarks
# - writes a slime --eval-config into the run log dir
# - launches train.py in debug-rollout-only eval mode with SGLang rollout

set -euo pipefail

export PYTHONUNBUFFERED=1

usage() {
   cat <<'EOF'
Usage:
  bash experiments/evals.sh [options] [-- extra slime args...]

Core:
  --model PATH                         HF model path.
  --model-config NAME                  scripts/models/<NAME>.sh. Defaults by --model-series.
  --model-series qwen3|qwen3.5         Default: qwen3.5.
  --benchmarks-root PATH               Default: experiments/artifacts/benchmarks.
  --include LIST                       Comma-separated benchmark names, or all.
                                       frontierscience expands to
                                       frontierscience_olympiad,frontierscience_research.
                                       Default: search_r1.
  --exclude LIST                       Extra comma-separated benchmark names to skip.
  --experiment-name NAME               Log/run name.
  --gpus N                             Total rollout GPUs. Default: 8.
  --gpus-per-engine N                  Tensor parallel size per SGLang engine. Default: 1.
  --ray-num-cpus N                     Ray CPU resources. Default: 64.

Generation/eval:
  --harness NAME                       Harness: bare, cot, react, gem, unified_gem,
                                       rllm_deepresearch (alias: rllm_dr). Default: cot.
  --user_prompt long|short             Web-search user prompt. Default: short.
  --rllm-dr-refine-server-url URLS     Comma-separated OpenAI-compatible Refine server base URLs.
  --unified-system-prompt              Sets harness to unified_gem unless --harness is later set.
  --no-unified-system-prompt           Sets harness to gem unless --harness is later set.
  --disable-thinking BOOL              Default: true.
  --discard-historical-thinking BOOL   Remove prior assistant <think> blocks before each new rollout step.
                                       Effective only when --disable-thinking is false. Default: false.
  --max-steps N                        Default: 128.
  --mcp-max-steps N                    Default: max-steps.
  --web-search-max-steps N             Default: max-steps.
  --cli-max-steps N                    Default: max-steps.
  --trajectory-timeout N               Default: 7200.
  --eval-trajectory-timeout N          Default: trajectory-timeout.
  --n-samples-per-eval-prompt N        Default: 1.
  --temperature X                      Default: 0.7.
  --top-p X                            Default: 1.0.
  --top-k N                            Default: 20.
  --eval-max-response-len N            Default: 38000; also the default cot per-step budget.
  --eval-max-prompt-len N              Default: 2048.
  --eval-max-context-len N             Default: prompt + response.
  --limit-per-benchmark N              Generate config with first N examples per benchmark. Default: 0 (all).
  --no-prefer-verl                     Normalize from raw files even when data_verl.parquet exists.
  --retrieval-concurrency N            Concurrent retrieval requests. Default: 176.
  --retrieval-cache-size N             Cross-episode retrieval LRU entries. Default: 4096.
  --eval-initial-inflight-tasks N      Initial scheduled eval trajectories. Default: cot=4/engine, agent=384.
  --eval-max-inflight-tasks N          Adaptive hard limit. Default: cot=8/engine, agent=576.
  --eval-adaptive-concurrency BOOL     Adjust inflight work from engine metrics. Default: true.
  --eval-trajectory-sample-rate X      Full trajectory dump fraction. Default: 1.
  --eval-dump-failures BOOL            Dump failed eval trajectories. Default: true.
  --native-sglang-session BOOL         Use verified incremental SGLang sessions. Default: true.

SGLang/runtime:
  --sglang-mem-fraction-static X       Default: 0.9.
  --sglang-server-concurrency N        Default: 60.
  --sglang-max-running-requests N      Per-engine limit. Default: cot=4, agent=96.
  --router-policy NAME                 Default: manual.
  --router-assignment-mode NAME        Default: min_load.
  --ray-dashboard-address URL          Default: http://127.0.0.1:8265.
  --ray-job-wait BOOL                  Default: false.
  --ray-job-follow-logs BOOL           Default: true.
  --cleanup BOOL                       Stop old Ray head first when true. Default: false.
  -h, --help                           Show this help.
EOF
}

is_truthy() {
   case "${1}" in
      1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
      *) return 1 ;;
   esac
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
BASE_DIR="$(cd -- "${REPO_ROOT}/.." &>/dev/null && pwd)"

MODEL_SERIES="${MODEL_SERIES:-qwen3.5}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
MODEL_DIR="${MODEL_DIR:-}"
BENCHMARKS_ROOT="${BENCHMARKS_ROOT:-${SCRIPT_DIR}/artifacts/benchmarks}"
#INCLUDE_BENCHMARKS="${INCLUDE_BENCHMARKS:-2wiki,bamboogle,gpqa_diamond,medqa}"
#INCLUDE_BENCHMARKS="${INCLUDE_BENCHMARKS:-browsecomp_plus}"
#INCLUDE_BENCHMARKS="${INCLUDE_BENCHMARKS:-search_r1}"
INCLUDE_BENCHMARKS="${INCLUDE_BENCHMARKS:-bamboogle}"
EXCLUDE_BENCHMARKS="${EXCLUDE_BENCHMARKS:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval-$(date +%Y%m%d-%H%M%S)}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
FUSED_HARNESS="${FUSED_HARNESS:-cot}"
USER_PROMPT="${USER_PROMPT:-short}"
RLLM_DR_REFINE_SERVER_URL="${RLLM_DR_REFINE_SERVER_URL:-}"
RLLM_DR_USE_REFINE="${RLLM_DR_USE_REFINE:-0}"
UNIFIED_SYSTEM_PROMPT="${UNIFIED_SYSTEM_PROMPT:-False}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"
DISCARD_HISTORICAL_THINKING="${DISCARD_HISTORICAL_THINKING:-false}"
MAX_STEPS="${MAX_STEPS:-128}"
MCP_MAX_STEPS="${MCP_MAX_STEPS:-${MAX_STEPS}}"
WEB_SEARCH_MAX_STEPS="${WEB_SEARCH_MAX_STEPS:-${MAX_STEPS}}"
CLI_MAX_STEPS="${CLI_MAX_STEPS:-${MAX_STEPS}}"
TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT:-7200}"
EVAL_TRAJECTORY_TIMEOUT="${EVAL_TRAJECTORY_TIMEOUT:-${TRAJECTORY_TIMEOUT}}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
#EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
#EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-23616}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-38000}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-2048}"
#EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-38000}"
#EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-2048}"
#EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-65536}"
#EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-131072}"
#EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-4096}"
EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-}"
LIMIT_PER_BENCHMARK="${LIMIT_PER_BENCHMARK:-0}"
PREFER_VERL="${PREFER_VERL:-1}"
RETRIEVAL_CONCURRENCY="${RETRIEVAL_CONCURRENCY:-176}"
RETRIEVAL_CACHE_SIZE="${RETRIEVAL_CACHE_SIZE:-4096}"
EVAL_INITIAL_INFLIGHT_TASKS="${EVAL_INITIAL_INFLIGHT_TASKS:-}"
EVAL_MAX_INFLIGHT_TASKS="${EVAL_MAX_INFLIGHT_TASKS:-}"
EVAL_ADAPTIVE_CONCURRENCY="${EVAL_ADAPTIVE_CONCURRENCY:-true}"
EVAL_TRAJECTORY_SAMPLE_RATE="${EVAL_TRAJECTORY_SAMPLE_RATE:-1}"
EVAL_DUMP_FAILURES="${EVAL_DUMP_FAILURES:-true}"
NATIVE_SGLANG_SESSION="${NATIVE_SGLANG_SESSION:-true}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.9}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-60}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
ROUTER_POLICY="${ROUTER_POLICY:-manual}"
ROUTER_ASSIGNMENT_MODE="${ROUTER_ASSIGNMENT_MODE:-min_load}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
RAY_JOB_WAIT="${RAY_JOB_WAIT:-0}"
RAY_JOB_FOLLOW_LOGS="${RAY_JOB_FOLLOW_LOGS:-1}"
CLEANUP="${CLEANUP:-false}"
harness_explicit=false
EXTRA_SLIME_ARGS=()

while [ "$#" -gt 0 ]; do
   case "$1" in
      --model) MODEL_DIR="${2:?Missing value for --model}"; shift 2 ;;
      --model-config) MODEL_CONFIG="${2:?Missing value for --model-config}"; shift 2 ;;
      --model-series) MODEL_SERIES="${2:?Missing value for --model-series}"; shift 2 ;;
      --benchmarks-root) BENCHMARKS_ROOT="${2:?Missing value for --benchmarks-root}"; shift 2 ;;
      --include) INCLUDE_BENCHMARKS="${2:?Missing value for --include}"; shift 2 ;;
      --exclude) EXCLUDE_BENCHMARKS="${2:?Missing value for --exclude}"; shift 2 ;;
      --experiment-name) EXPERIMENT_NAME="${2:?Missing value for --experiment-name}"; shift 2 ;;
      --gpus) ROLLOUT_GPUS="${2:?Missing value for --gpus}"; shift 2 ;;
      --gpus-per-engine) ROLLOUT_NUM_GPUS_PER_ENGINE="${2:?Missing value for --gpus-per-engine}"; shift 2 ;;
      --ray-num-cpus) RAY_NUM_CPUS="${2:?Missing value for --ray-num-cpus}"; shift 2 ;;
      --harness) FUSED_HARNESS="${2:?Missing value for --harness}"; harness_explicit=true; shift 2 ;;
      --user_prompt|--user-prompt) USER_PROMPT="${2:?Missing value for --user_prompt}"; shift 2 ;;
      --rllm-dr-refine-server-url) RLLM_DR_REFINE_SERVER_URL="${2:?Missing value for --rllm-dr-refine-server-url}"; shift 2 ;;
      --unified-system-prompt) UNIFIED_SYSTEM_PROMPT=True; if [ "${harness_explicit}" = "false" ]; then FUSED_HARNESS=unified_gem; fi; shift ;;
      --no-unified-system-prompt) UNIFIED_SYSTEM_PROMPT=False; if [ "${harness_explicit}" = "false" ]; then FUSED_HARNESS=gem; fi; shift ;;
      --disable-thinking) DISABLE_THINKING="${2:?Missing value for --disable-thinking}"; shift 2 ;;
      --discard-historical-thinking) DISCARD_HISTORICAL_THINKING="${2:?Missing value for --discard-historical-thinking}"; shift 2 ;;
      --max-steps) MAX_STEPS="${2:?Missing value for --max-steps}"; shift 2 ;;
      --mcp-max-steps) MCP_MAX_STEPS="${2:?Missing value for --mcp-max-steps}"; shift 2 ;;
      --web-search-max-steps) WEB_SEARCH_MAX_STEPS="${2:?Missing value for --web-search-max-steps}"; shift 2 ;;
      --cli-max-steps) CLI_MAX_STEPS="${2:?Missing value for --cli-max-steps}"; shift 2 ;;
      --trajectory-timeout) TRAJECTORY_TIMEOUT="${2:?Missing value for --trajectory-timeout}"; shift 2 ;;
      --eval-trajectory-timeout) EVAL_TRAJECTORY_TIMEOUT="${2:?Missing value for --eval-trajectory-timeout}"; shift 2 ;;
      --n-samples-per-eval-prompt) N_SAMPLES_PER_EVAL_PROMPT="${2:?Missing value for --n-samples-per-eval-prompt}"; shift 2 ;;
      --temperature) TEMPERATURE="${2:?Missing value for --temperature}"; shift 2 ;;
      --top-p) TOP_P="${2:?Missing value for --top-p}"; shift 2 ;;
      --top-k) TOP_K="${2:?Missing value for --top-k}"; shift 2 ;;
      --eval-max-response-len) EVAL_MAX_RESPONSE_LEN="${2:?Missing value for --eval-max-response-len}"; shift 2 ;;
      --eval-max-prompt-len) EVAL_MAX_PROMPT_LEN="${2:?Missing value for --eval-max-prompt-len}"; shift 2 ;;
      --eval-max-context-len) EVAL_MAX_CONTEXT_LEN="${2:?Missing value for --eval-max-context-len}"; shift 2 ;;
      --limit-per-benchmark) LIMIT_PER_BENCHMARK="${2:?Missing value for --limit-per-benchmark}"; shift 2 ;;
      --no-prefer-verl) PREFER_VERL=0; shift ;;
      --retrieval-concurrency) RETRIEVAL_CONCURRENCY="${2:?Missing value for --retrieval-concurrency}"; shift 2 ;;
      --retrieval-cache-size) RETRIEVAL_CACHE_SIZE="${2:?Missing value for --retrieval-cache-size}"; shift 2 ;;
      --eval-initial-inflight-tasks) EVAL_INITIAL_INFLIGHT_TASKS="${2:?Missing value for --eval-initial-inflight-tasks}"; shift 2 ;;
      --eval-max-inflight-tasks) EVAL_MAX_INFLIGHT_TASKS="${2:?Missing value for --eval-max-inflight-tasks}"; shift 2 ;;
      --eval-adaptive-concurrency) EVAL_ADAPTIVE_CONCURRENCY="${2:?Missing value for --eval-adaptive-concurrency}"; shift 2 ;;
      --eval-trajectory-sample-rate) EVAL_TRAJECTORY_SAMPLE_RATE="${2:?Missing value for --eval-trajectory-sample-rate}"; shift 2 ;;
      --eval-dump-failures) EVAL_DUMP_FAILURES="${2:?Missing value for --eval-dump-failures}"; shift 2 ;;
      --native-sglang-session) NATIVE_SGLANG_SESSION="${2:?Missing value for --native-sglang-session}"; shift 2 ;;
      --sglang-mem-fraction-static) SGLANG_MEM_FRACTION_STATIC="${2:?Missing value for --sglang-mem-fraction-static}"; shift 2 ;;
      --sglang-server-concurrency) SGLANG_SERVER_CONCURRENCY="${2:?Missing value for --sglang-server-concurrency}"; shift 2 ;;
      --sglang-max-running-requests) SGLANG_MAX_RUNNING_REQUESTS="${2:?Missing value for --sglang-max-running-requests}"; shift 2 ;;
      --router-policy) ROUTER_POLICY="${2:?Missing value for --router-policy}"; shift 2 ;;
      --router-assignment-mode) ROUTER_ASSIGNMENT_MODE="${2:?Missing value for --router-assignment-mode}"; shift 2 ;;
      --ray-dashboard-address) RAY_DASHBOARD_ADDRESS="${2:?Missing value for --ray-dashboard-address}"; shift 2 ;;
      --ray-job-wait) RAY_JOB_WAIT="${2:?Missing value for --ray-job-wait}"; shift 2 ;;
      --ray-job-follow-logs) RAY_JOB_FOLLOW_LOGS="${2:?Missing value for --ray-job-follow-logs}"; shift 2 ;;
      --cleanup) CLEANUP="${2:?Missing value for --cleanup}"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      --) shift; EXTRA_SLIME_ARGS=("$@"); break ;;
      *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
   esac
done

case "${MODEL_SERIES}" in
   qwen3|qwen3.5) ;;
   *) echo "Unsupported --model-series ${MODEL_SERIES}; expected qwen3 or qwen3.5." >&2; exit 2 ;;
esac
case "${USER_PROMPT}" in
   long|short) ;;
   *) echo "Unsupported --user_prompt ${USER_PROMPT}; expected long or short." >&2; exit 2 ;;
esac

if [ -z "${MODEL_CONFIG}" ]; then
   if [ "${MODEL_SERIES}" = "qwen3.5" ]; then
      MODEL_CONFIG="qwen3.5-4B"
   else
      MODEL_CONFIG="qwen3-4B"
   fi
fi

if [ -z "${MODEL_DIR}" ]; then
   if [ "${MODEL_SERIES}" = "qwen3.5" ]; then
      MODEL_DIR="/share/nlp/share/plm/Qwen3.5-4B"
   else
      MODEL_DIR="/share/nlp/share/plm/Qwen3-4B"
   fi
fi

MODEL_CONFIG_PATH="${REPO_ROOT}/scripts/models/${MODEL_CONFIG}.sh"
if [ ! -f "${MODEL_CONFIG_PATH}" ]; then
   echo "MODEL_CONFIG does not exist: ${MODEL_CONFIG_PATH}" >&2
   exit 2
fi
if [ ! -d "${MODEL_DIR}" ]; then
   echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
   exit 2
fi
if [ ! -d "${BENCHMARKS_ROOT}" ]; then
   echo "BENCHMARKS_ROOT does not exist: ${BENCHMARKS_ROOT}" >&2
   exit 2
fi

python3 - "${MODEL_DIR}" "${MODEL_SERIES}" <<'PY' || exit 2
import json
import pathlib
import sys

model_dir = pathlib.Path(sys.argv[1])
model_series = sys.argv[2]
config_path = model_dir / "config.json"
try:
    with config_path.open(encoding="utf-8") as f:
        model_type = str(json.load(f).get("model_type", ""))
except (OSError, ValueError) as exc:
    raise SystemExit(f"Unable to validate model series from {config_path}: {exc}")

normalized_type = model_type.lower().replace("-", "_")
if model_series == "qwen3.5":
    matches = normalized_type == "qwen3_5" or normalized_type.startswith("qwen3_5_")
else:
    matches = (
        normalized_type == "qwen3" or normalized_type.startswith("qwen3_")
    ) and not normalized_type.startswith("qwen3_5")
if not matches:
    raise SystemExit(
        f"--model-series {model_series} does not match {config_path} model_type={model_type!r}"
    )
PY

source "${MODEL_CONFIG_PATH}"

EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-$((EVAL_MAX_PROMPT_LEN + EVAL_MAX_RESPONSE_LEN))}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
if [ "${ROLLOUT_GPUS}" -lt 1 ]; then
   echo "--gpus must be >= 1" >&2
   exit 2
fi
if [ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -lt 1 ]; then
   echo "--gpus-per-engine must be >= 1" >&2
   exit 2
fi
case "${ROUTER_POLICY}" in
   manual|consistent_hashing|cache_aware|round_robin|random|power_of_two|prefix_hash) ;;
   *) echo "Unsupported --router-policy ${ROUTER_POLICY}" >&2; exit 2 ;;
esac
case "${ROUTER_ASSIGNMENT_MODE}" in
   random|min_load|min_group) ;;
   *) echo "Unsupported --router-assignment-mode ${ROUTER_ASSIGNMENT_MODE}" >&2; exit 2 ;;
esac
if [ "${RETRIEVAL_CACHE_SIZE}" -lt 0 ]; then
   echo "--retrieval-cache-size must be >= 0" >&2
   exit 2
fi
python3 - "${EVAL_TRAJECTORY_SAMPLE_RATE}" <<'PY' || exit 2
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("--eval-trajectory-sample-rate must be a number in [0, 1]")
if not 0 <= value <= 1:
    raise SystemExit("--eval-trajectory-sample-rate must be in [0, 1]")
PY
if [ $((ROLLOUT_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
   echo "--gpus (${ROLLOUT_GPUS}) must be divisible by --gpus-per-engine (${ROLLOUT_NUM_GPUS_PER_ENGINE})." >&2
   exit 2
fi
case "${FUSED_HARNESS}" in
   rllm_deepresearch|rllm_dr|rllm-dr|deepresearch)
      if is_truthy "${RLLM_DR_USE_REFINE}" && [ -z "${RLLM_DR_REFINE_SERVER_URL}" ]; then
         echo "rllm_deepresearch requires --rllm-dr-refine-server-url (or RLLM_DR_USE_REFINE=0)" >&2
         exit 2
      fi
      ;;
esac
ROLLOUT_NUM_ENGINES=$((ROLLOUT_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE))
case "${FUSED_HARNESS}" in
   cot|bare) UNIFIED_SYSTEM_PROMPT=False ;;
esac
if [ -z "${PER_STEP_MAX_TOKENS:-}" ]; then
   if [ "${FUSED_HARNESS}" = "cot" ]; then
      PER_STEP_MAX_TOKENS="${EVAL_MAX_RESPONSE_LEN}"
   else
      PER_STEP_MAX_TOKENS=38000
   fi
fi
if [ "${FUSED_HARNESS}" = "cot" ]; then
   # Long CoT requests retain generation state for the full response, so keep
   # actual engine concurrency low even while the rollout queue stays buffered.
   EVAL_INITIAL_INFLIGHT_TASKS="${EVAL_INITIAL_INFLIGHT_TASKS:-$((4 * ROLLOUT_NUM_ENGINES))}"
   EVAL_MAX_INFLIGHT_TASKS="${EVAL_MAX_INFLIGHT_TASKS:-$((8 * ROLLOUT_NUM_ENGINES))}"
   SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-4}"
else
   EVAL_INITIAL_INFLIGHT_TASKS="${EVAL_INITIAL_INFLIGHT_TASKS:-384}"
   EVAL_MAX_INFLIGHT_TASKS="${EVAL_MAX_INFLIGHT_TASKS:-576}"
   SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-96}"
fi
if [ "${RETRIEVAL_CONCURRENCY}" -lt 1 ] || [ "${EVAL_INITIAL_INFLIGHT_TASKS}" -lt 1 ] || [ "${EVAL_MAX_INFLIGHT_TASKS}" -lt 1 ] || [ "${SGLANG_MAX_RUNNING_REQUESTS}" -lt 1 ]; then
   echo "retrieval concurrency, eval inflight limits, and SGLang running requests must be >= 1" >&2
   exit 2
fi
if [ "${EVAL_INITIAL_INFLIGHT_TASKS}" -gt "${EVAL_MAX_INFLIGHT_TASKS}" ]; then
   echo "--eval-initial-inflight-tasks must not exceed --eval-max-inflight-tasks" >&2
   exit 2
fi

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/experiments/logs/evals/${EXPERIMENT_NAME}}"
EVAL_CONFIG="${EVAL_CONFIG:-${LOG_ROOT}/eval_config.yaml}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-${LOG_ROOT}/normalized_benchmarks}"
DUMP_DETAILS="${DUMP_DETAILS:-${LOG_ROOT}/debug}"
mkdir -p "${LOG_ROOT}" "${EVAL_CACHE_DIR}" "${DUMP_DETAILS}"

PREFER_VERL_ARG="--prefer-verl"
if ! is_truthy "${PREFER_VERL}"; then
   PREFER_VERL_ARG="--no-prefer-verl"
fi

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 -m slime_plugins.evals.fused_benchmark_config \
   --benchmarks-root "${BENCHMARKS_ROOT}" \
   --output-config "${EVAL_CONFIG}" \
   --cache-dir "${EVAL_CACHE_DIR}" \
   --include "${INCLUDE_BENCHMARKS}" \
   --exclude "${EXCLUDE_BENCHMARKS}" \
   ${PREFER_VERL_ARG} \
   --limit-per-benchmark "${LIMIT_PER_BENCHMARK}" \
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}" \
   --temperature "${TEMPERATURE}" \
   --top-p "${TOP_P}" \
   --top-k "${TOP_K}" \
   --long-response-len "${EVAL_MAX_RESPONSE_LEN}"

PROMPT_DATA="$(python3 - "${EVAL_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
datasets = cfg["eval"]["datasets"]
print(datasets[0]["path"])
PY
)"

if is_truthy "${CLEANUP}"; then
   ray stop --force 2>/dev/null || true
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if ray job list --address="${RAY_DASHBOARD_ADDRESS}" >/dev/null 2>&1; then
   echo "Reusing existing Ray head at ${RAY_DASHBOARD_ADDRESS}"
else
   ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ROLLOUT_GPUS}" --num-cpus "${RAY_NUM_CPUS}" --disable-usage-stats
fi

export SCRIPT_DIR REPO_ROOT
export CUDA_HOME="/cm/shared/apps/cuda12.9"
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${BASE_DIR}/Megatron-LM}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-10000000000}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export RETRIEVAL_SERVER_URL="${RETRIEVAL_SERVER_URL:-http://10.2.152.50:65432}"
export RLLM_RETRIEVAL_MODE="${RLLM_RETRIEVAL_MODE:-hybrid}"
export RLLM_RETRIEVAL_MAX_WORDS="${RLLM_RETRIEVAL_MAX_WORDS:-1024}"
export RLLM_RETRIEVAL_CONCURRENCY="${RETRIEVAL_CONCURRENCY}"
export RLLM_RETRIEVAL_CACHE_SIZE="${RETRIEVAL_CACHE_SIZE}"
case "${FUSED_HARNESS}" in
   rllm_deepresearch|rllm_dr|rllm-dr|deepresearch) export RETRIEVAL_MAX_RESULTS="${RETRIEVAL_MAX_RESULTS:-10}" ;;
   *) export RETRIEVAL_MAX_RESULTS="${RETRIEVAL_MAX_RESULTS:-4}" ;;
esac
export RLLM_RETRIEVAL_SUMMARIZE="${RLLM_RETRIEVAL_SUMMARIZE:-0}"
export RLLM_DR_REFINE_SERVER_URL
export RLLM_DR_USE_REFINE
export DOCKER_HOST="${DOCKER_HOST:-tcp://10.2.152.50:2375}"
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.44}"
export FUSED_HARNESS="${FUSED_HARNESS}"
export FUSED_WEB_SEARCH_USER_PROMPT="${USER_PROMPT}"
export FUSED_MODEL_SERIES="${MODEL_SERIES}"
export FUSED_UNIFIED_SYSTEM_PROMPT="${UNIFIED_SYSTEM_PROMPT}"
export FUSED_DISABLE_THINKING="${DISABLE_THINKING}"
export FUSED_DISCARD_HISTORICAL_THINKING="${DISCARD_HISTORICAL_THINKING}"
export FUSED_MAX_STEPS="${MAX_STEPS}"
export FUSED_MCP_MAX_STEPS="${MCP_MAX_STEPS}"
export FUSED_WEB_SEARCH_MAX_STEPS="${WEB_SEARCH_MAX_STEPS}"
export FUSED_CLI_MAX_STEPS="${CLI_MAX_STEPS}"
export FUSED_TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT}"
export FUSED_EVAL_TRAJECTORY_TIMEOUT="${EVAL_TRAJECTORY_TIMEOUT}"
export PER_STEP_MAX_TOKENS
export SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH="${SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH:-4096}"
export SLIME_FUSED_TERMINAL_LOG_STYLE="${SLIME_FUSED_TERMINAL_LOG_STYLE:-both}"
export SLIME_FUSED_PROGRESS_LOGS="${SLIME_FUSED_PROGRESS_LOGS:-false}"
export SLIME_EPISODE_LOG_DIR="${LOG_ROOT}"
export SLIME_FUSED_EVAL_TRAJECTORY_SAMPLE_RATE="${EVAL_TRAJECTORY_SAMPLE_RATE}"
export SLIME_FUSED_EVAL_DUMP_FAILURES="${EVAL_DUMP_FAILURES}"
export SLIME_FUSED_EVAL_USE_SGLANG_SESSION="${NATIVE_SGLANG_SESSION}"
export SLIME_FUSED_SESSION_CONTROL_TIMEOUT="${SLIME_FUSED_SESSION_CONTROL_TIMEOUT:-120}"
export SLIME_FUSED_SESSION_IDLE_TIMEOUT="${SLIME_FUSED_SESSION_IDLE_TIMEOUT:-600}"
# Agent turns commonly spend several seconds in retrieval. Keep engine sockets
# alive across that gap, while making the client expiry strictly shorter so it
# never reuses a connection already closed by Uvicorn.
export SGLANG_TIMEOUT_KEEP_ALIVE="${SGLANG_TIMEOUT_KEEP_ALIVE:-120}"
export SLIME_HTTP_KEEPALIVE_EXPIRY="${SLIME_HTTP_KEEPALIVE_EXPIRY:-60}"

RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

keys = (
    "HYDRA_FULL_ERROR", "TOKENIZERS_PARALLELISM", "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S", "VLLM_WORKER_MULTIPROC_METHOD",
    "PYTORCH_CUDA_ALLOC_CONF", "RETRIEVAL_SERVER_URL", "RLLM_RETRIEVAL_MODE",
    "RLLM_RETRIEVAL_MAX_WORDS", "RLLM_RETRIEVAL_CONCURRENCY", "RLLM_RETRIEVAL_CACHE_SIZE",
    "RETRIEVAL_MAX_RESULTS", "RLLM_RETRIEVAL_SUMMARIZE",
    "RLLM_DR_REFINE_SERVER_URL", "RLLM_DR_REFINE_MODEL", "RLLM_DR_USE_REFINE",
    "RLLM_DR_MAX_TURNS", "RLLM_DR_MAX_TOKENS", "RLLM_DR_MAX_CONTENT_LENGTH",
    "RLLM_DR_RETRIEVAL_MAX_RETRIES", "RLLM_DR_REFINE_MAX_RETRIES",
    "RLLM_DR_TEMPERATURE", "RLLM_DR_TOP_P", "RLLM_DR_TOP_K", "REFINE_SERVER_URL",
    "DOCKER_HOST", "DOCKER_API_VERSION", "OPENROUTER_API_KEY", "OPENROUTER_SITE_URL",
    "OPENROUTER_APP_NAME", "FUSED_HARNESS", "FUSED_WEB_SEARCH_USER_PROMPT",
    "FUSED_MODEL_SERIES", "FUSED_UNIFIED_SYSTEM_PROMPT",
    "FUSED_DISABLE_THINKING", "FUSED_DISCARD_HISTORICAL_THINKING",
    "FUSED_MAX_STEPS", "FUSED_MCP_MAX_STEPS",
    "FUSED_WEB_SEARCH_MAX_STEPS", "FUSED_CLI_MAX_STEPS", "FUSED_TRAJECTORY_TIMEOUT",
    "FUSED_EVAL_TRAJECTORY_TIMEOUT", "PER_STEP_MAX_TOKENS",
    "SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH", "SLIME_FUSED_TERMINAL_LOG_STYLE",
    "SLIME_FUSED_PROGRESS_LOGS", "SLIME_EPISODE_LOG_DIR",
    "SLIME_FUSED_EVAL_TRAJECTORY_SAMPLE_RATE", "SLIME_FUSED_EVAL_DUMP_FAILURES",
    "SLIME_FUSED_EVAL_USE_SGLANG_SESSION", "SLIME_FUSED_SESSION_CONTROL_TIMEOUT",
    "SLIME_FUSED_SESSION_IDLE_TIMEOUT", "SGLANG_TIMEOUT_KEEP_ALIVE",
    "SLIME_HTTP_KEEPALIVE_EXPIRY",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["PYTHONPATH"] = f"{os.environ['MEGATRON_LM_PATH']}:{os.environ['REPO_ROOT']}:{os.environ['SCRIPT_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
print(json.dumps({"env_vars": env}))
PY
)"

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_DIR} (${MODEL_CONFIG}; series=${MODEL_SERIES})"
echo "Benchmarks root: ${BENCHMARKS_ROOT}"
echo "Eval config: ${EVAL_CONFIG}"
echo "Log root: ${LOG_ROOT}"
echo "GPUs: ${ROLLOUT_GPUS}; gpus_per_engine=${ROLLOUT_NUM_GPUS_PER_ENGINE}; engines=${ROLLOUT_NUM_ENGINES}"
echo "Harness: ${FUSED_HARNESS}; user_prompt=${USER_PROMPT}; disable_thinking=${DISABLE_THINKING}; discard_historical_thinking=${DISCARD_HISTORICAL_THINKING}; n=${N_SAMPLES_PER_EVAL_PROMPT}; per_step_max_tokens=${PER_STEP_MAX_TOKENS}"
echo "Concurrency: eval=${EVAL_INITIAL_INFLIGHT_TASKS}-${EVAL_MAX_INFLIGHT_TASKS} adaptive=${EVAL_ADAPTIVE_CONCURRENCY}; sglang_http_per_engine=${SGLANG_SERVER_CONCURRENCY}; sglang_running_per_engine=${SGLANG_MAX_RUNNING_REQUESTS}; retrieval=${RETRIEVAL_CONCURRENCY}"

EVAL_ADAPTIVE_CONCURRENCY_ARG="--eval-adaptive-concurrency"
if ! is_truthy "${EVAL_ADAPTIVE_CONCURRENCY}"; then
   EVAL_ADAPTIVE_CONCURRENCY_ARG="--no-eval-adaptive-concurrency"
fi

RAY_JOB_SUBMIT_ARGS=()
if ! is_truthy "${RAY_JOB_WAIT}"; then
   RAY_JOB_SUBMIT_ARGS+=(--no-wait)
fi

SAFE_EXPERIMENT_NAME="$(printf '%s' "${EXPERIMENT_NAME}" | tr -c '[:alnum:]_' '_' | cut -c1-120)"
RAY_SUBMISSION_ID="${RAY_SUBMISSION_ID:-eval_${SAFE_EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)}"

cd "${REPO_ROOT}"
ray job submit --address="${RAY_DASHBOARD_ADDRESS}" \
   --submission-id="${RAY_SUBMISSION_ID}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   "${RAY_JOB_SUBMIT_ARGS[@]}" \
   -- python3 -u train.py \
   --debug-rollout-only \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ROLLOUT_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --hf-checkpoint "${MODEL_DIR}" \
   "${MODEL_ARGS[@]}" \
   --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
   --eval-function-path slime.rollout.sglang_rollout.generate_rollout \
   --prompt-data "${PROMPT_DATA}" \
   --input-key input \
   --label-key ground_truth_answer \
   --metadata-key extra_info \
   --num-rollout 0 \
   --rollout-batch-size 1 \
   --n-samples-per-prompt 1 \
   --global-batch-size 1 \
   --rollout-max-context-len "${EVAL_MAX_CONTEXT_LEN}" \
   --rollout-max-prompt-len "${EVAL_MAX_PROMPT_LEN}" \
   --rollout-max-response-len "${EVAL_MAX_RESPONSE_LEN}" \
   --rollout-temperature "${TEMPERATURE}" \
   --rollout-top-p "${TOP_P}" \
   --rollout-top-k "${TOP_K}" \
   --custom-generate-function-path slime.rollout.fused_agent.generate.generate \
   --apply-chat-template \
   --rm-type benchmark_verifier \
   --eval-interval 1 \
   --eval-config "${EVAL_CONFIG}" \
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}" \
   --eval-temperature "${TEMPERATURE}" \
   --eval-top-p "${TOP_P}" \
   --eval-top-k "${TOP_K}" \
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}" \
   --eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}" \
   --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}" \
   --eval-initial-inflight-tasks "${EVAL_INITIAL_INFLIGHT_TASKS}" \
   --eval-max-inflight-tasks "${EVAL_MAX_INFLIGHT_TASKS}" \
   "${EVAL_ADAPTIVE_CONCURRENCY_ARG}" \
   --custom-eval-rollout-log-function-path slime_plugins.evals.results_table.log_eval_results_table \
   --dump-details "${DUMP_DETAILS}" \
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}" \
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}" \
   --router-policy "${ROUTER_POLICY}" \
   --router-assignment-mode "${ROUTER_ASSIGNMENT_MODE}" \
   --sglang-context-length "${EVAL_MAX_CONTEXT_LEN}" \
   --sglang-disable-custom-all-reduce \
   "${EXTRA_SLIME_ARGS[@]}"

if ! is_truthy "${RAY_JOB_WAIT}" && is_truthy "${RAY_JOB_FOLLOW_LOGS}"; then
   echo "Following Ray job logs for ${RAY_SUBMISSION_ID}"
   ray job logs --address="${RAY_DASHBOARD_ADDRESS}" --follow "${RAY_SUBMISSION_ID}" \
      | sed -u -E '/^[[:space:]]*$/d; /^\([^)]*pid=[0-9]+[^)]*\)[[:space:]]*$/d'
fi
