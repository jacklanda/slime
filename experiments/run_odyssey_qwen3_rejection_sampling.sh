#!/bin/bash
# Offline rejection sampling over slime's fused-agent rollout path.
#
# This launcher intentionally reuses the same custom generate hook, harness,
# prompts, tool schemas, tool parser, and runtime env knobs as
# experiments/train_qwen3_fused_agent_sync.sh, but runs slime with
# --debug-rollout-only so no actor update is performed.

set -euo pipefail

export PYTHONUNBUFFERED=1

usage() {
   cat <<'EOF'
Usage:
  bash experiments/run_qwen3_rejection_sampling.sh [options]

Core rejection-sampling options:
  --sample-n N                       Trajectories sampled per prompt. Default: 32
  --max-trajectory-per-problem N     Accepted trajectories kept per problem. Default: 1
  --min-sample-trial N               Minimum sampled trajectories before saving a problem. Default: 1
  --reward-threshold X               Minimum reward accepted. Default: 0.6
  --min-steps N                      Minimum fused trajectory steps accepted. Default: 2
  --certainty-filter BOOL            Drop groups with pass_rate 0 or 1. Default: False
  --valid-groups-per-shard N         Deprecated compatibility option; ignored.
  --max-candidate-groups-per-shard N Deprecated compatibility option; ignored.
                                     Shards always contain rollout-batch-size groups.

Data/output options:
  --train-files LIST                 Comma-separated parquet files. Defaults to the final Search and MCP train sets.
  --prompt-data PATH                 Prepared prompt parquet. Default: OUTPUT_DIR/fused_train.parquet
  --output-dir DIR                   Output directory. Default: experiments/rejection_sampling/NAME
  --experiment-name NAME             Run name.
  --max-batches N                    Number of rollout batches. Default: one pass over data.
  --episodes-dir DIR                 Per-batch trajectory shards. Default: OUTPUT_DIR/episodes
  --checkpointing BOOL               Save RS checkpoint after each completed batch. Default: True
  --checkpoint-path PATH             Checkpoint JSON path. Default: OUTPUT_DIR/latest_checkpoint.json
  --resume BOOL                      Resume from checkpoint/shards. Default: True
  --results-mode MODE               full, manifest, or auto. Large runs use manifest. Default: auto

Fused-agent options, aligned with train_qwen3_fused_agent_sync.sh:
  --harness NAME                     Fused prompt harness: bare, cot, react, gem, unified_gem.
  --unified-system-prompt            Select unified_gem harness unless --harness is set later.
  --no-unified-system-prompt         Select gem harness unless --harness is set later.
  --model PATH                       HF model path.
  --enable-yarn BOOL                 Enable static YaRN for long contexts. Default: false
  --yarn-factor X                    YaRN scale factor. Default: 1.0
  --yarn-original-max-position-embeddings N
                                     Qwen3 native context used by YaRN. Default: 32768
  --disable-thinking BOOL            FUSED_DISABLE_THINKING. Default: false
  --discard-historical-thinking BOOL Remove prior assistant <think> blocks before each new rollout step.
                                     Effective only when --disable-thinking is false. Default: false
  --mcp-disable-step-penalty BOOL    MCP verifier step-penalty env. Default: True
  --max-steps N                      Fused agent max steps. Default: 64
  --mcp-max-steps N                  MCP max steps. Default: 64
  --mcp-max-tool-calls-per-turn N    Maximum MCP calls per assistant turn. Default: 1
  --web-search-max-steps N           Web-search max steps. Default: 64
  --cli-max-steps N                  CLI max steps. Default: 64
  --trajectory-timeout N             Fused rollout-group timeout. Default: 300
  --per-step-max-tokens N            Max tokens per model turn. Default: 8192
  --max-tool-output-length N         Fused max tool output length. Default: 4096
  --terminal-log-style STYLE         progress, rollouts, or both. Default: both
  --show-rollout-progress-logs BOOL  Show periodic fused rollout progress logs. Default: false

Rollout/system options:
  --model-config NAME                scripts/models config. Default: qwen3-8B
  --rollout-batch-size N             Task groups per persisted shard. Default: 4096
  --max-prompt-length N              Max prompt tokens. Default: 15472
  --max-response-length N            Max response tokens. Default: 24576
  --rollout-gpus N                   Rollout GPUs. Default: 8
  --rollout-num-gpus-per-engine N    GPUs per SGLang engine. Defaults by model:
                                     Qwen3-4B/8B=1, Qwen3-14B=2, Qwen3-30B-A3B/32B=4
  --gpu-memory-utilization X         SGLang static memory fraction. Default: 0.6
  --sglang-server-concurrency N      SGLang server concurrency. Default: 64
  --sglang-max-running-requests N    SGLang max running requests. Default: 64
  --sglang-router-request-timeout-secs N
                                     SGLang router request timeout. Default: 21600
  --fully-async-adaptive-concurrency BOOL
                                     Adapt in-flight prompt groups from SGLang load. Default: true
  --fully-async-initial-group-concurrency N
                                     Initial in-flight groups. Default: 4x rollout engine count
  --fully-async-max-group-concurrency N
                                     Maximum in-flight groups. Default: 8x rollout engine count
  --fully-async-concurrency-step N   Groups added/removed per adjustment. Default: engine count
  --fully-async-concurrency-poll-interval X
                                     Seconds between load samples. Default: 10
  --ray-num-cpus N                   Ray CPU resources. Default: 64
  --ray-job-wait 0|1                 Wait for Ray job submit. Default: 1
  -h, --help                         Show this help.
EOF
}

is_truthy() {
   case "${1}" in
      1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
      *) return 1 ;;
   esac
}

TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d%H%M%S")}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
BASE_DIR="$(cd -- "${REPO_ROOT}/.." &>/dev/null && pwd)"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-offline-rs-slime-fused-${TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/rejection_sampling/${EXPERIMENT_NAME}}"
RUNS_ROOT="${RUNS_ROOT:-/share/nlp/share/gem/runs}"
RUN_ROOT="${RUN_ROOT:-${RUNS_ROOT}/${EXPERIMENT_NAME}}"
MCP_ENV_ROOT="${MCP_ENV_ROOT:-${RUN_ROOT}/cache/mcp_envs}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_DIR}/logs}"
EPISODE_LOG_DIR_EXPLICIT=0
if [ -n "${EPISODE_LOG_DIR:-}" ]; then
   EPISODE_LOG_DIR_EXPLICIT=1
else
   EPISODE_LOG_DIR="${OUTPUT_DIR}/episodes"
fi
DUMP_DETAILS="${DUMP_DETAILS:-${OUTPUT_DIR}/debug}"
CHECKPOINT_PATH_EXPLICIT=0
if [ -n "${OFFLINE_RS_CHECKPOINT_PATH:-}" ]; then
   CHECKPOINT_PATH_EXPLICIT=1
else
   OFFLINE_RS_CHECKPOINT_PATH="${OUTPUT_DIR}/latest_checkpoint.json"
fi
PROMPT_DATA_EXPLICIT=0
if [ -n "${PROMPT_DATA:-}" ]; then
   PROMPT_DATA_EXPLICIT=1
else
   PROMPT_DATA="${OUTPUT_DIR}/fused_train.parquet"
fi

MODEL_CONFIG="${MODEL_CONFIG:-qwen3-4B}"
#MODEL_CONFIG="${MODEL_CONFIG:-qwen3-8B}"
#MODEL_CONFIG="${MODEL_CONFIG:-qwen3-14B}"
MODEL_DIR="${MODEL_DIR:-/share/nlp/share/plm/Qwen3-4B}"
#MODEL_DIR="${MODEL_DIR:-/share/nlp/share/plm/Qwen3-8B}"
#MODEL_DIR="${MODEL_DIR:-/share/nlp/share/plm/Qwen3-14B}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${BASE_DIR}/Megatron-LM}"

SAMPLE_N="${SAMPLE_N:-32}"
MAX_TRAJECTORY_PER_PROBLEM="${MAX_TRAJECTORY_PER_PROBLEM:-32}"
MIN_SAMPLE_TRIAL="${MIN_SAMPLE_TRIAL:-1}"
REWARD_THRESHOLD="${REWARD_THRESHOLD:-1.0}"
MIN_STEPS="${MIN_STEPS:-3}"
CERTAINTY_FILTER="${CERTAINTY_FILTER:-False}"
VALID_GROUPS_PER_SHARD="${VALID_GROUPS_PER_SHARD:-}"
MAX_CANDIDATE_GROUPS_PER_SHARD="${MAX_CANDIDATE_GROUPS_PER_SHARD:-}"
OFFLINE_RS_CHECKPOINT_ENABLE="${OFFLINE_RS_CHECKPOINT_ENABLE:-True}"
OFFLINE_RS_RESUME="${OFFLINE_RS_RESUME:-True}"
OFFLINE_RS_RESULTS_MODE="${OFFLINE_RS_RESULTS_MODE:-auto}"

DEFAULT_TRAIN_FILES=(
   "${SCRIPT_DIR}/artifacts/search_data_final/train.parquet"
   "${SCRIPT_DIR}/artifacts/mcp_data_final/train.parquet"
)
TRAIN_FILE_PATHS=("${DEFAULT_TRAIN_FILES[@]}")
SHUFFLE_TRAIN_DATA="${SHUFFLE_TRAIN_DATA:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

UNIFIED_SYSTEM_PROMPT="${UNIFIED_SYSTEM_PROMPT:-false}"
DISABLE_THINKING="${DISABLE_THINKING:-false}"
DISCARD_HISTORICAL_THINKING="${DISCARD_HISTORICAL_THINKING:-false}"
FUSED_HARNESS="${FUSED_HARNESS:-gem}"
harness_explicit=false

ENABLE_YARN="${ENABLE_YARN:-false}"
YARN_FACTOR="${YARN_FACTOR:-1.0}"
YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-32768}"
MAX_STEPS="${MAX_STEPS:-64}"
MCP_MAX_STEPS="${MCP_MAX_STEPS:-64}"
MCP_MAX_TOOL_CALLS_PER_TURN="${MCP_MAX_TOOL_CALLS_PER_TURN:-1}"
WEB_SEARCH_MAX_STEPS="${WEB_SEARCH_MAX_STEPS:-64}"
CLI_MAX_STEPS="${CLI_MAX_STEPS:-64}"
TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT:-7200}"
EVAL_TRAJECTORY_TIMEOUT="${EVAL_TRAJECTORY_TIMEOUT:-300}"
PER_STEP_MAX_TOKENS="${PER_STEP_MAX_TOKENS:-8192}"
MAX_TOOL_OUTPUT_LENGTH="${MAX_TOOL_OUTPUT_LENGTH:-4096}"
TERMINAL_LOG_STYLE="${TERMINAL_LOG_STYLE:-both}"
SHOW_ROLLOUT_PROGRESS_LOGS="${SHOW_ROLLOUT_PROGRESS_LOGS:-false}"
ACCEPTED_GROUP_UPDATE_MAX_GROUPS="${ACCEPTED_GROUP_UPDATE_MAX_GROUPS:-64}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-15472}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-24576}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-40960}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4096}"
NUM_ROLLOUT="${NUM_ROLLOUT:-}"
NUM_EPOCH="${NUM_EPOCH:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-${GPU_MEMORY_UTILIZATION:-0.9}}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-128}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-128}"
SGLANG_ROUTER_REQUEST_TIMEOUT_SECS="${SGLANG_ROUTER_REQUEST_TIMEOUT_SECS:-21600}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY:-true}"
FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY="${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY:-}"
FULLY_ASYNC_MAX_GROUP_CONCURRENCY="${FULLY_ASYNC_MAX_GROUP_CONCURRENCY:-}"
FULLY_ASYNC_CONCURRENCY_STEP="${FULLY_ASYNC_CONCURRENCY_STEP:-}"
FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL="${FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL:-10}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
RAY_JOB_WAIT="${RAY_JOB_WAIT:-1}"

ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-slime.rollout.fully_async_rollout.generate_rollout_fully_async}"
CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-slime.rollout.fused_agent.generate.generate}"

while [ "$#" -gt 0 ]; do
   case "$1" in
      --sample-n) SAMPLE_N="${2:?Missing value for --sample-n}"; shift 2 ;;
      --max-trajectory-per-problem) MAX_TRAJECTORY_PER_PROBLEM="${2:?Missing value for --max-trajectory-per-problem}"; shift 2 ;;
      --min-sample-trial) MIN_SAMPLE_TRIAL="${2:?Missing value for --min-sample-trial}"; shift 2 ;;
      --reward-threshold) REWARD_THRESHOLD="${2:?Missing value for --reward-threshold}"; shift 2 ;;
      --min-steps) MIN_STEPS="${2:?Missing value for --min-steps}"; shift 2 ;;
      --certainty-filter) CERTAINTY_FILTER="${2:?Missing value for --certainty-filter}"; shift 2 ;;
      --valid-groups-per-shard) VALID_GROUPS_PER_SHARD="${2:?Missing value for --valid-groups-per-shard}"; shift 2 ;;
      --max-candidate-groups-per-shard) MAX_CANDIDATE_GROUPS_PER_SHARD="${2:?Missing value for --max-candidate-groups-per-shard}"; shift 2 ;;
      --train-files) IFS=',' read -r -a TRAIN_FILE_PATHS <<< "${2:?Missing value for --train-files}"; shift 2 ;;
      --prompt-data) PROMPT_DATA="${2:?Missing value for --prompt-data}"; PROMPT_DATA_EXPLICIT=1; shift 2 ;;
      --output-dir)
         OUTPUT_DIR="${2:?Missing value for --output-dir}"
         LOG_ROOT="${OUTPUT_DIR}/logs"
         DUMP_DETAILS="${OUTPUT_DIR}/debug"
         if [ "${EPISODE_LOG_DIR_EXPLICIT}" = "0" ]; then
            EPISODE_LOG_DIR="${OUTPUT_DIR}/episodes"
         fi
         if [ "${CHECKPOINT_PATH_EXPLICIT}" = "0" ]; then
            OFFLINE_RS_CHECKPOINT_PATH="${OUTPUT_DIR}/latest_checkpoint.json"
         fi
         if [ "${PROMPT_DATA_EXPLICIT}" = "0" ]; then
            PROMPT_DATA="${OUTPUT_DIR}/fused_train.parquet"
         fi
         shift 2
         ;;
      --episodes-dir) EPISODE_LOG_DIR="${2:?Missing value for --episodes-dir}"; EPISODE_LOG_DIR_EXPLICIT=1; shift 2 ;;
      --checkpointing) OFFLINE_RS_CHECKPOINT_ENABLE="${2:?Missing value for --checkpointing}"; shift 2 ;;
      --checkpoint-path) OFFLINE_RS_CHECKPOINT_PATH="${2:?Missing value for --checkpoint-path}"; CHECKPOINT_PATH_EXPLICIT=1; shift 2 ;;
      --resume) OFFLINE_RS_RESUME="${2:?Missing value for --resume}"; shift 2 ;;
      --results-mode) OFFLINE_RS_RESULTS_MODE="${2:?Missing value for --results-mode}"; shift 2 ;;
      --experiment-name) EXPERIMENT_NAME="${2:?Missing value for --experiment-name}"; shift 2 ;;
      --max-batches|--num-rollout) NUM_ROLLOUT="${2:?Missing value for --max-batches}"; shift 2 ;;
      --harness) FUSED_HARNESS="${2:?Missing value for --harness}"; harness_explicit=true; shift 2 ;;
      --unified-system-prompt) UNIFIED_SYSTEM_PROMPT=True; if [ "${harness_explicit}" = "false" ]; then FUSED_HARNESS=unified_gem; fi; shift ;;
      --no-unified-system-prompt) UNIFIED_SYSTEM_PROMPT=False; if [ "${harness_explicit}" = "false" ]; then FUSED_HARNESS=gem; fi; shift ;;
      --model) MODEL_DIR="${2:?Missing value for --model}"; shift 2 ;;
      --enable-yarn) ENABLE_YARN="${2:?Missing value for --enable-yarn}"; shift 2 ;;
      --yarn-factor) YARN_FACTOR="${2:?Missing value for --yarn-factor}"; shift 2 ;;
      --yarn-original-max-position-embeddings) YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${2:?Missing value for --yarn-original-max-position-embeddings}"; shift 2 ;;
      --disable-thinking) DISABLE_THINKING="${2:?Missing value for --disable-thinking}"; shift 2 ;;
      --discard-historical-thinking) DISCARD_HISTORICAL_THINKING="${2:?Missing value for --discard-historical-thinking}"; shift 2 ;;
      --mcp-disable-step-penalty) RLLM_MCP_DISABLE_STEP_PENALTY="${2:?Missing value for --mcp-disable-step-penalty}"; shift 2 ;;
      --max-steps) MAX_STEPS="${2:?Missing value for --max-steps}"; shift 2 ;;
      --mcp-max-steps) MCP_MAX_STEPS="${2:?Missing value for --mcp-max-steps}"; shift 2 ;;
      --mcp-max-tool-calls-per-turn) MCP_MAX_TOOL_CALLS_PER_TURN="${2:?Missing value for --mcp-max-tool-calls-per-turn}"; shift 2 ;;
      --web-search-max-steps) WEB_SEARCH_MAX_STEPS="${2:?Missing value for --web-search-max-steps}"; shift 2 ;;
      --cli-max-steps) CLI_MAX_STEPS="${2:?Missing value for --cli-max-steps}"; shift 2 ;;
      --trajectory-timeout) TRAJECTORY_TIMEOUT="${2:?Missing value for --trajectory-timeout}"; shift 2 ;;
      --per-step-max-tokens) PER_STEP_MAX_TOKENS="${2:?Missing value for --per-step-max-tokens}"; shift 2 ;;
      --max-tool-output-length) MAX_TOOL_OUTPUT_LENGTH="${2:?Missing value for --max-tool-output-length}"; shift 2 ;;
      --terminal-log-style) TERMINAL_LOG_STYLE="${2:?Missing value for --terminal-log-style}"; shift 2 ;;
      --show-rollout-progress-logs) SHOW_ROLLOUT_PROGRESS_LOGS="${2:?Missing value for --show-rollout-progress-logs}"; shift 2 ;;
      --model-config) MODEL_CONFIG="${2:?Missing value for --model-config}"; shift 2 ;;
      --rollout-batch-size) ROLLOUT_BATCH_SIZE="${2:?Missing value for --rollout-batch-size}"; shift 2 ;;
      --max-prompt-length) MAX_PROMPT_LENGTH="${2:?Missing value for --max-prompt-length}"; shift 2 ;;
      --max-response-length) MAX_RESPONSE_LENGTH="${2:?Missing value for --max-response-length}"; shift 2 ;;
      --rollout-gpus) ROLLOUT_GPUS="${2:?Missing value for --rollout-gpus}"; shift 2 ;;
      --rollout-num-gpus-per-engine) ROLLOUT_NUM_GPUS_PER_ENGINE="${2:?Missing value for --rollout-num-gpus-per-engine}"; shift 2 ;;
      --gpu-memory-utilization) SGLANG_MEM_FRACTION_STATIC="${2:?Missing value for --gpu-memory-utilization}"; shift 2 ;;
      --sglang-server-concurrency) SGLANG_SERVER_CONCURRENCY="${2:?Missing value for --sglang-server-concurrency}"; shift 2 ;;
      --sglang-max-running-requests) SGLANG_MAX_RUNNING_REQUESTS="${2:?Missing value for --sglang-max-running-requests}"; shift 2 ;;
      --sglang-router-request-timeout-secs) SGLANG_ROUTER_REQUEST_TIMEOUT_SECS="${2:?Missing value for --sglang-router-request-timeout-secs}"; shift 2 ;;
      --fully-async-adaptive-concurrency) FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${2:?Missing value for --fully-async-adaptive-concurrency}"; shift 2 ;;
      --fully-async-initial-group-concurrency) FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY="${2:?Missing value for --fully-async-initial-group-concurrency}"; shift 2 ;;
      --fully-async-max-group-concurrency) FULLY_ASYNC_MAX_GROUP_CONCURRENCY="${2:?Missing value for --fully-async-max-group-concurrency}"; shift 2 ;;
      --fully-async-concurrency-step) FULLY_ASYNC_CONCURRENCY_STEP="${2:?Missing value for --fully-async-concurrency-step}"; shift 2 ;;
      --fully-async-concurrency-poll-interval) FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL="${2:?Missing value for --fully-async-concurrency-poll-interval}"; shift 2 ;;
      --ray-num-cpus) RAY_NUM_CPUS="${2:?Missing value for --ray-num-cpus}"; shift 2 ;;
      --ray-job-wait) RAY_JOB_WAIT="${2:?Missing value for --ray-job-wait}"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
   esac
done

case "${TERMINAL_LOG_STYLE}" in
   progress|rollouts|both) ;;
   *) echo "Invalid TERMINAL_LOG_STYLE=${TERMINAL_LOG_STYLE}; expected progress, rollouts, or both." >&2; exit 2 ;;
esac

if [ ! -d "${MODEL_DIR}" ]; then
   echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
   exit 1
fi
if [ ! -d "${MEGATRON_LM_PATH}" ]; then
   echo "MEGATRON_LM_PATH does not exist: ${MEGATRON_LM_PATH}" >&2
   exit 1
fi

source "${REPO_ROOT}/scripts/models/${MODEL_CONFIG}.sh"

case "${MODEL_CONFIG,,}" in
   qwen3-4b|qwen3-4b-*|qwen3-8b|qwen3-8b-*)
      DEFAULT_ROLLOUT_NUM_GPUS_PER_ENGINE=1
      ;;
   qwen3-14b|qwen3-14b-*)
      DEFAULT_ROLLOUT_NUM_GPUS_PER_ENGINE=2
      ;;
   qwen3-30b-a3b|qwen3-30b-a3b-*|qwen3-32b|qwen3-32b-*)
      DEFAULT_ROLLOUT_NUM_GPUS_PER_ENGINE=4
      ;;
   *)
      if [ -z "${ROLLOUT_NUM_GPUS_PER_ENGINE:-}" ]; then
         echo "MODEL_CONFIG=${MODEL_CONFIG} has no default rollout tensor-parallel size; set ROLLOUT_NUM_GPUS_PER_ENGINE explicitly." >&2
         exit 2
      fi
      DEFAULT_ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE}"
      ;;
esac
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${DEFAULT_ROLLOUT_NUM_GPUS_PER_ENGINE}}"

if [ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -lt 1 ] \
   || [ $((ROLLOUT_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
   echo "ROLLOUT_GPUS=${ROLLOUT_GPUS} must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}" >&2
   exit 2
fi
ROLLOUT_ENGINE_COUNT=$((ROLLOUT_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE))
FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY="${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY:-$((ROLLOUT_ENGINE_COUNT * 4))}"
# Rejection sampling has long tool/retrieval gaps between model turns. Keep
# enough trajectories in flight to refill every SGLang engine while allowing
# the adaptive controller to back off under KV-cache pressure. Training
# launchers do not use this script and retain their own defaults.
FULLY_ASYNC_MAX_GROUP_CONCURRENCY="${FULLY_ASYNC_MAX_GROUP_CONCURRENCY:-$((ROLLOUT_ENGINE_COUNT * 8))}"
FULLY_ASYNC_CONCURRENCY_STEP="${FULLY_ASYNC_CONCURRENCY_STEP:-${ROLLOUT_ENGINE_COUNT}}"
if [ "${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY}" -lt 1 ] \
   || [ "${FULLY_ASYNC_MAX_GROUP_CONCURRENCY}" -lt "${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY}" ] \
   || [ "${FULLY_ASYNC_CONCURRENCY_STEP}" -lt 1 ]; then
   echo "Fully-async concurrency requires 1 <= initial <= max and step >= 1; got initial=${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY}, max=${FULLY_ASYNC_MAX_GROUP_CONCURRENCY}, step=${FULLY_ASYNC_CONCURRENCY_STEP}." >&2
   exit 2
fi

if is_truthy "${ENABLE_YARN}"; then
   python3 - "${YARN_FACTOR}" "${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}" "${MAX_CONTEXT_LEN}" <<'PY' || exit 2
import math
import sys

factor = float(sys.argv[1])
original_context = int(sys.argv[2])
target_context = int(sys.argv[3])
if not math.isfinite(factor) or factor <= 1:
    raise SystemExit(f"YARN_FACTOR must be finite and greater than 1, got {factor}")
if original_context <= 0:
    raise SystemExit(
        "YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS must be positive, "
        f"got {original_context}"
    )
if target_context > int(factor * original_context):
    raise SystemExit(
        f"MAX_CONTEXT_LEN={target_context} exceeds the configured YaRN capacity "
        f"{int(factor * original_context)}"
    )
PY
   MODEL_ARGS+=(
      --use-yarn-rope
      --yarn-rope-scaling-factor "${YARN_FACTOR}"
      --yarn-original-max-position-embeddings "${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}"
   )
fi

RESOLVED_TRAIN_FILES=()
for train_file in "${TRAIN_FILE_PATHS[@]}"; do
   if [ -f "${train_file}" ]; then
      RESOLVED_TRAIN_FILES+=("${train_file}")
      continue
   fi
   rllm_fallback="${BASE_DIR}/rllm/${train_file#${REPO_ROOT}/}"
   if [ -f "${rllm_fallback}" ]; then
      RESOLVED_TRAIN_FILES+=("${rllm_fallback}")
      continue
   fi
   rllm_experiments_fallback="${BASE_DIR}/rllm/${train_file#${SCRIPT_DIR}/}"
   if [ -f "${rllm_experiments_fallback}" ]; then
      RESOLVED_TRAIN_FILES+=("${rllm_experiments_fallback}")
      continue
   fi
   echo "Training data file does not exist: ${train_file}" >&2
   exit 1
done

mkdir -p "${OUTPUT_DIR}" "$(dirname "${PROMPT_DATA}")" "${DUMP_DETAILS}" "${MCP_ENV_ROOT}"
python3 - "${PROMPT_DATA}" "${SHUFFLE_TRAIN_DATA}" "${SHUFFLE_SEED}" "${RESOLVED_TRAIN_FILES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

output = Path(sys.argv[1])
shuffle = sys.argv[2].lower() in {"1", "true", "yes", "on"}
seed = int(sys.argv[3])
paths = [Path(p) for p in sys.argv[4:]]

fingerprint_path = output.with_suffix(output.suffix + ".fingerprint.json")
fingerprint = {
    "version": 1,
    "shuffle": shuffle,
    "seed": seed,
    "inputs": [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ],
}
fingerprint["digest"] = hashlib.sha256(
    json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
if output.is_file() and fingerprint_path.is_file():
    try:
        cached_fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached_fingerprint = None
    if cached_fingerprint == fingerprint:
        print(f"Reusing cached fused train parquet: {output}")
        raise SystemExit(0)

tables = [pq.read_table(path) for path in paths]
schema = pa.unify_schemas([table.schema for table in tables], promote_options="permissive")
names = schema.names
types = {field.name: field.type for field in schema}

aligned = []
for table in tables:
    arrays = []
    for name in names:
        if name in table.column_names:
            column = table[name]
            if not column.type.equals(types[name]):
                column = column.cast(types[name])
            arrays.append(column)
        else:
            arrays.append(pa.nulls(table.num_rows, type=types[name]))
    aligned.append(pa.table(arrays, names=names))

combined = pa.concat_tables(aligned, promote_options="default")
if shuffle and combined.num_rows:
    indices = pa.array(np.random.default_rng(seed).permutation(combined.num_rows), type=pa.int64())
    combined = combined.take(indices)

temporary_output = output.with_suffix(output.suffix + ".tmp")
pq.write_table(combined, temporary_output)
os.replace(temporary_output, output)
temporary_fingerprint = fingerprint_path.with_suffix(fingerprint_path.suffix + ".tmp")
temporary_fingerprint.write_text(
    json.dumps(fingerprint, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary_fingerprint, fingerprint_path)
print(f"Wrote fused train parquet: {output}")
print(f"Rows: {combined.num_rows}")
for path, table in zip(paths, tables):
    print(f"  {path}: {table.num_rows}")
PY

TRAIN_NUM_ROWS=$(python3 - "${PROMPT_DATA}" <<'PY'
import sys
import pyarrow.parquet as pq
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)
if [ "${TRAIN_NUM_ROWS}" -le 0 ]; then
   echo "PROMPT_DATA has no rows: ${PROMPT_DATA}" >&2
   exit 2
fi
AUTO_NUM_ROLLOUT=$(( (TRAIN_NUM_ROWS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE * NUM_EPOCH ))
NUM_ROLLOUT="${NUM_ROLLOUT:-${AUTO_NUM_ROLLOUT}}"

case "${OFFLINE_RS_RESULTS_MODE}" in
   auto)
      expected_prompts=$((NUM_ROLLOUT * ROLLOUT_BATCH_SIZE))
      if [ "${expected_prompts}" -gt "${TRAIN_NUM_ROWS}" ]; then
         expected_prompts="${TRAIN_NUM_ROWS}"
      fi
      if [ $((expected_prompts * SAMPLE_N)) -gt 100000 ]; then
         RESOLVED_RESULTS_MODE=manifest
      else
         RESOLVED_RESULTS_MODE=full
      fi
      ;;
   full|manifest) RESOLVED_RESULTS_MODE="${OFFLINE_RS_RESULTS_MODE}" ;;
   *)
      echo "--results-mode must be full, manifest, or auto; got ${OFFLINE_RS_RESULTS_MODE}" >&2
      exit 2
      ;;
esac

mkdir -p "${EPISODE_LOG_DIR}" "$(dirname "${OFFLINE_RS_CHECKPOINT_PATH}")"
RESUME_START_ROLLOUT_ID=0
if is_truthy "${OFFLINE_RS_RESUME}"; then
   RESUME_START_ROLLOUT_ID=$(python3 - "${DUMP_DETAILS}/rollout_data" "${NUM_ROLLOUT}" <<'PY'
import sys
from pathlib import Path

rollout_dir = Path(sys.argv[1])
num_rollout = int(sys.argv[2])

finished = set()
if rollout_dir.exists():
    for path in rollout_dir.glob("*.pt"):
        try:
            if path.stat().st_size > 0:
                finished.add(int(path.stem))
        except ValueError:
            pass

continuous_next = 0
while continuous_next < num_rollout and continuous_next in finished:
    continuous_next += 1
print(min(num_rollout, continuous_next))
PY
)
fi
if [ "${RESUME_START_ROLLOUT_ID}" -gt 0 ]; then
   echo "Resume enabled: continuing from rollout batch ${RESUME_START_ROLLOUT_ID}/${NUM_ROLLOUT}"
else
   echo "Resume start: rollout batch 0/${NUM_ROLLOUT}"
fi
if [ "${RESUME_START_ROLLOUT_ID}" -ge "${NUM_ROLLOUT}" ]; then
   echo "All rollout batches already present; skipping Ray rollout and running final merge."
   SKIP_RAY_ROLLOUT=1
else
   SKIP_RAY_ROLLOUT="${SKIP_RAY_ROLLOUT:-0}"
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
HAS_NVLINK=$([ "${NVLINK_COUNT}" -gt 0 ] && echo 1 || echo 0)

export CUDA_HOME="/cm/shared/apps/cuda12.9"
export SCRIPT_DIR REPO_ROOT MEGATRON_LM_PATH HAS_NVLINK
export SLIME_EPISODE_LOG_DIR="${EPISODE_LOG_DIR}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NUM_GPUS="${NUM_GPUS:-${ROLLOUT_GPUS}}"

# These nodes currently have a CUDA 12.9 driver but a CUDA 13.1 CuTe DSL
# runtime. Use FlashInfer's CUDA JIT norm fallback until the node stacks align.
if [[ "$(hostname -s)" == "dgx-hyperplane17" || "$(hostname -s)" == "hgx-hyperplane09" ]]; then
   export FLASHINFER_USE_CUDA_NORM=1
   export LD_LIBRARY_PATH="/home/liuyang/app/anaconda3/envs/slime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
   echo "Enabled the FlashInfer CUDA norm fallback for $(hostname -s)."
fi
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export RAY_WARN_BLOCKING_GET_INSIDE_ASYNC="${RAY_WARN_BLOCKING_GET_INSIDE_ASYNC:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-10000000000}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export RUNS_ROOT RUN_ROOT MCP_ENV_ROOT
export SLIME_MCP_ENV_ROOT="${MCP_ENV_ROOT}"
export SLIME_MCP_ENV_COPY_CONCURRENCY="${SLIME_MCP_ENV_COPY_CONCURRENCY:-32}"
# Local MCP trajectory isolation is enabled by default. Keep the bounded
# process-pool knobs in the Ray runtime so rejection sampling workers receive
# the same isolation and fault-timeout policy as the launcher process.
export SLIME_LOCAL_MCP_PROCESS_ISOLATION="${SLIME_LOCAL_MCP_PROCESS_ISOLATION:-true}"
export SLIME_LOCAL_MCP_PROCESS_WORKERS="${SLIME_LOCAL_MCP_PROCESS_WORKERS:-}"
export SLIME_LOCAL_MCP_PROCESS_START_METHOD="${SLIME_LOCAL_MCP_PROCESS_START_METHOD:-forkserver}"
export SLIME_LOCAL_MCP_PROCESS_TIMEOUT="${SLIME_LOCAL_MCP_PROCESS_TIMEOUT:-120}"
export SLIME_LOCAL_MCP_PROCESS_WARM_TIMEOUT="${SLIME_LOCAL_MCP_PROCESS_WARM_TIMEOUT:-60}"

# Keep these defaults byte-for-byte aligned with train_fused_agent_sync.sh where
# they define agent behavior. The rollout code consumes this env to choose
# prompt harness, system prompt, tool schemas, parser behavior, and service IO.
export RETRIEVAL_SERVER_URL="${RETRIEVAL_SERVER_URL:-http://10.2.152.50:65432}"
export RLLM_RETRIEVAL_MODE="${RLLM_RETRIEVAL_MODE:-hybrid}"
export RLLM_RETRIEVAL_CONCURRENCY="${RLLM_RETRIEVAL_CONCURRENCY:-512}"
export RLLM_RETRIEVAL_CACHE_SIZE="${RLLM_RETRIEVAL_CACHE_SIZE:-4096}"
export RLLM_RETRIEVAL_MAX_WORDS="${RLLM_RETRIEVAL_MAX_WORDS:-1024}"
export RETRIEVAL_MAX_RESULTS="${RETRIEVAL_MAX_RESULTS:-${RLLM_RETRIEVAL_MAX_RESULTS:-8}}"
export FUSED_WEBQA_MIN_UNIQUE_SEARCHES="${FUSED_WEBQA_MIN_UNIQUE_SEARCHES:-1}"
export RLLM_RETRIEVAL_SUMMARIZE="${RLLM_RETRIEVAL_SUMMARIZE:-0}"
export RLLM_RETRIEVAL_RETRY_BUDGET="${RLLM_RETRIEVAL_RETRY_BUDGET:-8}"
export RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET="${RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET:-32}"
export RLLM_RETRIEVAL_LEXRANK_FALLBACK=0
export RLLM_RETRIEVAL_LEXRANK_MAX_WORDS="${RLLM_RETRIEVAL_LEXRANK_MAX_WORDS:-512}"
export RLLM_RETRIEVAL_LEXRANK_MAX_SENTENCES="${RLLM_RETRIEVAL_LEXRANK_MAX_SENTENCES:-32}"
export RLLM_RETRIEVAL_LEXRANK_MAX_INPUT_SENTENCES="${RLLM_RETRIEVAL_LEXRANK_MAX_INPUT_SENTENCES:-128}"
export RLLM_RETRIEVAL_LEXRANK_MULTIPROCESSING="${RLLM_RETRIEVAL_LEXRANK_MULTIPROCESSING:-1}"
export RLLM_RETRIEVAL_LEXRANK_WORKERS="${RLLM_RETRIEVAL_LEXRANK_WORKERS:-32}"
export DOCKER_HOST="${DOCKER_HOST:-tcp://10.2.152.50:2375}"
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.44}"
export RLLM_MCP_MIN_NOFILE="${RLLM_MCP_MIN_NOFILE:-4096}"
export RLLM_MCP_FD_THROTTLE_THRESHOLD="${RLLM_MCP_FD_THROTTLE_THRESHOLD:-4096}"
export RLLM_MCP_INIT_TIMEOUT="${RLLM_MCP_INIT_TIMEOUT:-64}"
export RLLM_MCP_START_RETRIES="${RLLM_MCP_START_RETRIES:-8}"
export RLLM_MCP_START_WAIT_TIMEOUT="${RLLM_MCP_START_WAIT_TIMEOUT:-0}"
export RLLM_MCP_TOOL_TIMEOUT="${RLLM_MCP_TOOL_TIMEOUT:-8}"
export RLLM_MCP_MAX_ACTIVE_SERVERS="${RLLM_MCP_MAX_ACTIVE_SERVERS:-128}"
export RLLM_MCP_PREFILTER_WORKERS="${RLLM_MCP_PREFILTER_WORKERS:-8}"
export RLLM_MCP_DISABLE_STEP_PENALTY="${RLLM_MCP_DISABLE_STEP_PENALTY:-True}"
# Reject verifier results that report an internal tool error while claiming success.
# This is rejection-sampling-only; ordinary training launchers leave it disabled.
export SLIME_MCP_STRICT_VERIFIER="${SLIME_MCP_STRICT_VERIFIER:-true}"
export FUSED_HARNESS="${FUSED_HARNESS}"
export FUSED_UNIFIED_SYSTEM_PROMPT="${UNIFIED_SYSTEM_PROMPT}"
export FUSED_DISABLE_THINKING="${DISABLE_THINKING}"
export FUSED_DISCARD_HISTORICAL_THINKING="${DISCARD_HISTORICAL_THINKING}"
export FUSED_MAX_STEPS="${FUSED_MAX_STEPS:-${MAX_STEPS}}"
export FUSED_MCP_MAX_STEPS="${FUSED_MCP_MAX_STEPS:-${MCP_MAX_STEPS}}"
export FUSED_MCP_MAX_TOOL_CALLS_PER_TURN="${MCP_MAX_TOOL_CALLS_PER_TURN}"
export FUSED_WEB_SEARCH_MAX_STEPS="${FUSED_WEB_SEARCH_MAX_STEPS:-${WEB_SEARCH_MAX_STEPS}}"
export FUSED_CLI_MAX_STEPS="${CLI_MAX_STEPS}"
export FUSED_TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT}"
export FUSED_EVAL_TRAJECTORY_TIMEOUT="${EVAL_TRAJECTORY_TIMEOUT}"
export PER_STEP_MAX_TOKENS="${PER_STEP_MAX_TOKENS:-${MAX_RESPONSE_LENGTH}}"
export SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH="${MAX_TOOL_OUTPUT_LENGTH}"
export SLIME_FUSED_TERMINAL_LOG_STYLE="${TERMINAL_LOG_STYLE}"
export SLIME_FUSED_PROGRESS_LOGS="${SHOW_ROLLOUT_PROGRESS_LOGS}"
export SLIME_FUSED_ACCEPTED_GROUP_UPDATE_MAX_GROUPS="${ACCEPTED_GROUP_UPDATE_MAX_GROUPS}"
export SLIME_TOOL_PARSER_ERROR_LOG_ENABLED="${SLIME_TOOL_PARSER_ERROR_LOG_ENABLED:-false}"
export SLIME_FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY}"
export SLIME_FULLY_ASYNC_INITIAL_CONCURRENCY="${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY}"
export SLIME_FULLY_ASYNC_MAX_CONCURRENCY="${FULLY_ASYNC_MAX_GROUP_CONCURRENCY}"
export SLIME_FULLY_ASYNC_CONCURRENCY_STEP="${FULLY_ASYNC_CONCURRENCY_STEP}"
export SLIME_FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL="${FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL}"
export SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS="${SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS:-true}"
export SLIME_FULLY_ASYNC_NO_DATASET_WRAP="${SLIME_FULLY_ASYNC_NO_DATASET_WRAP:-true}"
export SLIME_FULLY_ASYNC_CROSS_SHARD_PREFETCH="${SLIME_FULLY_ASYNC_CROSS_SHARD_PREFETCH:-true}"
export SLIME_FULLY_ASYNC_PREFETCH_GROUPS="${SLIME_FULLY_ASYNC_PREFETCH_GROUPS:-${ROLLOUT_BATCH_SIZE}}"
export SLIME_FUSED_PROFILE_DIR="${SLIME_FUSED_PROFILE_DIR:-${OUTPUT_DIR}/trajectory_profiles}"
export CREDIT_ASSIGNMENT_ENABLE="${CREDIT_ASSIGNMENT_ENABLE:-True}"
export CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR="${CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR:-True}"
export CREDIT_ASSIGNMENT_THINK_PARSER_ERROR="${CREDIT_ASSIGNMENT_THINK_PARSER_ERROR:-True}"
export CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY="${CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY:-True}"
export CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS="${CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS:-True}"
export CREDIT_ASSIGNMENT_NGRAM_REPETITION="${CREDIT_ASSIGNMENT_NGRAM_REPETITION:-True}"
export CREDIT_ASSIGNMENT_NGRAM_REPETITION_N="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_N:-8}"
export CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD:-0.5}"
export CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS:-160}"
export CREDIT_ASSIGNMENT_SEARCH_BYPASS="${CREDIT_ASSIGNMENT_SEARCH_BYPASS:-True}"
export CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL="${CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL:-True}"
export CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER="${CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER:-True}"
export CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP="${CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP:-False}"
export SLIME_FUSED_TAIL_GUARD="${TAIL_GUARD:-False}"
export SLIME_FUSED_TAIL_GUARD_TIME_GUARD="${TAIL_GUARD_TIME_GUARD:-True}"
export SLIME_FUSED_TAIL_GUARD_TIME_MULTIPLIER="${TAIL_GUARD_TIME_MULTIPLIER:-1.05}"
export SLIME_FUSED_TAIL_GUARD_TIME_SLACK_SECONDS="${TAIL_GUARD_TIME_SLACK_SECONDS:-16}"
export SLIME_FUSED_TAIL_GUARD_MIN_COMPLETION_RATIO="${TAIL_GUARD_MIN_COMPLETION_RATIO:-0.60}"
export FUSED_FILTER_MIN_MEAN_STEPS="${FUSED_FILTER_MIN_MEAN_STEPS:-0}"
export FUSED_FILTER_MIN_MCP_MEAN_STEPS="${FUSED_FILTER_MIN_MCP_MEAN_STEPS:-0}"
export FUSED_FILTER_MAX_ABNORMAL_RATIO="${FUSED_FILTER_MAX_ABNORMAL_RATIO:-0}"
export SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER="${SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER:-4}"

RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "CUDA_HOME", "HYDRA_FULL_ERROR", "NCCL_IB_DISABLE", "NCCL_TIMEOUT",
    "FLASHINFER_USE_CUDA_NORM", "LD_LIBRARY_PATH",
    "SLIME_EPISODE_LOG_DIR", "RAY_WARN_BLOCKING_GET_INSIDE_ASYNC",
    "TOKENIZERS_PARALLELISM", "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S", "VLLM_WORKER_MULTIPROC_METHOD",
    "PYTORCH_CUDA_ALLOC_CONF", "RETRIEVAL_SERVER_URL", "RLLM_RETRIEVAL_MODE",
    "RLLM_RETRIEVAL_CONCURRENCY", "RLLM_RETRIEVAL_CACHE_SIZE",
    "RLLM_RETRIEVAL_MAX_WORDS", "RETRIEVAL_MAX_RESULTS", "RLLM_RETRIEVAL_SUMMARIZE",
    "FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "RLLM_RETRIEVAL_RETRY_BUDGET",
    "RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET", "RLLM_RETRIEVAL_LEXRANK_FALLBACK",
    "RLLM_RETRIEVAL_LEXRANK_MAX_WORDS", "RLLM_RETRIEVAL_LEXRANK_MAX_SENTENCES",
    "RLLM_RETRIEVAL_LEXRANK_MAX_INPUT_SENTENCES", "RLLM_RETRIEVAL_LEXRANK_MULTIPROCESSING",
    "RLLM_RETRIEVAL_LEXRANK_WORKERS", "DOCKER_HOST", "DOCKER_API_VERSION",
    "WANDB_API_KEY", "RLLM_MCP_MIN_NOFILE", "RLLM_MCP_FD_THROTTLE_THRESHOLD",
    "RLLM_MCP_INIT_TIMEOUT", "RLLM_MCP_START_RETRIES", "RLLM_MCP_START_WAIT_TIMEOUT",
    "RLLM_MCP_TOOL_TIMEOUT", "RLLM_MCP_MAX_ACTIVE_SERVERS", "RLLM_MCP_PREFILTER_WORKERS",
    "RLLM_MCP_DISABLE_STEP_PENALTY", "SLIME_MCP_STRICT_VERIFIER", "FUSED_HARNESS", "FUSED_UNIFIED_SYSTEM_PROMPT",
    "FUSED_DISABLE_THINKING", "FUSED_DISCARD_HISTORICAL_THINKING",
    "FUSED_MAX_STEPS", "FUSED_MCP_MAX_STEPS", "FUSED_MCP_MAX_TOOL_CALLS_PER_TURN",
    "FUSED_WEB_SEARCH_MAX_STEPS", "FUSED_CLI_MAX_STEPS", "FUSED_TRAJECTORY_TIMEOUT",
    "FUSED_EVAL_TRAJECTORY_TIMEOUT", "PER_STEP_MAX_TOKENS",
    "SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH", "SLIME_FUSED_TERMINAL_LOG_STYLE",
    "SLIME_FUSED_PROGRESS_LOGS", "SLIME_FUSED_ACCEPTED_GROUP_UPDATE_MAX_GROUPS", "SLIME_FUSED_TAIL_GUARD",
    "SLIME_TOOL_PARSER_ERROR_LOG_ENABLED",
    "SLIME_FULLY_ASYNC_ADAPTIVE_CONCURRENCY", "SLIME_FULLY_ASYNC_INITIAL_CONCURRENCY",
    "SLIME_FULLY_ASYNC_MAX_CONCURRENCY", "SLIME_FULLY_ASYNC_CONCURRENCY_STEP",
    "SLIME_FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL", "SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS",
    "SLIME_FULLY_ASYNC_NO_DATASET_WRAP", "SLIME_FULLY_ASYNC_CROSS_SHARD_PREFETCH",
    "SLIME_FULLY_ASYNC_PREFETCH_GROUPS",
    "SLIME_FUSED_PROFILE_DIR",
    "RUNS_ROOT", "RUN_ROOT", "MCP_ENV_ROOT", "SLIME_MCP_ENV_ROOT", "SLIME_MCP_ENV_COPY_CONCURRENCY",
    "SLIME_LOCAL_MCP_PROCESS_ISOLATION", "SLIME_LOCAL_MCP_PROCESS_WORKERS",
    "SLIME_LOCAL_MCP_PROCESS_START_METHOD", "SLIME_LOCAL_MCP_PROCESS_TIMEOUT",
    "SLIME_LOCAL_MCP_PROCESS_WARM_TIMEOUT",
    "SLIME_FUSED_TAIL_GUARD_TIME_GUARD", "SLIME_FUSED_TAIL_GUARD_TIME_MULTIPLIER",
    "SLIME_FUSED_TAIL_GUARD_TIME_SLACK_SECONDS", "SLIME_FUSED_TAIL_GUARD_MIN_COMPLETION_RATIO",
    "CREDIT_ASSIGNMENT_ENABLE", "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR", "CREDIT_ASSIGNMENT_THINK_PARSER_ERROR",
    "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY", "CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS",
    "CREDIT_ASSIGNMENT_NGRAM_REPETITION", "CREDIT_ASSIGNMENT_NGRAM_REPETITION_N",
    "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD", "CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS",
    "CREDIT_ASSIGNMENT_SEARCH_BYPASS", "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL",
    "CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER", "CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP",
    "FUSED_FILTER_MIN_MEAN_STEPS", "FUSED_FILTER_MIN_MCP_MEAN_STEPS",
    "FUSED_FILTER_MAX_ABNORMAL_RATIO", "SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["PYTHONPATH"] = f"{os.environ['MEGATRON_LM_PATH']}:{os.environ['REPO_ROOT']}:{os.environ['SCRIPT_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = os.environ["HAS_NVLINK"]
print(json.dumps({"env_vars": env}))
PY
)

RUN_CONFIG="${OUTPUT_DIR}/run_config.json"
python3 - "${RUN_CONFIG}" \
   experiment_name="${EXPERIMENT_NAME}" output_dir="${OUTPUT_DIR}" prompt_data="${PROMPT_DATA}" \
   sample_n="${SAMPLE_N}" max_trajectory_per_problem="${MAX_TRAJECTORY_PER_PROBLEM}" \
   min_sample_trial="${MIN_SAMPLE_TRIAL}" reward_threshold="${REWARD_THRESHOLD}" \
   min_steps="${MIN_STEPS}" certainty_filter="${CERTAINTY_FILTER}" model="${MODEL_DIR}" \
   model_config="${MODEL_CONFIG}" rollout_num_gpus_per_engine="${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
   fused_harness="${FUSED_HARNESS}" unified_system_prompt="${UNIFIED_SYSTEM_PROMPT}" \
   rollout_function_path="${ROLLOUT_FUNCTION_PATH}" \
   fully_async_adaptive_concurrency="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY}" \
   fully_async_initial_group_concurrency="${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY}" \
   fully_async_max_group_concurrency="${FULLY_ASYNC_MAX_GROUP_CONCURRENCY}" \
   sglang_server_concurrency="${SGLANG_SERVER_CONCURRENCY}" \
   sglang_max_running_requests="${SGLANG_MAX_RUNNING_REQUESTS}" \
   custom_generate_function_path="${CUSTOM_GENERATE_FUNCTION_PATH}" num_rollout="${NUM_ROLLOUT}" \
   resume="${OFFLINE_RS_RESUME}" checkpointing="${OFFLINE_RS_CHECKPOINT_ENABLE}" \
   results_mode="${RESOLVED_RESULTS_MODE}" no_dataset_wrap="${SLIME_FULLY_ASYNC_NO_DATASET_WRAP}" \
   checkpoint_path="${OFFLINE_RS_CHECKPOINT_PATH}" start_rollout_id="${RESUME_START_ROLLOUT_ID}" \
   episodes_dir="${EPISODE_LOG_DIR}" <<'PY'
import json, sys
path = sys.argv[1]
config = dict(item.split("=", 1) for item in sys.argv[2:])
with open(path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4, ensure_ascii=False, sort_keys=True)
    f.write("\n")
print(f"Wrote {path}")
PY

ROLLOUT_ARGS=(
   --debug-rollout-only
   --rollout-only-inference-fast-path
   --rollout-only-skip-episode-dump
   --async-save-debug-rollout-data
   --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
   --prompt-data "${PROMPT_DATA}"
   --input-key "${INPUT_KEY:-prompt}"
   --label-key "${LABEL_KEY:-reward_model}"
   --metadata-key "${METADATA_KEY:-extra_info}"
   --tool-key "${TOOL_KEY:-tools}"
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --start-rollout-id "${RESUME_START_ROLLOUT_ID}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${SAMPLE_N}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-prompt-len "${MAX_PROMPT_LENGTH}"
   --rollout-max-response-len "${MAX_RESPONSE_LENGTH}"
   --rollout-temperature "${TEMPERATURE}"
   --rollout-top-p "${TOP_P}"
   --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}"
   --apply-chat-template
   --dump-details "${DUMP_DETAILS}"
)

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --load "${OUTPUT_DIR}"
   --save "${OUTPUT_DIR}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   --sglang-router-request-timeout-secs "${SGLANG_ROUTER_REQUEST_TIMEOUT_SECS}"
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
   --sglang-context-length "${MAX_CONTEXT_LEN}"
   --router-policy "${ROUTER_POLICY}"
)

if [[ "$(hostname -s)" == "dgx-hyperplane17" || "$(hostname -s)" == "hgx-hyperplane09" ]]; then
   SGLANG_ARGS+=(
      --sglang-attention-backend triton
      --sglang-sampling-backend pytorch
   )
fi

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Prompt data: ${PROMPT_DATA}; rows=${TRAIN_NUM_ROWS}; rollout_batch_size=${ROLLOUT_BATCH_SIZE}; sample_n=${SAMPLE_N}; num_rollout=${NUM_ROLLOUT}; start_rollout_id=${RESUME_START_ROLLOUT_ID}"
echo "Rollout function: ${ROLLOUT_FUNCTION_PATH}"
echo "SGLang: rollout_tp=${ROLLOUT_NUM_GPUS_PER_ENGINE}; rollout_engines=${ROLLOUT_ENGINE_COUNT}; mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}; server_concurrency=${SGLANG_SERVER_CONCURRENCY}; max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS}; router_policy=${ROUTER_POLICY}"
echo "Fully async task filling: adaptive=${FULLY_ASYNC_ADAPTIVE_CONCURRENCY}; group_concurrency=${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY}-${FULLY_ASYNC_MAX_GROUP_CONCURRENCY}; step=${FULLY_ASYNC_CONCURRENCY_STEP}; poll_interval=${FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL}s"
echo "Fused controls: harness=${FUSED_HARNESS}, unified_system_prompt=${UNIFIED_SYSTEM_PROMPT}, disable_thinking=${DISABLE_THINKING}, discard_historical_thinking=${DISCARD_HISTORICAL_THINKING}, max_steps=${MAX_STEPS}, mcp_max_steps=${MCP_MAX_STEPS}, mcp_max_tool_calls_per_turn=${MCP_MAX_TOOL_CALLS_PER_TURN}, web_search_max_steps=${WEB_SEARCH_MAX_STEPS}, cli_max_steps=${CLI_MAX_STEPS}, per_step_max_tokens=${PER_STEP_MAX_TOKENS}"
echo "Debug rollout dump: ${DUMP_DETAILS}/rollout_data/{rollout_id}.pt"
echo "Per-batch trajectory shards: ${EPISODE_LOG_DIR}/global_steps_{rollout_id}.json"
echo "Checkpoint: ${OFFLINE_RS_CHECKPOINT_PATH}"
echo "Results mode: ${RESOLVED_RESULTS_MODE}"

if [ "${SKIP_RAY_ROLLOUT}" != "1" ]; then
   RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
   if ray job list --address="${RAY_DASHBOARD_ADDRESS}" >/dev/null 2>&1; then
      echo "Reusing existing Ray head at ${RAY_DASHBOARD_ADDRESS}"
   else
      ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --num-cpus "${RAY_NUM_CPUS}" --disable-usage-stats
   fi

   cd "${REPO_ROOT}"

   SAFE_EXPERIMENT_NAME="$(printf '%s' "${EXPERIMENT_NAME}" | tr -c '[:alnum:]_' '_' | cut -c1-120)"
   RAY_SUBMISSION_ID="${RAY_SUBMISSION_ID:-raysubmit_${SAFE_EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)}"

   RAY_JOB_SUBMIT_ARGS=()
   if [ "${RAY_JOB_WAIT}" != "1" ]; then
      RAY_JOB_SUBMIT_ARGS+=(--no-wait)
   fi

   echo "Ray submission id: ${RAY_SUBMISSION_ID}"

   ray job submit --address="${RAY_DASHBOARD_ADDRESS}" \
      --submission-id="${RAY_SUBMISSION_ID}" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      "${RAY_JOB_SUBMIT_ARGS[@]}" \
      -- python3 -u train.py \
      --rollout-num-gpus "${ROLLOUT_GPUS}" \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${SGLANG_ARGS[@]}"

   if [ "${RAY_JOB_WAIT}" != "1" ]; then
      echo "Ray job submitted without waiting. Run the script again with --ray-job-wait 1 to checkpoint and merge after it finishes."
      exit 0
   fi

   RAY_JOB_STATUS="$(ray job status --address="${RAY_DASHBOARD_ADDRESS}" "${RAY_SUBMISSION_ID}" 2>/dev/null || true)"
   if ! grep -qi "succeeded" <<<"${RAY_JOB_STATUS}"; then
      echo "Ray job ${RAY_SUBMISSION_ID} did not succeed; refusing to merge incomplete rollout data." >&2
      printf '%s\n' "${RAY_JOB_STATUS}" >&2
      exit 1
   fi
fi

NUM_ROLLOUT=$(python3 - "${DUMP_DETAILS}/rollout_data" "${NUM_ROLLOUT}" <<'PY'
import sys
from pathlib import Path

import torch

rollout_dir = Path(sys.argv[1])
configured = int(sys.argv[2])
completed = 0
while completed < configured:
    path = rollout_dir / f"{completed}.pt"
    if not path.is_file():
        break
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("num_samples", len(payload.get("samples", ())))) <= 0:
        break
    completed += 1
print(completed)
PY
)
if [ "${NUM_ROLLOUT}" -le 0 ]; then
   echo "No non-empty rejection-sampling shards were completed." >&2
   exit 1
fi
echo "Completed non-empty rollout shards: ${NUM_ROLLOUT}"

for ((rollout_id = RESUME_START_ROLLOUT_ID; rollout_id < NUM_ROLLOUT; rollout_id++)); do
   rollout_path="${DUMP_DETAILS}/rollout_data/${rollout_id}.pt"
   if [ ! -s "${rollout_path}" ]; then
      echo "Missing completed rollout shard: ${rollout_path}" >&2
      exit 1
   fi
done

python3 - "${OUTPUT_DIR}" "${DUMP_DETAILS}/rollout_data" "${EPISODE_LOG_DIR}" \
   "${OFFLINE_RS_CHECKPOINT_PATH}" "${OFFLINE_RS_CHECKPOINT_ENABLE}" "${REWARD_THRESHOLD}" "${MIN_STEPS}" \
   "${MAX_TRAJECTORY_PER_PROBLEM}" "${MIN_SAMPLE_TRIAL}" "${CERTAINTY_FILTER}" "${NUM_ROLLOUT}" \
   "${RESOLVED_RESULTS_MODE}" <<'PY'
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from slime.utils.episode_dump import _episode_to_batch_dict

output_dir = Path(sys.argv[1])
rollout_dir = Path(sys.argv[2])
episodes_dir = Path(sys.argv[3])
checkpoint_path = Path(sys.argv[4])
checkpointing = sys.argv[5].lower() in {"1", "true", "yes", "on"}
reward_threshold = float(sys.argv[6])
min_steps = int(sys.argv[7])
max_keep = int(sys.argv[8])
min_trials = int(sys.argv[9])
certainty_filter = sys.argv[10].lower() in {"1", "true", "yes", "on"}
num_rollout = int(sys.argv[11])
results_mode = sys.argv[12]

episodes_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)


def reward_value(sample):
    reward = sample.get("reward", 0.0)
    if isinstance(reward, dict):
        for key in ("score", "reward", "acc"):
            if key in reward:
                return float(reward[key])
        return 0.0
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


def metadata(sample):
    value = sample.get("metadata")
    return value if isinstance(value, dict) else {}


def episode(sample):
    value = metadata(sample).get("rllm_episode")
    return value if isinstance(value, dict) else None


def problem_id(sample):
    ep = episode(sample)
    if ep and ep.get("id"):
        return str(ep["id"]).split(":", 1)[0]
    meta = metadata(sample)
    for key in ("id", "uid", "uuid", "instance_id", "question"):
        if meta.get(key) is not None:
            return str(meta[key])
    if sample.get("group_index") is not None:
        return f"group:{sample['group_index']}"
    if sample.get("index") is not None:
        return f"index:{sample['index']}"
    return "unknown"


def step_count(sample):
    meta = metadata(sample)
    if meta.get("fused_traj_steps") is not None:
        return int(meta["fused_traj_steps"])
    ep = episode(sample)
    if ep and ep.get("trajectories"):
        steps = ep["trajectories"][0].get("steps", [])
        if isinstance(steps, list):
            return len(steps)
    return 0


def rollout_id_from_path(path):
    try:
        return int(path.stem)
    except ValueError:
        return None


def sample_key(sample):
    ep = episode(sample)
    if ep and ep.get("id"):
        return str(ep["id"])
    return f"{problem_id(sample)}:{sample.get('index')}:{sample.get('rollout_id')}"


def batch_samples(path):
    dump = torch.load(path, map_location="cpu", weights_only=False)
    return [dict(sample, _rollout_dump=str(path)) for sample in dump.get("samples", [])]


def rollout_sort_key(path):
    rid = rollout_id_from_path(path)
    return rid if rid is not None else 10**12


def atomic_json_dump(path, payload, *, indent=4, sort_keys=False):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent, sort_keys=sort_keys, default=str)
        f.write("\n")
    temporary.replace(path)


if results_mode == "manifest":
    accepted_episodes_dir = output_dir / "accepted_episodes"
    accepted_episodes_dir.mkdir(parents=True, exist_ok=True)
    sample_index_path = output_dir / "rejection_sampling_sample_index.json"
    sample_index_tmp = sample_index_path.with_suffix(sample_index_path.suffix + ".tmp")
    results_path = output_dir / "rejection_sampling_results.json"
    summary_path = output_dir / "rejection_sampling_summary.json"

    accepted_keys = set()
    accepted_problem_ids = set()
    rejected_problem_ids = set()
    seen_problem_ids = set()
    group_summaries = []
    batch_summaries = []
    accepted_refs = []
    rollout_files = []
    num_samples = 0
    num_episodes = 0
    recovered_shards = {}

    if checkpointing and checkpoint_path.is_file():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checkpoint = {}
        if checkpoint.get("results_mode") == "manifest":
            for shard in checkpoint.get("completed_shards", []):
                if not isinstance(shard, dict):
                    continue
                try:
                    rollout_id = int(shard["rollout_id"])
                    episode_path = Path(shard["episode_path"])
                    accepted_episode_path = Path(shard["accepted_episode_path"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not episode_path.is_file() or not accepted_episode_path.is_file():
                    continue
                try:
                    accepted_payload = json.loads(accepted_episode_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                recovered_keys = accepted_payload.get("accepted_sample_keys")
                if not isinstance(recovered_keys, list):
                    recovered_keys = [
                        row.get("episode_id")
                        for row in accepted_payload.get("trajectories", [])
                        if isinstance(row, dict) and row.get("episode_id") is not None
                    ]
                recovered_shards[rollout_id] = (shard, {str(key) for key in recovered_keys})

    def annotate_episode(ep, accepted):
        value = json.loads(json.dumps(ep, ensure_ascii=False, default=str))
        value["is_accepted"] = bool(accepted)
        ep_metadata = value.setdefault("metadata", {})
        if isinstance(ep_metadata, dict):
            ep_metadata["is_accepted"] = bool(accepted)
            profile = ep_metadata.get("fused_profile")
            if isinstance(profile, dict):
                profile["accepted"] = bool(accepted)
        for traj in value.get("trajectories", []) or []:
            if isinstance(traj, dict):
                traj["is_accepted"] = bool(accepted)
        return value

    def episode_rows(batch, rollout_id, selected_keys, *, accepted_only):
        rows = []
        seen_episodes = set()
        for sample in batch:
            selected = sample_key(sample) in selected_keys
            if accepted_only and not selected:
                continue
            ep = episode(sample)
            if not ep:
                continue
            key = str(ep.get("id") or sample_key(sample))
            if key in seen_episodes:
                continue
            seen_episodes.add(key)
            row = _episode_to_batch_dict(
                annotate_episode(ep, selected),
                rollout_id,
                "train",
                0,
                args=None,
                sample_metadata=metadata(sample),
                eval_reward=None,
            )
            row["is_accepted"] = selected
            if isinstance(row.get("metadata"), dict):
                row["metadata"]["is_accepted"] = selected
            for traj in row.get("trajectories", []) or []:
                if isinstance(traj, dict):
                    traj["is_accepted"] = selected
            rows.append(row)
        return rows

    with sample_index_tmp.open("w", encoding="utf-8") as index_file:
        index_file.write("[\n")
        first_index_item = True
        for rollout_path in sorted(rollout_dir.glob("*.pt"), key=rollout_sort_key):
            rollout_id = rollout_id_from_path(rollout_path)
            if rollout_id is None:
                continue
            batch = batch_samples(rollout_path)
            groups = defaultdict(list)
            for sample in batch:
                groups[problem_id(sample)].append(sample)

            repeated = seen_problem_ids.intersection(groups)
            if repeated:
                preview = ", ".join(sorted(repeated)[:5])
                raise RuntimeError(
                    "Manifest merge requires each problem to stay in one rollout shard; "
                    f"repeated problem ids: {preview}"
                )
            seen_problem_ids.update(groups)

            recovered_shard = recovered_shards.get(rollout_id)
            recovered_keys = recovered_shard[1] if recovered_shard is not None else None
            batch_keys = {sample_key(sample) for sample in batch}
            if recovered_keys is not None and not recovered_keys.issubset(batch_keys):
                raise RuntimeError(f"Accepted episode shard {rollout_id} does not match its rollout .pt shard")
            batch_accepted_keys = set()
            for pid, items in sorted(groups.items()):
                trials = len(items)
                good = [
                    item for item in items
                    if reward_value(item) >= reward_threshold and step_count(item) >= min_steps
                ]
                pass_rate = len(good) / trials if trials else 0.0
                if recovered_keys is None:
                    kept = []
                    if trials >= min_trials and (not certainty_filter or (0.0 < pass_rate < 1.0)):
                        good.sort(
                            key=lambda item: (
                                reward_value(item),
                                step_count(item),
                                -len(str(item.get("response", ""))),
                            ),
                            reverse=True,
                        )
                        kept = good[:max_keep]
                    selected = {sample_key(item) for item in kept}
                else:
                    selected = {sample_key(item) for item in items if sample_key(item) in recovered_keys}
                batch_accepted_keys.update(selected)
                accepted_keys.update(selected)
                if selected:
                    accepted_problem_ids.add(pid)
                if len(selected) < trials:
                    rejected_problem_ids.add(pid)
                group_summaries.append(
                    {
                        "problem_id": pid,
                        "rollout_id": rollout_id,
                        "trials": trials,
                        "accepted": len(selected),
                        "pass_rate": pass_rate,
                        "max_reward": max((reward_value(item) for item in items), default=math.nan),
                    }
                )

            episode_path = episodes_dir / f"global_steps_{rollout_id}.json"
            accepted_episode_path = accepted_episodes_dir / f"global_steps_{rollout_id}.json"
            if recovered_shard is None:
                all_episode_rows = episode_rows(batch, rollout_id, batch_accepted_keys, accepted_only=False)
                accepted_episode_rows = episode_rows(batch, rollout_id, batch_accepted_keys, accepted_only=True)
                atomic_json_dump(
                    episode_path,
                    {
                        "training_step": rollout_id,
                        "epoch": 0,
                        "mode": "train",
                        "num_episodes": len(all_episode_rows),
                        "trajectories": all_episode_rows,
                    },
                )
                atomic_json_dump(
                    accepted_episode_path,
                    {
                        "training_step": rollout_id,
                        "epoch": 0,
                        "mode": "train",
                        "num_episodes": len(accepted_episode_rows),
                        "accepted_sample_keys": sorted(batch_accepted_keys),
                        "trajectories": accepted_episode_rows,
                    },
                )
                batch_episode_count = len(all_episode_rows)
            else:
                batch_episode_count = int(recovered_shard[0].get("num_episodes", len(batch)))

            for row_index, sample in enumerate(batch):
                key = sample_key(sample)
                selected = key in batch_accepted_keys
                profile = dict(metadata(sample).get("fused_profile") or {})
                profile["accepted"] = selected
                item = {
                    "rollout_id": rollout_id,
                    "row": row_index,
                    "sample_key": key,
                    "problem_id": problem_id(sample),
                    "group_index": sample.get("group_index"),
                    "index": sample.get("index"),
                    "reward": reward_value(sample),
                    "steps": step_count(sample),
                    "status": sample.get("status"),
                    "task_type": metadata(sample).get("fused_task_type"),
                    "termination": metadata(sample).get("fused_termination"),
                    "is_accepted": selected,
                    "profile": profile,
                    "rollout_shard": str(rollout_path),
                    "episode_shard": str(episode_path),
                }
                if not first_index_item:
                    index_file.write(",\n")
                rendered_item = json.dumps(item, ensure_ascii=False, indent=4, default=str)
                index_file.write("\n".join("    " + line for line in rendered_item.splitlines()))
                first_index_item = False
                if selected:
                    accepted_refs.append(item)

            num_samples += len(batch)
            num_episodes += batch_episode_count
            rollout_files.append(str(rollout_path))
            batch_summaries.append(
                {
                    "rollout_id": rollout_id,
                    "rollout_path": str(rollout_path),
                    "episode_path": str(episode_path),
                    "accepted_episode_path": str(accepted_episode_path),
                    "num_samples": len(batch),
                    "num_episodes": batch_episode_count,
                    "num_accepted": len(batch_accepted_keys),
                }
            )

            if checkpointing:
                completed = [item["rollout_id"] for item in batch_summaries]
                completed_set = set(completed)
                next_rollout_id = 0
                while next_rollout_id < num_rollout and next_rollout_id in completed_set:
                    next_rollout_id += 1
                atomic_json_dump(
                    checkpoint_path,
                    {
                        "next_rollout_id": next_rollout_id,
                        "completed_shards": batch_summaries,
                        "num_rollout": num_rollout,
                        "num_samples": num_samples,
                        "num_episodes": num_episodes,
                        "num_accepted": len(accepted_keys),
                        "results": str(results_path),
                        "episodes_dir": str(episodes_dir),
                        "results_mode": "manifest",
                    },
                    sort_keys=True,
                )

            del batch

        index_file.write("\n]\n")

    sample_index_tmp.replace(sample_index_path)
    accepted_problem_ids = sorted(accepted_problem_ids)
    rejected_problem_ids = sorted(rejected_problem_ids)
    summary = {
        "results_mode": "manifest",
        "rollout_dir": str(rollout_dir),
        "num_rollout_files": len(rollout_files),
        "episodes_dir": str(episodes_dir),
        "accepted_episodes_dir": str(accepted_episodes_dir),
        "num_episode_shards": len(batch_summaries),
        "num_samples": num_samples,
        "num_groups": len(group_summaries),
        "num_accepted": len(accepted_keys),
        "num_rejected": num_samples - len(accepted_keys),
        "num_accepted_problems": len(accepted_problem_ids),
        "num_rejected_problems": len(rejected_problem_ids),
        "reward_threshold": reward_threshold,
        "min_steps": min_steps,
        "max_trajectory_per_problem": max_keep,
        "min_sample_trial": min_trials,
        "certainty_filter": certainty_filter,
        "checkpoint_path": str(checkpoint_path),
        "sample_index": str(sample_index_path),
        "results": str(results_path),
    }
    results = {
        "summary": summary,
        "samples": {
            "storage": "rollout_shards_with_json_index",
            "num_samples": num_samples,
            "num_accepted": len(accepted_keys),
            "num_rejected": num_samples - len(accepted_keys),
            "index": str(sample_index_path),
            "rollout_shards": rollout_files,
            "accepted_items": accepted_refs,
        },
        "episodes": {
            "storage": "episode_shards",
            "num_episodes": num_episodes,
            "num_accepted": len(accepted_keys),
            "num_rejected": num_episodes - len(accepted_keys),
            "shards": batch_summaries,
        },
        "groups": {
            "num_groups": len(group_summaries),
            "accepted_problem_ids": accepted_problem_ids,
            "rejected_problem_ids": rejected_problem_ids,
            "items": group_summaries,
        },
        "global_steps": {
            "mode": "train",
            "num_shards": len(batch_summaries),
            "shards": batch_summaries,
            "num_episodes": num_episodes,
        },
    }
    atomic_json_dump(results_path, results)
    atomic_json_dump(summary_path, summary, sort_keys=True)
    print(json.dumps(summary, ensure_ascii=False, indent=4, sort_keys=True))
    raise SystemExit(0)


samples = []
samples_by_rollout = {}
for path in sorted(rollout_dir.glob("*.pt"), key=rollout_sort_key):
    rollout_id = rollout_id_from_path(path)
    if rollout_id is None:
        continue
    batch = batch_samples(path)
    samples_by_rollout[rollout_id] = batch
    samples.extend(batch)

groups = defaultdict(list)
for sample in samples:
    groups[problem_id(sample)].append(sample)

accepted = []
accepted_keys = set()
group_summaries = []
for pid, items in sorted(groups.items()):
    trials = len(items)
    good = [
        item for item in items
        if reward_value(item) >= reward_threshold and step_count(item) >= min_steps
    ]
    pass_rate = len(good) / trials if trials else 0.0
    kept = []
    if trials >= min_trials and (not certainty_filter or (0.0 < pass_rate < 1.0)):
        good.sort(key=lambda item: (reward_value(item), step_count(item), -len(str(item.get("response", "")))), reverse=True)
        kept = good[:max_keep]
        accepted.extend(kept)
        accepted_keys.update(sample_key(item) for item in kept)
    group_summaries.append(
        {
            "problem_id": pid,
            "trials": trials,
            "accepted": len(kept),
            "pass_rate": pass_rate,
            "max_reward": max((reward_value(item) for item in items), default=math.nan),
        }
    )

accepted_problem_ids = sorted({problem_id(sample) for sample in samples if sample_key(sample) in accepted_keys})
rejected_problem_ids = sorted({problem_id(sample) for sample in samples if sample_key(sample) not in accepted_keys})

results_path = output_dir / "rejection_sampling_results.json"
summary_path = output_dir / "rejection_sampling_summary.json"


def annotate_sample(sample):
    item = dict(sample)
    item.pop("_rollout_dump", None)
    item["is_accepted"] = sample_key(sample) in accepted_keys
    item_metadata = item.get("metadata")
    if isinstance(item_metadata, dict) and isinstance(item_metadata.get("fused_profile"), dict):
        item_metadata["fused_profile"]["accepted"] = item["is_accepted"]
    return item


def annotate_episode(ep, accepted):
    value = json.loads(json.dumps(ep, ensure_ascii=False, default=str))
    value["is_accepted"] = bool(accepted)
    metadata = value.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["is_accepted"] = bool(accepted)
        profile = metadata.get("fused_profile")
        if isinstance(profile, dict):
            profile["accepted"] = bool(accepted)
    for traj in value.get("trajectories", []) or []:
        if isinstance(traj, dict):
            traj["is_accepted"] = bool(accepted)
    return value


def write_batch_shard(rollout_id, batch):
    trajectories = []
    seen = set()
    for sample in batch:
        ep = episode(sample)
        if not ep:
            continue
        key = str(ep.get("id") or sample_key(sample))
        if key in seen:
            continue
        seen.add(key)
        accepted = sample_key(sample) in accepted_keys
        annotated_ep = annotate_episode(ep, accepted)
        row = _episode_to_batch_dict(
            annotated_ep,
            rollout_id,
            "train",
            0,
            args=None,
            sample_metadata=metadata(sample),
            eval_reward=None,
        )
        row["is_accepted"] = accepted
        if isinstance(row.get("metadata"), dict):
            row["metadata"]["is_accepted"] = accepted
        for traj in row.get("trajectories", []) or []:
            if isinstance(traj, dict):
                traj["is_accepted"] = accepted
        trajectories.append(row)
    path = episodes_dir / f"global_steps_{rollout_id}.json"
    payload = {
        "training_step": rollout_id,
        "epoch": 0,
        "mode": "train",
        "num_episodes": len(trajectories),
        "accepted_sample_keys": sorted(sample_key(sample) for sample in batch if sample_key(sample) in accepted_keys),
        "trajectories": trajectories,
    }
    atomic_json_dump(path, payload)
    return path, len(trajectories)

accepted_samples = [annotate_sample(sample) for sample in accepted]
all_samples = [annotate_sample(sample) for sample in samples]
rejected_samples = [sample for sample in all_samples if not sample["is_accepted"]]

accepted_episodes = []
for sample in accepted:
    ep = episode(sample)
    if ep:
        accepted_episodes.append(annotate_episode(ep, True))

all_episodes = []
seen = set()
for sample in samples:
    ep = episode(sample)
    if not ep:
        continue
    key = str(ep.get("id") or sample_key(sample))
    if key in seen:
        continue
    seen.add(key)
    all_episodes.append(annotate_episode(ep, sample_key(sample) in accepted_keys))

batch_summaries = []
for rollout_id in sorted(samples_by_rollout):
    path, count = write_batch_shard(rollout_id, samples_by_rollout[rollout_id])
    batch_summaries.append({"rollout_id": rollout_id, "path": str(path), "num_episodes": count})
    if checkpointing:
        completed = [item["rollout_id"] for item in batch_summaries]
        completed_set = set(completed)
        next_rollout_id = 0
        while next_rollout_id < num_rollout and next_rollout_id in completed_set:
            next_rollout_id += 1
        checkpoint = {
            "next_rollout_id": next_rollout_id,
            "completed_shards": batch_summaries,
            "num_rollout": num_rollout,
            "num_samples": len(samples),
            "num_episodes": sum(item["num_episodes"] for item in batch_summaries),
            "num_accepted": len(accepted_keys),
            "results": str(results_path),
            "episodes_dir": str(episodes_dir),
            "results_mode": "full",
        }
        atomic_json_dump(checkpoint_path, checkpoint, sort_keys=True)

combined_trajectories = []
pattern = re.compile(r"global_steps_(\d+)\.json$")
for shard in sorted(
    episodes_dir.glob("global_steps_*.json"),
    key=lambda p: int(pattern.match(p.name).group(1)) if pattern.match(p.name) else 10**12,
):
    data = json.loads(shard.read_text())
    combined_trajectories.extend(data.get("trajectories", []))

summary = {
    "rollout_dir": str(rollout_dir),
    "num_rollout_files": len(list(rollout_dir.glob("*.pt"))),
    "episodes_dir": str(episodes_dir),
    "num_episode_shards": len(list(episodes_dir.glob("global_steps_*.json"))),
    "num_samples": len(samples),
    "num_groups": len(groups),
    "num_accepted": len(accepted),
    "num_rejected": len(samples) - len(accepted_keys),
    "num_accepted_problems": len(accepted_problem_ids),
    "num_rejected_problems": len(rejected_problem_ids),
    "reward_threshold": reward_threshold,
    "min_steps": min_steps,
    "max_trajectory_per_problem": max_keep,
    "min_sample_trial": min_trials,
    "certainty_filter": certainty_filter,
    "checkpoint_path": str(checkpoint_path),
    "results": str(results_path),
}
results = {
    "summary": summary,
    "samples": {
        "num_samples": len(all_samples),
        "num_accepted": len(accepted_samples),
        "num_rejected": len(rejected_samples),
        "items": all_samples,
    },
    "episodes": {
        "num_episodes": len(all_episodes),
        "num_accepted": len(accepted_episodes),
        "num_rejected": len(all_episodes) - len(accepted_episodes),
        "items": all_episodes,
    },
    "groups": {
        "num_groups": len(group_summaries),
        "accepted_problem_ids": accepted_problem_ids,
        "rejected_problem_ids": rejected_problem_ids,
        "items": group_summaries,
    },
    "global_steps": {
        "mode": "train",
        "num_shards": len(list(episodes_dir.glob("global_steps_*.json"))),
        "shards": batch_summaries,
        "num_episodes": len(combined_trajectories),
        "trajectories": combined_trajectories,
    },
}
with results_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4, default=str)
    f.write("\n")
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4, sort_keys=True)
    f.write("\n")

print(json.dumps(summary, ensure_ascii=False, indent=4, sort_keys=True))
PY

echo "Offline rejection sampling completed. Output: ${OUTPUT_DIR}"
