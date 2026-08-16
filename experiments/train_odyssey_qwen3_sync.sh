#!/bin/bash
# Single-node synchronous fused-agent training launcher for current slime.
#
# The actor and rollout engines are colocated on all eight GPUs of this node.

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

usage() {
   cat <<'EOF'
Usage:
  bash experiments/train_odyssey_qwen3_sync.sh [options]

Options:
  --harness NAME                         Fused prompt harness: bare, cot, react, gem, unified_gem.
  --model PATH                           HF model path.
  --enable-yarn BOOL                     Enable static YaRN for long contexts. Default: false.
  --yarn-factor X                        YaRN scale factor. Default: 2.0.
  --yarn-original-max-position-embeddings N
                                         Qwen3 native context used by YaRN. Default: 32768.
  --disable-thinking BOOL                Stored in env for compatible fused code.
  --discard-historical-thinking BOOL     Remove prior assistant <think> blocks before each new rollout step.
                                         Effective only when --disable-thinking is false. Default: true.
  --mcp-disable-step-penalty BOOL        MCP verifier step-penalty env.
  --unified-system-prompt                Select unified_gem harness unless --harness is set later.
  --no-unified-system-prompt             Select gem harness unless --harness is set later.
  --partial-rollout / --no-partial-rollout
                                         Partial rollout is rejected for fused agents until resumable state is implemented.
  --terminal-log-style STYLE             progress, rollouts, or both. Stored in env for compatible fused code.
  --show-rollout-progress-logs BOOL      Show periodic fused rollout request/progress logs. Default: false.
  --temperature X                       Training rollout sampling temperature. Default: 1.0.
  --micro-batch-size N                   Training micro-batch size.
  --update-weights-interval N            Rollout weight update interval. Default: 1.
  --rollout-num-gpus-per-engine N        Tensor-parallel GPUs per rollout engine.
                                         Default: 1 on this eight-GPU launcher.
  --ckpt-step N                          Resume this exact iteration from --load instead of its latest tracker.
  --retrieval-backend local|serper       Set both train and eval backends (compatibility alias).
  --train-retrieval-backend local|serper Training retrieval backend. Default: local.
  --eval-retrieval-backend local|serper  Evaluation retrieval backend. Default: serper.
  --retrieval-concurrency N              Concurrent retrieval requests. Default: 176.
  --retrieval-mode MODE                  Retrieval mode env.
  --retrieval-cache-size N               Cross-episode retrieval LRU entries. Default: 4096.
  --retrieval-max-words N                Retrieval max words env.
  --retrieval-retry-budget N             Retrieval retry env.
  --retrieval-summary-retry-budget N     Retrieval summary retry env.
  --retrieval-lexrank-fallback BOOL      Accepted for compatibility; LexRank fallback is forced off.
  --retrieval-lexrank-max-words N        Retrieval LexRank max words env.
  --retrieval-lexrank-max-sentences N    Retrieval LexRank max sentences env.
  --retrieval-lexrank-max-input-sentences N
                                         Retrieval LexRank max input sentences env.
  --retrieval-lexrank-multiprocessing BOOL
                                         Retrieval LexRank multiprocessing env.
  --retrieval-lexrank-workers N          Retrieval LexRank workers env.
  --ray-num-cpus N                       Ray CPU resource count.
  --tail-guard BOOL                      Stored in env for compatible fused code.
  --tail-guard-time-guard BOOL           Stored in env for compatible fused code.
  --tail-guard-time-multiplier X         Stored in env for compatible fused code.
  --tail-guard-time-slack-seconds N      Stored in env for compatible fused code.
  --tail-guard-min-completion-ratio X    Stored in env for compatible fused code.
  --credit-assignment-enable BOOL        Enable fused credit-assignment masking strategies. Default: true.
  --credit-assignment-tool-parser-error BOOL
                                         Train only the parser-error turn for parser failures. Default: false.
  --credit-assignment-repeated-search-query BOOL
                                         Train only the repeated-query turn. Default: true.
  --credit-assignment-too-many-tool-calls BOOL
                                         Train only the excessive-tool-call turn. Default: true.
  --credit-assignment-search-bypass BOOL Keep search-bypass masks but force reward to 0. Default: true.
  --credit-assignment-direct-submit-without-tool BOOL
                                         Penalize direct finish/boxed answer before any non-finish tool. Default: true.
  --credit-assignment-mixed-tool-and-answer BOOL
                                         Penalize turns that contain both a non-finish tool call and boxed/submit answer. Default: false.
  --credit-assignment-tail-guard-early-stop BOOL
                                         Stored in env for compatible fused code.
  --horizon-reward-shaping BOOL         Enable bounded horizon penalty reward shaping. Default: false.
  --horizon-reward-min-multiplier X      Correct low-horizon reward multiplier floor. Default: 0.2.
  --horizon-reward-gamma X               Horizon progress exponent. Default: 1.0.
  --horizon-reward-step-weight X         Step progress weight. Default: 0.7.
  --horizon-reward-tool-call-weight X    Tool-call progress weight. Default: 0.3.
  --horizon-reward-target-steps X        Target steps for no horizon penalty. Default: 8.
  --horizon-reward-target-tool-calls X   Target tool calls for no horizon penalty. Default: target_steps - 1.
  --enable-dynamic-sampling-filter BOOL  Enable DAPO-style non-zero reward variance dynamic filtering. Default: true.
  --normalize-advantages / --no-normalize-advantages
                                         Whiten advantages across the data-parallel batch. Default: enabled.
  --enable_use_grm_train BOOL            Use OpenRouter GRM with rule-based fallback for WebQA training rewards.
                                         MCP retains environment verifier rewards. Default: true.
  --enable_use_grm_evals BOOL            Use OpenRouter GRM before rule-based fallback for WebQA interval evals.
                                         MCP retains dataset verifier rewards. Default: true.
  --train-grm-model NAME                 Training OpenRouter judge model. Default: deepseek/deepseek-v4-flash-0731.
  --eval-grm-model NAME                  Evaluation OpenRouter judge model. Default: google/gemini-3.7-flash.
  --grm-base-url URL                     OpenRouter-compatible judge endpoint.
  --grm-openrouter-api-key KEY           GRM API key. Prefer the OPENROUTER_API_KEY environment variable.
  --grm-mode score|equivalence           GRM protocol. Default: score.
  --grm-concurrency N                    Max concurrent GRM requests. Default: 128.
  --grm-timeout SECONDS                  GRM request timeout. Default: 60.
  --grm-max-retries N                    GRM retry attempts. Default: 32.
  --grm-max-input-tokens N               Maximum GRM input content tokens. Default: 131072.
  --max-steps N                          Fused agent max steps. Default: 96.
  --mcp-max-steps N                      Fused MCP max steps. Default: 96.
  --mcp-max-tool-calls-per-turn N        Maximum MCP calls emitted in one assistant turn. Default: 8.
  --web-search-max-steps N               Fused web-search max steps. Default: 96.
  --cli-max-steps N                      CLI fused agent max steps env.
  --trajectory-timeout N                 Fused trajectory timeout env.
  --eval-trajectory-timeout N            Fused eval trajectory timeout env.
  --rollout-group-timeout N              Per-attempt deadline for each prompt group's unfinished slots.
                                         Timeout classification uses the active trajectory stage. Default: 3600.
  --rollout-infra-retry-times N          Replacement attempts per logical slot after retryable infra failure.
                                         Default: 2.
  --eval-interval N                      Run interval eval every N rollout steps.
  --eval-config PATH                     Structured slime eval dataset config. Overrides generated benchmarks.
  --eval-prompt-data NAME PATH [...]     Legacy eval dataset name/path pairs.
  --eval-benchmarks-root PATH             Benchmark root used to generate the default eval config.
  --eval-include LIST                    Comma-separated default eval benchmarks. Default: asearcher.
  --eval-exclude LIST                    Comma-separated eval benchmarks to exclude.
  --eval-limit-per-benchmark N           Use the first N examples per benchmark. Default: 0 (all).
  --eval-temperature X                   Eval-only sampling temperature. Default: 0.6.
  --eval-top-p X                         Eval-only nucleus sampling threshold. Default: 0.95.
  --eval-top-k N                         Eval-only top-k sampling. Default: -1 (disabled).
  --per-step-max-tokens N                Per fused-agent turn. Default: 8192.
  --eval-max-response-len N              Eval-only max generated tokens. Default: 38000.
  --eval-max-prompt-len N                Eval-only max prompt tokens. Default: 2048.
  --eval-max-context-len N               Eval-only context length. Default: 40960.
  --eval-initial-inflight-tasks N        Initial scheduled eval trajectories. Default: 128.
  --eval-max-inflight-tasks N            Adaptive eval hard limit. Default: 256.
  --eval-adaptive-concurrency BOOL       Adjust eval concurrency from engine metrics. Default: true.
  --eval-mix-datasets BOOL               Interleave eval benchmarks under one inflight budget. Default: true.
  --eval-termination-retry-times N       Retry non-env_done eval trajectories. Default: 4.
  --eval-trajectory-sample-rate X        Full eval trajectory dump fraction. Default: 1.
  --eval-dump-failures BOOL              Dump failed eval trajectories. Default: true.
  --native-sglang-session BOOL           Use incremental SGLang sessions during eval. Default: true.
  --val_before_train BOOL                Run one eval before training starts. Default: true.
  --n-samples-per-eval-prompt N          Eval samples per prompt. Default: 1.
  --offload-train BOOL                   Offload trainer model between phases. Disabled by --release-train.
  --release-train BOOL                   Recreate trainer each step instead of pausing it. Default: true.
  --max-tool-output-length N             Fused max tool output length env.
  --sglang-server-concurrency N          Max concurrent requests per SGLang server. Default: 128.
  --sglang-router-request-timeout-secs N Router request timeout. Default: 21600.
  --sglang-max-running-requests N        SGLang max running requests. Default: 128.
  --colocate                             Share trainer and rollout GPUs with offload. Required by this launcher.
  --experiment-name NAME                 Experiment/run name. Defaults to the next dev suffix below.
  -h, --help                             Show this help.
EOF
}

is_truthy() {
   case "${1}" in
      1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
      *) return 1 ;;
   esac
}

PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-false}"
CKPT_STEP="${CKPT_STEP:-}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
TERMINAL_LOG_STYLE="${TERMINAL_LOG_STYLE:-both}"
SHOW_ROLLOUT_PROGRESS_LOGS="${SHOW_ROLLOUT_PROGRESS_LOGS:-false}"
# Alternate training and rollout across all eight GPUs on this node.
COLOCATE="${COLOCATE:-true}"
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
UNIFIED_SYSTEM_PROMPT="${UNIFIED_SYSTEM_PROMPT:-False}"
DISABLE_THINKING="${DISABLE_THINKING:-false}"
ENABLE_YARN="${ENABLE_YARN:-false}"
YARN_FACTOR="${YARN_FACTOR:-1.0}"
YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-32768}"
DISCARD_HISTORICAL_THINKING="${DISCARD_HISTORICAL_THINKING:-false}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
TRAIN_RETRIEVAL_BACKEND="${TRAIN_RETRIEVAL_BACKEND:-${RETRIEVAL_BACKEND:-local}}"
#TRAIN_RETRIEVAL_BACKEND="${TRAIN_RETRIEVAL_BACKEND:-${RETRIEVAL_BACKEND:-serper}}"
EVAL_RETRIEVAL_BACKEND="${EVAL_RETRIEVAL_BACKEND:-serper}"
RETRIEVAL_CONCURRENCY="${RETRIEVAL_CONCURRENCY:-512}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-${RLLM_RETRIEVAL_MODE:-hybrid}}"
RETRIEVAL_CACHE_SIZE="${RETRIEVAL_CACHE_SIZE:-4096}"
SERPER_SERVER_HOST="${SERPER_SERVER_HOST:-127.0.0.1}"
SERPER_SERVER_PORT="${SERPER_SERVER_PORT:-65433}"
SERPER_SEARCH_URL="${SERPER_SEARCH_URL:-http://10.2.152.50:9999/search}"
EVAL_SERPER_SERVICE_PID=""
TAIL_GUARD="${TAIL_GUARD:-False}"
TAIL_GUARD_TIME_GUARD="${TAIL_GUARD_TIME_GUARD:-True}"
TAIL_GUARD_TIME_MULTIPLIER="${TAIL_GUARD_TIME_MULTIPLIER:-1.05}"
TAIL_GUARD_TIME_SLACK_SECONDS="${TAIL_GUARD_TIME_SLACK_SECONDS:-16}"
TAIL_GUARD_MIN_COMPLETION_RATIO="${TAIL_GUARD_MIN_COMPLETION_RATIO:-0.60}"
CREDIT_ASSIGNMENT_ENABLE="${CREDIT_ASSIGNMENT_ENABLE:-True}"
CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR="${CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR:-True}"
CREDIT_ASSIGNMENT_THINK_PARSER_ERROR="${CREDIT_ASSIGNMENT_THINK_PARSER_ERROR:-True}"
CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY="${CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY:-True}"
CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS="${CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS:-True}"
CREDIT_ASSIGNMENT_NGRAM_REPETITION="${CREDIT_ASSIGNMENT_NGRAM_REPETITION:-True}"
CREDIT_ASSIGNMENT_NGRAM_REPETITION_N="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_N:-8}"
CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD:-0.5}"
CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS:-160}"
CREDIT_ASSIGNMENT_SEARCH_BYPASS="${CREDIT_ASSIGNMENT_SEARCH_BYPASS:-True}"
CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL="${CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL:-True}"
CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER="${CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER:-False}"
CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP="${CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP:-False}"
CREDIT_ASSIGNMENT_MAX_TURNS="${CREDIT_ASSIGNMENT_MAX_TURNS:-True}"
CREDIT_ASSIGNMENT_MAX_RESPONSE_LEN="${CREDIT_ASSIGNMENT_MAX_RESPONSE_LEN:-True}"
HORIZON_REWARD_SHAPING="${HORIZON_REWARD_SHAPING:-false}"
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-false}"
LR="${LR:-2e-6}"
KL_COEF="${KL_COEF:-0.0}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.00}"
# A zero-weight reference KL neither changes advantages nor the actor loss.
# Keep the expensive reference-model forward opt-in for this Qwen3 workload.
USE_KL_LOSS="${USE_KL_LOSS:-0}"
USE_WANDB="${USE_WANDB:-1}"
FUSED_HORIZON_REWARD_MIN_MULTIPLIER="${FUSED_HORIZON_REWARD_MIN_MULTIPLIER:-0.2}"
FUSED_HORIZON_REWARD_GAMMA="${FUSED_HORIZON_REWARD_GAMMA:-1.0}"
FUSED_HORIZON_REWARD_STEP_WEIGHT="${FUSED_HORIZON_REWARD_STEP_WEIGHT:-0.7}"
FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT="${FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT:-0.3}"
FUSED_HORIZON_REWARD_TARGET_STEPS="${FUSED_HORIZON_REWARD_TARGET_STEPS:-32}"
FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS="${FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS:-}"
MAX_STEPS="${MAX_STEPS:-64}"
MCP_MAX_STEPS="${MCP_MAX_STEPS:-64}"
MCP_MAX_TOOL_CALLS_PER_TURN="${MCP_MAX_TOOL_CALLS_PER_TURN:-8}"
WEB_SEARCH_MAX_STEPS="${WEB_SEARCH_MAX_STEPS:-64}"
CLI_MAX_STEPS="${CLI_MAX_STEPS:-64}"
TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT:-7200}"
EVAL_TRAJECTORY_TIMEOUT="${EVAL_TRAJECTORY_TIMEOUT:-7200}"
# Bound each group attempt. The rollout stage determines whether an unfinished
# slot is retryable infra, permanent task failure, or policy behavior.
ROLLOUT_GROUP_TIMEOUT="${ROLLOUT_GROUP_TIMEOUT:-3600}"
ROLLOUT_INFRA_RETRY_TIMES="${ROLLOUT_INFRA_RETRY_TIMES:-4}"
MAX_TOOL_OUTPUT_LENGTH="${MAX_TOOL_OUTPUT_LENGTH:-4096}"
# Keep enough queued requests to cover retrieval/tool I/O waits, but cap the
# running batch so growing agent contexts do not repeatedly exhaust the KV pool.
# Queued HTTP requests do not consume the running batch's KV allocation.
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-128}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-128}"
SGLANG_ROUTER_REQUEST_TIMEOUT_SECS="${SGLANG_ROUTER_REQUEST_TIMEOUT_SECS:-21600}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_CONFIG="${EVAL_CONFIG:-}"
EVAL_BENCHMARKS_ROOT="${EVAL_BENCHMARKS_ROOT:-}"
EVAL_INCLUDE_BENCHMARKS="${EVAL_INCLUDE_BENCHMARKS:-asearcher}"
EVAL_EXCLUDE_BENCHMARKS="${EVAL_EXCLUDE_BENCHMARKS:-}"
EVAL_LIMIT_PER_BENCHMARK="${EVAL_LIMIT_PER_BENCHMARK:-0}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_TOP_K="${EVAL_TOP_K:--1}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-2048}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-38000}"
EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-40960}"
EVAL_INITIAL_INFLIGHT_TASKS="${EVAL_INITIAL_INFLIGHT_TASKS:-128}"
EVAL_MAX_INFLIGHT_TASKS="${EVAL_MAX_INFLIGHT_TASKS:-256}"
EVAL_ADAPTIVE_CONCURRENCY="${EVAL_ADAPTIVE_CONCURRENCY:-true}"
EVAL_MIX_DATASETS="${EVAL_MIX_DATASETS:-true}"
EVAL_TERMINATION_RETRY_TIMES="${EVAL_TERMINATION_RETRY_TIMES:-4}"
EVAL_TRAJECTORY_SAMPLE_RATE="${EVAL_TRAJECTORY_SAMPLE_RATE:-1}"
EVAL_DUMP_FAILURES="${EVAL_DUMP_FAILURES:-true}"
NATIVE_SGLANG_SESSION="${NATIVE_SGLANG_SESSION:-true}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-${val_before_train:-false}}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_PROMPT_DATA=()
# Resolve the offload default after CLI parsing so --colocate/--no-colocate also
# changes it. Release-train overrides this below because it replaces the trainer
# actor instead of pausing it.
OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-${offload_train:-}}"
# Avoid torch_memory_saver.pause() for the trainer by default. Its native CUDA
# VMM path can terminate Ray workers on some CUDA/driver combinations. Release
# train uses the existing checkpoint/recreate lifecycle at the cost of disk I/O.
RELEASE_TRAIN="${RELEASE_TRAIN:-true}"
ENABLE_USE_GRM_TRAIN="${ENABLE_USE_GRM_TRAIN:-${enable_use_grm_train:-true}}"
ENABLE_USE_GRM_EVALS="${ENABLE_USE_GRM_EVALS:-${enable_use_grm_evals:-true}}"
GRM_CUSTOM_RM_PATH="${GRM_CUSTOM_RM_PATH:-slime.rollout.rm_hub.openrouter_grm.reward_func}"
TRAIN_GRM_MODEL="${TRAIN_GRM_MODEL:-deepseek/deepseek-v4-flash-0731}"
EVAL_GRM_MODEL="${EVAL_GRM_MODEL:-google/gemini-3-flash-preview}"
GRM_MODE="${GRM_MODE:-score}"
GRM_CONCURRENCY="${GRM_CONCURRENCY:-128}"
GRM_MAX_CONNECTIONS="${GRM_MAX_CONNECTIONS:-128}"
GRM_TIMEOUT="${GRM_TIMEOUT:-60}"
GRM_MAX_RETRIES="${GRM_MAX_RETRIES:-32}"
GRM_RETRY_BASE_DELAY="${GRM_RETRY_BASE_DELAY:-1}"
GRM_RETRY_MAX_DELAY="${GRM_RETRY_MAX_DELAY:-16.0}"
GRM_MAX_INPUT_TOKENS="${GRM_MAX_INPUT_TOKENS:-65536}"
GRM_MAX_NEW_TOKENS="${GRM_MAX_NEW_TOKENS:-1024}"
GRM_TEMPERATURE="${GRM_TEMPERATURE:-0.6}"
GRM_FAILURE_REWARD="${GRM_FAILURE_REWARD:-0.0}"
harness_explicit=false

while [ "$#" -gt 0 ]; do
   case "$1" in
      --harness) FUSED_HARNESS="${2:?Missing value for --harness}"; harness_explicit=true; shift 2 ;;
      --model) MODEL_DIR="${2:?Missing value for --model}"; shift 2 ;;
      --enable-yarn) ENABLE_YARN="${2:?Missing value for --enable-yarn}"; shift 2 ;;
      --yarn-factor) YARN_FACTOR="${2:?Missing value for --yarn-factor}"; shift 2 ;;
      --yarn-original-max-position-embeddings) YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${2:?Missing value for --yarn-original-max-position-embeddings}"; shift 2 ;;
      --disable-thinking) DISABLE_THINKING="${2:?Missing value for --disable-thinking}"; shift 2 ;;
      --discard-historical-thinking) DISCARD_HISTORICAL_THINKING="${2:?Missing value for --discard-historical-thinking}"; shift 2 ;;
      --mcp-disable-step-penalty) RLLM_MCP_DISABLE_STEP_PENALTY="${2:?Missing value for --mcp-disable-step-penalty}"; shift 2 ;;
      --unified-system-prompt) UNIFIED_SYSTEM_PROMPT=True; if [ "${harness_explicit}" = "false" ]; then FUSED_HARNESS=unified_gem; fi; shift ;;
      --no-unified-system-prompt) UNIFIED_SYSTEM_PROMPT=False; if [ "${harness_explicit}" = "false" ]; then FUSED_HARNESS=gem; fi; shift ;;
      --partial-rollout) PARTIAL_ROLLOUT=true; shift ;;
      --no-partial-rollout) PARTIAL_ROLLOUT=false; shift ;;
      --terminal-log-style) TERMINAL_LOG_STYLE="${2:?Missing value for --terminal-log-style}"; shift 2 ;;
      --show-rollout-progress-logs) SHOW_ROLLOUT_PROGRESS_LOGS="${2:?Missing value for --show-rollout-progress-logs}"; shift 2 ;;
      --temperature) TEMPERATURE="${2:?Missing value for --temperature}"; shift 2 ;;
      --micro-batch-size) MICRO_BATCH_SIZE="${2:?Missing value for --micro-batch-size}"; shift 2 ;;
      --update-weights-interval) UPDATE_WEIGHTS_INTERVAL="${2:?Missing value for --update-weights-interval}"; shift 2 ;;
      --rollout-num-gpus-per-engine) ROLLOUT_NUM_GPUS_PER_ENGINE="${2:?Missing value for --rollout-num-gpus-per-engine}"; shift 2 ;;
      --ckpt-step) CKPT_STEP="${2:?Missing value for --ckpt-step}"; shift 2 ;;
      --retrieval-backend) TRAIN_RETRIEVAL_BACKEND="${2:?Missing value for --retrieval-backend}"; EVAL_RETRIEVAL_BACKEND="${TRAIN_RETRIEVAL_BACKEND}"; shift 2 ;;
      --train-retrieval-backend) TRAIN_RETRIEVAL_BACKEND="${2:?Missing value for --train-retrieval-backend}"; shift 2 ;;
      --eval-retrieval-backend) EVAL_RETRIEVAL_BACKEND="${2:?Missing value for --eval-retrieval-backend}"; shift 2 ;;
      --retrieval-concurrency) RETRIEVAL_CONCURRENCY="${2:?Missing value for --retrieval-concurrency}"; shift 2 ;;
      --retrieval-mode) RLLM_RETRIEVAL_MODE="${2:?Missing value for --retrieval-mode}"; shift 2 ;;
      --retrieval-cache-size) RETRIEVAL_CACHE_SIZE="${2:?Missing value for --retrieval-cache-size}"; shift 2 ;;
      --retrieval-max-words) RLLM_RETRIEVAL_MAX_WORDS="${2:?Missing value for --retrieval-max-words}"; shift 2 ;;
      --retrieval-retry-budget) RLLM_RETRIEVAL_RETRY_BUDGET="${2:?Missing value for --retrieval-retry-budget}"; shift 2 ;;
      --retrieval-summary-retry-budget) RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET="${2:?Missing value for --retrieval-summary-retry-budget}"; shift 2 ;;
      --retrieval-lexrank-fallback) : "${2:?Missing value for --retrieval-lexrank-fallback}"; shift 2 ;;
      --retrieval-lexrank-max-words) RLLM_RETRIEVAL_LEXRANK_MAX_WORDS="${2:?Missing value for --retrieval-lexrank-max-words}"; shift 2 ;;
      --retrieval-lexrank-max-sentences) RLLM_RETRIEVAL_LEXRANK_MAX_SENTENCES="${2:?Missing value for --retrieval-lexrank-max-sentences}"; shift 2 ;;
      --retrieval-lexrank-max-input-sentences) RLLM_RETRIEVAL_LEXRANK_MAX_INPUT_SENTENCES="${2:?Missing value for --retrieval-lexrank-max-input-sentences}"; shift 2 ;;
      --retrieval-lexrank-multiprocessing) RLLM_RETRIEVAL_LEXRANK_MULTIPROCESSING="${2:?Missing value for --retrieval-lexrank-multiprocessing}"; shift 2 ;;
      --retrieval-lexrank-workers) RLLM_RETRIEVAL_LEXRANK_WORKERS="${2:?Missing value for --retrieval-lexrank-workers}"; shift 2 ;;
      --ray-num-cpus) RAY_NUM_CPUS="${2:?Missing value for --ray-num-cpus}"; shift 2 ;;
      --tail-guard) TAIL_GUARD="${2:?Missing value for --tail-guard}"; shift 2 ;;
      --tail-guard-time-guard) TAIL_GUARD_TIME_GUARD="${2:?Missing value for --tail-guard-time-guard}"; shift 2 ;;
      --tail-guard-time-multiplier) TAIL_GUARD_TIME_MULTIPLIER="${2:?Missing value for --tail-guard-time-multiplier}"; shift 2 ;;
      --tail-guard-time-slack-seconds) TAIL_GUARD_TIME_SLACK_SECONDS="${2:?Missing value for --tail-guard-time-slack-seconds}"; shift 2 ;;
      --tail-guard-min-completion-ratio) TAIL_GUARD_MIN_COMPLETION_RATIO="${2:?Missing value for --tail-guard-min-completion-ratio}"; shift 2 ;;
      --credit-assignment-enable) CREDIT_ASSIGNMENT_ENABLE="${2:?Missing value for --credit-assignment-enable}"; shift 2 ;;
      --credit-assignment-tool-parser-error) CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR="${2:?Missing value for --credit-assignment-tool-parser-error}"; shift 2 ;;
      --credit-assignment-repeated-search-query) CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY="${2:?Missing value for --credit-assignment-repeated-search-query}"; shift 2 ;;
      --credit-assignment-too-many-tool-calls) CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS="${2:?Missing value for --credit-assignment-too-many-tool-calls}"; shift 2 ;;
      --credit-assignment-ngram-repetition) CREDIT_ASSIGNMENT_NGRAM_REPETITION="${2:?Missing value for --credit-assignment-ngram-repetition}"; shift 2 ;;
      --credit-assignment-ngram-repetition-n) CREDIT_ASSIGNMENT_NGRAM_REPETITION_N="${2:?Missing value for --credit-assignment-ngram-repetition-n}"; shift 2 ;;
      --credit-assignment-ngram-repetition-threshold) CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD="${2:?Missing value for --credit-assignment-ngram-repetition-threshold}"; shift 2 ;;
      --credit-assignment-ngram-repetition-min-tokens) CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS="${2:?Missing value for --credit-assignment-ngram-repetition-min-tokens}"; shift 2 ;;
      --credit-assignment-search-bypass) CREDIT_ASSIGNMENT_SEARCH_BYPASS="${2:?Missing value for --credit-assignment-search-bypass}"; shift 2 ;;
      --credit-assignment-direct-submit-without-tool) CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL="${2:?Missing value for --credit-assignment-direct-submit-without-tool}"; shift 2 ;;
      --credit-assignment-mixed-tool-and-answer) CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER="${2:?Missing value for --credit-assignment-mixed-tool-and-answer}"; shift 2 ;;
      --credit-assignment-tail-guard-early-stop) CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP="${2:?Missing value for --credit-assignment-tail-guard-early-stop}"; shift 2 ;;
      --credit-assignment-max-turns) CREDIT_ASSIGNMENT_MAX_TURNS="${2:?Missing value for --credit-assignment-max-turns}"; shift 2 ;;
      --credit-assignment-max-response-len) CREDIT_ASSIGNMENT_MAX_RESPONSE_LEN="${2:?Missing value for --credit-assignment-max-response-len}"; shift 2 ;;
      --horizon-reward-shaping) HORIZON_REWARD_SHAPING="${2:?Missing value for --horizon-reward-shaping}"; shift 2 ;;
      --horizon-reward-min-multiplier) FUSED_HORIZON_REWARD_MIN_MULTIPLIER="${2:?Missing value for --horizon-reward-min-multiplier}"; shift 2 ;;
      --horizon-reward-gamma) FUSED_HORIZON_REWARD_GAMMA="${2:?Missing value for --horizon-reward-gamma}"; shift 2 ;;
      --horizon-reward-step-weight) FUSED_HORIZON_REWARD_STEP_WEIGHT="${2:?Missing value for --horizon-reward-step-weight}"; shift 2 ;;
      --horizon-reward-tool-call-weight) FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT="${2:?Missing value for --horizon-reward-tool-call-weight}"; shift 2 ;;
      --horizon-reward-target-steps) FUSED_HORIZON_REWARD_TARGET_STEPS="${2:?Missing value for --horizon-reward-target-steps}"; shift 2 ;;
      --horizon-reward-target-tool-calls) FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS="${2:?Missing value for --horizon-reward-target-tool-calls}"; shift 2 ;;
      --enable-dynamic-sampling-filter) ENABLE_DYNAMIC_SAMPLING_FILTER="${2:?Missing value for --enable-dynamic-sampling-filter}"; shift 2 ;;
      --normalize-advantages) NORMALIZE_ADVANTAGES=true; shift ;;
      --no-normalize-advantages) NORMALIZE_ADVANTAGES=false; shift ;;
      --enable_use_grm_train|--enable-use-grm-train) ENABLE_USE_GRM_TRAIN="${2:?Missing value for --enable_use_grm_train}"; shift 2 ;;
      --enable_use_grm_evals|--enable-use-grm-evals) ENABLE_USE_GRM_EVALS="${2:?Missing value for --enable_use_grm_evals}"; shift 2 ;;
      --train-grm-model) TRAIN_GRM_MODEL="${2:?Missing value for --train-grm-model}"; shift 2 ;;
      --eval-grm-model) EVAL_GRM_MODEL="${2:?Missing value for --eval-grm-model}"; shift 2 ;;
      --grm-base-url) GRM_BASE_URL="${2:?Missing value for --grm-base-url}"; shift 2 ;;
      --grm-openrouter-api-key)
         set +x
         export OPENROUTER_API_KEY="${2:?Missing value for --grm-openrouter-api-key}"
         set -x
         shift 2
         ;;
      --grm-mode) GRM_MODE="${2:?Missing value for --grm-mode}"; shift 2 ;;
      --grm-concurrency) GRM_CONCURRENCY="${2:?Missing value for --grm-concurrency}"; shift 2 ;;
      --grm-max-connections) GRM_MAX_CONNECTIONS="${2:?Missing value for --grm-max-connections}"; shift 2 ;;
      --grm-timeout) GRM_TIMEOUT="${2:?Missing value for --grm-timeout}"; shift 2 ;;
      --grm-max-retries) GRM_MAX_RETRIES="${2:?Missing value for --grm-max-retries}"; shift 2 ;;
      --grm-retry-base-delay) GRM_RETRY_BASE_DELAY="${2:?Missing value for --grm-retry-base-delay}"; shift 2 ;;
      --grm-retry-max-delay) GRM_RETRY_MAX_DELAY="${2:?Missing value for --grm-retry-max-delay}"; shift 2 ;;
      --grm-max-input-tokens) GRM_MAX_INPUT_TOKENS="${2:?Missing value for --grm-max-input-tokens}"; shift 2 ;;
      --grm-max-new-tokens) GRM_MAX_NEW_TOKENS="${2:?Missing value for --grm-max-new-tokens}"; shift 2 ;;
      --grm-temperature) GRM_TEMPERATURE="${2:?Missing value for --grm-temperature}"; shift 2 ;;
      --grm-failure-reward) GRM_FAILURE_REWARD="${2:?Missing value for --grm-failure-reward}"; shift 2 ;;
      --max-steps) MAX_STEPS="${2:?Missing value for --max-steps}"; shift 2 ;;
      --mcp-max-steps) MCP_MAX_STEPS="${2:?Missing value for --mcp-max-steps}"; shift 2 ;;
      --mcp-max-tool-calls-per-turn) MCP_MAX_TOOL_CALLS_PER_TURN="${2:?Missing value for --mcp-max-tool-calls-per-turn}"; shift 2 ;;
      --web-search-max-steps) WEB_SEARCH_MAX_STEPS="${2:?Missing value for --web-search-max-steps}"; shift 2 ;;
      --cli-max-steps) CLI_MAX_STEPS="${2:?Missing value for --cli-max-steps}"; shift 2 ;;
      --trajectory-timeout) TRAJECTORY_TIMEOUT="${2:?Missing value for --trajectory-timeout}"; shift 2 ;;
      --eval-trajectory-timeout) EVAL_TRAJECTORY_TIMEOUT="${2:?Missing value for --eval-trajectory-timeout}"; shift 2 ;;
      --rollout-group-timeout) ROLLOUT_GROUP_TIMEOUT="${2:?Missing value for --rollout-group-timeout}"; shift 2 ;;
      --rollout-infra-retry-times) ROLLOUT_INFRA_RETRY_TIMES="${2:?Missing value for --rollout-infra-retry-times}"; shift 2 ;;
      --eval-interval) EVAL_INTERVAL="${2:?Missing value for --eval-interval}"; shift 2 ;;
      --eval-config) EVAL_CONFIG="${2:?Missing value for --eval-config}"; shift 2 ;;
      --eval-benchmarks-root) EVAL_BENCHMARKS_ROOT="${2:?Missing value for --eval-benchmarks-root}"; shift 2 ;;
      --eval-include) EVAL_INCLUDE_BENCHMARKS="${2:?Missing value for --eval-include}"; shift 2 ;;
      --eval-exclude) EVAL_EXCLUDE_BENCHMARKS="${2:?Missing value for --eval-exclude}"; shift 2 ;;
      --eval-limit-per-benchmark) EVAL_LIMIT_PER_BENCHMARK="${2:?Missing value for --eval-limit-per-benchmark}"; shift 2 ;;
      --eval-temperature) EVAL_TEMPERATURE="${2:?Missing value for --eval-temperature}"; shift 2 ;;
      --eval-top-p) EVAL_TOP_P="${2:?Missing value for --eval-top-p}"; shift 2 ;;
      --eval-top-k) EVAL_TOP_K="${2:?Missing value for --eval-top-k}"; shift 2 ;;
      --per-step-max-tokens) PER_STEP_MAX_TOKENS="${2:?Missing value for --per-step-max-tokens}"; shift 2 ;;
      --eval-max-response-len) EVAL_MAX_RESPONSE_LEN="${2:?Missing value for --eval-max-response-len}"; shift 2 ;;
      --eval-max-prompt-len) EVAL_MAX_PROMPT_LEN="${2:?Missing value for --eval-max-prompt-len}"; shift 2 ;;
      --eval-max-context-len) EVAL_MAX_CONTEXT_LEN="${2:?Missing value for --eval-max-context-len}"; shift 2 ;;
      --eval-initial-inflight-tasks) EVAL_INITIAL_INFLIGHT_TASKS="${2:?Missing value for --eval-initial-inflight-tasks}"; shift 2 ;;
      --eval-max-inflight-tasks) EVAL_MAX_INFLIGHT_TASKS="${2:?Missing value for --eval-max-inflight-tasks}"; shift 2 ;;
      --eval-adaptive-concurrency) EVAL_ADAPTIVE_CONCURRENCY="${2:?Missing value for --eval-adaptive-concurrency}"; shift 2 ;;
      --eval-mix-datasets) EVAL_MIX_DATASETS="${2:?Missing value for --eval-mix-datasets}"; shift 2 ;;
      --eval-termination-retry-times) EVAL_TERMINATION_RETRY_TIMES="${2:?Missing value for --eval-termination-retry-times}"; shift 2 ;;
      --eval-trajectory-sample-rate) EVAL_TRAJECTORY_SAMPLE_RATE="${2:?Missing value for --eval-trajectory-sample-rate}"; shift 2 ;;
      --eval-dump-failures) EVAL_DUMP_FAILURES="${2:?Missing value for --eval-dump-failures}"; shift 2 ;;
      --native-sglang-session) NATIVE_SGLANG_SESSION="${2:?Missing value for --native-sglang-session}"; shift 2 ;;
      --val_before_train|--val-before-train) VAL_BEFORE_TRAIN="${2:?Missing value for --val_before_train}"; shift 2 ;;
      --eval-prompt-data)
         shift
         EVAL_PROMPT_DATA=()
         while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
            EVAL_PROMPT_DATA+=("$1")
            shift
         done
         ;;
      --n-samples-per-eval-prompt) N_SAMPLES_PER_EVAL_PROMPT="${2:?Missing value for --n-samples-per-eval-prompt}"; shift 2 ;;
      --offload-train) OFFLOAD_TRAIN="${2:?Missing value for --offload-train}"; shift 2 ;;
      --release-train) RELEASE_TRAIN="${2:?Missing value for --release-train}"; shift 2 ;;
      --max-tool-output-length) MAX_TOOL_OUTPUT_LENGTH="${2:?Missing value for --max-tool-output-length}"; shift 2 ;;
      --sglang-server-concurrency) SGLANG_SERVER_CONCURRENCY="${2:?Missing value for --sglang-server-concurrency}"; shift 2 ;;
      --sglang-max-running-requests) SGLANG_MAX_RUNNING_REQUESTS="${2:?Missing value for --sglang-max-running-requests}"; shift 2 ;;
      --sglang-router-request-timeout-secs) SGLANG_ROUTER_REQUEST_TIMEOUT_SECS="${2:?Missing value for --sglang-router-request-timeout-secs}"; shift 2 ;;
      --colocate) COLOCATE=true; shift ;;
      --no-colocate) COLOCATE=false; shift ;;
      --experiment-name) EXPERIMENT_NAME="${2:?Missing value for --experiment-name}"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
   esac
done

if ! is_truthy "${COLOCATE}"; then
   echo "This launcher is fixed to collocate mode; remove --no-colocate or set COLOCATE=true." >&2
   exit 2
fi
if [ "${ACTOR_NUM_GPUS_PER_NODE}" -ne 8 ]; then
   echo "This launcher requires ACTOR_NUM_GPUS_PER_NODE=8." >&2
   exit 2
fi

OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-${COLOCATE}}"
EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-40960}"

# Release-train avoids native TMS pause/resume failures by trading persistent
# trainer actors for checkpoint/reload I/O.
if is_truthy "${RELEASE_TRAIN}"; then
   if ! is_truthy "${COLOCATE}"; then
      echo "RELEASE_TRAIN=true requires COLOCATE=true in this launcher." >&2
      exit 2
   fi
   OFFLOAD_TRAIN=false
fi

if is_truthy "${ENABLE_USE_GRM_TRAIN}" || is_truthy "${ENABLE_USE_GRM_EVALS}" || [ "${CUSTOM_RM_PATH:-}" = "${GRM_CUSTOM_RM_PATH}" ]; then
   if [ -z "${OPENROUTER_API_KEY:-}" ] \
      && [ -n "${OPENAI_API_KEY:-}" ] \
      && [ -n "${GRM_BASE_URL:-${OPENAI_BASE_URL:-}}" ]; then
      set +x
      export OPENROUTER_API_KEY="${OPENAI_API_KEY}"
      set -x
      GRM_BASE_URL="${GRM_BASE_URL:-${OPENAI_BASE_URL}}"
   fi
   if [ -z "${OPENROUTER_API_KEY:-}" ]; then
      echo "GRM train/eval scoring requires OPENROUTER_API_KEY (or OPENAI_API_KEY with GRM_BASE_URL/OPENAI_BASE_URL)." >&2
      exit 2
   fi
fi

case "${TERMINAL_LOG_STYLE}" in
   progress|rollouts|both) ;;
   *) echo "Invalid TERMINAL_LOG_STYLE=${TERMINAL_LOG_STYLE}; expected progress, rollouts, or both." >&2; exit 2 ;;
esac
FUSED_HARNESS="${FUSED_HARNESS:-gem}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "${SCRIPT_DIR}/lib/retrieval_backend.sh"
RETRIEVAL_MODE="${RLLM_RETRIEVAL_MODE:-${RETRIEVAL_MODE}}"
RETRIEVAL_BACKEND="${TRAIN_RETRIEVAL_BACKEND}"
configure_retrieval_backend
TRAIN_RETRIEVAL_SERVER_URL="${RETRIEVAL_SERVER_URL}"
if [ -n "${EVAL_RETRIEVAL_SERVER_URL+x}" ]; then EVAL_RETRIEVAL_SERVER_URL_EXPLICIT=true; else EVAL_RETRIEVAL_SERVER_URL_EXPLICIT=false; fi
case "${EVAL_RETRIEVAL_BACKEND}" in
   local) EVAL_RETRIEVAL_SERVER_URL="${EVAL_RETRIEVAL_SERVER_URL:-http://10.2.152.50:65432}" ;;
   serper) EVAL_RETRIEVAL_SERVER_URL="${EVAL_RETRIEVAL_SERVER_URL:-http://${SERPER_SERVER_HOST}:${SERPER_SERVER_PORT}}" ;;
   *) echo "Unsupported --eval-retrieval-backend ${EVAL_RETRIEVAL_BACKEND}; expected local or serper." >&2; exit 2 ;;
esac
export TRAIN_RETRIEVAL_SERVER_URL EVAL_RETRIEVAL_SERVER_URL
if [ "${EVAL_RETRIEVAL_BACKEND}" = "serper" ] && [ -z "${SERPER_PROXY_TOKEN:-}" ]; then
   echo "SERPER_PROXY_TOKEN is required for Serper interval evals" >&2
   exit 2
fi
if [ "${EVAL_RETRIEVAL_BACKEND}" = "serper" ]; then
   export SERPER_PROXY_TOKEN SERPER_SEARCH_URL
fi
BASE_DIR="$(cd -- "${REPO_ROOT}/.." &>/dev/null && pwd)"
RUNS_ROOT="${RUNS_ROOT:-/share/nlp/share/gem/runs}"
EVAL_BENCHMARKS_ROOT="${EVAL_BENCHMARKS_ROOT:-${SCRIPT_DIR}/artifacts/benchmarks}"

default_experiment_name() {
   local prefix="odyssey-q3-4b-local-dev"
   #local prefix="odyssey-q3-8b-think-dev"
   #local prefix="fused-dapo-q3-8b-dht-gem-sync-dev"
   #local prefix="fused-dapo-q3-4b-rft-dht-gem-sync-dev"  # w/ rft warmup
   #local prefix="fused-dapo-q3-4b-dht-gem-sync-dev"  # w/o rft warmup
   #local prefix="fused-dapo-q3-4b-think-gem-sync-dev"
   #local prefix="fused-dapo-q3-4b-pet-gem-sync-dev"
   #local prefix="webqa-dapo-q3-4b-think-pet-gem-sync-dev"
   #local prefix="fused-dapo-q3-4b-no_think-gem-sync-dev"
   #local prefix="asearcher-dapo-q3-4b-no_think-gem-sync-dev"
   #local prefix="asearcher-dapo-q3-4b-think-gem-sync-dev"
   #local prefix="webqa-dapo-q3-4b-no_think-gem-sync-dev"
   #local prefix="webqa-dapo-q3-8b-no_think-gem-sync-dev"
   #local prefix="webqa-dapo-q3.5-4b-no_think-gem-sync-dev"
   #local prefix="mcp-dapo-q3-4b-think-gem-sync-dev"
   #local prefix="asearcher-dapo-q3.5-4b-no_think-gem-sync-dev"
   #local prefix="asearcher-dapo-q3-4b-think-gem-sync-dev"
   #local prefix="asearcher-dapo-q3-8b-no_think-gem-sync-dev"
   #local prefix="asearcher-dapo-q3-8b-no_think-gem-async-dev"
   #local prefix="webqa-dapo-q3-4b-no_think-gem-async-dev0"
   local max_dev=-1
   local root base suffix
   for root in "${RUNS_ROOT}" "${REPO_ROOT}/checkpoints/FusedRL" "${REPO_ROOT}/experiments/logs/FusedRL"; do
      [ -d "${root}" ] || continue
      while IFS= read -r base; do
         suffix="${base#${prefix}}"
         if [[ "${suffix}" =~ ^[0-9]+$ ]] && [ "${suffix}" -gt "${max_dev}" ]; then
            max_dev="${suffix}"
         fi
      done < <(find "${root}" -maxdepth 1 -type d -printf '%f\n' 2>/dev/null)
   done
   echo "${prefix}$((max_dev + 1))"
}

if [ "${SLIME_CLEANUP:-0}" = "1" ] && [ "${SLIME_CLEANUP_CONFIRM:-0}" = "1" ]; then
   ray stop --force 2>/dev/null || true
   pkill -9 -f 'ray::' 2>/dev/null || true
   pkill -9 -f 'sglang.*fused_agent' 2>/dev/null || true
   sleep 3
elif [ "${SLIME_CLEANUP:-0}" = "1" ]; then
   echo "SLIME_CLEANUP=1 ignored because SLIME_CLEANUP_CONFIRM=1 is not set; preserving existing processes and artifacts."
fi

MODEL_CONFIG="${MODEL_CONFIG:-qwen3-4B}"
#MODEL_CONFIG="${MODEL_CONFIG:-qwen3-8B}"
source "${REPO_ROOT}/scripts/models/${MODEL_CONFIG}.sh"

# Rollout TP must divide the visible rollout GPUs.  The actor TP/CP defaults
# remain Odyssey-specific; rollout TP follows the Qwen3 model-size contract.
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

case "${MODEL_CONFIG,,}" in
   qwen3-4b|qwen3-4b-*) DEFAULT_TP_SIZE=1 ;;
   *) DEFAULT_TP_SIZE=2 ;;
esac
DEFAULT_CP_SIZE=2

# In colocate mode both phases use the same eight GPU slots in alternation.
# ACTOR_GPUS and ROLLOUT_GPUS are global counts.
ACTOR_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
ROLLOUT_GPUS="${ROLLOUT_GPUS:-${ACTOR_GPUS}}"
if [ "${ROLLOUT_GPUS}" -ne "${ACTOR_GPUS}" ]; then
   echo "Colocate mode requires ROLLOUT_GPUS=ACTOR_GPUS=${ACTOR_GPUS}; got ${ROLLOUT_GPUS}." >&2
   exit 2
fi
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${DEFAULT_ROLLOUT_NUM_GPUS_PER_ENGINE}}"
CP_SIZE="${CP_SIZE:-${DEFAULT_CP_SIZE}}"
PP_SIZE="${PP_SIZE:-1}"

if [ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -lt 1 ] \
   || [ $((ROLLOUT_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
   echo "ROLLOUT_GPUS=${ROLLOUT_GPUS} must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}" >&2
   exit 2
fi
if [ $((ACTOR_NUM_GPUS_PER_NODE % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
   echo "ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE} must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}; otherwise a TP engine overlaps GPUs at a node boundary." >&2
   exit 2
fi
ROLLOUT_ENGINE_COUNT=$((ROLLOUT_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE))

# Explicit CP overrides clamp TP to the remaining actor-GPU dimension. The 8B
# resume topology stays at CP1/TP2 so its distributed optimizer can reshard DP.
if [ "${CP_SIZE}" -lt 1 ] || [ "${PP_SIZE}" -lt 1 ]; then
   echo "CP_SIZE and PP_SIZE must be positive; got CP_SIZE=${CP_SIZE}, PP_SIZE=${PP_SIZE}" >&2
   exit 2
fi
TP_CAPACITY=$((ACTOR_GPUS / CP_SIZE / PP_SIZE))
if [ "${TP_CAPACITY}" -lt 1 ]; then
   echo "ACTOR_GPUS=${ACTOR_GPUS} is insufficient for CP_SIZE=${CP_SIZE}, PP_SIZE=${PP_SIZE}" >&2
   exit 2
fi
if [ "${DEFAULT_TP_SIZE}" -gt "${TP_CAPACITY}" ]; then
   DEFAULT_TP_SIZE="${TP_CAPACITY}"
fi
TP_SIZE="${TP_SIZE:-${DEFAULT_TP_SIZE}}"
MODEL_PARALLEL_SIZE=$((TP_SIZE * CP_SIZE * PP_SIZE))
if [ $((ACTOR_GPUS % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
   echo "ACTOR_GPUS=${ACTOR_GPUS} must be divisible by TP_SIZE*CP_SIZE*PP_SIZE=${MODEL_PARALLEL_SIZE}" >&2
   exit 2
fi

if [ "${UPDATE_WEIGHTS_INTERVAL}" -lt 1 ]; then
   echo "UPDATE_WEIGHTS_INTERVAL must be at least 1; got ${UPDATE_WEIGHTS_INTERVAL}" >&2
   exit 2
fi
if is_truthy "${PARTIAL_ROLLOUT}"; then
   echo "Fused-agent partial rollout is unsupported: multi-turn environment and TiTO state cannot yet be resumed atomically." >&2
   exit 2
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(default_experiment_name)}"
MODEL_DIR="${MODEL_DIR:-/share/nlp/share/plm/Qwen3-4B}"
#MODEL_DIR="${MODEL_DIR:-/share/nlp/share/plm/Qwen3-8B}"
#MODEL_DIR="${MODEL_DIR:-/share/nlp/share/plm/Qwen3.5-4B}"
#MODEL_DIR="${MODEL_DIR:-/share/nlp/share/plm/Qwen3-4B}"
#MODEL_DIR="${MODEL_DIR:-/share/nlp/share/gem/Qwen3-4B-RFT-Warmup-v0}"  # init RL training with a self-distil wamrup ckpt
REF_LOAD="${REF_LOAD:-${MODEL_DIR}_torch_dist}"
RUN_ROOT="${RUN_ROOT:-${RUNS_ROOT}/${EXPERIMENT_NAME}}"
SAVE_DIR="${SAVE_DIR:-${RUN_ROOT}/checkpoints}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${BASE_DIR}/Megatron-LM}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
EPISODE_LOG_DIR="${EPISODE_LOG_DIR:-${LOG_ROOT}/episodes}"
DUMP_DETAILS="${DUMP_DETAILS:-${LOG_ROOT}/debug}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-${LOG_ROOT}/eval_cache}"
PREPARED_PROMPT_DATA="${PREPARED_PROMPT_DATA:-${RUN_ROOT}/data/prepared_train.parquet}"
# Keep full disk updates under the run directory so release-train checkpoints
# and rollout weight versions share one durable lifecycle.
UPDATE_WEIGHT_DISK_DIR="${UPDATE_WEIGHT_DISK_DIR:-${RUN_ROOT}/rollout_weights}"
WANDB_DIR="${WANDB_DIR:-${LOG_ROOT}/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${RUN_ROOT}/cache/wandb}"
HF_HOME="${HF_HOME:-${RUN_ROOT}/cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-${RUN_ROOT}/cache/torch}"
# Triton launchers are compiled concurrently by every SGLang engine. Keeping
# this cache on NFS can make os.replace() fail with EBUSY while another process
# has the same launcher loaded, so use the node-local filesystem by default.
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/slime-triton-cache/${USER:-$(id -un)}/${EXPERIMENT_NAME}}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/cache/xdg}"
MCP_ENV_ROOT="${MCP_ENV_ROOT:-${RUN_ROOT}/cache/mcp_envs}"
MCP_ENV_COPY_CONCURRENCY="${MCP_ENV_COPY_CONCURRENCY:-16}"

mkdir -p \
   "${SAVE_DIR}" \
   "${LOG_ROOT}" \
   "${EPISODE_LOG_DIR}" \
   "${DUMP_DETAILS}" \
   "${EVAL_CACHE_DIR}" \
   "$(dirname "${PREPARED_PROMPT_DATA}")" \
   "${UPDATE_WEIGHT_DISK_DIR}" \
   "${WANDB_DIR}" \
   "${WANDB_CACHE_DIR}" \
   "${HF_HOME}" \
   "${TORCH_HOME}" \
   "${TRITON_CACHE_DIR}" \
   "${XDG_CACHE_HOME}" \
   "${MCP_ENV_ROOT}"

if [ -n "${EVAL_INTERVAL}" ] && [ -z "${EVAL_CONFIG}" ] && [ "${#EVAL_PROMPT_DATA[@]}" -eq 0 ]; then
   EVAL_CONFIG="${LOG_ROOT}/eval_config.yaml"
   PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 -m slime_plugins.evals.fused_benchmark_config \
      --benchmarks-root "${EVAL_BENCHMARKS_ROOT}" \
      --output-config "${EVAL_CONFIG}" \
      --cache-dir "${EVAL_CACHE_DIR}" \
      --include "${EVAL_INCLUDE_BENCHMARKS}" \
      --exclude "${EVAL_EXCLUDE_BENCHMARKS}" \
      --limit-per-benchmark "${EVAL_LIMIT_PER_BENCHMARK}" \
      --n-samples-per-prompt "${N_SAMPLES_PER_EVAL_PROMPT}" \
      --temperature "${EVAL_TEMPERATURE}" \
      --top-p "${EVAL_TOP_P}" \
      --top-k "${EVAL_TOP_K}" \
      --long-response-len "${EVAL_MAX_RESPONSE_LEN}"
fi

if [ -n "${EVAL_INTERVAL}" ]; then
   if [ "${EVAL_INITIAL_INFLIGHT_TASKS}" -lt 1 ] || [ "${EVAL_MAX_INFLIGHT_TASKS}" -lt 1 ]; then
      echo "Eval inflight limits must be positive." >&2
      exit 2
   fi
   if [ "${EVAL_INITIAL_INFLIGHT_TASKS}" -gt "${EVAL_MAX_INFLIGHT_TASKS}" ]; then
      echo "--eval-initial-inflight-tasks must not exceed --eval-max-inflight-tasks." >&2
      exit 2
   fi
   if ! [[ "${EVAL_TERMINATION_RETRY_TIMES}" =~ ^[0-9]+$ ]]; then
      echo "--eval-termination-retry-times must be a non-negative integer." >&2
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
fi

if [ -n "${TRAIN_FILES:-}" ]; then
   IFS=',' read -r -a TRAIN_FILES <<< "${TRAIN_FILES}"
else
   # Current slime's stock RolloutDataSource takes one --prompt-data path. Keep
   # TRAIN_FILES as the source of truth; only build a prepared parquet when
   # multiple source files need to be merged.
   TRAIN_FILES=(
      "${SCRIPT_DIR}/artifacts/mcp_data_final/train.parquet"
      "${SCRIPT_DIR}/artifacts/search_data_final/train.parquet"
      #"${SCRIPT_DIR}/artifacts/asearcher.parquet"
   )
fi
SHUFFLE_TRAIN_DATA="${SHUFFLE_TRAIN_DATA:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-slime.rollout.fused_agent.generate.generate}"
CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-}"
CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"
if is_truthy "${HORIZON_REWARD_SHAPING}" && [ -z "${CUSTOM_REWARD_POST_PROCESS_PATH}" ]; then
   CUSTOM_REWARD_POST_PROCESS_PATH="slime.rollout.filter_hub.horizon_reward_shaping.post_process_rewards"
fi

if [ ! -d "${MODEL_DIR}" ]; then
   echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
   exit 1
fi
RESOLVED_TRAIN_FILES=()
for train_file in "${TRAIN_FILES[@]}"; do
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
   echo "Also tried: ${rllm_fallback}" >&2
   echo "Also tried: ${rllm_experiments_fallback}" >&2
   exit 1
done

if [ "${#RESOLVED_TRAIN_FILES[@]}" -eq 0 ]; then
   echo "No training data files resolved." >&2
   exit 1
fi

if [ "${#RESOLVED_TRAIN_FILES[@]}" -eq 1 ]; then
   PROMPT_DATA_FOR_SLIME="${RESOLVED_TRAIN_FILES[0]}"
else
   PROMPT_DATA_FOR_SLIME="${PREPARED_PROMPT_DATA:-${LOG_ROOT}/prepared_train.parquet}"
   mkdir -p "$(dirname "${PROMPT_DATA_FOR_SLIME}")"
   python3 - "${PROMPT_DATA_FOR_SLIME}" "${SHUFFLE_TRAIN_DATA}" "${SHUFFLE_SEED}" "${REPO_ROOT}" "${BASE_DIR}/rllm" "${RESOLVED_TRAIN_FILES[@]}" <<'PY'
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

output = Path(sys.argv[1])
shuffle = sys.argv[2].lower() in {"1", "true", "yes", "on"}
seed = int(sys.argv[3])
repo_root = Path(sys.argv[4])
rllm_root = Path(sys.argv[5])
paths = [Path(p) for p in sys.argv[6:]]

tables = [pq.read_table(path) for path in paths]
schema = pa.unify_schemas([table.schema for table in tables], promote_options="permissive")
all_columns = schema.names
types = {field.name: field.type for field in schema}

aligned = []
for table in tables:
    arrays = []
    for name in all_columns:
        if name in table.column_names:
            column = table[name]
            if not column.type.equals(types[name]):
                column = column.cast(types[name])
            arrays.append(column)
        else:
            arrays.append(pa.nulls(table.num_rows, type=types[name]))
    aligned.append(pa.table(arrays, names=all_columns))

combined = pa.concat_tables(aligned, promote_options="default")

missing_mcp_tools = Counter()
keep_rows = []
for data_source, extra_info in zip(combined["data_source"].to_pylist(), combined["extra_info"].to_pylist()):
    if str(data_source).lower() != "mcp":
        keep_rows.append(True)
        continue

    extra_info = extra_info or {}
    tools_py = extra_info.get("tools_py")
    if not tools_py and extra_info.get("data_root"):
        tools_py = str(Path(extra_info["data_root"]) / "tools.py")
    if not tools_py:
        missing_mcp_tools["<missing tools_py>"] += 1
        keep_rows.append(False)
        continue

    tools_path = Path(tools_py)
    candidates = [tools_path] if tools_path.is_absolute() else [
        repo_root / tools_path,
        repo_root / "experiments" / "fused" / "assets" / tools_path.name,
        rllm_root / tools_path,
    ]
    resolved = any(candidate.is_file() for candidate in candidates)
    keep_rows.append(resolved)
    if not resolved:
        missing_mcp_tools[str(tools_py)] += 1

if missing_mcp_tools:
    combined = combined.filter(pa.array(keep_rows, type=pa.bool_()))

if shuffle and combined.num_rows:
    indices = pa.array(np.random.default_rng(seed).permutation(combined.num_rows), type=pa.int64())
    combined = combined.take(indices)

# Write to a temp file then atomically rename, so a crash mid-write never leaves
# a partial prepared parquet.
tmp_output = output.with_suffix(output.suffix + ".tmp")
pq.write_table(combined, tmp_output)
tmp_output.replace(output)
print(f"Wrote fused train parquet: {output}")
print(f"Rows: {combined.num_rows}")
print("Inputs:")
for path, table in zip(paths, tables):
    print(f"  {path}: {table.num_rows}")
if missing_mcp_tools:
    print(f"Dropped unresolved MCP rows: {sum(missing_mcp_tools.values())}")
    for tools_py, count in sorted(missing_mcp_tools.items()):
        print(f"  {tools_py}: {count}")
print(f"Shuffle: {shuffle} seed={seed}")
PY
fi
if [ "${PREPARE_DATA_ONLY:-0}" = "1" ]; then
   exit 0
fi
if [ ! -d "${MEGATRON_LM_PATH}" ]; then
   echo "MEGATRON_LM_PATH does not exist: ${MEGATRON_LM_PATH}" >&2
   exit 1
fi
if [ ! -d "${REF_LOAD}" ]; then
   if [ "${AUTO_CONVERT_REF:-1}" != "1" ]; then
      echo "REF_LOAD does not exist: ${REF_LOAD}" >&2
      echo "Set REF_LOAD to an existing Megatron torch_dist checkpoint, or enable AUTO_CONVERT_REF=1." >&2
      exit 1
   fi
   echo "REF_LOAD does not exist: ${REF_LOAD}; converting ${MODEL_DIR} with tools/convert_hf_to_torch_dist.py"
   mkdir -p "$(dirname "${REF_LOAD}")"
   CONVERT_GPUS="${CONVERT_GPUS:-1}"
   CONVERT_MASTER_PORT="${CONVERT_MASTER_PORT:-12355}"
   PYTHONPATH="${MEGATRON_LM_PATH}:${REPO_ROOT}:${PYTHONPATH:-}" \
   torchrun --nproc_per_node "${CONVERT_GPUS}" --master_port "${CONVERT_MASTER_PORT}" \
      "${REPO_ROOT}/tools/convert_hf_to_torch_dist.py" \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${MODEL_DIR}" \
      --save "${REF_LOAD}"
fi

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-15472}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-24576}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-40960}"
# CP=2 splits one 40960-token sample across two GPUs.  Using the full context
# budget per GPU lets dynamic batching pack roughly twice that many tokens and
# can OOM in the vocabulary-parallel softmax on long rollout batches.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-20480}"
LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-20480}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
# Keep the post-filter training batch at 50/50 webqa and mcp. The synchronous
# collector keeps sampling each family until both accepted quotas are full.
ROLLOUT_TASK_FAMILY_QUOTAS="${ROLLOUT_TASK_FAMILY_QUOTAS:-webqa=0.5,mcp=0.5}"
# Bound aggressive admission even when low ROI or long-tail groups keep the
# collector refilling candidates before the previous wave fully drains.
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-128}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
# Keep enough prompt groups admitted to fill every decode engine even when
# older groups have only one or two long-tail trajectories left.
SYNC_MIN_PENDING_GROUPS="${SYNC_MIN_PENDING_GROUPS:-$((ROLLOUT_ENGINE_COUNT * 4))}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-360}"
if [ $((ROLLOUT_BATCH_SIZE % 2)) -ne 0 ]; then
   echo "ROLLOUT_BATCH_SIZE must be even for the 50/50 webqa/mcp training mix; got ${ROLLOUT_BATCH_SIZE}." >&2
   exit 2
fi
EFFECTIVE_GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT))}"
TRAIN_DATA_PARALLEL_SIZE=$((ACTOR_GPUS / TP_SIZE / CP_SIZE / PP_SIZE))
MICRO_BATCH_DATA_PARALLEL_SIZE=$((MICRO_BATCH_SIZE * TRAIN_DATA_PARALLEL_SIZE))
if [ $((EFFECTIVE_GLOBAL_BATCH_SIZE % MICRO_BATCH_DATA_PARALLEL_SIZE)) -ne 0 ]; then
   echo "GLOBAL_BATCH_SIZE=${EFFECTIVE_GLOBAL_BATCH_SIZE} must be divisible by MICRO_BATCH_SIZE*training_DP=${MICRO_BATCH_SIZE}*${TRAIN_DATA_PARALLEL_SIZE}=${MICRO_BATCH_DATA_PARALLEL_SIZE}." >&2
   exit 2
fi
ENABLE_DYNAMIC_SAMPLING_FILTER="${ENABLE_DYNAMIC_SAMPLING_FILTER:-true}"
DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
# This launcher requires strict reward-variance filtering. Zero disables the
# collector's fallback that would otherwise admit rejected groups after a
# candidate threshold; overwrite inherited environment values intentionally.
FULLY_ASYNC_FILTER_RELAX_AFTER_GROUPS=0
if ! is_truthy "${ENABLE_DYNAMIC_SAMPLING_FILTER}"; then
   echo "This launcher requires strict dynamic sampling; --enable-dynamic-sampling-filter must be true." >&2
   exit 2
fi

# Drain or abort every submitted trajectory before the next weight update.
# The fully-async collector intentionally keeps trajectories alive across
# training-step boundaries, which can mix SGLang weight versions within one
# fused trajectory.
ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-slime.rollout.sglang_rollout.generate_rollout}"
FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY:-true}"
FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY="${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY:-$((ROLLOUT_ENGINE_COUNT * 4))}"
FULLY_ASYNC_MAX_GROUP_CONCURRENCY="${FULLY_ASYNC_MAX_GROUP_CONCURRENCY:-$((ROLLOUT_ENGINE_COUNT * 8))}"
FULLY_ASYNC_CONCURRENCY_STEP="${FULLY_ASYNC_CONCURRENCY_STEP:-${ROLLOUT_ENGINE_COUNT}}"
FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL="${FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL:-10}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MAX_CONTEXT_LEN}}"
if [ "${MAX_MODEL_LEN}" -ne "${MAX_CONTEXT_LEN}" ]; then
   echo "MAX_MODEL_LEN=${MAX_MODEL_LEN} must equal MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN} for this slime launcher." >&2
   exit 2
fi
if [ $((MAX_TOKENS_PER_GPU * CP_SIZE)) -lt "${MAX_CONTEXT_LEN}" ]; then
   echo "MAX_TOKENS_PER_GPU*CP_SIZE=$((MAX_TOKENS_PER_GPU * CP_SIZE)) is smaller than MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN}." >&2
   exit 2
fi
if [ $((LOG_PROBS_MAX_TOKENS_PER_GPU * CP_SIZE)) -lt "${MAX_CONTEXT_LEN}" ]; then
   echo "LOG_PROBS_MAX_TOKENS_PER_GPU*CP_SIZE=$((LOG_PROBS_MAX_TOKENS_PER_GPU * CP_SIZE)) is smaller than MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN}." >&2
   exit 2
fi
if is_truthy "${ENABLE_YARN}"; then
   python3 - "${YARN_FACTOR}" "${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}" "${MAX_CONTEXT_LEN}" <<'PY'
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

TRAIN_NUM_ROWS=$(python3 - "${PROMPT_DATA_FOR_SLIME}" <<'PY'
import sys
import pyarrow.parquet as pq

print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)
if [ "${TRAIN_NUM_ROWS}" -le 0 ]; then
   echo "Prompt data has no rows: ${PROMPT_DATA_FOR_SLIME}" >&2
   exit 2
fi
CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   # Release-train keeps non-boundary checkpoints only as temporary reload
   # points. Save every rollout step by default so an interruption cannot roll
   # back the actor/optimizer state to an earlier training step. Larger values
   # remain available explicitly via SAVE_INTERVAL when checkpoint I/O matters
   # more than exact interruption recovery.
   --save-interval "${SAVE_INTERVAL:-10}"
)
# Resume training state (model/optimizer/rng/step + rollout data state) from an
# existing Megatron checkpoint dir. Defaults to SAVE_DIR so a re-launch with the
# same EXPERIMENT_NAME continues from the latest saved iteration. Set LOAD=""
# explicitly to force a fresh start from the HF/ref checkpoint.
LOAD="${LOAD-${SAVE_DIR}}"
if [ -n "${LOAD}" ]; then
   CKPT_ARGS+=(--load "${LOAD}")
fi
if [ -n "${CKPT_STEP}" ]; then
   if ! [[ "${CKPT_STEP}" =~ ^[0-9]+$ ]]; then
      echo "CKPT_STEP must be a non-negative integer; got ${CKPT_STEP}." >&2
      exit 2
   fi
   if [ -z "${LOAD}" ]; then
      echo "CKPT_STEP requires LOAD to point to a Megatron checkpoint root." >&2
      exit 2
   fi
   CKPT_ARGS+=(--ckpt-step "${CKPT_STEP}")
fi

# torch_dist checkpoints can reshard the distributed optimizer across a changed
# data-parallel world size. Model-parallel topology must remain unchanged.
if [ -n "${LOAD}" ] && [ -f "${LOAD}/latest_checkpointed_iteration.txt" ]; then
   python3 - "${LOAD}" "${CKPT_STEP}" "${TP_SIZE}" "${PP_SIZE}" "${CP_SIZE}" "${ACTOR_GPUS}" <<'PY'
import pathlib
import sys

import torch

load_dir = pathlib.Path(sys.argv[1])
iteration = int(sys.argv[2]) if sys.argv[2] else int((load_dir / "latest_checkpointed_iteration.txt").read_text().strip())
common_path = load_dir / f"iter_{iteration:07d}" / "common.pt"
if not common_path.is_file():
    raise SystemExit(f"Resume checkpoint metadata is missing: {common_path}")

state = torch.load(common_path, map_location="cpu", weights_only=False)
checkpoint_args = state.get("args")
if checkpoint_args is None:
    raise SystemExit(f"Resume checkpoint has no saved training arguments: {common_path}")

expected = {
    "tensor_model_parallel_size": int(sys.argv[3]),
    "pipeline_model_parallel_size": int(sys.argv[4]),
    "context_parallel_size": int(sys.argv[5]),
}
mismatches = {
    name: (getattr(checkpoint_args, name, None), value)
    for name, value in expected.items()
    if getattr(checkpoint_args, name, None) != value
}
if mismatches:
    details = ", ".join(
        f"{name}: checkpoint={actual}, requested={requested}"
        for name, (actual, requested) in mismatches.items()
    )
    raise SystemExit(f"Refusing inexact resume from iteration {iteration}: {details}")

checkpoint_world_size = getattr(checkpoint_args, "world_size", None)
requested_world_size = int(sys.argv[6])
print(
    f"Validated resume from iteration {iteration}: model_parallel={expected}, "
    f"world_size={checkpoint_world_size}->{requested_world_size} (data-parallel reshard)"
)
PY
fi

if is_truthy "${NO_LOAD_OPTIM:-false}"; then
   echo "NO_LOAD_OPTIM is not supported by this launcher because it discards optimizer and RNG resume state. Unset NO_LOAD_OPTIM." >&2
   exit 2
fi

ROLLOUT_ARGS=(
   --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"

   --prompt-data "${PROMPT_DATA_FOR_SLIME}"
   --input-key "${INPUT_KEY:-prompt}"
   --label-key "${LABEL_KEY:-reward_model}"
   --metadata-key "${METADATA_KEY:-extra_info}"
   --tool-key "${TOOL_KEY:-tools}"
   --rollout-shuffle

   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-prompt-len "${MAX_PROMPT_LENGTH}"
   --rollout-max-response-len "${MAX_RESPONSE_LENGTH}"
   --rollout-temperature "${TEMPERATURE}"
   --rollout-top-p "${TOP_P:-1.0}"
   --rollout-infra-retry-times "${ROLLOUT_INFRA_RETRY_TIMES}"

   --global-batch-size "${EFFECTIVE_GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --balance-data
)

if [ -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ]; then
   ROLLOUT_ARGS+=(
      --dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}"
      --fully-async-filter-relax-after-groups 0
   )
fi
if [ -n "${ROLLOUT_TASK_FAMILY_QUOTAS:-}" ]; then
   ROLLOUT_ARGS+=(--rollout-task-family-quotas "${ROLLOUT_TASK_FAMILY_QUOTAS}")
fi
if [ "${ENABLE_QUOTA_BUCKET_SAMPLING:-1}" = "1" ]; then
   ROLLOUT_ARGS+=(--enable-quota-bucket-sampling)
   if [ -z "${BUFFER_FILTER_PATH:-}" ]; then
      ROLLOUT_ARGS+=(--buffer-filter-path "slime.rollout.filter_hub.buffer_filters.quota_bucket_by_steps")
   fi
fi
if [ -n "${BUFFER_FILTER_PATH:-}" ]; then
   ROLLOUT_ARGS+=(--buffer-filter-path "${BUFFER_FILTER_PATH}")
fi
if [ -n "${CUSTOM_GENERATE_FUNCTION_PATH}" ]; then
   ROLLOUT_ARGS+=(--custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}")
fi
if [ "${APPLY_CHAT_TEMPLATE:-1}" = "1" ]; then
   ROLLOUT_ARGS+=(--apply-chat-template)
fi
if [ -n "${CUSTOM_RM_PATH}" ]; then
   ROLLOUT_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")
elif [ -n "${RM_TYPE:-}" ]; then
   ROLLOUT_ARGS+=(--rm-type "${RM_TYPE}")
fi
if is_truthy "${ENABLE_USE_GRM_TRAIN}" || is_truthy "${ENABLE_USE_GRM_EVALS}" || [ "${CUSTOM_RM_PATH:-}" = "${GRM_CUSTOM_RM_PATH}" ]; then
   ROLLOUT_ARGS+=(
      --grm-custom-rm-path "${GRM_CUSTOM_RM_PATH}"
      --train-grm-model "${TRAIN_GRM_MODEL}"
      --eval-grm-model "${EVAL_GRM_MODEL}"
      --grm-mode "${GRM_MODE}"
      --grm-concurrency "${GRM_CONCURRENCY}"
      --grm-max-connections "${GRM_MAX_CONNECTIONS}"
      --grm-timeout "${GRM_TIMEOUT}"
      --grm-max-retries "${GRM_MAX_RETRIES}"
      --grm-retry-base-delay "${GRM_RETRY_BASE_DELAY}"
      --grm-retry-max-delay "${GRM_RETRY_MAX_DELAY}"
      --grm-max-input-tokens "${GRM_MAX_INPUT_TOKENS}"
      --grm-max-new-tokens "${GRM_MAX_NEW_TOKENS}"
      --grm-temperature "${GRM_TEMPERATURE}"
      --grm-failure-reward "${GRM_FAILURE_REWARD}"
   )
   if is_truthy "${ENABLE_USE_GRM_TRAIN}"; then
      ROLLOUT_ARGS+=(--enable-use-grm-train)
   fi
   if is_truthy "${ENABLE_USE_GRM_EVALS}"; then
      ROLLOUT_ARGS+=(--enable-use-grm-evals)
   fi
   if [ -n "${GRM_BASE_URL:-}" ]; then
      ROLLOUT_ARGS+=(--grm-base-url "${GRM_BASE_URL}")
   fi
   if [ -n "${OPENROUTER_SITE_URL:-}" ]; then
      ROLLOUT_ARGS+=(--grm-openrouter-site-url "${OPENROUTER_SITE_URL}")
   fi
   if [ -n "${OPENROUTER_APP_NAME:-}" ]; then
      ROLLOUT_ARGS+=(--grm-openrouter-app-name "${OPENROUTER_APP_NAME}")
   fi
   if [ -n "${GRM_SYSTEM_PROMPT:-}" ]; then
      ROLLOUT_ARGS+=(--grm-system-prompt "${GRM_SYSTEM_PROMPT}")
   fi
fi
if [ -n "${CUSTOM_REWARD_POST_PROCESS_PATH}" ]; then
   ROLLOUT_ARGS+=(--custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}")
fi

EVAL_ARGS=()
if [ -n "${EVAL_INTERVAL}" ]; then
   EVAL_ARGS+=(
      --eval-function-path slime.rollout.sglang_rollout.generate_rollout
      --eval-interval "${EVAL_INTERVAL}"
   )
   if [ -n "${EVAL_CONFIG}" ]; then
      EVAL_ARGS+=(--eval-config "${EVAL_CONFIG}")
   elif [ "${#EVAL_PROMPT_DATA[@]}" -gt 0 ]; then
      EVAL_ARGS+=(--eval-prompt-data "${EVAL_PROMPT_DATA[@]}")
   else
      echo "--eval-interval requires --eval-config or --eval-prompt-data." >&2
      exit 2
   fi
   EVAL_ARGS+=(
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
      --eval-temperature "${EVAL_TEMPERATURE}"
      --eval-top-p "${EVAL_TOP_P}"
      --eval-top-k "${EVAL_TOP_K}"
      --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
      --eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}"
      --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
      --eval-initial-inflight-tasks "${EVAL_INITIAL_INFLIGHT_TASKS}"
      --eval-max-inflight-tasks "${EVAL_MAX_INFLIGHT_TASKS}"
      --eval-termination-retry-times "${EVAL_TERMINATION_RETRY_TIMES}"
      --custom-eval-rollout-log-function-path slime_plugins.evals.results_table.log_eval_results_table
   )
   if is_truthy "${EVAL_ADAPTIVE_CONCURRENCY}"; then
      EVAL_ARGS+=(--eval-adaptive-concurrency)
   else
      EVAL_ARGS+=(--no-eval-adaptive-concurrency)
   fi
   if is_truthy "${EVAL_MIX_DATASETS}"; then
      EVAL_ARGS+=(--eval-mix-datasets)
   else
      EVAL_ARGS+=(--no-eval-mix-datasets)
   fi
   if ! is_truthy "${VAL_BEFORE_TRAIN}"; then
      EVAL_ARGS+=(--skip-eval-before-train)
   fi
fi
if [ -n "${DUMP_DETAILS}" ]; then
   ROLLOUT_ARGS+=(--dump-details "${DUMP_DETAILS}")
fi

PERF_ARGS=(
   # Training and rollout tensor parallelism are configured independently.
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EP_SIZE:-1}"
   --expert-tensor-parallel-size "${ETP_SIZE:-1}"

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   #--calculate-per-token-loss

   --micro-batch-size "${MICRO_BATCH_SIZE}"
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

if [ "${USE_DYNAMIC_BATCH_SIZE:-1}" = "1" ]; then
   PERF_ARGS+=(--use-dynamic-batch-size)
fi

GRPO_ARGS=(
   --advantage-estimator "${ADVANTAGE_ESTIMATOR:-grpo}"
   --kl-coef "${KL_COEF}"
   --kl-loss-coef "${KL_LOSS_COEF}"
   --kl-loss-type low_var_kl
   --entropy-coef "${ENTROPY_COEF:-0.00}"
   --eps-clip "${EPS_CLIP:-0.2}"
   --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
)
if is_truthy "${USE_KL_LOSS}"; then
   GRPO_ARGS+=(--use-kl-loss)
fi
if is_truthy "${NORMALIZE_ADVANTAGES}"; then
   GRPO_ARGS+=(--normalize-advantages)
fi

# Rollout correction defaults:
# - TIS (Truncated Importance Sampling): soft correction. It multiplies pg_loss
#   by a clipped importance weight exp(train_log_probs - rollout_log_probs).
# - MIS (Masked Importance Sampling): hard correction through tis_mode=mask.
#   It masks out tokens/sequences whose importance ratio leaves the trust range.
# - RS (Rejection Sampling): an additional hard rejection mask, independent of
#   the TIS weighting mode. The stable default uses Slime's vanilla token TIS
#   [0, 2], avoiding abrupt effective-batch changes. Custom MIS/RS remains
#   opt-in via ROLLOUT_CORRECTION_CONFIG.
if is_truthy "${USE_TIS:-1}"; then
   GRPO_ARGS+=(--use-tis)
   if [ -n "${ROLLOUT_CORRECTION_CONFIG:-}" ]; then
      GRPO_ARGS+=(
         --custom-config-path "${ROLLOUT_CORRECTION_CONFIG}"
         --custom-tis-function-path "${ROLLOUT_CORRECTION_FUNCTION:-examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp}"
      )
   fi
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY:-0.05}"
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# Colocated SGLang shares each GPU with residual trainer allocations during
# weight synchronization. Keep the static pool at 0.6 so KV/CUDA graph resume
# has enough headroom while trainer CUDA allocations are still being released.
if [ -z "${SGLANG_MEM_FRACTION_STATIC:-}" ]; then
   if is_truthy "${COLOCATE}"; then
      SGLANG_MEM_FRACTION_STATIC=0.8
   else
      SGLANG_MEM_FRACTION_STATIC="${GPU_MEMORY_UTILIZATION:-0.9}"
   fi
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   --sglang-router-request-timeout-secs "${SGLANG_ROUTER_REQUEST_TIMEOUT_SECS}"
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
   --sglang-context-length "${MAX_CONTEXT_LEN}"
   --sglang-enable-deterministic-inference
   --sglang-attention-backend triton
   --sglang-disable-custom-all-reduce
   --sglang-disable-piecewise-cuda-graph
   --router-policy "${ROUTER_POLICY}"
)
WANDB_ARGS=()
if [ "${USE_WANDB}" = "1" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-mode "${WANDB_MODE:-online}"
      --wandb-project "${WANDB_PROJECT:-FusedRL}"
      --wandb-group "${WANDB_GROUP:-${EXPERIMENT_NAME}}"
      --disable-wandb-random-suffix
   )
   # Resume an existing wandb run (same curves) instead of creating a new one.
   if [ -n "${WANDB_RUN_ID:-}" ]; then
      WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
      if is_truthy "${WANDB_SKIP_RESUME_FIRST_STEP:-1}"; then
         WANDB_ARGS+=(--wandb-skip-resume-first-step)
      fi
   fi
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --deterministic-mode
   --moe-token-dispatcher-type alltoall
   --train-env-vars '{"TMS_INIT_ENABLE_CPU_BACKUP":"1"}'
)
if is_truthy "${CHECK_WEIGHT_UPDATE_EQUAL:-1}"; then
   # One tensor-level equality check validates the transport at startup;
   # lightweight weight-version checks cover every subsequent update.
   MISC_ARGS+=(--check-weight-update-equal)
fi

if [ "${PRINT_ROLLOUT_TRAJECTORY:-1}" = "1" ]; then
   MISC_ARGS+=(--print-rollout-trajectory)
fi
if [ "${PRINT_TRAIN_METRICS_TABLE:-1}" = "1" ]; then
   MISC_ARGS+=(--print-train-metrics-table)
fi

export SCRIPT_DIR REPO_ROOT MEGATRON_LM_PATH HAS_NVLINK
export RUNS_ROOT RUN_ROOT SAVE_DIR LOG_ROOT EPISODE_LOG_DIR DUMP_DETAILS EVAL_CACHE_DIR PREPARED_PROMPT_DATA
export UPDATE_WEIGHT_DISK_DIR WANDB_DIR WANDB_CACHE_DIR HF_HOME TORCH_HOME TRITON_CACHE_DIR XDG_CACHE_HOME MCP_ENV_ROOT
export SLIME_MCP_ENV_ROOT="${MCP_ENV_ROOT}"
export SLIME_MCP_ENV_COPY_CONCURRENCY="${MCP_ENV_COPY_CONCURRENCY}"
export SLIME_MCP_WORKSPACE_SCOPE="${SLIME_MCP_WORKSPACE_SCOPE:-task}"
export SLIME_FULLY_ASYNC_ADAPTIVE_CONCURRENCY="${FULLY_ASYNC_ADAPTIVE_CONCURRENCY}"
export SLIME_FULLY_ASYNC_INITIAL_CONCURRENCY="${FULLY_ASYNC_INITIAL_GROUP_CONCURRENCY}"
export SLIME_FULLY_ASYNC_MAX_CONCURRENCY="${FULLY_ASYNC_MAX_GROUP_CONCURRENCY}"
export SLIME_FULLY_ASYNC_CONCURRENCY_STEP="${FULLY_ASYNC_CONCURRENCY_STEP}"
export SLIME_FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL="${FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL}"
export SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS="${SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS:-false}"
export SLIME_EPISODE_LOG_DIR="${EPISODE_LOG_DIR}"
export SLIME_FUSED_EVAL_TRAJECTORY_SAMPLE_RATE="${EVAL_TRAJECTORY_SAMPLE_RATE}"
export SLIME_FUSED_EVAL_DUMP_FAILURES="${EVAL_DUMP_FAILURES}"
export SLIME_FUSED_EVAL_USE_SGLANG_SESSION="${NATIVE_SGLANG_SESSION}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if is_truthy "${COLOCATE}"; then
   export NUM_GPUS="${NUM_GPUS:-${ACTOR_GPUS}}"
else
   export NUM_GPUS="${NUM_GPUS:-$((ACTOR_GPUS + ROLLOUT_GPUS))}"
fi
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${ACTOR_NUM_GPUS_PER_NODE}}"
if [ "${NUM_GPUS_PER_NODE}" -ne 8 ]; then
   echo "This launcher requires NUM_GPUS_PER_NODE=8; got ${NUM_GPUS_PER_NODE}." >&2
   exit 2
fi

export CUDA_HOME="/cm/shared/apps/cuda12.9"
CUDA_RUNTIME_LIB_DIR="${CUDA_HOME}/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"
GCC_HOME="${GCC_HOME:-/cm/shared/apps/gcc11/11.3.0}"
export PATH="${GCC_HOME}/bin:${PATH}"
export CC="${GCC_HOME}/bin/gcc"
export CXX="${GCC_HOME}/bin/g++"
export CUDAHOSTCXX="${GCC_HOME}/bin/g++"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export RAY_WARN_BLOCKING_GET_INSIDE_ASYNC="${RAY_WARN_BLOCKING_GET_INSIDE_ASYNC:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-10000000000}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export OPENROUTER_APP_NAME="${OPENROUTER_APP_NAME:-GRM}"
export SLIME_FUSED_REQUIRE_WEIGHT_VERSION="${SLIME_FUSED_REQUIRE_WEIGHT_VERSION:-1}"

# Fused-agent service knobs inherited from the rllm launcher. They are runtime
# env only; current slime code consumes the subset used by selected generators.
export RLLM_RETRIEVAL_MAX_WORDS="${RLLM_RETRIEVAL_MAX_WORDS:-1024}"
export FUSED_WEBQA_MIN_UNIQUE_SEARCHES="${FUSED_WEBQA_MIN_UNIQUE_SEARCHES:-1}"
# Summarize is a ~2s LLM call per search with a large retry budget; on slow
# trajectories it stacks up and blows the 180s rollout collection timeout,
# causing groups to be dropped. Default off and use raw retrieve docs instead.
export RLLM_RETRIEVAL_SUMMARIZE="${RLLM_RETRIEVAL_SUMMARIZE:-0}"
export RLLM_RETRIEVAL_RETRY_BUDGET="${RLLM_RETRIEVAL_RETRY_BUDGET:-8}"
export RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET="${RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET:-32}"
# LexRank fallback option/env name is kept for compatibility with older launch
# commands, but this launcher never enables LexRank-based retrieval summaries.
export RLLM_RETRIEVAL_LEXRANK_FALLBACK=0
export RLLM_RETRIEVAL_LEXRANK_MAX_WORDS="${RLLM_RETRIEVAL_LEXRANK_MAX_WORDS:-512}"
export RLLM_RETRIEVAL_LEXRANK_MAX_SENTENCES="${RLLM_RETRIEVAL_LEXRANK_MAX_SENTENCES:-32}"
export RLLM_RETRIEVAL_LEXRANK_MAX_INPUT_SENTENCES="${RLLM_RETRIEVAL_LEXRANK_MAX_INPUT_SENTENCES:-128}"
export RLLM_RETRIEVAL_LEXRANK_MULTIPROCESSING="${RLLM_RETRIEVAL_LEXRANK_MULTIPROCESSING:-1}"
export RLLM_RETRIEVAL_LEXRANK_WORKERS="${RLLM_RETRIEVAL_LEXRANK_WORKERS:-32}"
export DOCKER_HOST="${DOCKER_HOST:-tcp://10.2.152.50:2375}"
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.44}"
export SLIME_LOCAL_MCP_PROCESS_ISOLATION="${SLIME_LOCAL_MCP_PROCESS_ISOLATION:-true}"
export SLIME_LOCAL_MCP_PROCESS_WORKERS="${SLIME_LOCAL_MCP_PROCESS_WORKERS:-16}"
export SLIME_LOCAL_MCP_PROCESS_START_METHOD="${SLIME_LOCAL_MCP_PROCESS_START_METHOD:-forkserver}"
export SLIME_LOCAL_MCP_PROCESS_TIMEOUT="${SLIME_LOCAL_MCP_PROCESS_TIMEOUT:-120}"
export SLIME_LOCAL_MCP_LEASE_TIMEOUT="${SLIME_LOCAL_MCP_LEASE_TIMEOUT:-120}"
export SLIME_LOCAL_MCP_DESCRIBE_CACHE_SIZE="${SLIME_LOCAL_MCP_DESCRIBE_CACHE_SIZE:-4096}"
export SLIME_LOCAL_MCP_PROCESS_WARM_TIMEOUT="${SLIME_LOCAL_MCP_PROCESS_WARM_TIMEOUT:-60}"
export RLLM_MCP_MIN_NOFILE="${RLLM_MCP_MIN_NOFILE:-4096}"
export RLLM_MCP_FD_THROTTLE_THRESHOLD="${RLLM_MCP_FD_THROTTLE_THRESHOLD:-4096}"
export RLLM_MCP_INIT_TIMEOUT="${RLLM_MCP_INIT_TIMEOUT:-32}"
export RLLM_MCP_START_RETRIES="${RLLM_MCP_START_RETRIES:-8}"
export RLLM_MCP_START_WAIT_TIMEOUT="${RLLM_MCP_START_WAIT_TIMEOUT:-0}"
export RLLM_MCP_TOOL_TIMEOUT="${RLLM_MCP_TOOL_TIMEOUT:-8}"
export RLLM_MCP_MAX_ACTIVE_SERVERS="${RLLM_MCP_MAX_ACTIVE_SERVERS:-128}"
export RLLM_MCP_PREFILTER_WORKERS="${RLLM_MCP_PREFILTER_WORKERS:-8}"
export RLLM_MCP_DISABLE_STEP_PENALTY="${RLLM_MCP_DISABLE_STEP_PENALTY:-True}"
export FUSED_HARNESS="${FUSED_HARNESS:-gem}"
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
export SLIME_ROLLOUT_GROUP_TIMEOUT="${ROLLOUT_GROUP_TIMEOUT}"
export SLIME_SGLANG_ABORT_TIMEOUT_SECONDS=10
export SLIME_SGLANG_ABORT_HTTP_TIMEOUT_SECONDS=2
export SLIME_SGLANG_ABORT_RETRY_INTERVAL_SECONDS=0.2
export PER_STEP_MAX_TOKENS="${PER_STEP_MAX_TOKENS:-8192}"
export SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH="${MAX_TOOL_OUTPUT_LENGTH}"
export SLIME_FUSED_TERMINAL_LOG_STYLE="${TERMINAL_LOG_STYLE}"
export SLIME_FUSED_PROGRESS_LOGS="${SHOW_ROLLOUT_PROGRESS_LOGS}"
export SLIME_FUSED_TAIL_GUARD="${TAIL_GUARD}"
export SLIME_FUSED_TAIL_GUARD_TIME_GUARD="${TAIL_GUARD_TIME_GUARD}"
export SLIME_FUSED_TAIL_GUARD_TIME_MULTIPLIER="${TAIL_GUARD_TIME_MULTIPLIER}"
export SLIME_FUSED_TAIL_GUARD_TIME_SLACK_SECONDS="${TAIL_GUARD_TIME_SLACK_SECONDS}"
export SLIME_FUSED_TAIL_GUARD_MIN_COMPLETION_RATIO="${TAIL_GUARD_MIN_COMPLETION_RATIO}"
export CREDIT_ASSIGNMENT_ENABLE="${CREDIT_ASSIGNMENT_ENABLE}"
export CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR="${CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR}"
export CREDIT_ASSIGNMENT_THINK_PARSER_ERROR="${CREDIT_ASSIGNMENT_THINK_PARSER_ERROR}"
export CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY="${CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY}"
export CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS="${CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS}"
export CREDIT_ASSIGNMENT_NGRAM_REPETITION="${CREDIT_ASSIGNMENT_NGRAM_REPETITION}"
export CREDIT_ASSIGNMENT_NGRAM_REPETITION_N="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_N}"
export CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD}"
export CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS}"
export CREDIT_ASSIGNMENT_SEARCH_BYPASS="${CREDIT_ASSIGNMENT_SEARCH_BYPASS}"
export CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL="${CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL}"
export CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER="${CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER}"
export CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP="${CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP}"
export CREDIT_ASSIGNMENT_MAX_TURNS="${CREDIT_ASSIGNMENT_MAX_TURNS}"
export CREDIT_ASSIGNMENT_MAX_RESPONSE_LEN="${CREDIT_ASSIGNMENT_MAX_RESPONSE_LEN}"
export FUSED_FILTER_MIN_MEAN_STEPS="${FUSED_FILTER_MIN_MEAN_STEPS:-0}"
export FUSED_FILTER_MIN_MCP_MEAN_STEPS="${FUSED_FILTER_MIN_MCP_MEAN_STEPS:-0}"
export FUSED_FILTER_MAX_ABNORMAL_RATIO="${FUSED_FILTER_MAX_ABNORMAL_RATIO:-0}"
export FUSED_HORIZON_REWARD_MIN_MULTIPLIER="${FUSED_HORIZON_REWARD_MIN_MULTIPLIER}"
export FUSED_HORIZON_REWARD_GAMMA="${FUSED_HORIZON_REWARD_GAMMA}"
export FUSED_HORIZON_REWARD_STEP_WEIGHT="${FUSED_HORIZON_REWARD_STEP_WEIGHT}"
export FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT="${FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT}"
export FUSED_HORIZON_REWARD_TARGET_STEPS="${FUSED_HORIZON_REWARD_TARGET_STEPS}"
if [ -n "${FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS}" ]; then
   export FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS
fi
export SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER="${SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER:-4}"
export SLIME_TASK_FAMILY_ROI_PRIOR="${SLIME_TASK_FAMILY_ROI_PRIOR:-0.5}"
export SLIME_TASK_FAMILY_ROI_PRIOR_STRENGTH="${SLIME_TASK_FAMILY_ROI_PRIOR_STRENGTH:-4}"
export SLIME_SYNC_MIN_PENDING_GROUPS="${SLIME_SYNC_MIN_PENDING_GROUPS:-${SYNC_MIN_PENDING_GROUPS}}"

# RUNTIME_ENV_JSON contains service credentials and is passed verbatim to Ray.
set +x
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "CUDA_HOME", "LD_LIBRARY_PATH", "PATH", "CC", "CXX", "CUDAHOSTCXX",
    "HYDRA_FULL_ERROR", "NCCL_IB_DISABLE", "NCCL_TIMEOUT",
    "OPENROUTER_API_KEY", "OPENROUTER_SITE_URL", "OPENROUTER_APP_NAME",
    "SLIME_EPISODE_LOG_DIR", "SLIME_FUSED_EVAL_TRAJECTORY_SAMPLE_RATE",
    "SLIME_FUSED_EVAL_DUMP_FAILURES", "SLIME_FUSED_EVAL_USE_SGLANG_SESSION",
    "SLIME_FUSED_REQUIRE_WEIGHT_VERSION",
    "RAY_WARN_BLOCKING_GET_INSIDE_ASYNC", "TOKENIZERS_PARALLELISM",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "VLLM_ENGINE_ITERATION_TIMEOUT_S",
    "VLLM_WORKER_MULTIPROC_METHOD", "PYTORCH_CUDA_ALLOC_CONF",
    "RETRIEVAL_SERVER_URL", "TRAIN_RETRIEVAL_SERVER_URL", "EVAL_RETRIEVAL_SERVER_URL", "SERPER_PROXY_TOKEN", "SERPER_SEARCH_URL", "RLLM_RETRIEVAL_MODE", "RLLM_RETRIEVAL_MAX_WORDS",
    "RLLM_RETRIEVAL_CONCURRENCY", "RLLM_RETRIEVAL_CACHE_SIZE",
    "RETRIEVAL_MAX_RESULTS", "RLLM_RETRIEVAL_SUMMARIZE",
    "FUSED_WEBQA_MIN_UNIQUE_SEARCHES",
    "RLLM_RETRIEVAL_RETRY_BUDGET", "RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET",
    "RLLM_RETRIEVAL_LEXRANK_FALLBACK", "RLLM_RETRIEVAL_LEXRANK_MAX_WORDS",
    "RLLM_RETRIEVAL_LEXRANK_MAX_SENTENCES", "RLLM_RETRIEVAL_LEXRANK_MAX_INPUT_SENTENCES",
    "RLLM_RETRIEVAL_LEXRANK_MULTIPROCESSING", "RLLM_RETRIEVAL_LEXRANK_WORKERS",
    "DOCKER_HOST", "DOCKER_API_VERSION", "WANDB_API_KEY",
    "WANDB_DIR", "WANDB_CACHE_DIR", "HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME",
    "MCP_ENV_ROOT", "SLIME_MCP_ENV_ROOT", "SLIME_MCP_ENV_COPY_CONCURRENCY", "SLIME_MCP_WORKSPACE_SCOPE",
    "SLIME_LOCAL_MCP_PROCESS_ISOLATION", "SLIME_LOCAL_MCP_PROCESS_WORKERS", "SLIME_LOCAL_MCP_PROCESS_START_METHOD",
    "SLIME_LOCAL_MCP_PROCESS_TIMEOUT", "SLIME_LOCAL_MCP_LEASE_TIMEOUT", "SLIME_LOCAL_MCP_PROCESS_WARM_TIMEOUT",
    "SLIME_LOCAL_MCP_DESCRIBE_CACHE_SIZE",
    "SLIME_FULLY_ASYNC_ADAPTIVE_CONCURRENCY", "SLIME_FULLY_ASYNC_INITIAL_CONCURRENCY",
    "SLIME_FULLY_ASYNC_MAX_CONCURRENCY", "SLIME_FULLY_ASYNC_CONCURRENCY_STEP",
    "SLIME_FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL", "SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS",
    "RLLM_MCP_MIN_NOFILE", "RLLM_MCP_FD_THROTTLE_THRESHOLD", "RLLM_MCP_INIT_TIMEOUT",
    "RLLM_MCP_START_RETRIES", "RLLM_MCP_START_WAIT_TIMEOUT", "RLLM_MCP_TOOL_TIMEOUT",
    "RLLM_MCP_MAX_ACTIVE_SERVERS", "RLLM_MCP_PREFILTER_WORKERS", "RLLM_MCP_DISABLE_STEP_PENALTY",
    "FUSED_HARNESS", "FUSED_UNIFIED_SYSTEM_PROMPT", "FUSED_DISABLE_THINKING",
    "FUSED_DISCARD_HISTORICAL_THINKING",
    "FUSED_MAX_STEPS", "FUSED_MCP_MAX_STEPS", "FUSED_MCP_MAX_TOOL_CALLS_PER_TURN",
    "FUSED_WEB_SEARCH_MAX_STEPS", "FUSED_CLI_MAX_STEPS", "FUSED_TRAJECTORY_TIMEOUT",
    "FUSED_EVAL_TRAJECTORY_TIMEOUT", "SLIME_ROLLOUT_GROUP_TIMEOUT", "SLIME_EVAL_ROLLOUT_GROUP_TIMEOUT",
    "PER_STEP_MAX_TOKENS", "SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH",
    "SLIME_FUSED_TERMINAL_LOG_STYLE", "SLIME_FUSED_PROGRESS_LOGS",
    "SLIME_FUSED_TAIL_GUARD", "SLIME_FUSED_TAIL_GUARD_TIME_GUARD",
    "SLIME_FUSED_TAIL_GUARD_TIME_MULTIPLIER", "SLIME_FUSED_TAIL_GUARD_TIME_SLACK_SECONDS",
    "SLIME_FUSED_TAIL_GUARD_MIN_COMPLETION_RATIO",
    "CREDIT_ASSIGNMENT_ENABLE", "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR", "CREDIT_ASSIGNMENT_THINK_PARSER_ERROR",
    "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY", "CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS",
    "CREDIT_ASSIGNMENT_NGRAM_REPETITION", "CREDIT_ASSIGNMENT_NGRAM_REPETITION_N",
    "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD", "CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS",
    "CREDIT_ASSIGNMENT_SEARCH_BYPASS", "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL",
    "CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER", "CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP",
    "CREDIT_ASSIGNMENT_MAX_TURNS", "CREDIT_ASSIGNMENT_MAX_RESPONSE_LEN",
    "FUSED_FILTER_MIN_MEAN_STEPS", "FUSED_FILTER_MIN_MCP_MEAN_STEPS",
    "FUSED_FILTER_MAX_ABNORMAL_RATIO",
    "FUSED_HORIZON_REWARD_MIN_MULTIPLIER", "FUSED_HORIZON_REWARD_GAMMA",
    "FUSED_HORIZON_REWARD_STEP_WEIGHT", "FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT",
    "FUSED_HORIZON_REWARD_TARGET_STEPS", "FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS",
    "SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER", "SLIME_TASK_FAMILY_ROI_PRIOR",
    "SLIME_TASK_FAMILY_ROI_PRIOR_STRENGTH", "SLIME_SYNC_MIN_PENDING_GROUPS",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["PYTHONPATH"] = f"{os.environ['MEGATRON_LM_PATH']}:{os.environ['REPO_ROOT']}:{os.environ['SCRIPT_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_ALGO"] = "Ring"
env["NCCL_NVLS_ENABLE"] = os.environ["HAS_NVLINK"]
env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
print(json.dumps({"env_vars": env}))
PY
)

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Run root: ${RUN_ROOT}"
echo "Model: ${MODEL_DIR}"
echo "Ref: ${REF_LOAD}"
echo "Prompt data: ${PROMPT_DATA_FOR_SLIME}"
echo "Train rows: ${TRAIN_NUM_ROWS}; rollout_batch_size=${ROLLOUT_BATCH_SIZE}; over_sampling_batch_size=${OVER_SAMPLING_BATCH_SIZE}; samples_per_prompt=${N_SAMPLES_PER_PROMPT}; num_rollout=${NUM_ROLLOUT}"
echo "Save dir: ${SAVE_DIR}"
echo "Log root: ${LOG_ROOT}"
echo "Episode dump root: ${EPISODE_LOG_DIR} (train/ and evals/)"
echo "Dump details dir: ${DUMP_DETAILS:-<disabled>}"
echo "W&B enabled: ${USE_WANDB}"
echo "Custom generate: ${CUSTOM_GENERATE_FUNCTION_PATH:-<stock slime rollout>}"
echo "Custom reward post-process: ${CUSTOM_REWARD_POST_PROCESS_PATH:-<vanilla>}"
echo "Rollout function: ${ROLLOUT_FUNCTION_PATH}"
echo "Actor GPUs: ${ACTOR_GPUS}, actor TP=${TP_SIZE}, CP=${CP_SIZE}, PP=${PP_SIZE}, rollout GPUs: ${ROLLOUT_GPUS}, rollout TP=${ROLLOUT_NUM_GPUS_PER_ENGINE}, rollout engines=${ROLLOUT_ENGINE_COUNT}, colocate=${COLOCATE}, offload_train=${OFFLOAD_TRAIN}, ray GPUs=${NUM_GPUS}"
echo "Training token budgets: max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}, log_probs_max_tokens_per_gpu=${LOG_PROBS_MAX_TOKENS_PER_GPU}, log_probs_chunk_size=${LOG_PROBS_CHUNK_SIZE}, max_context_len=${MAX_CONTEXT_LEN}"
echo "YaRN: enable=${ENABLE_YARN}, factor=${YARN_FACTOR}, original_max_position_embeddings=${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}"
echo "SGLang: mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}, server_concurrency=${SGLANG_SERVER_CONCURRENCY}, max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS}"
echo "SGLang abort: deadline=${SLIME_SGLANG_ABORT_TIMEOUT_SECONDS}s, http_timeout=${SLIME_SGLANG_ABORT_HTTP_TIMEOUT_SECONDS}s, retry_interval=${SLIME_SGLANG_ABORT_RETRY_INTERVAL_SECONDS}s"
echo "Sync rollout admission: min_pending_groups=${SLIME_SYNC_MIN_PENDING_GROUPS}, max_pending_groups=${OVER_SAMPLING_BATCH_SIZE}"
echo "Fused controls: harness=${FUSED_HARNESS}, unified_system_prompt=${UNIFIED_SYSTEM_PROMPT}, disable_thinking=${DISABLE_THINKING}, discard_historical_thinking=${DISCARD_HISTORICAL_THINKING}, max_steps=${FUSED_MAX_STEPS}, mcp_max_steps=${FUSED_MCP_MAX_STEPS}, mcp_max_tool_calls_per_turn=${FUSED_MCP_MAX_TOOL_CALLS_PER_TURN}, web_search_max_steps=${FUSED_WEB_SEARCH_MAX_STEPS}, cli_max_steps=${CLI_MAX_STEPS}, per_step_max_tokens=${PER_STEP_MAX_TOKENS}, partial_rollout=${PARTIAL_ROLLOUT}, terminal_log_style=${TERMINAL_LOG_STYLE}, show_rollout_progress_logs=${SHOW_ROLLOUT_PROGRESS_LOGS}"
echo "Rollout timeouts: trajectory=${FUSED_TRAJECTORY_TIMEOUT}s, group=${SLIME_ROLLOUT_GROUP_TIMEOUT}s, eval_trajectory=${FUSED_EVAL_TRAJECTORY_TIMEOUT}s"
echo "Training batches: micro_batch=${MICRO_BATCH_SIZE}, num_steps_per_rollout=${NUM_STEPS_PER_ROLLOUT}, update_weights_interval=${UPDATE_WEIGHTS_INTERVAL}, rollout_temperature=${TEMPERATURE}"
echo "Retrieval: train_backend=${TRAIN_RETRIEVAL_BACKEND}, train_url=${TRAIN_RETRIEVAL_SERVER_URL}, eval_backend=${EVAL_RETRIEVAL_BACKEND}, eval_url=${EVAL_RETRIEVAL_SERVER_URL}, mode=${RLLM_RETRIEVAL_MODE}, concurrency=${RLLM_RETRIEVAL_CONCURRENCY}, cache_size=${RLLM_RETRIEVAL_CACHE_SIZE}, max_words=${RLLM_RETRIEVAL_MAX_WORDS}, max_results=${RETRIEVAL_MAX_RESULTS}, retry=${RETRIEVAL_RETRY_BUDGET}, summary_retry=${RETRIEVAL_SUMMARY_RETRY_BUDGET}, lexrank_fallback=${RETRIEVAL_LEXRANK_FALLBACK}"
echo "Dynamic filter: enable=${ENABLE_DYNAMIC_SAMPLING_FILTER}, path=${DYNAMIC_SAMPLING_FILTER_PATH:-<none>}, strict=true, relax_after_groups=0; webqa_min_unique_searches=${FUSED_WEBQA_MIN_UNIQUE_SEARCHES}"
echo "Eval: interval=${EVAL_INTERVAL:-<disabled>}, benchmarks=${EVAL_INCLUDE_BENCHMARKS}, config=${EVAL_CONFIG:-<none>}, prompt_data=${EVAL_PROMPT_DATA[*]:-<none>}, n=${N_SAMPLES_PER_EVAL_PROMPT}, temperature=${EVAL_TEMPERATURE}, top_p=${EVAL_TOP_P}, top_k=${EVAL_TOP_K}, max_prompt_len=${EVAL_MAX_PROMPT_LEN}, max_response_len=${EVAL_MAX_RESPONSE_LEN}, max_context_len=${EVAL_MAX_CONTEXT_LEN}, val_before_train=${VAL_BEFORE_TRAIN}"
echo "Eval scheduling: inflight=${EVAL_INITIAL_INFLIGHT_TASKS}-${EVAL_MAX_INFLIGHT_TASKS}, adaptive=${EVAL_ADAPTIVE_CONCURRENCY}, mix_datasets=${EVAL_MIX_DATASETS}, termination_retries=${EVAL_TERMINATION_RETRY_TIMES}, trajectory_sample_rate=${EVAL_TRAJECTORY_SAMPLE_RATE}, dump_failures=${EVAL_DUMP_FAILURES}, native_session=${NATIVE_SGLANG_SESSION}"
echo "OpenRouter GRM: train=${ENABLE_USE_GRM_TRAIN}, train_model=${TRAIN_GRM_MODEL}, evals=${ENABLE_USE_GRM_EVALS}, eval_model=${EVAL_GRM_MODEL}, mode=${GRM_MODE}, concurrency=${GRM_CONCURRENCY}, max_connections=${GRM_MAX_CONNECTIONS}, timeout=${GRM_TIMEOUT}, retries=${GRM_MAX_RETRIES}, max_input_tokens=${GRM_MAX_INPUT_TOKENS}, max_new_tokens=${GRM_MAX_NEW_TOKENS}, custom_rm=${GRM_CUSTOM_RM_PATH}"
echo "GRPO: advantage_estimator=${ADVANTAGE_ESTIMATOR:-grpo}, normalize_advantages=${NORMALIZE_ADVANTAGES}, kl_coef=${KL_COEF}, lr=${LR}, eps_clip=${EPS_CLIP:-0.2}, eps_clip_high=${EPS_CLIP_HIGH:-0.28}"
echo "Buffer filter: enable_quota_bucket_sampling=${ENABLE_QUOTA_BUCKET_SAMPLING:-0}, path=${BUFFER_FILTER_PATH:-${ENABLE_QUOTA_BUCKET_SAMPLING:+slime.rollout.filter_hub.buffer_filters.quota_bucket_by_steps}}"
echo "Fused filter thresholds: min_mean_steps=${FUSED_FILTER_MIN_MEAN_STEPS}, min_mcp_mean_steps=${FUSED_FILTER_MIN_MCP_MEAN_STEPS}, max_abnormal_ratio=${FUSED_FILTER_MAX_ABNORMAL_RATIO}"
echo "Horizon reward shaping: enable=${HORIZON_REWARD_SHAPING}, min_multiplier=${FUSED_HORIZON_REWARD_MIN_MULTIPLIER}, gamma=${FUSED_HORIZON_REWARD_GAMMA}, step_weight=${FUSED_HORIZON_REWARD_STEP_WEIGHT}, tool_call_weight=${FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT}, target_steps=${FUSED_HORIZON_REWARD_TARGET_STEPS}, target_tool_calls=${FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS:-target_steps-1}"
echo "Post-filter rollout task family quotas: ${ROLLOUT_TASK_FAMILY_QUOTAS:-<none>}; candidate_multiplier=${SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER}"
echo "Tail guard: enable=${TAIL_GUARD}, time_guard=${TAIL_GUARD_TIME_GUARD}, time_multiplier=${TAIL_GUARD_TIME_MULTIPLIER}, time_slack=${TAIL_GUARD_TIME_SLACK_SECONDS}, min_completion_ratio=${TAIL_GUARD_MIN_COMPLETION_RATIO}"
echo "Credit assignment: enable=${CREDIT_ASSIGNMENT_ENABLE}, tool_parser_error=${CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR}, repeated_search_query=${CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY}, too_many_tool_calls=${CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS}, ngram_repetition=${CREDIT_ASSIGNMENT_NGRAM_REPETITION}(n=${CREDIT_ASSIGNMENT_NGRAM_REPETITION_N}, threshold=${CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD}, min_tokens=${CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS}), search_bypass=${CREDIT_ASSIGNMENT_SEARCH_BYPASS}, direct_submit_without_tool=${CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL}, mixed_tool_and_answer=${CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER}, tail_guard_early_stop=${CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP}"

CREDIT_ASSIGNMENT_ARGS=()
if is_truthy "${CREDIT_ASSIGNMENT_ENABLE}"; then
   CREDIT_ASSIGNMENT_ARGS+=(--credit-assignment-enable)
fi
if is_truthy "${CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR}"; then
   CREDIT_ASSIGNMENT_ARGS+=(--credit-assignment-tool-parser-error)
fi
if is_truthy "${CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY}"; then
   CREDIT_ASSIGNMENT_ARGS+=(--credit-assignment-repeated-search-query)
fi
if is_truthy "${CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS}"; then
   CREDIT_ASSIGNMENT_ARGS+=(--credit-assignment-too-many-tool-calls)
fi
if is_truthy "${CREDIT_ASSIGNMENT_NGRAM_REPETITION}"; then
   CREDIT_ASSIGNMENT_ARGS+=(--credit-assignment-ngram-repetition)
fi
if is_truthy "${CREDIT_ASSIGNMENT_SEARCH_BYPASS}"; then
   CREDIT_ASSIGNMENT_ARGS+=(--credit-assignment-search-bypass)
fi
if is_truthy "${CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER}"; then
   CREDIT_ASSIGNMENT_ARGS+=(--credit-assignment-mixed-tool-and-answer)
fi

RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"

if ray job list --address="${RAY_DASHBOARD_ADDRESS}" >/dev/null 2>&1; then
   echo "Reusing existing Ray head at ${RAY_DASHBOARD_ADDRESS}"
else
   ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --num-cpus "${RAY_NUM_CPUS}" --disable-usage-stats
fi

cd "${REPO_ROOT}"

RAY_JOB_SUBMIT_ARGS=()
if [ "${RAY_JOB_WAIT:-0}" != "1" ]; then
   RAY_JOB_SUBMIT_ARGS+=(--no-wait)
fi

SAFE_EXPERIMENT_NAME="$(printf '%s' "${EXPERIMENT_NAME}" | tr -c '[:alnum:]_' '_' | cut -c1-120)"
RAY_SUBMISSION_ID="${RAY_SUBMISSION_ID:-raysubmit_${SAFE_EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)}"
echo "Ray submission id: ${RAY_SUBMISSION_ID}"

CLUSTER_ARGS=(
   --actor-num-nodes "${ACTOR_NUM_NODES}"
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
   --num-gpus-per-node "${NUM_GPUS_PER_NODE}"
   --rollout-num-gpus "${ROLLOUT_GPUS}"
   --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}"
   --verify-rollout-weight-versions
)
if is_truthy "${COLOCATE}"; then
   CLUSTER_ARGS+=(--colocate)
fi
if is_truthy "${RELEASE_TRAIN}"; then
   CLUSTER_ARGS+=(
      --release-train
      --update-weight-mode full
      --update-weight-transport disk
      --update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR}"
   )
fi
if is_truthy "${OFFLOAD_TRAIN}"; then
   CLUSTER_ARGS+=(--offload-train)
else
   CLUSTER_ARGS+=(--no-offload-train)
fi

start_managed_retrieval_backend
start_managed_eval_serper() {
   if [ "${EVAL_RETRIEVAL_BACKEND}" != "serper" ] || [ "${EVAL_RETRIEVAL_SERVER_URL_EXPLICIT}" = "true" ]; then
      return
   fi
   python3 "${REPO_ROOT}/examples/search-r1/serper_search_server.py" \
      --host "${SERPER_SERVER_HOST}" --port "${SERPER_SERVER_PORT}" \
      >"${LOG_ROOT}/eval_serper_search_server.log" 2>&1 &
   EVAL_SERPER_SERVICE_PID=$!
   trap 'if [ -n "${EVAL_SERPER_SERVICE_PID}" ]; then kill "${EVAL_SERPER_SERVICE_PID}" 2>/dev/null || true; fi' EXIT
   python3 - "${EVAL_RETRIEVAL_SERVER_URL}" "${EVAL_SERPER_SERVICE_PID}" <<'PY'
import json, os, sys, time, urllib.request
url = sys.argv[1].rstrip('/') + '/health'
pid = int(sys.argv[2])
for _ in range(100):
    if not os.path.exists(f'/proc/{pid}'):
        break
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if json.load(response).get('status') == 'ok':
                raise SystemExit(0)
    except Exception:
        time.sleep(0.1)
raise SystemExit(f'Managed eval Serper service failed to start at {url}')
PY
}
start_managed_eval_serper
ray job submit --address="${RAY_DASHBOARD_ADDRESS}" \
   --submission-id="${RAY_SUBMISSION_ID}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   "${RAY_JOB_SUBMIT_ARGS[@]}" \
   -- python3 -u train.py \
   "${CLUSTER_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${CREDIT_ASSIGNMENT_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${WANDB_ARGS[@]}"
set -x

if [ "${RAY_JOB_WAIT:-0}" != "1" ] && [ "${RAY_JOB_FOLLOW_LOGS:-1}" = "1" ]; then
   echo "Following Ray job logs for ${RAY_SUBMISSION_ID}"
   ray job logs --address="${RAY_DASHBOARD_ADDRESS}" --follow "${RAY_SUBMISSION_ID}"
fi
