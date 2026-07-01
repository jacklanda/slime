#!/bin/bash
# Synchronous fused-agent training launcher for current slime.
#
# This keeps the launch style of experiments/run-qwen3.5-9B-fully_async.sh:
# array-based args, ray job submit, and a runtime PYTHONPATH rooted at this
# slime checkout. The fused-agent workload knobs mirror the rllm fused launcher
# where current slime exposes equivalent CLI arguments.

set -ex

export PYTHONUNBUFFERED=1

usage() {
   cat <<'EOF'
Usage:
  bash experiments/train_fused_agent_sync.sh [options]

Options:
  --harness NAME                         Fused prompt harness: bare, cot, react, gem, unified_gem.
  --model PATH                           HF model path.
  --disable-thinking BOOL                Stored in env for compatible fused code.
  --mcp-disable-step-penalty BOOL        MCP verifier step-penalty env.
  --unified-system-prompt                Select unified_gem harness unless --harness is set later.
  --no-unified-system-prompt             Select gem harness unless --harness is set later.
  --fully-async / --no-fully-async       Select fully-async rollout function. Default: disabled.
  --partial-rollout / --no-partial-rollout
                                         Recycle partial rollouts during abort/sync.
  --terminal-log-style STYLE             progress, rollouts, or both. Stored in env for compatible fused code.
  --accepted-group-update-min-groups N   Maps to ROLLOUT_BATCH_SIZE by default.
  --accepted-group-update-max-groups N   Stored in env for compatible fused code.
  --micro-batch-size N                   Training micro-batch size.
  --update-weights-interval N            Rollout weight update interval. Default: 1.
  --async-mini-batch-size N              Deprecated alias for --micro-batch-size.
  --async-trigger-parameter-sync-step N  Deprecated alias for --update-weights-interval.
  --async-fwd-bwd-group-size N           Deprecated no-op compatibility option.
  --async-staleness-threshold X          Deprecated no-op compatibility option.
  --retrieval-mode MODE                  Retrieval mode env.
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
                                         Train only the parser-error turn for parser failures. Default: true.
  --credit-assignment-repeated-search-query BOOL
                                         Train only the repeated-query turn. Default: true.
  --credit-assignment-too-many-tool-calls BOOL
                                         Train only the excessive-tool-call turn. Default: true.
  --credit-assignment-search-bypass BOOL Keep search-bypass masks but force reward to 0. Default: true.
  --credit-assignment-direct-submit-without-tool BOOL
                                         Penalize direct finish/boxed answer before any non-finish tool. Default: true.
  --credit-assignment-mixed-tool-and-answer BOOL
                                         Penalize turns that contain both a non-finish tool call and boxed/submit answer. Default: true.
  --credit-assignment-tail-guard-early-stop BOOL
                                         Stored in env for compatible fused code.
  --horizon-reward-shaping BOOL         Enable bounded horizon penalty reward shaping. Default: true.
  --horizon-reward-min-multiplier X      Correct low-horizon reward multiplier floor. Default: 0.2.
  --horizon-reward-gamma X               Horizon progress exponent. Default: 1.0.
  --horizon-reward-step-weight X         Step progress weight. Default: 0.7.
  --horizon-reward-tool-call-weight X    Tool-call progress weight. Default: 0.3.
  --horizon-reward-target-steps X        Target steps for no horizon penalty. Default: 8.
  --horizon-reward-target-tool-calls X   Target tool calls for no horizon penalty. Default: target_steps - 1.
  --enable-dynamic-sampling-filter BOOL  Enable DAPO-style non-zero reward variance dynamic filtering. Default: true.
  --enable_use_grm_evals BOOL            Use OpenRouter GRM before rule-based fallback for interval eval scoring. Default: false.
  --grm-model NAME                       OpenRouter judge model. Default: deepseek/deepseek-v4-flash.
  --grm-concurrency N                    Max concurrent GRM requests. Default: 128.
  --grm-timeout SECONDS                  GRM request timeout. Default: 60.
  --grm-max-retries N                    GRM retry attempts. Default: 3.
  --grm-max-trajectory-chars N           Trajectory chars sent to GRM. Default: 24000.
  --max-steps N                          Fused agent max steps.
  --mcp-max-steps N                      Fused MCP max steps. Default: 16.
  --web-search-max-steps N               Fused web-search max steps. Default: 4.
  --cli-max-steps N                      CLI fused agent max steps env.
  --trajectory-timeout N                 Fused trajectory timeout env.
  --eval-trajectory-timeout N            Fused eval trajectory timeout env.
  --eval-interval N                      Run interval eval every N rollout steps.
  --eval-config PATH                     Structured slime eval dataset config.
  --eval-prompt-data NAME PATH [...]     Legacy eval dataset name/path pairs.
  --eval-max-response-len N              Eval-only max generated tokens. Default: 16384.
  --eval-max-prompt-len N                Eval-only max prompt tokens. Default: 23616.
  --eval-max-context-len N               Eval-only context length. Default: prompt + response.
  --val_before_train BOOL                Run one eval before training starts. Default: true.
  --n-samples-per-eval-prompt N          Eval samples per prompt. Default: 1.
  --offload-train BOOL                   Offload trainer model between rollout/train phases. Default: true.
  --max-tool-output-length N             Fused max tool output length env.
  --sglang-server-concurrency N          Max concurrent requests per SGLang server. Default: 64.
  --sglang-max-running-requests N        SGLang max running requests. Default: 256.
  --colocate / --no-colocate             Share trainer and rollout GPUs with offload. Default: enabled.
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

FULLY_ASYNC="${FULLY_ASYNC:-false}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-false}"
TERMINAL_LOG_STYLE="${TERMINAL_LOG_STYLE:-both}"
# Default to non-colocate: train and rollout live on separate GPUs so training
# never runs the per-step torch_memory_saver offload/pause path. That pause path
# (cudaError 1 "invalid argument" in torch_memory_saver.cpp func=pause) crashes
# after ~20 offload cycles under colocate + enable_cpu_backup and has no upstream
# fix (already on the latest torch_memory_saver; see slime issues #1786/#71).
COLOCATE="${COLOCATE:-false}"
UNIFIED_SYSTEM_PROMPT="${UNIFIED_SYSTEM_PROMPT:-False}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"
ACCEPTED_GROUP_UPDATE_MIN_GROUPS="${ACCEPTED_GROUP_UPDATE_MIN_GROUPS:-16}"
ACCEPTED_GROUP_UPDATE_MAX_GROUPS="${ACCEPTED_GROUP_UPDATE_MAX_GROUPS:-${ACCEPTED_GROUP_UPDATE_MIN_GROUPS}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${ASYNC_MINI_BATCH_SIZE:-16}}"
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-${ASYNC_TRIGGER_PARAMETER_SYNC_STEP:-1}}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
TAIL_GUARD="${TAIL_GUARD:-False}"
TAIL_GUARD_TIME_GUARD="${TAIL_GUARD_TIME_GUARD:-True}"
TAIL_GUARD_TIME_MULTIPLIER="${TAIL_GUARD_TIME_MULTIPLIER:-1.05}"
TAIL_GUARD_TIME_SLACK_SECONDS="${TAIL_GUARD_TIME_SLACK_SECONDS:-16}"
TAIL_GUARD_MIN_COMPLETION_RATIO="${TAIL_GUARD_MIN_COMPLETION_RATIO:-0.60}"
CREDIT_ASSIGNMENT_ENABLE="${CREDIT_ASSIGNMENT_ENABLE:-True}"
CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR="${CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR:-True}"
CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY="${CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY:-True}"
CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS="${CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS:-True}"
CREDIT_ASSIGNMENT_NGRAM_REPETITION="${CREDIT_ASSIGNMENT_NGRAM_REPETITION:-True}"
CREDIT_ASSIGNMENT_NGRAM_REPETITION_N="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_N:-8}"
CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD:-0.5}"
CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS="${CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS:-160}"
CREDIT_ASSIGNMENT_SEARCH_BYPASS="${CREDIT_ASSIGNMENT_SEARCH_BYPASS:-True}"
CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL="${CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL:-True}"
CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER="${CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER:-True}"
CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP="${CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP:-False}"
HORIZON_REWARD_SHAPING="${HORIZON_REWARD_SHAPING:-false}"
FUSED_HORIZON_REWARD_MIN_MULTIPLIER="${FUSED_HORIZON_REWARD_MIN_MULTIPLIER:-0.2}"
FUSED_HORIZON_REWARD_GAMMA="${FUSED_HORIZON_REWARD_GAMMA:-1.0}"
FUSED_HORIZON_REWARD_STEP_WEIGHT="${FUSED_HORIZON_REWARD_STEP_WEIGHT:-0.7}"
FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT="${FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT:-0.3}"
FUSED_HORIZON_REWARD_TARGET_STEPS="${FUSED_HORIZON_REWARD_TARGET_STEPS:-8}"
FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS="${FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS:-}"
MAX_STEPS="${MAX_STEPS:-96}"
MCP_MAX_STEPS="${MCP_MAX_STEPS:-96}"
WEB_SEARCH_MAX_STEPS="${WEB_SEARCH_MAX_STEPS:-96}"
CLI_MAX_STEPS="${CLI_MAX_STEPS:-96}"
TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT:-3600}"
EVAL_TRAJECTORY_TIMEOUT="${EVAL_TRAJECTORY_TIMEOUT:-3600}"
MAX_TOOL_OUTPUT_LENGTH="${MAX_TOOL_OUTPUT_LENGTH:-4096}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-256}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-512}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_CONFIG="${EVAL_CONFIG:-experiments/eval_fused_agent_benchmarks.yaml}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-23616}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-${val_before_train:-true}}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_PROMPT_DATA=()
# Offload only makes sense under colocate (train/rollout share GPUs and take
# turns via torch_memory_saver). Under non-colocate they own separate GPUs, so
# default offload off — this is the whole point of the split: skip the crashing
# torch_memory_saver.pause() path entirely.
OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-${offload_train:-${COLOCATE}}}"
ENABLE_USE_GRM_EVALS="${ENABLE_USE_GRM_EVALS:-${enable_use_grm_evals:-true}}"
GRM_CUSTOM_RM_PATH="${GRM_CUSTOM_RM_PATH:-slime.rollout.rm_hub.openrouter_grm.reward_func}"
GRM_MODEL="${GRM_MODEL:-deepseek/deepseek-v4-flash}"
GRM_CONCURRENCY="${GRM_CONCURRENCY:-512}"
GRM_MAX_CONNECTIONS="${GRM_MAX_CONNECTIONS:-128}"
GRM_TIMEOUT="${GRM_TIMEOUT:-60}"
GRM_MAX_RETRIES="${GRM_MAX_RETRIES:-4}"
GRM_RETRY_BASE_DELAY="${GRM_RETRY_BASE_DELAY:-1}"
GRM_RETRY_MAX_DELAY="${GRM_RETRY_MAX_DELAY:-16.0}"
GRM_MAX_TRAJECTORY_CHARS="${GRM_MAX_TRAJECTORY_CHARS:-30000}"
GRM_MAX_TOKENS="${GRM_MAX_TOKENS:-128}"
GRM_TEMPERATURE="${GRM_TEMPERATURE:-0.6}"
GRM_FAILURE_REWARD="${GRM_FAILURE_REWARD:-0.0}"
harness_explicit=false

while [ "$#" -gt 0 ]; do
   case "$1" in
      --harness) FUSED_HARNESS="${2:?Missing value for --harness}"; harness_explicit=true; shift 2 ;;
      --model) MODEL_DIR="${2:?Missing value for --model}"; shift 2 ;;
      --disable-thinking) DISABLE_THINKING="${2:?Missing value for --disable-thinking}"; shift 2 ;;
      --mcp-disable-step-penalty) RLLM_MCP_DISABLE_STEP_PENALTY="${2:?Missing value for --mcp-disable-step-penalty}"; shift 2 ;;
      --unified-system-prompt) UNIFIED_SYSTEM_PROMPT=True; if [ "${harness_explicit}" = "false" ]; then FUSED_HARNESS=unified_gem; fi; shift ;;
      --no-unified-system-prompt) UNIFIED_SYSTEM_PROMPT=False; if [ "${harness_explicit}" = "false" ]; then FUSED_HARNESS=gem; fi; shift ;;
      --fully-async) FULLY_ASYNC=true; shift ;;
      --no-fully-async) FULLY_ASYNC=false; shift ;;
      --partial-rollout) PARTIAL_ROLLOUT=true; shift ;;
      --no-partial-rollout) PARTIAL_ROLLOUT=false; shift ;;
      --terminal-log-style) TERMINAL_LOG_STYLE="${2:?Missing value for --terminal-log-style}"; shift 2 ;;
      --accepted-group-update-min-groups) ACCEPTED_GROUP_UPDATE_MIN_GROUPS="${2:?Missing value for --accepted-group-update-min-groups}"; shift 2 ;;
      --accepted-group-update-max-groups) ACCEPTED_GROUP_UPDATE_MAX_GROUPS="${2:?Missing value for --accepted-group-update-max-groups}"; shift 2 ;;
      --micro-batch-size) MICRO_BATCH_SIZE="${2:?Missing value for --micro-batch-size}"; shift 2 ;;
      --update-weights-interval) UPDATE_WEIGHTS_INTERVAL="${2:?Missing value for --update-weights-interval}"; shift 2 ;;
      --async-mini-batch-size) MICRO_BATCH_SIZE="${2:?Missing value for --async-mini-batch-size}"; shift 2 ;;
      --async-trigger-parameter-sync-step) UPDATE_WEIGHTS_INTERVAL="${2:?Missing value for --async-trigger-parameter-sync-step}"; shift 2 ;;
      --async-fwd-bwd-group-size) : "${2:?Missing value for --async-fwd-bwd-group-size}"; shift 2 ;;
      --async-staleness-threshold) : "${2:?Missing value for --async-staleness-threshold}"; shift 2 ;;
      --retrieval-mode) RLLM_RETRIEVAL_MODE="${2:?Missing value for --retrieval-mode}"; shift 2 ;;
      --retrieval-max-words) RLLM_RETRIEVAL_MAX_WORDS="${2:?Missing value for --retrieval-max-words}"; shift 2 ;;
      --retrieval-retry-budget) RLLM_RETRIEVAL_RETRY_BUDGET="${2:?Missing value for --retrieval-retry-budget}"; shift 2 ;;
      --retrieval-summary-retry-budget) RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET="${2:?Missing value for --retrieval-summary-retry-budget}"; shift 2 ;;
      --retrieval-lexrank-fallback) RLLM_RETRIEVAL_LEXRANK_FALLBACK="${2:?Missing value for --retrieval-lexrank-fallback}"; shift 2 ;;
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
      --horizon-reward-shaping) HORIZON_REWARD_SHAPING="${2:?Missing value for --horizon-reward-shaping}"; shift 2 ;;
      --horizon-reward-min-multiplier) FUSED_HORIZON_REWARD_MIN_MULTIPLIER="${2:?Missing value for --horizon-reward-min-multiplier}"; shift 2 ;;
      --horizon-reward-gamma) FUSED_HORIZON_REWARD_GAMMA="${2:?Missing value for --horizon-reward-gamma}"; shift 2 ;;
      --horizon-reward-step-weight) FUSED_HORIZON_REWARD_STEP_WEIGHT="${2:?Missing value for --horizon-reward-step-weight}"; shift 2 ;;
      --horizon-reward-tool-call-weight) FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT="${2:?Missing value for --horizon-reward-tool-call-weight}"; shift 2 ;;
      --horizon-reward-target-steps) FUSED_HORIZON_REWARD_TARGET_STEPS="${2:?Missing value for --horizon-reward-target-steps}"; shift 2 ;;
      --horizon-reward-target-tool-calls) FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS="${2:?Missing value for --horizon-reward-target-tool-calls}"; shift 2 ;;
      --enable-dynamic-sampling-filter) ENABLE_DYNAMIC_SAMPLING_FILTER="${2:?Missing value for --enable-dynamic-sampling-filter}"; shift 2 ;;
      --enable_use_grm_evals|--enable-use-grm-evals) ENABLE_USE_GRM_EVALS="${2:?Missing value for --enable_use_grm_evals}"; shift 2 ;;
      --grm-model) GRM_MODEL="${2:?Missing value for --grm-model}"; shift 2 ;;
      --grm-concurrency) GRM_CONCURRENCY="${2:?Missing value for --grm-concurrency}"; shift 2 ;;
      --grm-max-connections) GRM_MAX_CONNECTIONS="${2:?Missing value for --grm-max-connections}"; shift 2 ;;
      --grm-timeout) GRM_TIMEOUT="${2:?Missing value for --grm-timeout}"; shift 2 ;;
      --grm-max-retries) GRM_MAX_RETRIES="${2:?Missing value for --grm-max-retries}"; shift 2 ;;
      --grm-retry-base-delay) GRM_RETRY_BASE_DELAY="${2:?Missing value for --grm-retry-base-delay}"; shift 2 ;;
      --grm-retry-max-delay) GRM_RETRY_MAX_DELAY="${2:?Missing value for --grm-retry-max-delay}"; shift 2 ;;
      --grm-max-trajectory-chars) GRM_MAX_TRAJECTORY_CHARS="${2:?Missing value for --grm-max-trajectory-chars}"; shift 2 ;;
      --grm-max-tokens) GRM_MAX_TOKENS="${2:?Missing value for --grm-max-tokens}"; shift 2 ;;
      --grm-temperature) GRM_TEMPERATURE="${2:?Missing value for --grm-temperature}"; shift 2 ;;
      --grm-failure-reward) GRM_FAILURE_REWARD="${2:?Missing value for --grm-failure-reward}"; shift 2 ;;
      --max-steps) MAX_STEPS="${2:?Missing value for --max-steps}"; shift 2 ;;
      --mcp-max-steps) MCP_MAX_STEPS="${2:?Missing value for --mcp-max-steps}"; shift 2 ;;
      --web-search-max-steps) WEB_SEARCH_MAX_STEPS="${2:?Missing value for --web-search-max-steps}"; shift 2 ;;
      --cli-max-steps) CLI_MAX_STEPS="${2:?Missing value for --cli-max-steps}"; shift 2 ;;
      --trajectory-timeout) TRAJECTORY_TIMEOUT="${2:?Missing value for --trajectory-timeout}"; shift 2 ;;
      --eval-trajectory-timeout) EVAL_TRAJECTORY_TIMEOUT="${2:?Missing value for --eval-trajectory-timeout}"; shift 2 ;;
      --eval-interval) EVAL_INTERVAL="${2:?Missing value for --eval-interval}"; shift 2 ;;
      --eval-config) EVAL_CONFIG="${2:?Missing value for --eval-config}"; shift 2 ;;
      --eval-max-response-len) EVAL_MAX_RESPONSE_LEN="${2:?Missing value for --eval-max-response-len}"; shift 2 ;;
      --eval-max-prompt-len) EVAL_MAX_PROMPT_LEN="${2:?Missing value for --eval-max-prompt-len}"; shift 2 ;;
      --eval-max-context-len) EVAL_MAX_CONTEXT_LEN="${2:?Missing value for --eval-max-context-len}"; shift 2 ;;
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
      --max-tool-output-length) MAX_TOOL_OUTPUT_LENGTH="${2:?Missing value for --max-tool-output-length}"; shift 2 ;;
      --sglang-server-concurrency) SGLANG_SERVER_CONCURRENCY="${2:?Missing value for --sglang-server-concurrency}"; shift 2 ;;
      --sglang-max-running-requests) SGLANG_MAX_RUNNING_REQUESTS="${2:?Missing value for --sglang-max-running-requests}"; shift 2 ;;
      --colocate) COLOCATE=true; shift ;;
      --no-colocate) COLOCATE=false; shift ;;
      --experiment-name) EXPERIMENT_NAME="${2:?Missing value for --experiment-name}"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
   esac
done

EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-$((EVAL_MAX_PROMPT_LEN + EVAL_MAX_RESPONSE_LEN))}"

# Keep the option/env name for compatibility with older launch commands, but do
# not allow this launcher to enable LexRank-based retrieval summaries.
RLLM_RETRIEVAL_LEXRANK_FALLBACK=0

case "${TERMINAL_LOG_STYLE}" in
   progress|rollouts|both) ;;
   *) echo "Invalid TERMINAL_LOG_STYLE=${TERMINAL_LOG_STYLE}; expected progress, rollouts, or both." >&2; exit 2 ;;
esac
FUSED_HARNESS="${FUSED_HARNESS:-gem}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
BASE_DIR="$(cd -- "${REPO_ROOT}/.." &>/dev/null && pwd)"

default_experiment_name() {
   #local prefix="fused-dapo-q3-4b-no_think-gem-async-dev"
   local prefix="asearcher-dapo-q3-4b-no_think-gem-sync-dev"
   #local prefix="asearcher-dapo-q3.5-4b-no_think-gem-sync-dev"
   #local prefix="asearcher-dapo-q3-4b-think-gem-sync-dev"
   #local prefix="asearcher-dapo-q3-8b-no_think-gem-sync-dev"
   #local prefix="asearcher-dapo-q3-8b-no_think-gem-async-dev"
   #local prefix="webqa-dapo-q3-4b-no_think-gem-async-dev0"
   local max_dev=-1
   local root base suffix
   for root in "${REPO_ROOT}/checkpoints/FusedRL" "${REPO_ROOT}/experiments/logs/FusedRL"; do
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

#MODEL_CONFIG="${MODEL_CONFIG:-qwen3-4B}"
#MODEL_CONFIG="${MODEL_CONFIG:-qwen3-8B}"
#MODEL_CONFIG="${MODEL_CONFIG:-qwen3.5-4B}"
MODEL_CONFIG="${MODEL_CONFIG:-qwen3-4B}"
source "${REPO_ROOT}/scripts/models/${MODEL_CONFIG}.sh"

# Default tensor-parallel size depends on the model. Gated attention
# (--attention-output-gate, qwen3.5) is broken in Megatron when
# num_query_groups < TP: the per-rank query head re-slice (attention.py step 4)
# is not mirrored on the gate tensor, so gate.view() fails with a size mismatch
# (factor = TP // num_query_groups). qwen3.5-4B has num_query_groups=4, so its TP
# must be <= 4. Non-gated models (qwen3-4B, num_query_groups=8) keep TP=8.
# NOTE: this is only the model-level cap; it is further clamped to ACTOR_GPUS
# below (a non-colocate 4-GPU actor cannot run TP=8).
if printf '%s\n' "${MODEL_ARGS[@]}" | grep -q -- "--attention-output-gate"; then
   DEFAULT_TP_SIZE=4
else
   DEFAULT_TP_SIZE=8
fi

# Non-colocate split on a single 8-GPU node: 4 GPUs train, 4 GPUs rollout.
# Defined here (before PERF_ARGS is built) so the TP clamp below actually takes
# effect — bash arrays expand their values at definition time.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

# TP cannot exceed the number of actor GPUs. Clamp the model-level default so a
# 4-GPU non-colocate actor uses TP=4 instead of the colocate-era TP=8 default.
if [ "${DEFAULT_TP_SIZE}" -gt "${ACTOR_GPUS}" ]; then
   DEFAULT_TP_SIZE="${ACTOR_GPUS}"
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(default_experiment_name)}"
#MODEL_DIR="${MODEL_DIR:-/share/nlp/share/plm/Qwen3-8B}"
#MODEL_DIR="${MODEL_DIR:-/share/nlp/share/plm/Qwen3.5-4B}"
MODEL_DIR="${MODEL_DIR:-/share/nlp/share/plm/Qwen3-4B}"
REF_LOAD="${REF_LOAD:-${MODEL_DIR}_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/FusedRL/${EXPERIMENT_NAME}}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${BASE_DIR}/Megatron-LM}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/experiments/logs/FusedRL/${EXPERIMENT_NAME}}"
EPISODE_LOG_DIR="${EPISODE_LOG_DIR:-${LOG_ROOT}}"
DUMP_DETAILS="${DUMP_DETAILS:-${LOG_ROOT}/debug}"

# Current slime's stock RolloutDataSource takes one --prompt-data path. Mirror
# the rllm fused launcher by accepting multiple train parquet files, then build
# one shuffled parquet before launching slime.
DEFAULT_TRAIN_FILES=(
   #"${SCRIPT_DIR}/artifacts/mcp_data_20260518/train.parquet"
   #"${SCRIPT_DIR}/artifacts/search_data_final/train.parquet"
   "${SCRIPT_DIR}/artifacts/asearcher.parquet"
)
TRAIN_FILE_PATHS=("${DEFAULT_TRAIN_FILES[@]}")
if [ -n "${TRAIN_FILES:-}" ]; then
   IFS=',' read -r -a TRAIN_FILE_PATHS <<< "${TRAIN_FILES}"
fi
#PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/artifacts/fused_mcp_search_train_shuffled.parquet}"
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/artifacts/asearcher.parquet}"
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
   echo "Also tried: ${rllm_fallback}" >&2
   echo "Also tried: ${rllm_experiments_fallback}" >&2
   exit 1
done

if [ "${#RESOLVED_TRAIN_FILES[@]}" -eq 0 ]; then
   echo "No training data files resolved." >&2
   exit 1
fi

mkdir -p "$(dirname "${PROMPT_DATA}")"
python3 - "${PROMPT_DATA}" "${SHUFFLE_TRAIN_DATA}" "${SHUFFLE_SEED}" "${RESOLVED_TRAIN_FILES[@]}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

output = Path(sys.argv[1])
shuffle = sys.argv[2].lower() in {"1", "true", "yes", "on"}
seed = int(sys.argv[3])
paths = [Path(p) for p in sys.argv[4:]]

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
if shuffle and combined.num_rows:
    indices = pa.array(np.random.default_rng(seed).permutation(combined.num_rows), type=pa.int64())
    combined = combined.take(indices)

# Write to a temp file then atomically rename. PROMPT_DATA may be the same path
# as an input file, so a crash mid-write must never corrupt the source parquet.
tmp_output = output.with_suffix(output.suffix + ".tmp")
pq.write_table(combined, tmp_output)
tmp_output.replace(output)
print(f"Wrote fused train parquet: {output}")
print(f"Rows: {combined.num_rows}")
print("Inputs:")
for path, table in zip(paths, tables):
    print(f"  {path}: {table.num_rows}")
print(f"Shuffle: {shuffle} seed={seed}")
PY
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

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-38000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${MAX_CONTEXT_LEN}}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-${ACCEPTED_GROUP_UPDATE_MIN_GROUPS}}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
NUM_EPOCH="${NUM_EPOCH:-100}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT))}"
ENABLE_DYNAMIC_SAMPLING_FILTER="${ENABLE_DYNAMIC_SAMPLING_FILTER:-true}"
DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
FULLY_ASYNC_FILTER_RELAX_AFTER_GROUPS="${FULLY_ASYNC_FILTER_RELAX_AFTER_GROUPS:-8192}"
if ! is_truthy "${ENABLE_DYNAMIC_SAMPLING_FILTER}"; then
   DYNAMIC_SAMPLING_FILTER_PATH=""
fi

if [ "${FULLY_ASYNC,,}" = "true" ] || [ "${FULLY_ASYNC}" = "1" ]; then
   ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-slime.rollout.fully_async_rollout.generate_rollout_fully_async}"
else
   ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-slime.rollout.sglang_rollout.generate_rollout}"
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MAX_CONTEXT_LEN}}"
if [ "${MAX_MODEL_LEN}" -ne "${MAX_CONTEXT_LEN}" ]; then
   echo "MAX_MODEL_LEN=${MAX_MODEL_LEN} must equal MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH (${MAX_CONTEXT_LEN}) for this slime launcher." >&2
   exit 2
fi
if [ "${MAX_TOKENS_PER_GPU}" -lt "${MAX_CONTEXT_LEN}" ]; then
   echo "MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU} is smaller than MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH (${MAX_CONTEXT_LEN})." >&2
   exit 2
fi
if [ "${LOG_PROBS_MAX_TOKENS_PER_GPU:-${MAX_CONTEXT_LEN}}" -lt "${MAX_CONTEXT_LEN}" ]; then
   echo "LOG_PROBS_MAX_TOKENS_PER_GPU=${LOG_PROBS_MAX_TOKENS_PER_GPU:-${MAX_CONTEXT_LEN}} is smaller than MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH (${MAX_CONTEXT_LEN})." >&2
   exit 2
fi

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

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
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

# When resuming a checkpoint saved under a different TP (e.g. migrating a TP=8
# colocate run to a TP=4 non-colocate layout), the DistributedOptimizer state
# uses sharding type dp_reshardable which cannot be resharded across TP. Set
# NO_LOAD_OPTIM=1 for that one migration launch to load only the (TP-reshardable)
# model weights and reinit the optimizer/RNG. New checkpoints this run saves are
# TP=4-native, so later resumes do NOT need this flag.
if is_truthy "${NO_LOAD_OPTIM:-false}"; then
   CKPT_ARGS+=(--no-load-optim --no-load-rng)
fi

ROLLOUT_ARGS=(
   --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"

   --prompt-data "${PROMPT_DATA}"
   --input-key "${INPUT_KEY:-prompt}"
   --label-key "${LABEL_KEY:-reward_model}"
   --metadata-key "${METADATA_KEY:-extra_info}"
   --tool-key "${TOOL_KEY:-tools}"
   --rollout-shuffle

   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-prompt-len "${MAX_PROMPT_LENGTH}"
   --rollout-max-response-len "${MAX_RESPONSE_LENGTH}"
   --rollout-temperature "${TEMPERATURE:-1.0}"
   --rollout-top-p "${TOP_P:-1.0}"

   --global-batch-size "${EFFECTIVE_GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --balance-data
)

if [ -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [ -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ] \
   && [ "${FULLY_ASYNC_FILTER_RELAX_AFTER_GROUPS}" -gt 0 ]; then
   ROLLOUT_ARGS+=(--fully-async-filter-relax-after-groups "${FULLY_ASYNC_FILTER_RELAX_AFTER_GROUPS}")
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
if [ "${PARTIAL_ROLLOUT,,}" = "true" ] || [ "${PARTIAL_ROLLOUT}" = "1" ]; then
   ROLLOUT_ARGS+=(--partial-rollout --mask-offpolicy-in-partial-rollout)
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
if is_truthy "${ENABLE_USE_GRM_EVALS}" || [ "${CUSTOM_RM_PATH:-}" = "${GRM_CUSTOM_RM_PATH}" ]; then
   ROLLOUT_ARGS+=(
      --grm-custom-rm-path "${GRM_CUSTOM_RM_PATH}"
      --grm-model "${GRM_MODEL}"
      --grm-concurrency "${GRM_CONCURRENCY}"
      --grm-max-connections "${GRM_MAX_CONNECTIONS}"
      --grm-timeout "${GRM_TIMEOUT}"
      --grm-max-retries "${GRM_MAX_RETRIES}"
      --grm-retry-base-delay "${GRM_RETRY_BASE_DELAY}"
      --grm-retry-max-delay "${GRM_RETRY_MAX_DELAY}"
      --grm-max-trajectory-chars "${GRM_MAX_TRAJECTORY_CHARS}"
      --grm-max-tokens "${GRM_MAX_TOKENS}"
      --grm-temperature "${GRM_TEMPERATURE}"
      --grm-failure-reward "${GRM_FAILURE_REWARD}"
   )
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
   EVAL_ARGS+=(--eval-interval "${EVAL_INTERVAL}")
   if [ -n "${EVAL_CONFIG}" ]; then
      EVAL_ARGS+=(--eval-config "${EVAL_CONFIG}")
   elif [ "${#EVAL_PROMPT_DATA[@]}" -gt 0 ]; then
      EVAL_ARGS+=(--eval-prompt-data "${EVAL_PROMPT_DATA[@]}")
   else
      echo "--eval-interval requires --eval-config or --eval-prompt-data." >&2
      exit 2
   fi
   EVAL_ARGS+=(--n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}")
   EVAL_ARGS+=(
      --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
      --eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}"
      --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
   )
   if ! is_truthy "${VAL_BEFORE_TRAIN}"; then
      EVAL_ARGS+=(--skip-eval-before-train)
   fi
fi
if [ -n "${DUMP_DETAILS}" ]; then
   ROLLOUT_ARGS+=(--dump-details "${DUMP_DETAILS}")
fi

PERF_ARGS=(
   # See DEFAULT_TP_SIZE above: qwen3.5 gated attention requires TP <= 4.
   --tensor-model-parallel-size "${TP_SIZE:-${DEFAULT_TP_SIZE}}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE:-1}"
   --context-parallel-size "${CP_SIZE:-1}"
   --expert-model-parallel-size "${EP_SIZE:-1}"
   --expert-tensor-parallel-size "${ETP_SIZE:-1}"

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size "${MICRO_BATCH_SIZE}"
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU:-${MAX_CONTEXT_LEN}}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-512}"
)

if [ "${USE_DYNAMIC_BATCH_SIZE:-1}" = "1" ]; then
   PERF_ARGS+=(--use-dynamic-batch-size)
fi

GRPO_ARGS=(
   --advantage-estimator "${ADVANTAGE_ESTIMATOR:-grpo}"
   --kl-coef "${KL_COEF:-0.01}"
   --kl-loss-coef "${KL_LOSS_COEF:-0.00}"
   --kl-loss-type low_var_kl
   --entropy-coef "${ENTROPY_COEF:-0.00}"
   --eps-clip "${EPS_CLIP:-0.2}"
   --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
)

# Rollout correction defaults:
# - TIS (Truncated Importance Sampling): soft correction. It multiplies pg_loss
#   by a clipped importance weight exp(train_log_probs - rollout_log_probs).
# - MIS (Masked Importance Sampling): hard correction through tis_mode=mask.
#   It masks out tokens/sequences whose importance ratio leaves the trust range.
# - RS (Rejection Sampling): an additional hard rejection mask, independent of
#   the TIS weighting mode. The recommended default here is TIS + RS: keep the
#   stable soft reweighting from TIS while rejecting severe rollout/trainer
#   mismatches with RS. To try pure MIS, point ROLLOUT_CORRECTION_CONFIG at a
#   config with tis_mode=mask and usually use_rs=false.
# Disable all rollout correction with USE_TIS=0. Override the YAML path or hook
# with ROLLOUT_CORRECTION_CONFIG / ROLLOUT_CORRECTION_FUNCTION.
if is_truthy "${USE_TIS:-1}"; then
   GRPO_ARGS+=(
      --use-tis
      --custom-config-path "${ROLLOUT_CORRECTION_CONFIG:-${REPO_ROOT}/experiments/fused_agent_tis_rs.yaml}"
      --custom-tis-function-path "${ROLLOUT_CORRECTION_FUNCTION:-examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp}"
   )
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-2e-6}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY:-0.05}"
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-${GPU_MEMORY_UTILIZATION:-0.9}}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
   --sglang-context-length "${MAX_CONTEXT_LEN}"
   --sglang-disable-custom-all-reduce
)

WANDB_ARGS=()
if [ "${USE_WANDB:-1}" = "1" ]; then
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
   fi
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

if [ "${PRINT_ROLLOUT_TRAJECTORY:-1}" = "1" ]; then
   MISC_ARGS+=(--print-rollout-trajectory)
fi
if [ "${PRINT_TRAIN_METRICS_TABLE:-1}" = "1" ]; then
   MISC_ARGS+=(--print-train-metrics-table)
fi

export SCRIPT_DIR REPO_ROOT MEGATRON_LM_PATH HAS_NVLINK
export SLIME_EPISODE_LOG_DIR="${EPISODE_LOG_DIR}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if is_truthy "${COLOCATE}"; then
   export NUM_GPUS="${NUM_GPUS:-${ACTOR_GPUS}}"
else
   export NUM_GPUS="${NUM_GPUS:-$((ACTOR_GPUS + ROLLOUT_GPUS))}"
fi

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export RAY_WARN_BLOCKING_GET_INSIDE_ASYNC="${RAY_WARN_BLOCKING_GET_INSIDE_ASYNC:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-10000000000}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export OPENROUTER_APP_NAME="${OPENROUTER_APP_NAME:-GRM}"

# Fused-agent service knobs inherited from the rllm launcher. They are runtime
# env only; current slime code consumes the subset used by selected generators.
export RETRIEVAL_SERVER_URL="${RETRIEVAL_SERVER_URL:-http://10.2.152.50:65432}"
export RLLM_RETRIEVAL_MODE="${RLLM_RETRIEVAL_MODE:-hybrid}"
export RLLM_RETRIEVAL_MAX_WORDS="${RLLM_RETRIEVAL_MAX_WORDS:-1024}"
export RETRIEVAL_MAX_RESULTS="${RETRIEVAL_MAX_RESULTS:-${RLLM_RETRIEVAL_MAX_RESULTS:-4}}"
export FUSED_WEBQA_MIN_UNIQUE_SEARCHES="${FUSED_WEBQA_MIN_UNIQUE_SEARCHES:-1}"
# Summarize is a ~2s LLM call per search with a large retry budget; on slow
# trajectories it stacks up and blows the 180s rollout collection timeout,
# causing groups to be dropped. Default off and use raw retrieve docs instead.
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
export FUSED_MAX_STEPS="${FUSED_MAX_STEPS:-${MAX_STEPS}}"
export FUSED_MCP_MAX_STEPS="${FUSED_MCP_MAX_STEPS:-${MCP_MAX_STEPS}}"
export FUSED_WEB_SEARCH_MAX_STEPS="${FUSED_WEB_SEARCH_MAX_STEPS:-${WEB_SEARCH_MAX_STEPS}}"
export FUSED_CLI_MAX_STEPS="${CLI_MAX_STEPS}"
export FUSED_TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT}"
export FUSED_EVAL_TRAJECTORY_TIMEOUT="${EVAL_TRAJECTORY_TIMEOUT}"
export PER_STEP_MAX_TOKENS="${PER_STEP_MAX_TOKENS:-2048}"
export SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH="${MAX_TOOL_OUTPUT_LENGTH}"
export SLIME_FUSED_TERMINAL_LOG_STYLE="${TERMINAL_LOG_STYLE}"
export SLIME_FUSED_ACCEPTED_GROUP_UPDATE_MAX_GROUPS="${ACCEPTED_GROUP_UPDATE_MAX_GROUPS}"
export SLIME_FUSED_TAIL_GUARD="${TAIL_GUARD}"
export SLIME_FUSED_TAIL_GUARD_TIME_GUARD="${TAIL_GUARD_TIME_GUARD}"
export SLIME_FUSED_TAIL_GUARD_TIME_MULTIPLIER="${TAIL_GUARD_TIME_MULTIPLIER}"
export SLIME_FUSED_TAIL_GUARD_TIME_SLACK_SECONDS="${TAIL_GUARD_TIME_SLACK_SECONDS}"
export SLIME_FUSED_TAIL_GUARD_MIN_COMPLETION_RATIO="${TAIL_GUARD_MIN_COMPLETION_RATIO}"
export CREDIT_ASSIGNMENT_ENABLE="${CREDIT_ASSIGNMENT_ENABLE}"
export CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR="${CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR}"
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
export SLIME_FUSED_CREDIT_ASSIGNMENT_ENABLE="${CREDIT_ASSIGNMENT_ENABLE}"
export SLIME_FUSED_CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR="${CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR}"
export SLIME_FUSED_CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY="${CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY}"
export SLIME_FUSED_CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS="${CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS}"
export SLIME_FUSED_CREDIT_ASSIGNMENT_NGRAM_REPETITION="${CREDIT_ASSIGNMENT_NGRAM_REPETITION}"
export SLIME_FUSED_CREDIT_ASSIGNMENT_SEARCH_BYPASS="${CREDIT_ASSIGNMENT_SEARCH_BYPASS}"
export SLIME_FUSED_CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL="${CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL}"
export SLIME_FUSED_CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER="${CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER}"
export SLIME_FUSED_CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP="${CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP}"
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

RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "HYDRA_FULL_ERROR", "NCCL_IB_DISABLE", "NCCL_TIMEOUT",
    "OPENROUTER_API_KEY", "OPENROUTER_SITE_URL", "OPENROUTER_APP_NAME",
    "SLIME_EPISODE_LOG_DIR",
    "RAY_WARN_BLOCKING_GET_INSIDE_ASYNC", "TOKENIZERS_PARALLELISM",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "VLLM_ENGINE_ITERATION_TIMEOUT_S",
    "VLLM_WORKER_MULTIPROC_METHOD", "PYTORCH_CUDA_ALLOC_CONF",
    "RETRIEVAL_SERVER_URL", "RLLM_RETRIEVAL_MODE", "RLLM_RETRIEVAL_MAX_WORDS",
    "RETRIEVAL_MAX_RESULTS", "RLLM_RETRIEVAL_SUMMARIZE",
    "FUSED_WEBQA_MIN_UNIQUE_SEARCHES",
    "RLLM_RETRIEVAL_RETRY_BUDGET", "RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET",
    "RLLM_RETRIEVAL_LEXRANK_FALLBACK", "RLLM_RETRIEVAL_LEXRANK_MAX_WORDS",
    "RLLM_RETRIEVAL_LEXRANK_MAX_SENTENCES", "RLLM_RETRIEVAL_LEXRANK_MAX_INPUT_SENTENCES",
    "RLLM_RETRIEVAL_LEXRANK_MULTIPROCESSING", "RLLM_RETRIEVAL_LEXRANK_WORKERS",
    "DOCKER_HOST", "DOCKER_API_VERSION", "WANDB_API_KEY",
    "RLLM_MCP_MIN_NOFILE", "RLLM_MCP_FD_THROTTLE_THRESHOLD", "RLLM_MCP_INIT_TIMEOUT",
    "RLLM_MCP_START_RETRIES", "RLLM_MCP_START_WAIT_TIMEOUT", "RLLM_MCP_TOOL_TIMEOUT",
    "RLLM_MCP_MAX_ACTIVE_SERVERS", "RLLM_MCP_PREFILTER_WORKERS", "RLLM_MCP_DISABLE_STEP_PENALTY",
    "FUSED_HARNESS", "FUSED_UNIFIED_SYSTEM_PROMPT", "FUSED_DISABLE_THINKING",
    "FUSED_MAX_STEPS", "FUSED_MCP_MAX_STEPS", "FUSED_WEB_SEARCH_MAX_STEPS", "FUSED_CLI_MAX_STEPS", "FUSED_TRAJECTORY_TIMEOUT",
    "FUSED_EVAL_TRAJECTORY_TIMEOUT", "SLIME_ROLLOUT_GROUP_TIMEOUT", "SLIME_EVAL_ROLLOUT_GROUP_TIMEOUT",
    "PER_STEP_MAX_TOKENS", "SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH",
    "SLIME_FUSED_TERMINAL_LOG_STYLE", "SLIME_FUSED_ACCEPTED_GROUP_UPDATE_MAX_GROUPS",
    "SLIME_FUSED_TAIL_GUARD", "SLIME_FUSED_TAIL_GUARD_TIME_GUARD",
    "SLIME_FUSED_TAIL_GUARD_TIME_MULTIPLIER", "SLIME_FUSED_TAIL_GUARD_TIME_SLACK_SECONDS",
    "SLIME_FUSED_TAIL_GUARD_MIN_COMPLETION_RATIO",
    "CREDIT_ASSIGNMENT_ENABLE", "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR",
    "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY", "CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS",
    "CREDIT_ASSIGNMENT_NGRAM_REPETITION", "CREDIT_ASSIGNMENT_NGRAM_REPETITION_N",
    "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD", "CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS",
    "CREDIT_ASSIGNMENT_SEARCH_BYPASS", "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL",
    "CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER", "CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP",
    "SLIME_FUSED_CREDIT_ASSIGNMENT_ENABLE", "SLIME_FUSED_CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR",
    "SLIME_FUSED_CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY",
    "SLIME_FUSED_CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS",
    "SLIME_FUSED_CREDIT_ASSIGNMENT_NGRAM_REPETITION",
    "SLIME_FUSED_CREDIT_ASSIGNMENT_SEARCH_BYPASS",
    "SLIME_FUSED_CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL",
    "SLIME_FUSED_CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER",
    "SLIME_FUSED_CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP",
    "FUSED_FILTER_MIN_MEAN_STEPS", "FUSED_FILTER_MIN_MCP_MEAN_STEPS",
    "FUSED_FILTER_MAX_ABNORMAL_RATIO",
    "FUSED_HORIZON_REWARD_MIN_MULTIPLIER", "FUSED_HORIZON_REWARD_GAMMA",
    "FUSED_HORIZON_REWARD_STEP_WEIGHT", "FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT",
    "FUSED_HORIZON_REWARD_TARGET_STEPS", "FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS",
    "SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["PYTHONPATH"] = f"{os.environ['MEGATRON_LM_PATH']}:{os.environ['REPO_ROOT']}:{os.environ['SCRIPT_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = os.environ["HAS_NVLINK"]
env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
print(json.dumps({"env_vars": env}))
PY
)

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_DIR}"
echo "Ref: ${REF_LOAD}"
echo "Prompt data: ${PROMPT_DATA}"
echo "Train rows: ${TRAIN_NUM_ROWS}; rollout_batch_size=${ROLLOUT_BATCH_SIZE}; num_epoch=${NUM_EPOCH}; num_rollout=${NUM_ROLLOUT}"
echo "Save dir: ${SAVE_DIR}"
echo "Log root: ${LOG_ROOT}"
echo "Episode dump root: ${EPISODE_LOG_DIR} (train/ and evals/)"
echo "Dump details dir: ${DUMP_DETAILS:-<disabled>}"
echo "W&B enabled: ${USE_WANDB:-0}"
echo "Custom generate: ${CUSTOM_GENERATE_FUNCTION_PATH:-<stock slime rollout>}"
echo "Custom reward post-process: ${CUSTOM_REWARD_POST_PROCESS_PATH:-<vanilla>}"
echo "Rollout function: ${ROLLOUT_FUNCTION_PATH}"
echo "Actor GPUs: ${ACTOR_GPUS}, rollout GPUs: ${ROLLOUT_GPUS}, colocate=${COLOCATE}, ray GPUs=${NUM_GPUS}"
echo "SGLang concurrency: server=${SGLANG_SERVER_CONCURRENCY}, max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS}"
echo "Fused controls: harness=${FUSED_HARNESS}, unified_system_prompt=${UNIFIED_SYSTEM_PROMPT}, disable_thinking=${DISABLE_THINKING}, max_steps=${FUSED_MAX_STEPS}, mcp_max_steps=${FUSED_MCP_MAX_STEPS}, web_search_max_steps=${FUSED_WEB_SEARCH_MAX_STEPS}, cli_max_steps=${CLI_MAX_STEPS}, per_step_max_tokens=${PER_STEP_MAX_TOKENS}, partial_rollout=${PARTIAL_ROLLOUT}, terminal_log_style=${TERMINAL_LOG_STYLE}"
echo "Accepted groups: min=${ACCEPTED_GROUP_UPDATE_MIN_GROUPS}, max=${ACCEPTED_GROUP_UPDATE_MAX_GROUPS}; micro_batch=${MICRO_BATCH_SIZE}, update_weights_interval=${UPDATE_WEIGHTS_INTERVAL}"
echo "Retrieval: mode=${RLLM_RETRIEVAL_MODE}, max_words=${RLLM_RETRIEVAL_MAX_WORDS}, max_results=${RETRIEVAL_MAX_RESULTS}, retry=${RLLM_RETRIEVAL_RETRY_BUDGET}, summary_retry=${RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET}, lexrank_fallback=${RLLM_RETRIEVAL_LEXRANK_FALLBACK}"
echo "Dynamic filter: enable=${ENABLE_DYNAMIC_SAMPLING_FILTER}, path=${DYNAMIC_SAMPLING_FILTER_PATH:-<none>}, relax_after_groups=${FULLY_ASYNC_FILTER_RELAX_AFTER_GROUPS}; webqa_min_unique_searches=${FUSED_WEBQA_MIN_UNIQUE_SEARCHES}"
echo "Eval: interval=${EVAL_INTERVAL:-<disabled>}, config=${EVAL_CONFIG:-<none>}, prompt_data=${EVAL_PROMPT_DATA[*]:-<none>}, n=${N_SAMPLES_PER_EVAL_PROMPT}, max_prompt_len=${EVAL_MAX_PROMPT_LEN}, max_response_len=${EVAL_MAX_RESPONSE_LEN}, max_context_len=${EVAL_MAX_CONTEXT_LEN}, val_before_train=${VAL_BEFORE_TRAIN}"
echo "OpenRouter GRM evals: enable=${ENABLE_USE_GRM_EVALS}, model=${GRM_MODEL}, concurrency=${GRM_CONCURRENCY}, timeout=${GRM_TIMEOUT}, retries=${GRM_MAX_RETRIES}, custom_rm=${GRM_CUSTOM_RM_PATH}"
echo "Buffer filter: enable_quota_bucket_sampling=${ENABLE_QUOTA_BUCKET_SAMPLING:-0}, path=${BUFFER_FILTER_PATH:-${ENABLE_QUOTA_BUCKET_SAMPLING:+slime.rollout.filter_hub.buffer_filters.quota_bucket_by_steps}}"
echo "Fused filter thresholds: min_mean_steps=${FUSED_FILTER_MIN_MEAN_STEPS}, min_mcp_mean_steps=${FUSED_FILTER_MIN_MCP_MEAN_STEPS}, max_abnormal_ratio=${FUSED_FILTER_MAX_ABNORMAL_RATIO}"
echo "Horizon reward shaping: enable=${HORIZON_REWARD_SHAPING}, min_multiplier=${FUSED_HORIZON_REWARD_MIN_MULTIPLIER}, gamma=${FUSED_HORIZON_REWARD_GAMMA}, step_weight=${FUSED_HORIZON_REWARD_STEP_WEIGHT}, tool_call_weight=${FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT}, target_steps=${FUSED_HORIZON_REWARD_TARGET_STEPS}, target_tool_calls=${FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS:-target_steps-1}"
echo "Rollout task family quotas: ${ROLLOUT_TASK_FAMILY_QUOTAS:-<none>}; candidate_multiplier=${SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER}"
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
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${ACTOR_GPUS}"
   --rollout-num-gpus "${ROLLOUT_GPUS}"
   --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}"
)
if is_truthy "${COLOCATE}"; then
   CLUSTER_ARGS+=(--colocate)
fi
if is_truthy "${OFFLOAD_TRAIN}"; then
   CLUSTER_ARGS+=(--offload-train)
else
   CLUSTER_ARGS+=(--no-offload-train)
fi

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

if [ "${RAY_JOB_WAIT:-0}" != "1" ] && [ "${RAY_JOB_FOLLOW_LOGS:-1}" = "1" ]; then
   echo "Following Ray job logs for ${RAY_SUBMISSION_ID}"
   ray job logs --address="${RAY_DASHBOARD_ADDRESS}" --follow "${RAY_SUBMISSION_ID}"
fi
