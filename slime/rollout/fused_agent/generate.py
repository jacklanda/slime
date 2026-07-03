from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.rollout.sglang_rollout import GenerateState, _extract_rollout_top_p_token_data
from slime.utils import http_utils
from slime.utils.types import Sample

from .env import FusedEnvironment, _format_retrieval, normalize_task, resolve_task_mode
from .parser import ToolCall, make_tool_parser
from .prompts import (
    COT_SYSTEM_PROMPT,
    COT_USER_PROMPT,
    FUSED_MCP_SYSTEM_PROMPT,
    FUSED_MCP_USER_PROMPT,
    FUSED_CLI_SYSTEM_PROMPT,
    FUSED_CLI_USER_PROMPT,
    FUSED_ET_SYSTEM_PROMPT,
    FUSED_ET_USER_PROMPT,
    FUSED_SEARCH_SYSTEM_PROMPT,
    FUSED_SEARCH_USER_PROMPT,
    FUSED_UNIFIED_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    REACT_USER_PROMPT,
    build_system_prompt,
    normalize_harness,
)

logger = logging.getLogger(__name__)
DEFAULT_SGLANG_CONTEXT_LENGTH_MARGIN = 256
_LAST_SGLANG_REQUEST_LOG_TS = 0.0


class SGLangContextLengthExceededError(ValueError):
    pass


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Run a fused-agent multi-turn workflow as a slime custom generate hook.

    This function is intentionally plugged into slime's stock
    ``sglang_rollout.generate_and_rm_group`` path, including the fully-async
    worker. It owns only per-sample agent interaction and returns train-ready
    ``Sample`` objects with token/logprob/loss-mask fields populated.
    """
    state = GenerateState(args)
    task = _task_from_sample(base_sample)
    env = FusedEnvironment(
        task,
        retrieval_url=os.environ.get("RETRIEVAL_SERVER_URL"),
        retrieval_max_results=int(os.environ.get("RETRIEVAL_MAX_RESULTS", "5")),
    )
    observation, info = env.reset()
    harness = normalize_harness(os.environ.get("FUSED_HARNESS", getattr(args, "fused_harness", "unified_gem")))
    base_max_steps = int(os.environ.get("FUSED_MAX_STEPS", getattr(args, "fused_max_steps", "16")))
    per_step_max_tokens = int(os.environ.get("PER_STEP_MAX_TOKENS", str(sampling_params.get("max_new_tokens", 2048))))
    disable_thinking = _env_bool("FUSED_DISABLE_THINKING", True)
    max_context_tokens = _effective_sglang_context_limit(args)
    max_tool_calls_per_turn = int(os.environ.get("FUSED_MAX_TOOL_CALLS_PER_TURN", os.environ.get("MAX_TOOL_CALLS_PER_TURN", "4")))
    credit_assignment_enable = _env_bool("CREDIT_ASSIGNMENT_ENABLE", True)
    credit_assignment_tool_parser_error = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR", True)
    credit_assignment_repeated_search_query = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY", True)
    credit_assignment_too_many_tool_calls = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS", True)
    credit_assignment_search_bypass = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_SEARCH_BYPASS", True)
    credit_assignment_direct_submit_without_tool = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL", True)
    credit_assignment_mixed_tool_and_answer = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER", True)
    credit_assignment_tail_guard_early_stop = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP", True)
    credit_assignment_ngram_repetition = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_NGRAM_REPETITION", True)
    credit_assignment_max_turns = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_MAX_TURNS", True)
    credit_assignment_max_response_len = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_MAX_RESPONSE_LEN", True)
    credit_assignment_parser_error_token_window = int(os.environ.get("CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR_TOKEN_WINDOW", "256"))
    ngram_repetition_n = int(os.environ.get("CREDIT_ASSIGNMENT_NGRAM_REPETITION_N", "8"))
    ngram_repetition_threshold = float(os.environ.get("CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD", "0.35"))
    ngram_repetition_min_tokens = int(os.environ.get("CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS", "128"))
    repeated_search_max_strikes = max(1, int(os.environ.get("FUSED_REPEATED_SEARCH_MAX_STRIKES", "2")))
    detect_abnormal_trajectories = not evaluation

    tools = env.tools()
    model_name = getattr(state.tokenizer, "name_or_path", None) or getattr(args, "hf_checkpoint", None)
    messages = _initial_messages(harness, info.get("task_type", ""), observation, tools, model_name)
    max_steps = _max_steps_for_mode(env.mode, base_max_steps)
    parser = make_tool_parser(model_name, valid_tools=_valid_tool_names(tools))
    manager = TrajectoryManager(fork_threshold_tokens=int(os.environ.get("SLIME_FUSED_FORK_THRESHOLD_TOKENS", "1024")))
    session_id = base_sample.session_id or uuid.uuid4().hex
    base_sample.session_id = session_id

    final_reward = 0.0
    final_done = False
    last_info: dict[str, Any] = {}
    total_steps = 0
    total_tool_call_turns = 0
    trajectory_steps: list[dict[str, Any]] = []
    episode_start_time = time.time()
    episode_start_timestamp = _utc_timestamp()
    llm_time = 0.0
    env_time = 0.0
    pending_turns: list[dict[str, Any]] = []
    seen_search_queries: set[str] = set()
    repeated_search_strikes = 0
    used_non_finish_tool = False
    credit_event: str | None = None
    credit_step_index: int | None = None
    try:
        for step_idx in range(max_steps):
            # Run the chat-template render off the event loop. The HF fast
            # tokenizer releases the GIL during tokenize, so offloading lets the
            # many concurrent trajectory coroutines actually overlap instead of
            # serializing on the single rollout event-loop thread (which
            # otherwise saturates one core and starves the SGLang engines).
            prompt_ids = await asyncio.to_thread(_render_prompt_ids, state.tokenizer, messages, disable_thinking=disable_thinking)
            if max_context_tokens and len(prompt_ids) >= max_context_tokens:
                final_done = True
                last_info = {
                    "termination_reason": "max_context_len_exceeded",
                    "prompt_tokens": len(prompt_ids),
                    "max_context_tokens": max_context_tokens,
                }
                break
            step_sampling = dict(sampling_params)
            step_sampling["max_new_tokens"] = max(0, min(int(step_sampling.get("max_new_tokens", per_step_max_tokens)), per_step_max_tokens))
            if max_context_tokens:
                step_sampling["max_new_tokens"] = min(step_sampling["max_new_tokens"], max_context_tokens - len(prompt_ids))
            if step_sampling["max_new_tokens"] <= 0:
                final_done = True
                last_info = {
                    "termination_reason": "max_context_len_exceeded",
                    "prompt_tokens": len(prompt_ids),
                    "max_context_tokens": max_context_tokens,
                }
                break

            llm_start = time.time()
            try:
                output = await _call_sglang(args, prompt_ids, step_sampling, session_id=session_id)
            except SGLangContextLengthExceededError as exc:
                final_done = True
                last_info = {
                    "termination_reason": "max_context_len_exceeded",
                    "prompt_tokens": len(prompt_ids),
                    "max_new_tokens": int(step_sampling.get("max_new_tokens", 0) or 0),
                    "max_context_tokens": max_context_tokens,
                    "error": str(exc),
                }
                break
            step_llm_time = time.time() - llm_start
            llm_time += step_llm_time
            output_ids = output["output_ids"]
            output_logprobs = output["output_logprobs"]
            raw_response = await asyncio.to_thread(state.tokenizer.decode, output_ids, skip_special_tokens=False) if output_ids else ""
            response = _strip_trailing_chat_template_stop(raw_response)
            finish_reason = output["finish_reason"]
            total_steps += 1
            parsed_actions = parser.parse(response)
            if any(action.name != "finish" for action in parsed_actions):
                total_tool_call_turns += 1

            assistant_msg = {"role": "assistant", "content": response}
            # Offload the second (no-generation-prompt) render off the event loop
            # for the same GIL-release reason as the line-124 render above.
            prompt_context_start_idx = await asyncio.to_thread(_last_assistant_context_start_idx, state.tokenizer, messages, disable_thinking=disable_thinking)
            pending_turns.append(
                {
                    "turn": TurnRecord(
                        prompt_ids=prompt_ids,
                        output_ids=output_ids,
                        finish_reason="tool_calls" if parsed_actions else finish_reason,
                        output_log_probs=output_logprobs,
                        loss_mask=_default_response_loss_mask(
                            state.tokenizer,
                            response,
                            output_len=len(output_ids),
                            disable_thinking=disable_thinking,
                            output_ids=output_ids,
                        ),
                        rollout_top_p_token_ids=output.get("rollout_top_p_token_ids"),
                        rollout_top_p_token_offsets=output.get("rollout_top_p_token_offsets"),
                        prompt_context_start_idx=prompt_context_start_idx,
                    ),
                    "prompt_messages": list(messages),
                    "response_message": assistant_msg,
                    "raw_response": response,
                    "metadata": {"sid": session_id, "step": step_idx},
                }
            )
            messages.append(assistant_msg)

            if detect_abnormal_trajectories and finish_reason == "length":
                final_done = True
                last_info = {"termination_reason": "max_response_len_exceeded"}
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action="",
                        reward=0.0,
                        done=True,
                        messages=messages,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break

            actions = parsed_actions
            if not actions:
                if detect_abnormal_trajectories and credit_assignment_tool_parser_error:
                    final_reward = 0.0
                    final_done = True
                    credit_event = "tool_parser_error"
                    credit_step_index = len(pending_turns) - 1
                    _set_pending_turn_error_span(
                        pending_turns[-1],
                        _response_span_to_output_token_span(
                            state.tokenizer,
                            response,
                            _parser_error_action_span(response),
                            output_len=len(output_ids),
                        ),
                        output_len=len(output_ids),
                    )
                    last_info = {
                        "termination_reason": "ABNORMAL_PARSE_ERROR",
                        "credit_assignment_event": credit_event,
                        "credit_assignment_error_step_index": credit_step_index,
                        "tool_parser_error_count": 1,
                    }
                    trajectory_steps.append(
                        _episode_step(
                            observation=observation,
                            response=response,
                            action="",
                            reward=0.0,
                            done=True,
                            messages=messages,
                            llm_time=step_llm_time,
                            env_time=0.0,
                            disable_thinking=disable_thinking,
                        )
                    )
                    break
                actions = [ToolCall("finish", {"command": "submit", "result": response})]
            if detect_abnormal_trajectories and max_tool_calls_per_turn > 0 and len(actions) > max_tool_calls_per_turn:
                final_reward = 0.0
                final_done = True
                if credit_assignment_too_many_tool_calls:
                    credit_event = "too_many_tool_calls"
                    credit_step_index = len(pending_turns) - 1
                    _set_pending_turn_error_span(
                        pending_turns[-1],
                        _response_span_to_output_token_span(
                            state.tokenizer,
                            response,
                            _actions_span(actions),
                            output_len=len(output_ids),
                        ),
                        output_len=len(output_ids),
                    )
                last_info = {
                    "termination_reason": "ABNORMAL_TOOL_BURST",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "too_many_tool_call_count": 1,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action="",
                        reward=0.0,
                        done=True,
                        messages=messages,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            if detect_abnormal_trajectories and credit_assignment_mixed_tool_and_answer and _has_mixed_tool_and_answer(response, actions):
                final_reward = 0.0
                final_done = True
                credit_event = "mixed_tool_and_answer"
                credit_step_index = len(pending_turns) - 1
                _set_pending_turn_error_span(
                    pending_turns[-1],
                    _response_span_to_output_token_span(
                        state.tokenizer,
                        response,
                        _mixed_tool_and_answer_span(response, actions),
                        output_len=len(output_ids),
                    ),
                    output_len=len(output_ids),
                )
                last_info = {
                    "termination_reason": "ABNORMAL_MIXED_TOOL_AND_ANSWER",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "mixed_tool_and_answer": True,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action=_format_action(actions[0]),
                        reward=0.0,
                        done=True,
                        messages=messages,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            direct_submit_without_tool = not used_non_finish_tool and _requires_non_finish_tool(tools) and all(action.name == "finish" for action in actions)
            if detect_abnormal_trajectories and direct_submit_without_tool and (step_idx == 0 or credit_assignment_direct_submit_without_tool):
                final_reward = 0.0
                final_done = True
                credit_event = "direct_submit_without_tool"
                credit_step_index = len(pending_turns) - 1
                last_info = {
                    "termination_reason": "ABNORMAL_DIRECT_SUBMIT_WITHOUT_TOOL",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "direct_submit_without_tool": True,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action=_format_action(actions[0]),
                        reward=0.0,
                        done=True,
                        messages=messages,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            repeated_action_span = _repeated_search_action_span(actions, seen_search_queries)
            repeated_query = _has_repeated_search_query(actions, seen_search_queries)
            if detect_abnormal_trajectories and repeated_query:
                repeated_search_strikes += 1
                duplicate_search_info = {
                    "duplicate_search_detected": True,
                    "duplicate_query_count": repeated_search_strikes,
                    "duplicate_query_max_strikes": repeated_search_max_strikes,
                }
                if repeated_search_strikes < repeated_search_max_strikes:
                    obs = "Repeated search query detected. Use different keywords, split the question into a new " "sub-query, or submit only if the existing evidence is sufficient."
                    formatted_obs = _format_tool_observation(actions[0].name, obs)
                    last_info = duplicate_search_info
                    trajectory_steps.append(
                        _episode_step(
                            observation=formatted_obs,
                            response=response,
                            action=_format_action(actions[0]),
                            reward=0.0,
                            done=False,
                            messages=messages,
                            llm_time=step_llm_time,
                            env_time=0.0,
                            disable_thinking=disable_thinking,
                        )
                    )
                    messages.append({"role": "user", "content": formatted_obs})
                    observation = formatted_obs
                    continue
                final_reward = 0.0
                final_done = True
                if credit_assignment_repeated_search_query:
                    credit_event = "repeated_search_query"
                    credit_step_index = len(pending_turns) - 1
                    _set_pending_turn_error_span(
                        pending_turns[-1],
                        _response_span_to_output_token_span(
                            state.tokenizer,
                            response,
                            repeated_action_span,
                            output_len=len(output_ids),
                        ),
                        output_len=len(output_ids),
                    )
                last_info = {
                    "termination_reason": "ABNORMAL_REPEATED_QUERY",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    **duplicate_search_info,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action=_format_action(actions[0]),
                        reward=0.0,
                        done=True,
                        messages=messages,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            repeated_output = _ngram_repetition_stats(
                output_ids,
                n=ngram_repetition_n,
                min_tokens=ngram_repetition_min_tokens,
            )
            repeated_action_span = _actions_span(actions)
            if detect_abnormal_trajectories and repeated_output["score"] > ngram_repetition_threshold and repeated_action_span is not None:
                final_reward = 0.0
                final_done = True
                if credit_assignment_ngram_repetition:
                    credit_event = "ngram_repetition"
                    credit_step_index = len(pending_turns) - 1
                    _set_pending_turn_error_span(
                        pending_turns[-1],
                        _response_span_to_output_token_span(
                            state.tokenizer,
                            response,
                            repeated_action_span,
                            output_len=len(output_ids),
                        ),
                        output_len=len(output_ids),
                    )
                last_info = {
                    "termination_reason": "ABNORMAL_NGRAM_REPETITION",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "ngram_repetition_detected": True,
                    "ngram_repetition_score": repeated_output["score"],
                    "ngram_repetition_n": repeated_output["n"],
                    "ngram_repetition_total": repeated_output["total"],
                    "ngram_repetition_unique": repeated_output["unique"],
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action=_format_action(actions[0]),
                        reward=0.0,
                        done=True,
                        messages=messages,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            action = actions[0]
            if action.name != "finish":
                used_non_finish_tool = True
            env_start = time.time()
            obs, reward, done, env_info = await env.step(action)
            step_env_time = time.time() - env_start
            env_time += step_env_time
            formatted_obs = _format_tool_observation(action.name, obs)
            final_reward = float(reward)
            final_done = bool(done)
            last_info = dict(env_info or {})
            if detect_abnormal_trajectories and last_info.get("credit_assignment") == "reasoning_step_only":
                final_reward = 0.0
                if last_info.get("termination_reason") == "ABNORMAL_SEARCH_BYPASS" and credit_assignment_search_bypass:
                    credit_event = "search_bypass"
                    credit_step_index = len(pending_turns) - 1
                else:
                    credit_event = "reasoning_step_only"
                    credit_step_index = len(pending_turns) - 1
            elif detect_abnormal_trajectories and last_info.get("termination_reason") == "ABNORMAL_SEARCH_BYPASS" and credit_assignment_search_bypass:
                final_reward = 0.0
                credit_event = "search_bypass"
                credit_step_index = len(pending_turns) - 1
            elif detect_abnormal_trajectories and last_info.get("termination_reason") in {"ABNORMAL_PARSE_ERROR", "INVALID_REACT_STRUCTURE", "INVALID_FINAL_STEP"} and credit_assignment_tool_parser_error:
                credit_event = "tool_parser_error"
                credit_step_index = len(pending_turns) - 1
            elif detect_abnormal_trajectories and last_info.get("termination_reason") == "ABNORMAL_NESTED_FINISH_PAYLOAD" and credit_assignment_tool_parser_error:
                credit_event = "tool_parser_error"
                credit_step_index = len(pending_turns) - 1
                _set_pending_turn_error_span(
                    pending_turns[-1],
                    _response_span_to_output_token_span(
                        state.tokenizer,
                        response,
                        _actions_span(actions),
                        output_len=len(output_ids),
                    ),
                    output_len=len(output_ids),
                )
            elif detect_abnormal_trajectories and last_info.get("termination_reason") == "ABNORMAL_TOOL_BURST" and credit_assignment_too_many_tool_calls:
                credit_event = "too_many_tool_calls"
                credit_step_index = len(pending_turns) - 1
            elif detect_abnormal_trajectories and last_info.get("termination_reason") == "ABNORMAL_REPEATED_QUERY" and credit_assignment_repeated_search_query:
                credit_event = "repeated_search_query"
                credit_step_index = len(pending_turns) - 1
            elif detect_abnormal_trajectories and last_info.get("termination_reason") == "ABNORMAL_NGRAM_REPETITION" and credit_assignment_ngram_repetition:
                credit_event = "ngram_repetition"
                credit_step_index = len(pending_turns) - 1
            elif detect_abnormal_trajectories and last_info.get("termination_reason") == "ABNORMAL_MIXED_TOOL_AND_ANSWER" and credit_assignment_mixed_tool_and_answer:
                credit_event = "mixed_tool_and_answer"
                credit_step_index = len(pending_turns) - 1
            trajectory_steps.append(
                _episode_step(
                    observation=formatted_obs,
                    response=response,
                    action=_format_action(action),
                    reward=final_reward if done else 0.0,
                    done=done,
                    messages=messages,
                    llm_time=step_llm_time,
                    env_time=step_env_time,
                    disable_thinking=disable_thinking,
                )
            )
            messages.append({"role": "user", "content": formatted_obs})
            observation = formatted_obs
            if done:
                break
        else:
            last_info = {**last_info, "termination_reason": "max_turns_exceeded"}
            if detect_abnormal_trajectories and credit_assignment_max_turns and pending_turns:
                credit_event = "max_turns_exceeded"
                credit_step_index = len(pending_turns) - 1
                response = str(pending_turns[-1].get("raw_response", ""))
                action_span = _actions_span(parser.parse(response))
                if action_span is not None:
                    _set_pending_turn_error_span(
                        pending_turns[-1],
                        _response_span_to_output_token_span(
                            state.tokenizer,
                            response,
                            action_span,
                            output_len=len(pending_turns[-1]["turn"].output_ids),
                        ),
                        output_len=len(pending_turns[-1]["turn"].output_ids),
                    )

        if not final_done and final_reward == 0.0:
            final_reward = env.compute_final_reward()
            last_info = {"reward_debug": env.reward_debug, **last_info}
    finally:
        env.close()

    termination_reason = last_info.get("termination_reason", "env_done" if final_done else "unknown")
    if detect_abnormal_trajectories and termination_reason == "TAIL_GUARD_EARLY_STOP" and credit_assignment_tail_guard_early_stop:
        credit_event = "tail_guard_early_stop"
    elif detect_abnormal_trajectories and termination_reason == "max_response_len_exceeded" and credit_assignment_max_response_len and pending_turns:
        credit_event = "max_response_len_exceeded"
        credit_step_index = len(pending_turns) - 1
    if credit_event is not None and credit_step_index is not None:
        last_info["credit_assignment_event"] = credit_event
        last_info["credit_assignment_error_step_index"] = credit_step_index

    _record_pending_turns(
        manager,
        session_id=session_id,
        pending_turns=pending_turns,
        credit_event=credit_event,
        credit_step_index=credit_step_index,
        parser_error_token_window=credit_assignment_parser_error_token_window,
    )

    episode_end_time = time.time()
    episode_timing = {
        "start_timestamp": episode_start_timestamp,
        "end_timestamp": _utc_timestamp(),
        "llm_time": llm_time,
        "env_time": env_time,
        "reward_time": 0.0,
        "total_time": episode_end_time - episode_start_time,
    }
    episode_dict = _rllm_episode_dict(
        base_sample=base_sample,
        task=task,
        session_id=session_id,
        reward=final_reward,
        termination_reason=termination_reason,
        reward_debug=env.reward_debug or last_info.get("reward_debug", {}),
        credit_event=credit_event,
        credit_step_index=credit_step_index,
        total_steps=total_steps,
        total_tool_call_turns=total_tool_call_turns,
        timing=episode_timing,
        steps=trajectory_steps,
        task_type=env.mode,
    )
    samples = manager.get_trajectory(
        session_id,
        base_sample=base_sample,
        reward=final_reward,
        allow_fully_masked=credit_event == "tail_guard_early_stop",
        extra_metadata={
            **dict(base_sample.metadata or {}),
            **last_info,
            "fused_task_type": env.mode,
            "fused_reward_debug": env.reward_debug or last_info.get("reward_debug", {}),
            "fused_termination": termination_reason,
            "credit_assignment_event": credit_event,
            "credit_assignment_error_step_index": credit_step_index,
            "fused_traj_steps": total_steps,
            "fused_tool_call_turns": total_tool_call_turns,
            "rllm_episode": episode_dict,
        },
    )
    if not samples:
        failed = Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            reward=0.0,
            status=Sample.Status.FAILED,
            metadata={
                **dict(base_sample.metadata or {}),
                "fused_error": "empty_trajectory",
                "fused_task_type": env.mode,
                "fused_termination": termination_reason,
                "credit_assignment_event": credit_event,
                "credit_assignment_error_step_index": credit_step_index,
                "fused_traj_steps": total_steps,
                "fused_tool_call_turns": total_tool_call_turns,
                "rllm_episode": episode_dict,
                **last_info,
            },
        )
        return failed
    for sample in samples:
        sample.reward = final_reward
        sample.status = Sample.Status.COMPLETED
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = [0.0] * sample.response_length
    return samples


def _task_from_sample(sample: Sample) -> dict[str, Any]:
    task: dict[str, Any] = {}
    if isinstance(sample.metadata, dict):
        task.update(sample.metadata)
    task["prompt"] = sample.prompt
    reward_model = sample.label
    ground_truth = _sample_ground_truth(sample)
    if isinstance(reward_model, dict):
        reward_model = dict(reward_model)
        if ground_truth is not None and not reward_model.get("ground_truth"):
            reward_model["ground_truth"] = ground_truth
    elif ground_truth is not None and reward_model is None:
        reward_model = {"ground_truth": ground_truth}
    task["reward_model"] = reward_model
    if ground_truth is not None:
        task["ground_truth"] = ground_truth
    return normalize_task(task)


def _sample_ground_truth(sample: Sample) -> Any:
    def _first_non_empty(values: list[Any]) -> Any:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, tuple, set, dict)) and not value:
                continue
            return value
        return None

    sources: list[Any] = []
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    extra_info = metadata.get("extra_info")
    if isinstance(extra_info, dict):
        sources.extend(extra_info.get(key) for key in ("ground_truth", "target", "answer", "answers"))
    sources.extend(metadata.get(key) for key in ("ground_truth", "target", "answer", "answers"))

    label = sample.label
    if isinstance(label, dict):
        sources.extend(label.get(key) for key in ("ground_truth", "target", "answer", "answers"))
    elif label is not None:
        sources.append(label)

    return _first_non_empty(list(sources))


def _initial_messages(harness: str, task_type: str, observation: str, tools: list[dict], model_name: str | None = None) -> list[dict[str, str]]:
    if harness == "bare":
        return [{"role": "user", "content": observation}]
    if harness == "cot":
        return [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": COT_USER_PROMPT.format(problem_statement=observation)},
        ]
    if harness == "react":
        system = build_system_prompt(REACT_SYSTEM_PROMPT, tools, model_name)
        user = REACT_USER_PROMPT.format(problem_statement=observation)
    elif task_type == "mcp":
        base = FUSED_UNIFIED_SYSTEM_PROMPT if harness == "unified_gem" else FUSED_MCP_SYSTEM_PROMPT
        system = build_system_prompt(base, tools, model_name)
        user = FUSED_MCP_USER_PROMPT.format(problem_statement=observation)
    elif task_type == "cli":
        base = FUSED_UNIFIED_SYSTEM_PROMPT if harness == "unified_gem" else FUSED_CLI_SYSTEM_PROMPT
        system = build_system_prompt(base, tools, model_name)
        user = FUSED_CLI_USER_PROMPT.format(problem_statement=observation)
    elif task_type == "et":
        base = FUSED_UNIFIED_SYSTEM_PROMPT if harness == "unified_gem" else FUSED_ET_SYSTEM_PROMPT
        system = build_system_prompt(base, tools, model_name)
        user = FUSED_ET_USER_PROMPT.format(problem_statement=observation)
    else:
        base = FUSED_UNIFIED_SYSTEM_PROMPT if harness == "unified_gem" else FUSED_SEARCH_SYSTEM_PROMPT
        system = build_system_prompt(base, tools, model_name)
        user = FUSED_SEARCH_USER_PROMPT.format(problem_statement=observation)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _valid_tool_names(tools: list[dict]) -> set[str]:
    names = set()
    for schema in tools:
        fn = schema.get("function", schema)
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.add(name)
            names.add(str(name).replace("-", "_"))
    names.update({"finish", "submit"})
    return names


def _requires_non_finish_tool(tools: list[dict]) -> bool:
    return any(name not in {"finish", "submit"} for name in _declared_tool_names(tools))


def _declared_tool_names(tools: list[dict]) -> set[str]:
    names = set()
    for schema in tools:
        fn = schema.get("function", schema)
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            normalized = str(name).strip().replace("-", "_")
            names.add(normalized)
    return names


def _strip_trailing_chat_template_stop(text: str) -> str:
    stripped = text
    while True:
        without_ws = stripped.rstrip()
        if not without_ws.endswith("<|im_end|>"):
            return stripped
        stripped = without_ws[: -len("<|im_end|>")]


def _assistant_response_for_prompt_replay(response: str, disable_thinking: bool) -> str:
    if not disable_thinking:
        return response
    empty_thinking_prefix = "<think>\n\n</think>\n\n"
    if response.startswith(empty_thinking_prefix):
        return response
    return empty_thinking_prefix + response


def _has_repeated_search_query(actions: list[ToolCall], seen_queries: set[str]) -> bool:
    repeated = False
    for action in actions:
        if not _is_web_search_tool(action.name):
            continue
        query = _normalize_search_query((action.arguments or {}).get("query") or "")
        if not query:
            continue
        if query in seen_queries:
            repeated = True
        else:
            seen_queries.add(query)
    return repeated


def _actions_span(actions: list[ToolCall]) -> tuple[int, int] | None:
    spans = [(action.start, action.end) for action in actions if action.start is not None and action.end is not None]
    if not spans:
        return None
    return min(start for start, _ in spans), max(end for _, end in spans)


def _repeated_search_action_span(actions: list[ToolCall], prior_seen_queries: set[str]) -> tuple[int, int] | None:
    spans = []
    seen = set(prior_seen_queries)
    for action in actions:
        if not _is_web_search_tool(action.name):
            continue
        query = _normalize_search_query((action.arguments or {}).get("query") or "")
        if not query:
            continue
        if query in seen and action.start is not None and action.end is not None:
            spans.append((action.start, action.end))
        seen.add(query)
    if not spans:
        return _actions_span([action for action in actions if _is_web_search_tool(action.name)])
    return min(start for start, _ in spans), max(end for _, end in spans)


def _normalize_search_query(query: Any) -> str:
    return " ".join(str(query or "").strip().lower().split())


def _has_mixed_tool_and_answer(response: str, actions: list[ToolCall]) -> bool:
    return any(action.name != "finish" for action in actions) and (any(action.name == "finish" for action in actions) or _answer_span(response, excluded_spans=_actions_spans(actions)) is not None)


def _mixed_tool_and_answer_span(response: str, actions: list[ToolCall]) -> tuple[int, int] | None:
    spans = [(action.start, action.end) for action in actions if action.start is not None and action.end is not None and action.name != "finish"]
    answer_span = _answer_span(response, excluded_spans=_actions_spans(actions))
    if answer_span is not None:
        spans.append(answer_span)
    else:
        spans.extend((action.start, action.end) for action in actions if action.start is not None and action.end is not None and action.name == "finish")
    if not spans:
        return _actions_span(actions)
    return min(start for start, _ in spans), max(end for _, end in spans)


def _actions_spans(actions: list[ToolCall]) -> list[tuple[int, int]]:
    return [(action.start, action.end) for action in actions if action.start is not None and action.end is not None]


def _answer_span(response: str, *, excluded_spans: list[tuple[int, int]] | None = None) -> tuple[int, int] | None:
    excluded_spans = excluded_spans or []
    answer_matches = list(re.finditer(r"<answer>\s*.*?\s*</answer>", response or "", flags=re.DOTALL | re.IGNORECASE))
    for match in reversed(answer_matches):
        if _span_overlaps_any((match.start(), match.end()), excluded_spans):
            continue
        return match.start(), match.end()
    boxed_start, boxed_end = _boxed_answer_span(response or "")
    if boxed_start is not None and boxed_end is not None and not _span_overlaps_any((boxed_start, boxed_end), excluded_spans):
        return boxed_start, boxed_end
    return None


def _span_overlaps_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in spans)


def _boxed_answer_span(text: str) -> tuple[int | None, int | None]:
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return None, None
    start = idx + len(marker)
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return idx, i + 1
    return None, None


def _ngram_repetition_stats(output_ids: list[int], *, n: int, min_tokens: int) -> dict[str, float | int]:
    if n <= 0 or len(output_ids) < max(min_tokens, n):
        return {"score": 0.0, "n": n, "total": 0, "unique": 0}
    ngrams = [tuple(output_ids[i : i + n]) for i in range(len(output_ids) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    score = 1.0 - unique / total if total else 0.0
    return {"score": score, "n": n, "total": total, "unique": unique}


def _parser_error_action_span(response: str) -> tuple[int, int] | None:
    span = _first_unclosed_tool_call_span(response)
    if span is not None:
        return span
    span = _first_malformed_tool_call_span(response)
    if span is not None:
        return span
    idx = response.rfind("</think>")
    if idx >= 0:
        start = idx + len("</think>")
        while start < len(response) and response[start].isspace():
            start += 1
        if start < len(response):
            return start, len(response)
    return (0, len(response)) if response else None


def _first_unclosed_tool_call_span(response: str) -> tuple[int, int] | None:
    start = response.find("<tool_call>")
    if start < 0:
        return None
    end = response.find("</tool_call>", start + len("<tool_call>"))
    if end >= 0:
        return None
    return start, len(response)


def _first_malformed_tool_call_span(response: str) -> tuple[int, int] | None:
    import re

    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL):
        try:
            json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return match.start(), match.end()
    return None


def _set_pending_turn_error_span(item: dict[str, Any], span: tuple[int, int] | None, *, output_len: int) -> None:
    if span is None:
        return
    start, end = span
    start = max(0, min(output_len, int(start)))
    end = max(start, min(output_len, int(end)))
    if end <= start:
        return
    item["credit_assignment_action_span"] = (start, end)


def _response_span_to_output_token_span(
    tokenizer,
    response: str,
    char_span: tuple[int, int] | None,
    *,
    output_len: int,
) -> tuple[int, int] | None:
    if char_span is None:
        return None
    char_start, char_end = char_span
    if len(response) == output_len:
        return char_start, char_end
    start = _encode_len(tokenizer, response[:char_start])
    end = _encode_len(tokenizer, response[:char_end])
    if start is None or end is None:
        return (0, output_len)
    return start, end


def _encode_len(tokenizer, text: str) -> int | None:
    try:
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text, add_special_tokens=False))
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict):
            return len(encoded["input_ids"])
        return len(encoded)
    except Exception:
        return None


def _record_pending_turns(
    manager: TrajectoryManager,
    *,
    session_id: str,
    pending_turns: list[dict[str, Any]],
    credit_event: str | None,
    credit_step_index: int | None,
    parser_error_token_window: int,
) -> None:
    for idx, item in enumerate(pending_turns):
        turn = item["turn"]
        policy_loss_mask = _credit_assignment_loss_mask(
            output_len=len(turn.output_ids),
            turn_index=idx,
            credit_event=credit_event,
            credit_step_index=credit_step_index,
            parser_error_token_window=parser_error_token_window,
            action_span=item.get("credit_assignment_action_span"),
        )
        metadata = dict(item["metadata"])
        if credit_event is not None:
            metadata["credit_assignment_event"] = credit_event
            metadata["credit_assignment_error_step_index"] = credit_step_index
            if idx == credit_step_index and item.get("credit_assignment_action_span") is not None:
                start, end = item["credit_assignment_action_span"]
                metadata["credit_assignment_action_start"] = start
                metadata["credit_assignment_action_end"] = end
        manager.record_turn(
            session_id,
            turn=TurnRecord(
                prompt_ids=turn.prompt_ids,
                output_ids=turn.output_ids,
                finish_reason=turn.finish_reason,
                output_log_probs=turn.output_log_probs,
                loss_mask=turn.loss_mask,
                policy_loss_mask=policy_loss_mask,
                prompt_context_start_idx=turn.prompt_context_start_idx,
                rollout_top_p_token_ids=turn.rollout_top_p_token_ids,
                rollout_top_p_token_offsets=turn.rollout_top_p_token_offsets,
            ),
            prompt_messages=item["prompt_messages"],
            response_message=item["response_message"],
            metadata=metadata,
        )


def _credit_assignment_loss_mask(
    *,
    output_len: int,
    turn_index: int,
    credit_event: str | None,
    credit_step_index: int | None,
    parser_error_token_window: int = 256,
    action_span: tuple[int, int] | None = None,
) -> list[int] | None:
    if credit_event is None:
        return None
    if credit_event == "search_bypass":
        return None
    if credit_event == "direct_submit_without_tool":
        return [0] * output_len
    if credit_event in {"tail_guard_early_stop"}:
        return [0] * output_len
    if credit_step_index is None:
        return [0] * output_len
    if credit_event == "mixed_tool_and_answer":
        return [1] * output_len if turn_index == credit_step_index else [0] * output_len
    if turn_index == credit_step_index and action_span is not None:
        start, end = action_span
        if 0 <= start < end <= output_len:
            return [0] * start + [1] * (end - start) + [0] * (output_len - end)
    if credit_event == "tool_parser_error" and turn_index == credit_step_index:
        trained_len = max(0, min(output_len, parser_error_token_window))
        return [0] * (output_len - trained_len) + [1] * trained_len
    if credit_event == "max_response_len_exceeded" and turn_index == credit_step_index:
        trained_len = max(0, min(output_len, parser_error_token_window))
        return [0] * (output_len - trained_len) + [1] * trained_len
    return [1] * output_len if turn_index == credit_step_index else [0] * output_len


def _default_response_loss_mask(
    tokenizer,
    response: str,
    *,
    output_len: int,
    disable_thinking: bool,
    output_ids: list[int] | None = None,
) -> list[int] | None:
    if output_len <= 0:
        return []
    if not disable_thinking:
        # enable-thinking: train the whole response (reasoning + answer). The
        # builder treats None as an all-ones mask.
        return None

    # disable-thinking: the empty think shell "<think>\n\n</think>\n\n" lives in
    # the prompt, so a well-behaved response carries no think block and trains in
    # full. Only when the model *mis-fires* a leading <think>...</think> despite
    # being told not to think do we mask that stray block out.
    #
    # Prefer locating </think> in token space: it is a single, non-mergeable
    # added token in both Qwen3 and Qwen3.5 tokenizers, so scanning output_ids
    # for its id gives an exact boundary and avoids the ±1 drift of re-encoding
    # a character substring (_encode_len(response[:think_end])).
    close_id = _think_close_token_id(tokenizer)
    if output_ids is not None and close_id is not None:
        try:
            j = output_ids.index(close_id)
        except ValueError:
            return [1] * output_len
        # Mask the mis-fired think block through </think> itself; the answer
        # (everything after </think>) stays trainable. Any trailing "\n\n"
        # separator sits in the trainable side but carries no real content.
        start = min(j + 1, output_len)
        return [0] * start + [1] * (output_len - start)

    # Fallback for tokenizers without a resolvable </think> id: character-level
    # boundary with the pre-existing ±1 re-encode behavior.
    think_end = _leading_think_block_end(response)
    if think_end is None:
        return [1] * output_len

    start = _encode_len(tokenizer, response[:think_end])
    if start is None:
        start = min(max(0, think_end), output_len) if len(response) == output_len else output_len
    start = max(0, min(output_len, start))
    return [0] * start + [1] * (output_len - start)


def _think_close_token_id(tokenizer) -> int | None:
    """Resolve the single-token id of ``</think>`` for this tokenizer, or None.

    Both Qwen3 (151668) and Qwen3.5 (248069) expose ``</think>`` as one added
    token; other tokenizers may lack it or split it, in which case we return None
    and callers fall back to character-level handling.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        tid = convert("</think>")
    except Exception:
        return None
    if tid is None:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and tid == unk_id:
        return None
    return tid


def _leading_think_block_end(response: str) -> int | None:
    text = str(response or "")
    stripped = text.lstrip()
    if not stripped.startswith("<think>"):
        return None
    prefix_len = len(text) - len(stripped)
    end = stripped.find("</think>")
    if end < 0:
        return None
    block_end = prefix_len + end + len("</think>")
    while block_end < len(text) and text[block_end] in {"\n", "\r"}:
        block_end += 1
    return block_end


def _render_prompt_ids(tokenizer, messages: list[dict[str, Any]], *, disable_thinking: bool = True) -> list[int]:
    rendered = _apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        disable_thinking=disable_thinking,
    )
    if hasattr(rendered, "data") and isinstance(rendered.data, dict):
        rendered = rendered.data["input_ids"]
    elif isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    return list(rendered)


def _last_assistant_context_start_idx(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    disable_thinking: bool = True,
) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return len(
                _render_messages_without_generation_prompt(
                    tokenizer,
                    messages[: idx + 1],
                    disable_thinking=disable_thinking,
                )
            )
    return None


def _render_messages_without_generation_prompt(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    disable_thinking: bool = True,
) -> list[int]:
    rendered = _apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        disable_thinking=disable_thinking,
    )
    if hasattr(rendered, "data") and isinstance(rendered.data, dict):
        rendered = rendered.data["input_ids"]
    elif isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    return list(rendered)


def _apply_chat_template(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    disable_thinking: bool,
):
    messages = _prepare_messages_for_chat_template(messages, disable_thinking=disable_thinking)
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": not disable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(
            _prepare_messages_for_fallback_chat_template(messages, disable_thinking=disable_thinking),
            **kwargs,
        )
        if not (disable_thinking and add_generation_prompt):
            return rendered
        return _append_disabled_thinking_generation_prefix(tokenizer, rendered, tokenize=tokenize)


def _prepare_messages_for_chat_template(messages: list[dict[str, Any]], *, disable_thinking: bool) -> list[dict[str, Any]]:
    if not disable_thinking:
        return messages
    prepared = []
    for message in messages:
        if message.get("role") != "assistant":
            prepared.append(message)
            continue
        content = str(message.get("content") or "")
        if "<think>" in content or message.get("reasoning_content") is not None:
            prepared.append(message)
            continue
        # Preserve an explicit empty think shell during prompt replay for
        # chat templates that suppress historical <think> blocks when the
        # reasoning content is falsy.
        prepared.append({**message, "reasoning_content": "\n"})
    return prepared


def _prepare_messages_for_fallback_chat_template(
    messages: list[dict[str, Any]],
    *,
    disable_thinking: bool,
) -> list[dict[str, Any]]:
    if not disable_thinking:
        return messages
    empty_thinking_prefix = "<think>\n\n</think>\n\n"
    prepared = []
    for message in messages:
        if message.get("role") != "assistant":
            prepared.append(message)
            continue
        content = str(message.get("content") or "")
        if content.startswith(empty_thinking_prefix):
            prepared.append({**message, "content": content})
            continue
        prepared.append({**message, "content": empty_thinking_prefix + content})
    return prepared


def _append_disabled_thinking_generation_prefix(tokenizer, rendered, *, tokenize: bool):
    empty_thinking_prefix = "<think>\n\n</think>\n\n"
    if not tokenize:
        return str(rendered) + empty_thinking_prefix

    prefix_ids = _tokenize_text(tokenizer, empty_thinking_prefix)
    if hasattr(rendered, "data") and isinstance(rendered.data, dict):
        data = dict(rendered.data)
        data["input_ids"] = list(data["input_ids"]) + prefix_ids
        if "attention_mask" in data:
            data["attention_mask"] = list(data["attention_mask"]) + [1] * len(prefix_ids)
        rendered.data = data
        return rendered
    if isinstance(rendered, dict):
        rendered = dict(rendered)
        rendered["input_ids"] = list(rendered["input_ids"]) + prefix_ids
        if "attention_mask" in rendered:
            rendered["attention_mask"] = list(rendered["attention_mask"]) + [1] * len(prefix_ids)
        return rendered
    return list(rendered) + prefix_ids


def _tokenize_text(tokenizer, text: str) -> list[int]:
    if not callable(tokenizer):
        return [ord(ch) for ch in text]
    encoded = tokenizer(text, add_special_tokens=False)
    if hasattr(encoded, "data") and isinstance(encoded.data, dict):
        encoded = encoded.data
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    return list(encoded)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _episode_step(
    *,
    observation: str,
    response: str,
    action: str,
    reward: float,
    done: bool,
    messages: list[dict[str, Any]],
    llm_time: float,
    env_time: float,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    return {
        "observation": observation,
        "thought": _extract_thought(response),
        "action": action,
        "reward": float(reward),
        "done": bool(done),
        "model_response": response,
        "chat_completions": _chat_completions_for_step(messages, response),
        "info": {
            "disable_thinking": bool(disable_thinking),
            "timing": {
                "start_timestamp": _utc_timestamp(),
                "end_timestamp": _utc_timestamp(),
                "llm_time": llm_time,
                "env_time": env_time,
            },
        },
    }


def _chat_completions_for_step(messages: list[dict[str, Any]], response: str) -> list[dict[str, Any]]:
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == response:
        return list(messages)
    return [*messages, {"role": "assistant", "content": response}]


def _extract_thought(response: str) -> str:
    start = response.find("<think>")
    end = response.find("</think>")
    if start >= 0 and end > start:
        return response[start : end + len("</think>")]
    return response


def _format_action(action: ToolCall) -> str:
    payload = {"name": action.name, "arguments": action.arguments or {}}
    return "<tool_call>" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "</tool_call>"


def _format_tool_observation(tool_name: str, output: Any) -> str:
    output_text = _format_observation_output(tool_name, output)
    return "<tool_response>\n" f"Execution output of [{tool_name}]:\n" f"{output_text}\n" "</tool_response>"


def _format_observation_output(tool_name: str, output: Any) -> str:
    if _is_web_search_tool(tool_name):
        return _format_web_search_output(output)
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False, default=str)


def _is_web_search_tool(tool_name: str) -> bool:
    return str(tool_name or "").strip().lower().replace("-", "_") in {"web_search", "search", "webqa"}


def _format_web_search_output(output: Any) -> str:
    if not isinstance(output, str):
        return _format_retrieval(output)
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return " ".join(output.split())
    return _format_retrieval(parsed)


def _rllm_episode_dict(
    *,
    base_sample: Sample,
    task: dict[str, Any],
    session_id: str,
    reward: float,
    termination_reason: str,
    reward_debug: dict[str, Any],
    credit_event: str | None,
    credit_step_index: int | None,
    total_steps: int,
    total_tool_call_turns: int,
    timing: dict[str, Any],
    steps: list[dict[str, Any]],
    task_type: str,
) -> dict[str, Any]:
    task_for_dump = _task_for_dump(task)
    episode_id = _episode_id(base_sample, task_for_dump)
    benchmark = _benchmark_metric_name(task_for_dump, task_type)
    metrics = {
        f"{benchmark}/pass@1": float(reward > 0),
        "traj/steps": float(total_steps),
        "turn/tool_call_turn": float(total_tool_call_turns),
    }
    suffix = _task_metric_suffix(task_type)
    metrics[f"traj/steps/{suffix}"] = float(total_steps)
    metrics[f"turn/tool_call_turn/{suffix}"] = float(total_tool_call_turns)
    for key, value in (reward_debug or {}).items():
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    trajectory = {
        "uid": str(uuid.uuid4()),
        "name": f"{benchmark}_0",
        "task": task_for_dump,
        "steps": steps,
        "reward": float(reward),
        "info": {"timing": timing},
    }
    metadata = {
        "reward_debug": reward_debug or {},
        "timing": timing,
    }
    if credit_event is not None:
        metadata["credit_assignment_event"] = credit_event
        metadata["credit_assignment_error_step_index"] = credit_step_index
    return {
        "id": episode_id,
        "task": task_for_dump,
        "termination_reason": termination_reason,
        "is_correct": bool(reward > 0),
        "session_id": session_id,
        "trajectories": [trajectory],
        "metrics": metrics,
        "metadata": metadata,
        "info": {"timing": timing},
    }


def _task_for_dump(task: dict[str, Any]) -> dict[str, Any]:
    cleaned = {k: v for k, v in task.items() if k not in {"image", "images", "prompt", "reward_model"}}
    if "question" not in cleaned and isinstance(task.get("prompt"), str):
        cleaned["question"] = task["prompt"]
    if "ground_truth" not in cleaned and task.get("reward_model") is not None:
        cleaned["ground_truth"] = task["reward_model"]
    if not cleaned.get("data_source"):
        mode = resolve_task_mode(task)
        if mode in {"mcp", "cli", "et"}:
            cleaned["data_source"] = mode
    return cleaned


def _episode_id(sample: Sample, task: dict[str, Any]) -> str:
    task_key = task.get("id") or task.get("uuid") or task.get("instance_id") or task.get("question") or task
    task_hash = hashlib.sha256(json.dumps(task_key, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:32]
    rollout_idx = sample.rollout_id if sample.rollout_id is not None else sample.index
    if rollout_idx is None:
        rollout_idx = 0
    return f"{task_hash}:{rollout_idx}"


def _benchmark_metric_name(task_for_dump: dict[str, Any], task_type: str) -> str:
    """Metric/trajectory name keyed by benchmark source (e.g. ``medqa``).

    Falls back to the normalized task_type suffix, then ``unknown``.
    """
    raw = task_for_dump.get("data_source") or task_for_dump.get("benchmark") or task_for_dump.get("dataset") or ""
    normalized = str(raw).strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    return normalized or _task_metric_suffix(task_type)


def _task_metric_suffix(task_type: str) -> str:
    normalized = str(task_type or "").lower().replace("-", "_").replace(" ", "_")
    if normalized in {"web_search", "search", "webqa"}:
        return "webqa"
    if normalized in {"mcp", "cli"}:
        return normalized
    if normalized in {"et", "endless_terminal", "endless_terminals", "swe"}:
        return "cli"
    return normalized or "unknown"


def _max_steps_for_mode(task_type: str, default: int) -> int:
    normalized = str(task_type or "").upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "WEB_SEARCH": ("FUSED_WEB_SEARCH_MAX_STEPS", "FUSED_WEBQA_MAX_STEPS"),
        "MCP": ("FUSED_MCP_MAX_STEPS",),
        "CLI": ("FUSED_CLI_MAX_STEPS",),
        "ET": ("FUSED_ET_MAX_STEPS", "FUSED_CLI_MAX_STEPS"),
    }
    for env_name in aliases.get(normalized, ()):
        value = os.environ.get(env_name)
        if value:
            return max(1, int(value))
    return default


def _effective_sglang_context_limit(args) -> int:
    limits = [
        int(value)
        for value in (
            getattr(args, "sglang_context_length", None),
            getattr(args, "rollout_max_context_len", None),
        )
        if value
    ]
    if not limits:
        return 0
    margin = max(0, int(os.environ.get("SGLANG_CONTEXT_LENGTH_MARGIN", DEFAULT_SGLANG_CONTEXT_LENGTH_MARGIN)))
    return max(0, min(limits) - margin)


async def _call_sglang(args, prompt_ids: list[int], sampling_params: dict[str, Any], *, session_id: str) -> dict[str, Any]:
    global _LAST_SGLANG_REQUEST_LOG_TS
    max_new_tokens = int(sampling_params.get("max_new_tokens", 0) or 0)
    max_context_tokens = _effective_sglang_context_limit(args)
    requested_tokens = len(prompt_ids) + max_new_tokens
    if max_context_tokens and requested_tokens > max_context_tokens:
        raise SGLangContextLengthExceededError(f"SGLang request would use {requested_tokens} tokens " f"({len(prompt_ids)} prompt + {max_new_tokens} new), exceeding local limit {max_context_tokens}.")
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    rid = uuid.uuid4().hex
    payload = {
        "rid": rid,
        "input_ids": prompt_ids,
        "sampling_params": {
            **sampling_params,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "no_stop_trim": True,
        },
        "return_logprob": True,
    }
    headers = {"X-SMG-Routing-Key": session_id} if getattr(args, "router_policy", None) == "consistent_hashing" else None
    started = time.time()
    now = started
    should_log = _env_bool("SLIME_FUSED_PROGRESS_LOGS", False) and now - _LAST_SGLANG_REQUEST_LOG_TS >= float(os.environ.get("SLIME_FUSED_SGLANG_LOG_INTERVAL", "10"))
    if should_log:
        _LAST_SGLANG_REQUEST_LOG_TS = now
        logger.info(
            "fused-agent sending SGLang generate request rid=%s prompt_tokens=%d max_new_tokens=%d url=%s",
            rid,
            len(prompt_ids),
            max_new_tokens,
            url,
        )
    try:
        output = await http_utils.post(url, payload, headers=headers)
    except (asyncio.CancelledError, httpx.TimeoutException):
        await _abort_sglang_request(args, rid)
        raise
    meta = output.get("meta_info") or {}
    token_logprobs = meta.get("output_token_logprobs") or []
    output_ids = [x[1] for x in token_logprobs]
    top_p_data = _extract_rollout_top_p_token_data(meta, expected_num_tokens=len(output_ids))
    finish_reason = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
    result = {
        "text": output.get("text") or "",
        "output_ids": output_ids,
        "output_logprobs": [float(x[0]) for x in token_logprobs],
        "finish_reason": finish_reason,
    }
    if top_p_data is not None:
        result["rollout_top_p_token_ids"], result["rollout_top_p_token_offsets"] = top_p_data
    if should_log:
        logger.info(
            "fused-agent received SGLang generate response rid=%s output_tokens=%d finish_reason=%s elapsed=%.2fs",
            rid,
            len(output_ids),
            finish_reason,
            time.time() - started,
        )
    return result


async def _abort_sglang_request(args, rid: str) -> None:
    client = http_utils._http_client
    if client is None:
        return
    try:
        await client.post(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/abort_request",
            json={"rid": rid},
            timeout=5.0,
        )
    except Exception:
        pass
