from __future__ import annotations

import json
import logging
import math
import re
import shutil
import warnings
from functools import lru_cache
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_MASKED_TOKEN_STYLE = "grey58 on grey11"
_EMPTY_THINK_SHELL_STYLE = "bold grey58"
_UNMASKED_TOKEN_STYLE = "bold bright_blue on grey15"
_REWARD_POS_STYLE = "bold black on bright_green"
_REWARD_NEG_STYLE = "bold white on red3"
_NO_CLIP_CHARS = 10**12
_WEB_SEARCH_OBSERVATION_MAX_WORDS = 256


def print_metrics_table(metrics: dict[str, Any], step: int, title: str | None = None) -> None:
    """Print metrics as a Rich table, with a plain-text fallback."""
    rows = [(key, _format_metric_value(value)) for key, value in sorted(metrics.items())]
    table_title = title or f"Step {step}"
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=table_title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=False)
        table.add_column("Value", justify="right", style="green")
        for key, value in rows:
            table.add_row(key, value)
        Console().print(table)
    except ImportError:
        print(f"\n{table_title}", flush=True)
        print("=" * 80, flush=True)
        for key, value in rows:
            print(f"{key:56s} {value:>20s}", flush=True)
        print("=" * 80, flush=True)


def maybe_print_rollout_group(args, group: list[Sample] | list[list[Sample]], *, group_id: int | str | None = None) -> None:
    if not getattr(args, "print_rollout_trajectory", False):
        return
    samples = _flatten_samples(group)
    sample = samples[0] if samples else None
    if sample is None:
        return
    print_rollout_sample(args, sample, group_id=group_id, related_samples=_same_rollout_samples(sample, samples))


def print_rollout_sample(
    args,
    sample: Sample,
    *,
    group_id: int | str | None = None,
    related_samples: list[Sample] | None = None,
) -> None:
    episode = _sample_rllm_episode(sample)
    if episode is not None:
        _print_rllm_episode(args, sample, episode, group_id=group_id, related_samples=related_samples)
        return

    tokenizer = _load_tokenizer_for_visualization(args)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            _print_standard_sample_rich(args, sample, tokenizer, group_id=group_id)
        return
    except ImportError:
        pass

    prompt_text, response_text, masked_action, unmasked_action = _sample_text_parts(sample, tokenizer)
    lines = [
        "",
        "=" * 80,
        _metadata_line(sample, group_id),
        f"reward: {_format_reward_value(sample.reward)} | status: {sample.status.value}",
        "legend: masked action = loss_mask 0, unmasked action = loss_mask 1",
        "prompt:",
        _clip_text(prompt_text),
        "masked action:",
        _clip_text(masked_action),
        "unmasked action:",
        _clip_text(unmasked_action),
        "full action:",
        _clip_text(response_text),
        "=" * 80,
    ]
    print("\n".join(lines), flush=True)


def _print_standard_sample_rich(args, sample: Sample, tokenizer, *, group_id: int | str | None) -> None:
    from rich.console import Group
    from rich.panel import Panel

    console = _trajectory_console(args)
    prompt_text, response_text, masked_action, unmasked_action = _sample_text_parts(sample, tokenizer)
    max_chars = _trajectory_max_chars(args)
    token_mask_view = _token_mask_text(sample, tokenizer)

    console.print()
    console.rule("Rollout Trajectory", style="cyan")
    console.print(Panel(_metadata_line(sample, group_id), title="Metadata", border_style="cyan"))
    console.print(Panel(_sample_result_text(sample), title="Result", border_style="green"))
    if token_mask_view is not None:
        console.print(Panel(Group(_token_mask_legend(), token_mask_view), title="Token Mask View", border_style="magenta"))
    console.print(_text_panel("Prompt", prompt_text, "blue", max_chars=max_chars))
    console.print(_text_panel("Masked Action", masked_action, "dim", max_chars=max_chars))
    console.print(_text_panel("Unmasked Action", unmasked_action, "bright_blue", max_chars=max_chars))
    console.print(_text_panel("Full Action", response_text, "green", max_chars=max_chars))
    console.rule(style="cyan")
    console.print()


def _print_rllm_episode(
    args,
    sample: Sample,
    episode: dict[str, Any],
    *,
    group_id: int | str | None,
    related_samples: list[Sample] | None = None,
) -> None:
    try:
        from rich.console import Group
        from rich.panel import Panel

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            tokenizer = _load_tokenizer_for_visualization(args)
            console = _trajectory_console(args)
            max_chars = _trajectory_max_chars(args)
            token_mask_view = _token_mask_text(sample, tokenizer, related_samples=related_samples)

            console.print()
            console.rule("Episode Trajectory", style="cyan")
            console.print(Panel(_episode_summary_table(sample, episode, group_id), title="Episode Trajectory", border_style="cyan"))
            console.print(_text_panel("Task", _task_text(episode), "blue", max_chars=max_chars))
            if token_mask_view is not None:
                console.print(Panel(Group(_token_mask_legend(), token_mask_view), title="Token Mask View", border_style="magenta"))

            trajectories = [traj for traj in episode.get("trajectories", []) if isinstance(traj, dict)]
            for traj_idx, trajectory in enumerate(trajectories):
                steps = [step for step in trajectory.get("steps", []) if isinstance(step, dict)]
                title = _trajectory_title(trajectory, traj_idx, len(steps))
                console.rule(title, style="green")
                console.print(Panel(_trajectory_table(trajectory, len(steps)), title="Trajectory Summary", border_style="green"))
                for step_idx, step in enumerate(steps):
                    _print_step(console, step, step_idx, len(steps), max_chars=max_chars)
            console.rule(style="cyan")
            console.print()
    except ImportError:
        print(_plain_rllm_episode(sample, episode, group_id=group_id, max_chars=_trajectory_max_chars(args)), flush=True)


def _episode_summary_table(sample: Sample, episode: dict[str, Any], group_id: int | str | None):
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    for key, value in _episode_summary_rows(sample, episode, group_id):
        table.add_row(key, _reward_text(sample.reward) if key == "reward" else value)
    return table


def _trajectory_table(trajectory: dict[str, Any], num_steps: int):
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    table.add_row("name", str(trajectory.get("name") or "trajectory"))
    if trajectory.get("uid"):
        table.add_row("uid", str(trajectory["uid"]))
    table.add_row("reward", _reward_text(trajectory.get("reward")))
    table.add_row("steps", str(num_steps))
    timing = _dict_or_empty(trajectory.get("timing") or _dict_or_empty(trajectory.get("info")).get("timing"))
    if timing:
        table.add_row("time", _format_timing(timing))
    return table


def _print_step(console, step: dict[str, Any], step_idx: int, num_steps: int, *, max_chars: int) -> None:
    from rich.panel import Panel
    from rich.table import Table

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="cyan", no_wrap=True)
    summary.add_column()
    summary.add_row("reward", _reward_text(step.get("reward")))
    summary.add_row("done", "yes" if step.get("done") else "no")
    timing = _dict_or_empty(_dict_or_empty(step.get("info")).get("timing") or step.get("timing"))
    if timing:
        summary.add_row("time", _format_timing(timing))

    console.print(
        Panel(
            summary,
            title=f"Step {step_idx + 1}/{num_steps}",
            border_style=_step_border_style(step),
            expand=True,
        )
    )

    thought, response = _step_thinking_and_response(step)
    disable_thinking = _step_disables_thinking(step)
    observation = _display_observation(step.get("observation"), max_chars=min(max_chars, 2000))

    if observation:
        console.print(_text_panel("Observation", observation, "dim", max_chars=min(max_chars, 2000)))
    if thought or disable_thinking:
        console.print(_text_panel("Thinking", _strip_think_tags(thought), "yellow", max_chars=min(max_chars, 2000)))
    if response:
        console.print(_text_panel("Action", response, "magenta", max_chars=min(max_chars, 2000)))


def _trajectory_console(args):
    from rich.console import Console

    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    configured_width = int(getattr(args, "print_rollout_trajectory_width", 88) or 88)
    prefix_width = int(getattr(args, "print_rollout_trajectory_prefix_width", 32) or 0)
    available_width = max(20, terminal_width - max(0, prefix_width))
    width = min(available_width, configured_width)
    if width < 20:
        width = 20
    force_terminal = bool(getattr(args, "print_rollout_trajectory_color", True))
    return Console(
        width=width,
        soft_wrap=False,
        force_terminal=force_terminal,
        color_system="truecolor" if force_terminal else None,
        no_color=not force_terminal,
    )


def _text_panel(title: str, value: str, border_style: str, *, max_chars: int):
    from rich.panel import Panel
    from rich.text import Text

    text = Text(_clip_text(value, max_chars), overflow="fold", no_wrap=False)
    return Panel(text, title=title, border_style=border_style, expand=True)


def _token_mask_legend():
    from rich.text import Text

    legend = Text()
    legend.append("legend: ")
    legend.append(" masked/prompt ", style=_MASKED_TOKEN_STYLE)
    legend.append(" = loss_mask 0, ")
    legend.append(" empty think shell ", style=_EMPTY_THINK_SHELL_STYLE)
    legend.append(" = injected prompt prefix, ")
    legend.append(" unmasked ", style=_UNMASKED_TOKEN_STYLE)
    legend.append(" = loss_mask 1, ")
    legend.append(" reward > 0 ", style=_REWARD_POS_STYLE)
    legend.append(" / ")
    legend.append(" reward <= 0 ", style=_REWARD_NEG_STYLE)
    return legend


def _token_mask_text(sample: Sample, tokenizer, *, related_samples: list[Sample] | None = None):
    from rich.text import Text

    if tokenizer is None or not sample.tokens:
        return None

    related_samples = [s for s in (related_samples or []) if isinstance(s, Sample) and s.tokens and s.response_length > 0]
    if related_samples:
        return _combined_token_mask_text(sample, tokenizer, related_samples)

    response_length = int(sample.response_length or 0)
    if response_length <= 0:
        return None

    tokens = list(sample.tokens)
    response_length = min(response_length, len(tokens))
    prompt_ids = tokens[:-response_length]
    response_ids = tokens[-response_length:]
    loss_mask = _normalized_response_loss_mask(sample.loss_mask, response_length)
    full_mask = [0] * len(prompt_ids) + loss_mask
    _promote_assistant_action_blocks(tokens, full_mask, tokenizer)
    shell_positions = _empty_think_shell_token_positions(tokens, response_length, tokenizer)

    rendered = Text(overflow="fold", no_wrap=False)

    reward_style = _reward_style(sample.reward)
    for idx, (token_id, mask) in enumerate(zip(tokens, full_mask, strict=False)):
        piece = _decode_token_piece(tokenizer, token_id)
        style = _UNMASKED_TOKEN_STYLE if mask else _MASKED_TOKEN_STYLE
        if idx in shell_positions:
            style = _EMPTY_THINK_SHELL_STYLE
        if reward_style is not None and idx == len(tokens) - 1:
            style = reward_style
        rendered.append(piece, style=style)
    return rendered


def _combined_token_mask_text(sample: Sample, tokenizer, related_samples: list[Sample]):
    from rich.text import Text

    base_sample = max(related_samples, key=lambda item: len(item.tokens or []), default=sample)
    tokens = list(base_sample.tokens)
    if not tokens:
        return None

    full_mask = [0] * len(tokens)
    search_start = 0
    for item in sorted(related_samples, key=lambda s: (len(s.tokens or []), int(s.response_length or 0))):
        response_length = min(int(item.response_length or 0), len(item.tokens or []))
        if response_length <= 0:
            continue
        response_ids = list(item.tokens[-response_length:])
        loss_mask = _normalized_response_loss_mask(item.loss_mask, response_length)
        for start, end in _unmasked_token_spans(loss_mask):
            span_ids = response_ids[start:end]
            if not span_ids:
                continue
            match = _find_subsequence(tokens, span_ids, start=search_start)
            if match is None:
                match = _find_subsequence(tokens, span_ids, start=0)
            if match is None:
                continue
            for idx in range(match, match + len(span_ids)):
                full_mask[idx] = 1
            search_start = match + len(span_ids)
    _promote_assistant_action_blocks(tokens, full_mask, tokenizer)
    shell_positions = _empty_think_shell_token_positions(tokens, 0, tokenizer)

    rendered = Text(overflow="fold", no_wrap=False)
    reward_style = _reward_style(sample.reward)
    for idx, (token_id, mask) in enumerate(zip(tokens, full_mask, strict=False)):
        piece = _decode_token_piece(tokenizer, token_id)
        style = _UNMASKED_TOKEN_STYLE if mask else _MASKED_TOKEN_STYLE
        if idx in shell_positions:
            style = _EMPTY_THINK_SHELL_STYLE
        if reward_style is not None and idx == len(tokens) - 1:
            style = reward_style
        rendered.append(piece, style=style)
    return rendered


def _empty_think_shell_token_positions(tokens: list[int], response_length: int, tokenizer) -> set[int]:
    pieces = [_decode_token_piece(tokenizer, token_id) for token_id in tokens]
    full_text = "".join(pieces)
    prompt_end = max(0, len(tokens) - max(0, response_length))

    offsets = []
    cursor = 0
    for piece in pieces:
        start = cursor
        cursor += len(piece)
        offsets.append((start, cursor))

    shell_spans = []
    shell_text = "<think>\\n\\n</think>\\n\\n"
    for match in re.finditer(re.escape(shell_text), full_text):
        shell_spans.append(match.span())

    positions = set()
    for span_start, span_end in shell_spans:
        for idx, (token_start, token_end) in enumerate(offsets):
            if idx >= prompt_end:
                break
            if token_start < span_end and token_end > span_start:
                positions.add(idx)
    return positions


def _unmasked_token_spans(loss_mask: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = None
    for idx, value in enumerate(loss_mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(loss_mask)))
    return spans


def _find_subsequence(tokens: list[int], span_ids: list[int], *, start: int) -> int | None:
    if not span_ids or len(span_ids) > len(tokens):
        return None
    first = span_ids[0]
    limit = len(tokens) - len(span_ids) + 1
    for idx in range(max(0, start), limit):
        if tokens[idx] == first and tokens[idx : idx + len(span_ids)] == span_ids:
            return idx
    return None


def _promote_assistant_action_blocks(tokens: list[int], mask: list[int], tokenizer) -> None:
    pieces = [_decode_token_piece(tokenizer, token_id) for token_id in tokens]
    full_text = "".join(pieces)
    if "<tool_call>" not in full_text:
        return

    offsets = []
    cursor = 0
    for piece in pieces:
        start = cursor
        cursor += len(piece)
        offsets.append((start, cursor))

    assistant_spans = _assistant_message_spans(full_text)
    if not assistant_spans:
        return

    for action_match in re.finditer(r"<tool_call>.*?</tool_call>", full_text, flags=re.DOTALL):
        action_start, action_end = action_match.span()
        if not any(span_start <= action_start < action_end <= span_end for span_start, span_end in assistant_spans):
            continue
        for idx, (token_start, token_end) in enumerate(offsets):
            if token_start < action_end and token_end > action_start:
                mask[idx] = 1


def _assistant_message_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    header = "<|im_start|>assistant\\n"
    end_marker = "<|im_end|>"
    cursor = 0
    while True:
        start = text.find(header, cursor)
        if start < 0:
            break
        content_start = start + len(header)
        end = text.find(end_marker, content_start)
        if end < 0:
            end = len(text)
        spans.append((content_start, end))
        cursor = end + len(end_marker)
    return spans


def _sample_result_text(sample: Sample):
    from rich.text import Text

    text = Text()
    text.append("reward: ")
    text.append_text(_reward_text(sample.reward))
    text.append(" | status: ")
    text.append(sample.status.value, style="bold cyan")
    return text


def _reward_text(value: Any):
    from rich.text import Text

    style = _reward_style(value)
    return Text(_format_reward_value(value), style=style or "bold")


def _format_reward_value(value: Any) -> str:
    reward = _reward_float(value)
    if reward is None:
        return _format_metric_value(value)
    return f"{reward:.1f}"


def _reward_style(value: Any) -> str | None:
    reward = _reward_float(value)
    if reward is None:
        return None
    return _REWARD_POS_STYLE if reward > 0 else _REWARD_NEG_STYLE


def _reward_float(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("reward", "score", "raw_reward"):
            if key in value:
                return _reward_float(value[key])
        return None
    try:
        reward = float(value)
    except (TypeError, ValueError):
        return None
    return reward if math.isfinite(reward) else None


def _normalized_response_loss_mask(loss_mask: list[int] | None, response_length: int) -> list[int]:
    if loss_mask is None:
        return [1] * response_length
    normalized = list(loss_mask)
    if len(normalized) < response_length:
        normalized = [0] * (response_length - len(normalized)) + normalized
    return normalized[-response_length:]


def _decode_token_piece(tokenizer, token_id: int) -> str:
    try:
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        piece = tokenizer.decode([token_id])
    return _escape_token_piece(piece)


def _escape_token_piece(piece: str) -> str:
    return piece.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6f}" if abs(value) < 1000 else f"{value:.2f}"
        return str(value)
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "None"
    return str(value)


def _load_tokenizer_for_visualization(args):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            return _cached_load_tokenizer(args.hf_checkpoint)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to load tokenizer for rollout trajectory visualization; falling back to raw text")
        return None


@lru_cache(maxsize=4)
def _cached_load_tokenizer(hf_checkpoint: str):
    from slime.utils.processing_utils import load_tokenizer

    return load_tokenizer(hf_checkpoint, trust_remote_code=True)


def _flatten_samples(group: list[Sample] | list[list[Sample]]) -> list[Sample]:
    samples = []
    stack: list[Any] = list(group)
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack[:0] = item
        elif isinstance(item, Sample):
            samples.append(item)
    return samples


def _first_sample(group: list[Sample] | list[list[Sample]]) -> Sample | None:
    samples = _flatten_samples(group)
    return samples[0] if samples else None


def _same_rollout_samples(sample: Sample, samples: list[Sample]) -> list[Sample]:
    rollout_id = sample.rollout_id
    if rollout_id is None:
        return [sample]
    return [item for item in samples if item.rollout_id == rollout_id] or [sample]


def _sample_text_parts(sample: Sample, tokenizer) -> tuple[str, str, str, str]:
    if tokenizer is None or not sample.tokens or sample.response_length <= 0:
        response_text = sample.response or ""
        return str(sample.prompt), response_text, "", response_text

    response_ids = sample.tokens[-sample.response_length :]
    prompt_ids = sample.tokens[: -sample.response_length]
    loss_mask = sample.loss_mask or [1] * len(response_ids)
    if len(loss_mask) < len(response_ids):
        loss_mask = [0] * (len(response_ids) - len(loss_mask)) + list(loss_mask)

    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    masked_ids = [token_id for token_id, mask in zip(response_ids, loss_mask, strict=False) if not mask]
    unmasked_ids = [token_id for token_id, mask in zip(response_ids, loss_mask, strict=False) if mask]
    masked_action = tokenizer.decode(masked_ids, skip_special_tokens=False) if masked_ids else ""
    unmasked_action = tokenizer.decode(unmasked_ids, skip_special_tokens=False) if unmasked_ids else ""
    return prompt_text, response_text, masked_action, unmasked_action


def _metadata_line(sample: Sample, group_id: int | str | None) -> str:
    parts = []
    if group_id is not None:
        parts.append(f"group={group_id}")
    for name in ("index", "group_index", "rollout_id"):
        value = getattr(sample, name, None)
        if value is not None:
            parts.append(f"{name}={value}")
    metadata = sample.metadata or {}
    for key in ("fused_task_type", "fused_termination", "data_source"):
        if key in metadata:
            parts.append(f"{key}={metadata[key]}")
    return " | ".join(parts) or "rollout trajectory"


def _sample_rllm_episode(sample: Sample) -> dict[str, Any] | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    episode = metadata.get("rllm_episode")
    return episode if isinstance(episode, dict) else None


def _episode_summary_rows(sample: Sample, episode: dict[str, Any], group_id: int | str | None) -> list[tuple[str, str]]:
    rows = []
    if group_id is not None:
        rows.append(("group", str(group_id)))
    rows.extend(
        (name, str(value))
        for name in ("index", "group_index", "rollout_id")
        if (value := getattr(sample, name, None)) is not None
    )
    task = _dict_or_empty(episode.get("task"))
    metrics = _dict_or_empty(episode.get("metrics"))
    rows.extend(
        [
            ("episode_id", str(episode.get("id") or "n/a")),
            ("session_id", str(episode.get("session_id") or "n/a")),
            ("source", _episode_source(sample, task)),
            ("reward", _format_reward_value(sample.reward)),
            ("is_correct", "yes" if episode.get("is_correct") else "no"),
            ("termination", str(episode.get("termination_reason") or "unknown")),
            ("trajectories", str(len(episode.get("trajectories") or []))),
        ]
    )
    for key in ("traj/steps", "turn/tool_call_turn", "default_traj_name_acc"):
        if key in metrics:
            rows.append((key, _format_metric_value(metrics[key])))
    return rows


def _episode_source(sample: Sample, task: dict[str, Any]) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(
        metadata.get("fused_task_type")
        or task.get("data_source")
        or task.get("task_type")
        or metadata.get("data_source")
        or "unknown"
    )


def _task_text(episode: dict[str, Any]) -> str:
    task = _dict_or_empty(episode.get("task"))
    if not task:
        return "No task metadata."

    preferred_keys = (
        "question",
        "problem",
        "prompt",
        "instruction",
        "ground_truth",
        "answer",
        "data_source",
        "difficulty",
        "split",
        "id",
        "uid",
    )
    lines = []
    used = set()
    for key in preferred_keys:
        if key in task:
            used.add(key)
            lines.append(f"{key}: {_clip_text(_stringify(task[key]), 2000)}")
    extras = {key: value for key, value in task.items() if key not in used}
    if extras:
        lines.append("extra:")
        lines.append(_clip_text(json.dumps(extras, indent=2, ensure_ascii=False, default=str), 2000))
    return "\n".join(lines)


def _trajectory_title(trajectory: dict[str, Any], traj_idx: int, num_steps: int) -> str:
    name = trajectory.get("name") or f"trajectory-{traj_idx}"
    return f"{name} | steps={num_steps} | reward={_format_reward_value(trajectory.get('reward'))}"


def _step_border_style(step: dict[str, Any]) -> str:
    if step.get("done"):
        return "green" if float(step.get("reward") or 0.0) > 0 else "red"
    return "yellow"


def _plain_rllm_episode(
    sample: Sample,
    episode: dict[str, Any],
    *,
    group_id: int | str | None,
    max_chars: int,
) -> str:
    lines = ["", "=" * 100, "Episode Trajectory"]
    lines.extend(f"{key}: {value}" for key, value in _episode_summary_rows(sample, episode, group_id))
    lines.extend(["", "Task:", _clip_text(_task_text(episode), max_chars)])
    for traj_idx, trajectory in enumerate(episode.get("trajectories") or []):
        if not isinstance(trajectory, dict):
            continue
        steps = [step for step in trajectory.get("steps", []) if isinstance(step, dict)]
        lines.extend(["", "-" * 100, _trajectory_title(trajectory, traj_idx, len(steps))])
        for step_idx, step in enumerate(steps):
            thought, response = _step_thinking_and_response(step)
            lines.extend(
                [
                    "",
                    f"Step {step_idx + 1}/{len(steps)} | reward={_format_reward_value(step.get('reward'))} | done={bool(step.get('done'))}",
                    "Observation:",
                    _clip_text(_display_observation(step.get("observation"), max_chars=max_chars), max_chars),
                    "Thinking:",
                    _clip_text(_strip_think_tags(thought), max_chars),
                    "Action:",
                    _clip_text(response, max_chars),
                ]
            )
    lines.append("=" * 100)
    return "\n".join(lines)


def _step_thinking_and_response(step: dict[str, Any]) -> tuple[str, str]:
    model_response = str(step.get("model_response") or "")
    thought, response = _extract_thinking_and_response(model_response)
    if thought or "<think>" in model_response:
        return thought, response
    if _step_disables_thinking(step):
        return "", response
    return str(step.get("thought") or ""), response


def _step_disables_thinking(step: dict[str, Any]) -> bool:
    info = _dict_or_empty(step.get("info"))
    return bool(step.get("disable_thinking") or info.get("disable_thinking"))


def _extract_thinking_and_response(model_response: str) -> tuple[str, str]:
    if not model_response:
        return "", ""
    match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
    if not match:
        return "", model_response.strip()
    return match.group(1).strip(), model_response.split("</think>", 1)[-1].strip()


def _strip_think_tags(text: str) -> str:
    text = text.strip()
    if text.startswith("<think>") and text.endswith("</think>"):
        return text[len("<think>") : -len("</think>")].strip()
    return text


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _display_observation(value: Any, *, max_chars: int) -> str:
    text = _stringify(value)
    if not text:
        return ""
    formatted = _format_tool_response_for_display(text, max_chars=max_chars)
    if _is_web_search_tool_response(text):
        return formatted
    return _clip_text(formatted, max_chars)


def _format_tool_response_for_display(text: str, *, max_chars: int) -> str:
    match = re.fullmatch(
        r"\s*<tool_response>\s*\n(?P<header>Execution output of \[(?P<tool>[^\]]+)\]:)\s*\n(?P<body>.*?)\n?</tool_response>\s*",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return _format_json_payload_for_display(text, max_chars=max_chars) or text

    body = match.group("body").strip()
    tool_name = match.group("tool").strip().lower()
    if tool_name in {"web_search", "search", "webqa"}:
        compact_body = _format_search_payload_for_display(body, max_chars=_NO_CLIP_CHARS) or _strip_urls(body)
        compact_body = _limit_search_words(compact_body)
    else:
        compact_body = _format_json_payload_for_display(body, max_chars=max_chars) or body
    return f"<tool_response>\n{match.group('header')}\n{compact_body}\n</tool_response>"


def _is_web_search_tool_response(text: str) -> bool:
    match = re.match(r"\s*<tool_response>\s*\nExecution output of \[(?P<tool>[^\]]+)\]:", text, flags=re.DOTALL)
    return bool(match and match.group("tool").strip().lower() in {"web_search", "search", "webqa"})


def _format_search_payload_for_display(text: str, *, max_chars: int) -> str | None:
    compact = _format_search_json_payload_for_display(text, max_chars=max_chars)
    if compact:
        return compact

    blocks = _extract_search_blocks_from_text(text)
    if blocks:
        return _clip_text("\n".join(blocks), max_chars)

    stripped = text.strip()
    if stripped[:1] in {"[", "{"} and ('"content"' in stripped or '"chunk_text"' in stripped):
        cleaned = _strip_json_search_noise(text)
    else:
        cleaned = text
    cleaned = _strip_urls(cleaned)
    cleaned = _compact_search_text(cleaned)
    return _clip_text(cleaned, max_chars) if cleaned else None


def _format_search_json_payload_for_display(text: str, *, max_chars: int) -> str | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None

    rows = _search_payload_rows(payload)
    if not rows:
        return _clip_text(_plain_search_text(payload), max_chars)
    blocks = [_format_json_observation_item(item, idx) for idx, item in enumerate(rows, start=1)]
    blocks = [block for block in blocks if block]
    return _clip_text("\n".join(blocks), max_chars) if blocks else None


def _search_payload_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("result", "results", "data", "documents", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return []


def _format_json_payload_for_display(text: str, *, max_chars: int) -> str | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None

    if isinstance(payload, list):
        blocks = [_format_json_observation_item(item, idx) for idx, item in enumerate(payload, start=1)]
        blocks = [block for block in blocks if block]
        if blocks:
            return _clip_text("\n".join(blocks), max_chars)
    if isinstance(payload, dict):
        block = _format_json_observation_item(payload, 1)
        if block:
            return _clip_text(block, max_chars)
    return None


def _extract_search_blocks_from_text(text: str) -> list[str]:
    title_matches = list(re.finditer(r'"title"\s*:\s*"(?P<title>(?:\\.|[^"\\])*)"', text, flags=re.DOTALL))
    if not title_matches:
        return []

    blocks = []
    for idx, match in enumerate(title_matches, start=1):
        start = match.end()
        end = title_matches[idx].start() if idx < len(title_matches) else len(text)
        segment = text[start:end]
        body = _extract_json_string_field(segment, "chunk_text")
        if body is None:
            body = _extract_json_string_field(segment, "summary")
        if body is None:
            body = _extract_json_string_field(segment, "snippet")
        if body is None:
            body = _extract_json_string_field(segment, "text")

        title = _decode_jsonish_string(match.group("title"))
        blocks.append(_format_search_result_line(idx, title, body))
    return blocks


def _extract_json_string_field(text: str, field: str) -> str | None:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if not match:
        return None
    return _decode_jsonish_string(match.group("value"))


def _decode_jsonish_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value


def _format_json_observation_item(item: Any, idx: int) -> str:
    if not isinstance(item, dict):
        return _clip_text(_stringify(item), 600)

    content = item.get("content")
    if isinstance(content, dict):
        source = content
    else:
        source = item

    title = source.get("title") or source.get("name") or source.get("id") or item.get("id")
    url = source.get("url") or source.get("link")
    body = (
        source.get("chunk_text")
        or source.get("summary")
        or source.get("snippet")
        or source.get("text")
        or source.get("content")
        or item.get("text")
        or item.get("output")
    )

    lines = []
    if title is not None:
        lines.append(f"[{idx}] {title}")
    elif url is not None:
        lines.append(f"[{idx}]")
    else:
        lines.append(f"[{idx}]")
    if body is not None:
        return _format_search_result_line(idx, title, _stringify(body))

    extras = []
    for key in ("score", "source", "published", "date"):
        if key in source:
            extras.append(f"{key}={source[key]}")
    if extras:
        lines.append(" | ".join(extras))
    return "\n".join(lines)


def _format_search_result_line(idx: int, title: Any, body: Any) -> str:
    title_text = _compact_search_text(_stringify(title)) if title is not None else ""
    body_text = _compact_search_text(_stringify(body)) if body is not None else ""
    if title_text and body_text:
        return f"[{idx}] {title_text}: {body_text}"
    if title_text:
        return f"[{idx}] {title_text}"
    if body_text:
        return f"[{idx}] {body_text}"
    return f"[{idx}]"


def _limit_search_words(text: str, max_words: int = _WEB_SEARCH_OBSERVATION_MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _clean_observation_text(text: str) -> str:
    text = text.replace("\\n", "\n").replace("\\r", "\n").replace("\\t", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _compact_search_text(text: str) -> str:
    text = text.replace("\\n", "\n").replace("\\r", "\n").replace("\\t", " ")
    return " ".join(text.split())


def _strip_urls(text: str) -> str:
    text = re.sub(r'https?://\S+', "", text)
    text = re.sub(r'"url"\s*:\s*"[^"]*",?\s*', "", text)
    text = re.sub(r'"link"\s*:\s*"[^"]*",?\s*', "", text)
    return text


def _strip_json_search_noise(text: str) -> str:
    text = re.sub(r'^\s*\[\s*', "", text)
    text = re.sub(r'\s*\]\s*$', "", text)
    text = re.sub(r'^\s*\{\s*', "", text)
    text = re.sub(r'\s*\}\s*$', "", text)
    text = re.sub(r'"id"\s*:\s*[^,\n]+,?\s*', "", text)
    text = re.sub(r'"content"\s*:\s*\{', "", text)
    text = re.sub(r'"(?:title|chunk_text|summary|snippet|text)"\s*:\s*', "", text)
    text = text.replace('",', "\n").replace('"', "")
    text = re.sub(r"^[,\s{}]+|[,\s{}]+$", "", text)
    return text


def _plain_search_text(value: Any) -> str:
    parts: list[str] = []

    def collect(item: Any, key: str = "") -> None:
        normalized_key = key.lower()
        if normalized_key in {
            "url",
            "link",
            "id",
            "lexical_rank",
            "lexical_score",
            "fusion_score",
            "title_overlap_boost",
            "fusion_profile",
            "query_class",
            "retrieval_channels",
        }:
            return
        if isinstance(item, dict):
            for child_key, child_value in item.items():
                collect(child_value, str(child_key))
            return
        if isinstance(item, list):
            for child in item:
                collect(child, key)
            return
        if item is not None:
            text = _compact_search_text(str(item))
            if text:
                parts.append(text)

    collect(value)
    return " ".join(parts)


def _format_timing(timing: dict[str, Any]) -> str:
    parts = []
    for key in ("llm_time", "env_time", "reward_time", "total_time"):
        if key in timing:
            parts.append(f"{key}={_format_metric_value(timing[key])}s")
    if not parts:
        for key in ("start_timestamp", "end_timestamp"):
            if key in timing:
                parts.append(f"{key}={timing[key]}")
    return ", ".join(parts) if parts else "n/a"


def _trajectory_max_chars(args) -> int:
    return int(getattr(args, "print_rollout_trajectory_max_chars", 4096))


def _clip_text(value: str, max_chars: int = 4096) -> str:
    value = value.replace("\r", "\\r")
    if len(value) <= max_chars:
        return value
    head = max_chars // 2
    tail = max_chars - head
    return f"{value[:head]}\n... <skipped {len(value) - max_chars} chars> ...\n{value[-tail:]}"
