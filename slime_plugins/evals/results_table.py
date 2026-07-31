from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from slime.ray.rollout import (
    _compute_eval_source_metrics,
    _eval_dataset_group_size,
    _eval_source_from_sample,
    _eval_sample_steps,
    _eval_sample_tool_calls,
)
from slime.utils.metric_utils import compute_pass_at_k_and_pass_all
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_BENCHMARK_COLUMN = ("Benchmark", 21, "<")
_METRIC_COLUMN_WIDTH = 15
_COUNT_COLUMNS = (("# steps", "steps", 10, ">"), ("# tool calls", "tool_calls", 12, ">"))
_TERMINATION_COLUMNS = (
    ("# abnormal / all", "abnormal", 16, ">"),
    ("# max turns / all", "max_turns", 17, ">"),
    ("# clip / all", "clip", 12, ">"),
)


def log_eval_results_table(rollout_id: int, args: Any, data: dict[str, Any], extra_metrics: dict[str, Any] | None) -> bool:
    table = format_eval_results_table(args, data)
    if table:
        logger.info("eval %s results:\n\n%s\n", rollout_id, table)
    return False


def format_eval_results_table(args: Any, data: dict[str, Any]) -> str:
    rows: list[tuple[str, dict[str, float | str]]] = []
    for dataset_name, dataset_data in data.items():
        rewards = dataset_data["rewards"]
        group_size = _eval_dataset_group_size(args, dataset_name)
        dataset_metrics = compute_pass_at_k_and_pass_all(flat_rewards=rewards, group_size=group_size)
        source_metrics: dict[str, float | str] = {}
        if (samples := dataset_data.get("samples")) is not None:
            source_metrics = _compute_eval_source_metrics(samples, rewards, group_size)
            dataset_metrics.update(_format_termination_counts(samples, group_size))
            source_metrics.update(_source_termination_counts(samples, group_size))
            steps = [value for sample in samples if (value := _eval_sample_steps(sample)) is not None]
            if steps:
                dataset_metrics["steps"] = float(np.mean(steps))
            tool_calls = [value for sample in samples if (value := _eval_sample_tool_calls(sample)) is not None]
            if tool_calls:
                dataset_metrics["tool_calls"] = float(np.mean(tool_calls))

        overall_name = f"overall / {dataset_name}" if source_metrics else dataset_name
        if overall_row := _result_row(overall_name, dataset_metrics):
            rows.append(overall_row)
        rows.extend(_source_rows(source_metrics))

    if not rows:
        return ""
    return _render_table(rows)


def _source_rows(metrics: dict[str, float | str]) -> list[tuple[str, dict[str, float | str]]]:
    source_names: list[str] = []
    for key in metrics:
        parts = key.split("/")
        if len(parts) == 3 and parts[1] == "pass@1" and parts[2] == "mean":
            source_names.append(parts[0])
    return [row for source in source_names if (row := _result_row(source, metrics, prefix=f"{source}/"))]


def _result_row(
    name: str,
    metrics: dict[str, float | str],
    prefix: str = "",
) -> tuple[str, dict[str, float | str]] | None:
    row_metrics = {
        key.removeprefix(prefix): value
        if key.removeprefix(prefix) in {column[1] for column in _TERMINATION_COLUMNS}
        else float(value)
        for key, value in metrics.items()
        if key.startswith(prefix) and _is_table_metric_key(key.removeprefix(prefix))
    }
    if not row_metrics:
        return None
    return (name, row_metrics)


def _render_table(rows: list[tuple[str, dict[str, float | str]]]) -> str:
    metric_keys = _table_metric_keys(rows)
    columns = (
        _BENCHMARK_COLUMN,
        *[(f"{key.replace('/', ' ')} (%)", _METRIC_COLUMN_WIDTH, ">") for key in metric_keys],
        *[(header, width, align) for header, _, width, align in _COUNT_COLUMNS],
        *[(header, width, align) for header, _, width, align in _TERMINATION_COLUMNS],
    )
    header = _render_cells(tuple(column[0] for column in columns), columns)
    heavy_rule = "  ".join("━" * width for _, width, _ in columns)
    light_rule = "  ".join("─" * width for _, width, _ in columns)
    lines = [header, heavy_rule]
    for index, row in enumerate(rows):
        name, metrics = row
        values = tuple(_format_metric(metrics[key]) if key in metrics else "" for key in metric_keys)
        counts = tuple(_format_count(metrics[key]) if key in metrics else "" for _, key, _, _ in _COUNT_COLUMNS)
        terminations = tuple(str(metrics.get(key, "")) for _, key, _, _ in _TERMINATION_COLUMNS)
        lines.append(_render_cells((name, *values, *counts, *terminations), columns))
        if index != len(rows) - 1:
            lines.append(light_rule)
    return "\n".join(lines)


def _render_cells(values: tuple[str, ...], columns: tuple[tuple[str, int, str], ...]) -> str:
    cells = []
    for value, (_, width, align) in zip(values, columns, strict=True):
        cells.append(f"{value:{align}{width}}")
    return "  ".join(cells)


def _format_metric(value: float | str) -> str:
    return f"{float(value) * 100:.1f}"


def _format_count(value: float | str) -> str:
    return f"{float(value):.1f}"


def _is_table_metric_key(key: str) -> bool:
    return re.fullmatch(r"pass[@^]\d+/(mean|std)", key) is not None or key in {
        "steps",
        "tool_calls",
        "abnormal",
        "max_turns",
        "clip",
    }


def _table_metric_keys(rows: list[tuple[str, dict[str, float | str]]]) -> list[str]:
    non_metric_keys = {"steps", "tool_calls", *(column[1] for column in _TERMINATION_COLUMNS)}
    keys = {key for _, metrics in rows for key in metrics if key not in non_metric_keys}
    return sorted(keys, key=_metric_key_sort_key)


def _metric_key_sort_key(key: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"pass([@^])(\d+)/(mean|std)", key)
    if match is None:
        return (2, 0, 0)
    symbol, k, stat = match.groups()
    symbol_order = 0 if symbol == "@" else 1
    stat_order = 0 if stat == "mean" else 1
    return (symbol_order, int(k), stat_order)


def _format_termination_counts(samples: list[Any], group_size: int) -> dict[str, str]:
    group_size = max(1, group_size)
    groups = [samples[start : start + group_size] for start in range(0, len(samples), group_size)]
    abnormal = sum(any(_eval_sample_termination(sample).startswith("abnormal_") for sample in group) for group in groups)
    max_turns = sum(any(_eval_sample_termination(sample) == "max_turns_exceeded" for sample in group) for group in groups)
    clipped = sum(sample.status == Sample.Status.TRUNCATED for sample in samples)
    return {
        "abnormal": f"{abnormal} / {len(groups)}",
        "max_turns": f"{max_turns} / {len(groups)}",
        "clip": f"{clipped} / {len(samples)}",
    }


def _source_termination_counts(samples: list[Any], group_size: int) -> dict[str, str]:
    samples_by_source: dict[str, list[Any]] = {}
    for sample in samples:
        if source := _eval_source_from_sample(sample):
            samples_by_source.setdefault(source, []).append(sample)
    counts = {}
    for source, source_samples in samples_by_source.items():
        counts.update(
            {f"{source}/{key}": value for key, value in _format_termination_counts(source_samples, group_size).items()}
        )
    return counts


def _eval_sample_termination(sample: Any) -> str:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    reason = metadata.get("fused_termination") or metadata.get("termination_reason")
    return str(reason).strip().lower() if reason is not None else ""
