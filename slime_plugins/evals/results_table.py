from __future__ import annotations

import logging
import re
from typing import Any

from slime.ray.rollout import _compute_eval_source_metrics, _eval_dataset_group_size
from slime.utils.metric_utils import compute_pass_at_k_and_pass_all

logger = logging.getLogger(__name__)

_BENCHMARK_COLUMN = ("Benchmark", 21, "<")
_METRIC_COLUMN_WIDTH = 15


def log_eval_results_table(rollout_id: int, args: Any, data: dict[str, Any], extra_metrics: dict[str, Any] | None) -> bool:
    table = format_eval_results_table(args, data)
    if table:
        logger.info("eval %s results:\n\n%s\n", rollout_id, table)
    return False


def format_eval_results_table(args: Any, data: dict[str, Any]) -> str:
    rows: list[tuple[str, dict[str, float]]] = []
    for dataset_name, dataset_data in data.items():
        rewards = dataset_data["rewards"]
        group_size = _eval_dataset_group_size(args, dataset_name)
        dataset_metrics = compute_pass_at_k_and_pass_all(flat_rewards=rewards, group_size=group_size)
        source_metrics = {}
        if (samples := dataset_data.get("samples")) is not None:
            source_metrics = _compute_eval_source_metrics(samples, rewards, group_size)

        overall_name = f"overall / {dataset_name}" if source_metrics else dataset_name
        if overall_row := _result_row(overall_name, dataset_metrics):
            rows.append(overall_row)
        rows.extend(_source_rows(source_metrics))

    if not rows:
        return ""
    return _render_table(rows)


def _source_rows(metrics: dict[str, float]) -> list[tuple[str, dict[str, float]]]:
    source_names: list[str] = []
    for key in metrics:
        parts = key.split("/")
        if len(parts) == 3 and parts[1] == "pass@1" and parts[2] == "mean":
            source_names.append(parts[0])
    return [row for source in source_names if (row := _result_row(source, metrics, prefix=f"{source}/"))]


def _result_row(
    name: str,
    metrics: dict[str, float],
    prefix: str = "",
) -> tuple[str, dict[str, float]] | None:
    row_metrics = {
        key.removeprefix(prefix): float(value)
        for key, value in metrics.items()
        if key.startswith(prefix) and _is_table_metric_key(key.removeprefix(prefix))
    }
    if not row_metrics:
        return None
    return (name, row_metrics)


def _render_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    metric_keys = _table_metric_keys(rows)
    columns = (_BENCHMARK_COLUMN, *[(f"{key.replace('/', ' ')} (%)", _METRIC_COLUMN_WIDTH, ">") for key in metric_keys])
    header = _render_cells(tuple(column[0] for column in columns), columns)
    heavy_rule = "  ".join("━" * width for _, width, _ in columns)
    light_rule = "  ".join("─" * width for _, width, _ in columns)
    lines = [header, heavy_rule]
    for index, row in enumerate(rows):
        name, metrics = row
        values = tuple(_format_metric(metrics[key]) if key in metrics else "" for key in metric_keys)
        lines.append(_render_cells((name, *values), columns))
        if index != len(rows) - 1:
            lines.append(light_rule)
    return "\n".join(lines)


def _render_cells(values: tuple[str, ...], columns: tuple[tuple[str, int, str], ...]) -> str:
    cells = []
    for value, (_, width, align) in zip(values, columns, strict=True):
        cells.append(f"{value:{align}{width}}")
    return "  ".join(cells)


def _format_metric(value: float) -> str:
    return f"{value * 100:.1f}"


def _is_table_metric_key(key: str) -> bool:
    return re.fullmatch(r"pass[@^]\d+/(mean|std)", key) is not None


def _table_metric_keys(rows: list[tuple[str, dict[str, float]]]) -> list[str]:
    keys = {key for _, metrics in rows for key in metrics}
    return sorted(keys, key=_metric_key_sort_key)


def _metric_key_sort_key(key: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"pass([@^])(\d+)/(mean|std)", key)
    if match is None:
        return (2, 0, 0)
    symbol, k, stat = match.groups()
    symbol_order = 0 if symbol == "@" else 1
    stat_order = 0 if stat == "mean" else 1
    return (symbol_order, int(k), stat_order)
