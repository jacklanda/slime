import logging
from types import SimpleNamespace

import numpy as np

from slime.utils.metric_utils import format_metrics_for_display
from slime.utils.train_metric_utils import log_perf_data_raw


def test_format_metrics_for_display_rounds_floats_without_mutating_source():
    metrics = {
        "python_float": 1.234567,
        "numpy_float": np.float32(2.34567),
        "whole_float": 6.0,
        "integer": 3,
        "boolean": True,
        "nested": [4.56789, (5.67891,)],
    }

    display = format_metrics_for_display(metrics)

    assert display == {
        "python_float": 1.23,
        "numpy_float": 2.35,
        "whole_float": 6.0,
        "integer": 3,
        "boolean": True,
        "nested": [4.57, (5.68,)],
    }
    assert metrics["python_float"] == 1.234567
    assert metrics["nested"] == [4.56789, (5.67891,)]
    assert repr(display["whole_float"]) == "6.00"


def test_perf_terminal_log_rounds_without_reducing_tracking_precision(monkeypatch, caplog):
    class FakeTimer:
        seq_lens = []

        def log_dict(self):
            return {"train": 1.234567}

        def reset(self):
            pass

    tracked = {}
    monkeypatch.setattr("slime.utils.train_metric_utils.Timer", FakeTimer)
    monkeypatch.setattr("slime.utils.train_metric_utils.compute_rollout_step", lambda args, rollout_id: 7)
    monkeypatch.setattr(
        "slime.utils.train_metric_utils.logging_utils.log",
        lambda args, metrics, **kwargs: tracked.update(metrics),
    )

    with caplog.at_level(logging.INFO, logger="slime.utils.train_metric_utils"):
        log_perf_data_raw(7, SimpleNamespace(), True, None)

    assert "'perf/trainer_compute_time': 1.23" in caplog.text
    assert tracked["perf/trainer_compute_time"] == 1.234567
