from types import SimpleNamespace

from slime.ray.rollout import compute_metrics_from_samples
from slime.utils.types import Sample


def test_compute_metrics_from_samples_omits_response_aborted_ratio():
    args = SimpleNamespace(
        advantage_estimator="ppo",
        log_reward_category=None,
        rollout_max_response_len=8,
        rollout_max_prompt_len=16,
    )
    samples = [
        Sample(
            index=0,
            tokens=[1, 2, 3, 4],
            response_length=2,
            reward=0.0,
            response="ok",
            metadata={},
        ),
        Sample(
            index=1,
            tokens=[5, 6, 7, 8, 9],
            response_length=3,
            reward=0.0,
            response="done",
            metadata={},
            status=Sample.Status.ABORTED,
        ),
    ]

    metrics = compute_metrics_from_samples(args, samples)

    assert "response/aborted_ratio" not in metrics
    assert "response_length/mean" in metrics
    assert "prompt_length/mean" in metrics
