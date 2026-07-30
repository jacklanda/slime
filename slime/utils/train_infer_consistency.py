from collections.abc import Callable, Sequence

import torch


MISMATCH_BUCKET_DIMENSIONS = (
    (0, ("webqa", "mcp", "cli", "other")),
    (1, ("single_segment", "multi_segment")),
    (2, ("short", "medium", "long")),
)


def validate_rollout_weight_versions(expected: object, engine_versions: Sequence[object | None]) -> None:
    """Require every reporting SGLang engine rank to match the updater version."""
    reported = [(index, version) for index, version in enumerate(engine_versions) if version is not None]
    if not reported:
        raise RuntimeError("No SGLang engine reported a weight version after rollout update")
    mismatches = [(index, version) for index, version in reported if str(version) != str(expected)]
    if mismatches:
        raise RuntimeError(
            "Weight version mismatch after rollout update: " f"expected {expected!r}, engines={mismatches!r}"
        )


def mismatch_bucket_contributions(
    abs_logprob_diff: torch.Tensor,
    local_templates: Sequence[torch.Tensor],
    bucket_ids: Sequence[Sequence[int]],
    reducer: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Split a linear mismatch metric into fixed, additive bucket contributions."""
    if len(local_templates) != len(bucket_ids):
        raise ValueError(
            f"mismatch bucket count {len(bucket_ids)} does not match sequence count {len(local_templates)}"
        )

    metrics = {}
    for dimension, names in MISMATCH_BUCKET_DIMENSIONS:
        for bucket_id, name in enumerate(names):
            selector = torch.cat(
                [
                    torch.full_like(template, float(ids[dimension] == bucket_id))
                    for template, ids in zip(local_templates, bucket_ids, strict=True)
                ],
                dim=0,
            )
            metrics[f"train_rollout_logprob_abs_diff/{name}_contribution"] = reducer(
                abs_logprob_diff * selector
            ).clone().detach()
    return metrics
