"""Tests for the vanilla (non-segment) GRPO group reward normalization path.

This path had no coverage: every existing ``rewards_normalization=True`` test
routes through the prompt-equal / multi-segment branch of
``_post_process_rewards`` instead.
"""

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from slime.ray.rollout import _group_normalize_rewards, _post_process_rewards
from slime.utils.types import Sample


NUM_GPUS = 0


def _args(**overrides):
    values = dict(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=4,
        rollout_batch_size=3,
        reward_key=None,
    )
    values.update(overrides)
    return Namespace(**values)


def _samples(group_ids, rewards):
    return [Sample(group_index=group_id, index=i, rollout_id=i, reward=reward) for i, (group_id, reward) in enumerate(zip(group_ids, rewards, strict=True))]


def _legacy_positional_impl(args, raw_rewards):
    """The implementation this path used before grouping by ``group_index``."""
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
        rewards = rewards.reshape(-1, args.n_samples_per_prompt)
    else:
        rewards = rewards.view(-1, rewards.shape[-1])
    rewards = rewards - rewards.mean(dim=-1, keepdim=True)
    if args.advantage_estimator in ["grpo", "gspo", "cispo"] and args.grpo_std_normalization:
        rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)
    return rewards.flatten().tolist()


@pytest.mark.parametrize("use_std", [True, False])
@pytest.mark.parametrize(("n_per_prompt", "batch_size"), [(32, 8), (4, 3), (2, 5), (8, 1)])
def test_group_index_grouping_matches_legacy_positional_reshape(use_std, n_per_prompt, batch_size):
    """Bit-exact equivalence whenever the old positional layout was valid.

    The data source emits contiguous groups, so grouping by ``group_index``
    must reproduce the positional reshape exactly.
    """
    args = _args(
        n_samples_per_prompt=n_per_prompt,
        rollout_batch_size=batch_size,
        grpo_std_normalization=use_std,
    )
    n = n_per_prompt * batch_size
    group_ids = [i // n_per_prompt for i in range(n)]
    raw = [float(i % 3 == 0) if i % 7 else 0.35 for i in range(n)]

    result = _group_normalize_rewards(args, _samples(group_ids, raw), raw)

    torch.testing.assert_close(
        torch.tensor(result),
        torch.tensor(_legacy_positional_impl(args, raw)),
        rtol=0,
        atol=0,
    )


def test_uneven_group_sizes_normalize_per_group_not_whole_batch():
    """Regression: the legacy fallback collapsed every group into one.

    ``view(-1, N)`` on a 1-D tensor yields shape ``[1, N]``, so a sample count
    that was not exactly ``n_samples_per_prompt * rollout_batch_size`` silently
    turned group norm into batch norm.
    """
    args = _args(n_samples_per_prompt=4, rollout_batch_size=3)
    group_ids = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    raw = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert len(raw) != args.n_samples_per_prompt * args.rollout_batch_size

    result = _group_normalize_rewards(args, _samples(group_ids, raw), raw)

    # Each group is centered on its own mean.
    torch.testing.assert_close(torch.tensor(result[0:4]), torch.tensor([0.75, -0.25, -0.25, -0.25]))
    torch.testing.assert_close(torch.tensor(result[4:7]), torch.tensor([1 / 3, 1 / 3, -2 / 3]))
    torch.testing.assert_close(torch.tensor(result[7:11]), torch.tensor([0.25, 0.25, 0.25, -0.75]))
    # The legacy behaviour normalized against the batch mean instead.
    assert result != _legacy_positional_impl(args, raw)


def test_each_group_is_zero_mean_regardless_of_group_sizes():
    args = _args(n_samples_per_prompt=3, rollout_batch_size=3)
    group_ids = [0, 0, 1, 1, 1, 1, 1, 2]
    raw = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0]

    result = _group_normalize_rewards(args, _samples(group_ids, raw), raw)

    for group_id in {0, 1, 2}:
        values = [r for r, g in zip(result, group_ids, strict=True) if g == group_id]
        torch.testing.assert_close(torch.tensor(values).mean(), torch.tensor(0.0), atol=1e-6, rtol=0)


def test_singleton_group_yields_zero_not_nan_with_std_normalization():
    """Unbiased std over one element is NaN; a singleton is already all-zero."""
    args = _args(n_samples_per_prompt=1, rollout_batch_size=2, grpo_std_normalization=True)
    group_ids = [0, 1]
    raw = [1.0, 0.0]

    result = _group_normalize_rewards(args, _samples(group_ids, raw), raw)

    assert result == [0.0, 0.0]
    assert not any(value != value for value in result)


def test_group_index_none_falls_back_to_rollout_id_with_a_warning(caplog):
    args = _args(n_samples_per_prompt=2, rollout_batch_size=1)
    samples = [
        Sample(group_index=None, index=0, rollout_id=5, reward=1.0),
        Sample(group_index=None, index=1, rollout_id=5, reward=0.0),
    ]

    with caplog.at_level("WARNING"):
        result = _group_normalize_rewards(args, samples, [1.0, 0.0])

    # Both samples share rollout_id=5, so they form one group.
    torch.testing.assert_close(torch.tensor(result), torch.tensor([0.5, -0.5]))
    assert "group_index=None" in caplog.text


def test_ordering_is_preserved_when_groups_are_interleaved():
    """Group ids need not arrive contiguously; output stays positionally aligned."""
    args = _args(n_samples_per_prompt=2, rollout_batch_size=2)
    group_ids = [0, 1, 0, 1]
    raw = [1.0, 0.0, 0.0, 1.0]

    result = _group_normalize_rewards(args, _samples(group_ids, raw), raw)

    torch.testing.assert_close(torch.tensor(result), torch.tensor([0.5, -0.5, -0.5, 0.5]))


def test_post_process_rewards_returns_raw_rewards_unchanged():
    args = _args(n_samples_per_prompt=2, rollout_batch_size=1)
    raw = [1.0, 0.0]
    samples = _samples([0, 0], raw)

    raw_rewards, normalized = _post_process_rewards(args, samples)

    assert raw_rewards == raw
    torch.testing.assert_close(torch.tensor(normalized), torch.tensor([0.5, -0.5]))


def test_normalization_is_skipped_when_disabled():
    args = _args(rewards_normalization=False, n_samples_per_prompt=2, rollout_batch_size=1)
    raw = [1.0, 0.0]

    raw_rewards, normalized = _post_process_rewards(args, _samples([0, 0], raw))

    assert raw_rewards == raw
    assert normalized == raw


# ---------------------------------------------------------------------------
# Interaction with the global advantage-whitening pass in
# ``compute_advantages_and_returns``.
# ---------------------------------------------------------------------------


def _whiten(advantages, weights):
    """Token-weighted whitening, matching ``distributed_masked_whiten``."""
    mean = (advantages * weights).sum() / weights.sum()
    var = (((advantages - mean) ** 2) * weights).sum() / weights.sum()
    return (advantages - mean) * torch.rsqrt(var + 1e-8)


def _rollout(length_penalty, seed=0, n_per_prompt=32, num_groups=8):
    """A rollout where wrong answers run longer, as agentic trajectories do.

    Every draw goes through an explicit ``generator`` so the fixture is
    independent of global RNG state (and therefore of test execution order).
    ``Distribution.sample`` has no generator argument, so the lognormal draw is
    built from ``torch.randn`` directly.
    """
    generator = torch.Generator().manual_seed(seed)
    pass_rate = torch.rand(num_groups, generator=generator) * 0.9 + 0.05
    rewards = (torch.rand(num_groups, n_per_prompt, generator=generator) < pass_rate[:, None]).float()
    base = torch.exp(8.0 + 0.6 * torch.randn(num_groups, n_per_prompt, generator=generator))
    lengths = (base * (1.0 + length_penalty * (1 - rewards))).clamp(64, 32768).round()
    # DAPO-style dynamic sampling drops zero-variance groups before training.
    keep = rewards.std(dim=1) > 1e-6
    return rewards[keep], lengths[keep]


def _normalized_advantages(args, rewards, lengths):
    group_ids = [i for i in range(rewards.shape[0]) for _ in range(rewards.shape[1])]
    raw = rewards.flatten().tolist()
    normalized = _group_normalize_rewards(args, _samples(group_ids, raw), raw)
    return torch.tensor(normalized).view_as(rewards), lengths


def test_group_norm_alone_keeps_every_group_zero_mean_in_token_space():
    """The invariant GRPO relies on: no group gets a net push in either direction.

    Group normalization centers per group with every sequence weighted equally.
    That also holds token-weighted only because the centering is exact per
    group -- which is precisely what the global whitening pass destroys.
    """
    args = _args(n_samples_per_prompt=32, rollout_batch_size=8, grpo_std_normalization=True)
    rewards, lengths = _rollout(length_penalty=0.6, seed=3)
    advantages, lengths = _normalized_advantages(args, rewards, lengths)

    for group in range(advantages.shape[0]):
        torch.testing.assert_close(advantages[group].mean(), torch.tensor(0.0), atol=1e-5, rtol=0)


def _mean_token_weighted_advantage(args, length_penalty, seeds=range(20)):
    """Average the token-weighted mean advantage over many rollouts.

    A single rollout's token-weighted mean has a per-seed spread of ~0.04, so
    averaging is what separates the systematic bias from sampling noise.
    """
    values = []
    for seed in seeds:
        rewards, lengths = _rollout(length_penalty=length_penalty, seed=seed)
        advantages, lengths = _normalized_advantages(args, rewards, lengths)
        values.append(((advantages * lengths).sum() / lengths.sum()).item())
    return torch.tensor(values).mean().item()


def test_global_whitening_breaks_per_group_zero_mean_when_length_correlates_with_reward():
    """Regression guard for the redundant ``--normalize-advantages`` pass.

    Group norm centers on a SEQUENCE-weighted mean; whitening re-centers on a
    TOKEN-weighted one. Wrong trajectories are longer, so the token-weighted
    mean of an already-centered advantage is negative and whitening adds a
    uniform positive constant to every token.
    """
    args = _args(n_samples_per_prompt=32, rollout_batch_size=8, grpo_std_normalization=True)

    # Systematically negative -- the bias whitening will invert into a push.
    assert _mean_token_weighted_advantage(args, length_penalty=0.6) < -0.10

    rewards, lengths = _rollout(length_penalty=0.6, seed=3)
    advantages, lengths = _normalized_advantages(args, rewards, lengths)
    whitened = _whiten(advantages, lengths)

    # Groups are pushed off zero, and the penalty on wrong answers weakens.
    group_means = torch.stack([whitened[g].mean() for g in range(whitened.shape[0])])
    assert group_means.abs().max() > 0.05
    assert group_means.abs().max() > torch.stack([advantages[g].mean() for g in range(advantages.shape[0])]).abs().max()

    wrong = rewards == 0
    assert whitened[wrong].mean() > advantages[wrong].mean()

    # The spurious push lands mostly on wrong trajectories, since they hold a
    # token share larger than their sequence share.
    assert lengths[wrong].sum() / lengths.sum() > wrong.float().mean()


def test_no_bias_is_injected_when_length_is_uncorrelated_with_reward():
    """Confirms the diagnosis: the bias comes from the correlation, not whitening itself."""
    args = _args(n_samples_per_prompt=32, rollout_batch_size=8, grpo_std_normalization=True)

    assert abs(_mean_token_weighted_advantage(args, length_penalty=0.0)) < 0.02


def test_bias_grows_monotonically_with_the_length_correlation():
    """Dose-response: the stronger the correlation, the larger the injected push."""
    args = _args(n_samples_per_prompt=32, rollout_batch_size=8, grpo_std_normalization=True)

    biases = [_mean_token_weighted_advantage(args, length_penalty=lp) for lp in (0.0, 0.4, 0.6, 1.0)]

    assert all(later < earlier for earlier, later in zip(biases, biases[1:], strict=False)), biases


def test_launcher_disables_normalize_advantages_by_default():
    """The Gemma4 fused-agent launcher must not stack whitening on group norm."""
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "train_gemma4_fused_agent_sync.sh"
    text = launcher.read_text()

    assert 'NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-false}"' in text
    assert 'NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-true}"' not in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
