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
from slime.utils.credit_assignment import CreditAssignmentConfig, excluded_from_reward_baseline
from slime.utils.prompt_equal import process_segment_rewards
from slime.utils.types import Sample


NUM_GPUS = 0
ODYSSEY_GEMMA4_LAUNCHERS = (
    "train_odyssey_gemma4_sync.sh",
    "train_odyssey_gemma4_multinode_sync.sh",
    "train_odyssey_gemma4_async.sh",
)


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


@pytest.mark.parametrize("launcher_name", ODYSSEY_GEMMA4_LAUNCHERS)
def test_launcher_disables_normalize_advantages_by_default(launcher_name):
    """The Gemma4 fused-agent launcher must not stack whitening on group norm."""
    launcher = Path(__file__).resolve().parents[1] / "experiments" / launcher_name
    text = launcher.read_text()

    assert 'NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-false}"' in text
    assert 'NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-true}"' not in text


# ---------------------------------------------------------------------------
# Dr.GRPO mean-only advantages.
#
# Dividing by the group std scales each group by the inverse of its own
# evidence. DAPO's zero-variance filter drops all-wrong groups, so what survives
# is dominated by near-degenerate ones -- exactly where the division blows up.
# ---------------------------------------------------------------------------


def _binary_group_advantage(k, n, use_std, path):
    """Winner advantage for a group with ``k`` of ``n`` correct, on either path."""
    rewards = [1.0] * k + [0.0] * (n - k)
    args = _args(n_samples_per_prompt=n, rollout_batch_size=1, grpo_std_normalization=use_std)
    if path == "group":
        samples = _samples([0] * n, rewards)
        return max(_group_normalize_rewards(args, samples, rewards))
    # The fused-agent rollout marks every sample prompt_equal_loss, which routes
    # _post_process_rewards through process_segment_rewards instead.
    samples = [
        Sample(
            group_index=0,
            index=i,
            rollout_id=i,
            reward=reward,
            metadata={"prompt_equal_loss": True, "parent_traj_id": f"t{i}", "segment_index": 0, "segment_count": 1},
        )
        for i, reward in enumerate(rewards)
    ]
    return max(process_segment_rewards(args, samples, rewards, rewards))


@pytest.mark.parametrize("path", ["group", "segment"])
def test_std_normalization_amplifies_the_least_certain_groups(path):
    """Documents the behaviour being switched off, on both reward paths.

    A lone winner in 32 samples gets ~5.6x the pull of a winner in a balanced
    group -- the group carrying the least evidence moves the weights the most.
    """
    lone = _binary_group_advantage(1, 32, use_std=True, path=path)
    balanced = _binary_group_advantage(16, 32, use_std=True, path=path)

    assert lone > 5.0
    # ~1.0, but not exactly: the two paths disagree on the std convention
    # (_group_normalize_rewards uses unbiased, process_segment_rewards
    # population), which is a pre-existing divergence this change makes moot.
    assert balanced == pytest.approx(1.0, abs=0.02)
    assert lone / balanced > 5.0


@pytest.mark.parametrize("path", ["group", "segment"])
def test_mean_only_bounds_every_group_advantage_by_one(path):
    """Mean-only centering caps |advantage| at (n-1)/n regardless of sparsity."""
    for k in (1, 2, 8, 16, 31):
        advantage = _binary_group_advantage(k, 32, use_std=False, path=path)
        assert advantage <= 31 / 32 + 1e-6, (k, advantage)

    assert _binary_group_advantage(1, 32, use_std=False, path=path) == pytest.approx(31 / 32)


@pytest.mark.parametrize("path", ["group", "segment"])
def test_mean_only_preserves_the_sign_and_ordering_of_advantages(path):
    """Only the per-group scale changes; the gradient direction does not.

    Winners stay positive, losers negative, and the group stays zero-mean --
    so this is a variance-reduction change, not a change of objective.
    """
    rewards = [1.0] * 6 + [0.0] * 26
    args_std = _args(n_samples_per_prompt=32, rollout_batch_size=1, grpo_std_normalization=True)
    args_mean = _args(n_samples_per_prompt=32, rollout_batch_size=1, grpo_std_normalization=False)
    samples = _samples([0] * 32, rewards)

    with_std = _group_normalize_rewards(args_std, samples, rewards)
    mean_only = _group_normalize_rewards(args_mean, samples, rewards)

    for reward, a, b in zip(rewards, with_std, mean_only, strict=True):
        assert (a > 0) == (b > 0) == (reward > 0.5)
    # Same vector up to a single positive scale factor.
    scale = with_std[0] / mean_only[0]
    assert scale > 1
    torch.testing.assert_close(torch.tensor(with_std), torch.tensor(mean_only) * scale, atol=1e-5, rtol=0)
    torch.testing.assert_close(torch.tensor(mean_only).mean(), torch.tensor(0.0), atol=1e-6, rtol=0)


def test_mean_only_flattens_the_advantage_mass_held_by_sparse_groups():
    """The batch-level effect: sparse groups stop dominating the update.

    Group shapes are taken from odyssey-gemma4-e4b-think-dev38 step 40, whose
    surviving groups were [1, 1, 2, 2, 5, 11, 12, 18] correct out of 32.
    """
    observed_group_sizes = [1, 1, 2, 2, 5, 11, 12, 18]

    def sparse_mass_share(use_std):
        per_group = []
        for k in observed_group_sizes:
            rewards = [1.0] * k + [0.0] * (32 - k)
            args = _args(n_samples_per_prompt=32, rollout_batch_size=1, grpo_std_normalization=use_std)
            normalized = _group_normalize_rewards(args, _samples([0] * 32, rewards), rewards)
            per_group.append(sum(abs(x) for x in normalized))
        total = sum(per_group)
        return sum(m for k, m in zip(observed_group_sizes, per_group, strict=True) if k <= 2) / total

    with_std = sparse_mass_share(use_std=True)
    mean_only = sparse_mass_share(use_std=False)

    # Two groups out of eight held 31% of the update; mean-only takes them to
    # 17.5%, just under their 2/8 = 25% share of the sequences.
    assert with_std == pytest.approx(0.314, abs=0.01)
    assert mean_only == pytest.approx(0.175, abs=0.01)
    assert mean_only < 2 / len(observed_group_sizes) < with_std


@pytest.mark.parametrize("launcher_name", ODYSSEY_GEMMA4_LAUNCHERS)
def test_gemma4_launcher_keeps_grpo_std_normalization_by_default(launcher_name):
    """Odyssey Gemma4 launchers intentionally retain standard GRPO scaling."""
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments" / launcher_name).read_text()
    assert 'GRPO_STD_NORMALIZATION="${GRPO_STD_NORMALIZATION:-true}"' in launcher
    assert 'GRPO_STD_NORMALIZATION="${GRPO_STD_NORMALIZATION:-false}"' not in launcher


def test_gemma4_multinode_sync_uses_strict_qwen_rollout_admission():
    """Keep a deep candidate reservoir while rejecting zero-variance groups."""
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments" / "train_odyssey_gemma4_multinode_sync.sh").read_text()

    assert 'SYNC_MIN_PENDING_GROUPS=96' in launcher
    assert 'SYNC_WEBQA_MIN_PENDING_GROUPS="${SYNC_WEBQA_MIN_PENDING_GROUPS:-32}"' in launcher
    assert 'SYNC_MCP_MIN_PENDING_GROUPS="${SYNC_MCP_MIN_PENDING_GROUPS:-32}"' in launcher
    assert 'SYNC_MCP_ONLY_MIN_PENDING_GROUPS="${SYNC_MCP_ONLY_MIN_PENDING_GROUPS:-96}"' in launcher
    assert 'ROLLOUT_TASK_FAMILY_TOP_MEAN_STEPS="${ROLLOUT_TASK_FAMILY_TOP_MEAN_STEPS:-true}"' in launcher
    assert 'FULLY_ASYNC_FILTER_RELAX_AFTER_GROUPS=0' in launcher
    assert 'This launcher requires strict dynamic sampling' in launcher
    assert 'SLIME_SYNC_MIN_PENDING_GROUPS="${SYNC_MIN_PENDING_GROUPS}"' in launcher
    assert 'SLIME_SYNC_MCP_ONLY_MIN_PENDING_GROUPS="${SYNC_MCP_ONLY_MIN_PENDING_GROUPS}"' in launcher
    assert '--fully-async-filter-relax-after-groups "${FULLY_ASYNC_FILTER_RELAX_AFTER_GROUPS}"' in launcher
    assert 'ROLLOUT_ARGS+=(--rollout-task-family-top-mean-steps)' in launcher


# ---------------------------------------------------------------------------
# Credit-assignment penalties must not define the baseline they are scored
# against. Covered on both reward paths, since the fused-agent rollout marks
# every sample prompt_equal_loss and therefore uses the segment path.
# ---------------------------------------------------------------------------


def _credit_args(enable=True, **overrides):
    values = dict(
        credit_assignment_enable=enable,
        credit_assignment_tool_parser_error=True,
        credit_assignment_repeated_search_query=True,
        credit_assignment_too_many_tool_calls=True,
        credit_assignment_search_bypass=True,
        credit_assignment_mixed_tool_and_answer=True,
        credit_assignment_ngram_repetition=True,
    )
    values.update(overrides)
    return _args(**values)


def _penalized_samples(rewards, events, *, segment):
    """Build a one-group batch where ``events[i]`` marks sample i as penalized."""
    samples = []
    for i, (reward, event) in enumerate(zip(rewards, events, strict=True)):
        metadata = {}
        if segment:
            metadata = {
                "prompt_equal_loss": True,
                "parent_traj_id": f"t{i}",
                "segment_index": 0,
                "segment_count": 1,
            }
        if event:
            metadata["credit_assignment_event"] = event
        samples.append(Sample(group_index=0, index=i, rollout_id=i, reward=reward, metadata=metadata))
    return samples


def _normalize(args, rewards, events, *, segment):
    samples = _penalized_samples(rewards, events, segment=segment)
    if segment:
        return process_segment_rewards(args, samples, rewards, rewards)
    return _group_normalize_rewards(args, samples, rewards)


@pytest.mark.parametrize("segment", [False, True])
def test_penalized_samples_are_excluded_from_the_group_baseline(segment):
    """The baseline is the mean over clean samples only."""
    rewards = [1.0, 1.0, 0.0, 0.0]
    events = [None, None, None, "tool_parser_error"]

    result = _normalize(_credit_args(), rewards, events, segment=segment)

    # Clean mean is 2/3, not the all-sample mean of 1/2.
    torch.testing.assert_close(
        torch.tensor(result),
        torch.tensor([1 / 3, 1 / 3, -2 / 3, -2 / 3]),
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("segment", [False, True])
def test_penalty_survives_and_strengthens_while_winners_are_less_over_credited(segment):
    """The directional claim: dilution was happening on both sides at once."""
    rewards = [1.0] * 4 + [0.0] * 2 + [0.0] * 4
    events = [None] * 6 + ["tool_parser_error"] * 4

    before = _normalize(_credit_args(enable=False), rewards, events, segment=segment)
    after = _normalize(_credit_args(), rewards, events, segment=segment)

    winner_before, winner_after = before[0], after[0]
    penalty_before, penalty_after = before[-1], after[-1]

    assert winner_after < winner_before  # less over-credited
    assert penalty_after < penalty_before  # more strongly penalized
    assert penalty_after < 0  # still a penalty, not a reward


@pytest.mark.parametrize("segment", [False, True])
def test_penalized_samples_keep_a_nonzero_advantage_rather_than_being_dropped(segment):
    """Excluded from the baseline is not the same as excluded from training."""
    rewards = [1.0, 0.0, 0.0]
    events = [None, None, "tool_parser_error"]

    result = _normalize(_credit_args(), rewards, events, segment=segment)

    assert result[-1] != 0.0
    torch.testing.assert_close(torch.tensor(result[-1]), torch.tensor(-0.5), atol=1e-6, rtol=0)


@pytest.mark.parametrize("segment", [False, True])
def test_all_penalized_group_falls_back_to_the_full_group(segment):
    """A baseline over an empty set is undefined; degrade, do not crash."""
    rewards = [1.0, 0.0, 0.0, 0.0]
    events = ["tool_parser_error"] * 4

    result = _normalize(_credit_args(), rewards, events, segment=segment)

    # Identical to the old all-sample behaviour: centered on 0.25.
    torch.testing.assert_close(
        torch.tensor(result),
        torch.tensor([0.75, -0.25, -0.25, -0.25]),
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("segment", [False, True])
def test_disabled_credit_assignment_leaves_normalization_untouched(segment):
    """No behaviour change for runs that do not opt into credit assignment."""
    rewards = [1.0, 1.0, 0.0, 0.0]
    events = [None, None, None, "tool_parser_error"]

    result = _normalize(_credit_args(enable=False), rewards, events, segment=segment)

    torch.testing.assert_close(
        torch.tensor(result),
        torch.tensor([0.5, 0.5, -0.5, -0.5]),
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("segment", [False, True])
def test_std_normalization_uses_the_same_population_as_the_mean(segment):
    """Mean and scale must come from one population, or advantages are skewed."""
    rewards = [1.0, 1.0, 0.0, 0.0, 0.0]
    events = [None, None, None, None, "tool_parser_error"]
    args = _credit_args(grpo_std_normalization=True)

    result = _normalize(args, rewards, events, segment=segment)

    clean = torch.tensor([1.0, 1.0, 0.0, 0.0])
    centered = clean - clean.mean()
    expected_scale = centered.std(correction=0 if segment else 1) + 1e-6
    torch.testing.assert_close(
        torch.tensor(result[0]),
        torch.tensor(((1.0 - clean.mean()) / expected_scale).item()),
        atol=1e-5,
        rtol=0,
    )
    # The clean samples remain zero-mean after scaling.
    torch.testing.assert_close(torch.tensor(result[:4]).mean(), torch.tensor(0.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("segment", [False, True])
def test_constant_clean_baseline_does_not_amplify_credit_penalties(segment):
    """A mostly parser-error group must stay finite when std normalization is on."""
    rewards = [0.0] * 30 + [1.0, 1.0]
    events = ["tool_parser_error"] * 30 + [None, None]

    result = _normalize(_credit_args(grpo_std_normalization=True), rewards, events, segment=segment)

    assert all(torch.isfinite(torch.tensor(result)))
    torch.testing.assert_close(torch.tensor(result[:30]), torch.full((30,), -1.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.tensor(result[30:]), torch.zeros(2), atol=1e-6, rtol=0)


def test_every_credit_assignment_event_is_excluded_from_the_baseline():
    """All guard events assign a synthetic 0.0, so all must be excluded."""
    events = [
        "tool_parser_error",
        "direct_submit_without_tool",
        "ngram_repetition",
        "mixed_tool_and_answer",
        "max_response_len_exceeded",
        "max_turns_exceeded",
        "repeated_search_query",
        "too_many_tool_calls",
        "tail_guard_early_stop",
        "search_bypass",
    ]
    config = CreditAssignmentConfig.from_args(_credit_args())

    for event in events:
        assert excluded_from_reward_baseline({"credit_assignment_event": event}, config), event
    assert not excluded_from_reward_baseline({}, config)
    assert not excluded_from_reward_baseline(None, config)


def test_nothing_is_excluded_when_credit_assignment_is_disabled():
    config = CreditAssignmentConfig.from_args(_credit_args(enable=False))

    assert not excluded_from_reward_baseline({"credit_assignment_event": "tool_parser_error"}, config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
