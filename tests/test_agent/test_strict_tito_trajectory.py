import pytest

from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.utils.types import Sample


NUM_GPUS = 0


def _lp(token_id: int) -> float:
    return -token_id / 1_000_000.0


def _assert_trainable_tokens_keep_their_logprobs(samples):
    for sample in samples:
        assert len(sample.loss_mask) == sample.response_length
        assert len(sample.rollout_log_probs) == sample.response_length
        response_start = len(sample.tokens) - sample.response_length
        for rel_idx, (mask, logprob) in enumerate(zip(sample.loss_mask, sample.rollout_log_probs, strict=True)):
            token_id = sample.tokens[response_start + rel_idx]
            if int(mask) == 1:
                assert logprob == pytest.approx(_lp(token_id))
            else:
                assert logprob == 0.0


def _record_strict_turn(
    manager: TrajectoryManager,
    sid: str,
    *,
    prompt_ids: list[int],
    context_delta_ids: list[int],
    output_ids: list[int],
    turn_idx: int,
    loss_mask: list[int] | None = None,
    policy_loss_mask: list[int] | None = None,
    top_p: bool = False,
) -> None:
    top_p_ids = None
    top_p_offsets = None
    if top_p:
        top_p_ids = []
        top_p_offsets = [0]
        for token_id in output_ids:
            top_p_ids.extend([token_id * 10, token_id * 10 + 1])
            top_p_offsets.append(len(top_p_ids))
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_ids,
            context_delta_ids=context_delta_ids,
            output_ids=output_ids,
            output_log_probs=[_lp(token_id) for token_id in output_ids],
            loss_mask=loss_mask,
            policy_loss_mask=policy_loss_mask,
            rollout_top_p_token_ids=top_p_ids,
            rollout_top_p_token_offsets=top_p_offsets,
            finish_reason="tool_calls",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            *(
                message
                for prev in range(turn_idx)
                for message in (
                    {"role": "assistant", "content": f"a{prev}"},
                    {"role": "tool", "content": f"tool{prev}"},
                )
            ),
        ],
        response_message={"role": "assistant", "content": f"a{turn_idx}"},
    )


def test_strict_context_delta_survives_thousand_turn_long_horizon():
    manager = TrajectoryManager()
    sid = "strict-thousand-turns"
    prefix: list[int] = []
    expected_tokens: list[int] = []
    expected_loss_mask: list[int] = []
    expected_policy_loss_mask: list[int] = []
    expected_logprobs: list[float] = []

    for turn_idx in range(1_005):
        context_delta = [10_000 + turn_idx, 20_000 + (turn_idx % 17)]
        output_ids = [700_000 + turn_idx]
        prompt_ids = prefix + context_delta
        if turn_idx % 113 == 0:
            # Strict TITO must trust the adapter's exact delta, not mutate
            # already-generated history to match a replay-rendered prompt.
            prompt_ids = prefix[:3] + [999_999_000 + turn_idx] + prefix[3:] + context_delta
        loss_mask = [0] if turn_idx % 97 == 0 else [1]
        policy_loss_mask = [0] if turn_idx % 41 == 0 else loss_mask
        _record_strict_turn(
            manager,
            sid,
            prompt_ids=prompt_ids,
            context_delta_ids=context_delta,
            output_ids=output_ids,
            turn_idx=turn_idx,
            loss_mask=loss_mask,
            policy_loss_mask=policy_loss_mask,
            top_p=turn_idx in {777, 1_004},
        )
        prefix.extend(context_delta)
        prefix.extend(output_ids)
        expected_tokens.extend(context_delta)
        expected_tokens.extend(output_ids)
        expected_loss_mask.extend([0] * len(context_delta))
        expected_loss_mask.extend(loss_mask)
        expected_policy_loss_mask.extend([0] * len(context_delta))
        expected_policy_loss_mask.extend(policy_loss_mask)
        expected_logprobs.extend([0.0] * len(context_delta))
        expected_logprobs.extend([_lp(output_ids[0]) if loss_mask[0] else 0.0])

    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == expected_tokens
    assert sample.response_length == len(expected_loss_mask) - 2
    assert sample.loss_mask == expected_loss_mask[2:]
    assert sample.policy_loss_mask == expected_policy_loss_mask[2:]
    assert sample.rollout_log_probs == expected_logprobs[2:]
    assert sample.rollout_top_p_token_offsets is not None
    assert len(sample.rollout_top_p_token_offsets) == sample.response_length + 1
    assert sample.rollout_top_p_token_offsets[-1] == len(sample.rollout_top_p_token_ids)
    _assert_trainable_tokens_keep_their_logprobs(samples)


def test_explicit_tito_boundary_splits_strict_delta_without_token_realigning():
    manager = TrajectoryManager()
    sid = "strict-boundary"
    p1 = [1, 2, 3]
    r1 = [701_001, 701_002]
    p2 = [9, 9, 9, 4]
    r2 = [702_001]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=p1,
            context_delta_ids=p1,
            output_ids=r1,
            output_log_probs=[_lp(token_id) for token_id in r1],
            finish_reason="tool_calls",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a1"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=[123_456, *p2],
            context_delta_ids=p2,
            tito_boundary_before=True,
            output_ids=r2,
            output_log_probs=[_lp(token_id) for token_id in r2],
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "tool"},
        ],
        response_message={"role": "assistant", "content": "a2"},
    )

    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)

    assert len(samples) == 2
    assert samples[0].tokens == p1 + r1
    assert samples[1].tokens == p2 + r2
    assert samples[1].loss_mask == [1]
    _assert_trainable_tokens_keep_their_logprobs(samples)


def test_tito_boundary_keeps_top_p_and_policy_masks_aligned_per_split_sample():
    manager = TrajectoryManager()
    sid = "boundary-top-p-policy"
    prompt_1 = [1, 2, 3]
    response_1 = [701_001, 701_002]
    delta_2 = [4, 5]
    response_2 = [702_001, 702_002, 702_003]
    boundary_delta = [9, 9, 9]
    response_3 = [703_001, 703_002]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_1,
            context_delta_ids=prompt_1,
            output_ids=response_1,
            output_log_probs=[_lp(token_id) for token_id in response_1],
            loss_mask=[1, 0],
            policy_loss_mask=[1, 0],
            rollout_top_p_token_ids=[11, 12, 13, 14],
            rollout_top_p_token_offsets=[0, 2, 4],
            finish_reason="tool_calls",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a1"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_1 + response_1 + delta_2,
            context_delta_ids=delta_2,
            output_ids=response_2,
            output_log_probs=[_lp(token_id) for token_id in response_2],
            loss_mask=[1, 1, 0],
            policy_loss_mask=[0, 1, 0],
            rollout_top_p_token_ids=[21, 22, 23, 24, 25, 26],
            rollout_top_p_token_offsets=[0, 1, 3, 6],
            finish_reason="tool_calls",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "tool1"},
        ],
        response_message={"role": "assistant", "content": "a2"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=[123_456, *boundary_delta],
            context_delta_ids=boundary_delta,
            tito_boundary_before=True,
            output_ids=response_3,
            output_log_probs=[_lp(token_id) for token_id in response_3],
            loss_mask=[0, 1],
            policy_loss_mask=[0, 0],
            rollout_top_p_token_ids=[31, 32, 33],
            rollout_top_p_token_offsets=[0, 2, 3],
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "tool1"},
            {"role": "assistant", "content": "a2"},
            {"role": "tool", "content": "tool2"},
        ],
        response_message={"role": "assistant", "content": "a3"},
    )

    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=3.0)

    assert len(samples) == 2
    assert samples[0].tokens == prompt_1 + response_1 + delta_2 + response_2
    assert samples[0].response_length == len(response_1) + len(delta_2) + len(response_2)
    assert samples[0].loss_mask == [1, 0, 0, 0, 1, 1, 0]
    assert samples[0].policy_loss_mask == [1, 0, 0, 0, 0, 1, 0]
    assert samples[0].rollout_top_p_token_ids == [11, 12, 13, 14, 21, 22, 23, 24, 25, 26]
    assert samples[0].rollout_top_p_token_offsets == [0, 2, 4, 4, 4, 5, 7, 10]

    assert samples[1].tokens == boundary_delta + response_3
    assert samples[1].response_length == len(response_3)
    assert samples[1].loss_mask == [0, 1]
    assert samples[1].policy_loss_mask == [0, 0]
    assert samples[1].rollout_top_p_token_ids == [31, 32, 33]
    assert samples[1].rollout_top_p_token_offsets == [0, 2, 3]
    assert [sample.reward for sample in samples] == [1.5, 1.5]
    _assert_trainable_tokens_keep_their_logprobs(samples)


def test_all_zero_response_loss_mask_is_dropped_unless_explicitly_allowed():
    manager = TrajectoryManager()
    sid = "all-zero-loss"
    prompt = [1, 2]
    response = [701_001, 701_002]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt,
            context_delta_ids=prompt,
            output_ids=response,
            output_log_probs=[_lp(token_id) for token_id in response],
            loss_mask=[0, 0],
            finish_reason="stop",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a"},
    )

    assert manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0) == []

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt,
            context_delta_ids=prompt,
            output_ids=response,
            output_log_probs=[_lp(token_id) for token_id in response],
            loss_mask=[0, 0],
            finish_reason="stop",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a"},
    )

    samples = manager.get_trajectory(
        sid,
        base_sample=Sample(index=0, prompt=""),
        reward=1.0,
        allow_fully_masked=True,
    )

    assert len(samples) == 1
    assert samples[0].loss_mask == [0, 0]
    assert samples[0].rollout_log_probs == [0.0, 0.0]


def test_policy_mask_can_zero_pg_without_dropping_rollout_sample():
    manager = TrajectoryManager()
    sid = "zero-policy-mask"
    prompt = [1, 2]
    response = [701_001, 701_002, 701_003]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt,
            context_delta_ids=prompt,
            output_ids=response,
            output_log_probs=[_lp(token_id) for token_id in response],
            loss_mask=[1, 1, 1],
            policy_loss_mask=[0, 0, 0],
            finish_reason="stop",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a"},
    )

    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)

    assert len(samples) == 1
    assert samples[0].loss_mask == [1, 1, 1]
    assert samples[0].policy_loss_mask == [0, 0, 0]
    assert samples[0].rollout_log_probs == [_lp(token_id) for token_id in response]


def test_shared_assistant_branch_replay_does_not_retrain_policy_or_top_p():
    manager = TrajectoryManager()
    sid = "shared-assistant-branch-policy-top-p"
    prompt_1 = [1, 2]
    shared_response = [701_001, 701_002, 701_003]
    branch_a_delta = [31, 32]
    branch_a_response = [702_001, 702_002]
    branch_b_delta = [41, 42, 43]
    branch_b_response = [703_001]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_1,
            context_delta_ids=prompt_1,
            output_ids=shared_response,
            output_log_probs=[_lp(token_id) for token_id in shared_response],
            loss_mask=[1, 0, 1],
            policy_loss_mask=[0, 0, 1],
            rollout_top_p_token_ids=[11, 12, 13, 14, 15],
            rollout_top_p_token_offsets=[0, 2, 2, 5],
            finish_reason="tool_calls",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "shared"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_1 + shared_response + branch_a_delta,
            context_delta_ids=branch_a_delta,
            output_ids=branch_a_response,
            output_log_probs=[_lp(token_id) for token_id in branch_a_response],
            loss_mask=[1, 1],
            policy_loss_mask=[1, 0],
            rollout_top_p_token_ids=[21, 22, 23],
            rollout_top_p_token_offsets=[0, 1, 3],
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "shared"},
            {"role": "tool", "content": "tool-a"},
        ],
        response_message={"role": "assistant", "content": "branch-a"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_1 + shared_response + branch_b_delta,
            context_delta_ids=branch_b_delta,
            output_ids=branch_b_response,
            output_log_probs=[_lp(token_id) for token_id in branch_b_response],
            loss_mask=[1],
            policy_loss_mask=[0],
            rollout_top_p_token_ids=[31, 32],
            rollout_top_p_token_offsets=[0, 2],
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "shared"},
            {"role": "tool", "content": "tool-b"},
        ],
        response_message={"role": "assistant", "content": "branch-b"},
    )

    samples = manager.get_trajectory(sid, base_sample=Sample(index=7, prompt="", rollout_id=77), reward=3.0)

    assert len(samples) == 2
    assert [sample.reward for sample in samples] == [1.5, 1.5]

    first, second = samples
    assert first.tokens == prompt_1 + shared_response + branch_a_delta + branch_a_response
    assert first.response_length == len(shared_response) + len(branch_a_delta) + len(branch_a_response)
    assert first.loss_mask == [1, 0, 1, 0, 0, 1, 1]
    assert first.policy_loss_mask == [0, 0, 1, 0, 0, 1, 0]
    assert first.rollout_log_probs == [
        _lp(shared_response[0]),
        0.0,
        _lp(shared_response[2]),
        0.0,
        0.0,
        _lp(branch_a_response[0]),
        _lp(branch_a_response[1]),
    ]
    assert first.rollout_top_p_token_ids == [11, 12, 13, 14, 15, 21, 22, 23]
    assert first.rollout_top_p_token_offsets == [0, 2, 2, 5, 5, 5, 6, 8]

    assert second.tokens == prompt_1 + shared_response + branch_b_delta + branch_b_response
    assert second.response_length == len(shared_response) + len(branch_b_delta) + len(branch_b_response)
    assert second.loss_mask == [0, 0, 0, 0, 0, 0, 1]
    assert second.policy_loss_mask == [0, 0, 0, 0, 0, 0, 0]
    assert second.rollout_log_probs == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _lp(branch_b_response[0])]
    assert second.rollout_top_p_token_ids == [31, 32]
    assert second.rollout_top_p_token_offsets == [0, 0, 0, 0, 0, 0, 0, 2]
    _assert_trainable_tokens_keep_their_logprobs(samples)


def test_strict_context_delta_never_realigns_replayed_prompt_tokens():
    manager = TrajectoryManager()
    sid = "strict-context-delta"
    prompt_1 = [1, 2, 3]
    response_1 = [701_001, 701_002]
    delta_2 = [4, 5, 6]
    response_2 = [702_001, 702_002]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_1,
            context_delta_ids=prompt_1,
            output_ids=response_1,
            output_log_probs=[_lp(token_id) for token_id in response_1],
            finish_reason="tool_calls",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a1"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            # This replay prompt intentionally differs from prompt_1 + response_1.
            # Strict TITO must ignore it for training construction and consume only
            # the supplied context delta.
            prompt_ids=[1, 2, 999_999, 3, *delta_2],
            context_delta_ids=delta_2,
            output_ids=response_2,
            output_log_probs=[_lp(token_id) for token_id in response_2],
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "tool"},
        ],
        response_message={"role": "assistant", "content": "a2"},
    )

    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)

    assert len(samples) == 1
    assert samples[0].tokens == prompt_1 + response_1 + delta_2 + response_2
    assert samples[0].loss_mask == [1, 1, 0, 0, 0, 1, 1]
    _assert_trainable_tokens_keep_their_logprobs(samples)


def test_legacy_prompt_drift_forks_instead_of_realigning_logprobs():
    manager = TrajectoryManager()
    sid = "legacy-drift"
    prompt_1 = [1, 2, 3]
    response_1 = [701_001, 701_002]
    prompt_2 = [1, 2, 3, 701_001, 999_999, 4, 5]
    response_2 = [702_001]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_1,
            output_ids=response_1,
            output_log_probs=[_lp(token_id) for token_id in response_1],
            finish_reason="tool_calls",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a1"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_2,
            output_ids=response_2,
            output_log_probs=[_lp(token_id) for token_id in response_2],
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "tool"},
        ],
        response_message={"role": "assistant", "content": "a2"},
    )

    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)

    assert len(samples) == 2
    assert samples[0].tokens == prompt_1 + response_1
    assert samples[1].tokens == prompt_2 + response_2
    _assert_trainable_tokens_keep_their_logprobs(samples)


def test_late_top_p_metadata_is_padded_over_existing_context():
    manager = TrajectoryManager()
    sid = "late-top-p"
    prompt_1 = [1, 2, 3]
    response_1 = [701_001, 701_002]
    delta_2 = [4, 5, 6, 7]
    response_2 = [702_001, 702_002, 702_003]
    top_p_ids = [11, 12, 13, 14, 15, 16]
    top_p_offsets = [0, 2, 4, 6]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_1,
            context_delta_ids=prompt_1,
            output_ids=response_1,
            output_log_probs=[_lp(token_id) for token_id in response_1],
            finish_reason="tool_calls",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a1"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_1 + response_1 + delta_2,
            context_delta_ids=delta_2,
            output_ids=response_2,
            output_log_probs=[_lp(token_id) for token_id in response_2],
            rollout_top_p_token_ids=top_p_ids,
            rollout_top_p_token_offsets=top_p_offsets,
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "tool"},
        ],
        response_message={"role": "assistant", "content": "a2"},
    )

    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)

    assert len(samples) == 1
    assert len(samples[0].rollout_top_p_token_offsets) == samples[0].response_length + 1
    assert samples[0].rollout_top_p_token_offsets[0] == 0
    assert samples[0].rollout_top_p_token_offsets[-1] == len(samples[0].rollout_top_p_token_ids)
    _assert_trainable_tokens_keep_their_logprobs(samples)


def test_first_turn_strips_actual_context_delta_not_full_prompt_length():
    manager = TrajectoryManager()
    sid = "short-first-delta"
    prompt = [1, 2, 3, 4, 5]
    context_delta = [4, 5]
    response = [701_001, 701_002]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt,
            context_delta_ids=context_delta,
            output_ids=response,
            output_log_probs=[_lp(token_id) for token_id in response],
            finish_reason="stop",
        ),
        prompt_messages=[{"role": "user", "content": "compacted"}],
        response_message={"role": "assistant", "content": "a"},
    )

    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)

    assert len(samples) == 1
    assert samples[0].tokens == context_delta + response
    assert samples[0].response_length == len(response)
    assert samples[0].loss_mask == [1, 1]
    _assert_trainable_tokens_keep_their_logprobs(samples)


def test_masked_response_tokens_zero_their_rollout_logprobs():
    manager = TrajectoryManager()
    sid = "masked-logprobs"
    prompt = [1, 2, 3]
    response = [701_001, 701_002, 701_003]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt,
            context_delta_ids=prompt,
            output_ids=response,
            output_log_probs=[_lp(token_id) for token_id in response],
            loss_mask=[0, 1, 0],
            finish_reason="stop",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a"},
    )

    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)

    assert len(samples) == 1
    assert samples[0].loss_mask == [0, 1, 0]
    assert samples[0].rollout_log_probs == [0.0, _lp(response[1]), 0.0]
    _assert_trainable_tokens_keep_their_logprobs(samples)


@pytest.mark.parametrize("logprobs", [[], [float("nan")], [float("inf")]])
def test_strict_trainable_tokens_require_complete_finite_logprobs(logprobs):
    manager = TrajectoryManager()

    with pytest.raises(ValueError, match="logprob"):
        manager.record_turn(
            "strict-logprobs",
            turn=TurnRecord(
                prompt_ids=[1],
                context_delta_ids=[1],
                output_ids=[2],
                output_log_probs=logprobs,
                finish_reason="stop",
                require_rollout_logprobs=True,
            ),
            prompt_messages=[{"role": "user", "content": "u"}],
            response_message={"role": "assistant", "content": "a"},
        )


def test_strict_weight_version_rejects_missing_and_mixed_turns():
    manager = TrajectoryManager()
    common = {
        "prompt_ids": [1],
        "context_delta_ids": [1],
        "output_ids": [2],
        "output_log_probs": [-0.1],
        "finish_reason": "stop",
        "require_weight_version": True,
    }
    messages = [{"role": "user", "content": "u"}]

    with pytest.raises(ValueError, match="requires SGLang"):
        manager.record_turn(
            "missing-version",
            turn=TurnRecord(**common),
            prompt_messages=messages,
            response_message={"role": "assistant", "content": "a"},
        )

    manager.record_turn(
        "mixed-version",
        turn=TurnRecord(**common, weight_version="v1"),
        prompt_messages=messages,
        response_message={"role": "assistant", "content": "a"},
    )
    with pytest.raises(ValueError, match="mixed rollout weight versions"):
        manager.record_turn(
            "mixed-version",
            turn=TurnRecord(**common, weight_version="v2"),
            prompt_messages=messages,
            response_message={"role": "assistant", "content": "b"},
        )


def test_strict_append_only_ignores_message_replay_forks_and_emits_one_sample():
    manager = TrajectoryManager(strict_append_only=True)
    sid = "strict-linear"
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=[1, 2],
            output_ids=[3, 4],
            finish_reason="tool_calls",
            output_log_probs=[-0.3, -0.4],
            context_delta_ids=[1, 2],
            tito_context_reason="initial",
        ),
        prompt_messages=[{"role": "user", "content": "question"}],
        response_message={"role": "assistant", "content": "raw action"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=[1, 2, 3, 4, 5],
            output_ids=[6],
            finish_reason="stop",
            output_log_probs=[-0.6],
            context_delta_ids=[5],
            tito_context_reason="append_delta",
        ),
        prompt_messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "unrelated structured replay"},
        ],
        response_message={"role": "assistant", "content": "answer"},
    )

    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)

    assert len(samples) == 1
    assert samples[0].tokens == [1, 2, 3, 4, 5, 6]
    assert samples[0].loss_mask == [1, 1, 0, 1]
    assert samples[0].rollout_log_probs == [-0.3, -0.4, 0.0, -0.6]
    assert samples[0].metadata["tito_context_reasons"] == ["initial", "append_delta"]


@pytest.mark.parametrize(
    "turn, message",
    [
        (
            TurnRecord(prompt_ids=[1], output_ids=[2], finish_reason="stop"),
            "context_delta_ids",
        ),
        (
            TurnRecord(
                prompt_ids=[1],
                output_ids=[2],
                finish_reason="stop",
                context_delta_ids=[1],
                tito_boundary_before=True,
            ),
            "boundary",
        ),
    ],
)
def test_strict_append_only_rejects_unproven_context(turn, message):
    manager = TrajectoryManager(strict_append_only=True)

    with pytest.raises(ValueError, match=message):
        manager.record_turn(
            "strict-invalid",
            turn=turn,
            prompt_messages=[{"role": "user", "content": "question"}],
            response_message={"role": "assistant", "content": "answer"},
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
