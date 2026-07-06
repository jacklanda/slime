from argparse import Namespace

import pytest

from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.ray.rollout import convert_samples_to_train_data
from slime.utils.credit_assignment import CreditAssignmentConfig, build_policy_loss_mask
from slime.utils.types import Sample


NUM_GPUS = 0


def _args(**overrides):
    values = dict(
        credit_assignment_enable=True,
        credit_assignment_tool_parser_error=True,
        credit_assignment_repeated_search_query=True,
        credit_assignment_too_many_tool_calls=True,
        credit_assignment_search_bypass=True,
        credit_assignment_mixed_tool_and_answer=True,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=1,
        rollout_batch_size=1,
        grpo_std_normalization=True,
        rollout_top_p=1.0,
        use_rollout_logprobs=False,
        use_rollout_routing_replay=False,
        use_opd=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_credit_assignment_masks_only_valid_action_span():
    config = CreditAssignmentConfig(enable=True, tool_parser_error=True)
    mask, event = build_policy_loss_mask(
        metadata={
            "tool_call_parse_error": True,
            "credit_assignment_action_start": 2,
            "credit_assignment_action_end": 5,
        },
        loss_mask=[1, 1, 0, 1, 1, 1],
        config=config,
    )

    assert event == "tool_parser_error"
    assert mask == [0, 0, 0, 1, 1, 0]


def test_credit_assignment_invalid_span_masks_all_policy_tokens():
    config = CreditAssignmentConfig(enable=True, repeated_search_query=True)
    mask, event = build_policy_loss_mask(
        metadata={
            "repeated_query": True,
            "credit_assignment_action_start": 4,
            "credit_assignment_action_end": 99,
        },
        loss_mask=[1, 1, 1, 1],
        config=config,
    )

    assert event == "repeated_search_query"
    assert mask == [0, 0, 0, 0]


def test_credit_assignment_missing_span_keeps_existing_action_mask():
    config = CreditAssignmentConfig(enable=True, too_many_tool_calls=True)
    mask, event = build_policy_loss_mask(
        metadata={"termination_reason": "ABNORMAL_TOOL_BURST"},
        loss_mask=[0, 1, 1, 0],
        config=config,
    )

    assert event == "too_many_tool_calls"
    assert mask == [0, 1, 1, 0]


def test_credit_assignment_search_bypass_keeps_full_action_mask():
    config = CreditAssignmentConfig(enable=True, search_bypass=True)
    mask, event = build_policy_loss_mask(
        metadata={"search_bypass": True},
        loss_mask=[1, 0, 1, 1],
        config=config,
    )

    assert event == "search_bypass"
    assert mask == [1, 0, 1, 1]


def test_credit_assignment_metadata_event_is_honored_for_fused_events():
    config = CreditAssignmentConfig(enable=True)

    mask, event = build_policy_loss_mask(
        metadata={
            "credit_assignment_event": "max_turns_exceeded",
            "credit_assignment_action_start": 1,
            "credit_assignment_action_end": 4,
        },
        loss_mask=[1, 0, 1, 1, 1],
        config=config,
    )
    assert event == "max_turns_exceeded"
    assert mask == [0, 0, 1, 1, 0]

    mask, event = build_policy_loss_mask(
        metadata={"credit_assignment_event": "direct_submit_without_tool"},
        loss_mask=[1, 0, 1],
        config=config,
    )
    assert event == "direct_submit_without_tool"
    assert mask == [0, 0, 0]

    mask, event = build_policy_loss_mask(
        metadata={"credit_assignment_event": "tail_guard_early_stop"},
        loss_mask=[1, 1, 0],
        config=config,
    )
    assert event == "tail_guard_early_stop"
    assert mask == [0, 0, 0]


def test_credit_assignment_mixed_tool_and_answer_config_reason():
    config = CreditAssignmentConfig(enable=True, mixed_tool_and_answer=True)
    mask, event = build_policy_loss_mask(
        metadata={
            "termination_reason": "ABNORMAL_MIXED_TOOL_AND_ANSWER",
            "credit_assignment_action_start": 2,
            "credit_assignment_action_end": 5,
        },
        loss_mask=[1, 1, 0, 1, 1, 1],
        config=config,
    )

    assert event == "mixed_tool_and_answer"
    assert mask == [0, 0, 0, 1, 1, 0]


def test_rollout_conversion_separates_advantage_and_policy_masks():
    samples = [
        Sample(
            index=0,
            rollout_id=7,
            tokens=[10, 20, 1, 2, 3, 4],
            response_length=4,
            reward=1.0,
            loss_mask=[1, 1, 1, 1],
            metadata={
                "tool_call_parse_error": True,
                "credit_assignment_action_start": 1,
                "credit_assignment_action_end": 3,
            },
        ),
        Sample(
            index=1,
            rollout_id=7,
            tokens=[10, 20, 5, 6, 7],
            response_length=3,
            reward=0.0,
            loss_mask=[1, 0, 1],
            metadata={"search_bypass": True},
        ),
    ]

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["loss_masks"] == [[1, 1, 1, 1], [1, 0, 1]]
    assert train_data["policy_loss_masks"] == [[0, 1, 1, 0], [1, 0, 1]]
    assert train_data["rollout_mask_sums"] == [6, 6]
    assert train_data["policy_rollout_mask_sums"] == [4, 4]
    assert train_data["episode_metrics_data"]["loss_mask_sums"] == [4, 2]
    assert train_data["episode_metrics_data"]["policy_loss_mask_sums"] == [2, 2]


def test_rollout_conversion_preserves_explicit_tito_policy_masks_across_split_samples():
    samples = [
        Sample(
            index=10,
            rollout_id=99,
            tokens=[1, 2, 701, 702, 703],
            response_length=3,
            reward=1.0,
            loss_mask=[1, 1, 1],
            policy_loss_mask=[0, 1, 0],
            metadata={"credit_assignment_event": "tool_parser_error"},
        ),
        Sample(
            index=10,
            rollout_id=99,
            tokens=[9, 9, 801, 802, 803, 804],
            response_length=4,
            reward=1.0,
            loss_mask=[1, 0, 1, 1],
            policy_loss_mask=[0, 0, 1, 0],
            metadata={"credit_assignment_event": "max_turns_exceeded"},
        ),
        Sample(
            index=11,
            rollout_id=100,
            tokens=[5, 6, 901, 902],
            response_length=2,
            reward=0.0,
            loss_mask=[1, 1],
            # No explicit policy mask: because another sample has one, this
            # sample should still get a concrete fallback mask equal to loss_mask.
            metadata={},
        ),
    ]

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["loss_masks"] == [[1, 1, 1], [1, 0, 1, 1], [1, 1]]
    assert train_data["policy_loss_masks"] == [[0, 1, 0], [0, 0, 1, 0], [1, 1]]
    assert train_data["rollout_mask_sums"] == [6, 6, 2]
    assert train_data["policy_rollout_mask_sums"] == [2, 2, 2]
    assert train_data["episode_metrics_data"]["policy_loss_mask_sums"] == [1, 1, 2]
    assert train_data["episode_metrics_data"]["credit_assignment_events"] == [
        "tool_parser_error",
        "max_turns_exceeded",
        None,
    ]


def test_rollout_conversion_mixes_metadata_explicit_and_zero_policy_masks_per_rollout():
    samples = [
        Sample(
            index=12,
            rollout_id=1200,
            tokens=[1, 2, 301, 302, 303, 304],
            response_length=4,
            reward=0.0,
            loss_mask=[1, 1, 0, 1],
            metadata={
                "credit_assignment_event": "repeated_search_query",
                "credit_assignment_action_start": 1,
                "credit_assignment_action_end": 4,
            },
        ),
        Sample(
            index=12,
            rollout_id=1200,
            tokens=[3, 4, 401, 402, 403],
            response_length=3,
            reward=0.0,
            loss_mask=[1, 0, 1],
            policy_loss_mask=[1, 1, 0],
            metadata={"credit_assignment_event": "tool_parser_error"},
        ),
        Sample(
            index=12,
            rollout_id=1200,
            tokens=[5, 6, 501, 502],
            response_length=2,
            reward=0.0,
            loss_mask=[0, 0],
            metadata={"credit_assignment_event": "tail_guard_early_stop"},
        ),
        Sample(
            index=13,
            rollout_id=1300,
            tokens=[7, 8, 601, 602, 603],
            response_length=3,
            reward=1.0,
            loss_mask=[1, 1, 1],
            metadata={},
        ),
    ]

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["loss_masks"] == [[1, 1, 0, 1], [1, 0, 1], [0, 0], [1, 1, 1]]
    assert train_data["policy_loss_masks"] == [[0, 1, 0, 1], [1, 0, 0], [0, 0], [1, 1, 1]]
    assert train_data["rollout_mask_sums"] == [5, 5, 5, 3]
    assert train_data["policy_rollout_mask_sums"] == [3, 3, 3, 3]
    assert train_data["episode_metrics_data"]["policy_loss_mask_sums"] == [2, 1, 0, 3]
    assert train_data["episode_metrics_data"]["credit_assignment_events"] == [
        "repeated_search_query",
        "tool_parser_error",
        "tail_guard_early_stop",
        None,
    ]


def test_explicit_policy_masks_survive_when_metadata_credit_assignment_is_disabled():
    samples = [
        Sample(
            index=20,
            rollout_id=200,
            tokens=[1, 2, 701, 702, 703],
            response_length=3,
            reward=0.0,
            loss_mask=[1, 1, 1],
            policy_loss_mask=[0, 1, 0],
            metadata={"credit_assignment_event": "tool_parser_error"},
        ),
        Sample(
            index=21,
            rollout_id=200,
            tokens=[3, 4, 801, 802, 803, 804],
            response_length=4,
            reward=0.0,
            loss_mask=[1, 1, 0, 1],
            metadata={
                "credit_assignment_event": "max_turns_exceeded",
                "credit_assignment_action_start": 1,
                "credit_assignment_action_end": 3,
            },
        ),
    ]

    train_data = convert_samples_to_train_data(_args(credit_assignment_enable=False), samples)

    assert train_data["loss_masks"] == [[1, 1, 1], [1, 1, 0, 1]]
    assert train_data["policy_loss_masks"] == [[0, 1, 0], [1, 1, 0, 1]]
    assert train_data["rollout_mask_sums"] == [6, 6]
    assert train_data["policy_rollout_mask_sums"] == [4, 4]
    assert train_data["episode_metrics_data"]["credit_assignment_events"] == [
        "tool_parser_error",
        None,
    ]


def test_trajectory_tito_split_policy_masks_are_only_narrowed_by_trainer_conversion():
    manager = TrajectoryManager()
    sid = "tito-split-to-trainer-policy-mask"
    first_prompt = [1, 2]
    first_response = [701, 702]
    boundary_delta = [9, 10, 11]
    boundary_response = [801, 802, 803]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=first_prompt,
            context_delta_ids=first_prompt,
            output_ids=first_response,
            output_log_probs=[-0.1, -0.2],
            loss_mask=[1, 0],
            # Deliberately stale on token 2. TITO should preserve explicit
            # rollout evidence; trainer conversion is the final narrowing gate.
            policy_loss_mask=[0, 1],
            finish_reason="tool_calls",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a1"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=[123, *boundary_delta],
            context_delta_ids=boundary_delta,
            tito_boundary_before=True,
            output_ids=boundary_response,
            output_log_probs=[-0.3, -0.4, -0.5],
            loss_mask=[1, 1, 0],
            # Deliberately stale on token 3; conversion must not unmask it.
            policy_loss_mask=[1, 0, 1],
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "tool"},
        ],
        response_message={"role": "assistant", "content": "a2"},
    )

    samples = manager.get_trajectory(
        sid,
        base_sample=Sample(index=3, rollout_id=303, prompt=""),
        reward=2.0,
        extra_metadata={"credit_assignment_event": "tool_parser_error"},
    )

    assert len(samples) == 2
    assert samples[0].loss_mask == [1, 0]
    assert samples[0].policy_loss_mask == [0, 1]
    assert samples[1].loss_mask == [1, 1, 0]
    assert samples[1].policy_loss_mask == [1, 0, 1]

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["loss_masks"] == [[1, 0], [1, 1, 0]]
    assert train_data["policy_loss_masks"] == [[0, 0], [1, 0, 0]]
    assert train_data["rollout_mask_sums"] == [3, 3]
    assert train_data["policy_rollout_mask_sums"] == [1, 1]
    assert train_data["episode_metrics_data"]["policy_loss_mask_sums"] == [0, 1]


def test_shared_tito_branch_replay_context_does_not_inflate_policy_rollout_denominator():
    manager = TrajectoryManager()
    sid = "shared-branch-to-trainer-mask-sums"
    prompt = [1, 2]
    shared_response = [701, 702, 703]
    branch_a_delta = [31]
    branch_a_response = [801, 802]
    branch_b_delta = [41, 42]
    branch_b_response = [901]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt,
            context_delta_ids=prompt,
            output_ids=shared_response,
            output_log_probs=[-0.1, -0.2, -0.3],
            loss_mask=[1, 1, 0],
            policy_loss_mask=[1, 0, 0],
            finish_reason="tool_calls",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "shared"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt + shared_response + branch_a_delta,
            context_delta_ids=branch_a_delta,
            output_ids=branch_a_response,
            output_log_probs=[-0.4, -0.5],
            loss_mask=[1, 1],
            policy_loss_mask=[0, 1],
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
            prompt_ids=prompt + shared_response + branch_b_delta,
            context_delta_ids=branch_b_delta,
            output_ids=branch_b_response,
            output_log_probs=[-0.6],
            loss_mask=[1],
            policy_loss_mask=[1],
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "shared"},
            {"role": "tool", "content": "tool-b"},
        ],
        response_message={"role": "assistant", "content": "branch-b"},
    )

    samples = manager.get_trajectory(
        sid,
        base_sample=Sample(index=4, rollout_id=404, prompt=""),
        reward=2.0,
        extra_metadata={"credit_assignment_event": "tool_parser_error"},
    )

    assert len(samples) == 2
    assert samples[0].loss_mask == [1, 1, 0, 0, 1, 1]
    assert samples[0].policy_loss_mask == [1, 0, 0, 0, 0, 1]
    assert samples[1].loss_mask == [0, 0, 0, 0, 0, 1]
    assert samples[1].policy_loss_mask is None

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["loss_masks"] == [[1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1]]
    assert train_data["policy_loss_masks"] == [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]
    assert train_data["rollout_mask_sums"] == [5, 5]
    assert train_data["policy_rollout_mask_sums"] == [3, 3]
    assert train_data["episode_metrics_data"]["loss_mask_sums"] == [4, 1]
    assert train_data["episode_metrics_data"]["policy_loss_mask_sums"] == [2, 1]


def test_metadata_action_span_over_tito_context_only_trains_unmasked_response_tokens():
    manager = TrajectoryManager()
    sid = "metadata-span-over-tito-context"
    prompt = [1, 2]
    response_1 = [701, 702]
    delta_2 = [31, 32, 33]
    response_2 = [801, 802, 803, 804]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt,
            context_delta_ids=prompt,
            output_ids=response_1,
            output_log_probs=[-0.1, -0.2],
            loss_mask=[1, 1],
            finish_reason="tool_calls",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a1"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt + response_1 + delta_2,
            context_delta_ids=delta_2,
            output_ids=response_2,
            output_log_probs=[-0.3, -0.4, -0.5, -0.6],
            loss_mask=[1, 0, 1, 1],
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "tool"},
        ],
        response_message={"role": "assistant", "content": "a2"},
    )

    samples = manager.get_trajectory(
        sid,
        base_sample=Sample(index=5, rollout_id=505, prompt=""),
        reward=1.0,
        extra_metadata={
            "credit_assignment_event": "max_turns_exceeded",
            # Response-region indices after stripping the first prompt:
            # [r1, r1, delta, delta, delta, r2, r2, r2, r2].
            # This span crosses context delta and a loss-masked response token.
            "credit_assignment_action_start": 2,
            "credit_assignment_action_end": 8,
        },
    )

    assert len(samples) == 1
    assert samples[0].loss_mask == [1, 1, 0, 0, 0, 1, 0, 1, 1]

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["loss_masks"] == [[1, 1, 0, 0, 0, 1, 0, 1, 1]]
    assert train_data["policy_loss_masks"] == [[0, 0, 0, 0, 0, 1, 0, 1, 0]]
    assert train_data["rollout_mask_sums"] == [5]
    assert train_data["policy_rollout_mask_sums"] == [2]
    assert train_data["episode_metrics_data"]["credit_assignment_events"] == ["max_turns_exceeded"]


def test_invalid_metadata_action_span_zeroes_policy_mask_without_dropping_tito_sample():
    manager = TrajectoryManager()
    sid = "invalid-span-tito-to-trainer"
    prompt = [1, 2]
    delta_2 = [31, 32]
    response_1 = [701, 702]
    response_2 = [801, 802, 803]

    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt,
            context_delta_ids=prompt,
            output_ids=response_1,
            output_log_probs=[-0.1, -0.2],
            loss_mask=[1, 1],
            finish_reason="tool_calls",
        ),
        prompt_messages=[{"role": "user", "content": "u"}],
        response_message={"role": "assistant", "content": "a1"},
    )
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt + response_1 + delta_2,
            context_delta_ids=delta_2,
            output_ids=response_2,
            output_log_probs=[-0.3, -0.4, -0.5],
            loss_mask=[1, 0, 1],
            finish_reason="stop",
        ),
        prompt_messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "tool"},
        ],
        response_message={"role": "assistant", "content": "a2"},
    )

    samples = manager.get_trajectory(
        sid,
        base_sample=Sample(index=8, rollout_id=808, prompt=""),
        reward=1.0,
        extra_metadata={
            "credit_assignment_event": "repeated_search_query",
            "credit_assignment_action_start": 3,
            "credit_assignment_action_end": 999,
        },
    )

    assert len(samples) == 1
    assert samples[0].loss_mask == [1, 1, 0, 0, 1, 0, 1]

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["loss_masks"] == [[1, 1, 0, 0, 1, 0, 1]]
    assert train_data["policy_loss_masks"] == [[0, 0, 0, 0, 0, 0, 0]]
    assert train_data["rollout_mask_sums"] == [4]
    assert train_data["policy_rollout_mask_sums"] == [0]
    assert train_data["episode_metrics_data"]["policy_loss_mask_sums"] == [0]
    assert train_data["episode_metrics_data"]["credit_assignment_events"] == ["repeated_search_query"]


def test_long_horizon_tito_metadata_action_span_survives_trainer_conversion():
    manager = TrajectoryManager()
    sid = "long-horizon-tito-to-trainer-action-span"
    prefix: list[int] = []
    expected_loss_mask: list[int] = []

    for turn_idx in range(1_005):
        delta = [10_000 + turn_idx]
        response = [700_000 + turn_idx]
        loss_mask = [0] if turn_idx % 101 == 0 else [1]
        prompt_ids = prefix + delta
        manager.record_turn(
            sid,
            turn=TurnRecord(
                prompt_ids=prompt_ids,
                context_delta_ids=delta,
                output_ids=response,
                output_log_probs=[-0.01 * (turn_idx + 1)],
                loss_mask=loss_mask,
                finish_reason="tool_calls" if turn_idx < 1_004 else "stop",
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
        prefix.extend(delta)
        prefix.extend(response)
        expected_loss_mask.extend([0])
        expected_loss_mask.extend(loss_mask)

    action_start = len(expected_loss_mask) - 12
    action_end = len(expected_loss_mask) - 2
    samples = manager.get_trajectory(
        sid,
        base_sample=Sample(index=6, rollout_id=606, prompt=""),
        reward=1.0,
        extra_metadata={
            "credit_assignment_event": "max_turns_exceeded",
            "credit_assignment_action_start": action_start,
            "credit_assignment_action_end": action_end,
        },
    )

    assert len(samples) == 1
    sample = samples[0]
    # The first strict delta is stripped as prompt; the response region starts
    # at the first generated token and includes later context deltas.
    assert sample.response_length == len(expected_loss_mask) - 1
    assert sample.loss_mask == expected_loss_mask[1:]

    train_data = convert_samples_to_train_data(_args(), samples)

    response_action_start = action_start - 1
    response_action_end = action_end - 1
    expected_policy_mask = [0] * sample.response_length
    for idx in range(response_action_start, response_action_end):
        expected_policy_mask[idx] = sample.loss_mask[idx]

    assert train_data["loss_masks"] == [sample.loss_mask]
    assert train_data["policy_loss_masks"] == [expected_policy_mask]
    assert train_data["rollout_mask_sums"] == [sum(sample.loss_mask)]
    assert train_data["policy_rollout_mask_sums"] == [sum(expected_policy_mask)]
    assert sum(expected_policy_mask) == 5


def test_rollout_conversion_intersects_explicit_policy_mask_with_loss_mask():
    samples = [
        Sample(
            index=0,
            rollout_id=101,
            tokens=[1, 2, 3, 4, 5],
            response_length=4,
            reward=0.0,
            loss_mask=[1, 0, 1, 0],
            # External/debug data may carry a stale policy mask. The trainer
            # conversion boundary must not let it train loss-masked tokens.
            policy_loss_mask=[1, 1, 0, 1],
            metadata={"credit_assignment_event": "tool_parser_error"},
        )
    ]

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["loss_masks"] == [[1, 0, 1, 0]]
    assert train_data["policy_loss_masks"] == [[1, 0, 0, 0]]
    assert train_data["policy_rollout_mask_sums"] == [1]
    assert train_data["episode_metrics_data"]["policy_loss_mask_sums"] == [1]


def test_rollout_conversion_remove_sample_zeros_loss_and_policy_masks_in_same_rollout():
    samples = [
        Sample(
            index=0,
            rollout_id=8,
            tokens=[1, 2, 3, 4],
            response_length=2,
            reward=1.0,
            loss_mask=[1, 1],
            policy_loss_mask=[1, 0],
        ),
        Sample(
            index=1,
            rollout_id=8,
            tokens=[5, 6, 7, 8, 9],
            response_length=3,
            reward=1.0,
            loss_mask=[1, 1, 1],
            policy_loss_mask=[1, 1, 1],
            remove_sample=True,
        ),
    ]

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["loss_masks"] == [[1, 1], [0, 0, 0]]
    assert train_data["policy_loss_masks"] == [[1, 0], [0, 0, 0]]
    assert train_data["rollout_mask_sums"] == [2, 2]
    assert train_data["policy_rollout_mask_sums"] == [1, 1]
    assert train_data["episode_metrics_data"]["remove_sample"] == [False, True]


def test_rollout_conversion_policy_rollout_sums_support_zero_policy_samples():
    samples = [
        Sample(
            index=0,
            rollout_id=42,
            tokens=[1, 2, 3],
            response_length=2,
            reward=0.0,
            loss_mask=[1, 1],
            policy_loss_mask=[0, 0],
            metadata={"credit_assignment_event": "direct_submit_without_tool"},
        ),
        Sample(
            index=1,
            rollout_id=42,
            tokens=[4, 5, 6, 7],
            response_length=3,
            reward=0.0,
            loss_mask=[1, 1, 1],
            policy_loss_mask=[0, 1, 0],
            metadata={"credit_assignment_event": "tool_parser_error"},
        ),
        Sample(
            index=2,
            rollout_id=43,
            tokens=[8, 9],
            response_length=1,
            reward=0.0,
            loss_mask=[1],
            policy_loss_mask=[0],
            metadata={"credit_assignment_event": "tail_guard_early_stop"},
        ),
    ]

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["rollout_mask_sums"] == [5, 5, 1]
    assert train_data["policy_rollout_mask_sums"] == [1, 1, 0]
    assert train_data["episode_metrics_data"]["policy_loss_mask_sums"] == [0, 1, 0]


def test_rollout_conversion_builds_policy_masks_from_metadata_credit_events():
    samples = [
        Sample(
            index=0,
            rollout_id=50,
            tokens=[1, 2, 3, 4],
            response_length=3,
            reward=0.0,
            loss_mask=[1, 1, 1],
            metadata={"credit_assignment_event": "direct_submit_without_tool"},
        ),
        Sample(
            index=1,
            rollout_id=50,
            tokens=[5, 6, 7, 8, 9],
            response_length=4,
            reward=0.0,
            loss_mask=[1, 0, 1, 1],
            metadata={
                "credit_assignment_event": "max_response_len_exceeded",
                "credit_assignment_action_start": 1,
                "credit_assignment_action_end": 4,
            },
        ),
    ]

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["policy_loss_masks"] == [[0, 0, 0], [0, 0, 1, 1]]
    assert train_data["rollout_mask_sums"] == [6, 6]
    assert train_data["policy_rollout_mask_sums"] == [2, 2]
    assert train_data["episode_metrics_data"]["credit_assignment_events"] == [
        "direct_submit_without_tool",
        "max_response_len_exceeded",
    ]


def test_fully_masked_tito_tail_guard_sample_survives_trainer_conversion():
    sid = "fully-masked-tito-tail-guard"
    manager = TrajectoryManager()
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=[10, 11],
            context_delta_ids=[],
            output_ids=[20, 21, 22],
            finish_reason="stop",
            loss_mask=[0, 0, 0],
            policy_loss_mask=[0, 0, 0],
            output_log_probs=[-0.1, -0.2, -0.3],
        ),
        prompt_messages=[{"role": "user", "content": "q"}],
        response_message={"role": "assistant", "content": "a"},
        metadata={"credit_assignment_event": "tail_guard_early_stop"},
    )

    samples = manager.get_trajectory(
        sid,
        base_sample=Sample(index=0, rollout_id=7, tokens=[10, 11], response_length=0, reward=0.0),
        reward=0.0,
        allow_fully_masked=True,
        extra_metadata={"credit_assignment_event": "tail_guard_early_stop"},
    )

    assert len(samples) == 1
    assert samples[0].loss_mask == [0, 0, 0]
    assert samples[0].policy_loss_mask is None

    train_data = convert_samples_to_train_data(_args(), samples)

    assert train_data["loss_masks"] == [[0, 0, 0]]
    assert train_data["policy_loss_masks"] == [[0, 0, 0]]
    assert train_data["rollout_mask_sums"] == [0]
    assert train_data["policy_rollout_mask_sums"] == [0]
    assert train_data["episode_metrics_data"]["loss_mask_sums"] == [0]
    assert train_data["episode_metrics_data"]["policy_loss_mask_sums"] == [0]
    assert train_data["episode_metrics_data"]["credit_assignment_events"] == ["tail_guard_early_stop"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
