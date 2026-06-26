from argparse import Namespace

from slime.ray.rollout import convert_samples_to_train_data
from slime.utils.credit_assignment import CreditAssignmentConfig, build_policy_loss_mask
from slime.utils.types import Sample


def _args(**overrides):
    values = dict(
        credit_assignment_enable=True,
        credit_assignment_tool_parser_error=True,
        credit_assignment_repeated_search_query=True,
        credit_assignment_too_many_tool_calls=True,
        credit_assignment_search_bypass=True,
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
