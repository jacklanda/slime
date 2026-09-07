import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.data import rollout_logprob_dtype
from slime.backends.sglang_utils.arguments import validate_args as validate_sglang_args
from slime.rollout import prefill_logprobs
from slime.utils.ppo_utils import (
    _calculate_log_probs_and_entropy_true_on_policy,
    _gather_true_on_policy_full_logits,
    _prepare_true_on_policy_full_logits,
    _split_replicated_loss_gather_grad,
    calculate_log_probs_and_entropy,
)
from slime.utils.types import Sample


def test_true_on_policy_rollout_logprob_dtype_matches_training_precision():
    assert rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=True, fp16=False)) is torch.bfloat16
    assert rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=False, fp16=True)) is torch.float16
    assert rollout_logprob_dtype(Namespace(true_on_policy_mode=False, bf16=True, fp16=False)) is torch.float32


def test_true_on_policy_full_vocab_gather_trims_padding_and_splits_gradient():
    shards = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
    full_logits = _prepare_true_on_policy_full_logits(shards, vocab_size=3)
    assert full_logits.tolist() == [[1.0, 2.0, 3.0]]

    grad = torch.arange(8).view(2, 4)
    local_grad = _split_replicated_loss_gather_grad(
        grad,
        rank=1,
        world_size=2,
        local_last_dim=2,
    )
    assert local_grad.tolist() == [[2, 3], [6, 7]]


def test_true_on_policy_logprob_excludes_padded_vocab(monkeypatch):
    monkeypatch.setenv("SLIME_SGLANG_BATCH_INVARIANT_LOGPROB", "1")
    logits = torch.tensor([[1.0, 2.0, 3.0, 100.0]], dtype=torch.bfloat16, requires_grad=True)
    tokens = torch.tensor([2])

    log_probs, entropy = calculate_log_probs_and_entropy(
        logits,
        tokens,
        tp_group=None,
        with_entropy=False,
        true_on_policy=True,
        vocab_size=3,
    )

    expected = torch.log_softmax(logits[:, :3], dim=-1)[:, 2]
    torch.testing.assert_close(log_probs, expected, rtol=0.0, atol=0.0)
    assert entropy is None
    log_probs.sum().backward()
    assert logits.grad is not None
    assert logits.grad[0, 3] == 0


def test_true_on_policy_full_vocab_softmax_never_uses_default_world_group(monkeypatch):
    logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    tokens = torch.tensor([1])

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fail_on_world_group(*args, **kwargs):
        raise AssertionError("full-vocabulary true-on-policy loss used the default WORLD group")

    monkeypatch.setattr(torch.distributed, "get_world_size", fail_on_world_group)
    monkeypatch.setattr(torch.distributed, "all_reduce", fail_on_world_group)

    gathered = _gather_true_on_policy_full_logits(logits, None, vocab_size=3)
    log_probs, entropy = _calculate_log_probs_and_entropy_true_on_policy(
        gathered,
        tokens,
        None,
        with_entropy=True,
        vocab_size=3,
    )

    expected_full = torch.log_softmax(logits, dim=-1)
    torch.testing.assert_close(log_probs, expected_full[:, 1])
    torch.testing.assert_close(entropy, -(expected_full.exp() * expected_full).sum(dim=-1))
    log_probs.sum().backward()
    assert logits.grad is not None


def test_true_on_policy_derives_sglang_deterministic_contract():
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=1,
        sglang_pp_size=1,
        sglang_dp_size=1,
        true_on_policy_mode=True,
        sglang_enable_deterministic_inference=False,
        sglang_rl_on_policy_target=None,
        sglang_attention_backend="triton",
        sglang_router_ip=None,
        prefill_num_servers=None,
        rollout_external=False,
        sglang_config=None,
    )

    validate_sglang_args(args)

    assert args.sglang_enable_deterministic_inference is True
    assert args.sglang_enable_prefill_only_deterministic_inference is True
    assert args.sglang_true_on_policy_contract == "qwen3_dense_true_on_policy_v1"
    assert args.sglang_rl_on_policy_target is None
    assert args.sglang_attention_backend == "fa3"


def test_prefill_recompute_uses_response_tail_and_validates_tokens(monkeypatch):
    sample = Sample(
        tokens=[10, 11, 12, 20, 21],
        response_length=2,
        rollout_log_probs=[-9.0, -9.0],
        status=Sample.Status.COMPLETED,
    )
    calls = []

    async def fake_post(url, payload, headers=None):
        calls.append((url, payload, headers))
        if url.endswith("/flush_cache"):
            return {}
        return {
            "meta_info": {
                "input_token_logprobs": [(None, 12), (-0.25, 20), (-0.5, 21)]
            }
        }

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)
    args = SimpleNamespace(recompute_logprobs_via_prefill=True, router_policy="consistent_hashing")

    asyncio.run(
        prefill_logprobs.recompute_rollout_logprobs_via_prefill(
            args,
            [sample],
            url="http://localhost/generate",
            sampling_params={"temperature": 1.0, "max_new_tokens": 128},
        )
    )

    assert sample.rollout_log_probs == [-0.25, -0.5]
    assert sample.metadata["rollout_log_probs_source"] == "sglang_prefill_recompute"
    assert calls[0][0] == "http://localhost/flush_cache"
    assert calls[1][1]["logprob_start_len"] == 2
    assert calls[1][1]["sampling_params"]["max_new_tokens"] == 0
    assert calls[1][1]["sampling_params"]["temperature"] == 0


def test_prefill_recompute_rejects_token_misalignment(monkeypatch):
    sample = Sample(tokens=[10, 11, 20], response_length=1, status=Sample.Status.COMPLETED)

    async def fake_post(url, payload, headers=None):
        if url.endswith("/flush_cache"):
            return {}
        return {"meta_info": {"input_token_logprobs": [(None, 11), (-0.1, 999)]}}

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)
    args = SimpleNamespace(recompute_logprobs_via_prefill=True, router_policy="consistent_hashing")

    with pytest.raises(ValueError, match="token alignment mismatch"):
        asyncio.run(
            prefill_logprobs.recompute_rollout_logprobs_via_prefill(
                args,
                [sample],
                url="http://localhost/generate",
                sampling_params={},
            )
        )


@pytest.mark.parametrize(
    "launcher_name",
    ["train_odyssey_qwen3_sync.sh", "train_odyssey_qwen3_multinode_sync.sh"],
)
def test_odyssey_launcher_exposes_one_true_on_policy_switch(launcher_name):
    from pathlib import Path

    launcher = (Path(__file__).resolve().parents[1] / "experiments" / launcher_name).read_text()
    assert 'TRUE_ON_POLICY="${TRUE_ON_POLICY:-true}"' in launcher
    assert "--true-on-policy) TRUE_ON_POLICY=true; shift ;;" in launcher
    assert "--true-on-policy-mode" in launcher
    assert "--recompute-logprobs-via-prefill" in launcher
    assert "--sglang-enable-prefill-only-deterministic-inference" in launcher
    assert "--sglang-true-on-policy-contract qwen3_dense_true_on_policy_v1" in launcher
    assert "--sglang-attention-backend fa3" in launcher
    assert "--batch-invariant-mode" not in launcher
    assert 'ROLLOUT_NUM_GPUS_PER_ENGINE="${TP_SIZE}"' in launcher
    assert 'if is_truthy "${TRUE_ON_POLICY}" && [ "${TP_SIZE}" -ne "${ROLLOUT_NUM_GPUS_PER_ENGINE}" ]; then' in launcher
    assert 'SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-${TRUE_ON_POLICY}}"' in launcher
    assert 'f"{os.environ[\'MILES_PATH\']}:"' in launcher
    assert 'env["NCCL_ALGO"] = "Ring"' in launcher
