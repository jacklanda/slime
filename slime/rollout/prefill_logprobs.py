from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


def _build_prefill_scoring_payload(
    sample: Sample,
    sampling_params: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_len = len(sample.tokens) - sample.response_length
    if prompt_len <= 0:
        raise ValueError(
            "Cannot recompute rollout logprobs via prefill without a prompt token: "
            f"tokens={len(sample.tokens)}, response_length={sample.response_length}"
        )

    payload: dict[str, Any] = {
        "input_ids": sample.tokens,
        "sampling_params": {
            **dict(sampling_params),
            "max_new_tokens": 0,
            "temperature": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # Include the token immediately before the response because SGLang's
        # first input logprob is None.
        "logprob_start_len": prompt_len - 1,
    }
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        payload["image_data"] = [
            encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
        ]
    return payload


def _build_batch_prefill_scoring_payload(
    samples: list[Sample],
    sampling_params: Mapping[str, Any],
) -> dict[str, Any]:
    payloads = [_build_prefill_scoring_payload(sample, sampling_params) for sample in samples]
    logprob_start_len = payloads[0]["logprob_start_len"]
    if any(payload["logprob_start_len"] != logprob_start_len for payload in payloads):
        raise ValueError("Batched SGLang prefill scoring requires a shared logprob_start_len")
    return {
        "input_ids": [payload["input_ids"] for payload in payloads],
        "sampling_params": payloads[0]["sampling_params"],
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
    }


def _extract_response_logprobs(sample: Sample, meta_info: Mapping[str, Any]) -> list[float]:
    input_token_logprobs = meta_info.get("input_token_logprobs")
    if not input_token_logprobs:
        raise ValueError("SGLang prefill scoring response did not include input_token_logprobs")

    response_items = input_token_logprobs[-sample.response_length :]
    response_tokens = sample.tokens[-sample.response_length :]
    scored_tokens = [item[1] for item in response_items]
    if scored_tokens != response_tokens:
        raise ValueError(
            "SGLang prefill scoring token alignment mismatch: "
            f"expected {response_tokens[:8]} (len={len(response_tokens)}), "
            f"got {scored_tokens[:8]} (len={len(scored_tokens)})"
        )

    response_logprobs = [item[0] for item in response_items]
    if any(logprob is None for logprob in response_logprobs):
        raise ValueError("SGLang prefill scoring returned None for a response-token logprob")
    return response_logprobs


async def _score_one(
    sample: Sample,
    *,
    url: str,
    sampling_params: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> None:
    payload = _build_prefill_scoring_payload(sample, sampling_params)
    output = await post(url, payload, headers=headers)
    sample.rollout_log_probs = _extract_response_logprobs(sample, output["meta_info"])
    sample.metadata["rollout_log_probs_source"] = "sglang_prefill_recompute"


async def recompute_rollout_logprobs_via_prefill(
    args: Any,
    samples: list[Sample],
    *,
    url: str,
    sampling_params: Mapping[str, Any],
) -> None:
    """Replace decode logprobs with clean-prefill scores for accepted samples."""
    if not getattr(args, "recompute_logprobs_via_prefill", False):
        return

    samples = [
        sample
        for sample in samples
        if sample.response_length and sample.status != Sample.Status.ABORTED
    ]
    if not samples:
        return

    flush_url = url.rsplit("/", 1)[0] + "/flush_cache"
    use_consistent_hashing = getattr(args, "router_policy", None) == "consistent_hashing"
    has_images = any(sample.multimodal_inputs and sample.multimodal_inputs.get("images") for sample in samples)

    if not use_consistent_hashing and not has_images:
        samples_by_start_len: dict[int, list[Sample]] = defaultdict(list)
        for sample in samples:
            samples_by_start_len[len(sample.tokens) - sample.response_length - 1].append(sample)
        for batch_samples in samples_by_start_len.values():
            await post(flush_url, {})
            outputs = await post(url, _build_batch_prefill_scoring_payload(batch_samples, sampling_params))
            if not isinstance(outputs, list) or len(outputs) != len(batch_samples):
                raise ValueError(
                    "SGLang batch prefill scoring output count mismatch: "
                    f"expected {len(batch_samples)}, got {len(outputs) if isinstance(outputs, list) else type(outputs).__name__}"
                )
            for sample, output in zip(batch_samples, outputs, strict=True):
                sample.rollout_log_probs = _extract_response_logprobs(sample, output["meta_info"])
                sample.metadata["rollout_log_probs_source"] = "sglang_prefill_recompute"
        return

    for sample in samples:
        headers = {"X-SMG-Routing-Key": sample.session_id} if use_consistent_hashing and sample.session_id else None
        await post(flush_url, {}, headers=headers)
        await _score_one(sample, url=url, sampling_params=sampling_params, headers=headers)
