from __future__ import annotations

import asyncio

import pytest

from slime.rollout.rm_hub.benchmark_verifier import _extract_final_answer, reward_func
from slime.utils.types import Sample

NUM_GPUS = 0


def _sample(response: str, label, metadata: dict) -> Sample:
    return Sample(response=response, label=label, metadata=metadata)


@pytest.mark.unit
def test_browsecomp_plus_extracts_finish_result_for_token_f1():
    response = (
        '<tool_call>{"name":"finish","arguments":{"command":"submit",'
        '"result":"Ada Lovelace"}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label={"ground_truth": {"metric": "token_f1", "target": "Ada Lovelace"}},
        metadata={"data_source": "browsecomp_plus"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_browsecomp_plus_extracts_boxed_inside_finish_result():
    response = (
        '<tool_call>{"name":"finish","arguments":{"command":"submit",'
        '"result":"The final answer is \\\\boxed{Ada Lovelace}."}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label={"ground_truth": {"metric": "token_f1", "target": "Ada Lovelace"}},
        metadata={"data_source": "browsecomp_plus"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_browsecomp_plus_extracts_raw_boxed_before_answer_tag_fallback():
    response = "<answer>Wrong Person</answer>\nFinal: \\boxed{Ada Lovelace}"
    sample = _sample(
        response=response,
        label={"ground_truth": {"metric": "token_f1", "target": "Ada Lovelace"}},
        metadata={"data_source": "browsecomp_plus"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_browsecomp_plus_keeps_answer_tag_as_compatibility_fallback():
    sample = _sample(
        response="<answer>Ada Lovelace</answer>",
        label={"ground_truth": {"metric": "token_f1", "target": "Ada Lovelace"}},
        metadata={"data_source": "browsecomp_plus"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_browsecomp_plus_uses_token_f1_for_partial_overlap():
    sample = _sample(
        response="<answer>The title of the book is Grafias</answer>",
        label={"ground_truth": {"metric": "token_f1", "target": "Jaja of Opobo: The slave who became a king"}},
        metadata={"data_source": "browsecomp_plus"},
    )

    assert asyncio.run(reward_func(None, sample)) > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("data_source", ["2wiki", "bamboogle", "simpleqa_verified", "musique"])
def test_short_answer_benchmarks_extract_finish_result(data_source):
    response = (
        '<tool_call>{"name":"finish","arguments":{"command":"submit",'
        '"result":"Marie Curie"}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label="Marie Curie",
        metadata={"data_source": data_source},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_bamboogle_strict_exact_match_rejects_answer_in_explanatory_text():
    response = (
        '<tool_call>{"name":"finish","arguments":{"command":"submit",'
        '"result":"The answer is Pine Tree State"}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label="Pine Tree State",
        metadata={"data_source": "bamboogle", "strict_exact_match": True},
    )

    assert asyncio.run(reward_func(None, sample)) == 0.0


@pytest.mark.unit
def test_bamboogle_strict_exact_match_keeps_standard_normalization():
    response = (
        '<tool_call>{"name":"finish","arguments":{"command":"submit",'
        '"result":"The Pine Tree State"}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label="pine tree state",
        metadata={"data_source": "bamboogle", "strict_exact_match": True},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "response",
    [
        r"\boxed{Pine Tree State}",
        "<answer>Pine Tree State</answer>",
        "Final answer: Pine Tree State",
    ],
)
def test_bamboogle_strict_exact_match_accepts_supported_final_answer_formats(response):
    sample = _sample(
        response=response,
        label="Pine Tree State",
        metadata={"data_source": "bamboogle", "strict_exact_match": True},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("data_source", ["2wiki", "bamboogle", "simpleqa_verified", "musique"])
def test_short_answer_benchmarks_extract_raw_boxed_answer(data_source):
    sample = _sample(
        response="<think>checking evidence</think>\nFinal: \\boxed{Marie Curie}",
        label="Marie Curie",
        metadata={"data_source": data_source},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_short_answer_benchmarks_accept_submit_alias_as_finish():
    response = (
        '<tool_call>{"name":"submit","arguments":{"command":"submit",'
        '"result":"Marie Curie"}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label="Marie Curie",
        metadata={"data_source": "simpleqa_verified"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_benchmark_verifier_rejects_detected_eval_response_anomaly():
    response = (
        '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"Marie Curie"}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label="Marie Curie",
        metadata={"data_source": "bamboogle", "eval_response_anomalies": ["ngram_repetition"]},
    )

    assert asyncio.run(reward_func(None, sample)) == 0.0


@pytest.mark.unit
def test_short_answer_benchmarks_accept_plain_final_response_after_parse_error():
    sample = _sample(
        response="The machine used to extract honey from honeycombs uses **centrifugal force**.",
        label=["Centrifugal Force"],
        metadata={"data_source": "bamboogle"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_short_answer_benchmarks_do_not_score_unsubmitted_search_query():
    response = (
        '<tool_call>{"name":"web_search","arguments":'
        "{\"query\":\"director age Let's Make Money Short Term 12\",\"max_results\":3}}</tool_call>"
    )
    sample = _sample(
        response=response,
        label="Let'S Make Money",
        metadata={"data_source": "2wiki"},
    )

    assert asyncio.run(reward_func(None, sample)) == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "label"),
    [
        ("Final: \\boxed{\\textbf{Paris}}", "Paris"),
        ("Final: \\boxed{\\mathrm{D}}", "D"),
        ("Final: \\boxed{\\text{120,000 euros}}", "120000"),
    ],
)
def test_short_answer_benchmarks_strip_latex_wrappers_and_units(response, label):
    sample = _sample(
        response=response,
        label=label,
        metadata={"data_source": "simpleqa_verified"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("prediction", "label"),
    [
        ("Final: \\boxed{1,142}", "1142"),
        ("Final: \\boxed{120,000 euros}", "120000"),
        ("Final: \\boxed{12/03/1988}", "03/12/1988"),
    ],
)
def test_short_answer_benchmarks_normalize_numbers_units_and_dates(prediction, label):
    sample = _sample(
        response=prediction,
        label=label,
        metadata={"data_source": "simpleqa_verified"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_submitted_answer_is_not_truncated_by_plain_text_cascade():
    response = (
        '<tool_call>{"name":"finish","arguments":{"command":"submit",'
        '"result":"Short Term 12"}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label="Short Term 12",
        metadata={"data_source": "2wiki"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_scienceqa_extracts_boxed_choice_letter():
    sample = _sample(
        response="<think>A and B are less significant</think>Final: \\boxed{D}",
        label="D",
        metadata={"data_source": "scienceqa", "type": "选择题"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_scienceqa_extracts_finish_choice_letter():
    response = (
        '<tool_call>{"name":"finish","arguments":{"command":"submit",'
        '"result":"D"}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label="D",
        metadata={"data_source": "scienceqa", "type": "选择题"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_scienceqa_extracts_final_choice_commitment():
    sample = _sample(
        response="After checking the options, I will go with option D.",
        label="D",
        metadata={"data_source": "scienceqa", "type": "选择题"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_medqa_extracts_letter_from_finish_result():
    response = (
        '<tool_call>{"name":"finish","arguments":{"command":"submit",'
        '"result":"The answer is C"}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label="C",
        metadata={"data_source": "medqa", "options": ["alpha", "beta", "gamma", "delta"]},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_medqa_extracts_boxed_letter():
    sample = _sample(
        response="<think>B is wrong</think>Final: \\boxed{C}",
        label="C",
        metadata={"data_source": "medqa", "options": ["alpha", "beta", "gamma", "delta"]},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_medqa_accepts_plain_final_option_text():
    sample = _sample(
        response=(
            "Based on the information provided, the most likely diagnosis is "
            "**D. Tracheobronchial rupture**."
        ),
        label="Tracheobronchial rupture",
        metadata={
            "data_source": "medqa",
            "options": [
                "A. Tension pneumothorax",
                "B. Diaphragmatic rupture",
                "C. Pulmonary contusion",
                "D. Tracheobronchial rupture",
            ],
        },
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_medqa_maps_bare_letter_to_option_text_label():
    sample = _sample(
        response='<tool_call>{"name":"finish","arguments":{"command":"submit","result":"D"}}</tool_call>',
        label="Tracheobronchial rupture",
        metadata={
            "data_source": "medqa",
            "options": [
                "A. Tension pneumothorax",
                "B. Diaphragmatic rupture",
                "C. Pulmonary contusion",
                "D. Tracheobronchial rupture",
            ],
        },
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_gpqa_extracts_finish_result():
    response = (
        '<tool_call>{"name":"finish","arguments":{"command":"submit",'
        '"result":"D"}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label="D",
        metadata={"data_source": "gpqa_diamond"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_gpqa_extracts_raw_boxed_letter():
    sample = _sample(
        response="<think>A is tempting</think>Final: \\boxed{D}",
        label="D",
        metadata={"data_source": "gpqa_diamond"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_gpqa_does_not_extract_letter_from_unsubmitted_search_query():
    response = (
        '<tool_call>{"name":"web_search","arguments":'
        '{"query":"stellar photosphere LTE Ti energy levels ratio change with spots",'
        '"max_results":3}}</tool_call>'
    )
    sample = _sample(
        response=response,
        label="C",
        metadata={"data_source": "gpqa_diamond"},
    )

    assert asyncio.run(reward_func(None, sample)) == 0.0


@pytest.mark.unit
def test_gpqa_extracts_rich_final_choice_commitment():
    sample = _sample(
        response="The evidence points to the cold atomic medium. My final choice is C.",
        label="C",
        metadata={"data_source": "gpqa_diamond"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_gpqa_ignores_tentative_choice_commitment():
    sample = _sample(
        response="If the answer is C, the premise would fail. Therefore, the final answer is D.",
        label="D",
        metadata={"data_source": "gpqa_diamond"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_frontierscience_olympiad_scores_short_final_answer():
    sample = _sample(
        response="Derivation omitted.\nFINAL ANSWER: \\(2.31 \\times 10^6 K\\)",
        label="`\\( 2.31 \\times 10^6 K\\)`",
        metadata={"data_source": "frontierscience_olympiad"},
    )

    assert asyncio.run(reward_func(None, sample)) == 1.0


@pytest.mark.unit
def test_frontierscience_research_scores_long_answer_with_rouge_l():
    reference = "Preselect the initial state. Postselect a nearly orthogonal final state. The amplification factor follows from the weak value."
    sample = _sample(
        response=(
            '<tool_call>{"name":"finish","arguments":{"command":"submit",'
            '"result":"Preselect the initial state, then postselect a nearly orthogonal final state."}}</tool_call>'
        ),
        label=reference,
        metadata={"data_source": "frontierscience_research"},
    )

    reward = asyncio.run(reward_func(None, sample))

    assert 0.0 < reward < 1.0


@pytest.mark.unit
def test_extract_final_answer_supports_nested_boxed_braces():
    assert _extract_final_answer("Final: \\boxed{answer {with braces}}") == "answer {with braces}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
