import pytest

from xmodel_kv.cli.summarize_hotpot_agents import (
    _agent_a_gold_cells,
    _bootstrap_delta,
    _deduplicate_questions,
    _has_memo_schema,
    _paired_em_flips,
    _quantile,
)


def test_quantile_interpolates():
    assert _quantile([0.0, 1.0], 0.25) == pytest.approx(0.25)


def test_paired_delta_observed_value():
    rows = [
        {"a": {"f1": 1.0}, "b": {"f1": 0.0}},
        {"a": {"f1": 0.5}, "b": {"f1": 0.5}},
    ]
    result = _bootstrap_delta(rows, "f1", "a", "b", samples=20, seed=0)
    assert result["value"] == pytest.approx(0.5)


def test_paired_em_flips_uses_reference_direction():
    rows = [
        {"reference": {"em": 1.0}, "candidate": {"em": 0.0}},
        {"reference": {"em": 0.0}, "candidate": {"em": 1.0}},
        {"reference": {"em": 1.0}, "candidate": {"em": 0.0}},
    ]
    result = _paired_em_flips(rows, "reference", "candidate")
    assert result["reference_correct_candidate_wrong"] == 2
    assert result["reference_wrong_candidate_correct"] == 1


def test_memo_schema_detection_is_case_insensitive():
    assert _has_memo_schema("BRIDGE: X\nEVIDENCE: Y\nNEXT_LOOKUP: Z")
    assert not _has_memo_schema("The answer is X")


def test_agent_a_gold_cells_split_rows():
    def row(contains_gold, identity_em, reuse_em):
        return {
            "agent_a": {"report_contains_gold": contains_gold},
            "direct_b": {"em": 0.0, "f1": 0.0},
            "identity_ab": {"em": identity_em, "f1": identity_em, "answer": "x"},
            "reuse_ab": {"em": reuse_em, "f1": reuse_em, "answer": "x"},
        }

    cells = _agent_a_gold_cells([row(True, 1.0, 0.0), row(False, 0.0, 1.0)])
    assert cells[0]["report_contains_gold"] is True
    assert cells[0]["reuse_minus_identity_em"] == -1.0
    assert cells[1]["report_contains_gold"] is False
    assert cells[1]["reuse_minus_identity_em"] == 1.0


def test_question_deduplication_keeps_first_casefolded_row():
    rows = [
        {"question": "Which answer?", "value": 1},
        {"question": "  WHICH   ANSWER? ", "value": 2},
        {"question": "A different question", "value": 3},
    ]
    assert [row["value"] for row in _deduplicate_questions(rows)] == [1, 3]
