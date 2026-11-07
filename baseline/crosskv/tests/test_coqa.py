import pytest

from xmodel_kv.coqa import (
    answer_from_generation,
    context_at_turn,
    coqa_f1,
    gold_answers_at_turn,
    select_conversations,
    token_f1,
)
from xmodel_kv.cli.summarize_coqa import summarize


def _document(domain="mctest", number=0):
    return {
        "id": str(number),
        "source": domain,
        "story": "A short story.",
        "questions": {"input_text": ["First?", "Second?"]},
        "answers": {"input_text": ["the red cat", "yes"]},
        "additional_answers": {
            "0": {"input_text": ["red cat", "Yes."]},
            "1": {"input_text": ["a red cat", "yes"]},
        },
    }


def test_context_and_gold_answers_match_lm_eval_layout():
    document = _document()
    assert context_at_turn(document, 2) == (
        "A short story.\n\nQ: First?\n\nA: the red cat\n\nQ: Second?\n\nA:"
    )
    assert gold_answers_at_turn(document, 1) == ["the red cat", "red cat", "a red cat"]
    assert answer_from_generation("  yes\nQ: next") == "yes"


def test_f1_normalization_and_leave_one_out():
    assert token_f1("Red cat!", "the red cat") == 1.0
    assert coqa_f1("red cat", ["the red cat", "red animal"]) == pytest.approx(0.75)


def test_balanced_selection_retains_indices():
    domains = ("mctest", "race", "cnn", "wikipedia", "gutenberg")
    rows = []
    for domain in domains:
        rows.extend(_document(domain, number) for number in range(3))
    selected = select_conversations(enumerate(rows), per_domain=1)
    assert [index for index, _ in selected] == [0, 3, 6, 9, 12]


def test_coqa_summary_checks_composite_coverage():
    rows = []
    for turn in (1, 3):
        for index in range(2):
            rows.append(
                {
                    "index": index,
                    "turn": turn,
                    "domain": "mctest",
                    "standalone_f1": 0.8,
                    "transfer_f1": 0.7,
                }
            )
    result = summarize(rows, expected_conversations=2, expected_turns=(1, 3))
    assert result["overall_drift_percentage_points"] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + rows[:1], expected_conversations=2, expected_turns=(1, 3))
