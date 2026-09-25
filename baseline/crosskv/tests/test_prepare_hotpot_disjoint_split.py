from xmodel_kv.cli.prepare_hotpot_disjoint_split import (
    disjoint_questions,
    normalize_question,
)


def test_question_normalization_and_reference_exclusion():
    assert normalize_question("  Who  Is\nThis? ") == "who is this?"
    candidate = [
        {"_id": "a", "input": "Question A"},
        {"_id": "b", "input": " QUESTION  B "},
        {"_id": "c", "input": "question b"},
    ]
    reference = [{"_id": "r", "input": "question a"}]
    kept, excluded = disjoint_questions(candidate, reference)
    assert [row["_id"] for row in kept] == ["b"]
    assert [row["_id"] for row in excluded] == ["a", "c"]
