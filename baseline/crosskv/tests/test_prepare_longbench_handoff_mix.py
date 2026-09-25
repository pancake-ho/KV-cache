from xmodel_kv.cli.prepare_longbench_handoff_mix import (
    build_handoff_cases,
    parse_passages,
    unique_answer_groups,
)


def _row(index, answer):
    return {
        "dataset": "test",
        "input": f"question {index}",
        "context": f"Passage 1:\nTitle {index}\nText {index}",
        "answer": answer,
        "raw_id": str(index),
    }


def test_unique_answer_groups_do_not_create_ambiguous_lookup_keys():
    rows = [_row(index, "yes" if index < 3 else f"answer {index}") for index in range(10)]

    groups = unique_answer_groups(rows, size=4, seed=7)

    assert groups
    for group in groups:
        assert len({row["answer"].casefold() for row in group}) == 4


def test_handoff_rotates_final_answer_and_marks_one_gold_candidate():
    rows = [_row(index, f"answer {index}") for index in range(4)]

    cases = build_handoff_cases(rows, candidate_count=4, seed=3)

    assert len(cases) == 4
    for case in cases:
        candidates = case["agent_b"]["candidates"]
        assert sum(candidate["is_gold"] for candidate in candidates) == 1
        gold = next(candidate for candidate in candidates if candidate["is_gold"])
        assert gold["bridge_answer"] == case["agent_a"]["gold_answer"]
        assert gold["final_answer"] == case["agent_b"]["gold_answer"]
        assert gold["final_answer"] != gold["bridge_answer"]


def test_passage_parser_preserves_titles_and_text():
    documents = parse_passages(
        "Passage 1:\nAlpha\nFirst text\n\nPassage 2:\nBeta\nSecond text",
        source_id="row",
    )

    assert [document["title"] for document in documents] == ["Alpha", "Beta"]
    assert documents[1]["text"] == "Second text"
