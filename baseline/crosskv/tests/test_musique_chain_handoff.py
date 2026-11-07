from xmodel_kv.musique_chain_handoff import (
    build_chain_cases,
    collect_candidate_records,
    collect_distractor_documents,
    is_linear_chain,
)


def _row(row_id: str, answers: list[str], *, distractor_count: int = 2):
    paragraphs = []
    decomposition = []
    for index, answer in enumerate(answers):
        previous = answers[index - 1] if index else ""
        question = f"Find stage {index + 1}" if index == 0 else f"#{index} >> relation"
        evidence = (
            f"The requested result is {answer}."
            if index == 0
            else f"The lookup key {previous} maps to requested result {answer}."
        )
        paragraphs.append(
            {
                "idx": index,
                "title": f"Support {row_id} {index}",
                "paragraph_text": evidence,
                "is_supporting": True,
            }
        )
        decomposition.append(
            {"question": question, "answer": answer, "paragraph_support_idx": index}
        )
    for index in range(distractor_count):
        paragraphs.append(
            {
                "idx": len(answers) + index,
                "title": f"Distractor {row_id} {index}",
                "paragraph_text": "Unrelated evidence text. " * 20,
                "is_supporting": False,
            }
        )
    return {
        "id": row_id,
        "question": "Global chained question",
        "answer": answers[-1],
        "question_decomposition": decomposition,
        "paragraphs": paragraphs,
    }


def test_linear_chain_requires_immediate_predecessor():
    row = _row("dev", ["One", "Two", "Three"])
    assert is_linear_chain(row, hops=3)
    row["question_decomposition"][2]["question"] = "#1 >> wrong dependency"
    assert not is_linear_chain(row, hops=3)


def test_build_three_agent_case_has_unique_candidates():
    train = [_row(f"train-{i}", [f"A{i}", f"B{i}", f"C{i}"]) for i in range(12)]
    dev = [_row("dev", ["Dev-A", "Dev-B", "Dev-C"])]
    cases = build_chain_cases(
        dev,
        collect_candidate_records(train),
        collect_distractor_documents(train),
        hops=3,
        samples=1,
        candidate_count=8,
        min_agent_a_document_chars=200,
    )
    case = cases[0]
    assert case["agent_count"] == 3
    assert len(case["stages"]) == 2
    for stage in case["stages"]:
        assert len(stage["candidates"]) == 8
        assert sum(candidate["is_gold"] for candidate in stage["candidates"]) == 1
        assert len({candidate["lookup_key"].casefold() for candidate in stage["candidates"]}) == 8
        assert len({candidate["result"].casefold() for candidate in stage["candidates"]}) == 8
