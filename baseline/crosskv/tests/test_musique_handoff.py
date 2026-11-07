from xmodel_kv.musique_handoff import (
    build_cases,
    build_training_pools,
    normalize_surface,
    usable_two_hop,
)


def _row(row_id, bridge, final, relation="#1 >> spouse"):
    return {
        "id": row_id,
        "question": f"What is related to {bridge}?",
        "answer": final,
        "question_decomposition": [
            {
                "question": "Find the bridge",
                "answer": bridge,
                "paragraph_support_idx": 0,
            },
            {
                "question": relation,
                "answer": final,
                "paragraph_support_idx": 1,
            },
        ],
        "paragraphs": [
            {
                "idx": 0,
                "title": f"First {bridge}",
                "paragraph_text": f"The intermediate answer is {bridge}.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": f"Second {bridge}",
                "paragraph_text": f"{bridge} has the requested relation to {final}.",
                "is_supporting": True,
            },
            {
                "idx": 2,
                "title": f"Distractor {row_id}",
                "paragraph_text": "Unrelated dossier text. " * 20,
                "is_supporting": False,
            },
        ],
    }


def test_usable_two_hop_requires_explicit_evidence():
    assert usable_two_hop(_row("dev", "Bridge", "Final"))
    broken = _row("broken", "Bridge", "Final")
    broken["paragraphs"][1]["paragraph_text"] = "No named entities here."
    assert not usable_two_hop(broken)


def test_build_case_has_unique_eight_way_candidates():
    train = [_row(f"train-{i}", f"Bridge-{i}", f"Final-{i}") for i in range(10)]
    dev = [_row("dev", "Dev-Bridge", "Dev-Final")]
    pools = build_training_pools(train)
    cases = build_cases(
        dev,
        pools,
        samples=1,
        candidate_count=8,
        min_a_document_chars=100,
    )
    case = cases[0]
    candidates = case["agent_b"]["candidates"]
    assert len(candidates) == 8
    assert sum(candidate["is_gold"] for candidate in candidates) == 1
    assert len({normalize_surface(candidate["bridge_answer"]) for candidate in candidates}) == 8
    assert len({normalize_surface(candidate["final_answer"]) for candidate in candidates}) == 8
    assert case["agent_b"]["chance_accuracy"] == 0.125


def test_build_case_records_non_dev_training_provenance():
    rows = [_row(f"train-{i}", f"Bridge-{i}", f"Final-{i}") for i in range(10)]
    pools = build_training_pools(rows)
    case = build_cases(
        rows,
        pools,
        samples=1,
        candidate_count=8,
        min_a_document_chars=100,
        case_split="train",
        decoy_source_split="train",
    )[0]
    assert case["construction"]["evaluation_split"] == "train"
    assert case["construction"]["decoy_source_split"] == "train"
    assert case["agent_a"]["document_chars"] >= 100


def test_build_cases_excludes_frozen_case_ids_before_sampling():
    rows = [_row(f"train-{i}", f"Bridge-{i}", f"Final-{i}") for i in range(12)]
    pools = build_training_pools(rows)

    cases = build_cases(
        rows,
        pools,
        samples=5,
        candidate_count=8,
        min_a_document_chars=100,
        case_split="train",
        excluded_case_ids={"train-0", "train-1", "train-2"},
    )

    assert len(cases) == 5
    assert not {case["id"] for case in cases} & {"train-0", "train-1", "train-2"}
