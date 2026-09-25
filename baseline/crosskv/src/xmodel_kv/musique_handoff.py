from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from typing import Any


def normalize_surface(text: str) -> str:
    """Normalize a MuSiQue entity or relation for structural checks."""

    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def relation_key(row: dict[str, Any]) -> str:
    return normalize_surface(row["question_decomposition"][1]["question"])


def usable_two_hop(row: dict[str, Any]) -> bool:
    """Return whether a row supports an explicit, auditable A->B split."""

    decomposition = row.get("question_decomposition", [])
    paragraphs = row.get("paragraphs", [])
    if len(decomposition) != 2 or "#1" not in decomposition[1].get("question", ""):
        return False
    first_index = decomposition[0].get("paragraph_support_idx")
    second_index = decomposition[1].get("paragraph_support_idx")
    if not isinstance(first_index, int) or not isinstance(second_index, int):
        return False
    if not (0 <= first_index < len(paragraphs) and 0 <= second_index < len(paragraphs)):
        return False

    bridge = normalize_surface(decomposition[0].get("answer", ""))
    final = normalize_surface(decomposition[1].get("answer", ""))
    first_text = normalize_surface(paragraphs[first_index].get("paragraph_text", ""))
    second_text = normalize_surface(paragraphs[second_index].get("paragraph_text", ""))
    return bool(
        bridge
        and final
        and bridge != final
        and bridge in first_text
        and bridge in second_text
        and final in second_text
    )


def compact_training_record(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed for an eight-way B candidate and A distractors."""

    decomposition = row["question_decomposition"]
    second_index = decomposition[1]["paragraph_support_idx"]
    distractors = [
        _document(paragraph, source_id=row["id"])
        for paragraph in row["paragraphs"]
        if not paragraph.get("is_supporting", False)
    ]
    return {
        "id": row["id"],
        "relation_key": relation_key(row),
        "bridge_answer": decomposition[0]["answer"],
        "final_answer": decomposition[1]["answer"],
        "b_document": _document(row["paragraphs"][second_index], source_id=row["id"]),
        "distractors": distractors,
    }


def build_training_pools(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if usable_two_hop(row):
            record = compact_training_record(row)
            pools[record["relation_key"]].append(record)
    return dict(pools)


def eligible_dev_rows(
    rows: Iterable[dict[str, Any]],
    training_pools: dict[str, list[dict[str, Any]]],
    *,
    candidate_count: int = 8,
) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        if not usable_two_hop(row):
            continue
        bridge = normalize_surface(row["question_decomposition"][0]["answer"])
        final = normalize_surface(row["answer"])
        decoys = _select_unique_decoys(
            training_pools.get(relation_key(row), []),
            bridge=bridge,
            final=final,
            source_id=row["id"],
            count=candidate_count - 1,
        )
        if len(decoys) >= candidate_count - 1:
            eligible.append(row)
    return eligible


def build_cases(
    dev_rows: Sequence[dict[str, Any]],
    training_pools: dict[str, list[dict[str, Any]]],
    *,
    samples: int = 197,
    seed: int = 2027,
    candidate_count: int = 8,
    min_a_document_chars: int = 24_000,
    case_split: str = "dev",
    decoy_source_split: str = "train",
    excluded_case_ids: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Build a frozen, train-free MuSiQue A->B handoff evaluation set.

    Training-split rows only provide same-relation decoy documents. No model
    parameters, cache parameters, or selection thresholds are fit here.
    """

    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    eligible = eligible_dev_rows(
        dev_rows, training_pools, candidate_count=candidate_count
    )
    excluded = frozenset(excluded_case_ids)
    eligible = [row for row in eligible if row["id"] not in excluded]
    if samples > len(eligible):
        raise ValueError(f"requested {samples} cases from {len(eligible)} eligible dev rows")
    selection_rng = random.Random(seed)
    selected = selection_rng.sample(eligible, samples)
    return [
        _build_case(
            row,
            training_pools[relation_key(row)],
            seed=seed,
            candidate_count=candidate_count,
            min_a_document_chars=min_a_document_chars,
            case_split=case_split,
            decoy_source_split=decoy_source_split,
        )
        for row in selected
    ]


def _build_case(
    row: dict[str, Any],
    pool: Sequence[dict[str, Any]],
    *,
    seed: int,
    candidate_count: int,
    min_a_document_chars: int,
    case_split: str,
    decoy_source_split: str,
) -> dict[str, Any]:
    decomposition = row["question_decomposition"]
    bridge = decomposition[0]["answer"]
    final = row["answer"]
    bridge_normalized = normalize_surface(bridge)
    final_normalized = normalize_surface(final)
    rng = random.Random(f"{seed}:{row['id']}")

    decoys = _select_unique_decoys(
        pool,
        bridge=bridge_normalized,
        final=final_normalized,
        source_id=row["id"],
        count=candidate_count - 1,
        rng=rng,
    )
    if len(decoys) != candidate_count - 1:
        raise ValueError(f"not enough unique decoys for {row['id']}")

    first_index = decomposition[0]["paragraph_support_idx"]
    second_index = decomposition[1]["paragraph_support_idx"]
    a_documents = [_document(row["paragraphs"][first_index], source_id=row["id"])]
    a_distractors = [
        _document(paragraph, source_id=row["id"])
        for paragraph in row["paragraphs"]
        if paragraph["idx"] not in {first_index, second_index}
    ]
    for record in decoys:
        a_distractors.extend(record["distractors"])
    # Some training rows and their seven selected candidate records do not
    # contain enough local distractor text to reach the long-context floor.
    # Fill from other same-relation training records, using only paragraphs
    # already marked non-supporting in those records.
    selected_decoy_ids = {record["id"] for record in decoys}
    remaining_records = [
        record
        for record in pool
        if record["id"] != row["id"] and record["id"] not in selected_decoy_ids
    ]
    rng.shuffle(remaining_records)
    for record in remaining_records:
        a_distractors.extend(record["distractors"])
    rng.shuffle(a_distractors)
    seen_documents = {_document_key(a_documents[0])}
    current_chars = _document_chars(a_documents)
    for document in a_distractors:
        key = _document_key(document)
        if key in seen_documents:
            continue
        a_documents.append(document)
        seen_documents.add(key)
        current_chars += _document_chars([document])
        if current_chars >= min_a_document_chars:
            break
    if current_chars < min_a_document_chars:
        raise ValueError(
            f"not enough distractor text to build {row['id']} to "
            f"{min_a_document_chars} characters"
        )
    rng.shuffle(a_documents)

    true_candidate = {
        "source_id": row["id"],
        "bridge_answer": bridge,
        "final_answer": final,
        "document": _document(row["paragraphs"][second_index], source_id=row["id"]),
        "is_gold": True,
    }
    candidates = [true_candidate] + [
        {
            "source_id": record["id"],
            "bridge_answer": record["bridge_answer"],
            "final_answer": record["final_answer"],
            "document": record["b_document"],
            "is_gold": False,
        }
        for record in decoys
    ]
    rng.shuffle(candidates)
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"CAND-{index}"

    gold_candidate = next(candidate for candidate in candidates if candidate["is_gold"])
    return {
        "benchmark": "musique_split_handoff_v1",
        "id": row["id"],
        "global_question": row["question"],
        "relation_key": relation_key(row),
        "agent_a": {
            "question": decomposition[0]["question"],
            "gold_answer": bridge,
            "documents": a_documents,
            "document_chars": _document_chars(a_documents),
        },
        "agent_b": {
            "question_template": decomposition[1]["question"],
            "gold_answer": final,
            "gold_candidate_id": gold_candidate["candidate_id"],
            "candidates": candidates,
            "candidate_count": candidate_count,
            "chance_accuracy": 1.0 / candidate_count,
        },
        "construction": {
            "source_dataset": "MuSiQue-Answerable v1.0",
            "evaluation_split": case_split,
            "decoy_source_split": decoy_source_split,
            "decoys_are_training_not_parameters": decoy_source_split == "train",
            "seed": seed,
            "min_a_document_chars": min_a_document_chars,
        },
    }


def _document(paragraph: dict[str, Any], *, source_id: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "title": paragraph["title"],
        "text": paragraph["paragraph_text"],
    }


def _select_unique_decoys(
    pool: Sequence[dict[str, Any]],
    *,
    bridge: str,
    final: str,
    source_id: str,
    count: int,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Find a bridge-to-final matching so both candidate fields stay unique."""

    excluded_bridge = normalize_surface(bridge)
    excluded_final = normalize_surface(final)
    by_bridge: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for record in pool:
        candidate_bridge = normalize_surface(record["bridge_answer"])
        candidate_final = normalize_surface(record["final_answer"])
        if (
            record["id"] == source_id
            or candidate_bridge == excluded_bridge
            or candidate_final == excluded_final
        ):
            continue
        by_bridge[candidate_bridge].append((candidate_final, record))

    bridges = sorted(by_bridge)
    if rng is not None:
        rng.shuffle(bridges)
        for records in by_bridge.values():
            rng.shuffle(records)
    matched_finals: dict[str, tuple[str, dict[str, Any]]] = {}

    def augment(candidate_bridge: str, seen_finals: set[str]) -> bool:
        for candidate_final, record in by_bridge[candidate_bridge]:
            if candidate_final in seen_finals:
                continue
            seen_finals.add(candidate_final)
            previous = matched_finals.get(candidate_final)
            if previous is None or augment(previous[0], seen_finals):
                matched_finals[candidate_final] = (candidate_bridge, record)
                return True
        return False

    for candidate_bridge in bridges:
        augment(candidate_bridge, set())
    matches = [entry[1] for entry in matched_finals.values()]
    if rng is not None:
        rng.shuffle(matches)
    return matches[:count]


def _document_key(document: dict[str, Any]) -> tuple[str, str]:
    return (normalize_surface(document["title"]), normalize_surface(document["text"]))


def _document_chars(documents: Sequence[dict[str, Any]]) -> int:
    return sum(len(document["title"]) + len(document["text"]) + 2 for document in documents)
