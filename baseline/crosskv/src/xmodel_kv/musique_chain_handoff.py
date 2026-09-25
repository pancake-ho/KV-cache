from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from typing import Any

from .musique_handoff import normalize_surface


_REFERENCE_RE = re.compile(r"#(\d+)")


def is_linear_chain(row: dict[str, Any], *, hops: int) -> bool:
    """Return whether a MuSiQue row is an auditable A->B->... chain.

    Every downstream decomposition step must depend on exactly the immediately
    preceding step.  Its supporting paragraph must explicitly contain both the
    lookup key (previous answer) and the new result.
    """

    decomposition = row.get("question_decomposition", [])
    paragraphs = row.get("paragraphs", [])
    if len(decomposition) != hops:
        return False
    for stage_index, step in enumerate(decomposition):
        references = [int(value) for value in _REFERENCE_RE.findall(step.get("question", ""))]
        if stage_index == 0:
            if references:
                return False
        elif references != [stage_index]:
            return False
        support_index = step.get("paragraph_support_idx")
        if not isinstance(support_index, int) or not 0 <= support_index < len(paragraphs):
            return False
        evidence = normalize_surface(paragraphs[support_index].get("paragraph_text", ""))
        result = normalize_surface(step.get("answer", ""))
        if not result or result not in evidence:
            return False
        if stage_index:
            lookup_key = normalize_surface(decomposition[stage_index - 1].get("answer", ""))
            if not lookup_key or lookup_key == result or lookup_key not in evidence:
                return False
    return True


def collect_candidate_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect real MuSiQue relation records used only as structured decoys."""

    records: list[dict[str, Any]] = []
    for row in rows:
        decomposition = row.get("question_decomposition", [])
        paragraphs = row.get("paragraphs", [])
        for stage_index, step in enumerate(decomposition[1:], start=1):
            references = [int(value) for value in _REFERENCE_RE.findall(step.get("question", ""))]
            if references != [stage_index]:
                continue
            support_index = step.get("paragraph_support_idx")
            if not isinstance(support_index, int) or not 0 <= support_index < len(paragraphs):
                continue
            lookup_key = decomposition[stage_index - 1].get("answer", "")
            result = step.get("answer", "")
            evidence = paragraphs[support_index].get("paragraph_text", "")
            normalized_evidence = normalize_surface(evidence)
            if (
                not normalize_surface(lookup_key)
                or not normalize_surface(result)
                or normalize_surface(lookup_key) == normalize_surface(result)
                or normalize_surface(lookup_key) not in normalized_evidence
                or normalize_surface(result) not in normalized_evidence
            ):
                continue
            records.append(
                {
                    "source_id": row["id"],
                    "lookup_key": lookup_key,
                    "result": result,
                    "question_template": step["question"],
                    "document": _document(paragraphs[support_index], row["id"]),
                }
            )
    return records


def collect_distractor_documents(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for row in rows:
        for paragraph in row.get("paragraphs", []):
            if not paragraph.get("is_supporting", False):
                documents.append(_document(paragraph, row["id"]))
    return documents


def build_chain_cases(
    dev_rows: Sequence[dict[str, Any]],
    candidate_records: Sequence[dict[str, Any]],
    distractor_documents: Sequence[dict[str, Any]],
    *,
    hops: int,
    samples: int,
    seed: int = 2027,
    candidate_count: int = 8,
    min_agent_a_document_chars: int = 24_000,
) -> list[dict[str, Any]]:
    if hops not in {3, 4}:
        raise ValueError("hops must be 3 or 4")
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    eligible = [row for row in dev_rows if is_linear_chain(row, hops=hops)]
    if samples > len(eligible):
        raise ValueError(f"requested {samples} cases from {len(eligible)} eligible rows")
    rng = random.Random(f"select:{seed}:{hops}")
    selected = rng.sample(eligible, samples)
    return [
        _build_case(
            row,
            candidate_records,
            distractor_documents,
            seed=seed,
            candidate_count=candidate_count,
            min_agent_a_document_chars=min_agent_a_document_chars,
        )
        for row in selected
    ]


def _build_case(
    row: dict[str, Any],
    candidate_records: Sequence[dict[str, Any]],
    distractor_documents: Sequence[dict[str, Any]],
    *,
    seed: int,
    candidate_count: int,
    min_agent_a_document_chars: int,
) -> dict[str, Any]:
    decomposition = row["question_decomposition"]
    paragraphs = row["paragraphs"]
    rng = random.Random(f"case:{seed}:{row['id']}")
    support_indices = {step["paragraph_support_idx"] for step in decomposition}

    first_support = _document(paragraphs[decomposition[0]["paragraph_support_idx"]], row["id"])
    a_documents = [first_support]
    local_distractors = [
        _document(paragraph, row["id"])
        for paragraph in paragraphs
        if paragraph.get("idx") not in support_indices
    ]
    global_distractors = list(distractor_documents)
    rng.shuffle(local_distractors)
    rng.shuffle(global_distractors)
    seen = {_document_key(first_support)}
    for document in local_distractors + global_distractors:
        key = _document_key(document)
        if key in seen:
            continue
        a_documents.append(document)
        seen.add(key)
        if _document_chars(a_documents) >= min_agent_a_document_chars:
            break
    if _document_chars(a_documents) < min_agent_a_document_chars:
        raise ValueError("not enough distractor text to build the Agent-A dossier")
    rng.shuffle(a_documents)

    stages: list[dict[str, Any]] = []
    for stage_index, step in enumerate(decomposition[1:], start=1):
        lookup_key = decomposition[stage_index - 1]["answer"]
        result = step["answer"]
        true_candidate = {
            "source_id": row["id"],
            "lookup_key": lookup_key,
            "result": result,
            "document": _document(paragraphs[step["paragraph_support_idx"]], row["id"]),
            "is_gold": True,
        }
        decoys = _select_decoys(
            candidate_records,
            excluded_source_id=row["id"],
            excluded_lookup=lookup_key,
            excluded_result=result,
            count=candidate_count - 1,
            rng=rng,
        )
        if len(decoys) != candidate_count - 1:
            raise ValueError(f"not enough unique candidates for {row['id']} stage {stage_index + 1}")
        candidates = [true_candidate] + [
            {
                "source_id": record["source_id"],
                "lookup_key": record["lookup_key"],
                "result": record["result"],
                "document": record["document"],
                "is_gold": False,
            }
            for record in decoys
        ]
        rng.shuffle(candidates)
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate["candidate_id"] = f"CAND-{candidate_index}"
        gold = next(candidate for candidate in candidates if candidate["is_gold"])
        stages.append(
            {
                "agent_index": stage_index + 1,
                "question_template": step["question"],
                "lookup_gold": lookup_key,
                "gold_answer": result,
                "gold_candidate_id": gold["candidate_id"],
                "candidates": candidates,
                "candidate_count": candidate_count,
                "chance_accuracy": 1.0 / candidate_count,
            }
        )

    return {
        "benchmark": "musique_linear_chain_handoff_v1",
        "id": row["id"],
        "agent_count": len(decomposition),
        "global_question": row["question"],
        "agent_a": {
            "question": decomposition[0]["question"],
            "gold_answer": decomposition[0]["answer"],
            "documents": a_documents,
            "document_chars": _document_chars(a_documents),
        },
        "stages": stages,
        "construction": {
            "source_dataset": "MuSiQue-Answerable v1.0",
            "evaluation_split": "dev",
            "decoy_source_split": "train",
            "controlled_structured_candidates": True,
            "decoys_are_training_records_not_fitted_parameters": True,
            "seed": seed,
            "min_agent_a_document_chars": min_agent_a_document_chars,
        },
    }


def _select_decoys(
    records: Sequence[dict[str, Any]],
    *,
    excluded_source_id: str,
    excluded_lookup: str,
    excluded_result: str,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    lookup_values = {normalize_surface(excluded_lookup)}
    result_values = {normalize_surface(excluded_result)}
    selected: list[dict[str, Any]] = []
    for record in shuffled:
        lookup = normalize_surface(record["lookup_key"])
        result = normalize_surface(record["result"])
        if (
            record["source_id"] == excluded_source_id
            or not lookup
            or not result
            or lookup in lookup_values
            or result in result_values
        ):
            continue
        selected.append(record)
        lookup_values.add(lookup)
        result_values.add(result)
        if len(selected) == count:
            break
    return selected


def _document(paragraph: dict[str, Any], source_id: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "title": paragraph["title"],
        "text": paragraph["paragraph_text"],
    }


def _document_key(document: dict[str, Any]) -> tuple[str, str]:
    return normalize_surface(document["title"]), normalize_surface(document["text"])


def _document_chars(documents: Sequence[dict[str, Any]]) -> int:
    return sum(len(document["title"]) + len(document["text"]) + 2 for document in documents)
