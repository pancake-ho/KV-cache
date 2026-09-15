from __future__ import annotations

import hashlib
import json
from typing import Any


CANONICAL_FIELDS = (
    "record_id",
    "dataset",
    "repo_id",
    "dataset_revision",
    "source_config",
    "source_split",
    "source_index",
    "source_row_id",
    "domain",
    "task_type",
    "source_format",
    "question",
    "context",
    "choices",
    "answer",
    "answer_text",
    "rationale",
    "subject",
    "topic",
    "paper_text",
    "normalized_text",
    "paper_token_count",
    "normalized_token_count",
)


def clean_text(value: Any) -> str:
    """
    text만 분리
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def stable_record_id(
    *,
    dataset: str,
    source_config: str,
    source_split: str,
    source_index: int,
    question: str,
) -> str:
    """
    Create a deterministic ID even when the upstream dataset
    does not expose a unique row ID.
    """
    payload = {
        "dataset": dataset,
        "source_config": source_config,
        "source_split": source_split,
        "source_index": int(source_index),
        "question": question,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    digest = hashlib.sha256(serialized).hexdigest()[:24]

    return f"{dataset}:{digest}"


def validate_canonical_record(record: dict[str, Any]) -> None:
    missing = [
        field
        for field in CANONICAL_FIELDS
        if field not in record
    ]

    if missing:
        raise ValueError(
            f"Canonical record is missing fields: {missing}"
        )

    if not clean_text(record["record_id"]):
        raise ValueError("record_id is empty")

    if not clean_text(record["question"]):
        raise ValueError(
            f"question is empty for {record['record_id']}"
        )

    if not clean_text(record["answer"]):
        raise ValueError(
            f"answer is empty for {record['record_id']}"
        )