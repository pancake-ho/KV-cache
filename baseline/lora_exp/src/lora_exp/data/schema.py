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
    Convert a value into a stripped text representation.

    None is represented as an empty string so that data-quality
    filtering can be handled consistently later in the pipeline.
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

    The question may be empty for malformed source rows.
    Such rows are intentionally given a deterministic ID first,
    then removed later by filter_records() so that the removal
    can be counted in Phase-1 statistics.
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

    digest = hashlib.sha256(
        serialized
    ).hexdigest()[:24]

    return f"{dataset}:{digest}"


def validate_canonical_record(
    record: dict[str, Any],
) -> None:
    """
    Validate the STRUCTURE of a canonical record.

    Important separation of responsibilities:

    - This function:
        schema / required-field validation

    - filter_records():
        content-quality validation such as
        empty question/answer and over-length samples

    We intentionally do NOT reject empty question/answer here.
    Some upstream datasets contain malformed empty text fields.
    Those samples must reach filter_records() so their removal
    is explicitly counted in stats.json instead of causing the
    entire Phase-1 preparation job to fail.
    """
    missing = [
        field
        for field in CANONICAL_FIELDS
        if field not in record
    ]

    if missing:
        raise ValueError(
            "Canonical record is missing fields: "
            f"{missing}"
        )

    if not clean_text(
        record["record_id"]
    ):
        raise ValueError(
            "record_id is empty"
        )

    # Structural type checks.
    if not isinstance(
        record["source_index"],
        int,
    ):
        raise TypeError(
            "source_index must be int, "
            f"got {type(record['source_index'])}"
        )

    if not isinstance(
        record["choices"],
        list,
    ):
        raise TypeError(
            "choices must be list, "
            f"got {type(record['choices'])}"
        )

    if not isinstance(
        record["paper_text"],
        str,
    ):
        raise TypeError(
            "paper_text must be str, "
            f"got {type(record['paper_text'])}"
        )

    if not isinstance(
        record["normalized_text"],
        str,
    ):
        raise TypeError(
            "normalized_text must be str, "
            f"got {type(record['normalized_text'])}"
        )