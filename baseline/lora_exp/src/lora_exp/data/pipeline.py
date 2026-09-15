from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict

from lora_exp.data.formatters import (
    format_flashcard_paper,
    format_medmcqa_paper,
    format_normalized_qa,
    format_pubmedqa_paper,
)
from lora_exp.data.registry import LoadedShard
from lora_exp.data.schema import (
    clean_text,
    stable_record_id,
    validate_canonical_record,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _percentile(
    values: list[int],
    p: float,
) -> float:
    if not values:
        return float("nan")

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    pos = (len(ordered) - 1) * p
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return float(ordered[lower])

    fraction = pos - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def _token_summary(
    values: list[int],
) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
        }

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _medmcqa_records(
    shards: list[LoadedShard],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(shards) != 1:
        raise RuntimeError(
            "MedMCQA should have exactly one source shard"
        )

    shard = shards[0]

    records: list[dict[str, Any]] = []

    letters = ["A", "B", "C", "D"]

    for source_index, row in enumerate(
        shard.dataset
    ):
        question = clean_text(row.get("question"))

        choices = [
            clean_text(row.get("opa")),
            clean_text(row.get("opb")),
            clean_text(row.get("opc")),
            clean_text(row.get("opd")),
        ]

        try:
            answer_index = int(row.get("cop"))
        except Exception:
            continue

        if answer_index not in range(4):
            continue

        answer_letter = letters[answer_index]
        answer_text = choices[answer_index]

        source_row_id = clean_text(
            row.get("id")
        )

        record_id = (
            f"medmcqa:{source_row_id}"
            if source_row_id
            else stable_record_id(
                dataset="medmcqa",
                source_config=shard.config_name,
                source_split=shard.split_name,
                source_index=source_index,
                question=question,
            )
        )

        paper_text = format_medmcqa_paper(
            question=question,
            choices=choices,
            answer_letter=answer_letter,
        )

        normalized_text = format_normalized_qa(
            question=question,
            choices=choices,
            answer=answer_letter,
        )

        record = {
            "record_id": record_id,
            "dataset": "medmcqa",
            "repo_id": shard.repo_id,
            "dataset_revision": shard.revision,
            "source_config": shard.config_name,
            "source_split": shard.split_name,
            "source_index": source_index,
            "source_row_id": source_row_id,
            "domain": cfg["domain"],
            "task_type": cfg["task_type"],
            "source_format": "mcq_four_choice",
            "question": question,
            "context": "",
            "choices": choices,
            "answer": answer_letter,
            "answer_text": answer_text,
            "rationale": clean_text(
                row.get("exp")
            ),
            "subject": clean_text(
                row.get("subject_name")
            ),
            "topic": clean_text(
                row.get("topic_name")
            ),
            "paper_text": paper_text,
            "normalized_text": normalized_text,
            "paper_token_count": -1,
            "normalized_token_count": -1,
        }

        validate_canonical_record(record)
        records.append(record)

    return records


def _pubmedqa_records(
    shards: list[LoadedShard],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    global_index = 0

    for shard in shards:
        for local_index, row in enumerate(
            shard.dataset
        ):
            question = clean_text(
                row.get("QUESTION")
            )

            contexts = row.get(
                "CONTEXTS",
                [],
            )

            if contexts is None:
                contexts = []

            context = "\n".join(
                clean_text(x)
                for x in contexts
                if clean_text(x)
            )

            answer = clean_text(
                row.get("final_decision")
            ).lower()

            # The paper explicitly retains only
            # yes / no / maybe examples.
            if answer not in {
                "yes",
                "no",
                "maybe",
            }:
                global_index += 1
                continue

            record_id = stable_record_id(
                dataset="pubmedqa",
                source_config=shard.config_name,
                source_split=shard.split_name,
                source_index=local_index,
                question=question,
            )

            paper_text = (
                format_pubmedqa_paper(
                    question=question,
                    context=context,
                    answer=answer,
                )
            )

            normalized_text = (
                format_normalized_qa(
                    question=question,
                    context=context,
                    answer=answer,
                )
            )

            record = {
                "record_id": record_id,
                "dataset": "pubmedqa",
                "repo_id": shard.repo_id,
                "dataset_revision": shard.revision,
                "source_config": shard.config_name,
                "source_split": shard.split_name,
                "source_index": global_index,
                "source_row_id": "",
                "domain": cfg["domain"],
                "task_type": cfg["task_type"],
                "source_format": (
                    "contextual_closed_qa"
                ),
                "question": question,
                "context": context,
                "choices": [],
                "answer": answer,
                "answer_text": answer,
                "rationale": clean_text(
                    row.get("LONG_ANSWER")
                ),
                "subject": "",
                "topic": "",
                "paper_text": paper_text,
                "normalized_text": normalized_text,
                "paper_token_count": -1,
                "normalized_token_count": -1,
            }

            validate_canonical_record(record)
            records.append(record)

            global_index += 1

    return records


def _flashcard_records(
    shards: list[LoadedShard],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(shards) != 1:
        raise RuntimeError(
            "Flashcards should have exactly one source shard"
        )

    shard = shards[0]

    records: list[dict[str, Any]] = []

    for source_index, row in enumerate(
        shard.dataset
    ):
        question = clean_text(
            row.get("input")
        )

        answer = clean_text(
            row.get("output")
        )

        record_id = stable_record_id(
            dataset="flashcards",
            source_config=shard.config_name,
            source_split=shard.split_name,
            source_index=source_index,
            question=question,
        )

        paper_text = format_flashcard_paper(
            question=question,
            answer=answer,
        )

        normalized_text = (
            format_normalized_qa(
                question=question,
                answer=answer,
            )
        )

        record = {
            "record_id": record_id,
            "dataset": "flashcards",
            "repo_id": shard.repo_id,
            "dataset_revision": shard.revision,
            "source_config": shard.config_name,
            "source_split": shard.split_name,
            "source_index": source_index,
            "source_row_id": "",
            "domain": cfg["domain"],
            "task_type": cfg["task_type"],
            "source_format": "open_qa",
            "question": question,
            "context": "",
            "choices": [],
            "answer": answer,
            "answer_text": answer,
            "rationale": "",
            "subject": "",
            "topic": "",
            "paper_text": paper_text,
            "normalized_text": normalized_text,
            "paper_token_count": -1,
            "normalized_token_count": -1,
        }

        validate_canonical_record(record)
        records.append(record)

    return records


def convert_shards_to_records(
    *,
    logical_dataset: str,
    shards: list[LoadedShard],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if logical_dataset == "medmcqa":
        return _medmcqa_records(
            shards,
            cfg,
        )

    if logical_dataset == "pubmedqa":
        return _pubmedqa_records(
            shards,
            cfg,
        )

    if logical_dataset == "flashcards":
        return _flashcard_records(
            shards,
            cfg,
        )

    raise ValueError(
        f"Unsupported logical_dataset={logical_dataset}"
    )


def add_token_counts(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    batch_size: int,
) -> None:
    for start in range(
        0,
        len(records),
        batch_size,
    ):
        batch_records = records[
            start:start + batch_size
        ]

        paper_texts = [
            row["paper_text"]
            for row in batch_records
        ]

        normalized_texts = [
            row["normalized_text"]
            for row in batch_records
        ]

        paper_encoded = tokenizer(
            paper_texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )

        normalized_encoded = tokenizer(
            normalized_texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )

        for row, paper_ids, normalized_ids in zip(
            batch_records,
            paper_encoded["input_ids"],
            normalized_encoded["input_ids"],
        ):
            row["paper_token_count"] = len(
                paper_ids
            )

            row["normalized_token_count"] = len(
                normalized_ids
            )


def filter_records(
    records: list[dict[str, Any]],
    *,
    max_input_length: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
]:
    filtered: list[dict[str, Any]] = []

    removed_empty = 0
    removed_overlength = 0

    for row in records:
        if (
            not clean_text(row["question"])
            or not clean_text(row["answer"])
        ):
            removed_empty += 1
            continue

        # Primary Phase-2 training representation:
        # paper_text.
        if (
            int(row["paper_token_count"])
            > max_input_length
        ):
            removed_overlength += 1
            continue

        filtered.append(row)

    return (
        filtered,
        {
            "removed_empty": removed_empty,
            "removed_overlength_paper_text": (
                removed_overlength
            ),
        },
    )


def build_dataset_stats(
    *,
    logical_dataset: str,
    records_before_filter: list[
        dict[str, Any]
    ],
    records_after_filter: list[
        dict[str, Any]
    ],
    split_sizes: dict[str, int],
    filter_stats: dict[str, int],
) -> dict[str, Any]:
    paper_counts = [
        int(row["paper_token_count"])
        for row in records_after_filter
    ]

    normalized_counts = [
        int(row["normalized_token_count"])
        for row in records_after_filter
    ]

    duplicate_texts = (
        len(records_after_filter)
        - len(
            {
                row["paper_text"]
                for row in records_after_filter
            }
        )
    )

    duplicate_ids = (
        len(records_after_filter)
        - len(
            {
                row["record_id"]
                for row in records_after_filter
            }
        )
    )

    answer_distribution = Counter(
        row["answer"]
        for row in records_after_filter
    )

    source_format_distribution = Counter(
        row["source_format"]
        for row in records_after_filter
    )

    stats: dict[str, Any] = {
        "dataset": logical_dataset,
        "records_before_filter": len(
            records_before_filter
        ),
        "records_after_filter": len(
            records_after_filter
        ),
        "filter": filter_stats,
        "split_sizes": split_sizes,
        "duplicate_record_ids_retained": (
            duplicate_ids
        ),
        "duplicate_paper_texts_retained": (
            duplicate_texts
        ),
        "paper_token_count": _token_summary(
            paper_counts
        ),
        "normalized_token_count": (
            _token_summary(
                normalized_counts
            )
        ),
        "source_format_distribution": dict(
            source_format_distribution
        ),
    }

    # Free-text answer distribution for flashcards
    # is not useful and may make the manifest huge.
    if logical_dataset != "flashcards":
        stats["answer_distribution"] = dict(
            answer_distribution
        )

    if logical_dataset == "medmcqa":
        stats["subject_distribution"] = dict(
            Counter(
                row["subject"]
                for row in records_after_filter
                if row["subject"]
            )
        )

    return stats


def save_prepared_dataset(
    *,
    logical_dataset: str,
    records: list[dict[str, Any]],
    output_root: Path,
    validation_fraction: float,
    seed: int,
    preview_rows: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    dataset = Dataset.from_list(records)

    split = dataset.train_test_split(
        test_size=validation_fraction,
        seed=seed,
        shuffle=True,
    )

    dataset_dict = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
        }
    )

    target_dir = (
        output_root
        / logical_dataset
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        target_dir
        / "train.parquet"
    )

    validation_path = (
        target_dir
        / "validation.parquet"
    )

    dataset_dict["train"].to_parquet(
        str(train_path),
        compression="zstd",
    )

    dataset_dict["validation"].to_parquet(
        str(validation_path),
        compression="zstd",
    )

    preview_path = (
        target_dir
        / "preview.jsonl"
    )

    with preview_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        n = min(
            preview_rows,
            len(dataset_dict["train"]),
        )

        for idx in range(n):
            f.write(
                json.dumps(
                    dataset_dict["train"][idx],
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats["split_sizes"] = {
        "train": len(
            dataset_dict["train"]
        ),
        "validation": len(
            dataset_dict["validation"]
        ),
    }

    stats_path = (
        target_dir
        / "stats.json"
    )

    with stats_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "train_path": str(train_path),
        "validation_path": str(
            validation_path
        ),
        "stats_path": str(stats_path),
        "preview_path": str(preview_path),
        "train_sha256": _sha256_file(
            train_path
        ),
        "validation_sha256": (
            _sha256_file(
                validation_path
            )
        ),
        "train_rows": len(
            dataset_dict["train"]
        ),
        "validation_rows": len(
            dataset_dict["validation"]
        ),
    }


def source_manifest(
    shards: Iterable[LoadedShard],
) -> list[dict[str, Any]]:
    return [
        {
            "logical_dataset": (
                shard.logical_dataset
            ),
            "repo_id": shard.repo_id,
            "revision": shard.revision,
            "config_name": (
                shard.config_name
            ),
            "split_name": (
                shard.split_name
            ),
            "rows": len(
                shard.dataset
            ),
        }
        for shard in shards
    ]