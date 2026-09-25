#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lora_exp.model.factory import load_tokenizer
from lora_exp.train.packing import (
    build_packed_blocks,
    estimate_available_full_blocks,
    load_parquet_dataset,
)
from lora_exp.train.runner import train_one_adapter


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def git_output(args: list[str]) -> str:
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return f"<git failed: {result.stderr.strip()}>"
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            candidate = (destination / member.name).resolve()
            if candidate != root and root not in candidate.parents:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
        tar.extractall(destination)


def clean_subject(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_subject_index(dataset) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for row_idx, subject in enumerate(dataset["subject"]):
        key = clean_subject(subject)
        if key:
            index[key].append(row_idx)
    return dict(index)


def subset(dataset, indices: list[int]):
    return dataset.select(indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/topic_userpair.yaml",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Audit/select topics and write topic_plan.json, then exit before GPU training.",
    )
    args = parser.parse_args()

    cfg = read_yaml(PROJECT_ROOT / args.config)["phase3b"]

    model_cfg = read_yaml(
        PROJECT_ROOT / cfg["model_config"]
    )["model"]
    lora_cfg = read_yaml(
        PROJECT_ROOT / cfg["lora_config"]
    )["lora"]
    base_train_cfg = read_yaml(
        PROJECT_ROOT / cfg["base_train_config"]
    )["phase2"]

    work_dir = Path(os.path.expandvars(args.work_dir)).resolve()
    if work_dir.exists():
        raise RuntimeError(f"work_dir already exists: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=False)

    receipt_path = PROJECT_ROOT / cfg["phase1_receipt"]
    receipt = read_json(receipt_path)
    if receipt.get("status") != "PASS":
        raise RuntimeError("Phase-1 receipt is not PASS")

    archive_path = Path(receipt["archive"])
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    actual_sha = sha256_file(archive_path)
    expected_sha = receipt["archive_sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError(
            "Phase-1 archive SHA mismatch\n"
            f"expected={expected_sha}\nactual={actual_sha}"
        )

    extract_root = work_dir / "phase1"
    safe_extract(archive_path, extract_root)

    dataset_root = extract_root / cfg["phase1_dataset_root"]
    source_name = str(cfg["source_dataset"])

    train_ds = load_parquet_dataset(
        dataset_root / source_name / "train.parquet"
    )
    val_ds = load_parquet_dataset(
        dataset_root / source_name / "validation.parquet"
    )

    if "subject" not in train_ds.column_names:
        raise RuntimeError("MedMCQA subject column is missing")

    tokenizer = load_tokenizer(model_cfg)

    seq_len = int(cfg["sequence_length"])
    token_count_field = str(cfg["token_count_field"])
    text_field = str(cfg["text_field"])
    append_eos = bool(cfg["append_eos_between_samples"])

    train_index = build_subject_index(train_ds)
    val_index = build_subject_index(val_ds)

    candidate_rows: list[dict[str, Any]] = []

    shared_subjects = sorted(
        set(train_index).intersection(val_index)
    )

    print("\n[TOPIC AUDIT]")
    for subject in shared_subjects:
        train_sub = subset(train_ds, train_index[subject])
        val_sub = subset(val_ds, val_index[subject])

        train_blocks = estimate_available_full_blocks(
            train_sub,
            token_count_field=token_count_field,
            sequence_length=seq_len,
            append_eos=append_eos,
        )
        val_blocks = estimate_available_full_blocks(
            val_sub,
            token_count_field=token_count_field,
            sequence_length=seq_len,
            append_eos=append_eos,
        )

        row = {
            "subject": subject,
            "train_rows": len(train_sub),
            "validation_rows": len(val_sub),
            "available_train_blocks": int(train_blocks),
            "available_validation_blocks": int(val_blocks),
        }
        candidate_rows.append(row)

    min_train_blocks = int(cfg["min_train_blocks"])
    min_val_blocks = int(cfg["min_validation_blocks"])

    eligible = [
        row
        for row in candidate_rows
        if row["available_train_blocks"] >= min_train_blocks
        and row["available_validation_blocks"] >= min_val_blocks
    ]

    explicit_subjects = cfg.get("subjects")
    num_topics = int(cfg["num_topics"])

    if explicit_subjects:
        requested = [str(x).strip() for x in explicit_subjects]
        by_subject = {row["subject"]: row for row in eligible}
        missing = [x for x in requested if x not in by_subject]
        if missing:
            raise RuntimeError(
                "Requested subjects do not satisfy topic budget: "
                f"{missing}"
            )
        selected = [by_subject[x] for x in requested]
    else:
        eligible.sort(
            key=lambda x: (
                -x["available_train_blocks"],
                -x["available_validation_blocks"],
                x["subject"],
            )
        )
        selected = eligible[:num_topics]

    if len(selected) != num_topics:
        raise RuntimeError(
            f"Need {num_topics} eligible subjects, got {len(selected)}"
        )

    max_common = int(cfg["max_common_train_blocks"])
    common_train_blocks = min(
        max_common,
        min(row["available_train_blocks"] for row in selected),
    )
    common_eval_blocks = min(
        int(cfg["validation_blocks"]),
        min(row["available_validation_blocks"] for row in selected),
    )

    if common_train_blocks <= 0 or common_eval_blocks <= 0:
        raise RuntimeError("Common topic block budget is zero")

    output_root = Path(
        os.path.expandvars(cfg["topic_adapter_output_root"])
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    topics: list[dict[str, Any]] = []

    print("\n[SELECTED TOPICS]")
    for idx, row in enumerate(selected):
        topic_id = f"topic{idx}"
        item = {
            "topic_id": topic_id,
            **row,
            "train_blocks": int(common_train_blocks),
            "eval_blocks": int(common_eval_blocks),
            "packing_seed": int(cfg["seed"]) + idx * 1000,
            "adapter_output": str(
                output_root / topic_id / f"seed{int(cfg['seed'])}"
            ),
        }
        topics.append(item)
        print(
            f"  {topic_id}: {row['subject']} | "
            f"train_rows={row['train_rows']} | "
            f"val_rows={row['validation_rows']} | "
            f"available_blocks={row['available_train_blocks']}"
        )

    plan = {
        "status": "READY",
        "git_branch": git_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status": git_output(["git", "status", "--short"]),
        "phase1_receipt": str(receipt_path),
        "phase1_archive": str(archive_path),
        "phase1_archive_sha256": actual_sha,
        "source_dataset": source_name,
        "common_train_blocks": int(common_train_blocks),
        "common_eval_blocks": int(common_eval_blocks),
        "sequence_length": seq_len,
        "seed": int(cfg["seed"]),
        "topics": topics,
    }

    plan_path = output_root / "topic_plan.json"
    save_json(plan, plan_path)

    print("\n[PLAN]")
    print(f"plan={plan_path}")
    print(
        f"common_train_tokens_per_epoch="
        f"{common_train_blocks * seq_len}"
    )

    if args.plan_only:
        print("\n[PLAN-ONLY] Topic audit/selection complete; training was not started.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; CPU fallback forbidden for topic-adapter training."
        )

    train_cfg = dict(base_train_cfg)
    train_cfg["num_train_epochs"] = base_train_cfg["num_train_epochs"]
    train_cfg["resume_completed"] = True

    summaries: dict[str, Any] = {}

    for topic in topics:
        topic_id = topic["topic_id"]
        subject = topic["subject"]
        packing_seed = int(topic["packing_seed"])

        train_sub = subset(train_ds, train_index[subject])
        val_sub = subset(val_ds, val_index[subject])

        packed_train = build_packed_blocks(
            train_sub,
            tokenizer=tokenizer,
            text_field=text_field,
            sequence_length=seq_len,
            num_blocks=common_train_blocks,
            seed=packing_seed,
            append_eos=append_eos,
        )

        packed_eval = build_packed_blocks(
            val_sub,
            tokenizer=tokenizer,
            text_field=text_field,
            sequence_length=seq_len,
            num_blocks=common_eval_blocks,
            seed=packing_seed + 1,
            append_eos=append_eos,
        )

        summary = train_one_adapter(
            dataset_name=f"{topic_id}:{subject}",
            train_dataset=packed_train,
            eval_dataset=packed_eval,
            model_cfg=model_cfg,
            lora_cfg=lora_cfg,
            train_cfg=train_cfg,
            output_dir=Path(topic["adapter_output"]),
            seed=int(cfg["seed"]),
        )

        summary["topic_id"] = topic_id
        summary["subject"] = subject
        summaries[topic_id] = summary

        save_json(
            {
                "status": "RUNNING",
                "topics": topics,
                "summaries": summaries,
            },
            output_root / "topic_train_summary.json",
        )

        del packed_train
        del packed_eval
        torch.cuda.empty_cache()

    save_json(
        {
            "status": "PASS",
            "topics": topics,
            "summaries": summaries,
        },
        output_root / "topic_train_summary.json",
    )

    print("\n" + "=" * 80)
    print("PHASE 3B TOPIC-ADAPTER TRAINING PASS")
    print("=" * 80)
    for topic_id, summary in summaries.items():
        print(
            f"{topic_id:8s} | {summary['subject']} | "
            f"eval_loss={summary['eval_metrics']['eval_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
