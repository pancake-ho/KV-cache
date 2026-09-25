#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lora_exp.model.factory import load_base_model, load_tokenizer
from lora_exp.train.packing import (
    FixedBlockCausalCollator,
    build_packed_blocks,
    load_parquet_dataset,
)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            candidate = (destination / member.name).resolve()
            if candidate != root and root not in candidate.parents:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
        tar.extractall(destination)


def clean_subject(value: Any) -> str:
    return "" if value is None else str(value).strip()


def subject_indices(dataset, subject: str) -> list[int]:
    return [
        idx
        for idx, value in enumerate(dataset["subject"])
        if clean_subject(value) == subject
    ]


def evaluate_blocks(model, dataset) -> float:
    collator = FixedBlockCausalCollator()
    losses: list[float] = []

    model.eval()

    with torch.inference_mode():
        for idx in range(len(dataset)):
            batch = collator([dataset[idx]])
            batch = {
                key: value.cuda(non_blocking=True)
                for key, value in batch.items()
            }

            output = model(
                **batch,
                use_cache=False,
                return_dict=True,
            )

            loss = float(output.loss.detach().cpu())
            if not math.isfinite(loss):
                raise RuntimeError("Non-finite eval loss")
            losses.append(loss)

    return float(sum(losses) / len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/topic_userpair.yaml",
    )
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")

    cfg = read_yaml(PROJECT_ROOT / args.config)["phase3b"]

    model_cfg = read_yaml(
        PROJECT_ROOT / cfg["model_config"]
    )["model"]

    work_dir = Path(os.path.expandvars(args.work_dir)).resolve()
    if work_dir.exists():
        raise RuntimeError(f"work_dir exists: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=False)

    output_root = Path(
        os.path.expandvars(cfg["topic_adapter_output_root"])
    ).resolve()

    plan_path = output_root / "topic_plan.json"
    plan = read_json(plan_path)

    if plan.get("status") != "READY":
        raise RuntimeError("topic_plan.json is not READY")

    receipt = read_json(PROJECT_ROOT / cfg["phase1_receipt"])
    archive = Path(receipt["archive"])
    extract_root = work_dir / "phase1"
    safe_extract(archive, extract_root)

    dataset_root = extract_root / cfg["phase1_dataset_root"]
    val_ds = load_parquet_dataset(
        dataset_root / cfg["source_dataset"] / "validation.parquet"
    )

    tokenizer = load_tokenizer(model_cfg)

    text_field = str(cfg["text_field"])
    seq_len = int(cfg["sequence_length"])
    append_eos = bool(cfg["append_eos_between_samples"])

    eval_sets = {}

    for topic in plan["topics"]:
        indices = subject_indices(val_ds, topic["subject"])
        sub = val_ds.select(indices)

        eval_sets[topic["topic_id"]] = build_packed_blocks(
            sub,
            tokenizer=tokenizer,
            text_field=text_field,
            sequence_length=seq_len,
            num_blocks=int(plan["common_eval_blocks"]),
            seed=int(topic["packing_seed"]) + 1,
            append_eos=append_eos,
        )

    base = load_base_model(
        model_cfg,
        device="cuda",
        use_cache=False,
    )

    topics = plan["topics"]
    first = topics[0]

    def adapter_dir(topic: dict[str, Any]) -> Path:
        return Path(topic["adapter_output"]) / "adapter_final"

    for topic in topics:
        if not adapter_dir(topic).exists():
            raise FileNotFoundError(adapter_dir(topic))

    model = PeftModel.from_pretrained(
        base,
        adapter_dir(first),
        adapter_name=first["topic_id"],
        is_trainable=False,
    )

    for topic in topics[1:]:
        model.load_adapter(
            adapter_dir(topic),
            adapter_name=topic["topic_id"],
            is_trainable=False,
        )

    model.eval()

    rows: list[dict[str, Any]] = []

    print("\n[EVAL BASE]")
    with model.disable_adapter():
        for target in topics:
            topic_id = target["topic_id"]
            loss = evaluate_blocks(model, eval_sets[topic_id])
            rows.append(
                {
                    "model": "base",
                    "target_topic_id": topic_id,
                    "target_subject": target["subject"],
                    "eval_loss": loss,
                }
            )
            print(f"  base -> {topic_id}: {loss:.6f}")

    for source in topics:
        source_id = source["topic_id"]
        model.set_adapter(source_id)
        print(f"\n[EVAL {source_id}]")

        for target in topics:
            target_id = target["topic_id"]
            loss = evaluate_blocks(model, eval_sets[target_id])
            rows.append(
                {
                    "model": source_id,
                    "target_topic_id": target_id,
                    "target_subject": target["subject"],
                    "eval_loss": loss,
                }
            )
            print(f"  {source_id} -> {target_id}: {loss:.6f}")

    base_loss = {
        row["target_topic_id"]: row["eval_loss"]
        for row in rows
        if row["model"] == "base"
    }

    for row in rows:
        row["improvement_vs_base"] = (
            base_loss[row["target_topic_id"]]
            - row["eval_loss"]
        )
        row["is_own_topic"] = (
            row["model"] == row["target_topic_id"]
        )

    csv_path = output_root / "topic_cross_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "target_topic_id",
                "target_subject",
                "eval_loss",
                "improvement_vs_base",
                "is_own_topic",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    own_rows = [
        row
        for row in rows
        if row["model"] != "base"
        and row["is_own_topic"]
    ]

    own_improved = all(
        row["improvement_vs_base"] > 0.0
        for row in own_rows
    )

    diagonal_best = {}
    for target in topics:
        target_id = target["topic_id"]
        candidates = [
            row
            for row in rows
            if row["model"] != "base"
            and row["target_topic_id"] == target_id
        ]
        best = min(candidates, key=lambda x: x["eval_loss"])
        diagonal_best[target_id] = (
            best["model"] == target_id
        )

    summary = {
        "status": "PASS",
        "own_topic_improves_vs_base": own_improved,
        "own_topic_is_best_adapter": diagonal_best,
        "rows": rows,
    }

    save_json(
        summary,
        output_root / "topic_cross_eval.json",
    )

    print("\n" + "=" * 80)
    print("PHASE 3B TOPIC CROSS-EVAL PASS")
    print(f"own_topic_improves_vs_base={own_improved}")
    print(f"own_topic_is_best_adapter={diagonal_best}")
    print(f"csv={csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
