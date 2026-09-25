#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lora_exp.model.factory import load_tokenizer


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def git_output(args: list[str]) -> str:
    result = subprocess.run(
        args, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "<git-failed>"


def resolve_revision(repo_id: str) -> str:
    info = HfApi().dataset_info(repo_id)
    if not info.sha:
        raise RuntimeError(f"Could not resolve immutable revision for {repo_id}")
    return str(info.sha)


def load_split(
    repo_id: str,
    revision: str,
    split: str,
    cache_dir: Path,
    config_name: str | None = None,
) -> Dataset:
    kwargs = {
        "path": repo_id,
        "revision": revision,
        "split": split,
        "cache_dir": str(cache_dir),
        "trust_remote_code": False,
    }
    if config_name:
        kwargs["name"] = config_name
    return load_dataset(**kwargs)


def clean(x: Any) -> str:
    return "" if x is None else str(x).strip()


def normalize_options(options: Any) -> list[str]:
    if options is None:
        return []
    if isinstance(options, dict):
        return [clean(options[k]) for k in sorted(options.keys())]
    if isinstance(options, (list, tuple)):
        out = []
        for item in options:
            if isinstance(item, dict) and "value" in item:
                out.append(clean(item["value"]))
            else:
                out.append(clean(item))
        return out
    raise TypeError(f"Unsupported options type={type(options)}")


def format_mcq_input(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question.strip()}", "", "Choices:"]
    for i, choice in enumerate(choices):
        lines.append(f"{LABELS[i]}. {choice}")
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def format_mcq_train(question: str, choices: list[str], answer_letter: str) -> str:
    return format_mcq_input(question, choices) + f" {answer_letter}\n"


def format_sentiment_input(text: str) -> str:
    return (
        "Task: Classify the sentiment of the following financial text "
        "as negative, neutral, or positive.\n\n"
        f"Input: {text.strip()}\n\n"
        "Answer:"
    )


def format_sentiment_train(text: str, answer: str) -> str:
    return format_sentiment_input(text) + f" {answer.strip().lower()}\n"


def canonicalize_medqa(ds: Dataset, split_name: str) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(ds):
        question = clean(row.get("question"))
        choices = normalize_options(row.get("options"))
        answer_idx = clean(row.get("answer_idx")).upper()
        if not question or len(choices) < 4 or answer_idx not in LABELS[: len(choices)]:
            continue
        rid = f"medqa::{split_name}::{idx}"
        rows.append(
            {
                "record_id": rid,
                "dataset": "medqa",
                "domain": "medical",
                "task": "closed_qa",
                "question": question,
                "content": question,
                "choices": choices,
                "answer": answer_idx,
                "native_text": format_mcq_train(question, choices, answer_idx),
                "probe_prompt": format_mcq_input(question, choices),
                "source_split": split_name,
            }
        )
    return rows


def canonicalize_medmcqa(ds: Dataset, split_name: str) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(ds):
        question = clean(row.get("question"))
        choices = [
            clean(row.get("opa")),
            clean(row.get("opb")),
            clean(row.get("opc")),
            clean(row.get("opd")),
        ]
        cop = row.get("cop")
        try:
            cop_i = int(cop)
        except Exception:
            continue
        # HF MedMCQA uses 0..3 in some revisions and 1..4 in others.
        if 0 <= cop_i <= 3:
            answer_idx = LABELS[cop_i]
        elif 1 <= cop_i <= 4:
            answer_idx = LABELS[cop_i - 1]
        else:
            continue
        if not question or any(not x for x in choices):
            continue
        rid = clean(row.get("id")) or f"medmcqa::{split_name}::{idx}"
        rows.append(
            {
                "record_id": rid,
                "dataset": "medmcqa",
                "domain": "medical",
                "task": "closed_qa",
                "question": question,
                "content": question,
                "choices": choices,
                "answer": answer_idx,
                "native_text": format_mcq_train(question, choices, answer_idx),
                "probe_prompt": format_mcq_input(question, choices),
                "source_split": split_name,
            }
        )
    return rows


def canonicalize_finance(
    ds: Dataset,
    dataset_name: str,
    split_name: str,
) -> list[dict[str, Any]]:
    rows = []
    valid_answers = {"negative", "neutral", "positive"}
    for idx, row in enumerate(ds):
        text = clean(row.get("text"))
        answer = clean(row.get("answer")).lower()
        if not text:
            # Defensive fallback for compatible repositories.
            text = clean(row.get("sentence"))
        if not answer and row.get("label") is not None:
            answer = clean(row.get("label")).lower()
        if not text or answer not in valid_answers:
            continue
        rid = clean(row.get("id")) or f"{dataset_name}::{split_name}::{idx}"
        rows.append(
            {
                "record_id": rid,
                "dataset": dataset_name,
                "domain": "finance",
                "task": "sentiment",
                "question": "",
                "content": text,
                "choices": ["negative", "neutral", "positive"],
                "answer": answer,
                "native_text": format_sentiment_train(text, answer),
                "probe_prompt": format_sentiment_input(text),
                "source_split": split_name,
            }
        )
    return rows


def deterministic_split(
    rows: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(round(len(rows) * validation_fraction)))
    val_set = set(indices[:n_val])
    train_rows = [row for i, row in enumerate(rows) if i not in val_set]
    val_rows = [row for i, row in enumerate(rows) if i in val_set]
    return train_rows, val_rows


def attach_token_counts(rows: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["native_token_count"] = len(
            tokenizer(
                item["native_text"],
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )["input_ids"]
        )
        item["probe_token_count"] = len(
            tokenizer(
                item["probe_prompt"],
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )["input_ids"]
        )
        out.append(item)
    return out


def available_blocks(rows: list[dict[str, Any]], seq_len: int) -> int:
    # EOS separators between examples are included conservatively.
    total = sum(int(row["native_token_count"]) + 1 for row in rows)
    return total // seq_len


def save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3c_cross_domain.yaml")
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    cfg = read_yaml(PROJECT_ROOT / args.config)["phase3c"]
    model_cfg = read_yaml(PROJECT_ROOT / cfg["model_config"])["model"]

    root = Path(os.path.expandvars(cfg["dataset_root"])).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.path.expandvars(args.cache_dir)).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_cfg)
    seq_len = int(cfg["sequence_length"])
    seed = int(cfg["seed"])

    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "git_branch": git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status": git_output(["git", "status", "--short"]),
        "model_id": model_cfg["model_id"],
        "model_revision": model_cfg.get("revision", "main"),
        "sequence_length": seq_len,
        "seed": seed,
        "datasets": {},
    }

    for ds_idx, (name, dcfg) in enumerate(cfg["datasets"].items()):
        repo_id = dcfg["repo_id"]
        revision = resolve_revision(repo_id)
        config_name = dcfg.get("config_name")
        print(f"\n[{name}] repo={repo_id} revision={revision}")

        source_train = load_split(
            repo_id, revision, dcfg["source_train"], cache_dir, config_name
        )

        if name == "medqa":
            full_train = canonicalize_medqa(source_train, "train")
        elif name == "medmcqa":
            full_train = canonicalize_medmcqa(source_train, "train")
        else:
            full_train = canonicalize_finance(source_train, name, "train")

        if "internal_validation_fraction" in dcfg:
            train_rows, val_rows = deterministic_split(
                full_train,
                float(dcfg["internal_validation_fraction"]),
                seed + ds_idx * 1000,
            )
        else:
            source_val = load_split(
                repo_id,
                revision,
                dcfg["source_validation"],
                cache_dir,
                config_name,
            )
            if name == "fpb" or name == "fiqasa":
                val_rows = canonicalize_finance(
                    source_val, name, dcfg["source_validation"]
                )
            else:
                raise RuntimeError(f"Unexpected validation path for {name}")
            train_rows = full_train

        source_test = load_split(
            repo_id, revision, dcfg["source_test"], cache_dir, config_name
        )
        if name == "medqa":
            test_rows = canonicalize_medqa(source_test, "test")
        elif name == "medmcqa":
            test_rows = canonicalize_medmcqa(source_test, "validation")
        else:
            test_rows = canonicalize_finance(
                source_test, name, dcfg["source_test"]
            )

        train_rows = attach_token_counts(train_rows, tokenizer)
        val_rows = attach_token_counts(val_rows, tokenizer)
        test_rows = attach_token_counts(test_rows, tokenizer)

        if not train_rows or not val_rows or not test_rows:
            raise RuntimeError(f"{name}: empty canonical split")

        ds_dir = root / name
        save_rows(train_rows, ds_dir / "train.parquet")
        save_rows(val_rows, ds_dir / "validation.parquet")
        save_rows(test_rows, ds_dir / "test.parquet")

        blocks = available_blocks(train_rows, seq_len)

        manifest["datasets"][name] = {
            "domain": dcfg["domain"],
            "task": dcfg["task"],
            "repo_id": repo_id,
            "revision": revision,
            "config_name": config_name,
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "test_rows": len(test_rows),
            "available_train_blocks": int(blocks),
            "train_token_sum_with_eos": int(
                sum(int(x["native_token_count"]) + 1 for x in train_rows)
            ),
            "train_token_min": int(min(x["native_token_count"] for x in train_rows)),
            "train_token_max": int(max(x["native_token_count"] for x in train_rows)),
        }

        print(
            f"  rows train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)}"
        )
        print(f"  available {seq_len}-token blocks={blocks}")

    min_blocks = min(
        item["available_train_blocks"]
        for item in manifest["datasets"].values()
    )
    common_blocks = min(int(cfg["max_common_train_blocks"]), int(min_blocks))

    manifest["minimum_available_train_blocks"] = int(min_blocks)
    manifest["common_train_blocks"] = int(common_blocks)
    manifest["common_train_tokens_per_epoch"] = int(common_blocks * seq_len)
    manifest["recommended_gradient_accumulation_steps"] = int(
        cfg["gradient_accumulation_steps"]
    )

    if common_blocks < int(cfg["min_common_train_blocks"]):
        manifest["status"] = "FAIL_BUDGET"
        save_json(manifest, root / "manifest.json")
        raise RuntimeError(
            f"Common budget too small: {common_blocks} < "
            f"{cfg['min_common_train_blocks']}. Do not train yet."
        )

    manifest["status"] = "PASS"
    save_json(manifest, root / "manifest.json")

    print("\n" + "=" * 80)
    print("PHASE 3C DATA PREPARATION / AUDIT PASS")
    print(f"dataset_root={root}")
    print(f"common_train_blocks={common_blocks}")
    print(f"tokens_per_epoch={common_blocks * seq_len}")
    print("=" * 80)


if __name__ == "__main__":
    main()
