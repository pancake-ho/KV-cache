#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lora_exp.model.factory import load_tokenizer
from lora_exp.train.packing import build_packed_blocks, load_parquet_dataset
from lora_exp.train.runner import train_one_adapter


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3c_cross_domain.yaml")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden.")

    cfg = read_yaml(PROJECT_ROOT / args.config)["phase3c"]
    model_cfg = read_yaml(PROJECT_ROOT / cfg["model_config"])["model"]
    lora_cfg = read_yaml(PROJECT_ROOT / cfg["lora_config"])["lora"]
    base_train_cfg = read_yaml(PROJECT_ROOT / cfg["base_train_config"])["phase2"]

    dataset_root = Path(os.path.expandvars(cfg["dataset_root"])).resolve()
    manifest = read_json(dataset_root / "manifest.json")
    if manifest.get("status") != "PASS":
        raise RuntimeError("Cross-domain data manifest is not PASS.")

    common_blocks = int(manifest["common_train_blocks"])
    seq_len = int(cfg["sequence_length"])
    val_blocks = int(cfg["validation_blocks"])
    seed = int(cfg["seed"])

    tokenizer = load_tokenizer(model_cfg)

    train_cfg = dict(base_train_cfg)
    train_cfg["gradient_accumulation_steps"] = int(
        cfg["gradient_accumulation_steps"]
    )
    train_cfg["resume_completed"] = True

    output_root = Path(
        os.path.expandvars(cfg["adapter_output_root"])
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = {}

    for idx, name in enumerate(cfg["datasets"].keys()):
        train_ds = load_parquet_dataset(dataset_root / name / "train.parquet")
        val_ds = load_parquet_dataset(dataset_root / name / "validation.parquet")

        packed_train = build_packed_blocks(
            train_ds,
            tokenizer=tokenizer,
            text_field="native_text",
            sequence_length=seq_len,
            num_blocks=common_blocks,
            seed=seed + idx * 1000,
            append_eos=True,
        )

        # Validation budget must be feasible for each dataset.
        total_val_tokens = sum(
            int(x) + 1 for x in val_ds["native_token_count"]
        )
        feasible_val_blocks = total_val_tokens // seq_len
        use_val_blocks = min(val_blocks, feasible_val_blocks)
        if use_val_blocks < 4:
            raise RuntimeError(
                f"{name}: too few validation blocks ({use_val_blocks})"
            )

        packed_val = build_packed_blocks(
            val_ds,
            tokenizer=tokenizer,
            text_field="native_text",
            sequence_length=seq_len,
            num_blocks=use_val_blocks,
            seed=seed + idx * 1000 + 1,
            append_eos=True,
        )

        summary = train_one_adapter(
            dataset_name=name,
            train_dataset=packed_train,
            eval_dataset=packed_val,
            model_cfg=model_cfg,
            lora_cfg=lora_cfg,
            train_cfg=train_cfg,
            output_dir=output_root / name / f"seed{seed}",
            seed=seed,
        )
        summary["domain"] = cfg["datasets"][name]["domain"]
        summary["task"] = cfg["datasets"][name]["task"]
        summaries[name] = summary

        save_json(
            {"status": "RUNNING", "summaries": summaries},
            output_root / "train_summary.json",
        )

        del packed_train, packed_val
        torch.cuda.empty_cache()

    save_json(
        {
            "status": "PASS",
            "common_train_blocks": common_blocks,
            "sequence_length": seq_len,
            "gradient_accumulation_steps": train_cfg[
                "gradient_accumulation_steps"
            ],
            "summaries": summaries,
        },
        output_root / "train_summary.json",
    )

    print("\n" + "=" * 80)
    print("PHASE 3C CROSS-DOMAIN ADAPTER TRAINING PASS")
    for name, s in summaries.items():
        print(
            f"{name:10s} domain={s['domain']:7s} "
            f"eval_loss={s['eval_metrics']['eval_loss']:.6f} "
            f"epoch={s['train_metrics']['epoch']}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
