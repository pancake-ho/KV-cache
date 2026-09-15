#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def eval_blocks(model, dataset) -> float:
    collator = FixedBlockCausalCollator()
    losses = []
    model.eval()
    with torch.inference_mode():
        for i in range(len(dataset)):
            batch = collator([dataset[i]])
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            out = model(**batch, use_cache=False, return_dict=True)
            loss = float(out.loss.detach().cpu())
            if not math.isfinite(loss):
                raise RuntimeError("Non-finite eval loss.")
            losses.append(loss)
    return float(sum(losses) / len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3c_cross_domain.yaml")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden.")

    cfg = read_yaml(PROJECT_ROOT / args.config)["phase3c"]
    model_cfg = read_yaml(PROJECT_ROOT / cfg["model_config"])["model"]
    dataset_root = Path(os.path.expandvars(cfg["dataset_root"])).resolve()
    output_root = Path(
        os.path.expandvars(cfg["adapter_output_root"])
    ).resolve()
    seed = int(cfg["seed"])
    seq_len = int(cfg["sequence_length"])
    val_blocks_target = int(cfg["validation_blocks"])

    manifest = read_json(dataset_root / "manifest.json")
    if manifest.get("status") != "PASS":
        raise RuntimeError("Dataset manifest not PASS.")

    tokenizer = load_tokenizer(model_cfg)
    eval_sets = {}

    for idx, name in enumerate(cfg["datasets"].keys()):
        val_ds = load_parquet_dataset(dataset_root / name / "validation.parquet")
        feasible = sum(int(x) + 1 for x in val_ds["native_token_count"]) // seq_len
        n_blocks = min(val_blocks_target, feasible)
        eval_sets[name] = build_packed_blocks(
            val_ds,
            tokenizer=tokenizer,
            text_field="native_text",
            sequence_length=seq_len,
            num_blocks=n_blocks,
            seed=seed + idx * 1000 + 1,
            append_eos=True,
        )

    base = load_base_model(model_cfg, device="cuda", use_cache=False)

    names = list(cfg["datasets"].keys())

    def adapter_dir(name: str) -> Path:
        return output_root / name / f"seed{seed}" / "adapter_final"

    model = PeftModel.from_pretrained(
        base,
        adapter_dir(names[0]),
        adapter_name=names[0],
        is_trainable=False,
    )
    for name in names[1:]:
        model.load_adapter(
            adapter_dir(name),
            adapter_name=name,
            is_trainable=False,
        )
    model.eval()

    rows = []

    with model.disable_adapter():
        for target in names:
            loss = eval_blocks(model, eval_sets[target])
            rows.append(
                {
                    "model": "base",
                    "model_domain": "base",
                    "target": target,
                    "target_domain": cfg["datasets"][target]["domain"],
                    "eval_loss": loss,
                }
            )
            print(f"base -> {target}: {loss:.6f}")

    for source in names:
        model.set_adapter(source)
        for target in names:
            loss = eval_blocks(model, eval_sets[target])
            rows.append(
                {
                    "model": source,
                    "model_domain": cfg["datasets"][source]["domain"],
                    "target": target,
                    "target_domain": cfg["datasets"][target]["domain"],
                    "eval_loss": loss,
                }
            )
            print(f"{source} -> {target}: {loss:.6f}")

    base_losses = {
        row["target"]: row["eval_loss"] for row in rows if row["model"] == "base"
    }
    for row in rows:
        row["improvement_vs_base"] = (
            base_losses[row["target"]] - row["eval_loss"]
        )
        row["is_own_dataset"] = row["model"] == row["target"]
        row["same_domain"] = (
            row["model"] != "base"
            and row["model_domain"] == row["target_domain"]
        )

    csv_path = output_root / "cross_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    own = [
        row
        for row in rows
        if row["model"] != "base" and row["is_own_dataset"]
    ]
    own_improves = all(row["improvement_vs_base"] > 0 for row in own)

    payload = {
        "status": "PASS",
        "own_dataset_improves_vs_base": own_improves,
        "rows": rows,
    }
    save_json(payload, output_root / "cross_eval.json")

    print("\n" + "=" * 80)
    print("PHASE 3C CROSS-EVAL PASS")
    print(f"own_dataset_improves_vs_base={own_improves}")
    print(f"csv={csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
