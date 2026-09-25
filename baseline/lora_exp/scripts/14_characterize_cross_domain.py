#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lora_exp.analysis.similarity import compare_feature_sets
from lora_exp.instrument.qkv_capture import PrefillQKVCapture
from lora_exp.model.factory import load_base_model, load_tokenizer
from lora_exp.train.packing import load_parquet_dataset


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


def git_output(args: list[str]) -> str:
    p = subprocess.run(
        args, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return p.stdout.strip() if p.returncode == 0 else "<git-failed>"


def select_probes(cfg, dataset_root: Path, tokenizer) -> list[dict[str, Any]]:
    per_dataset = int(cfg["probe_records_per_dataset"])
    max_tokens = int(cfg["max_probe_tokens"])
    seed = int(cfg["seed"])
    probes = []

    for idx, name in enumerate(cfg["datasets"].keys()):
        ds = load_parquet_dataset(dataset_root / name / "test.parquet")
        indices = list(range(len(ds)))
        random.Random(seed + idx * 1000).shuffle(indices)

        selected = 0
        for j in indices:
            row = ds[int(j)]
            prompt = str(row["probe_prompt"])
            n = len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                )["input_ids"]
            )
            if n <= 0 or n > max_tokens:
                continue
            probes.append(
                {
                    "probe_id": f"{name}::{row['record_id']}",
                    "record_id": str(row["record_id"]),
                    "probe_dataset": name,
                    "probe_domain": cfg["datasets"][name]["domain"],
                    "probe_task": cfg["datasets"][name]["task"],
                    "prompt": prompt,
                    "token_count": n,
                }
            )
            selected += 1
            if selected >= per_dataset:
                break

        if selected != per_dataset:
            raise RuntimeError(
                f"{name}: requested {per_dataset} probes, got {selected}"
            )

    return probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3c_cross_domain.yaml")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden.")

    cfg = read_yaml(PROJECT_ROOT / args.config)["phase3c"]
    model_yaml = read_yaml(PROJECT_ROOT / cfg["model_config"])
    model_cfg = model_yaml["model"]
    arch = model_yaml["expected_architecture"]
    seed = int(cfg["seed"])

    dataset_root = Path(os.path.expandvars(cfg["dataset_root"])).resolve()
    adapter_root = Path(
        os.path.expandvars(cfg["adapter_output_root"])
    ).resolve()
    output_root = Path(
        os.path.expandvars(cfg["characterization_output_root"])
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_summary = read_json(adapter_root / "train_summary.json")
    cross_eval = read_json(adapter_root / "cross_eval.json")
    if train_summary.get("status") != "PASS":
        raise RuntimeError("Adapter training summary is not PASS.")
    if cross_eval.get("status") != "PASS":
        raise RuntimeError("Cross-eval is not PASS.")

    tokenizer = load_tokenizer(model_cfg)
    probes = select_probes(cfg, dataset_root, tokenizer)
    save_json(probes, output_root / "probes.json")

    names = list(cfg["datasets"].keys())
    base = load_base_model(model_cfg, device="cuda", use_cache=True)

    def adapter_dir(name: str) -> Path:
        return adapter_root / name / f"seed{seed}" / "adapter_final"

    model = PeftModel.from_pretrained(
        base,
        adapter_dir(names[0]),
        adapter_name=names[0],
        is_trainable=False,
    )
    for name in names[1:]:
        model.load_adapter(
            adapter_dir(name), adapter_name=name, is_trainable=False
        )

    model.eval()
    model.config.use_cache = True

    capture = PrefillQKVCapture(
        model,
        num_layers=int(arch["num_hidden_layers"]),
        num_q_heads=int(arch["num_attention_heads"]),
        num_kv_heads=int(arch["num_key_value_heads"]),
        head_dim=int(arch["head_dim"]),
    )
    print(f"[Q CAPTURE MODE] {capture.q_mode}")

    pairs = list(itertools.combinations(names, 2))
    detail_path = output_root / "same_input_detail.csv"

    fieldnames = [
        "probe_id",
        "record_id",
        "probe_dataset",
        "probe_domain",
        "probe_task",
        "token_count",
        "pair_a",
        "pair_b",
        "pair_relation",
        "tensor_type",
        "layer",
        "num_vectors",
        "seq_len",
        "num_heads",
        "cosine_mean",
        "cosine_median",
        "cosine_std",
        "cosine_p05",
        "cosine_p95",
        "fraction_cos_ge_095",
        "fraction_cos_ge_099",
        "global_cosine",
        "symmetric_rel_l2",
        "delta_global_cosine",
        "delta_norm_ratio_a",
        "delta_norm_ratio_b",
        "delta_distance_vs_base",
    ]

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            for probe_idx, probe in enumerate(probes, start=1):
                batch = tokenizer(
                    probe["prompt"],
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                )
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()

                with model.disable_adapter():
                    base_features = capture.capture(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                adapter_features = {}
                for name in names:
                    model.set_adapter(name)
                    adapter_features[name] = capture.capture(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                for a, b in pairs:
                    relation = (
                        "within_domain"
                        if cfg["datasets"][a]["domain"]
                        == cfg["datasets"][b]["domain"]
                        else "cross_domain"
                    )
                    rows = compare_feature_sets(
                        adapter_features[a],
                        adapter_features[b],
                        base_features=base_features,
                    )
                    for row in rows:
                        writer.writerow(
                            {
                                **{k: probe[k] for k in [
                                    "probe_id",
                                    "record_id",
                                    "probe_dataset",
                                    "probe_domain",
                                    "probe_task",
                                    "token_count",
                                ]},
                                "pair_a": a,
                                "pair_b": b,
                                "pair_relation": relation,
                                **row,
                            }
                        )

                f.flush()
                print(f"[PROBE] {probe_idx}/{len(probes)} {probe['probe_id']}")

                del base_features, adapter_features, batch, input_ids, attention_mask
                torch.cuda.empty_cache()

        finally:
            capture.close()

    manifest = {
        "status": "PASS",
        "git_branch": git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status": git_output(["git", "status", "--short"]),
        "q_capture_mode": capture.q_mode,
        "num_probes": len(probes),
        "adapter_names": names,
        "pairs": [
            {
                "a": a,
                "b": b,
                "relation": (
                    "within_domain"
                    if cfg["datasets"][a]["domain"]
                    == cfg["datasets"][b]["domain"]
                    else "cross_domain"
                ),
            }
            for a, b in pairs
        ],
        "detail_csv": str(detail_path),
    }
    save_json(manifest, output_root / "characterization_manifest.json")

    print("\n" + "=" * 80)
    print("PHASE 3C SAME-INPUT QKV CHARACTERIZATION PASS")
    print(f"probes={len(probes)}")
    print(f"detail={detail_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
