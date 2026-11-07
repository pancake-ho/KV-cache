"""Estimate Figure-2 disk requirements before allocating large tensor stores.

No model load, package install or GPU required. Imported only by launchers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil


def estimate_bytes(cfg: dict, *, smoke: bool) -> int:
    sequences = 12 if smoke else cfg["train_sequences"] + cfg["heldout_sequences"]
    length = 128 if smoke else cfg["sequence_length"]
    samples_per_sequence = math.ceil(length / cfg["stride"])
    layers = cfg["expected_source_layers"] + cfg["expected_target_layers"]
    # Three cache variants (K RoPE, K stripped, V); float16 storage.
    return sequences * samples_per_sequence * layers * cfg["expected_kv_heads"] * cfg["expected_head_dim"] * 3 * 2


def check_space(cfg: dict, *, dataset: Path, scratch: Path, smoke: bool, staging: bool = False) -> dict:
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not scratch.is_dir():
        raise NotADirectoryError(scratch)
    estimated = estimate_bytes(cfg, smoke=smoke)
    # Headroom for dataset copy, plot/checkpoints, temporary model/download files.
    headroom_gib = int(os.environ.get("KV_FIG2_HEADROOM_GIB", "20"))
    if headroom_gib < 1:
        raise ValueError("KV_FIG2_HEADROOM_GIB must be a positive integer")
    reserved = headroom_gib * (1024 ** 3)
    dataset_bytes = dataset.stat().st_size
    # Include the dataset copy in the initial pre-staging free-space check.
    required = estimated + reserved + (dataset_bytes if staging else 0)
    free = shutil.disk_usage(scratch).free
    report = {
        "mode": "smoke" if smoke else "full",
        "estimated_kv_arrays_bytes": estimated,
        "dataset_bytes": dataset_bytes,
        "staging_dataset": staging,
        "required_free_bytes": required,
        "free_bytes_at_check": free,
        "headroom_bytes": reserved,
        "headroom_for_model_files_and_temporary_outputs": True,
        "storage_note": "raw per-layer K/V arrays stay in compute-node scratch; no automatic NAS copy",
    }
    if free < required:
        raise RuntimeError(f"Insufficient scratch: need {required / 2**30:.1f} GiB, "
                           f"available {free / 2**30:.1f} GiB")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--staging", action="store_true", help="Include dataset copy in storage estimate")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    report = check_space(cfg, dataset=args.dataset, scratch=args.scratch,
                         smoke=args.mode == "smoke", staging=args.staging)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
