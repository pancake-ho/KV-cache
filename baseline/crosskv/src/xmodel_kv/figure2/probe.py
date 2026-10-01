from __future__ import annotations

import argparse
import csv
from pathlib import Path
import os
import numpy as np

from .common import KINDS, digest, lock, provenance, read_json, write_json
from .regression import Statistics, fit_pair, ols_inverse
from .store import ProbeStore, validate_pair


def save_checkpoint(path, arrays):
    temporary = Path(str(path) + ".pending.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def run_probe(run_dir, device="cuda:0"):
    root = Path(run_dir)
    source, target = ProbeStore(root / "source"), ProbeStore(root / "target")
    validate_pair(source, target)
    sm, tm = source.metadata, target.metadata
    cfg = sm["config"]
    n_train = sm["train_observations"]
    n_total = sm["num_observations"]
    if n_train < 2 or n_total - n_train < 2:
        raise ValueError("fit and heldout both require at least two observations")
    out = root / "results"
    out.mkdir(exist_ok=True)
    engine = Statistics(device, cfg["covariance_dtype"])
    shape = (len(KINDS), sm["num_layers"], tm["num_layers"], tm["num_kv_heads"])
    identity = {
        "source_identity": sm["identity_hash"], "target_identity": tm["identity_hash"],
        "covariance_dtype": cfg["covariance_dtype"], "covariance_rcond": cfg["covariance_rcond"],
        "regression": "centered single-source matched-head OLS with numerical pseudoinverse; no ridge",
        "primary_reduction": "uniform mean across nonconstant output channels, then equal mean across KV heads",
        "axis_order": ["kind", "source_layer", "target_layer", "kv_head"],
        "device": device,
    }
    identity_hash = digest(identity)
    manifest_path, checkpoint = out / "probe.json", out / "pairwise_r2.npz"
    with lock(out / ".probe.lock"):
        if checkpoint.exists():
            old = read_json(manifest_path)
            if old["identity_hash"] != identity_hash:
                raise ValueError("existing probe checkpoint belongs to different inputs/settings/device")
            with np.load(checkpoint) as saved:
                arrays = {name: np.array(saved[name]) for name in saved.files}
            if arrays["train_uniform"].shape != shape:
                raise ValueError("checkpoint shape mismatch")
        else:
            arrays = {name: np.full(shape, np.nan, dtype=np.float64)
                      for name in ("train_uniform", "test_uniform", "train_weighted", "test_weighted")}
            arrays.update(done=np.zeros(shape[:3], dtype=bool),
                          source_rank=np.zeros((len(KINDS), sm["num_layers"], sm["num_kv_heads"]), dtype=np.int32),
                          source_min_eigenvalue=np.zeros((len(KINDS), sm["num_layers"], sm["num_kv_heads"])),
                          train_valid_channels=np.zeros(shape, dtype=np.int32),
                          test_valid_channels=np.zeros(shape, dtype=np.int32),
                          train_roundoff_channels=np.zeros(shape, dtype=np.int32),
                          test_roundoff_channels=np.zeros(shape, dtype=np.int32))
            write_json(manifest_path, {**identity, "identity_hash": identity_hash,
                                      "complete": False, "provenance": provenance()})
        for kind_index, kind in enumerate(KINDS):
            for source_layer in range(sm["num_layers"]):
                if arrays["done"][kind_index, source_layer].all():
                    continue
                data = source.open(kind, source_layer)
                source_train = engine.moments(data[:n_train], covariance=True)
                source_test = engine.moments(data[n_train:], covariance=True)
                inverse, rank, smallest = ols_inverse(source_train.covariance, cfg["covariance_rcond"])
                arrays["source_rank"][kind_index, source_layer] = rank
                arrays["source_min_eigenvalue"][kind_index, source_layer] = smallest
                for target_layer in range(tm["num_layers"]):
                    position = (kind_index, source_layer, target_layer)
                    if arrays["done"][position]:
                        continue
                    target_data = target.open(kind, target_layer)
                    target_train = engine.moments(target_data[:n_train], covariance=False)
                    target_test = engine.moments(target_data[n_train:], covariance=False)
                    train, test = fit_pair(engine, source_train, target_train, source_test, target_test, inverse)
                    for name, result in (("train", train), ("test", test)):
                        arrays[f"{name}_uniform"][position] = result["uniform"]
                        arrays[f"{name}_weighted"][position] = result["weighted"]
                        arrays[f"{name}_valid_channels"][position] = result["valid_channels"]
                        arrays[f"{name}_roundoff_channels"][position] = result["roundoff_clamped_channels"]
                    arrays["done"][position] = True
                    del target_train, target_test, target_data
                save_checkpoint(checkpoint, arrays)
                print(f"{kind}: source layer {source_layer + 1}/{sm['num_layers']} persisted", flush=True)
                del data, source_train, source_test, inverse
        if not arrays["done"].all():
            raise AssertionError("missing layer pairs")
        summary = {}
        with (out / "pairwise_r2.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("kind", "source_layer", "target_layer", "head", "train_uniform_r2",
                             "heldout_uniform_r2", "train_variance_weighted_r2", "heldout_variance_weighted_r2",
                             "source_rank", "train_valid_channels", "heldout_valid_channels",
                             "train_roundoff_channels", "heldout_roundoff_channels"))
            for ki, kind in enumerate(KINDS):
                summary[kind] = {}
                for split in ("train", "test"):
                    heatmap = arrays[f"{split}_uniform"][ki].mean(axis=-1)
                    if not np.isfinite(heatmap).all():
                        raise ValueError("NaN/Inf in final heatmap")
                    summary[kind][split] = {
                        "mean": float(heatmap.mean()), "minimum": float(heatmap.min()),
                        "maximum": float(heatmap.max()), "negative_cells": int((heatmap < 0).sum()),
                        "mean_best_source_per_target": float(heatmap.max(axis=0).mean()),
                        "best_source_per_target": heatmap.argmax(axis=0).tolist(),
                    }
                for sl in range(sm["num_layers"]):
                    for tl in range(tm["num_layers"]):
                        for head in range(tm["num_kv_heads"]):
                            pos = (ki, sl, tl, head)
                            writer.writerow((kind, sl, tl, head, *(float(arrays[name][pos]) for name in
                                ("train_uniform", "test_uniform", "train_weighted", "test_weighted")),
                                int(arrays["source_rank"][ki, sl, head]),
                                *(int(arrays[name][pos]) for name in ("train_valid_channels", "test_valid_channels",
                                                                    "train_roundoff_channels", "test_roundoff_channels"))))
        summary["rank_minimum"] = int(arrays["source_rank"].min())
        summary["rank_maximum"] = int(arrays["source_rank"].max())
        summary["train_observations"] = n_train
        summary["heldout_observations"] = n_total - n_train
        write_json(out / "summary.json", summary)
        manifest = read_json(manifest_path)
        manifest["complete"] = True
        write_json(manifest_path, manifest)
        return arrays


def main():
    parser = argparse.ArgumentParser(description="All-layer OLS R2; source rows, target columns")
    parser.add_argument("--run-dir", required=True)
    # CPU is an explicit numerical reference, never a fallback for a CUDA request.
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_probe(args.run_dir, args.device)


if __name__ == "__main__":
    main()
