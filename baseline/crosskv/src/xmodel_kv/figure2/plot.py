from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .common import KINDS, read_json


def plot(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(run_dir)
    results = root / "results"
    manifest = read_json(results / "probe.json")
    if not manifest.get("complete"):
        raise ValueError("probe is incomplete")
    prepared = read_json(root / "prepare.json")
    cfg = prepared["config"]
    source_name = cfg['source_model'].split('/')[-1]
    target_name = cfg['target_model'].split('/')[-1]
    with np.load(results / "pairwise_r2.npz") as stored:
        if not stored["done"].all():
            raise ValueError("missing layer-pair scores")
        scores = {split: stored[f"{split}_uniform"].mean(axis=-1) for split in ("train", "test")}
    labels = ("K (with RoPE)", "K (RoPE stripped)", "V")
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_under("#567fa3")
    for split, title, filename in (("train", "Calibration (in-sample)", "figure2_calibration"),
                                   ("test", "Held-out documents", "figure2_heldout")):
        data = scores[split]
        if not np.isfinite(data).all():
            raise ValueError("nonfinite plot data")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained", sharex=True, sharey=True)
        for kind_index, axis in enumerate(axes):
            artist = axis.imshow(data[kind_index], origin="upper", interpolation="nearest",
                                 aspect="auto", vmin=0, vmax=1, cmap=cmap)
            axis.set_title(labels[kind_index], fontweight="bold")
            axis.set_xlabel(f"Target layer ({target_name})")
            axis.set_xticks(range(0, data.shape[2], 5 if data.shape[2] >= 5 else 1))
            axis.set_yticks(range(0, data.shape[1], 5 if data.shape[1] >= 5 else 1))
        axes[0].set_ylabel(f"Source layer ({source_name})")
        negative = int((data < 0).sum())
        colorbar = fig.colorbar(artist, ax=axes, shrink=0.84, extend="min" if negative else "neither")
        colorbar.set_label("Head-averaged $R^2$")
        n = prepared["train_observations"] if split == "train" else prepared["heldout_observations"]
        status = "SMOKE TEST - " if cfg["smoke"] else ""
        fig.suptitle(f"{status}{source_name} → {target_name} | {title}\n"
                     f"Single-source matched-head OLS · N={n:,} · {cfg['sequence_length']} tokens · stride {cfg['stride']}"
                     + (f" · blue: {negative} negative cells" if negative else ""), fontsize=13)
        fig.savefig(results / f"{filename}.png", dpi=300)
        fig.savefig(results / f"{filename}.pdf")
        plt.close(fig)
        print(results / f"{filename}.png", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Figure 2 style: three panels and one common colorbar")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    plot(args.run_dir)


if __name__ == "__main__":
    main()
