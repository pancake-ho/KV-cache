"""Render saved training MSE or a source-layer R² probe for the neural mapper."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter

from .spec import MapperSpec
from .training_metrics import (
    HeadTrainingMetric,
    load_training_metrics,
    score_source_layers,
    write_training_metrics_csv,
)


PANELS = (("k", "K · RoPE stripped"), ("v", "V"))


def _model_name(model_id: str) -> str:
    path = Path(model_id)
    for part in path.parts:
        if part.startswith("models--"):
            return part.split("--")[-1]
    return path.name


def _draw_panel(ax, values: np.ndarray, label: str, first_layer: int) -> None:
    heads, layers = values.shape
    present = values[np.isfinite(values)]
    positive = present[present > 0]
    colors = values.copy()
    if positive.size:
        low, high = float(positive.min()), float(positive.max())
        if present.size != positive.size:
            low /= 10
        if low == high:
            low, high = low / 1.2, high * 1.2
        colors[values == 0] = low
        norm = LogNorm(low, high)
        ticks = np.geomspace(low, high, 4)
    else:
        norm = Normalize(0, 1)
        ticks = [0]

    cmap = matplotlib.colormaps["YlOrRd"].with_extremes(bad="#e4e8ed")
    image = ax.imshow(np.ma.masked_invalid(colors), cmap=cmap, norm=norm,
                      aspect="auto", origin="upper", interpolation="nearest")
    ax.set_title(label, loc="left", fontweight="bold", fontsize=13, pad=12)
    ticks_x = sorted({*range(0, layers, max(1, math.ceil(layers / 12))), layers - 1})
    ax.set_xticks(ticks_x, [first_layer + tick for tick in ticks_x])
    ax.set_yticks(range(heads))
    ax.set_xlabel("Target layer")
    ax.set_xticks(np.arange(-0.5, layers, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heads, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not present.size:
        ax.text(0.5, 0.5, "No saved metrics", ha="center", va="center", transform=ax.transAxes)
        return

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.05, pad=0.035, ticks=ticks)
    colorbar.outline.set_visible(False)
    colorbar.minorticks_off()
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1e}"))
    colorbar.ax.tick_params(length=0, labelsize=8)
    if layers <= 8:
        for head, layer in np.argwhere(np.isfinite(values)):
            color = "white" if norm(colors[head, layer]) > 0.72 else "#1d2939"
            ax.text(layer, head, f"{values[head, layer]:.2e}", ha="center", va="center",
                    color=color, fontsize=8)


def plot_training_heatmap(spec: MapperSpec, metrics: list[HeadTrainingMetric], output_dir: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first_layer = min(item.target_layer for item in metrics)
    last_layer = max(item.target_layer for item in metrics)
    layers = last_layer - first_layer + 1
    total = spec.target_layers * spec.target_kv_heads * len(PANELS)
    matrices = {kind: np.full((spec.target_kv_heads, layers), np.nan) for kind, _ in PANELS}
    for item in metrics:
        matrices[item.kind][item.head, item.target_layer - first_layer] = item.train_mse

    fig, axes = plt.subplots(1, 2, figsize=(max(9, min(22, 6 + 0.4 * layers)), 5.3))
    fig.subplots_adjust(left=0.085, right=0.915, top=0.76, bottom=0.17, wspace=0.35)
    pair = f"{_model_name(spec.source_model)} → {_model_name(spec.target_model)}"
    top_k = len(spec.selected_layers[0])
    fig.text(0.085, 0.94, "Neural KV Mapper · Final-epoch Training MSE", fontsize=18, fontweight="bold")
    fig.text(0.085, 0.875, f"{pair}   |   Top-{top_k}   |   {len(metrics)}/{total} saved K/V head metrics",
             fontsize=10, color="#475467")
    for ax, (kind, label) in zip(axes, PANELS):
        _draw_panel(ax, matrices[kind], label, first_layer)
    axes[0].set_ylabel("Target KV head")
    fig.text(0.085, 0.045, "Training data · Lower is better · Separate color scales · Gray = missing metric",
             fontsize=9, color="#475467")
    try:
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"mlp_train_mse_heatmap.{extension}", dpi=240, facecolor="white")
    finally:
        plt.close(fig)


def plot_source_layer_r2(spec: MapperSpec, scores: np.ndarray, samples: int, output_dir: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, target = _model_name(spec.source_model), _model_name(spec.target_model)
    cmap = matplotlib.colormaps["YlOrRd"].with_extremes(bad="#e7e7e7", under="#567fa3")
    with plt.rc_context({"font.size": 8, "axes.linewidth": 0.55, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.5), layout="constrained", sharex=True, sharey=True)
        for index, (ax, label) in enumerate(zip(axes, ("K (RoPE stripped)", "V"))):
            image = ax.imshow(np.ma.masked_invalid(scores[index]), cmap=cmap, vmin=0, vmax=1,
                              origin="upper", interpolation="nearest", aspect="equal")
            ax.set_title(f"({chr(97 + index)}) {label}", fontsize=9, fontweight="bold")
            ax.set_xlabel(f"Target layer ({target})")
            ax.set_xticks(range(0, spec.target_layers, 5))
            ax.set_yticks(range(0, spec.source_layers, 5))
            ax.tick_params(length=2.5, width=0.55)
        axes[0].set_ylabel(f"Source layer ({source})")
        has_negative = (scores < 0).any()
        bar = fig.colorbar(image, ax=axes, shrink=0.81, pad=0.018, extend="min" if has_negative else "neither")
        bar.set_label(r"Head-averaged $R^2$")
        top_k = len(spec.selected_layers[0])
        fig.suptitle(f"Fixed top-{top_k} MLP, one source layer varied\nCalibration ($N={samples:,}$)", fontsize=9)
        caption = "Gray: unavailable or unselected. Other inputs fixed at their mean."
        if has_negative:
            caption += " Blue: negative $R^2$."
        fig.supxlabel(caption, fontsize=7)
        try:
            for extension in ("png", "pdf"):
                fig.savefig(output_dir / f"mlp_source_layer_r2.{extension}", dpi=300)
        finally:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapper-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for figures and metrics")
    parser.add_argument("--source-dir", type=Path, help="Training source KV store, for the R² probe")
    parser.add_argument("--target-dir", type=Path, help="Training target KV store, for the R² probe")
    parser.add_argument("--samples", type=int, default=2048, help="Calibration observations for the R² probe")
    parser.add_argument("--seed", type=int, default=2026, help="Calibration sampling seed for the R² probe")
    args = parser.parse_args()
    if (args.source_dir is None) != (args.target_dir is None):
        parser.error("provide both --source-dir and --target-dir for the R² probe")
    try:
        if args.output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {args.output_dir}")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".mapper-heatmap-", dir=args.output_dir.parent) as temporary:
            staging = Path(temporary)
            if args.source_dir is not None:
                spec, scores, rows = score_source_layers(args.mapper_dir, args.source_dir, args.target_dir,
                                                        args.samples, args.seed)
                plot_source_layer_r2(spec, scores, len(rows), staging)
                np.savez_compressed(staging / "source_layer_r2.npz", scores=scores,
                                    source_model=spec.source_model, target_model=spec.target_model,
                                    samples=len(rows), seed=args.seed, sample_rows=rows,
                                    top_k=len(spec.selected_layers[0]),
                                    method="fixed_mapper_source_only_mean_fill_calibration")
                summary = f"Plotted {np.isfinite(scores).sum()} source–target K/V scores"
            else:
                spec, metrics = load_training_metrics(args.mapper_dir)
                plot_training_heatmap(spec, metrics, staging)
                write_training_metrics_csv(staging / "mlp_train_mse_cells.csv", metrics)
                total = spec.target_layers * spec.target_kv_heads * len(PANELS)
                summary = f"Plotted {len(metrics)}/{total} saved K/V metrics"
            staging.rename(args.output_dir)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"{summary} in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
