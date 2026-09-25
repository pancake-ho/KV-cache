#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TENSORS = ["q_attn", "k_cache", "v_cache"]
TENSOR_LABELS = {
    "q_attn": "Q",
    "k_cache": "K cache",
    "v_cache": "V cache",
}
RELATION_LABELS = {
    "within_domain": "Same-domain adapters",
    "cross_domain": "Different-domain adapters",
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_phase3c_tables(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects_path = out_root / "summary" / "domain_effects.csv"
    layer_path = out_root / "summary" / "layer_stats.csv"

    if not effects_path.exists():
        raise FileNotFoundError(
            f"Missing existing summary file: {effects_path}\n"
            "This script is plot-only. Do not rerun the experiment; "
            "use the Phase-3C summary outputs that already exist."
        )
    if not layer_path.exists():
        raise FileNotFoundError(
            f"Missing existing summary file: {layer_path}"
        )

    effects = pd.read_csv(effects_path, low_memory=False)
    layer_stats = pd.read_csv(layer_path, low_memory=False)

    required_effect_cols = {
        "metric",
        "scope",
        "tensor_type",
        "within_mean",
        "cross_mean",
        "effect_within_minus_cross",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value_bh",
    }
    required_layer_cols = {
        "metric",
        "tensor_type",
        "layer",
        "pair_relation",
        "mean",
    }

    missing_effect = required_effect_cols - set(effects.columns)
    missing_layer = required_layer_cols - set(layer_stats.columns)

    if missing_effect:
        raise RuntimeError(
            f"domain_effects.csv missing columns: {sorted(missing_effect)}"
        )
    if missing_layer:
        raise RuntimeError(
            f"layer_stats.csv missing columns: {sorted(missing_layer)}"
        )

    return effects, layer_stats


def get_effect_row(
    effects: pd.DataFrame,
    metric: str,
    tensor: str,
) -> pd.Series:
    rows = effects[
        (effects["scope"] == "all")
        & (effects["metric"] == metric)
        & (effects["tensor_type"] == tensor)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one row for "
            f"metric={metric}, tensor={tensor}, scope=all; got {len(rows)}."
        )
    return rows.iloc[0]


def plot_layer_curves(
    ax,
    layer_stats: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    subset = layer_stats[layer_stats["metric"] == metric]

    line_styles = {
        "within_domain": "-",
        "cross_domain": "--",
    }

    for tensor in TENSORS:
        for relation in ["within_domain", "cross_domain"]:
            group = subset[
                (subset["tensor_type"] == tensor)
                & (subset["pair_relation"] == relation)
            ].sort_values("layer")

            ax.plot(
                group["layer"],
                group["mean"],
                linestyle=line_styles[relation],
                linewidth=1.9,
                label=f"{TENSOR_LABELS[tensor]} / {RELATION_LABELS[relation]}",
            )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(
        frameon=False,
        fontsize=8.5,
        ncol=2,
    )


def annotate_grouped_bars(
    ax,
    bars,
    values,
    *,
    decimals: int,
    y_offset: float,
) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + y_offset,
            f"{float(value):.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )


def plot_delta_summary(
    ax,
    effects: pd.DataFrame,
) -> None:
    x = np.arange(len(TENSORS), dtype=float)
    width = 0.34

    same_domain = []
    different_domain = []

    for tensor in TENSORS:
        row = get_effect_row(
            effects,
            "delta_global_cosine",
            tensor,
        )
        same_domain.append(float(row["within_mean"]))
        different_domain.append(float(row["cross_mean"]))

    bars_same = ax.bar(
        x - width / 2,
        same_domain,
        width=width,
        label="Same-domain adapters",
    )
    bars_diff = ax.bar(
        x + width / 2,
        different_domain,
        width=width,
        label="Different-domain adapters",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([TENSOR_LABELS[t] for t in TENSORS])
    ax.set_ylim(0.50, 0.86)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(
        "C. Average LoRA-induced Q/K/V similarity",
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    annotate_grouped_bars(
        ax,
        bars_same,
        same_domain,
        decimals=3,
        y_offset=0.008,
    )
    annotate_grouped_bars(
        ax,
        bars_diff,
        different_domain,
        decimals=3,
        y_offset=0.008,
    )

    ax.text(
        0.5,
        0.03,
        "Higher = more similar LoRA-induced change",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )


def plot_domain_advantage(
    ax,
    effects: pd.DataFrame,
) -> None:
    x = np.arange(len(TENSORS), dtype=float)

    means = []
    lower_err = []
    upper_err = []

    for tensor in TENSORS:
        row = get_effect_row(
            effects,
            "delta_global_cosine",
            tensor,
        )

        mean = float(row["effect_within_minus_cross"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])

        means.append(mean)
        lower_err.append(mean - low)
        upper_err.append(high - mean)

    bars = ax.bar(
        x,
        means,
        width=0.58,
        yerr=np.vstack([lower_err, upper_err]),
        capsize=5,
    )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([TENSOR_LABELS[t] for t in TENSORS])
    ax.set_ylim(0.0, 0.22)
    ax.set_ylabel(
        "Same-domain advantage\n"
        "(cosine similarity difference)"
    )
    ax.set_title(
        "D. How much more similar are same-domain adapters?",
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.007,
            f"+{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.text(
        0.5,
        0.03,
        "Positive value → same-domain pair is more similar\n"
        "Error bars: 95% bootstrap CI",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def build_figure(
    effects: pd.DataFrame,
    layer_stats: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 10.0),
    )

    # A. Keep the original absolute-similarity layer style.
    plot_layer_curves(
        axes[0, 0],
        layer_stats,
        "cosine_mean",
        title="A. Absolute Q/K/V similarity with the same input",
        ylabel="Cosine similarity",
        ylim=(0.9925, 1.0002),
    )

    # B. Keep the original base-relative layer style.
    plot_layer_curves(
        axes[0, 1],
        layer_stats,
        "delta_global_cosine",
        title="B. Similarity of LoRA-induced change after removing the base",
        ylabel="Cosine similarity of ΔQ / ΔK / ΔV",
        ylim=(0.0, 1.01),
    )

    # C/D. Replace the difficult old panels with direct summaries.
    plot_delta_summary(
        axes[1, 0],
        effects,
    )
    plot_domain_advantage(
        axes[1, 1],
        effects,
    )

    fig.suptitle(
        "Cross-domain LoRA Q/K/V Commonality — Identical Input",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.953,
        "Same-domain adapters: Medical↔Medical or Finance↔Finance    |    "
        "Different-domain adapters: Medical↔Finance",
        ha="center",
        fontsize=10.5,
    )

    fig.text(
        0.5,
        0.015,
        "Note: Medical datasets use QA tasks and Finance datasets use sentiment tasks; "
        "the observed separation reflects domain/task training heterogeneity.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0.02, 0.045, 0.98, 0.925),
        h_pad=2.4,
        w_pad=2.0,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def print_key_numbers(effects: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("PROFESSOR-FACING KEY RESULT")
    print("=" * 88)

    for tensor in TENSORS:
        row = get_effect_row(
            effects,
            "delta_global_cosine",
            tensor,
        )
        print(
            f"{TENSOR_LABELS[tensor]:8s} | "
            f"same-domain={float(row['within_mean']):.3f} | "
            f"different-domain={float(row['cross_mean']):.3f} | "
            f"advantage={float(row['effect_within_minus_cross']):+.3f} | "
            f"95% CI=[{float(row['ci_low']):+.3f}, "
            f"{float(row['ci_high']):+.3f}]"
        )

    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replot existing Phase-3C summary results only. "
            "No training, no model loading, and no GPU inference."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/phase3c_cross_domain.yaml",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output PNG. Default: "
            "<characterization_output_root>/professor_figures/"
            "phase3c_professor_simple.png"
        ),
    )
    args = parser.parse_args()

    cfg = read_yaml(
        PROJECT_ROOT / args.config
    )["phase3c"]

    out_root = Path(
        os.path.expandvars(
            cfg["characterization_output_root"]
        )
    ).resolve()

    effects, layer_stats = load_phase3c_tables(
        out_root
    )

    if args.output is None:
        output_path = (
            out_root
            / "professor_figures"
            / "phase3c_professor_simple.png"
        )
    else:
        output_path = Path(
            os.path.expandvars(args.output)
        ).resolve()

    build_figure(
        effects,
        layer_stats,
        output_path,
    )

    print_key_numbers(effects)

    print("\n[PLOT-ONLY PASS]")
    print("No experiment was re-run.")
    print(f"figure={output_path}")


if __name__ == "__main__":
    main()
