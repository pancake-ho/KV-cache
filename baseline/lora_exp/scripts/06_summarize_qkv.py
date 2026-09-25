#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def read_yaml(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def summarize(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    metric: str,
) -> pd.DataFrame:
    grouped = (
        df.groupby(
            group_cols,
            dropna=False,
        )[metric]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
        .reset_index()
    )

    grouped["sem"] = (
        grouped["std"]
        / grouped[
            "count"
        ].pow(0.5)
    )

    grouped["ci95_low"] = (
        grouped["mean"]
        - 1.96
        * grouped["sem"]
    )

    grouped["ci95_high"] = (
        grouped["mean"]
        + 1.96
        * grouped["sem"]
    )

    return grouped


def summarize_paired_difference(
    df: pd.DataFrame,
    *,
    variant_a: str,
    variant_b: str,
    output_path: Path,
):
    # Cross-user LoRA pairs only.
    work = df[
        (df["pair_a"] != "base")
        & (df["pair_b"] != "base")
        & (
            df["variant"].isin(
                [
                    variant_a,
                    variant_b,
                ]
            )
        )
    ].copy()

    pivot = work.pivot_table(
        index=[
            "base_record_id",
            "dataset",
            "pair_a",
            "pair_b",
            "tensor_type",
            "layer",
        ],
        columns="variant",
        values="cosine_mean",
        aggfunc="first",
    ).reset_index()

    if (
        variant_a not in pivot.columns
        or variant_b not in pivot.columns
    ):
        print(
            f"[SKIP] paired effect "
            f"{variant_a} -> {variant_b}"
        )
        return

    pivot = pivot.dropna(
        subset=[
            variant_a,
            variant_b,
        ]
    )

    pivot["difference"] = (
        pivot[variant_b]
        - pivot[variant_a]
    )

    summary = summarize(
        pivot,
        [
            "dataset",
            "pair_a",
            "pair_b",
            "tensor_type",
            "layer",
        ],
        metric="difference",
    )

    summary.to_csv(
        output_path,
        index=False,
    )


def plot_layer_curves(
    df: pd.DataFrame,
    *,
    tensor_type: str,
    output_path: Path,
):
    work = df[
        (df["tensor_type"] == tensor_type)
        & (df["pair_a"] != "base")
        & (df["pair_b"] != "base")
    ].copy()

    grouped = (
        work.groupby(
            [
                "pair_a",
                "pair_b",
                "layer",
            ]
        )["cosine_mean"]
        .mean()
        .reset_index()
    )

    fig = plt.figure(
        figsize=(9, 5)
    )

    ax = fig.add_subplot(111)

    for (
        pair_a,
        pair_b,
    ), group in grouped.groupby(
        [
            "pair_a",
            "pair_b",
        ]
    ):
        group = group.sort_values(
            "layer"
        )

        ax.plot(
            group["layer"],
            group["cosine_mean"],
            label=(
                f"{pair_a} vs {pair_b}"
            ),
        )

    ax.set_xlabel(
        "Transformer layer"
    )

    ax.set_ylabel(
        "Mean cosine similarity"
    )

    ax.set_title(
        f"Cross-LoRA {tensor_type} similarity"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def plot_delta_curves(
    df: pd.DataFrame,
    *,
    tensor_type: str,
    output_path: Path,
):
    work = df[
        (df["tensor_type"] == tensor_type)
        & (df["pair_a"] != "base")
        & (df["pair_b"] != "base")
        & df[
            "delta_global_cosine"
        ].notna()
    ].copy()

    grouped = (
        work.groupby(
            [
                "pair_a",
                "pair_b",
                "layer",
            ]
        )[
            "delta_global_cosine"
        ]
        .mean()
        .reset_index()
    )

    fig = plt.figure(
        figsize=(9, 5)
    )

    ax = fig.add_subplot(111)

    for (
        pair_a,
        pair_b,
    ), group in grouped.groupby(
        [
            "pair_a",
            "pair_b",
        ]
    ):
        group = group.sort_values(
            "layer"
        )

        ax.plot(
            group["layer"],
            group[
                "delta_global_cosine"
            ],
            label=(
                f"{pair_a} vs {pair_b}"
            ),
        )

    ax.set_xlabel(
        "Transformer layer"
    )

    ax.set_ylabel(
        "Cosine similarity of LoRA-induced delta"
    )

    ax.set_title(
        f"Base-relative Δ{tensor_type}"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/characterize.yaml"
        ),
    )

    args = parser.parse_args()

    cfg = read_yaml(
        PROJECT_ROOT
        / args.config
    )["phase3"]

    output_root = Path(
        os.path.expandvars(
            cfg["output_root"]
        )
    ).resolve()

    detail_path = (
        output_root
        / "qkv_detail.csv"
    )

    if not detail_path.exists():
        raise FileNotFoundError(
            detail_path
        )

    df = pd.read_csv(
        detail_path
    )

    result_dir = (
        output_root / "summary"
    )

    figure_dir = (
        output_root / "figures"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Overall cross-prompt statistics
    # --------------------------------------------------------

    overall = summarize(
        df,
        [
            "pair_a",
            "pair_b",
            "tensor_type",
            "layer",
        ],
        metric="cosine_mean",
    )

    overall.to_csv(
        result_dir
        / "overall_cosine.csv",
        index=False,
    )

    delta_df = df[
        df[
            "delta_global_cosine"
        ].notna()
    ]

    delta_summary = summarize(
        delta_df,
        [
            "pair_a",
            "pair_b",
            "tensor_type",
            "layer",
        ],
        metric="delta_global_cosine",
    )

    delta_summary.to_csv(
        result_dir
        / "overall_delta_cosine.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Dataset / form statistics
    # --------------------------------------------------------

    by_dataset_variant = summarize(
        df,
        [
            "dataset",
            "variant",
            "pair_a",
            "pair_b",
            "tensor_type",
            "layer",
        ],
        metric="cosine_mean",
    )

    by_dataset_variant.to_csv(
        result_dir
        / "by_dataset_variant.csv",
        index=False,
    )

    # --------------------------------------------------------
    # MedMCQA medical sub-topic statistics
    # --------------------------------------------------------

    med = df[
        (df["dataset"] == "medmcqa")
        & (df["variant"] == "normalized")
        & (df["subject"].fillna("") != "")
    ].copy()

    med_subject = summarize(
        med,
        [
            "subject",
            "pair_a",
            "pair_b",
            "tensor_type",
            "layer",
        ],
        metric="cosine_mean",
    )

    med_subject = med_subject[
        med_subject["count"]
        >= int(
            cfg["min_group_size"]
        )
    ]

    med_subject.to_csv(
        result_dir
        / "medmcqa_subject.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Paired form effect:
    # same semantic sample, native -> normalized
    # --------------------------------------------------------

    summarize_paired_difference(
        df,
        variant_a="native",
        variant_b="normalized",
        output_path=(
            result_dir
            / "paired_form_effect.csv"
        ),
    )

    # --------------------------------------------------------
    # Paired context effect:
    # PubMedQA normalized with context
    # vs exact same question without context
    # --------------------------------------------------------

    summarize_paired_difference(
        df[
            df["dataset"]
            == "pubmedqa"
        ],
        variant_a="normalized",
        variant_b=(
            "normalized_no_context"
        ),
        output_path=(
            result_dir
            / "paired_context_effect.csv"
        ),
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    for tensor_type in [
        "q_attn",
        "k_cache",
        "v_cache",
    ]:
        plot_layer_curves(
            df,
            tensor_type=(
                tensor_type
            ),
            output_path=(
                figure_dir
                / (
                    f"{tensor_type}"
                    "_absolute_similarity.png"
                )
            ),
        )

        plot_delta_curves(
            df,
            tensor_type=(
                tensor_type
            ),
            output_path=(
                figure_dir
                / (
                    f"{tensor_type}"
                    "_delta_similarity.png"
                )
            ),
        )

    print(
        "===================================="
    )
    print(
        "PHASE 3 SUMMARY PASS"
    )
    print(
        f"summary={result_dir}"
    )
    print(
        f"figures={figure_dir}"
    )
    print(
        "===================================="
    )


if __name__ == "__main__":
    main()