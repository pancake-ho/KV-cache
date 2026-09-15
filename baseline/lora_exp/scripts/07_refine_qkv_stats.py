#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

TENSORS = [
    "q_attn",
    "k_cache",
    "v_cache",
]

PRIMARY_METRICS = [
    "cosine_mean",
    "delta_global_cosine",
    "symmetric_rel_l2",
    "delta_norm_mean",
    "delta_distance_vs_base",
]


# ============================================================
# I/O
# ============================================================

def read_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def save_json(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def load_detail(
    path: Path,
) -> pd.DataFrame:
    """
    Load qkv_detail.csv with explicit metadata dtypes.

    This also removes the DtypeWarning previously produced for
    subject/topic columns.
    """

    df = pd.read_csv(
        path,
        dtype={
            "dataset": "string",
            "subject": "string",
            "topic": "string",
            "variant": "string",
            "context_mode": "string",
            "pair_a": "string",
            "pair_b": "string",
            "tensor_type": "string",
        },
        low_memory=False,
    )

    required = {
        "probe_id",
        "base_record_id",
        "dataset",
        "subject",
        "topic",
        "variant",
        "pair_a",
        "pair_b",
        "tensor_type",
        "layer",
        "cosine_mean",
        "symmetric_rel_l2",
        "delta_global_cosine",
        "delta_norm_ratio_a",
        "delta_norm_ratio_b",
        "delta_distance_vs_base",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "qkv_detail.csv is missing columns: "
            f"{missing}"
        )

    # A/B pair의 LoRA-induced magnitude를
    # 하나의 대칭적 representative 값으로 만든다.
    df["delta_norm_mean"] = (
        df["delta_norm_ratio_a"].astype(float)
        + df["delta_norm_ratio_b"].astype(float)
    ) / 2.0

    return df


# ============================================================
# Integrity
# ============================================================

def audit_detail(
    df: pd.DataFrame,
) -> dict:
    key = [
        "probe_id",
        "pair_a",
        "pair_b",
        "tensor_type",
        "layer",
    ]

    duplicate_keys = int(
        df.duplicated(key).sum()
    )

    if duplicate_keys != 0:
        raise RuntimeError(
            "Duplicate characterization keys found: "
            f"{duplicate_keys}"
        )

    tensor_types = sorted(
        df["tensor_type"]
        .dropna()
        .unique()
        .tolist()
    )

    if set(tensor_types) != set(TENSORS):
        raise RuntimeError(
            f"Unexpected tensor types: "
            f"{tensor_types}"
        )

    layers = sorted(
        int(x)
        for x in df["layer"]
        .dropna()
        .unique()
    )

    if layers != list(range(36)):
        raise RuntimeError(
            "Expected layers 0..35, "
            f"got {layers}"
        )

    pairs = (
        df[
            [
                "pair_a",
                "pair_b",
            ]
        ]
        .drop_duplicates()
    )

    if len(pairs) != 6:
        raise RuntimeError(
            f"Expected 6 model pairs, "
            f"got {len(pairs)}"
        )

    absolute_cols = [
        "cosine_mean",
        "symmetric_rel_l2",
    ]

    if not np.isfinite(
        df[
            absolute_cols
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Non-finite absolute metrics found."
        )

    cross = df[
        (df["pair_a"] != "base")
        & (df["pair_b"] != "base")
    ]

    delta_cols = [
        "delta_global_cosine",
        "delta_norm_ratio_a",
        "delta_norm_ratio_b",
        "delta_distance_vs_base",
    ]

    if not np.isfinite(
        cross[
            delta_cols
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Non-finite delta metrics found."
        )

    return {
        "rows": int(
            len(df)
        ),
        "unique_probes": int(
            df["probe_id"].nunique()
        ),
        "duplicate_keys": (
            duplicate_keys
        ),
        "tensor_types": (
            tensor_types
        ),
        "num_layers": int(
            len(layers)
        ),
        "num_pairs": int(
            len(pairs)
        ),
    }


# ============================================================
# Bootstrap utilities
# ============================================================

def bootstrap_equal_dataset_mean(
    record_df: pd.DataFrame,
    *,
    value_col: str,
    replicates: int,
    rng: np.random.Generator,
    ci_level: float,
) -> dict:
    """
    One row = one source base_record_id.

    Each dataset is bootstrapped independently and receives
    equal final weight:

        (mean_med + mean_pub + mean_flash) / 3

    Therefore PubMedQA's extra prompt variant count cannot
    dominate the primary statistic.
    """

    clean = (
        record_df[
            [
                "dataset",
                "base_record_id",
                value_col,
            ]
        ]
        .dropna()
        .copy()
    )

    if clean.empty:
        raise RuntimeError(
            f"No data for {value_col}"
        )

    if clean.duplicated(
        [
            "dataset",
            "base_record_id",
        ]
    ).any():
        raise RuntimeError(
            "Bootstrap input must contain "
            "one value per source record."
        )

    observed_dataset_means = []

    bootstrap_dataset_means = []

    for _, group in clean.groupby(
        "dataset",
        observed=True,
    ):
        values = (
            group[value_col]
            .to_numpy(
                dtype=float
            )
        )

        observed_dataset_means.append(
            float(
                values.mean()
            )
        )

        indices = rng.integers(
            low=0,
            high=len(values),
            size=(
                replicates,
                len(values),
            ),
        )

        bootstrap_dataset_means.append(
            values[
                indices
            ].mean(
                axis=1
            )
        )

    bootstrap_values = (
        np.vstack(
            bootstrap_dataset_means
        )
        .mean(
            axis=0
        )
    )

    alpha = (
        1.0 - ci_level
    ) / 2.0

    ci_low, ci_high = (
        np.quantile(
            bootstrap_values,
            [
                alpha,
                1.0 - alpha,
            ],
        )
    )

    return {
        "mean": float(
            np.mean(
                observed_dataset_means
            )
        ),
        "ci_low": float(
            ci_low
        ),
        "ci_high": float(
            ci_high
        ),
        "n_records": int(
            clean[
                "base_record_id"
            ].nunique()
        ),
        "n_datasets": int(
            clean[
                "dataset"
            ].nunique()
        ),
    }


def bootstrap_simple_mean(
    record_df: pd.DataFrame,
    *,
    value_col: str,
    replicates: int,
    rng: np.random.Generator,
    ci_level: float,
) -> dict:
    """
    Bootstrap one dataset's paired-record effects.
    """

    clean = (
        record_df[
            [
                "base_record_id",
                value_col,
            ]
        ]
        .dropna()
        .copy()
    )

    if clean.duplicated(
        "base_record_id"
    ).any():
        raise RuntimeError(
            "Expected one row per base_record_id."
        )

    values = clean[
        value_col
    ].to_numpy(
        dtype=float
    )

    if len(values) == 0:
        raise RuntimeError(
            f"No values for {value_col}"
        )

    indices = rng.integers(
        low=0,
        high=len(values),
        size=(
            replicates,
            len(values),
        ),
    )

    bootstrap_values = (
        values[
            indices
        ].mean(
            axis=1
        )
    )

    alpha = (
        1.0 - ci_level
    ) / 2.0

    ci_low, ci_high = (
        np.quantile(
            bootstrap_values,
            [
                alpha,
                1.0 - alpha,
            ],
        )
    )

    return {
        "mean": float(
            values.mean()
        ),
        "ci_low": float(
            ci_low
        ),
        "ci_high": float(
            ci_high
        ),
        "n_records": int(
            len(values)
        ),
        "n_datasets": 1,
    }


# ============================================================
# Paired randomization tests
# ============================================================

def sign_flip_equal_dataset(
    record_df: pd.DataFrame,
    *,
    value_col: str,
    replicates: int,
    rng: np.random.Generator,
) -> float:
    """
    Two-sided paired sign-flip test while preserving equal
    dataset weighting.
    """

    clean = (
        record_df[
            [
                "dataset",
                "base_record_id",
                value_col,
            ]
        ]
        .dropna()
        .copy()
    )

    observed = abs(
        clean.groupby(
            "dataset",
            observed=True,
        )[value_col]
        .mean()
        .mean()
    )

    null_dataset_means = []

    for _, group in clean.groupby(
        "dataset",
        observed=True,
    ):
        values = (
            group[value_col]
            .to_numpy(
                dtype=float
            )
        )

        signs = rng.choice(
            np.array(
                [
                    -1.0,
                    1.0,
                ]
            ),
            size=(
                replicates,
                len(values),
            ),
        )

        null_dataset_means.append(
            (
                signs
                * values
            ).mean(
                axis=1
            )
        )

    null_distribution = np.abs(
        np.vstack(
            null_dataset_means
        ).mean(
            axis=0
        )
    )

    return float(
        (
            np.count_nonzero(
                null_distribution
                >= observed
            )
            + 1
        )
        / (
            replicates
            + 1
        )
    )


def sign_flip_simple(
    record_df: pd.DataFrame,
    *,
    value_col: str,
    replicates: int,
    rng: np.random.Generator,
) -> float:
    values = (
        record_df[
            value_col
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    observed = abs(
        float(
            values.mean()
        )
    )

    signs = rng.choice(
        np.array(
            [
                -1.0,
                1.0,
            ]
        ),
        size=(
            replicates,
            len(values),
        ),
    )

    null_distribution = np.abs(
        (
            signs
            * values
        ).mean(
            axis=1
        )
    )

    return float(
        (
            np.count_nonzero(
                null_distribution
                >= observed
            )
            + 1
        )
        / (
            replicates
            + 1
        )
    )


def benjamini_hochberg(
    p_values,
) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    """

    p_values = np.asarray(
        list(
            p_values
        ),
        dtype=float,
    )

    n = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    ranked = (
        p_values[
            order
        ]
    )

    adjusted = (
        ranked
        * n
        / np.arange(
            1,
            n + 1,
        )
    )

    adjusted = (
        np.minimum.accumulate(
            adjusted[
                ::-1
            ]
        )[
            ::-1
        ]
    )

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    output = np.empty_like(
        adjusted
    )

    output[
        order
    ] = adjusted

    return output


# ============================================================
# Primary normalized-only analysis
# ============================================================

def build_primary_record_table(
    df: pd.DataFrame,
    *,
    primary_variant: str,
) -> pd.DataFrame:
    """
    Primary scientific comparison:

      - cross-LoRA pairs only
      - normalized prompts only
      - three LoRA pairs receive equal weight
      - pair average is performed inside each source record
    """

    primary = df[
        (df["pair_a"] != "base")
        & (df["pair_b"] != "base")
        & (
            df["variant"]
            == primary_variant
        )
    ].copy()

    if primary.empty:
        raise RuntimeError(
            "Primary normalized analysis "
            "contains no rows."
        )

    return (
        primary.groupby(
            [
                "dataset",
                "base_record_id",
                "tensor_type",
                "layer",
            ],
            observed=True,
        )[
            PRIMARY_METRICS
        ]
        .mean()
        .reset_index()
    )


def build_primary_layer_stats(
    record_df: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    rows = []

    counter = 0

    for tensor_type in TENSORS:
        tensor_df = record_df[
            record_df[
                "tensor_type"
            ]
            == tensor_type
        ]

        for layer in range(36):
            layer_df = tensor_df[
                tensor_df[
                    "layer"
                ]
                == layer
            ]

            for metric in PRIMARY_METRICS:
                stats = (
                    bootstrap_equal_dataset_mean(
                        layer_df,
                        value_col=metric,
                        replicates=(
                            bootstrap_replicates
                        ),
                        rng=(
                            np.random.default_rng(
                                seed
                                + counter
                            )
                        ),
                        ci_level=(
                            ci_level
                        ),
                    )
                )

                rows.append(
                    {
                        "tensor_type": (
                            tensor_type
                        ),
                        "layer": layer,
                        "metric": metric,
                        **stats,
                    }
                )

                counter += 1

    return pd.DataFrame(
        rows
    )


def build_primary_overall_stats(
    record_df: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    """
    Layers are first averaged inside each source record.
    Therefore 36 transformer layers are NOT treated as
    independent statistical samples.
    """

    rows = []

    counter = 10000

    for tensor_type in TENSORS:
        tensor_df = (
            record_df[
                record_df[
                    "tensor_type"
                ]
                == tensor_type
            ]
            .groupby(
                [
                    "dataset",
                    "base_record_id",
                ],
                observed=True,
            )[
                PRIMARY_METRICS
            ]
            .mean()
            .reset_index()
        )

        for metric in PRIMARY_METRICS:
            stats = (
                bootstrap_equal_dataset_mean(
                    tensor_df,
                    value_col=metric,
                    replicates=(
                        bootstrap_replicates
                    ),
                    rng=(
                        np.random.default_rng(
                            seed
                            + counter
                        )
                    ),
                    ci_level=(
                        ci_level
                    ),
                )
            )

            rows.append(
                {
                    "tensor_type": (
                        tensor_type
                    ),
                    "metric": metric,
                    **stats,
                }
            )

            counter += 1

    return pd.DataFrame(
        rows
    )


# ============================================================
# Pairwise normalized-only table
# ============================================================

def build_pairwise_overall(
    df: pd.DataFrame,
    *,
    primary_variant: str,
    bootstrap_replicates: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    work = df[
        (df["pair_a"] != "base")
        & (df["pair_b"] != "base")
        & (
            df["variant"]
            == primary_variant
        )
    ].copy()

    # Average layers inside each record.
    record_df = (
        work.groupby(
            [
                "dataset",
                "base_record_id",
                "pair_a",
                "pair_b",
                "tensor_type",
            ],
            observed=True,
        )[
            [
                "cosine_mean",
                "delta_global_cosine",
            ]
        ]
        .mean()
        .reset_index()
    )

    rows = []

    counter = 20000

    for (
        pair_a,
        pair_b,
        tensor_type,
    ), group in record_df.groupby(
        [
            "pair_a",
            "pair_b",
            "tensor_type",
        ],
        observed=True,
    ):
        for metric in [
            "cosine_mean",
            "delta_global_cosine",
        ]:
            stats = (
                bootstrap_equal_dataset_mean(
                    group,
                    value_col=metric,
                    replicates=(
                        bootstrap_replicates
                    ),
                    rng=(
                        np.random.default_rng(
                            seed
                            + counter
                        )
                    ),
                    ci_level=(
                        ci_level
                    ),
                )
            )

            rows.append(
                {
                    "pair_a": pair_a,
                    "pair_b": pair_b,
                    "tensor_type": (
                        tensor_type
                    ),
                    "metric": metric,
                    **stats,
                }
            )

            counter += 1

    return pd.DataFrame(
        rows
    )


# ============================================================
# Controlled form/context effects
# ============================================================

def matched_dataset_pair_mask(
    df: pd.DataFrame,
) -> pd.Series:
    """
    True if the pair includes the adapter trained on the
    dataset currently supplying the prompt.
    """

    return (
        (
            df["dataset"]
            == df["pair_a"]
        )
        | (
            df["dataset"]
            == df["pair_b"]
        )
    )


def build_form_effect_records(
    df: pd.DataFrame,
    *,
    scope: str,
) -> pd.DataFrame:
    """
    Per source record:

        normalized cosine - native cosine

    Positive:
        normalizing the prompt makes different LoRAs
        more similar.
    """

    work = df[
        (df["pair_a"] != "base")
        & (df["pair_b"] != "base")
        & (
            df["variant"].isin(
                [
                    "native",
                    "normalized",
                ]
            )
        )
    ].copy()

    pivot = (
        work.pivot_table(
            index=[
                "dataset",
                "base_record_id",
                "pair_a",
                "pair_b",
                "tensor_type",
                "layer",
            ],
            columns="variant",
            values="cosine_mean",
            aggfunc="first",
        )
        .reset_index()
    )

    pivot = pivot.dropna(
        subset=[
            "native",
            "normalized",
        ]
    )

    pivot[
        "effect_value"
    ] = (
        pivot["normalized"]
        - pivot["native"]
    )

    if scope == "matched":
        pivot = pivot[
            matched_dataset_pair_mask(
                pivot
            )
        ]

    elif scope != "all":
        raise ValueError(
            f"Unsupported form scope: "
            f"{scope}"
        )

    # Average matched pairs and all layers
    # inside one semantic source record.
    return (
        pivot.groupby(
            [
                "dataset",
                "base_record_id",
                "tensor_type",
            ],
            observed=True,
        )[
            "effect_value"
        ]
        .mean()
        .reset_index()
    )


def build_context_effect_records(
    df: pd.DataFrame,
    *,
    scope: str,
) -> pd.DataFrame:
    """
    PubMedQA paired comparison:

        normalized_no_context
        -
        normalized_with_context

    Positive:
        removing context makes LoRAs more similar.
    """

    work = df[
        (
            df["dataset"]
            == "pubmedqa"
        )
        & (
            df["pair_a"]
            != "base"
        )
        & (
            df["pair_b"]
            != "base"
        )
        & (
            df["variant"].isin(
                [
                    "normalized",
                    "normalized_no_context",
                ]
            )
        )
    ].copy()

    pivot = (
        work.pivot_table(
            index=[
                "base_record_id",
                "pair_a",
                "pair_b",
                "tensor_type",
                "layer",
            ],
            columns="variant",
            values="cosine_mean",
            aggfunc="first",
        )
        .reset_index()
    )

    pivot = pivot.dropna(
        subset=[
            "normalized",
            "normalized_no_context",
        ]
    )

    pivot[
        "effect_value"
    ] = (
        pivot[
            "normalized_no_context"
        ]
        - pivot["normalized"]
    )

    pair_contains_pubmed = (
        (
            pivot["pair_a"]
            == "pubmedqa"
        )
        | (
            pivot["pair_b"]
            == "pubmedqa"
        )
    )

    if scope == "matched":
        pivot = pivot[
            pair_contains_pubmed
        ]

    elif scope == "control":
        pivot = pivot[
            ~pair_contains_pubmed
        ]

    elif scope != "all":
        raise ValueError(
            f"Unsupported context scope: "
            f"{scope}"
        )

    return (
        pivot.groupby(
            [
                "base_record_id",
                "tensor_type",
            ],
            observed=True,
        )[
            "effect_value"
        ]
        .mean()
        .reset_index()
    )


def build_controlled_effect_stats(
    df: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    specs = [
        (
            "form_normalization",
            "matched",
            True,
        ),
        (
            "form_normalization",
            "all",
            True,
        ),
        (
            "context_removal",
            "matched",
            False,
        ),
        (
            "context_removal",
            "control",
            False,
        ),
    ]

    rows = []

    counter = 30000

    for (
        effect_name,
        scope,
        equal_dataset,
    ) in specs:
        if (
            effect_name
            == "form_normalization"
        ):
            records = (
                build_form_effect_records(
                    df,
                    scope=scope,
                )
            )

        else:
            records = (
                build_context_effect_records(
                    df,
                    scope=scope,
                )
            )

        for tensor_type in TENSORS:
            group = records[
                records[
                    "tensor_type"
                ]
                == tensor_type
            ].copy()

            rng_boot = (
                np.random.default_rng(
                    seed
                    + counter
                )
            )

            rng_perm = (
                np.random.default_rng(
                    seed
                    + counter
                    + 1
                )
            )

            if equal_dataset:
                stats = (
                    bootstrap_equal_dataset_mean(
                        group,
                        value_col=(
                            "effect_value"
                        ),
                        replicates=(
                            bootstrap_replicates
                        ),
                        rng=rng_boot,
                        ci_level=(
                            ci_level
                        ),
                    )
                )

                p_value = (
                    sign_flip_equal_dataset(
                        group,
                        value_col=(
                            "effect_value"
                        ),
                        replicates=(
                            permutation_replicates
                        ),
                        rng=rng_perm,
                    )
                )

            else:
                stats = (
                    bootstrap_simple_mean(
                        group,
                        value_col=(
                            "effect_value"
                        ),
                        replicates=(
                            bootstrap_replicates
                        ),
                        rng=rng_boot,
                        ci_level=(
                            ci_level
                        ),
                    )
                )

                p_value = (
                    sign_flip_simple(
                        group,
                        value_col=(
                            "effect_value"
                        ),
                        replicates=(
                            permutation_replicates
                        ),
                        rng=rng_perm,
                    )
                )

            rows.append(
                {
                    "effect": (
                        effect_name
                    ),
                    "scope": scope,
                    "tensor_type": (
                        tensor_type
                    ),
                    **stats,
                    "p_value": (
                        p_value
                    ),
                }
            )

            counter += 2

    result = pd.DataFrame(
        rows
    )

    result[
        "q_value_bh"
    ] = (
        benjamini_hochberg(
            result[
                "p_value"
            ].to_numpy()
        )
    )

    return result


# ============================================================
# Topic sampling audit
# ============================================================

def build_subject_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    med = (
        df[
            (
                df["dataset"]
                == "medmcqa"
            )
            & (
                df["variant"]
                == "normalized"
            )
        ][
            [
                "base_record_id",
                "subject",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    med["subject"] = (
        med["subject"]
        .fillna("")
        .astype(str)
    )

    med = med[
        med["subject"]
        .str.len()
        > 0
    ]

    return (
        med.groupby(
            "subject",
            observed=True,
        )[
            "base_record_id"
        ]
        .nunique()
        .sort_values(
            ascending=False
        )
        .rename(
            "n_records"
        )
        .reset_index()
    )


# ============================================================
# Single final figure
# ============================================================

def plot_overview(
    layer_stats: pd.DataFrame,
    overall_stats: pd.DataFrame,
    controlled: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    """
    One final research figure:

      A. absolute Q/K/V similarity
      B. base-relative ΔQ/ΔK/ΔV similarity
      C. magnitude-aware differences
      D. controlled form/context effects
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            15,
            10,
        ),
    )

    # --------------------------------------------------------
    # A. Absolute similarity
    # --------------------------------------------------------

    ax = axes[
        0,
        0,
    ]

    absolute = layer_stats[
        layer_stats[
            "metric"
        ]
        == "cosine_mean"
    ]

    for tensor_type in TENSORS:
        group = (
            absolute[
                absolute[
                    "tensor_type"
                ]
                == tensor_type
            ]
            .sort_values(
                "layer"
            )
        )

        line = ax.plot(
            group["layer"],
            group["mean"],
            label=tensor_type,
        )[0]

        ax.fill_between(
            group[
                "layer"
            ].to_numpy(),
            group[
                "ci_low"
            ].to_numpy(),
            group[
                "ci_high"
            ].to_numpy(),
            color=(
                line.get_color()
            ),
            alpha=0.15,
        )

    ax.set_title(
        "A. Cross-LoRA absolute similarity\n"
        "(normalized prompts)"
    )

    ax.set_xlabel(
        "Transformer layer"
    )

    ax.set_ylabel(
        "Cosine similarity"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()


    # --------------------------------------------------------
    # B. Base-relative delta similarity
    # --------------------------------------------------------

    ax = axes[
        0,
        1,
    ]

    delta = layer_stats[
        layer_stats[
            "metric"
        ]
        == "delta_global_cosine"
    ]

    for tensor_type in TENSORS:
        group = (
            delta[
                delta[
                    "tensor_type"
                ]
                == tensor_type
            ]
            .sort_values(
                "layer"
            )
        )

        line = ax.plot(
            group["layer"],
            group["mean"],
            label=tensor_type,
        )[0]

        ax.fill_between(
            group[
                "layer"
            ].to_numpy(),
            group[
                "ci_low"
            ].to_numpy(),
            group[
                "ci_high"
            ].to_numpy(),
            color=(
                line.get_color()
            ),
            alpha=0.15,
        )

    ax.set_title(
        "B. Base-relative LoRA-induced\n"
        "direction similarity"
    )

    ax.set_xlabel(
        "Transformer layer"
    )

    ax.set_ylabel(
        "Cosine similarity of "
        "ΔQ / ΔK / ΔV"
    )

    ax.set_ylim(
        -0.05,
        1.0,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()


    # --------------------------------------------------------
    # C. Magnitude-aware results
    # --------------------------------------------------------

    ax = axes[
        1,
        0,
    ]

    magnitude_metrics = [
        (
            "symmetric_rel_l2",
            "Relative L2",
        ),
        (
            "delta_norm_mean",
            "||ΔX|| / ||X₀||",
        ),
        (
            "delta_distance_vs_base",
            "||ΔXa-ΔXb|| / ||X₀||",
        ),
    ]

    x_positions = np.arange(
        len(
            magnitude_metrics
        ),
        dtype=float,
    )

    width = 0.24

    for index, tensor_type in enumerate(
        TENSORS
    ):
        means = []
        lower_errors = []
        upper_errors = []

        for metric, _ in (
            magnitude_metrics
        ):
            row = overall_stats[
                (
                    overall_stats[
                        "tensor_type"
                    ]
                    == tensor_type
                )
                & (
                    overall_stats[
                        "metric"
                    ]
                    == metric
                )
            ].iloc[0]

            means.append(
                float(
                    row["mean"]
                )
            )

            lower_errors.append(
                float(
                    row["mean"]
                    - row[
                        "ci_low"
                    ]
                )
            )

            upper_errors.append(
                float(
                    row[
                        "ci_high"
                    ]
                    - row["mean"]
                )
            )

        ax.bar(
            (
                x_positions
                + (
                    index - 1
                )
                * width
            ),
            means,
            width=width,
            yerr=np.vstack(
                [
                    lower_errors,
                    upper_errors,
                ]
            ),
            capsize=3,
            label=tensor_type,
        )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        [
            label
            for _, label
            in magnitude_metrics
        ]
    )

    ax.set_title(
        "C. Magnitude-aware "
        "cross-LoRA differences"
    )

    ax.set_ylabel(
        "Dimensionless ratio"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend()


    # --------------------------------------------------------
    # D. Controlled form/context effects
    # --------------------------------------------------------

    ax = axes[
        1,
        1,
    ]

    effect_specs = [
        (
            "form_normalization",
            "matched",
            "Form normalization\n(matched)",
        ),
        (
            "context_removal",
            "matched",
            "Context removal\n(matched)",
        ),
        (
            "context_removal",
            "control",
            "Context removal\n(control)",
        ),
    ]

    x_positions = np.arange(
        len(
            effect_specs
        ),
        dtype=float,
    )

    for index, tensor_type in enumerate(
        TENSORS
    ):
        means = []
        lower_errors = []
        upper_errors = []

        for (
            effect,
            scope,
            _,
        ) in effect_specs:
            row = controlled[
                (
                    controlled[
                        "effect"
                    ]
                    == effect
                )
                & (
                    controlled[
                        "scope"
                    ]
                    == scope
                )
                & (
                    controlled[
                        "tensor_type"
                    ]
                    == tensor_type
                )
            ].iloc[0]

            # Small cosine changes are easier to read as ×10^3.
            means.append(
                1000.0
                * float(
                    row[
                        "mean"
                    ]
                )
            )

            lower_errors.append(
                1000.0
                * float(
                    row["mean"]
                    - row[
                        "ci_low"
                    ]
                )
            )

            upper_errors.append(
                1000.0
                * float(
                    row[
                        "ci_high"
                    ]
                    - row["mean"]
                )
            )

        ax.bar(
            (
                x_positions
                + (
                    index - 1
                )
                * width
            ),
            means,
            width=width,
            yerr=np.vstack(
                [
                    lower_errors,
                    upper_errors,
                ]
            ),
            capsize=3,
            label=tensor_type,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        [
            label
            for _, _, label
            in effect_specs
        ]
    )

    ax.set_title(
        "D. Controlled form/context effects"
    )

    ax.set_ylabel(
        "Δ cosine similarity × 10³"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend()


    fig.suptitle(
        "Phase 3A — Same-domain heterogeneous "
        "LoRA Q/K/V characterization",
        fontsize=14,
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.97,
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Human-readable summary
# ============================================================

def write_key_results(
    overall: pd.DataFrame,
    controlled: pd.DataFrame,
    subject_counts: pd.DataFrame,
    *,
    min_group_size: int,
    path: Path,
) -> None:
    lines = [
        "PHASE 3A REFINED KEY RESULTS",
        "=" * 72,
        (
            "Primary analysis: normalized-only, "
            "cross-LoRA only, pair-average within "
            "source record, equal dataset weight."
        ),
        "",
    ]

    for metric in PRIMARY_METRICS:
        lines.append(
            f"[{metric}]"
        )

        for tensor_type in TENSORS:
            row = overall[
                (
                    overall[
                        "metric"
                    ]
                    == metric
                )
                & (
                    overall[
                        "tensor_type"
                    ]
                    == tensor_type
                )
            ].iloc[0]

            lines.append(
                f"  {tensor_type:8s}: "
                f"{row['mean']:.6f} "
                f"[{row['ci_low']:.6f}, "
                f"{row['ci_high']:.6f}]"
            )

        lines.append("")

    lines.append(
        "[controlled_effects]"
    )

    for _, row in (
        controlled.iterrows()
    ):
        lines.append(
            f"  {row['effect']:20s} "
            f"{row['scope']:8s} "
            f"{row['tensor_type']:8s}: "
            f"{row['mean']:+.6f} "
            f"[{row['ci_low']:+.6f}, "
            f"{row['ci_high']:+.6f}], "
            f"p={row['p_value']:.6g}, "
            f"q={row['q_value_bh']:.6g}"
        )

    max_subject_count = (
        int(
            subject_counts[
                "n_records"
            ].max()
        )
        if not subject_counts.empty
        else 0
    )

    topic_ready = (
        max_subject_count
        >= min_group_size
    )

    lines.extend(
        [
            "",
            (
                "topic/sub-topic readiness: "
                f"{'READY' if topic_ready else 'NOT READY'} "
                f"(max MedMCQA records/subject="
                f"{max_subject_count}, "
                f"required={min_group_size})"
            ),
        ]
    )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

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

    stat_cfg = cfg.get(
        "refined_stats",
        {},
    )

    primary_variant = str(
        stat_cfg.get(
            "primary_variant",
            "normalized",
        )
    )

    bootstrap_replicates = int(
        stat_cfg.get(
            "bootstrap_replicates",
            5000,
        )
    )

    permutation_replicates = int(
        stat_cfg.get(
            "permutation_replicates",
            20000,
        )
    )

    ci_level = float(
        stat_cfg.get(
            "ci_level",
            0.95,
        )
    )

    seed = int(
        stat_cfg.get(
            "seed",
            2026,
        )
    )

    output_root = Path(
        os.path.expandvars(
            cfg[
                "output_root"
            ]
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

    result_dir = (
        output_root
        / "refined_stats"
    )

    figure_dir = (
        output_root
        / "figures"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )

    print(
        "PHASE 3A — REFINED STATISTICAL ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"input={detail_path}"
    )

    print(
        f"primary_variant={primary_variant}"
    )

    print(
        f"bootstrap={bootstrap_replicates}"
    )

    print(
        f"permutation={permutation_replicates}"
    )


    # --------------------------------------------------------
    # Load + audit
    # --------------------------------------------------------

    df = load_detail(
        detail_path
    )

    integrity = audit_detail(
        df
    )

    print(
        "\n[INTEGRITY]"
    )

    print(
        integrity
    )


    # --------------------------------------------------------
    # Primary normalized-only statistics
    # --------------------------------------------------------

    record_df = (
        build_primary_record_table(
            df,
            primary_variant=(
                primary_variant
            ),
        )
    )

    layer_stats = (
        build_primary_layer_stats(
            record_df,
            bootstrap_replicates=(
                bootstrap_replicates
            ),
            ci_level=(
                ci_level
            ),
            seed=seed,
        )
    )

    overall_stats = (
        build_primary_overall_stats(
            record_df,
            bootstrap_replicates=(
                bootstrap_replicates
            ),
            ci_level=(
                ci_level
            ),
            seed=seed,
        )
    )

    pairwise_stats = (
        build_pairwise_overall(
            df,
            primary_variant=(
                primary_variant
            ),
            bootstrap_replicates=(
                bootstrap_replicates
            ),
            ci_level=(
                ci_level
            ),
            seed=seed,
        )
    )


    # --------------------------------------------------------
    # Controlled effects
    # --------------------------------------------------------

    controlled = (
        build_controlled_effect_stats(
            df,
            bootstrap_replicates=(
                bootstrap_replicates
            ),
            permutation_replicates=(
                permutation_replicates
            ),
            ci_level=(
                ci_level
            ),
            seed=seed,
        )
    )


    # --------------------------------------------------------
    # Topic sampling audit
    # --------------------------------------------------------

    subject_counts = (
        build_subject_counts(
            df
        )
    )


    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    layer_stats.to_csv(
        result_dir
        / "primary_layer_bootstrap.csv",
        index=False,
    )

    overall_stats.to_csv(
        result_dir
        / "primary_overall_bootstrap.csv",
        index=False,
    )

    pairwise_stats.to_csv(
        result_dir
        / "pairwise_normalized_overall.csv",
        index=False,
    )

    controlled.to_csv(
        result_dir
        / "controlled_effects.csv",
        index=False,
    )

    subject_counts.to_csv(
        result_dir
        / "medmcqa_subject_counts.csv",
        index=False,
    )


    # --------------------------------------------------------
    # One final overview figure
    # --------------------------------------------------------

    overview_path = (
        figure_dir
        / "phase3a_overview.png"
    )

    plot_overview(
        layer_stats,
        overall_stats,
        controlled,
        output_path=(
            overview_path
        ),
    )


    # --------------------------------------------------------
    # Human-readable key results
    # --------------------------------------------------------

    key_results_path = (
        result_dir
        / "phase3a_key_results.txt"
    )

    min_group_size = int(
        cfg.get(
            "min_group_size",
            4,
        )
    )

    write_key_results(
        overall_stats,
        controlled,
        subject_counts,
        min_group_size=(
            min_group_size
        ),
        path=(
            key_results_path
        ),
    )


    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    max_subject_count = (
        int(
            subject_counts[
                "n_records"
            ].max()
        )
        if not subject_counts.empty
        else 0
    )

    manifest = {
        "status": "PASS",
        "input": str(
            detail_path
        ),
        "integrity": (
            integrity
        ),
        "primary_variant": (
            primary_variant
        ),
        "primary_scope": (
            "cross-LoRA only; pair-average within "
            "source record; equal dataset weighting; "
            "base_record_id bootstrap"
        ),
        "bootstrap_replicates": (
            bootstrap_replicates
        ),
        "permutation_replicates": (
            permutation_replicates
        ),
        "ci_level": (
            ci_level
        ),
        "seed": seed,
        "topic_analysis_ready": (
            max_subject_count
            >= min_group_size
        ),
        "max_medmcqa_records_per_subject": (
            max_subject_count
        ),
        "outputs": {
            "primary_layer": str(
                result_dir
                / "primary_layer_bootstrap.csv"
            ),
            "primary_overall": str(
                result_dir
                / "primary_overall_bootstrap.csv"
            ),
            "pairwise": str(
                result_dir
                / "pairwise_normalized_overall.csv"
            ),
            "controlled_effects": str(
                result_dir
                / "controlled_effects.csv"
            ),
            "subject_counts": str(
                result_dir
                / "medmcqa_subject_counts.csv"
            ),
            "key_results": str(
                key_results_path
            ),
            "overview_figure": str(
                overview_path
            ),
        },
    }

    save_json(
        manifest,
        result_dir
        / "refined_manifest.json",
    )


    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        "\n[PRIMARY OVERALL]"
    )

    print(
        overall_stats.to_string(
            index=False
        )
    )

    print(
        "\n[CONTROLLED EFFECTS]"
    )

    print(
        controlled.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PHASE 3A REFINED ANALYSIS PASS"
    )

    print(
        f"overview={overview_path}"
    )

    print(
        f"key_results={key_results_path}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()