#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TENSORS = ["q_attn", "k_cache", "v_cache"]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def bootstrap_mean(values, reps, ci, rng):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    boot = values[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def sign_flip_p(values, reps, rng):
    values = np.asarray(values, dtype=float)
    obs = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= obs) + 1) / (reps + 1))


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def paired_relation_table(df, metric):
    # First average pair identities inside each relation.
    pair_avg = (
        df.groupby(
            [
                "probe_id",
                "probe_dataset",
                "probe_domain",
                "tensor_type",
                "layer",
                "pair_relation",
            ],
            observed=True,
        )[metric]
        .mean()
        .reset_index()
    )

    pivot = (
        pair_avg.pivot_table(
            index=[
                "probe_id",
                "probe_dataset",
                "probe_domain",
                "tensor_type",
                "layer",
            ],
            columns="pair_relation",
            values=metric,
            aggfunc="first",
        )
        .dropna(subset=["within_domain", "cross_domain"])
        .reset_index()
    )
    pivot["effect"] = pivot["within_domain"] - pivot["cross_domain"]
    return pivot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3c_cross_domain.yaml")
    args = parser.parse_args()

    cfg = read_yaml(PROJECT_ROOT / args.config)["phase3c"]
    out_root = Path(
        os.path.expandvars(cfg["characterization_output_root"])
    ).resolve()

    detail_path = out_root / "same_input_detail.csv"
    df = pd.read_csv(detail_path, low_memory=False)

    expected_probes = 4 * int(cfg["probe_records_per_dataset"])
    if df["probe_id"].nunique() != expected_probes:
        raise RuntimeError(
            f"Probe count mismatch: {df['probe_id'].nunique()} != {expected_probes}"
        )
    if set(df["tensor_type"].unique()) != set(TENSORS):
        raise RuntimeError("Unexpected tensor types.")
    if df["layer"].nunique() != 36:
        raise RuntimeError("Expected 36 layers.")

    reps = int(cfg["bootstrap_replicates"])
    perm_reps = int(cfg["permutation_replicates"])
    ci = float(cfg["ci_level"])
    seed = int(cfg["stats_seed"])

    stats_dir = out_root / "summary"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Layer curves for within/cross domain.
    # --------------------------------------------------------
    layer_rows = []
    for metric in ["cosine_mean", "delta_global_cosine", "symmetric_rel_l2"]:
        pair_avg = (
            df.groupby(
                ["probe_id", "tensor_type", "layer", "pair_relation"],
                observed=True,
            )[metric]
            .mean()
            .reset_index()
        )
        for keys, group in pair_avg.groupby(
            ["tensor_type", "layer", "pair_relation"],
            observed=True,
        ):
            tensor, layer, relation = keys
            values = group[metric].to_numpy(dtype=float)
            mean, lo, hi = bootstrap_mean(
                values,
                reps,
                ci,
                np.random.default_rng(seed + len(layer_rows)),
            )
            layer_rows.append(
                {
                    "metric": metric,
                    "tensor_type": tensor,
                    "layer": int(layer),
                    "pair_relation": relation,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_probes": len(values),
                }
            )
    layer_stats = pd.DataFrame(layer_rows)
    layer_stats.to_csv(stats_dir / "layer_stats.csv", index=False)

    # --------------------------------------------------------
    # Overall within-vs-cross effects.
    # Layers are averaged within each probe BEFORE inference.
    # --------------------------------------------------------
    effect_rows = []
    counter = 10000
    for metric in ["cosine_mean", "delta_global_cosine", "symmetric_rel_l2"]:
        pivot = paired_relation_table(df, metric)
        probe_avg = (
            pivot.groupby(
                ["probe_id", "probe_dataset", "probe_domain", "tensor_type"],
                observed=True,
            )[["within_domain", "cross_domain", "effect"]]
            .mean()
            .reset_index()
        )

        for tensor in TENSORS:
            for scope, scope_df in [
                ("all", probe_avg),
                ("medical_probe", probe_avg[probe_avg["probe_domain"] == "medical"]),
                ("finance_probe", probe_avg[probe_avg["probe_domain"] == "finance"]),
            ]:
                g = scope_df[scope_df["tensor_type"] == tensor]
                values = g["effect"].to_numpy(dtype=float)
                mean, lo, hi = bootstrap_mean(
                    values,
                    reps,
                    ci,
                    np.random.default_rng(seed + counter),
                )
                p = sign_flip_p(
                    values,
                    perm_reps,
                    np.random.default_rng(seed + counter + 1),
                )
                effect_rows.append(
                    {
                        "metric": metric,
                        "scope": scope,
                        "tensor_type": tensor,
                        "within_mean": float(g["within_domain"].mean()),
                        "cross_mean": float(g["cross_domain"].mean()),
                        "effect_within_minus_cross": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "p_value": p,
                        "n_probes": len(values),
                    }
                )
                counter += 2

    effects = pd.DataFrame(effect_rows)
    effects["q_value_bh"] = bh(effects["p_value"].tolist())
    effects.to_csv(stats_dir / "domain_effects.csv", index=False)

    # --------------------------------------------------------
    # Pair table, useful to show 2 within-domain vs 4 cross-domain pairs.
    # --------------------------------------------------------
    pair_table = (
        df.groupby(
            ["pair_a", "pair_b", "pair_relation", "tensor_type"],
            observed=True,
        )[["cosine_mean", "delta_global_cosine", "symmetric_rel_l2"]]
        .mean()
        .reset_index()
    )
    pair_table.to_csv(stats_dir / "pair_table.csv", index=False)

    # --------------------------------------------------------
    # One professor-facing overview figure.
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    def plot_layer(ax, metric, title, ylabel):
        sub = layer_stats[layer_stats["metric"] == metric]
        styles = {"within_domain": "-", "cross_domain": "--"}
        for tensor in TENSORS:
            for relation in ["within_domain", "cross_domain"]:
                g = sub[
                    (sub["tensor_type"] == tensor)
                    & (sub["pair_relation"] == relation)
                ].sort_values("layer")
                line = ax.plot(
                    g["layer"],
                    g["mean"],
                    linestyle=styles[relation],
                    label=f"{tensor}/{relation.replace('_domain','')}",
                )[0]
                ax.fill_between(
                    g["layer"].to_numpy(),
                    g["ci_low"].to_numpy(),
                    g["ci_high"].to_numpy(),
                    alpha=0.08,
                )
        ax.set_title(title)
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    plot_layer(
        axes[0, 0],
        "cosine_mean",
        "A. Same-input Q/K/V: within-domain vs cross-domain LoRA pairs",
        "Absolute cosine similarity",
    )
    plot_layer(
        axes[0, 1],
        "delta_global_cosine",
        "B. Base-relative LoRA-induced direction",
        "Cosine similarity of ΔQ / ΔK / ΔV",
    )

    # C: overall absolute similarity within vs cross.
    ax = axes[1, 0]
    overall_abs = effects[
        (effects["metric"] == "cosine_mean")
        & (effects["scope"] == "all")
    ]
    x = np.arange(len(TENSORS))
    width = 0.34
    within = []
    cross = []
    for tensor in TENSORS:
        row = overall_abs[overall_abs["tensor_type"] == tensor].iloc[0]
        within.append(float(row["within_mean"]))
        cross.append(float(row["cross_mean"]))
    ax.bar(x - width / 2, within, width=width, label="Within-domain LoRA pair")
    ax.bar(x + width / 2, cross, width=width, label="Cross-domain LoRA pair")
    ax.set_xticks(x)
    ax.set_xticklabels(TENSORS)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("C. Overall absolute similarity (same input)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    # D: main domain effect for absolute and delta similarity.
    ax = axes[1, 1]
    metric_specs = [
        ("cosine_mean", "Absolute cosine"),
        ("delta_global_cosine", "Δ-direction cosine"),
    ]
    x = np.arange(len(metric_specs))
    width = 0.24
    for ti, tensor in enumerate(TENSORS):
        means, lows, highs = [], [], []
        for metric, _ in metric_specs:
            row = effects[
                (effects["metric"] == metric)
                & (effects["scope"] == "all")
                & (effects["tensor_type"] == tensor)
            ].iloc[0]
            means.append(1000.0 * float(row["effect_within_minus_cross"]))
            lows.append(
                1000.0
                * float(row["effect_within_minus_cross"] - row["ci_low"])
            )
            highs.append(
                1000.0
                * float(row["ci_high"] - row["effect_within_minus_cross"])
            )
        ax.bar(
            x + (ti - 1) * width,
            means,
            width=width,
            yerr=np.vstack([lows, highs]),
            capsize=3,
            label=tensor,
        )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_specs])
    ax.set_ylabel("(Within-domain − Cross-domain) × 10³")
    ax.set_title("D. Domain-trained LoRA effect")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle(
        "Phase 3C — Cross-domain LoRA Q/K/V Characterization "
        "(Medical vs Finance, Identical Input)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figure_path = out_root / "phase3c_professor_overview.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # Human-readable results.
    # --------------------------------------------------------
    lines = [
        "PHASE 3C CROSS-DOMAIN SAME-INPUT RESULTS",
        "=" * 80,
        "Primary question: when the exact same prompt is given to all four LoRA",
        "adapters, are Q/K/V more similar for adapters trained in the same domain",
        "(medical-medical or finance-finance) than across domains?",
        "",
        "[DOMAIN EFFECTS: ALL PROBES]",
    ]
    for _, row in effects[effects["scope"] == "all"].iterrows():
        lines.append(
            f"{row['metric']:22s} | {row['tensor_type']:8s} | "
            f"within={row['within_mean']:.6f} | cross={row['cross_mean']:.6f} | "
            f"effect={row['effect_within_minus_cross']:+.6f} "
            f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
            f"p={row['p_value']:.6g} | q={row['q_value_bh']:.6g}"
        )

    key_path = out_root / "phase3c_key_results.txt"
    key_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    save_json(
        {
            "status": "PASS",
            "num_probes": int(df["probe_id"].nunique()),
            "outputs": {
                "domain_effects": str(stats_dir / "domain_effects.csv"),
                "pair_table": str(stats_dir / "pair_table.csv"),
                "layer_stats": str(stats_dir / "layer_stats.csv"),
                "overview": str(figure_path),
                "key_results": str(key_path),
            },
        },
        out_root / "summary_manifest.json",
    )

    print("\n" + "=" * 80)
    print("PHASE 3C SUMMARY PASS")
    print(f"overview={figure_path}")
    print(f"key_results={key_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
