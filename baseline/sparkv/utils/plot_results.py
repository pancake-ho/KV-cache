from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGY_LABELS = {
    "sparkv": "SparKV",
    "strong_hybrid": "Strong Hybrid",
    "local_sparse": "Local Prefill (SpargeAttention)",
    "all_stream": "All Stream (control)",
    "local_full": "Dense Local (diagnostic)",
}

COLORS = {
    "sparkv": "#0072B2",
    "strong_hybrid": "#E69F00",
    "local_sparse": "#009E73",
    "all_stream": "#CC79A7",
    "local_full": "#7F7F7F",
}


def load_result_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Read JSONL results or mixed stdout logs containing JSON records."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped.startswith("{"):
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError:
                    # Pretty-printed scheduler metadata in stdout begins with
                    # "{" but spans multiple lines; it is not a result row.
                    continue
                if not isinstance(value, dict) or "ttft_ms" not in value:
                    continue
                value.setdefault("strategy", "sparkv")
                value.setdefault("repeat", 0)
                value.setdefault("sample_index", len(rows))
                value["_source_file"] = str(path)
                value["_source_line"] = line_number
                rows.append(value)
    if not rows:
        raise ValueError("no JSON result records with ttft_ms were found")
    return rows


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _strategy_order(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "local_sparse",
        "strong_hybrid",
        "all_stream",
        "local_full",
        "sparkv",
    ]
    present = [str(value) for value in frame["strategy"].dropna().unique()]
    return [x for x in preferred if x in present] + sorted(
        set(present) - set(preferred)
    )


def _save(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: list[str],
) -> list[str]:
    paths = []
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)


def _plot_ttft(
    frame: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    order = _strategy_order(frame)
    values = [
        _numeric(frame[frame["strategy"] == strategy], "ttft_ms").dropna()
        / 1000.0
        for strategy in order
    ]
    fig, ax = plt.subplots(figsize=(max(6.5, 1.45 * len(order)), 4.2))
    parts = ax.boxplot(values, patch_artist=True, showmeans=True)
    for patch, strategy in zip(parts["boxes"], order):
        patch.set_facecolor(COLORS.get(strategy, "#56B4E9"))
        patch.set_alpha(0.65)
    for index, series in enumerate(values, start=1):
        if series.empty:
            continue
        offsets = np.linspace(-0.08, 0.08, len(series)) if len(series) > 1 else [0]
        ax.scatter(
            index + np.asarray(offsets),
            series,
            s=24,
            color="black",
            alpha=0.55,
            zorder=3,
        )
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([STRATEGY_LABELS.get(x, x) for x in order], rotation=12)
    ax.set_ylabel("TTFT (s)")
    ax.set_title("End-to-end time to first token")
    _style_axes(ax)
    return _save(fig, output_dir, "01_ttft", formats)


def _plot_energy_quality(
    frame: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    order = _strategy_order(frame)
    labels = [STRATEGY_LABELS.get(x, x) for x in order]
    energy = [
        float(_numeric(frame[frame["strategy"] == x], "ttft_energy_j").median())
        for x in order
    ]
    quality = frame[_numeric(frame, "repeat").fillna(0) == 0]
    f1 = [
        float(_numeric(quality[quality["strategy"] == x], "f1").mean())
        for x in order
    ]

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.3 * len(order)), 4.1))
    axes[0].bar(labels, energy, color=[COLORS.get(x, "#56B4E9") for x in order])
    axes[0].set_ylabel("Energy during TTFT (J)")
    axes[0].set_title("Measured GPU energy scope")
    axes[0].tick_params(axis="x", rotation=15)
    _style_axes(axes[0])

    axes[1].bar(labels, f1, color=[COLORS.get(x, "#56B4E9") for x in order])
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0, max(1.0, np.nanmax(f1) * 1.15 if f1 else 1.0))
    axes[1].set_title("TriviaQA response quality (repeat 0)")
    axes[1].tick_params(axis="x", rotation=15)
    _style_axes(axes[1])
    fig.suptitle("Energy and quality are separate objectives", y=1.02)
    return _save(fig, output_dir, "02_energy_quality", formats)


def _sparkv_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["strategy"] == "sparkv"].copy()


def _plot_prediction_gap(
    frame: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    sparkv = _sparkv_rows(frame)
    predicted = _numeric(sparkv, "schedule_predicted_makespan_ms") / 1000.0
    actual = _numeric(sparkv, "actual_context_rebuild_ms") / 1000.0
    keep = predicted.notna() & actual.notna()
    if not keep.any():
        return []
    sparkv = sparkv.loc[keep]
    predicted = predicted.loc[keep]
    actual = actual.loc[keep]
    labels = [
        f"S{int(sample)} R{int(repeat)}"
        for sample, repeat in zip(sparkv["sample_index"], sparkv["repeat"])
    ]
    positions = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(labels)), 4.2))
    ax.bar(positions - width / 2, predicted, width, label="Scheduler prediction")
    ax.bar(positions + width / 2, actual, width, label="Physical cache rebuild")
    ax.set_xticks(positions, labels, rotation=25)
    ax.set_ylabel("Time (s)")
    ax.set_title("Scheduler estimate versus physical realization")
    ax.legend(frameon=False)
    _style_axes(ax)
    return _save(fig, output_dir, "03_prediction_gap", formats)


def _plot_route_mix(
    frame: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    sparkv = _sparkv_rows(frame)
    local = _numeric(sparkv, "local_units")
    streamed = _numeric(sparkv, "streamed_units")
    total = local + streamed
    keep = total > 0
    if not keep.any():
        return []
    sparkv = sparkv.loc[keep]
    local_share = (local.loc[keep] / total.loc[keep]) * 100.0
    stream_share = (streamed.loc[keep] / total.loc[keep]) * 100.0
    labels = [
        f"S{int(sample)} R{int(repeat)}"
        for sample, repeat in zip(sparkv["sample_index"], sparkv["repeat"])
    ]
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 4.1))
    ax.bar(positions, local_share, label="Local compute", color="#009E73")
    ax.bar(
        positions,
        stream_share,
        bottom=local_share,
        label="Cloud stream",
        color="#CC79A7",
    )
    ax.set_xticks(positions, labels, rotation=25)
    ax.set_ylabel("Final unit ownership (%)")
    ax.set_ylim(0, 100)
    ax.set_title("SparKV route mix after runtime migration")
    ax.legend(frameon=False, ncol=2)
    _style_axes(ax)
    return _save(fig, output_dir, "04_route_mix", formats)


def _plot_non_additive_diagnostics(
    frame: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    sparkv = _sparkv_rows(frame)
    rebuild = _numeric(sparkv, "actual_context_rebuild_ms")
    if not (rebuild > 0).any():
        return []
    metrics = [
        ("wire_ms", "Wire simulation"),
        ("decode_ms", "Huffman + dequantize"),
        ("h2d_ms", "Host to device"),
        ("compute_effective_ms", "Effective local compute"),
        ("physical_dependency_wait_ms", "Physical dependency wait"),
        ("support_attention_ms", "Support attention"),
    ]
    ratios = [
        float((_numeric(sparkv, key) / rebuild).replace([np.inf, -np.inf], np.nan).median())
        * 100.0
        for key, _ in metrics
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars = ax.barh(
        [label for _, label in metrics],
        ratios,
        color=["#56B4E9", "#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7"],
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("Median ratio to context rebuild wall time")
    ax.set_title("Diagnostic ratios (overlapping; do not sum)")
    _style_axes(ax)
    return _save(fig, output_dir, "05_non_additive_diagnostics", formats)


def _plot_stage_profiles(
    frame: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    max_stage_plots: int,
) -> list[str]:
    created: list[str] = []
    sparkv = _sparkv_rows(frame)
    candidates = sparkv[
        sparkv.get("stage_records", pd.Series(index=sparkv.index, dtype=object)).map(
            lambda value: isinstance(value, list) and bool(value)
        )
    ]
    candidates = candidates.sort_values(["sample_index", "repeat"]).head(max_stage_plots)
    for _, row in candidates.iterrows():
        stages = pd.DataFrame(row["stage_records"])
        if stages.empty:
            continue
        x = _numeric(stages, "stage")
        window = max(1, min(15, len(stages) // 20))
        fig, ax = plt.subplots(figsize=(9.2, 4.3))
        series = [
            ("predicted_compute_ms", "Predicted compute", "#009E73"),
            ("actual_compute_worker_ms", "Actual compute worker", "#0072B2"),
            ("predicted_stream_ms", "Predicted stream", "#CC79A7"),
            ("actual_stream_ms", "Actual stream", "#D55E00"),
            ("dependency_wait_ms", "Dependency wait", "#E69F00"),
        ]
        for key, label, color in series:
            values = _numeric(stages, key).rolling(window, min_periods=1).median()
            ax.plot(x, values, label=label, color=color, linewidth=1.35)
        sample = int(row["sample_index"])
        repeat = int(row["repeat"])
        ax.set_xlabel("Stage")
        ax.set_ylabel(f"Rolling median time (ms), window={window}")
        ax.set_title(f"Runtime controller trace: sample {sample}, repeat {repeat}")
        ax.legend(frameon=False, ncol=2)
        _style_axes(ax)
        created.extend(
            _save(fig, output_dir, f"06_stage_profile_s{sample:03d}_r{repeat:02d}", formats)
        )
    return created


def _summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in frame.groupby("strategy", sort=False):
        quality = group[_numeric(group, "repeat").fillna(0) == 0]
        ttft = _numeric(group, "ttft_ms")
        rows.append(
            {
                "strategy": strategy,
                "runs": len(group),
                "samples": group["sample_index"].nunique(),
                "median_ttft_ms": float(ttft.median()),
                "p95_ttft_ms": float(np.nanpercentile(ttft, 95)),
                "median_energy_j": float(_numeric(group, "ttft_energy_j").median()),
                "mean_f1": float(_numeric(quality, "f1").mean()),
                "median_predicted_makespan_ms": float(
                    _numeric(group, "schedule_predicted_makespan_ms").median()
                ),
                "median_context_rebuild_ms": float(
                    _numeric(group, "actual_context_rebuild_ms").median()
                ),
            }
        )
    return pd.DataFrame(rows)


def generate_plots(
    *,
    inputs: list[Path],
    output_dir: Path,
    formats: list[str],
    max_stage_plots: int = 4,
) -> dict[str, Any]:
    unsupported = set(formats) - {"png", "pdf", "svg"}
    if unsupported:
        raise ValueError(f"unsupported formats: {sorted(unsupported)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_result_records(inputs)
    frame = pd.DataFrame(rows)
    frame["strategy"] = frame["strategy"].astype(str)

    created: list[str] = []
    created.extend(_plot_ttft(frame, output_dir, formats))
    created.extend(_plot_energy_quality(frame, output_dir, formats))
    created.extend(_plot_prediction_gap(frame, output_dir, formats))
    created.extend(_plot_route_mix(frame, output_dir, formats))
    created.extend(_plot_non_additive_diagnostics(frame, output_dir, formats))
    created.extend(
        _plot_stage_profiles(frame, output_dir, formats, max_stage_plots)
    )

    summary_path = output_dir / "plot_summary.csv"
    _summary_table(frame).to_csv(summary_path, index=False)
    created.append(str(summary_path))

    manifest = {
        "schema": "sparkv-plot-manifest-v1",
        "inputs": [str(path) for path in inputs],
        "result_records": len(rows),
        "strategies": _strategy_order(frame),
        "formats": formats,
        "files": created,
        "diagnostic_note": (
            "Decode, dependency wait, transfer, and compute counters overlap in wall time; "
            "the diagnostic ratio plot is intentionally non-additive."
        ),
    }
    manifest_path = output_dir / "plot_manifest.json"
    manifest["files"].append(str(manifest_path))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--max-stage-plots", type=int, default=4)
    args = parser.parse_args()

    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    manifest = generate_plots(
        inputs=[Path(name) for name in args.inputs],
        output_dir=Path(args.output_dir),
        formats=formats,
        max_stage_plots=args.max_stage_plots,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
