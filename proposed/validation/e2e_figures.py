"""Render complete serving reports as paper-ready PDF, PNG and CSV files."""

import csv
from pathlib import Path
from statistics import mean, median

from .e2e_inputs import read


GIB = 1 << 30
STYLE = {
    "full": ("Full prefill", "#555555", "o", "-"),
    "exact": ("Same-model reuse", "#107f91", "s", "--"),
    "ridge": ("Ridge", "#c66026", "^", "-."),
    "cachebridge": ("CacheBridge (adapted)", "#7854a3", "D", ":"),
    "cachebridge_v2": ("CacheBridge v2 mapper", "#b12c56", "P", "--"),
    "rarecache": ("RaReCache", "#1966b2", "o", "-"),
    "mlp": ("MLP (streamed)", "#395a3f", "v", "--"),
    "lora_base": ("Base mapper on LoRA KV", "#9b6535", "^", ":"),
    "lora_fitted": ("Proposed (LoRA-pair)", "#1966b2", "o", "-"),
    "lora_cachebridge_v2": ("CacheBridge v2 (LoRA-pair)", "#b12c56", "P", "--"),
}


def table(report):
    lora_comparison = bool(report["settings"]["profile"].get("lora_cachebridge_v2_history"))
    for key, case in report["results"].items():
        for method, result in case["methods"].items():
            m = result["summary"]
            delta = result.get("paired_vs_full", {})
            f1_ci = delta.get("f1_points", {}).get("document_bootstrap_ci95", [None, None])
            yield {
                "pair": report["settings"]["pair"], "scenario": key, "method": method,
                "codec": case["scenario"]["codec"], "cache_budget_mib": case["scenario"]["cache_mib"],
                "qa_documents": m["qa_documents"], "trace_requests": m["trace_requests_per_replay"],
                "offered_cross_pct": 100 * m["offered_kinds"].get("cross", 0) / m["qa_documents"],
                "offered_same_pct": 100 * m["offered_kinds"].get("same", 0) / m["qa_documents"],
                "qa_hit_pct": 100 * m["qa_hit_rate"], "cross_hit_pct": 100 * m["cross_hit_rate"],
                "ttft_p50_ms": m["ttft_ms"]["p50"], "ttft_p95_ms": m["ttft_ms"]["p95"],
                "e2e_p50_ms": m["application_ms"]["p50"], "e2e_p95_ms": m["application_ms"]["p95"],
                "trace_requests_per_s": m["trace_requests_per_second"],
                "peak_allocated_gib": m["peak_allocated_bytes"] / GIB,
                "peak_reserved_gib": m["peak_reserved_bytes"] / GIB,
                "peak_resident_kv_gib": m["peak_resident_kv_bytes"] / GIB,
                "qa_f1": 100 * m["qa_f1"], "qa_em": 100 * m["qa_em"],
                "delta_f1_vs_full": delta.get("f1_points", {}).get("mean"),
                "delta_f1_ci95_low": f1_ci[0], "delta_f1_ci95_high": f1_ci[1],
                **(reuse_columns(case["scenario"], m) if "reuse" in m else {}),
                **({"elapsed_ms_per_target_qa": m["elapsed_ms_per_target_qa"],
                    "mapping_p50_ms": m["mapping_ms"]["p50"]}
                   if "elapsed_ms_per_target_qa" in m else {}),
                **({"ttft_mean_ms": mean(
                    row["ttft_ms"] for replay in result["replays"] for row in replay["requests"]
                    if row["kind"] != "producer")}
                   if lora_comparison else {}),
            }


def reuse_columns(scenario, summary):
    metrics = summary["reuse"]
    hit_ttft = metrics["cross_hit_ttft_ms"]
    return {
        "reuse_count": scenario["reuse_count"],
        "context_documents": metrics["context_documents"],
        "target_qa": metrics["target_qa_per_replay"],
        "producer_prefills": metrics["producer_prefills_per_replay"],
        "cross_source_prefills": metrics["cross_source_prefills_per_replay"],
        "cross_opportunities": metrics["cross_opportunities_per_replay"],
        "elapsed_per_target_qa_ms": metrics["elapsed_ms_per_target_qa"]["p50"],
        "producer_ms_per_replay": metrics["producer_application_ms_per_replay_p50"],
        "cross_hit_ttft_ms": hit_ttft["p50"] if hit_ttft else None,
        "mapping_p50_ms": summary["mapping_ms"]["p50"],
        "eligible_cross_hit_pct": 100 * metrics["cross_hit_rate_given_eligible"],
        "cache_evictions": median(metrics["cache_evictions_by_replay"]),
        "cache_fill_before_qa_pct": 100 * metrics["cache_fill_before_qa_ratio_p50"],
    }


def validate_complete(report):
    if report.get("status") != "complete" or report.get("schema_version") != 1:
        raise ValueError("Figures require a completed top-system E2E report")
    settings = report["settings"]
    planned = {case["id"] for case in settings["profile"]["scenarios"]}
    if set(report["results"]) != planned or any(
            set(case["methods"]) != set(settings["methods"]) for case in report["results"].values()):
        raise ValueError("The report is missing scenarios or baselines")


def save(plt, fig, path):
    for extension in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + extension), dpi=320, bbox_inches="tight")
    plt.close(fig)


def compare(plt, rows, path, watermark):
    # Paired bars share axes with common units and use hatching that survives grayscale.
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.45), layout="constrained")
    items = [
        ("ttft_p50_ms", "ttft_p95_ms", "TTFT (ms)", "(a)", ("p50", "p95")),
        ("e2e_p50_ms", "e2e_p95_ms", "Request latency (ms)", "(b)", ("p50", "p95")),
        ("peak_allocated_gib", "peak_reserved_gib", "Peak GPU memory (GiB)", "(c)", ("Allocated", "Reserved")),
        ("qa_f1", "qa_em", "QA score (%)", "(d)", ("F1", "Exact match")),
    ]
    for ax, (first, second, ylabel, panel, legend) in zip(axs.flat, items, strict=True):
        colors = [STYLE[r["method"]][1] for r in rows]
        ax.bar([x - .18 for x in range(len(rows))], [r[first] for r in rows], .34,
               color=colors, edgecolor="black", linewidth=.35, label=legend[0])
        ax.bar([x + .18 for x in range(len(rows))], [r[second] for r in rows], .34,
               color="white", edgecolor=colors, hatch="///", linewidth=.8, label=legend[1])
        ax.set_title(panel, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(rows)), [STYLE[r["method"]][0] for r in rows], rotation=24, ha="right")
        ax.legend(loc="upper center", bbox_to_anchor=(.58, 1.12), ncol=2, frameon=False,
                  handlelength=1.25, columnspacing=.8)
        ax.margins(y=.13)
        if ylabel.endswith("(%)"):
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle(watermark, fontsize=10)
    save(plt, fig, path)


def lines(plt, rows, path, watermark):
    rows = [r for r in rows if r["scenario"] in {"no_cross", "mixed", "high_cross"}]
    fig, axs = plt.subplots(1, 3, figsize=(8.25, 2.95), layout="constrained")
    for i, (ax, field, unit) in enumerate(zip(
            axs, ("ttft_p50_ms", "cross_hit_pct", "trace_requests_per_s"),
            ("Median TTFT (ms)", "Actual cross-model hits (%)", "Trace throughput (req/s)"),
            strict=True)):
        for method in dict.fromkeys(r["method"] for r in rows):
            values = sorted((r["offered_cross_pct"], r[field]) for r in rows if r["method"] == method)
            label, col, marker, linestyle = STYLE[method]
            ax.plot(*zip(*values), color=col, marker=marker, linestyle=linestyle,
                    label=label, linewidth=1.35, markersize=3.7)
        ax.set_ylabel(unit)
        ax.set_xlabel("Eligible cross-model requests (%)")
        ax.set_title(f"({chr(97 + i)})", loc="left")
        ax.set_xticks(sorted({r["offered_cross_pct"] for r in rows}))
        if field == "cross_hit_pct":
            ax.set_ylim(0, 100)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    fig.suptitle(watermark, fontsize=10)
    save(plt, fig, path)


def cache_tradeoff(plt, rows, path, watermark):
    rows = [r for r in rows if r["scenario"] in {"mixed", "small_cache", "bitpack5"}]
    order = ("small_cache", "mixed", "bitpack5")
    fig, axs = plt.subplots(1, 3, figsize=(8.3, 3.0), layout="constrained")
    for i, (ax, field, title) in enumerate(zip(
            axs, ("ttft_p50_ms", "peak_resident_kv_gib", "qa_hit_pct"),
            ("Median TTFT (ms)", "Resident cache peak (GiB)", "Actual cache hits (%)"), strict=True)):
        for method in dict.fromkeys(r["method"] for r in rows):
            label, color, marker, linestyle = STYLE[method]
            vals = {r["scenario"]: r[field] for r in rows if r["method"] == method}
            ax.plot(range(len(order)), [vals.get(k) for k in order], color=color, marker=marker,
                    linestyle=linestyle, label=label, linewidth=1.35, markersize=3.7)
        ax.set_title(f"({chr(97 + i)})", loc="left")
        ax.set_ylabel(title)
        if field == "qa_hit_pct":
            ax.set_ylim(0, 100)
        elif field == "peak_resident_kv_gib":
            ax.set_ylim(bottom=0)
        sizes = {r["scenario"]: r["cache_budget_mib"] for r in rows}
        ax.set_xticks(range(len(order)), [f"{sizes[k]} MiB\n{('5-bit' if k == 'bitpack5' else 'Raw')}"
                                         for k in order])
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    fig.suptitle(watermark, fontsize=10)
    save(plt, fig, path)


def render(report_path, out):
    report = read(report_path)
    validate_complete(report)
    rows = list(table(report))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reuse_mode = report["settings"]["profile"].get("reuse_experiment")
    csv_name = "reuse_metrics.csv" if reuse_mode else "metrics.csv"
    with (out / csv_name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e1e5e8", "grid.linewidth": .45,
    })
    if reuse_mode:
        plot_reuse(plt, rows, out, reuse_mode)
        print(f"Figures {out}", flush=True)
        return
    info = report["settings"]
    if info["profile"].get("lora_stage") == "evaluate":
        compare_v2 = bool(info["profile"].get("lora_cachebridge_v2_history"))
        name = "lora_cachebridge_v2_comparison" if compare_v2 else "lora_transfer_comparison"
        plot_lora_routes(plt, rows, info["lora_routes"], out / name, mean_ttft=compare_v2)
        print(f"Figures {out}", flush=True)
        return
    if info["profile"].get("compare_cachebridge_v2"):
        plot_sweep_panels(plt, rows, out / "cachebridge_v2_comparison", "offered_cross_pct",
                          "Eligible cross-model QA requests (%)", (
                              ("ttft_p50_ms", "Median TTFT (ms)"),
                              ("e2e_p50_ms", "Median E2E latency (ms)"),
                              ("peak_allocated_gib", "Peak GPU allocated (GiB)"),
                              ("qa_f1", "QA F1 (%)"),
                          ))
        print(f"Figures {out}", flush=True)
        return
    scope = f"{info['pair']}   ·   {info['profile']['documents']} QA documents"
    if info["profile_name"] == "pilot":
        scope += "   ·   pilot"
    cases = set(report["results"])
    primary = "mixed" if "mixed" in cases else next(iter(report["results"]))
    compare(plt, [row for row in rows if row["scenario"] == primary], out / "e2e_baselines", scope)
    if {"no_cross", "mixed", "high_cross"} <= cases:
        lines(plt, rows, out / "cross_model_share", scope)
    if {"mixed", "small_cache", "bitpack5"} <= cases:
        cache_tradeoff(plt, rows, out / "cache_tradeoff", scope)
    print(f"Figures {out}", flush=True)


def plot_lora_routes(plt, rows, routes, path, *, mean_ttft=False):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45), layout="constrained", sharex=True)
    panels = (
        ("ttft_mean_ms", "Mean TTFT (ms)") if mean_ttft else ("ttft_p50_ms", "Median TTFT (ms)"),
        ("elapsed_ms_per_target_qa", "Total time per target QA (ms)"),
        ("peak_allocated_gib", "Peak GPU allocated (GiB)"),
        ("qa_f1", "QA F1 (%)"),
    )
    methods = list(dict.fromkeys(row["method"] for row in rows))
    values = {(row["scenario"], row["method"]): row for row in rows}
    positions = range(len(routes))
    for ax, (key, label) in zip(axes.flat, panels, strict=True):
        for i, method in enumerate(methods):
            name, color, marker, _ = STYLE[method]
            offset = .13 * (i - (len(methods) - 1) / 2)
            ax.scatter([x + offset for x in positions], [values[route, method][key] for route in routes],
                       color=color, marker=marker, label=name, s=24)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xticks(list(positions), [route["target"]["dataset"] for route in routes.values()])
        ax.set_xlabel("Target LoRA training dataset")
    axes[0, 0].legend(loc="lower right", fontsize=7, framealpha=.93)
    save(plt, fig, path)


def plot_reuse(plt, rows, out, mode):
    if mode == "amortization":
        x_key, x_label = "reuse_count", "Target queries per cross-model source prefill"
    else:
        x_key, x_label = "cache_budget_mib", "KV cache budget (MiB)"
    figures = {
        "reuse_e2e": (
            ("elapsed_per_target_qa_ms", "Elapsed time per target QA (ms)"),
            ("ttft_p50_ms", "Median TTFT (ms)"),
            ("peak_allocated_gib", "Peak GPU allocated (GiB)"),
            ("qa_f1", "QA F1 (%)"),
        ),
        "reuse_cache": (
            ("eligible_cross_hit_pct", "Eligible cross-model hit rate (%)"),
            ("peak_resident_kv_gib", "Peak resident KV (GiB)"),
            ("cache_evictions", "Cache evictions per replay"),
            ("cache_fill_before_qa_pct", "Median KV budget used (%)"),
        ),
    }
    for name, metrics in figures.items():
        plot_sweep_panels(plt, rows, out / name, x_key, x_label, metrics)


def plot_sweep_panels(plt, rows, path, x_key, x_label, metrics):
    ticks = sorted({row[x_key] for row in rows})
    methods = dict.fromkeys(row["method"] for row in rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45), layout="constrained", sharex=True)
    for ax, (key, label) in zip(axes.flat, metrics, strict=True):
        for method in methods:
            values = sorted((row[x_key], row[key]) for row in rows if row["method"] == method)
            method_label, color, marker, linestyle = STYLE[method]
            if method == "cachebridge" and "cachebridge_v2" in methods:
                method_label = "CacheBridge v1 mapper"
            ax.plot(*zip(*values), label="Proposed" if method == "rarecache" else method_label,
                    color=color, marker=marker, linestyle=linestyle, linewidth=1.35, markersize=4)
        ax.set(ylabel=label, xticks=ticks)
        ax.grid(axis="x", visible=False)
        if label.endswith("(%)"):
            ax.set_ylim(-4, 104)
    for ax in axes[-1]:
        ax.set_xlabel(x_label)
    axes[0, 0].legend(loc="lower right", fontsize=7, handlelength=1.5, framealpha=.93)
    save(plt, fig, path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    path = parser.parse_args().report
    render(path, path.parent / "figures")
