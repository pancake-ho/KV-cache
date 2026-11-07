from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(
    "outputs/real_handoff_full_le10k"
)

SUMMARY = ROOT / "summary.json"

OUT = ROOT / "plots"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


with SUMMARY.open(
    encoding="utf-8"
) as f:
    summary = json.load(f)


bins = [
    "0-3k",
    "3-5k",
    "5-8k",
    "8-10k",
]

data = summary[
    "by_context_bin"
]


# ============================================================
# Figure 1
# Context Length vs Switching Latency
# ============================================================

target_latency = [
    data[b][
        "target_full_prefill_mean_s"
    ]
    for b in bins
]

crosskv_latency = [
    data[b][
        "crosskv_switch_mean_s"
    ]
    for b in bins
]


plt.figure(
    figsize=(7.2, 4.8)
)

plt.plot(
    bins,
    target_latency,
    marker="o",
    linewidth=2,
    label="Target Full Re-prefill",
)

plt.plot(
    bins,
    crosskv_latency,
    marker="o",
    linewidth=2,
    label="CrossKV Switch",
)

plt.xlabel(
    "Context Length (tokens)"
)

plt.ylabel(
    "Latency (s)"
)

plt.title(
    "CrossKV Switching Latency vs Context Length"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "01_latency_vs_context.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 2
# Overall downstream quality
# ============================================================

overall = summary[
    "overall"
]

models = [
    "Source\nQwen3-1.7B",
    "Target\nQwen3-4B",
    "CrossKV\n1.7B→4B",
]

f1 = [
    overall[
        "source_native_f1"
    ],
    overall[
        "target_native_f1"
    ],
    overall[
        "crosskv_f1"
    ],
]


plt.figure(
    figsize=(6.6, 4.8)
)

bars = plt.bar(
    models,
    f1,
)

plt.ylabel(
    "Mean Token F1"
)

plt.title(
    "Downstream Quality on LongBench-E HotpotQA"
)

plt.ylim(
    0,
    max(f1) * 1.30,
)

plt.grid(
    axis="y",
    alpha=0.3,
)

for bar, value in zip(
    bars,
    f1,
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height()
        + 0.008,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()

plt.savefig(
    OUT / "02_quality_f1.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Figure 3
# Peak GPU memory by context length
# ============================================================

target_memory = [
    data[b][
        "target_prefill_peak_allocated_gib_mean"
    ]
    for b in bins
]

crosskv_memory = [
    data[b][
        "crosskv_switch_peak_allocated_gib_mean"
    ]
    for b in bins
]


x = list(
    range(
        len(bins)
    )
)

width = 0.36


plt.figure(
    figsize=(7.2, 4.8)
)

plt.bar(
    [
        value - width / 2
        for value in x
    ],
    target_memory,
    width=width,
    label="Target Full Re-prefill",
)

plt.bar(
    [
        value + width / 2
        for value in x
    ],
    crosskv_memory,
    width=width,
    label="CrossKV Switch",
)

plt.xticks(
    x,
    bins,
)

plt.xlabel(
    "Context Length (tokens)"
)

plt.ylabel(
    "Peak Allocated GPU Memory (GiB)"
)

plt.title(
    "GPU Memory Overhead of CrossKV"
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "03_peak_memory.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Text summary
# ============================================================

print()
print(
    "Generated:"
)

for path in sorted(
    OUT.glob("*.png")
):
    print(
        " -",
        path,
    )

print()

print(
    "[Overall]"
)

print(
    "Source F1 =",
    overall[
        "source_native_f1"
    ],
)

print(
    "Target F1 =",
    overall[
        "target_native_f1"
    ],
)

print(
    "CrossKV F1 =",
    overall[
        "crosskv_f1"
    ],
)

