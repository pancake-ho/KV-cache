from __future__ import annotations

import argparse
import json
import math
import re
import string
from pathlib import Path

import numpy as np

from xmodel_kv.hotpotqa import (
    answer_em,
    answer_f1,
)


# ============================================================
# Robust diagnostic answer-hit
#
# Previous implementation used:
#
#     normalized_gold in normalized_prediction
#
# which can incorrectly match:
#
#     gold = "no"
#     prediction contains "known"
#
# Here we require a CONTIGUOUS TOKEN SEQUENCE match.
# ============================================================

def normalize_tokens(text: str) -> list[str]:

    text = text.lower()

    text = "".join(
        ch
        for ch in text
        if ch not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    return text.split()


def token_phrase_hit(
    prediction: str,
    golds: list[str],
) -> float:

    pred = normalize_tokens(
        prediction
    )

    for gold in golds:

        g = normalize_tokens(
            str(gold)
        )

        if not g:
            continue

        width = len(g)

        for start in range(
            0,
            len(pred) - width + 1,
        ):

            if (
                pred[
                    start:
                    start + width
                ]
                == g
            ):
                return 1.0

    return 0.0


def load_rows(
    path: Path,
) -> list[dict]:

    rows = []

    with path.open(
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            rows.append(
                json.loads(line)
            )

    return rows


def rescore(
    row: dict,
) -> dict:

    golds = [
        str(x)
        for x in row[
            "gold_answers"
        ]
    ]

    for key in (
        "source_native",
        "target_native",
        "crosskv_transfer",
    ):

        prediction = row[key][
            "clean"
        ]

        row[key][
            "policy_em"
        ] = answer_em(
            prediction,
            golds,
        )

        row[key][
            "policy_f1"
        ] = answer_f1(
            prediction,
            golds,
        )

        row[key][
            "token_answer_hit"
        ] = token_phrase_hit(
            prediction,
            golds,
        )

    return row


# ============================================================
# Policy evaluation
# ============================================================

def evaluate_policy(
    rows: list[dict],
    *,
    name: str,
    choose_crosskv,
) -> dict:

    baseline_latencies = []

    chosen_latencies = []

    target_f1 = []
    chosen_f1 = []

    target_hits = []
    chosen_hits = []

    transfer_count = 0

    selected_peak_memory = []

    target_peak_memory = []

    lost_target_hits = 0
    gained_over_target_hits = 0

    for row in rows:

        use_crosskv = bool(
            choose_crosskv(row)
        )

        if use_crosskv:
            transfer_count += 1

        baseline_latency = (
            row[
                "timing_seconds"
            ][
                "target_full_prefill"
            ]
        )

        if use_crosskv:

            latency = (
                row[
                    "timing_seconds"
                ][
                    "crosskv_switch_total"
                ]
            )

            quality = row[
                "crosskv_transfer"
            ]

            peak_memory = (
                row[
                    "memory"
                ][
                    "crosskv_switch"
                ][
                    "peak_allocated_gib"
                ]
            )

        else:

            latency = (
                baseline_latency
            )

            quality = row[
                "target_native"
            ]

            peak_memory = (
                row[
                    "memory"
                ][
                    "target_full_prefill"
                ][
                    "peak_allocated_gib"
                ]
            )

        target = row[
            "target_native"
        ]

        target_hit = float(
            target[
                "token_answer_hit"
            ]
        )

        chosen_hit = float(
            quality[
                "token_answer_hit"
            ]
        )

        baseline_latencies.append(
            baseline_latency
        )

        chosen_latencies.append(
            latency
        )

        target_f1.append(
            target[
                "policy_f1"
            ]
        )

        chosen_f1.append(
            quality[
                "policy_f1"
            ]
        )

        target_hits.append(
            target_hit
        )

        chosen_hits.append(
            chosen_hit
        )

        target_peak_memory.append(
            row[
                "memory"
            ][
                "target_full_prefill"
            ][
                "peak_allocated_gib"
            ]
        )

        selected_peak_memory.append(
            peak_memory
        )

        if (
            target_hit == 1.0
            and
            chosen_hit == 0.0
        ):
            lost_target_hits += 1

        if (
            target_hit == 0.0
            and
            chosen_hit == 1.0
        ):
            gained_over_target_hits += 1

    baseline_latencies = np.asarray(
        baseline_latencies,
        dtype=np.float64,
    )

    chosen_latencies = np.asarray(
        chosen_latencies,
        dtype=np.float64,
    )

    target_f1 = np.asarray(
        target_f1,
        dtype=np.float64,
    )

    chosen_f1 = np.asarray(
        chosen_f1,
        dtype=np.float64,
    )

    target_hits = np.asarray(
        target_hits,
        dtype=np.float64,
    )

    chosen_hits = np.asarray(
        chosen_hits,
        dtype=np.float64,
    )

    target_peak_memory = np.asarray(
        target_peak_memory,
        dtype=np.float64,
    )

    selected_peak_memory = np.asarray(
        selected_peak_memory,
        dtype=np.float64,
    )

    baseline_mean = float(
        baseline_latencies.mean()
    )

    chosen_mean = float(
        chosen_latencies.mean()
    )

    return {
        "name":
            name,

        "n":
            len(rows),

        "transfer_count":
            transfer_count,

        "transfer_fraction":
            transfer_count
            / len(rows),

        "latency_mean_s":
            chosen_mean,

        "all_reprefill_latency_mean_s":
            baseline_mean,

        # Ratio of means, not mean of per-sample ratios.
        "speedup_vs_all_reprefill":
            baseline_mean
            / chosen_mean,

        "latency_saved_percent":
            100.0
            * (
                1.0
                - chosen_mean
                / baseline_mean
            ),

        "target_f1_mean":
            float(
                target_f1.mean()
            ),

        "selected_f1_mean":
            float(
                chosen_f1.mean()
            ),

        "f1_delta":
            float(
                chosen_f1.mean()
                - target_f1.mean()
            ),

        "target_answer_hit_rate":
            float(
                target_hits.mean()
            ),

        "selected_answer_hit_rate":
            float(
                chosen_hits.mean()
            ),

        "answer_hit_delta":
            float(
                chosen_hits.mean()
                - target_hits.mean()
            ),

        "lost_target_answer_hits":
            lost_target_hits,

        "gained_answer_hits_over_target":
            gained_over_target_hits,

        "target_peak_memory_mean_gib":
            float(
                target_peak_memory.mean()
            ),

        "selected_peak_memory_mean_gib":
            float(
                selected_peak_memory.mean()
            ),
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    rows = load_rows(
        Path(
            args.input
        )
    )

    rows = [
        rescore(row)
        for row in rows
    ]

    # ========================================================
    # 1. Baselines
    # ========================================================

    policies = []

    policies.append(
        evaluate_policy(
            rows,
            name="always_reprefill",
            choose_crosskv=lambda row:
                False,
        )
    )

    policies.append(
        evaluate_policy(
            rows,
            name="always_crosskv",
            choose_crosskv=lambda row:
                True,
        )
    )

    # ========================================================
    # 2. Length-only policies
    #
    # CrossKV only when context >= threshold.
    # ========================================================

    thresholds = [
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
    ]

    for threshold in thresholds:

        policies.append(
            evaluate_policy(
                rows,
                name=(
                    f"length_ge_{threshold}"
                ),
                choose_crosskv=(
                    lambda row,
                    threshold=threshold:
                        row[
                            "context_tokens"
                        ]
                        >= threshold
                ),
            )
        )

    # ========================================================
    # 3. Oracle upper bound: answer-hit preserving
    #
    # Use CrossKV iff:
    #   1) it is actually faster
    #   2) it does not destroy target token-answer-hit
    #
    # This is NOT deployable because it uses gold labels.
    # It is an upper bound for a future predictor.
    # ========================================================

    def oracle_hit_safe(
        row: dict,
    ) -> bool:

        faster = (
            row[
                "timing_seconds"
            ][
                "crosskv_switch_total"
            ]
            <
            row[
                "timing_seconds"
            ][
                "target_full_prefill"
            ]
        )

        target_hit = (
            row[
                "target_native"
            ][
                "token_answer_hit"
            ]
        )

        transfer_hit = (
            row[
                "crosskv_transfer"
            ][
                "token_answer_hit"
            ]
        )

        non_degrading = (
            transfer_hit
            >= target_hit
        )

        return (
            faster
            and
            non_degrading
        )

    policies.append(
        evaluate_policy(
            rows,
            name=(
                "oracle_hit_preserving"
            ),
            choose_crosskv=(
                oracle_hit_safe
            ),
        )
    )

    # ========================================================
    # 4. Oracle upper bound: F1 non-degrading
    #
    # Transfer only when:
    #   CrossKV is faster AND CrossKV F1 >= target F1.
    # ========================================================

    def oracle_f1_safe(
        row: dict,
    ) -> bool:

        faster = (
            row[
                "timing_seconds"
            ][
                "crosskv_switch_total"
            ]
            <
            row[
                "timing_seconds"
            ][
                "target_full_prefill"
            ]
        )

        non_degrading = (
            row[
                "crosskv_transfer"
            ][
                "policy_f1"
            ]
            >=
            row[
                "target_native"
            ][
                "policy_f1"
            ]
        )

        return (
            faster
            and
            non_degrading
        )

    policies.append(
        evaluate_policy(
            rows,
            name=(
                "oracle_f1_nondegrading"
            ),
            choose_crosskv=(
                oracle_f1_safe
            ),
        )
    )

    # ========================================================
    # Print
    # ========================================================

    print()
    print(
        "=" * 115
    )

    print(
        "CROSSKV HANDOFF POLICY ANALYSIS"
    )

    print(
        "=" * 115
    )

    print(
        f"{'Policy':<30}"
        f"{'Tx%':>8}"
        f"{'Latency':>11}"
        f"{'Speedup':>10}"
        f"{'Save%':>9}"
        f"{'F1':>9}"
        f"{'ΔF1':>9}"
        f"{'Hit':>9}"
        f"{'ΔHit':>9}"
        f"{'LostHit':>10}"
    )

    print(
        "-" * 115
    )

    for p in policies:

        print(
            f"{p['name']:<30}"
            f"{100*p['transfer_fraction']:8.1f}"
            f"{p['latency_mean_s']:11.4f}"
            f"{p['speedup_vs_all_reprefill']:10.3f}"
            f"{p['latency_saved_percent']:9.2f}"
            f"{p['selected_f1_mean']:9.3f}"
            f"{p['f1_delta']:9.3f}"
            f"{p['selected_answer_hit_rate']:9.3f}"
            f"{p['answer_hit_delta']:9.3f}"
            f"{p['lost_target_answer_hits']:10d}"
        )

    result = {
        "metric_note": (
            "token_answer_hit uses contiguous normalized "
            "token-sequence matching. Oracle policies use "
            "ground-truth quality and are upper bounds, "
            "not deployable policies."
        ),

        "documents":
            len(rows),

        "policies":
            policies,
    }

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
