"""Conditions and pairing for the UD–ES repair selection experiment."""

import math

from .metrics.record import summarize_trials


VARIANTS = {
    "rank30": {"repair_fraction": 0.30, "first_tokens": 0, "last_tokens": 0},
    "ends30": {"repair_fraction": 0.30, "first_tokens": 4, "last_tokens": 32},
    "rank60": {"repair_fraction": 0.60, "first_tokens": 0, "last_tokens": 0},
}


def measurement_order(index):
    arms = ("full", *VARIANTS, "native")
    block, offset = divmod(index, len(arms))
    rotated = arms[offset:] + arms[:offset]
    return rotated[::-1] if block % 2 else rotated


def selection_metrics(positions, prefix_tokens, options):
    count = math.ceil(options["repair_fraction"] * prefix_tokens)
    first = min(options["first_tokens"], count)
    last = min(options["last_tokens"], count - first)
    if (len(positions) != count or positions[:first] != list(range(first))
            or positions[count - last:] != list(range(prefix_tokens - last, prefix_tokens))):
        raise RuntimeError("Repair positions differ from the configured budget or reserved ends")
    return {
        "count": count,
        "positions": positions,
        "first_4_count": sum(pos < 4 for pos in positions),
        "last_32_count": sum(pos >= prefix_tokens - 32 for pos in positions),
    }


def cross_measurement(full, reuse, setup):
    saved = full["ttft_ms"] - reuse["ttft_ms"]
    download = reuse["target_download"]
    return {
        "reuse": reuse, "source_setup": setup, "saved_ms": saved,
        "headroom_without_download_tcp_ms": saved + download["tcp_ms"],
        "headroom_without_download_total_ms": saved + download["total_ms"],
    }


def summarize_alternatives(questions, repeats):
    return {
        name: summarize_trials(
            [[{**trial, **trial["repair_alternatives"][name]} for trial in trials]
             for trials in questions],
            repeats,
        )
        for name in VARIANTS if name != "rank30"
    }
