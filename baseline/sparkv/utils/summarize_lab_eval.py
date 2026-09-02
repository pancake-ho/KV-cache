from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _p95(
    values,
) -> float:
    return float(
        np.percentile(
            values,
            95,
        )
    )


def _safe_median(
    frame: pd.DataFrame,
    column: str,
) -> float:
    if column not in frame:
        return float("nan")
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()
    if values.empty:
        return float("nan")
    return float(
        values.median()
    )


def _ratio(
    numerator: float,
    denominator: float,
) -> float:
    if (
        not np.isfinite(
            numerator
        )
        or not np.isfinite(
            denominator
        )
        or denominator <= 0
    ):
        return float("nan")
    return float(
        numerator
        / denominator
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
    )
    parser.add_argument(
        "--csv",
        required=True,
    )
    parser.add_argument(
        "--report",
        required=True,
    )
    parser.add_argument(
        "--predictor",
        default=None,
    )
    args = parser.parse_args()

    rows = []
    with Path(
        args.input
    ).open(
        encoding="utf-8"
    ) as handle:
        rows.extend(
            json.loads(line)
            for line in handle
            if line.strip()
        )

    frame = pd.DataFrame(
        rows
    )
    required = {
        "strategy",
        "sample_index",
        "repeat",
        "ttft_ms",
        "f1",
    }
    missing = (
        required
        - set(
            frame.columns
        )
    )
    if missing:
        raise ValueError(
            "missing evaluation fields: "
            f"{sorted(missing)}"
        )

    summary_rows = []
    for strategy, group in (
        frame.groupby(
            "strategy",
            sort=True,
        )
    ):
        quality = group[
            group[
                "repeat"
            ]
            == 0
        ]

        ttft = pd.to_numeric(
            group[
                "ttft_ms"
            ],
            errors="raise",
        )

        summary_rows.append(
            {
                "strategy":
                    strategy,
                "runs":
                    int(
                        len(group)
                    ),
                "samples":
                    int(
                        group[
                            "sample_index"
                        ].nunique()
                    ),
                "repeats":
                    int(
                        group[
                            "repeat"
                        ].nunique()
                    ),
                "median_ttft_ms":
                    float(
                        ttft.median()
                    ),
                "p95_ttft_ms":
                    _p95(
                        ttft
                    ),
                "mean_ttft_ms":
                    float(
                        ttft.mean()
                    ),
                "std_ttft_ms":
                    float(
                        ttft.std(
                            ddof=0
                        )
                    ),
                "mean_f1":
                    float(
                        pd.to_numeric(
                            quality[
                                "f1"
                            ],
                            errors="coerce",
                        )
                        .mean()
                    ),
                "peak_vram_mib":
                    float(
                        pd.to_numeric(
                            group[
                                "peak_vram_mib"
                            ],
                            errors="coerce",
                        )
                        .max()
                    ),
                "median_energy_j":
                    _safe_median(
                        group,
                        "ttft_energy_j",
                    ),
                "median_rebuild_ms":
                    _safe_median(
                        group,
                        "actual_context_rebuild_ms",
                    ),
                "median_predicted_makespan_ms":
                    _safe_median(
                        group,
                        "schedule_predicted_makespan_ms",
                    ),
                "median_decode_ms":
                    _safe_median(
                        group,
                        "decode_ms",
                    ),
                "median_wire_ms":
                    _safe_median(
                        group,
                        "wire_ms",
                    ),
                "median_h2d_ms":
                    _safe_median(
                        group,
                        "h2d_ms",
                    ),
                "median_dependency_wait_ms":
                    _safe_median(
                        group,
                        "physical_dependency_wait_ms",
                    ),
                "median_compute_effective_ms":
                    _safe_median(
                        group,
                        "compute_effective_ms",
                    ),
                "median_runtime_migrations":
                    _safe_median(
                        group,
                        "runtime_migrations",
                    ),
                "max_forced_recovery":
                    float(
                        pd.to_numeric(
                            group.get(
                                "forced_stream_recovery",
                                pd.Series(
                                    [0]
                                ),
                            ),
                            errors="coerce",
                        )
                        .fillna(0)
                        .max()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )
    Path(
        args.csv
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary.to_csv(
        args.csv,
        index=False,
    )

    paired = frame.pivot_table(
        index=[
            "sample_index",
            "repeat",
        ],
        columns=
            "strategy",
        values=
            "ttft_ms",
        aggfunc=
            "first",
    )

    paired_metrics = {}
    if {
        "local_full",
        "sparkv",
    }.issubset(
        paired.columns
    ):
        values = (
            paired[
                "local_full"
            ]
            / paired[
                "sparkv"
            ]
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        paired_metrics[
            "median_local_over_sparkv"
        ] = float(
            values.median()
        )

    if {
        "all_stream",
        "sparkv",
    }.issubset(
        paired.columns
    ):
        values = (
            paired[
                "all_stream"
            ]
            / paired[
                "sparkv"
            ]
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        paired_metrics[
            "median_all_stream_over_sparkv"
        ] = float(
            values.median()
        )

    warnings = []

    sparkv = frame[
        frame[
            "strategy"
        ]
        == "sparkv"
    ]
    if not sparkv.empty:
        rebuild = _safe_median(
            sparkv,
            "actual_context_rebuild_ms",
        )
        predicted = _safe_median(
            sparkv,
            "schedule_predicted_makespan_ms",
        )
        decode = _safe_median(
            sparkv,
            "decode_ms",
        )
        wait = _safe_median(
            sparkv,
            "physical_dependency_wait_ms",
        )

        scheduler_gap = _ratio(
            rebuild,
            predicted,
        )
        decode_share = _ratio(
            decode,
            rebuild,
        )
        wait_share = _ratio(
            wait,
            rebuild,
        )

        paired_metrics[
            "median_scheduler_gap_ratio"
        ] = scheduler_gap
        paired_metrics[
            "median_decode_share_of_rebuild"
        ] = decode_share
        paired_metrics[
            "median_dependency_wait_share_of_rebuild"
        ] = wait_share

        forced = pd.to_numeric(
            sparkv.get(
                "forced_stream_recovery",
                pd.Series(
                    [0]
                ),
            ),
            errors="coerce",
        ).fillna(0)
        if (
            forced.max()
            > 0
        ):
            warnings.append(
                "forced_stream_recovery > 0: "
                "the physical executor did not "
                "finish the planned local routes "
                "without recovery."
            )

        if (
            np.isfinite(
                scheduler_gap
            )
            and scheduler_gap
            > 1.25
        ):
            warnings.append(
                "actual context rebuild is more "
                "than 25% above scheduler-predicted "
                "makespan; do not present predicted "
                "and physical TTFT as equivalent."
            )

        if (
            np.isfinite(
                decode_share
            )
            and decode_share
            > 0.50
        ):
            warnings.append(
                "Huffman/dequantization decode "
                "exceeds 50% of context rebuild; "
                "prototype codec overhead dominates "
                "the measured wall-clock result."
            )

        if (
            np.isfinite(
                wait_share
            )
            and wait_share
            > 0.25
        ):
            warnings.append(
                "physical dependency waiting exceeds "
                "25% of context rebuild; head-level "
                "scheduler readiness and full-layer "
                "execution are materially separated."
            )

    sample_count = int(
        frame[
            "sample_index"
        ].nunique()
    )
    repeat_count = int(
        frame[
            "repeat"
        ].nunique()
    )
    if sample_count < 4:
        warnings.append(
            "fewer than 4 samples: treat this as "
            "a smoke/preliminary evaluation."
        )
    if repeat_count < 3:
        warnings.append(
            "fewer than 3 timing repeats: variance "
            "is not well characterized."
        )

    predictor_metrics = None
    if args.predictor:
        payload = torch.load(
            args.predictor,
            map_location="cpu",
            weights_only=False,
        )
        predictor_metrics = payload.get(
            "metrics",
            None,
        )
        if predictor_metrics:
            test_mape = float(
                predictor_metrics.get(
                    "test_mape",
                    float("nan"),
                )
            )
            if (
                np.isfinite(
                    test_mape
                )
                and test_mape
                > 1.0
            ):
                warnings.append(
                    "predictor test MAPE > 100%; "
                    "scheduler cost prediction is "
                    "not yet reliable."
                )

    quality = (
        frame[
            frame[
                "repeat"
            ]
            == 0
        ]
        .groupby(
            "strategy"
        )[
            "f1"
        ]
        .mean()
    )
    if (
        "local_full"
        in quality.index
        and "sparkv"
        in quality.index
    ):
        paired_metrics[
            "mean_f1_delta_sparkv_minus_local"
        ] = float(
            quality[
                "sparkv"
            ]
            - quality[
                "local_full"
            ]
        )

    lines = [
        "# SparKV lab evaluation",
        "",
        "## Aggregate",
        "",
        summary.to_markdown(
            index=False
        ),
        "",
        "## Paired diagnostics",
        "",
    ]
    for key, value in (
        paired_metrics.items()
    ):
        lines.append(
            f"- {key}: "
            f"{value:.6g}"
        )

    if predictor_metrics:
        lines.extend(
            [
                "",
                "## Predictor",
                "",
            ]
        )
        for key, value in (
            predictor_metrics.items()
        ):
            lines.append(
                f"- {key}: {value}"
            )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
        ]
    )
    if warnings:
        lines.extend(
            f"- WARNING: {message}"
            for message in warnings
        )
    else:
        lines.append(
            "- No automatic guardrail warning."
        )

    lines.extend(
        [
            "",
            "Speedup convention: baseline TTFT / "
            "SparKV TTFT. Values > 1 mean SparKV "
            "is faster than that baseline.",
            "",
            "The scheduler-predicted makespan and "
            "the physical prototype wall-clock TTFT "
            "are reported separately by design.",
        ]
    )

    report = "\n".join(
        lines
    ) + "\n"
    Path(
        args.report
    ).write_text(
        report,
        encoding="utf-8",
    )

    print(
        summary.to_string(
            index=False
        )
    )
    print()
    print(report)


if __name__ == "__main__":
    main()
