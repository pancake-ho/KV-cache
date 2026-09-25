from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def p95(values) -> float:
    return float(
        np.percentile(
            values,
            95,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()

    rows = []
    for name in args.inputs:
        with Path(name).open(
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

    latency = (
        frame.groupby(
            [
                "strategy",
                "bandwidth_mbps",
                "jitter_cv",
            ],
            dropna=False,
        )
        .agg(
            runs=(
                "ttft_ms",
                "size",
            ),
            median_ttft_ms=(
                "ttft_ms",
                "median",
            ),
            p95_ttft_ms=(
                "ttft_ms",
                p95,
            ),
            median_context_rebuild_ms=(
                "actual_context_rebuild_ms",
                "median",
            ),
            median_energy_j=(
                "ttft_energy_j",
                "median",
            ),
            peak_vram_mib=(
                "peak_vram_mib",
                "max",
            ),
            mean_wire_mib=(
                "wire_bytes",
                lambda x: float(
                    np.mean(x)
                    / 2**20
                ),
            ),
            mean_runtime_migrations=(
                "runtime_migrations",
                "mean",
            ),
            max_forced_recovery=(
                "forced_stream_recovery",
                "max",
            ),
        )
        .reset_index()
    )

    quality = (
        frame[
            frame[
                "repeat"
            ]
            == 0
        ]
        .groupby(
            [
                "strategy",
                "bandwidth_mbps",
                "jitter_cv",
            ],
            dropna=False,
        )
        .agg(
            quality_runs=(
                "f1",
                "size",
            ),
            mean_f1=(
                "f1",
                "mean",
            ),
        )
        .reset_index()
    )

    result = (
        latency.merge(
            quality,
            on=[
                "strategy",
                "bandwidth_mbps",
                "jitter_cv",
            ],
            how="left",
        )
    )

    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        output,
        index=False,
    )

    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
