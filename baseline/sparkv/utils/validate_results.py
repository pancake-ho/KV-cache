from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
    )
    parser.add_argument(
        "--allow-recovery",
        action="store_true",
    )
    args = parser.parse_args()

    rows = []
    with Path(
        args.result
    ).open(
        encoding="utf-8"
    ) as handle:
        rows = [
            json.loads(line)
            for line in handle
            if line.strip()
        ]

    if not rows:
        raise SystemExit(
            "no SparKV results"
        )

    failures = []

    for row in rows:
        if row.get(
            "runtime_device"
        ) != "cuda:0":
            failures.append(
                "non-CUDA runtime"
            )

        ttft = float(
            row[
                "ttft_ms"
            ]
        )
        if (
            not math.isfinite(
                ttft
            )
            or ttft <= 0
        ):
            failures.append(
                "invalid TTFT"
            )

        if not row.get(
            "actual_huffman_bitstream",
            False,
        ):
            failures.append(
                "Huffman bitstream disabled"
            )

        if (
            "official Ampere CUDA binding"
            not in str(
                row.get(
                    "spargeattention",
                    "",
                )
            )
        ):
            failures.append(
                "SpargeAttention path not active"
            )

        total_units = int(
            row.get(
                "local_units",
                0,
            )
        ) + int(
            row.get(
                "streamed_units",
                0,
            )
        )
        if total_units <= 0:
            failures.append(
                "no KV units processed"
            )

        if (
            not args.allow_recovery
            and int(
                row.get(
                    "forced_stream_recovery",
                    0,
                )
            )
            != 0
        ):
            failures.append(
                "forced stream recovery occurred"
            )

        if (
            int(
                row.get(
                    "repeat",
                    0,
                )
            )
            == 0
            and row.get(
                "f1"
            )
            is None
        ):
            failures.append(
                "missing quality metric"
            )

    if failures:
        unique = sorted(
            set(
                failures
            )
        )
        raise SystemExit(
            "SparKV validation failed: "
            + ", ".join(
                unique
            )
        )

    print(
        "SparKV direct validation passed "
        f"for {len(rows)} run(s)."
    )


if __name__ == "__main__":
    main()
