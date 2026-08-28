import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def p95(values):
    return float(np.percentile(values, 95))


parser = argparse.ArgumentParser()
parser.add_argument("inputs", nargs="+")
parser.add_argument("--output", required=True)
args = parser.parse_args()

rows = []
for name in args.inputs:
    with Path(name).open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())

frame = pd.DataFrame(rows)
summary = (
    frame.groupby(["format", "strategy", "bandwidth_mbps", "jitter_cv"], dropna=False)
    .agg(
        runs=("ttft_ms", "size"),
        median_ttft_ms=("ttft_ms", "median"),
        p95_ttft_ms=("ttft_ms", p95),
        mean_f1=("f1", "mean"),
        median_energy_j=("ttft_energy_j", "median"),
        peak_vram_mib=("peak_vram_mib", "max"),
        mean_wire_mib=("wire_bytes", lambda x: float(np.mean(x) / 2**20)),
    )
    .reset_index()
    .sort_values(["bandwidth_mbps", "format", "median_ttft_ms"])
)
summary.to_csv(args.output, index=False)
print(summary.to_string(index=False))

