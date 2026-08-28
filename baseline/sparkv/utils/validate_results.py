import argparse
import json
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("inputs", nargs="+")
args = parser.parse_args()

rows = []
for name in args.inputs:
    with Path(name).open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())

frame = pd.DataFrame(rows)
raw = frame[(frame["format"] == "raw") & (frame["repeat"] == 0)]
failures = []
for (sample, bandwidth), group in raw.groupby(["sample_index", "bandwidth_mbps"]):
    predictions = group.set_index("strategy")["prediction"].to_dict()
    if "local" not in predictions:
        continue
    reference = predictions["local"]
    for strategy, prediction in predictions.items():
        if prediction != reference:
            failures.append(
                {
                    "sample": int(sample),
                    "bandwidth": float(bandwidth),
                    "strategy": strategy,
                }
            )

if failures:
    print(json.dumps(failures[:20], indent=2))
    raise SystemExit(f"raw-cache equivalence failed in {len(failures)} cases")
print(f"raw-cache prediction equivalence passed for {len(raw)} records")

