"""Research inventory. No model loading, server changes, or job submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = {
    "crosskv-figure2": {
        "scope": "baseline", "source": "baseline/crosskv",
        "seraph": "run/seraph/baseline/crosskv_figure2.sh",
        "lab": "run/lab/baseline/crosskv_figure2.sh",
        "status": "runners implemented; actual server smoke not yet verified",
        "method": "2026-10-07 matched-head, single-source OLS R2",
    },
    "lora-qkv": {
        "scope": "baseline", "source": "baseline/lora_exp",
        "seraph": "run/seraph/baseline/lora_qkv.sh",
        "lab": None,
        "status": "Seraph legacy config only; lab paths unverified",
        "method": "same-base heterogeneous LoRA representation analysis",
    },
    "sparkv": {
        "scope": "baseline", "source": "baseline/sparkv",
        "seraph": None, "lab": None,
        "status": "existing launchers assert exp/sparkv-test; no portable wrapper",
        "method": "SparKV reproduction",
    },
    "semcache": {
        "scope": "baseline", "source": "baseline/GP_semcache",
        "seraph": None, "lab": None,
        "status": "external Git submodule; runtime unverified",
        "method": "SemCache baseline",
    },
    "proposed": {
        "scope": "proposed", "source": "proposed",
        "seraph": None, "lab": None,
        "status": "method not yet implemented; package boundary only",
        "method": "future original research",
    },
}


def inventory(root: Path = ROOT) -> list[dict]:
    return [
        {
            "name": name, **info,
            "source_exists": (root / info["source"]).exists(),
            "seraph_exists": (
                (root / info["seraph"]).is_file() if info["seraph"] else False
            ),
            "lab_exists": (
                (root / info["lab"]).is_file() if info["lab"] else False
            ),
        }
        for name, info in EXPERIMENTS.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = inventory()
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if args.check:
        missing = [
            row["name"] for row in rows
            if (not row["source_exists"] and row["name"] != "semcache")
            or (row["seraph"] and not row["seraph_exists"])
            or (row["lab"] and not row["lab_exists"])
        ]
        if missing:
            raise SystemExit("Missing research paths: " + ", ".join(missing))


if __name__ == "__main__":
    main()
