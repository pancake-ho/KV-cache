from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_orbit_distillation import RUN_NAMES
from .analyze_paired_transfer import _load_rows, paired_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Four-arm in-domain characterization of orbit distillation."
    )
    for name in RUN_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--orbit-canonical-shifted")
    parser.add_argument("--expected-cases", type=int, default=80)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.expected_cases, args.bootstrap_replicates) < 1:
        parser.error("case and bootstrap counts must be positive")
    paths = {name: getattr(args, name) for name in RUN_NAMES}
    result = analyze_musique_orbit(
        paths,
        expected_cases=args.expected_cases,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        shifted_path=args.orbit_canonical_shifted,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def analyze_musique_orbit(
    paths: dict[str, str | Path],
    *,
    expected_cases: int,
    bootstrap_replicates: int,
    seed: int,
    shifted_path: str | Path | None = None,
) -> dict:
    if set(paths) != set(RUN_NAMES):
        raise ValueError(f"expected run paths {RUN_NAMES}")
    rows = {
        name: _load_rows(
            _results_path(path), quant_bits=4, layer_pattern="first_16"
        )
        for name, path in paths.items()
    }
    if any(len(table) != expected_cases for table in rows.values()):
        raise ValueError("unexpected case count in four-arm characterization")
    _require_paired(rows)
    shifted = None
    if shifted_path is not None:
        shifted = _load_rows(
            _results_path(shifted_path), quant_bits=4, layer_pattern="first_16"
        )
        if len(shifted) != expected_cases:
            raise ValueError("unexpected shifted-source case count")
        _require_shift_pair(rows["orbit_canonical"], shifted)
    endpoints = {
        endpoint: _analyze_endpoint(
            rows,
            endpoint=endpoint,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            shifted=shifted,
        )
        for endpoint in ("student", "bridge")
    }
    wire = _audit_wire(rows)
    return {
        "analysis": "positional_orbit_musique_characterization",
        "cases": expected_cases,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "paths": {name: str(path) for name, path in paths.items()},
        "orbit_canonical_shifted": (
            None if shifted_path is None else str(shifted_path)
        ),
        "endpoints": endpoints,
        "wire_audit": wire,
    }


def _analyze_endpoint(
    rows: dict[str, dict[str, dict]],
    *,
    endpoint: str,
    bootstrap_replicates: int,
    seed: int,
    shifted: dict[str, dict] | None,
) -> dict:
    comparisons = {}
    pairs = {
        "target_retention_Ot_minus_Pt": ("orbit_target", "parent_target"),
        "canonical_safety_Oc_minus_Ot": ("orbit_canonical", "orbit_target"),
        "parent_canonical_Pc_minus_Pt": ("parent_canonical", "parent_target"),
        "canonical_candidate_Oc_minus_Pc": (
            "orbit_canonical",
            "parent_canonical",
        ),
    }
    for label, (candidate, reference) in pairs.items():
        comparisons[label] = _compare(
            rows[candidate],
            rows[reference],
            endpoint=endpoint,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
    if shifted is not None:
        comparisons["source_causality_Oc_correct_minus_shifted"] = _compare(
            rows["orbit_canonical"],
            shifted,
            endpoint=endpoint,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
    return {
        "absolute": {
            name: {
                "accuracy": _mean(table, f"{endpoint}_exact"),
                "mean_f1": _mean(table, f"{endpoint}_f1"),
            }
            for name, table in rows.items()
        },
        "comparisons": comparisons,
        "difference_in_differences": _did(
            rows,
            endpoint=endpoint,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
    }


def _compare(
    candidate: dict[str, dict],
    reference: dict[str, dict],
    *,
    endpoint: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    ids = sorted(candidate)
    return paired_summary(
        [candidate[case_id][f"{endpoint}_exact"] for case_id in ids],
        [reference[case_id][f"{endpoint}_exact"] for case_id in ids],
        [candidate[case_id][f"{endpoint}_f1"] for case_id in ids],
        [reference[case_id][f"{endpoint}_f1"] for case_id in ids],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )


def _did(
    rows: dict[str, dict[str, dict]],
    *,
    endpoint: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    ids = sorted(rows["parent_target"])
    output = {}
    for offset, metric in enumerate(("f1", "exact")):
        field = f"{endpoint}_{metric}"
        values = np.asarray(
            [
                rows["orbit_canonical"][case_id][field]
                - rows["orbit_target"][case_id][field]
                - rows["parent_canonical"][case_id][field]
                + rows["parent_target"][case_id][field]
                for case_id in ids
            ],
            dtype=np.float64,
        )
        generator = np.random.default_rng(seed + offset)
        indices = generator.integers(
            0, len(values), size=(bootstrap_replicates, len(values))
        )
        low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
        output[metric] = {
            "mean": float(values.mean()),
            "bootstrap_95ci": [float(low), float(high)],
        }
    return output


def _require_paired(rows: dict[str, dict[str, dict]]) -> None:
    ids = set(rows[RUN_NAMES[0]])
    for name in RUN_NAMES[1:]:
        if set(rows[name]) != ids:
            raise ValueError(f"result IDs differ for {name}")
    for case_id in ids:
        first = rows[RUN_NAMES[0]][case_id]
        for name in RUN_NAMES[1:]:
            current = rows[name][case_id]
            for field in ("final_gold", "bridge_gold"):
                if current.get(field) != first.get(field):
                    raise ValueError(f"{field} differs for {name} case {case_id}")


def _audit_wire(rows: dict[str, dict[str, dict]]) -> dict:
    audits = {}
    for name, table in rows.items():
        expected_mode = "canonical" if name.endswith("canonical") else "target"
        values = list(table.values())
        checks = {
            "position_mode": all(
                row.get("packet_position_mode") == expected_mode for row in values
            ),
            "base_bytes": all(row.get("base_packet_bytes") == 67_642 for row in values),
            "delta_bytes": all(row.get("delta_packet_bytes") == 4_252 for row in values),
            "cold_bytes": all(row.get("actual_packet_bytes") == 71_894 for row in values),
            "incremental_bytes": all(
                row.get("incremental_packet_bytes") == 4_252 for row in values
            ),
        }
        if expected_mode == "canonical":
            checks["target_invariant"] = all(
                row.get("packet_target_invariant") is True for row in values
            )
        audits[name] = {"checks": checks, "pass": all(checks.values())}
    return {"runs": audits, "pass": all(run["pass"] for run in audits.values())}


def _require_shift_pair(
    correct: dict[str, dict], shifted: dict[str, dict]
) -> None:
    if set(correct) != set(shifted):
        raise ValueError("correct and shifted result IDs differ")
    for case_id in correct:
        for field in ("final_gold", "bridge_gold"):
            if correct[case_id].get(field) != shifted[case_id].get(field):
                raise ValueError(f"shifted {field} differs for case {case_id}")
        if correct[case_id].get("source_id") == shifted[case_id].get("source_id"):
            raise ValueError(f"shift did not change source for case {case_id}")


def _mean(rows: dict[str, dict], field: str) -> float:
    return float(np.mean([row[field] for row in rows.values()]))


def _results_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "results.jsonl" if path.is_dir() else path


if __name__ == "__main__":
    main()
