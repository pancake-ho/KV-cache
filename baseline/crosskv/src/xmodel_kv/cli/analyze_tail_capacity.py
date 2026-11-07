from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_paired_transfer import paired_summary
from .analyze_summary_transfer_variants import load_protocol_rows


CAPS = (4, 8, 16)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired generated-Tail capacity diagnostic."
    )
    for cap in CAPS:
        parser.add_argument(f"--cap-{cap}", required=True)
    parser.add_argument(
        "--protocol",
        choices=("state_readout", "question_conditioned"),
        default="question_conditioned",
    )
    parser.add_argument("--expected-cases", type=int, default=150)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {cap: getattr(args, f"cap_{cap}") for cap in CAPS}
    result = analyze_tail_capacity(
        paths,
        protocol=args.protocol,
        expected_cases=args.expected_cases,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def analyze_tail_capacity(
    paths: dict[int, str | Path],
    *,
    protocol: str,
    expected_cases: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if set(paths) != set(CAPS):
        raise ValueError(f"expected paths for caps {CAPS}")
    rows = {
        cap: load_protocol_rows(
            [path], protocol=protocol, expected_cases=expected_cases
        )
        for cap, path in paths.items()
    }
    _require_paired(rows)
    absolute = {}
    for cap, table in rows.items():
        values = list(table.values())
        if any(row["generated_tail_tokens"] > cap for row in values):
            raise ValueError(f"generated tail exceeds cap {cap}")
        absolute[str(cap)] = {
            "mean_f1": float(np.mean([row["generated_tail_f1"] for row in values])),
            "accuracy": float(np.mean([row["generated_tail_em"] for row in values])),
            "mean_realized_tokens": float(
                np.mean([row["generated_tail_tokens"] for row in values])
            ),
            "mean_packet_bytes": float(
                np.mean([row["generated_tail_packet_bytes"] for row in values])
            ),
        }
    comparisons = {
        label: _compare(
            rows[candidate],
            rows[reference],
            ids=sorted(rows[candidate]),
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        for label, candidate, reference in (
            ("cap8_minus_cap4", 8, 4),
            ("cap16_minus_cap8", 16, 8),
            ("cap16_minus_cap4", 16, 4),
        )
    }
    strata = {}
    ids_by_stratum = _answer_length_strata(rows[4])
    for stratum, ids in ids_by_stratum.items():
        strata[stratum] = {
            "cases": len(ids),
            "cap16_minus_cap4": _compare(
                rows[16],
                rows[4],
                ids=ids,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            ),
        }
    interaction = _stratum_interaction(
        rows[16],
        rows[4],
        long_ids=ids_by_stratum["9_plus_words"],
        short_ids=ids_by_stratum["1_to_4_words"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    global_comparison = comparisons["cap16_minus_cap4"]
    long_comparison = strata["9_plus_words"]["cap16_minus_cap4"]
    gates = {
        "global_capacity_support": (
            global_comparison["mean_f1_delta"] > 0
            and global_comparison["mean_f1_delta_bootstrap_95ci"][0] > 0
        ),
        "long_answer_support": (
            long_comparison["mean_f1_delta"] > 0
            and long_comparison["mean_f1_delta_bootstrap_95ci"][0] > 0
        ),
        "length_interaction": (
            interaction["mean"] > 0 and interaction["bootstrap_95ci"][0] > 0
        ),
    }
    return {
        "analysis": "generated_tail_capacity_diagnostic",
        "protocol": protocol,
        "cases": expected_cases,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "paths": {str(cap): str(path) for cap, path in paths.items()},
        "absolute": absolute,
        "comparisons": comparisons,
        "answer_length_strata": strata,
        "long_minus_short_interaction": interaction,
        "gates": gates,
    }


def _compare(
    candidate: dict[str, dict],
    reference: dict[str, dict],
    *,
    ids: list[str],
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if not ids:
        raise ValueError("capacity comparison stratum is empty")
    return paired_summary(
        [candidate[case_id]["generated_tail_em"] for case_id in ids],
        [reference[case_id]["generated_tail_em"] for case_id in ids],
        [candidate[case_id]["generated_tail_f1"] for case_id in ids],
        [reference[case_id]["generated_tail_f1"] for case_id in ids],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )


def _answer_length_strata(table: dict[str, dict]) -> dict[str, list[str]]:
    output = {
        "1_to_4_words": [],
        "5_to_8_words": [],
        "9_plus_words": [],
    }
    for case_id, row in table.items():
        words = row["gold_answers"][0].split()
        if len(words) <= 4:
            output["1_to_4_words"].append(case_id)
        elif len(words) <= 8:
            output["5_to_8_words"].append(case_id)
        else:
            output["9_plus_words"].append(case_id)
    return {name: sorted(ids) for name, ids in output.items()}


def _stratum_interaction(
    candidate: dict[str, dict],
    reference: dict[str, dict],
    *,
    long_ids: list[str],
    short_ids: list[str],
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    long_values = np.asarray(
        [
            candidate[case_id]["generated_tail_f1"]
            - reference[case_id]["generated_tail_f1"]
            for case_id in long_ids
        ],
        dtype=np.float64,
    )
    short_values = np.asarray(
        [
            candidate[case_id]["generated_tail_f1"]
            - reference[case_id]["generated_tail_f1"]
            for case_id in short_ids
        ],
        dtype=np.float64,
    )
    if not long_values.size or not short_values.size:
        raise ValueError("long and short answer strata must be non-empty")
    generator = np.random.default_rng(seed)
    long_indices = generator.integers(
        0, len(long_values), size=(bootstrap_replicates, len(long_values))
    )
    short_indices = generator.integers(
        0, len(short_values), size=(bootstrap_replicates, len(short_values))
    )
    samples = long_values[long_indices].mean(axis=1) - short_values[
        short_indices
    ].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(long_values.mean() - short_values.mean()),
        "bootstrap_95ci": [float(low), float(high)],
        "long_cases": int(len(long_values)),
        "short_cases": int(len(short_values)),
    }


def _require_paired(rows: dict[int, dict[str, dict]]) -> None:
    ids = set(rows[CAPS[0]])
    for cap in CAPS[1:]:
        if set(rows[cap]) != ids:
            raise ValueError(f"result IDs differ for cap {cap}")
    for case_id in ids:
        first = rows[CAPS[0]][case_id]
        for cap in CAPS[1:]:
            current = rows[cap][case_id]
            for field in ("question", "gold_answers"):
                if current.get(field) != first.get(field):
                    raise ValueError(f"{field} differs for cap {cap}, case {case_id}")


if __name__ == "__main__":
    main()
