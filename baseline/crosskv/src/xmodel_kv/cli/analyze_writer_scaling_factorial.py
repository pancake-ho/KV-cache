from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_paired_transfer import _load_rows, paired_summary


LAYERS = ("all", "last_16")
QUANT_BITS = (16, 4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen N=128/N=512 writer layer-by-quant factorial."
    )
    parser.add_argument(
        "--n128-results", "--reference-results", dest="reference_results", required=True
    )
    parser.add_argument(
        "--n512-results", "--candidate-results", dest="candidate_results", required=True
    )
    parser.add_argument("--reference-name", default="n128")
    parser.add_argument("--candidate-name", default="n512")
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")

    analysis = analyze_factorial(
        load_factorial(args.reference_results),
        load_factorial(args.candidate_results),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
    )
    analysis["inputs"] = {
        args.reference_name: args.reference_results,
        args.candidate_name: args.candidate_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


def load_factorial(path: str | Path) -> dict[tuple[str, int], dict[str, dict]]:
    cells = {
        (layer, quant): _load_rows(path, quant, layer_pattern=layer)
        for layer in LAYERS
        for quant in QUANT_BITS
    }
    expected_ids = set(next(iter(cells.values())))
    if any(set(rows) != expected_ids for rows in cells.values()):
        raise ValueError(f"factorial cells in {path} have different result IDs")
    return cells


def analyze_factorial(
    n128: dict[tuple[str, int], dict[str, dict]],
    n512: dict[tuple[str, int], dict[str, dict]],
    *,
    bootstrap_replicates: int,
    seed: int,
    reference_name: str = "n128",
    candidate_name: str = "n512",
) -> dict:
    if not reference_name or not candidate_name or reference_name == candidate_name:
        raise ValueError("writer names must be distinct and non-empty")
    if set(n128) != set(n512):
        raise ValueError("writer factorial cells differ")
    for cell in n128:
        if set(n128[cell]) != set(n512[cell]):
            raise ValueError(f"writer result IDs differ in cell {cell}")

    writers = {reference_name: n128, candidate_name: n512}
    cells = {
        writer: {
            cell_name(layer, quant): summarize_cell(rows)
            for (layer, quant), rows in table.items()
        }
        for writer, table in writers.items()
    }
    scaling = {
        cell_name(layer, quant): compare_cells(
            n512[(layer, quant)],
            n128[(layer, quant)],
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        for layer in LAYERS
        for quant in QUANT_BITS
    }
    within_writer = {}
    for writer, table in writers.items():
        contrasts = {}
        for layer in LAYERS:
            contrasts[f"{layer}_int4_vs_bf16"] = compare_cells(
                table[(layer, 4)],
                table[(layer, 16)],
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
        for quant in QUANT_BITS:
            contrasts[f"last_16_vs_all_{precision_name(quant)}"] = compare_cells(
                table[("last_16", quant)],
                table[("all", quant)],
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
        within_writer[writer] = contrasts
    return {
        "cases": len(next(iter(n128.values()))),
        "cells": cells,
        f"{candidate_name}_vs_{reference_name}": scaling,
        "within_writer": within_writer,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }


def summarize_cell(rows: dict[str, dict]) -> dict:
    values = list(rows.values())

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in values) / len(values)

    summary = {
        "cases": len(values),
        "accuracy": mean("student_exact"),
        "mean_f1": mean("student_f1"),
        "bridge_accuracy": mean("bridge_exact"),
        "bridge_mean_f1": mean("bridge_f1"),
        "no_summary_accuracy": mean("no_summary_exact"),
        "no_summary_mean_f1": mean("no_summary_f1"),
    }
    for key in ("active_layers", "wire_payload_bytes"):
        if all(key in row for row in values):
            summary[key] = mean(key)
    return summary


def compare_cells(
    candidate: dict[str, dict],
    reference: dict[str, dict],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if set(candidate) != set(reference):
        raise ValueError("paired cell IDs differ")
    case_ids = sorted(candidate)
    return paired_summary(
        [candidate[case_id]["student_exact"] for case_id in case_ids],
        [reference[case_id]["student_exact"] for case_id in case_ids],
        [candidate[case_id]["student_f1"] for case_id in case_ids],
        [reference[case_id]["student_f1"] for case_id in case_ids],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )


def cell_name(layer: str, quant_bits: int) -> str:
    return f"{layer}_{precision_name(quant_bits)}"


def precision_name(quant_bits: int) -> str:
    return "bf16" if quant_bits == 16 else f"int{quant_bits}"


if __name__ == "__main__":
    main()
