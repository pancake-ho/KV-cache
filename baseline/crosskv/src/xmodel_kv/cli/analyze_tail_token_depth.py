from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_summary_transfer_variants import compare_variants, load_protocol_rows
from .analyze_tail_capacity import analyze_tail_capacity


CAPS = (4, 8, 16)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generated Tail-KV token-cap by layer-depth factorial."
    )
    for depth in ("first16", "all"):
        for cap in CAPS:
            parser.add_argument(f"--{depth}-cap{cap}", required=True)
    parser.add_argument("--all-cap24", required=True)
    parser.add_argument(
        "--protocol",
        choices=("question_conditioned", "state_readout"),
        default="question_conditioned",
    )
    parser.add_argument("--expected-cases", type=int, default=150)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    first16 = {cap: getattr(args, f"first16_cap{cap}") for cap in CAPS}
    all32 = {cap: getattr(args, f"all_cap{cap}") for cap in CAPS}
    result = analyze_tail_token_depth(
        first16,
        all32,
        all_cap24=args.all_cap24,
        protocol=args.protocol,
        expected_cases=args.expected_cases,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def analyze_tail_token_depth(
    first16_paths,
    all32_paths,
    *,
    all_cap24,
    protocol: str,
    expected_cases: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    first16 = {
        cap: load_protocol_rows(
            [path], protocol=protocol, expected_cases=expected_cases
        )
        for cap, path in first16_paths.items()
    }
    all32 = {
        cap: load_protocol_rows(
            [path], protocol=protocol, expected_cases=expected_cases
        )
        for cap, path in all32_paths.items()
    }
    cap24 = load_protocol_rows(
        [all_cap24], protocol=protocol, expected_cases=expected_cases
    )
    capacity_all = analyze_tail_capacity(
        all32_paths,
        protocol=protocol,
        expected_cases=expected_cases,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    depth_effects = {
        str(cap): compare_variants(
            all32[cap],
            first16[cap],
            candidate_arm="generated_tail",
            reference_arm="generated_tail",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        for cap in CAPS
    }
    cap8_vs_cap24 = compare_variants(
        all32[8],
        cap24,
        candidate_arm="generated_tail",
        reference_arm="generated_tail",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    cap8_vs_no = compare_variants(
        all32[8],
        all32[8],
        candidate_arm="generated_tail",
        reference_arm="no_summary",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    mean_bytes = lambda table: float(
        np.mean([row["generated_tail_packet_bytes"] for row in table.values()])
    )
    cap8_bytes = mean_bytes(all32[8])
    cap24_bytes = mean_bytes(cap24)
    ratio = cap8_bytes / cap24_bytes
    gates = {
        "cap8_quality_noninferior_to_cap24": (
            cap8_vs_cap24["mean_f1_delta"] >= -0.03
            and cap8_vs_cap24["mean_f1_delta_bootstrap_95ci"][0] > -0.06
        ),
        "cap8_better_than_no_state": (
            cap8_vs_no["mean_f1_delta"] > 0
            and cap8_vs_no["mean_f1_delta_bootstrap_95ci"][0] > 0
        ),
        "cap8_bytes_at_most_60pct": ratio <= 0.60,
    }
    return {
        "analysis": "generated_tail_token_depth_factorial",
        "protocol": protocol,
        "cases": expected_cases,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "capacity_within_all32": capacity_all,
        "all32_minus_first16": depth_effects,
        "cap8_all32_minus_cap24_all32": cap8_vs_cap24,
        "cap8_all32_minus_no_state": cap8_vs_no,
        "bytes": {
            "cap8_all32": cap8_bytes,
            "cap24_all32": cap24_bytes,
            "ratio": ratio,
        },
        "gates": {**gates, "paper_frontier_authorized": all(gates.values())},
    }


if __name__ == "__main__":
    main()
