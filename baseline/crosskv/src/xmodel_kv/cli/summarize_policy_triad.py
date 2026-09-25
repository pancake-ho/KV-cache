from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict summary for policy-triad JSONL")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [row for path in args.inputs for row in _jsonl(Path(path))]
    identifiers = [row["triad_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        duplicates = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        raise SystemExit(f"duplicate triad ids: {duplicates[:10]}")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(rows)}")

    fine_groups = defaultdict(list)
    aggregate_groups = defaultdict(list)
    for row in rows:
        triad = row["triad"]
        fine_groups[
            (triad["family"], triad["direction"], triad["arm"], triad["layout"])
        ].append(row)
        aggregate_groups[(triad["family"], triad["arm"], triad["layout"])].append(row)
    summary = {
        "rows": len(rows),
        "unique_rows": len(set(identifiers)),
        "chance_levels": sorted({row["triad"]["chance_level"] for row in rows}),
        "fine_cells": [
            _summarize(key, group, include_direction=True)
            for key, group in sorted(fine_groups.items())
        ],
        "aggregate_cells": [
            _summarize(key, group, include_direction=False)
            for key, group in sorted(aggregate_groups.items())
        ],
        "paired_controls": _paired_controls(rows),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(summary["aggregate_cells"], ensure_ascii=False, indent=2))
    print({"rows": len(rows), "output": str(output_path)})


def _summarize(key, rows, *, include_direction: bool):
    if include_direction:
        family, direction, arm, layout = key
    else:
        family, arm, layout = key
        direction = None
    eligible = [
        row
        for row in rows
        if row["triad_validity"]["target_policy_correct"]
        and row["triad_validity"]["source_policy_correct"]
    ]
    result = {
        "family": family,
        "arm": arm,
        "layout": layout,
        "directions": sorted({row["triad"]["direction"] for row in rows}),
        "rows": len(rows),
        "eligible": len(eligible),
        "eligible_rate": _rate(len(eligible), len(rows)),
        "prefix_shift_min": min(row["triad"]["actual_prefix_shift"] for row in rows),
        "prefix_shift_max": max(row["triad"]["actual_prefix_shift"] for row in rows),
        "identity_target_agreement": sum(
            row["identity"]["target_native_agreement"] for row in rows
        ),
    }
    if direction is not None:
        result["direction"] = direction
    if not eligible:
        return result
    counts = {
        "hybrid_matches_target": sum(
            row["triad_outcome"]["hybrid_matches_target_native"] for row in eligible
        ),
        "hybrid_matches_source": sum(
            row["triad_outcome"]["hybrid_matches_source_native"] for row in eligible
        ),
        "hybrid_matches_third": sum(
            row["triad_outcome"]["hybrid_matches_third_policy"] for row in eligible
        ),
        "full_replay_matches_target": sum(
            row["hybrid"][-1]["target_native_agreement"] for row in eligible
        ),
        "positive_source_leakage": sum(
            row["hybrid"][0]["source_leakage_shift"] > 0 for row in eligible
        ),
    }
    for name, count in counts.items():
        result[name] = count
        result[f"{name}_rate"] = _rate(count, len(eligible))
    result["median_mean_kl"] = statistics.median(
        row["hybrid"][0]["distribution"]["mean_kl"] for row in eligible
    )
    result["median_source_leakage"] = statistics.median(
        row["hybrid"][0]["source_leakage_shift"] for row in eligible
    )
    return result


def _rate(successes: int, trials: int):
    if trials == 0:
        return None
    lower, upper = _wilson(successes, trials)
    return {
        "value": successes / trials,
        "wilson_95_low": lower,
        "wilson_95_high": upper,
    }


def _wilson(successes: int, trials: int, z: float = 1.959963984540054):
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _paired_controls(rows):
    groups = defaultdict(dict)
    for row in rows:
        triad = row["triad"]
        key = (
            triad["family"],
            triad["direction"],
            triad["layout"],
            triad["seed"],
            triad["replicate"],
        )
        arm = triad["arm"]
        if arm in groups[key]:
            raise ValueError(f"duplicate arm {arm!r} for paired triad {key}")
        groups[key][arm] = row

    cells = defaultdict(list)
    for key, arms in groups.items():
        if set(arms) != {"null", "matched", "third"}:
            continue
        family, direction, layout, _, _ = key
        cells[(family, direction, layout)].append(arms)

    results = []
    for (family, direction, layout), arm_groups in sorted(cells.items()):
        eligible = [
            arms
            for arms in arm_groups
            if all(
                row["triad_validity"]["target_policy_correct"]
                and row["triad_validity"]["source_policy_correct"]
                for row in arms.values()
            )
        ]
        result = {
            "family": family,
            "direction": direction,
            "layout": layout,
            "paired_rows": len(arm_groups),
            "eligible": len(eligible),
            "eligible_rate": _rate(len(eligible), len(arm_groups)),
        }
        if eligible:
            result["third_vs_null_on_third_action"] = _paired_binary(
                [
                    (
                        arms["third"]["triad_outcome"][
                            "hybrid_matches_third_policy"
                        ],
                        arms["null"]["triad_outcome"][
                            "hybrid_matches_third_policy"
                        ],
                    )
                    for arms in eligible
                ],
                treatment_name="third_arm",
                control_name="null_arm",
            )
            result["matched_vs_third_on_target_action"] = _paired_binary(
                [
                    (
                        arms["matched"]["triad_outcome"][
                            "hybrid_matches_target_native"
                        ],
                        arms["third"]["triad_outcome"][
                            "hybrid_matches_target_native"
                        ],
                    )
                    for arms in eligible
                ],
                treatment_name="matched_arm",
                control_name="third_arm",
            )
        results.append(result)
    return results


def _paired_binary(pairs, *, treatment_name: str, control_name: str):
    treatment_successes = sum(treatment for treatment, _ in pairs)
    control_successes = sum(control for _, control in pairs)
    treatment_only = sum(treatment and not control for treatment, control in pairs)
    control_only = sum(control and not treatment for treatment, control in pairs)
    return {
        f"{treatment_name}_successes": treatment_successes,
        f"{control_name}_successes": control_successes,
        "paired_effect": (treatment_successes - control_successes) / len(pairs),
        f"{treatment_name}_only": treatment_only,
        f"{control_name}_only": control_only,
        "discordant_pairs": treatment_only + control_only,
        "exact_mcnemar_two_sided_p": _exact_mcnemar(treatment_only, control_only),
    }


def _exact_mcnemar(first_only: int, second_only: int) -> float:
    if first_only < 0 or second_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, successes)
        for successes in range(min(first_only, second_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def _jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
