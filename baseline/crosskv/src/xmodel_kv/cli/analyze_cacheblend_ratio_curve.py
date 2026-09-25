from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .analyze_cacheblend_recovery import normalize_answer
from .analyze_hotpot_competitors import load_unique_rows
from .analyze_paired_transfer import paired_summary


RATIO_INPUTS = (("ratio_015", 0.15), ("ratio_050", 0.50), ("ratio_100", 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a fixed CacheBlend ratio curve.")
    parser.add_argument("--cold-results", nargs="+", required=True)
    for name, _ in RATIO_INPUTS:
        parser.add_argument(f"--{name.replace('_', '-')}-results", nargs="+", required=True)
    parser.add_argument("--expected-cases", type=int, default=290)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.expected_cases, args.bootstrap_replicates) < 1:
        parser.error("case and bootstrap counts must be positive")

    cold = load_unique_rows(args.cold_results, name="CacheBlend cold")
    ratio_rows = {
        name: load_unique_rows(getattr(args, f"{name}_results"), name=name)
        for name, _ in RATIO_INPUTS
    }
    joined = join_curve(cold, ratio_rows, expected_cases=args.expected_cases)
    analysis = analyze_curve(
        joined,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    analysis["inputs"] = {
        "cold": args.cold_results,
        **{
            name: getattr(args, f"{name}_results")
            for name, _ in RATIO_INPUTS
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


def join_curve(cold, ratio_rows, *, expected_cases: int) -> list[dict]:
    expected_ids = set(cold)
    if len(expected_ids) != expected_cases:
        raise ValueError("cold input does not contain the expected cases")
    if any(set(rows) != expected_ids for rows in ratio_rows.values()):
        raise ValueError("CacheBlend curve ID sets differ")
    joined = []
    expected_ratios = dict(RATIO_INPUTS)
    for case_id in sorted(cold, key=lambda value: int(cold[value]["index"])):
        root = cold[case_id]
        row = {"cold": root}
        for name, rows in ratio_rows.items():
            candidate = rows[case_id]
            for key in (
                "index",
                "question",
                "gold_answers",
                "prompt_tokens",
                "document_tokens",
                "document_kv_bf16_bytes",
            ):
                if candidate[key] != root[key]:
                    raise ValueError(f"cold/{name} mismatch for {case_id}: {key}")
            recorded_ratio = candidate.get("recompute_ratio")
            if recorded_ratio is not None and float(recorded_ratio) != expected_ratios[name]:
                raise ValueError(f"unexpected recompute ratio in {name}")
            if not 0 < int(candidate["cacheblend_cached_tokens"]) < int(
                candidate["prompt_tokens"]
            ):
                raise ValueError(f"invalid partial cache hit in {name}: {case_id}")
            row[name] = candidate
        joined.append(row)
    return joined


def analyze_curve(rows, *, bootstrap_replicates: int, seed: int) -> dict:
    def mean(arm: str, key: str) -> float:
        return sum(float(row[arm][key]) for row in rows) / len(rows)

    quality = {
        "cold": {
            "em": mean("cold", "cold_full_em"),
            "f1": mean("cold", "cold_full_f1"),
            "wall_ms": mean("cold", "cold_full_wall_ms"),
            "ttft_ms": mean("cold", "cold_full_ttft_ms"),
        }
    }
    comparisons = {}
    output_stability = {}
    for name, ratio in RATIO_INPUTS:
        quality[name] = {
            "recompute_ratio": ratio,
            "em": mean(name, "cacheblend_em"),
            "f1": mean(name, "cacheblend_f1"),
            "wall_ms": mean(name, "cacheblend_wall_ms"),
            "ttft_ms": mean(name, "cacheblend_ttft_ms"),
        }
        comparisons[f"{name}_vs_cold"] = paired_summary(
            [row[name]["cacheblend_em"] for row in rows],
            [row["cold"]["cold_full_em"] for row in rows],
            [row[name]["cacheblend_f1"] for row in rows],
            [row["cold"]["cold_full_f1"] for row in rows],
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        raw = [
            row[name]["cacheblend"] == row["cold"]["cold_full"] for row in rows
        ]
        normalized = [
            normalize_answer(row[name]["cacheblend"])
            == normalize_answer(row["cold"]["cold_full"])
            for row in rows
        ]
        trigram_counts = [max_ngram_count(row[name]["cacheblend"], n=3) for row in rows]
        output_stability[name] = {
            "raw_agreement_with_cold": sum(raw) / len(rows),
            "normalized_agreement_with_cold": sum(normalized) / len(rows),
            "mean_max_trigram_count": sum(trigram_counts) / len(rows),
            "fraction_max_trigram_at_least_3": (
                sum(count >= 3 for count in trigram_counts) / len(rows)
            ),
        }
    length_strata = {}
    for label, lower, upper in (
        ("le_4096", 0, 4096),
        ("4097_8192", 4097, 8192),
        ("8193_12288", 8193, 12288),
        ("gt_12288", 12289, None),
    ):
        selected = [
            row
            for row in rows
            if int(row["cold"]["prompt_tokens"]) >= lower
            and (upper is None or int(row["cold"]["prompt_tokens"]) <= upper)
        ]
        if not selected:
            continue
        length_strata[label] = {
            "cases": len(selected),
            "mean_prompt_tokens": sum(
                int(row["cold"]["prompt_tokens"]) for row in selected
            )
            / len(selected),
            "cold_f1": sum(row["cold"]["cold_full_f1"] for row in selected)
            / len(selected),
            **{
                f"{name}_f1": sum(row[name]["cacheblend_f1"] for row in selected)
                / len(selected)
                for name, _ in RATIO_INPUTS
            },
        }
    return {
        "cases": len(rows),
        "quality_and_timing": quality,
        "comparisons": comparisons,
        "output_stability": output_stability,
        "length_strata": length_strata,
        "cache": {
            "mean_prompt_tokens": mean("ratio_015", "prompt_tokens"),
            "mean_cached_tokens": mean("ratio_015", "cacheblend_cached_tokens"),
            "mean_cached_fraction": mean("ratio_015", "cacheblend_cached_fraction"),
        },
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }


def max_ngram_count(text: str, *, n: int) -> int:
    tokens = normalize_answer(text).split()
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return max(counts.values(), default=0)


if __name__ == "__main__":
    main()
