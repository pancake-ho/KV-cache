from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_paired_transfer import paired_summary


CAPSULE_ARMS = ("no_summary", "capsule", "shifted_capsule", "generated_tail")
TEXT_ARMS = (
    "plaintext_source",
    "plaintext_shifted",
    "plaintext_gold",
    "full_recompute",
)
ALL_ARMS = (*CAPSULE_ARMS, "source_full_kv", *TEXT_ARMS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join compact-KV and plaintext/full-recompute Hotpot baselines."
    )
    parser.add_argument("--capsule-results", required=True)
    parser.add_argument("--baseline-results", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=290)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--capsule-payload-bytes", type=float, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    args = parser.parse_args()
    if min(
        args.expected_cases,
        args.bootstrap_replicates,
        args.capsule_payload_bytes,
        args.num_layers,
        args.num_kv_heads,
        args.head_dim,
    ) < 1:
        parser.error("counts, dimensions, and payload must be positive")

    capsule = load_capsule_rows(args.capsule_results)
    baseline = load_baseline_rows(args.baseline_results)
    rows = join_rows(capsule, baseline, expected_cases=args.expected_cases)
    summary = analyze(
        rows,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        capsule_payload_bytes=args.capsule_payload_bytes,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )
    summary["inputs"] = {
        "capsule_results": args.capsule_results,
        "baseline_results": args.baseline_results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _results_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "results.jsonl" if path.is_dir() else path


def load_capsule_rows(path: str | Path) -> dict[str, dict]:
    selected = {}
    with _results_path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("protocol") != "question_conditioned":
                continue
            if row["id"] in selected:
                raise ValueError(f"duplicate capsule row: {row['id']}")
            selected[row["id"]] = row
    if not selected:
        raise ValueError("no question-conditioned capsule rows")
    return selected


def load_baseline_rows(paths) -> dict[str, dict]:
    selected = {}
    for path in paths:
        with _results_path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in selected:
                    raise ValueError(f"duplicate baseline row: {row['id']}")
                selected[row["id"]] = row
    if not selected:
        raise ValueError("no baseline rows")
    return selected


def join_rows(capsule, baseline, *, expected_cases: int) -> list[dict]:
    if capsule.keys() != baseline.keys():
        raise ValueError("capsule and baseline ID sets differ")
    if len(capsule) != expected_cases:
        raise ValueError("joined inputs do not contain the expected cases")
    rows = []
    for case_id in sorted(capsule, key=lambda value: int(capsule[value]["index"])):
        left, right = capsule[case_id], baseline[case_id]
        for field in ("index", "question", "gold_answers", "source_answer"):
            if left[field] != right[field]:
                raise ValueError(f"capsule/baseline field differs: {field}")
        row = dict(left)
        for arm in TEXT_ARMS:
            for suffix in (
                "",
                "_em",
                "_f1",
                "_prompt_tokens",
                "_prefill_ms",
                "_generation_ms",
                "_handoff_tokens",
                "_handoff_utf8_bytes",
            ):
                key = f"{arm}{suffix}"
                if key in right:
                    row[key] = right[key]
        row["source_full_kv"] = row["source_answer"]
        row["source_full_kv_em"] = row["source_answer_em"]
        row["source_full_kv_f1"] = row["source_answer_f1"]
        rows.append(row)
    indices = {int(row["index"]) for row in rows}
    if indices != set(range(expected_cases)):
        raise ValueError("joined rows do not cover contiguous indices")
    return rows


def analyze(
    rows,
    *,
    bootstrap_replicates: int,
    seed: int,
    capsule_payload_bytes: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict:
    mean = lambda key: sum(float(row[key]) for row in rows) / len(rows)
    quality = {
        arm: {"em": mean(f"{arm}_em"), "f1": mean(f"{arm}_f1")}
        for arm in ALL_ARMS
    }
    pairs = (
        ("capsule", "no_summary"),
        ("capsule", "shifted_capsule"),
        ("generated_tail", "capsule"),
        ("plaintext_source", "capsule"),
        ("plaintext_source", "generated_tail"),
        ("full_recompute", "capsule"),
        ("full_recompute", "generated_tail"),
        ("source_full_kv", "capsule"),
        ("plaintext_gold", "full_recompute"),
    )
    comparisons = {
        f"{candidate}_vs_{reference}": paired_summary(
            [row[f"{candidate}_em"] for row in rows],
            [row[f"{reference}_em"] for row in rows],
            [row[f"{candidate}_f1"] for row in rows],
            [row[f"{reference}_f1"] for row in rows],
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        for candidate, reference in pairs
    }
    int4_bytes_per_token = (
        num_layers * num_kv_heads * head_dim * 2 * 4 / 8
        + num_layers * num_kv_heads * 2 * 2
    )
    bf16_bytes_per_token = num_layers * num_kv_heads * head_dim * 2 * 2
    mean_source_tokens = mean("source_tokens")
    mean_tail_tokens = mean("generated_tail_tokens")
    transport = {
        "capsule_payload_bytes": capsule_payload_bytes,
        "generated_tail_mean_tokens": mean_tail_tokens,
        "generated_tail_int4_payload_bytes": mean_tail_tokens * int4_bytes_per_token,
        "plaintext_source_mean_tokens": mean("plaintext_source_handoff_tokens"),
        "plaintext_source_mean_utf8_bytes": mean(
            "plaintext_source_handoff_utf8_bytes"
        ),
        "full_kv_mean_tokens": mean_source_tokens,
        "full_kv_theoretical_int4_payload_bytes": (
            mean_source_tokens * int4_bytes_per_token
        ),
        "full_kv_bf16_payload_bytes": mean_source_tokens * bf16_bytes_per_token,
        "int4_bytes_per_kv_token": int4_bytes_per_token,
        "bf16_bytes_per_kv_token": bf16_bytes_per_token,
    }
    capsule_path = [
        row["capsule_emission_ms"]
        + row["capsule_prepare_ms"]
        + row["capsule_generation_ms"]
        for row in rows
    ]
    tail_path = [
        row["source_decode_ms"]
        + row["generated_tail_prepare_ms"]
        + row["generated_tail_generation_ms"]
        for row in rows
    ]
    plaintext_path = [
        row["source_decode_ms"]
        + row["plaintext_source_prefill_ms"]
        + row["plaintext_source_generation_ms"]
        for row in rows
    ]
    full_recompute_path = [
        row["full_recompute_prefill_ms"] + row["full_recompute_generation_ms"]
        for row in rows
    ]
    timing = {
        "capsule_post_source_prefill_ms": sum(capsule_path) / len(rows),
        "generated_tail_post_source_prefill_ms": sum(tail_path) / len(rows),
        "plaintext_source_post_source_prefill_ms": sum(plaintext_path) / len(rows),
        "full_recompute_receiver_ms": sum(full_recompute_path) / len(rows),
        "plaintext_source_receiver_prefill_ms": mean("plaintext_source_prefill_ms"),
        "plaintext_source_receiver_generation_ms": mean(
            "plaintext_source_generation_ms"
        ),
        "full_recompute_receiver_prefill_ms": mean("full_recompute_prefill_ms"),
        "full_recompute_receiver_generation_ms": mean(
            "full_recompute_generation_ms"
        ),
    }
    return {
        "cases": len(rows),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "mean_source_tokens": mean_source_tokens,
        "quality": quality,
        "comparisons": comparisons,
        "transport": transport,
        "timing_ms": timing,
    }


if __name__ == "__main__":
    main()
