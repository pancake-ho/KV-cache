from __future__ import annotations

import argparse
import json
from pathlib import Path

from xmodel_kv.hotpotqa import answer_em

from .analyze_paired_transfer import paired_summary


ARMS = ("no_summary", "capsule", "shifted_capsule", "generated_tail")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and analyze frozen HotpotQA KV-summary transfer shards."
    )
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--capsule-payload-bytes", type=float, default=126_720)
    parser.add_argument(
        "--generated-tail-bytes-per-token", type=float, default=38_016
    )
    args = parser.parse_args()
    if min(
        args.expected_cases,
        args.bootstrap_replicates,
        args.capsule_payload_bytes,
        args.generated_tail_bytes_per_token,
    ) < 1:
        parser.error("case and bootstrap counts must be positive")
    rows = load_and_validate(args.results, expected_cases=args.expected_cases)
    summary = analyze(
        rows,
        input_paths=args.results,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        capsule_payload_bytes=args.capsule_payload_bytes,
        generated_tail_bytes_per_token=args.generated_tail_bytes_per_token,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (int(row["index"]), row["protocol"]))
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered)
    )
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_and_validate(paths, *, expected_cases: int) -> list[dict]:
    rows = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            path = path / "results.jsonl"
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    required = {
        "id",
        "index",
        "protocol",
        "gold_answers",
        "source_answer",
        "source_answer_em",
        "source_answer_f1",
        "source_tokens",
    }
    required.update(f"{arm}_em" for arm in ARMS)
    required.update(f"{arm}_f1" for arm in ARMS)
    if any(not required.issubset(row) for row in rows):
        raise ValueError("one or more result rows are missing required fields")
    keys = [(row["id"], row["protocol"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (id, protocol) result row")
    protocols = set(row["protocol"] for row in rows)
    if protocols != {"state_readout", "question_conditioned"}:
        raise ValueError(f"unexpected protocol set: {sorted(protocols)}")
    id_sets = {
        protocol: {row["id"] for row in rows if row["protocol"] == protocol}
        for protocol in protocols
    }
    if any(len(ids) != expected_cases for ids in id_sets.values()):
        raise ValueError("protocol does not contain the expected number of cases")
    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        raise ValueError("protocols contain different case IDs")
    indices = {row["index"] for row in rows}
    if indices != set(range(expected_cases)):
        raise ValueError("merged results do not cover the expected contiguous indices")
    by_key = {(row["id"], row["protocol"]): row for row in rows}
    for case_id in id_sets["state_readout"]:
        left = by_key[(case_id, "state_readout")]
        right = by_key[(case_id, "question_conditioned")]
        for field in (
            "index",
            "source_answer",
            "source_answer_em",
            "source_answer_f1",
            "source_tokens",
        ):
            if left[field] != right[field]:
                raise ValueError(f"source field differs across protocols: {field}")
    return rows


def analyze(
    rows: list[dict],
    *,
    input_paths,
    bootstrap_replicates: int,
    seed: int,
    capsule_payload_bytes: float = 126_720,
    generated_tail_bytes_per_token: float = 38_016,
) -> dict:
    output = {
        "input_paths": list(input_paths),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "by_protocol": {},
    }
    for protocol in ("state_readout", "question_conditioned"):
        selected = sorted(
            (row for row in rows if row["protocol"] == protocol),
            key=lambda row: int(row["index"]),
        )
        protocol_summary = analyze_subset(
            selected,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            capsule_payload_bytes=capsule_payload_bytes,
            generated_tail_bytes_per_token=generated_tail_bytes_per_token,
        )
        source_correct = [row for row in selected if float(row["source_answer_em"]) == 1]
        protocol_summary["source_correct_diagnostic"] = analyze_subset(
            source_correct,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            capsule_payload_bytes=capsule_payload_bytes,
            generated_tail_bytes_per_token=generated_tail_bytes_per_token,
        )
        output["by_protocol"][protocol] = protocol_summary
    return output


def analyze_subset(
    rows,
    *,
    bootstrap_replicates: int,
    seed: int,
    capsule_payload_bytes: float,
    generated_tail_bytes_per_token: float,
) -> dict:
    if not rows:
        raise ValueError("cannot analyze an empty subset")
    mean = lambda key: sum(float(row[key]) for row in rows) / len(rows)
    arm_metrics = {
        arm: {
            "em": mean(f"{arm}_em"),
            "f1": mean(f"{arm}_f1"),
        }
        for arm in ARMS
    }
    comparisons = {}
    for candidate, reference in (
        ("capsule", "shifted_capsule"),
        ("capsule", "no_summary"),
        ("generated_tail", "capsule"),
        ("generated_tail", "no_summary"),
    ):
        comparisons[f"{candidate}_vs_{reference}"] = paired_summary(
            [row[f"{candidate}_em"] for row in rows],
            [row[f"{reference}_em"] for row in rows],
            [row[f"{candidate}_f1"] for row in rows],
            [row[f"{reference}_f1"] for row in rows],
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
    result = {
        "cases": len(rows),
        "source_answer_em": mean("source_answer_em"),
        "source_answer_f1": mean("source_answer_f1"),
        "mean_source_tokens": mean("source_tokens"),
        "arms": arm_metrics,
        "comparisons": comparisons,
        "generated_tail_matches_source_answer": sum(
            answer_em(row["generated_tail"], [row["source_answer"]]) for row in rows
        )
        / len(rows),
    }
    mean_tail_tokens = mean("generated_tail_tokens")
    mean_tail_payload = mean_tail_tokens * generated_tail_bytes_per_token
    result["transport"] = {
        "capsule_slots": mean("capsule_tokens"),
        "capsule_payload_bytes": capsule_payload_bytes,
        "mean_generated_tail_tokens": mean_tail_tokens,
        "mean_generated_tail_payload_bytes": mean_tail_payload,
        "capsule_payload_reduction_vs_tail": 1
        - capsule_payload_bytes / mean_tail_payload,
    }
    timing_fields = (
        "source_prefill_ms",
        "capsule_emission_ms",
        "source_decode_ms",
        "capsule_prepare_ms",
        "capsule_generation_ms",
        "generated_tail_prepare_ms",
        "generated_tail_generation_ms",
    )
    if all(all(field in row for field in timing_fields) for row in rows):
        capsule_postprefill = [
            row["capsule_emission_ms"]
            + row["capsule_prepare_ms"]
            + row["capsule_generation_ms"]
            for row in rows
        ]
        tail_postprefill = [
            row["source_decode_ms"]
            + row["generated_tail_prepare_ms"]
            + row["generated_tail_generation_ms"]
            for row in rows
        ]
        result["timing_ms"] = {
            **{f"mean_{field}": mean(field) for field in timing_fields},
            "mean_capsule_postprefill_handoff": sum(capsule_postprefill) / len(rows),
            "mean_tail_postprefill_handoff": sum(tail_postprefill) / len(rows),
            "mean_capsule_minus_tail_postprefill": sum(
                capsule - tail
                for capsule, tail in zip(capsule_postprefill, tail_postprefill, strict=True)
            )
            / len(rows),
        }
    return result


if __name__ == "__main__":
    main()
