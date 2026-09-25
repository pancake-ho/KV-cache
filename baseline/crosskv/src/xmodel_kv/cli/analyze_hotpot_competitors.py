from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_paired_transfer import paired_summary


BASE_ARMS = (
    "capsule",
    "generated_tail",
    "plaintext_source",
    "full_recompute",
)
COMPETITOR_ARMS = (
    "kvpacket_full_recompute",
    "kvpacket_no_recompute",
    "kvpacket",
    "cacheblend_cold",
    "cacheblend",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join capsule, KVPacket, and CacheBlend on HotpotQA-E."
    )
    parser.add_argument("--base-matrix", required=True)
    parser.add_argument("--kvpacket-results", nargs="+", required=True)
    parser.add_argument("--cacheblend-cold-results", nargs="+", required=True)
    parser.add_argument("--cacheblend-results", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=290)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--capsule-payload-bytes", type=float, required=True)
    args = parser.parse_args()
    if min(
        args.expected_cases,
        args.bootstrap_replicates,
        args.capsule_payload_bytes,
    ) < 1:
        parser.error("counts and payload must be positive")

    base = load_unique_rows([args.base_matrix], name="base")
    kvpacket = load_unique_rows(args.kvpacket_results, name="KVPacket")
    cacheblend_cold = load_unique_rows(
        args.cacheblend_cold_results, name="CacheBlend cold"
    )
    cacheblend = load_unique_rows(args.cacheblend_results, name="CacheBlend")
    rows = join_rows(
        base,
        kvpacket,
        cacheblend_cold,
        cacheblend,
        expected_cases=args.expected_cases,
    )
    analysis = analyze(
        rows,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        capsule_payload_bytes=args.capsule_payload_bytes,
    )
    analysis["inputs"] = {
        "base_matrix": args.base_matrix,
        "kvpacket_results": args.kvpacket_results,
        "cacheblend_cold_results": args.cacheblend_cold_results,
        "cacheblend_results": args.cacheblend_results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


def results_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "matrix.jsonl" if path.is_dir() else path


def load_unique_rows(paths, *, name: str) -> dict[str, dict]:
    rows = {}
    for path in paths:
        with results_path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                case_id = row["id"]
                if case_id in rows:
                    raise ValueError(f"duplicate {name} row: {case_id}")
                rows[case_id] = row
    if not rows:
        raise ValueError(f"no {name} rows")
    return rows


def join_rows(
    base,
    kvpacket,
    cacheblend_cold,
    cacheblend,
    *,
    expected_cases: int,
) -> list[dict]:
    id_sets = {
        "base": set(base),
        "kvpacket": set(kvpacket),
        "cacheblend_cold": set(cacheblend_cold),
        "cacheblend": set(cacheblend),
    }
    if any(ids != id_sets["base"] for ids in id_sets.values()):
        sizes = {name: len(ids) for name, ids in id_sets.items()}
        raise ValueError(f"competitor ID sets differ: {sizes}")
    if len(base) != expected_cases:
        raise ValueError("competitor inputs do not contain the expected cases")

    joined = []
    for case_id in sorted(base, key=lambda value: int(base[value]["index"])):
        root = base[case_id]
        packet = kvpacket[case_id]
        cold = cacheblend_cold[case_id]
        blend = cacheblend[case_id]
        for candidate_name, candidate in (
            ("KVPacket", packet),
            ("CacheBlend cold", cold),
            ("CacheBlend", blend),
        ):
            for key in ("index", "question", "gold_answers"):
                if candidate[key] != root[key]:
                    raise ValueError(
                        f"{candidate_name}/base mismatch for {case_id}: {key}"
                    )
        for key in (
            "prompt_tokens",
            "document_tokens",
            "document_kv_bf16_bytes",
        ):
            if cold[key] != blend[key]:
                raise ValueError(f"CacheBlend cold/blend mismatch: {case_id} {key}")
        cached_tokens = int(blend["cacheblend_cached_tokens"])
        if not 0 < cached_tokens < int(blend["prompt_tokens"]):
            raise ValueError(
                "CacheBlend must hit documents and miss target query: "
                f"{case_id} cached={cached_tokens} prompt={blend['prompt_tokens']}"
            )

        row = dict(root)
        for source, target in (
            ("full_recompute", "kvpacket_full_recompute"),
            ("no_recompute", "kvpacket_no_recompute"),
            ("kv_packet", "kvpacket"),
        ):
            for suffix in (
                "",
                "_em",
                "_f1",
                "_ttft_ms",
                "_flops",
                "_online_total_flops",
                "_peak_gpu_delta_bytes",
            ):
                row[f"{target}{suffix}"] = packet[f"{source}{suffix}"]
        row.update(
            {
                "kvpacket_document_tokens": packet["document_tokens"],
                "kvpacket_raw_document_kv_bf16_bytes": packet[
                    "raw_document_kv_bf16_bytes"
                ],
                "kvpacket_packet_kv_bf16_bytes": packet[
                    "packet_kv_bf16_bytes"
                ],
                "cacheblend_cold": cold["cold_full"],
                "cacheblend_cold_em": cold["cold_full_em"],
                "cacheblend_cold_f1": cold["cold_full_f1"],
                "cacheblend_cold_wall_ms": cold["cold_full_wall_ms"],
                "cacheblend_cold_ttft_ms": cold["cold_full_ttft_ms"],
                "cacheblend": blend["cacheblend"],
                "cacheblend_em": blend["cacheblend_em"],
                "cacheblend_f1": blend["cacheblend_f1"],
                "cacheblend_wall_ms": blend["cacheblend_wall_ms"],
                "cacheblend_ttft_ms": blend["cacheblend_ttft_ms"],
                "cacheblend_cached_tokens": blend["cacheblend_cached_tokens"],
                "cacheblend_uncached_tokens": blend["cacheblend_uncached_tokens"],
                "cacheblend_cached_fraction": blend[
                    "cacheblend_cached_fraction"
                ],
                "cacheblend_population_wall_ms": blend["population_wall_ms"],
                "cacheblend_population_prompt_tokens": blend[
                    "population_prompt_tokens"
                ],
                "cacheblend_prompt_tokens": blend["prompt_tokens"],
                "cacheblend_document_tokens": blend["document_tokens"],
                "cacheblend_document_kv_bf16_bytes": blend[
                    "document_kv_bf16_bytes"
                ],
            }
        )
        joined.append(row)
    if {int(row["index"]) for row in joined} != set(range(expected_cases)):
        raise ValueError("joined rows do not cover contiguous expected indices")
    return joined


def analyze(
    rows,
    *,
    bootstrap_replicates: int,
    seed: int,
    capsule_payload_bytes: float,
) -> dict:
    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    quality = {
        arm: {"em": mean(f"{arm}_em"), "f1": mean(f"{arm}_f1")}
        for arm in (*BASE_ARMS, *COMPETITOR_ARMS)
    }
    pairs = (
        ("kvpacket", "kvpacket_no_recompute"),
        ("kvpacket", "kvpacket_full_recompute"),
        ("kvpacket", "capsule"),
        ("kvpacket", "generated_tail"),
        ("cacheblend", "cacheblend_cold"),
        ("cacheblend", "capsule"),
        ("cacheblend_cold", "full_recompute"),
        ("cacheblend", "kvpacket"),
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
    bf16_bytes_per_token = 32 * 8 * 128 * 2 * 2
    packet_payload = mean("kvpacket_packet_kv_bf16_bytes")
    raw_payload = mean("kvpacket_raw_document_kv_bf16_bytes")
    blend_payload = mean("cacheblend_document_kv_bf16_bytes")
    blend_retrieved_payload = mean("cacheblend_cached_tokens") * bf16_bytes_per_token
    transport = {
        "capsule_payload_bytes": capsule_payload_bytes,
        "kvpacket_raw_document_kv_bf16_bytes": raw_payload,
        "kvpacket_packet_kv_bf16_bytes": packet_payload,
        "kvpacket_wrapper_overhead_fraction": packet_payload / raw_payload - 1,
        "cacheblend_document_kv_bf16_bytes": blend_payload,
        "cacheblend_retrieved_kv_bf16_bytes": blend_retrieved_payload,
        "kvpacket_to_capsule_payload_ratio": packet_payload
        / capsule_payload_bytes,
        "cacheblend_document_to_capsule_payload_ratio": blend_payload
        / capsule_payload_bytes,
        "cacheblend_retrieved_to_capsule_payload_ratio": blend_retrieved_payload
        / capsule_payload_bytes,
    }
    timing = {
        "kvpacket_full_recompute_ttft_ms": mean(
            "kvpacket_full_recompute_ttft_ms"
        ),
        "kvpacket_no_recompute_ttft_ms": mean("kvpacket_no_recompute_ttft_ms"),
        "kvpacket_ttft_ms": mean("kvpacket_ttft_ms"),
        "cacheblend_cold_wall_ms": mean("cacheblend_cold_wall_ms"),
        "cacheblend_wall_ms": mean("cacheblend_wall_ms"),
        "cacheblend_population_wall_ms": mean("cacheblend_population_wall_ms"),
        "cacheblend_population_plus_reuse_wall_ms": mean(
            "cacheblend_population_wall_ms"
        )
        + mean("cacheblend_wall_ms"),
        "cacheblend_cold_ttft_ms": mean("cacheblend_cold_ttft_ms"),
        "cacheblend_ttft_ms": mean("cacheblend_ttft_ms"),
        "capsule_receiver_ms": mean("capsule_prepare_ms")
        + mean("capsule_generation_ms"),
        "capsule_post_source_prefill_ms": mean("capsule_emission_ms")
        + mean("capsule_prepare_ms")
        + mean("capsule_generation_ms"),
    }
    return {
        "cases": len(rows),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "quality": quality,
        "comparisons": comparisons,
        "transport": transport,
        "timing_ms": timing,
        "cacheblend": {
            "mean_prompt_tokens": mean("cacheblend_prompt_tokens"),
            "mean_cached_tokens": mean("cacheblend_cached_tokens"),
            "mean_uncached_tokens": mean("cacheblend_uncached_tokens"),
            "mean_cached_fraction": mean("cacheblend_cached_fraction"),
            "zero_hit_cases": sum(
                int(row["cacheblend_cached_tokens"]) == 0 for row in rows
            ),
            "full_hit_cases": sum(
                int(row["cacheblend_cached_tokens"])
                == int(row["cacheblend_prompt_tokens"])
                for row in rows
            ),
        },
    }


if __name__ == "__main__":
    main()
