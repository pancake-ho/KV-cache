"""Measure and compare the in-process serving workload."""

from collections import defaultdict
from math import ceil
from statistics import fmean, median
from time import perf_counter


def measure_replays(run, *, device, repeats):
    """Warm the mode once, then time fresh traces and sample CUDA allocator peaks."""
    import torch

    run()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start_allocated = torch.cuda.memory_allocated(device)
        start_reserved = torch.cuda.memory_reserved(device)

        started = perf_counter()
        rows = run()
        torch.cuda.synchronize(device)
        seconds = perf_counter() - started

        samples.append((rows, {
            "wall_seconds": seconds,
            "start_gpu_allocated_bytes": start_allocated,
            "start_gpu_reserved_bytes": start_reserved,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }))
    return samples


def _latency_stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": fmean(ordered),
        "p50": median(ordered),
        "p95": ordered[ceil(0.95 * len(ordered)) - 1],
    }


def _reduction_pct(baseline, candidate):
    return 100 * (baseline - candidate) / baseline if baseline > 0 else None


def _summarize_mode(samples):
    rows = [row for replay_rows, _ in samples for row in replay_rows]
    replay_stats = [stats for _, stats in samples]
    wall_seconds = sum(stats["wall_seconds"] for stats in replay_stats)
    reused = [row for row in rows if row["reused_prefix_tokens"] > 0]
    prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    generated_tokens = sum(len(row["generated_ids"]) for row in rows)

    return {
        "replay_count": len(samples),
        "request_count": len(rows),
        "hit_count": len(reused),
        "request_hit_rate_pct": 100 * len(reused) / len(rows),
        "prefix_token_reuse_pct": 100 * sum(row["reused_prefix_tokens"] for row in rows) / prompt_tokens,
        "ttft_ms": _latency_stats([row["ttft_ms"] for row in rows]),
        "hit_ttft_ms": _latency_stats([row["ttft_ms"] for row in reused]),
        "request_latency_ms": _latency_stats([row["application_ms"] for row in rows]),
        "replay_wall_seconds": wall_seconds,
        "requests_per_second": len(rows) / wall_seconds,
        "output_tokens_per_second": generated_tokens / wall_seconds,
        "peak_gpu_allocated_bytes": max(stats["peak_gpu_allocated_bytes"] for stats in replay_stats),
        "peak_gpu_reserved_bytes": max(stats["peak_gpu_reserved_bytes"] for stats in replay_stats),
        "peak_resident_kv_bytes": max(row["resident_kv_bytes"] for row in rows),
        "replays": replay_stats,
    }


def _compare_modes(baseline, candidate, baseline_summary, candidate_summary, request_cases=None):
    hit_baseline = []
    hit_candidate = []
    first_token_agreements = 0
    all_token_agreements = 0
    cases = defaultdict(list)

    for (reference_rows, _), (candidate_rows, _) in zip(baseline, candidate, strict=True):
        for reference, row in zip(reference_rows, candidate_rows, strict=True):
            if row["request_id"] != reference["request_id"]:
                raise ValueError("performance comparisons require identical ordered request IDs")
            reference_ids, candidate_ids = reference["generated_ids"], row["generated_ids"]
            first_token_agreements += reference_ids[0] == candidate_ids[0]
            all_token_agreements += reference_ids == candidate_ids
            if request_cases is not None:
                cases[request_cases[row["request_id"]]].append((reference, row))
            if row["reused_prefix_tokens"] > 0:
                hit_baseline.append(reference["ttft_ms"])
                hit_candidate.append(row["ttft_ms"])

    comparisons = {
        "ttft_mean_reduction_pct": _reduction_pct(baseline_summary["ttft_ms"]["mean"],
                                                 candidate_summary["ttft_ms"]["mean"]),
        "ttft_p95_reduction_pct": _reduction_pct(baseline_summary["ttft_ms"]["p95"],
                                                candidate_summary["ttft_ms"]["p95"]),
        "hit_ttft_mean_reduction_pct": (
            _reduction_pct(fmean(hit_baseline), fmean(hit_candidate)) if hit_candidate else None
        ),
        "hit_ttft_baseline_ms": _latency_stats(hit_baseline),
        "hit_ttft_candidate_ms": _latency_stats(hit_candidate),
        "request_latency_mean_reduction_pct": _reduction_pct(
            baseline_summary["request_latency_ms"]["mean"], candidate_summary["request_latency_ms"]["mean"]
        ),
        "output_throughput_speedup": (
            candidate_summary["output_tokens_per_second"] / baseline_summary["output_tokens_per_second"]
        ),
        "first_token_agreement_pct": 100 * first_token_agreements / candidate_summary["request_count"],
        "all_tokens_agreement_pct": 100 * all_token_agreements / candidate_summary["request_count"],
    }
    for metric in ("peak_gpu_allocated", "peak_gpu_reserved", "peak_resident_kv"):
        key = f"{metric}_bytes"
        comparisons[f"{metric}_reduction_pct"] = _reduction_pct(baseline_summary[key], candidate_summary[key])
    if request_cases is not None:
        comparisons["by_case"] = {}
        for case, pairs in cases.items():
            reference_ttft = _latency_stats([reference["ttft_ms"] for reference, _ in pairs])
            candidate_ttft = _latency_stats([row["ttft_ms"] for _, row in pairs])
            comparisons["by_case"][case] = {
                "baseline_ttft_ms": reference_ttft,
                "candidate_ttft_ms": candidate_ttft,
                "ttft_mean_reduction_pct": _reduction_pct(reference_ttft["mean"], candidate_ttft["mean"]),
                "hit_count": sum(row["reused_prefix_tokens"] > 0 for _, row in pairs),
                "all_tokens_agreement_pct": 100 * sum(
                    reference["generated_ids"] == row["generated_ids"] for reference, row in pairs
                ) / len(pairs),
            }
    return comparisons


def performance_report(samples_by_mode, *, request_cases=None):
    """Keep whole-trace and paired cache-hit comparisons separate."""
    modes = {name: _summarize_mode(samples) for name, samples in samples_by_mode.items()}
    vs_full = {
        name: _compare_modes(
            samples_by_mode["full"], samples, modes["full"], modes[name], request_cases,
        )
        for name, samples in samples_by_mode.items() if name != "full"
    }
    bitpack_vs_raw = {}
    for name, samples in samples_by_mode.items():
        if name.endswith("_bitpack"):
            raw = name.removesuffix("_bitpack")
            bitpack_vs_raw[name] = _compare_modes(
                samples_by_mode[raw], samples, modes[raw], modes[name], request_cases,
            )

    return {
        "measurement": {
            "warmup_replays_per_mode": 1,
            "request_latency_scope": "in-process application_ms; no network or queueing",
            "memory_scope": "PyTorch CUDA allocator; peaks reset before each replay",
            "resident_kv_scope": "cache bytes sampled at request completion",
            "p95_method": "nearest rank across measured requests",
        },
        "modes": modes,
        "vs_full": vs_full,
        "bitpack_vs_raw": bitpack_vs_raw,
    }
