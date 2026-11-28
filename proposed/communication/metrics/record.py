"""Record CPU wall times, TCP sizes, target GPU peaks and paired QA summaries."""
import json
from statistics import mean, median
from time import perf_counter


def elapsed_ms(start):
    return (perf_counter() - start) * 1000


def frame_metrics(started, size, overhead):
    return {"tcp_ms": elapsed_ms(started), "payload_bytes": size,
            "tcp_application_bytes": size + overhead}


def measure_request(engine, request, budget, device):
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    row = engine.serve(request, budget)
    row.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
               peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return row


def write_row(file, row):
    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    file.flush()


def summarize_trials(paired_by_question, repeats):
    from ...validation.e2e_workload import distribution

    def stat(side, metric, aggregate):
        return [aggregate(trial[side][metric] for trial in trials) for trials in paired_by_question]

    saved = [median(trial["saved_ms"] for trial in trials) for trials in paired_by_question]

    summary = {
        "kind": "summary", "questions": len(saved), "repeats": repeats,
        "median_paired_saved_ms": median(saved), "faster_questions": sum(value > 0 for value in saved),
        "full_ttft_ms": distribution(stat("full", "ttft_ms", median)),
        "reuse_ttft_ms": distribution(stat("reuse", "ttft_ms", median)),
        "full_f1": mean(stat("full", "f1", mean)), "reuse_f1": mean(stat("reuse", "f1", mean)),
        "full_em": mean(stat("full", "exact_match", mean)),
        "reuse_em": mean(stat("reuse", "exact_match", mean)),
    }
    for scope in ("download_tcp", "download_total"):
        key = f"headroom_without_{scope}_ms"
        if key in paired_by_question[0][0]:
            margins = [median(trial[key] for trial in trials) for trials in paired_by_question]
            summary[key] = {
                "median_ms": median(margins),
                "positive_questions": sum(value > 0 for value in margins),
            }
    if "miss_extra_ms" in paired_by_question[0][0]:
        hit_saved = mean(
            mean(trial["saved_ms"] for trial in trials) for trials in paired_by_question
        )
        miss_extra = mean(
            mean(trial["miss_extra_ms"] for trial in trials) for trials in paired_by_question
        )
        summary.update(
            miss_ttft_ms=distribution(stat("miss", "ttft_ms", median)),
            mean_hit_saved_ms=hit_saved,
            mean_miss_extra_ms=miss_extra,
            break_even_hit_rate=(
                miss_extra / (hit_saved + miss_extra)
                if hit_saved > 0 and miss_extra >= 0 else None
            ),
        )
    return summary
