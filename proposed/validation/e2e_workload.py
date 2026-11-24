"""Build paired serving traces and summarize consumer and whole-trace metrics."""

from collections import Counter, defaultdict
import re
import string
from dataclasses import dataclass
from random import Random
from statistics import mean, median, quantiles

from ..src.serving.request import Request


@dataclass(frozen=True)
class Event:
    request: Request
    kind: str
    document: dict
    prefix_request_id: str | None


def documents(cohort, prompts, count, seed):
    by_group = defaultdict(list)
    for row, prompt in zip(cohort, prompts, strict=True):
        by_group[row["group"]].append((row, prompt))
    rng = Random(seed)
    for group in by_group.values():
        rng.shuffle(group)
    # Preserve all strata in the pilot and interleave strata in the sweep.
    selected = []
    while len(selected) < count:
        for group in by_group.values():
            if group and len(selected) < count:
                selected.append(group.pop())
    return selected


def workload_kinds(count, scenario, seed):
    same = round(count * scenario["same_share"])
    cross = min(round(count * scenario["cross_share"]), count - same)
    kinds = ["same"] * same + ["cross"] * cross + ["cold"] * (count - same - cross)
    Random(seed).shuffle(kinds)
    return kinds


def make_trace(selected, tokenizer, model_ids, scenario, window, seed):
    count = len(selected)
    source, target = model_ids
    types = workload_kinds(count, scenario, seed)
    steps = []
    for index, ((document, (tokens, boundary)), kind) in enumerate(zip(selected, types, strict=True)):
        tokens = tuple(tokens)
        key = document["document_sha256"]
        consumer = Request(f"query-{index}", "query-user", target, tokenizer.decode(tokens), tokens, key)
        if kind == "cold":
            steps.append((None, Event(consumer, kind, document, None)))
            continue
        prefix = tokens[:boundary]
        producer = Request(f"prefix-{index}", "prefix-user", source if kind == "cross" else target,
                           tokenizer.decode(prefix), prefix, key)
        steps.append((Event(producer, "producer", document, None),
                      Event(consumer, kind, document, producer.request_id)))
    trace = []
    for start in range(0, count, window):
        group = steps[start:start + window]
        trace += [p for p, _ in group if p is not None]
        trace += [q for _, q in group]
    return trace


def route_check(event, row, method):
    kind = event.kind
    path = row["path"]
    if method == "full" and path != "full_recompute":
        raise ValueError("Full prefill unexpectedly accessed the cache")
    if row["reused_from"] is not None:
        expected = {"same": "native_kv_hit", "cross": "cross_model_kv_hit"}.get(kind)
        if (path != expected or row["reused_from"] != event.prefix_request_id
                or row["reused_prefix_tokens"] != event.document["source_prefix_tokens"]
                or row["forwarded_prefill_tokens"] != len(event.request.token_ids) - row["reused_prefix_tokens"]
                or (method == "exact" and kind == "cross")):
            raise ValueError(f"Unexpected prefix route for {row['request_id']}: {path}")


def distribution(samples):
    return {"p50": median(samples),
            "p95": quantiles(samples, n=20, method="inclusive")[-1] if len(samples) > 1 else samples[0]}


def summarize(replays, *, trace=None, cache_capacity_bytes=None):
    consumers = defaultdict(list)
    for replay in replays:
        for row in replay["requests"]:
            if row["kind"] != "producer":
                consumers[row["request_id"]].append(row)
    # Each target QA contributes one observation to the latency percentiles.
    metrics = ("ttft_ms", "application_ms", "mapping_ms", "prefill_ms")
    rows = [{
        "request_id": key, "group": values[0]["group"], "kind": values[0]["kind"],
        "source_prefix_tokens": values[0]["source_prefix_tokens"],
        "path": values[0]["path"], "f1": mean(row["f1"] for row in values),
        "exact_match": mean(row["exact_match"] for row in values),
        "repeat_tokens_match": all(row["generated_ids"] == values[0]["generated_ids"]
                                   for row in values),
        **({"paper_id": values[0]["paper_id"]} if "paper_id" in values[0] else {}),
        **{name: median(row[name] for row in values) for name in metrics},
    } for key, values in consumers.items()]
    hits = {"native_kv_hit", "cross_model_kv_hit"}
    all_rows = [row for replay in replays for row in replay["requests"]]
    consumer_rows = [row for row in all_rows if row["kind"] != "producer"]
    path_counts = dict(Counter(row["path"] for row in consumer_rows))
    kind_counts = dict(Counter(row["kind"] for row in rows))
    summary = {
        "qa_documents": len(rows),
        "trace_requests_per_replay": len(replays[0]["requests"]),
        "qa_f1": mean(row["f1"] for row in rows),
        "qa_em": mean(row["exact_match"] for row in rows),
        "qa_paths": path_counts, "offered_kinds": kind_counts,
        "qa_hit_rate": sum(path_counts.get(path, 0) for path in hits) / len(consumer_rows),
        "cross_hit_rate": path_counts.get("cross_model_kv_hit", 0) / len(consumer_rows),
        "trace_requests_per_second": median(replay["requests_per_second"] for replay in replays),
        "peak_allocated_bytes": max(replay["peak_allocated_bytes"] for replay in replays),
        "peak_reserved_bytes": max(replay["peak_reserved_bytes"] for replay in replays),
        "peak_resident_kv_bytes": max(row["resident_kv_bytes"] for row in all_rows),
        "model_allocated_bytes": replays[0]["model_allocated_bytes"],
        "mapper_allocated_bytes": replays[0]["mapper_allocated_bytes"],
        **{name: distribution([row[name] for row in rows]) for name in metrics},
    }
    summary["by_offered_kind"] = {
        kind: {"ttft_ms": distribution([row["ttft_ms"] for row in rows if row["kind"] == kind]),
               "qa_f1": mean(row["f1"] for row in rows if row["kind"] == kind)}
        for kind in kind_counts
    }
    if trace is not None:
        summary["reuse"] = summarize_reuse(replays, consumers, trace, cache_capacity_bytes)
    return summary, rows


def summarize_reuse(replays, consumers, trace, capacity_bytes):
    if capacity_bytes is None or capacity_bytes < 1:
        raise ValueError("Reuse metrics require a positive cache capacity")
    targets = [event for event in trace if event.kind != "producer"]
    producers = [event for event in trace if event.kind == "producer"]
    producer_kind = {event.prefix_request_id: event.kind for event in targets if event.prefix_request_id}
    opportunities = sum(event.kind == "cross" for event in targets)
    target_count = len(targets)
    if not opportunities or target_count != len(consumers):
        raise ValueError("Invalid target QA or cross-model opportunity count")
    trace_ids = [event.request.request_id for event in trace]
    cross_hits, evictions, occupancy, producer_ms = [], [], [], []
    for replay in replays:
        requests = replay["requests"]
        if [row["request_id"] for row in requests] != trace_ids:
            raise ValueError("Replay requests do not match the repeated-prefix trace")
        rows = [row for row in requests if row["kind"] != "producer"]
        cross_hits.append(sum(row["kind"] == "cross" and row["path"] == "cross_model_kv_hit"
                              for row in rows))
        evictions.append(sum(row["cache_evictions"] for row in requests))
        occupancy.append(median(row["resident_kv_bytes_before"] / capacity_bytes for row in rows))
        producer_ms.append(sum(row["application_ms"] for row in requests if row["kind"] == "producer"))

    hit_ttfts = []
    for samples in consumers.values():
        hits = [row for row in samples if row["path"] == "cross_model_kv_hit"]
        if hits:
            hit_ttfts.append(median(row["ttft_ms"] for row in hits))
    return {
        "context_documents": len({event.document["paper_id"] for event in targets}),
        "target_qa_per_replay": target_count,
        "producer_prefills_per_replay": len(producers),
        "cross_source_prefills_per_replay": sum(kind == "cross" for kind in producer_kind.values()),
        "producer_application_ms_per_replay_p50": median(producer_ms),
        "cross_opportunities_per_replay": opportunities,
        "cross_hits_by_replay": cross_hits,
        "cross_hit_rate_given_eligible": sum(cross_hits) / (opportunities * len(replays)),
        "cross_hit_qa_count": len(hit_ttfts),
        "cross_hit_ttft_ms": distribution(hit_ttfts) if hit_ttfts else None,
        "elapsed_ms_per_target_qa": distribution(
            [1000 * replay["wall_seconds"] / target_count for replay in replays]),
        "cache_evictions_by_replay": evictions,
        "cache_fill_before_qa_ratio_p50": median(occupancy),
    }


def paired_deltas(rows, full, seed):
    reference = {row["request_id"]: row for row in full}
    if set(reference) != {row["request_id"] for row in rows}:
        raise ValueError("Methods were not measured on the same paired document IDs")
    paired = {"f1_points": defaultdict(list), "ttft_ms": defaultdict(list)}
    for row in rows:
        old = reference[row["request_id"]]
        paper = row.get("paper_id")
        if (row["group"], row["kind"], paper) != (old["group"], old["kind"], old.get("paper_id")):
            raise ValueError("Paired workload labels differ")
        key = row["group"], paper
        paired["f1_points"][key].append(100 * (row["f1"] - old["f1"]))
        paired["ttft_ms"][key].append(row["ttft_ms"] - old["ttft_ms"])
    result = {}
    for name, blocks in paired.items():
        groups = defaultdict(list)
        for (group, paper), values in blocks.items():
            groups[group].extend(values if paper is None else [mean(values)])
        result[name] = {
            "mean": mean(value for block in blocks.values() for value in block),
            "document_bootstrap_ci95": interval(groups, seed, draws=2000),
        }
        if any(paper is not None for _, paper in blocks):
            result[name]["bootstrap_unit"] = "paper"
    return result
PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(text):
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text.lower().translate(PUNCTUATION)).split())


def score_answer(answer, references):
    prediction = normalize(answer)
    predicted = prediction.split()
    f1, exact = 0.0, 0
    for reference in references:
        truth = normalize(reference)
        words = truth.split()
        exact = max(exact, int(prediction == truth))
        common = sum((Counter(predicted) & Counter(words)).values())
        if common:
            f1 = max(f1, 2 * common / (len(predicted) + len(words)))
    return {"f1": f1, "exact_match": exact}


def interval(groups, seed, draws=5000):
    rng = Random(seed)
    samples = sorted(
        mean(value for group in groups.values() for value in rng.choices(group, k=len(group)))
        for _ in range(draws)
    )
    return [samples[draws // 40], samples[draws - 1 - draws // 40]]
