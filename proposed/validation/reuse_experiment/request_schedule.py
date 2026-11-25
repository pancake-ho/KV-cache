"""Schedule source prefills and repeated queries over shared document prefixes."""

from collections import defaultdict

from ...src.serving.request import Request
from ..e2e_workload import Event, workload_kinds


def make_reuse_trace(selected, tokenizer, model_ids, scenario, profile, seed):
    """Use the same ordered target questions and vary cross-model session boundaries."""
    papers = defaultdict(list)
    for item in selected:
        papers[item[0]["paper_id"]].append(item)
    count = len(papers)
    per_paper = profile["questions_per_document"]
    reuse_count = scenario["reuse_count"]
    kinds = workload_kinds(count, scenario, seed)
    pressure = profile["reuse_experiment"] == "pressure"
    if (not 1 <= reuse_count <= per_paper or per_paper % reuse_count
            or "cross" not in kinds
            or (pressure and (reuse_count != per_paper or profile["pressure_window"] < 1))):
        raise ValueError("Invalid reuse count, pressure profile, or workload shares")
    source, target = model_ids
    sessions = []
    for index, ((paper_id, questions), kind) in enumerate(zip(papers.items(), kinds, strict=True)):
        if len(questions) != per_paper:
            raise ValueError(f"Incorrect number of questions for paper {paper_id}")
        session_size = {"same": per_paper, "cross": reuse_count, "cold": per_paper}[kind]
        for start in range(0, per_paper, session_size):
            cohort = questions[start:start + session_size]
            metadata, (first, boundary) = cohort[0]
            prefix = tuple(first[:boundary])
            key = f"{metadata['document_sha256']}:{start}"
            producer = None
            if kind != "cold":
                producer = Request(f"prefix-{index}-{start}", "prefix-user",
                                   source if kind == "cross" else target,
                                   tokenizer.decode(prefix), prefix, key)
            consumers = []
            for offset, (document, (ids, boundary)) in enumerate(cohort):
                tokens = tuple(ids)
                if tokens[:boundary] != prefix or not 0 < boundary < len(tokens):
                    raise ValueError(f"Different token prefix at paper {paper_id}")
                request = Request(f"query-{index}-{start + offset}", "query-user", target,
                                  tokenizer.decode(tokens), tokens,
                                  f"{key}:{offset}" if kind == "cold" else key)
                consumers.append(Event(request, kind, document, producer.request_id if producer else None))
            producer_event = Event(producer, "producer", metadata, None) if producer else None
            sessions.append((producer_event, consumers))
    if pressure:
        # Prefill a group of papers, then interleave questions for that group.
        window = profile["pressure_window"]
        trace = []
        for start in range(0, len(sessions), window):
            batch = sessions[start:start + window]
            trace.extend(producer for producer, _ in batch if producer is not None)
            trace.extend(rows[turn] for turn in range(per_paper) for _, rows in reversed(batch))
        return trace
    trace = []
    for producer, consumers in sessions:
        if producer is not None:
            trace.append(producer)
        trace.extend(consumers)
    return trace
