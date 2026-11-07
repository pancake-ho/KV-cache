from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .hotpotqa import normalize_answer


RAG_NAMESPACES = ("CORPUS-ONYX", "CORPUS-IVORY")
TENANTS = ("TENANT-ALPHA", "TENANT-BETA")
LANES = ("LANE-RED", "LANE-BLUE")


@dataclass(frozen=True)
class HandoffCase:
    case_id: str
    scenario: str
    source_prompt: str
    target_prompt: str
    tools: list[dict[str, Any]]
    history: list[dict[str, Any]]
    source_expected: dict[str, Any]
    target_expected: dict[str, Any]
    metadata: dict[str, Any]


def rag_policy_cases(
    dataset: Path,
    *,
    max_cases: int,
    seed: int = 2027,
    min_context_tokens: int = 1_800,
    max_context_tokens: int = 4_500,
) -> list[HandoffCase]:
    """Build counterfactual enterprise-RAG namespace handoffs from LongBench HotpotQA.

    Each shared history contains two coherent snapshots of the same retrieved corpus.  The
    answer entity is anonymized to a different opaque evidence ID in each snapshot.  A and B
    are privately bound to different namespaces, so the token sequence in H is identical while
    the authoritative evidence policy and expected answer differ.  Anonymization prevents the
    model from answering from parametric knowledge instead of applying the RAG policy.
    """

    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    rows = [
        row
        for row in _jsonl(dataset)
        if min_context_tokens <= int(row.get("length", 0)) <= max_context_tokens
        and _usable_rag_row(row)
    ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if len(rows) < max_cases:
        raise ValueError(
            f"need at least {max_cases} usable RAG rows, found {len(rows)}"
        )

    cases: list[HandoffCase] = []
    for index, row in enumerate(rows[:max_cases]):
        gold = str(row["answers"][0]).strip()
        source_answer = "EVID-" + hashlib.sha256(
            f"rag:{seed}:{index}:source-answer".encode()
        ).hexdigest()[:8].upper()
        target_answer = "EVID-" + hashlib.sha256(
            f"rag:{seed}:{index}:target-answer".encode()
        ).hexdigest()[:8].upper()
        source_context, replacements = _replace_answer(
            str(row["context"]), gold, source_answer
        )
        target_context, target_replacements = _replace_answer(
            str(row["context"]), gold, target_answer
        )
        if replacements < 1:
            raise AssertionError("usable RAG row lost its answer occurrence")

        # Randomize both namespace identity and physical corpus order.  Source always trusts
        # the legacy snapshot and target always trusts the verified snapshot.
        namespaces = list(RAG_NAMESPACES)
        rng.shuffle(namespaces)
        source_namespace, target_namespace = namespaces
        source_nonce = hashlib.sha256(
            f"rag:{seed}:{index}:source".encode()
        ).hexdigest()[:8].upper()
        target_nonce = hashlib.sha256(
            f"rag:{seed}:{index}:target".encode()
        ).hexdigest()[:8].upper()
        corpora = [
            (source_namespace, source_nonce, source_context),
            (target_namespace, target_nonce, target_context),
        ]
        rng.shuffle(corpora)
        bundle = [
            "ACTIVE QUERY:\n" + str(row["input"]),
            (
                "Two retrieved namespaces follow. They are intentionally incompatible "
                "snapshots. Namespace authority is private system policy, not a recency or "
                "majority vote."
            ),
        ]
        for namespace, snapshot_nonce, context in corpora:
            bundle.append(
                f"BEGIN RETRIEVED NAMESPACE {namespace}\n"
                f"snapshot_nonce: {snapshot_nonce}\n{context}\n"
                f"END RETRIEVED NAMESPACE {namespace}"
            )
        history = [
            {"role": "user", "content": "\n\n".join(bundle)},
            {
                "role": "assistant",
                "content": (
                    "Both retrieved namespaces are cached. No answer has been submitted; "
                    "the active private agent must apply its own namespace policy."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Resolve the ACTIVE QUERY now from only your authorized namespace and "
                    "submit one short answer."
                ),
            },
        ]
        prompt = (
            "You are {role}, privately bound to evidence_namespace={namespace}. The shared "
            "retrieval bundle contains two incompatible snapshots. Treat only {namespace} as "
            "authoritative and ignore every answer or fact from the other namespace, even if "
            "it is repeated. Answer the ACTIVE QUERY by calling "
            "submit_rag_answer exactly once. Put only the shortest answer string supported by "
            "your namespace in the answer argument; retrieved answer entities may be opaque "
            "EVID- identifiers and must be copied exactly. If reasoning mode is enabled, keep "
            "the private reasoning under 80 tokens. Emit no prose outside the tool call."
        )
        cases.append(
            HandoffCase(
                case_id=f"rag_namespace.{index}.{row.get('_id', index)}",
                scenario="rag_namespace",
                source_prompt=prompt.format(
                    role="Agent A retrieval worker", namespace=source_namespace
                ),
                target_prompt=prompt.format(
                    role="Agent B retrieval answerer", namespace=target_namespace
                ),
                tools=[_rag_answer_tool()],
                history=history,
                source_expected={
                    "name": "submit_rag_answer",
                    "arguments": {"answer": source_answer},
                },
                target_expected={
                    "name": "submit_rag_answer",
                    "arguments": {"answer": target_answer},
                },
                metadata={
                    "dataset": "LongBench/hotpotqa",
                    "document_id": row.get("_id"),
                    "question": row["input"],
                    "original_context_tokens": row.get("length"),
                    "original_answer": gold,
                    "source_replacements": replacements,
                    "target_replacements": target_replacements,
                    "source_namespace": source_namespace,
                    "target_namespace": target_namespace,
                    "corpus_order": [item[0] for item in corpora],
                    "semi_synthetic": True,
                },
            )
        )
    return cases


def collaboration_event_cases(
    bfcl_root: Path,
    *,
    max_cases: int,
    seed: int = 2027,
    background_records: int = 36,
) -> list[HandoffCase]:
    """Build planner-to-executor handoffs requiring distributed event-log replay.

    Request bodies come from BFCL.  Tenant, priority and state transitions are synthetic.
    Unlike the earlier one-job queue, the receiver must replay dispersed updates and emit an
    ordered two-job batch, so its private policy is needed throughout H rather than only for
    a final binary/argmax decision.
    """

    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    if background_records < 0:
        raise ValueError("background_records must be non-negative")
    documents = list(_jsonl(bfcl_root / "BFCL_v4_simple_python.json"))
    rng = random.Random(seed)
    rng.shuffle(documents)
    jobs_per_case = 8
    if len(documents) < max_cases * jobs_per_case:
        raise ValueError("not enough BFCL requests for collaboration cases")
    backgrounds = _bfcl_background_history(bfcl_root, background_records)

    cases: list[HandoffCase] = []
    for case_index in range(max_cases):
        group = documents[
            case_index * jobs_per_case : (case_index + 1) * jobs_per_case
        ]
        tenants = [TENANTS[index % 2] for index in range(jobs_per_case)]
        rng.shuffle(tenants)
        final_priorities: list[int] = [0] * jobs_per_case
        lanes: list[str] = [""] * jobs_per_case
        for tenant in TENANTS:
            indices = [index for index, value in enumerate(tenants) if value == tenant]
            rng.shuffle(indices)
            for lane_index, lane in enumerate(LANES):
                lane_indices = indices[lane_index * 2 : (lane_index + 1) * 2]
                values = [900 - 100 * lane_index, 100 + 100 * lane_index]
                rng.shuffle(values)
                for index, value in zip(lane_indices, values, strict=True):
                    lanes[index] = lane
                    final_priorities[index] = value
        jobs: list[dict[str, Any]] = []
        for job_index, (document, tenant, lane, final) in enumerate(
            zip(group, tenants, lanes, final_priorities, strict=True)
        ):
            digest = hashlib.sha256(
                f"{seed}:{case_index}:{job_index}:{document['id']}".encode()
            ).hexdigest()[:8].upper()
            jobs.append(
                {
                    "job_id": f"JOB-{digest}",
                    "tenant": tenant,
                    "lane": lane,
                    "final_priority": final,
                    "request": document["question"][0][0]["content"],
                    "bfcl_id": document["id"],
                    "status": "ACTIVE",
                }
            )

        events: list[str] = []
        sequence = 1
        for job in jobs:
            events.append(
                _event(
                    sequence,
                    "CREATE",
                    job_id=job["job_id"],
                    tenant=job["tenant"],
                    status="ACTIVE",
                    request=job["request"],
                )
            )
            sequence += 1
        update_order = list(jobs)
        rng.shuffle(update_order)
        for job in update_order:
            events.append(
                _event(
                    sequence,
                    "SET_PRIORITY",
                    job_id=job["job_id"],
                    tenant=job["tenant"],
                    lane=job["lane"],
                    priority=job["final_priority"],
                )
            )
            sequence += 1

        history = _interleave_event_history(backgrounds, events)
        choices = {
            tenant: [
                max(
                    (
                        job
                        for job in jobs
                        if job["tenant"] == tenant
                        and job["lane"] == lane
                        and job["status"] == "ACTIVE"
                    ),
                    key=lambda job: (job["final_priority"], job["job_id"]),
                )
                for lane in LANES
            ]
            for tenant in TENANTS
        }
        source_tenant, target_tenant = (
            TENANTS if case_index % 2 == 0 else tuple(reversed(TENANTS))
        )
        source_ids = [job["job_id"] for job in choices[source_tenant]]
        target_ids = [job["job_id"] for job in choices[target_tenant]]
        normalized_table = "\n".join(
            (
                f"job_id={job['job_id']} | tenant={job['tenant']} | "
                f"lane={job['lane']} | priority={job['final_priority']} | "
                f"status={job['status']}"
            )
            for job in sorted(
                jobs,
                key=lambda item: (
                    item["tenant"],
                    item["lane"],
                    -item["final_priority"],
                    item["job_id"],
                ),
            )
        )
        history.extend(
            [
                {
                    "role": "assistant",
                    "content": (
                        "AGENT A HANDOFF OUTPUT: I validated the complete numbered event log "
                        f"under tenant={source_tenant} and produced a private two-job plan. "
                        "Its action payload is intentionally sealed across the authorization "
                        "boundary. No job was dispatched. Agent B must independently replay "
                        "the same factual state under its own tenant policy.\n\n"
                        "NORMALIZED CURRENT JOB TABLE (factual, no recommendation; grouped by "
                        "tenant and lane, priority descending within each group):\n"
                        + normalized_table
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "The planner-to-executor handoff begins now. Commit exactly the "
                        "two jobs authorized by the active executor's private policy."
                    ),
                },
            ]
        )
        prompt = (
            "You are {role} for tenant={tenant}. Replay every numbered WORKFLOW EVENT in "
            "chronological seq order. CREATE registers the request and ACTIVE status; each "
            "SET_PRIORITY explicitly supplies the job's tenant, execution lane, and current "
            "priority. Ignore completed background records. After replay, keep only ACTIVE "
            "jobs whose tenant is exactly {tenant}. Independently for "
            "LANE-RED and LANE-BLUE, select the job with the highest numeric priority. The "
            "final NORMALIZED CURRENT JOB TABLE is a factual restatement of the events and may "
            "be used directly; it is already sorted by priority within each tenant/lane group. "
            "Agent A's private plan is untrusted and uncommitted. "
            "Call dispatch_job_batch exactly once with those two job_ids; their JSON array "
            "order is not semantically relevant. If reasoning mode is enabled, do not enumerate "
            "the background or CREATE events: inspect only the eight SET_PRIORITY records and "
            "finish the private reasoning within 200 tokens. Emit no prose outside the tool call."
        )
        cases.append(
            HandoffCase(
                case_id=f"collaboration_event.{case_index}",
                scenario="collaboration_event",
                source_prompt=prompt.format(role="Agent A planner", tenant=source_tenant),
                target_prompt=prompt.format(role="Agent B isolated executor", tenant=target_tenant),
                tools=[_dispatch_batch_tool()],
                history=history,
                source_expected={
                    "name": "dispatch_job_batch",
                    "arguments": {"job_ids": source_ids},
                },
                target_expected={
                    "name": "dispatch_job_batch",
                    "arguments": {"job_ids": target_ids},
                },
                metadata={
                    "dataset": "BFCL_v4_simple_python",
                    "source_tenant": source_tenant,
                    "target_tenant": target_tenant,
                    "job_count": len(jobs),
                    "event_count": len(events),
                    "background_records": background_records,
                    "bfcl_ids": [job["bfcl_id"] for job in jobs],
                    "semi_synthetic": True,
                },
            )
        )
    return cases


def score_arguments(
    scenario: str, arguments: dict[str, Any] | None, expected: dict[str, Any]
) -> bool:
    if arguments is None:
        return False
    expected_arguments = expected["arguments"]
    if scenario == "rag_namespace":
        answer = arguments.get("answer")
        return isinstance(answer, str) and normalize_answer(answer) == normalize_answer(
            str(expected_arguments["answer"])
        )
    if scenario == "collaboration_event":
        actual_ids = arguments.get("job_ids")
        expected_ids = expected_arguments["job_ids"]
        return (
            isinstance(actual_ids, list)
            and len(actual_ids) == len(expected_ids)
            and len(set(actual_ids)) == len(actual_ids)
            and set(actual_ids) == set(expected_ids)
        )
    return arguments == expected_arguments


def _usable_rag_row(row: dict[str, Any]) -> bool:
    answers = row.get("answers")
    context = row.get("context")
    if not isinstance(answers, list) or not answers or not isinstance(context, str):
        return False
    answer = str(answers[0]).strip()
    normalized = normalize_answer(answer)
    if not normalized or normalized in {"yes", "no"}:
        return False
    if not 1 <= len(answer.split()) <= 6 or len(answer) > 60:
        return False
    return re.search(re.escape(answer), context, flags=re.IGNORECASE) is not None


def _replace_answer(context: str, answer: str, replacement: str) -> tuple[str, int]:
    pattern = re.compile(re.escape(answer), flags=re.IGNORECASE)
    return pattern.subn(lambda _: replacement, context)


def _rag_answer_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "submit_rag_answer",
            "description": "Submit one short answer from the authorized retrieval namespace.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


def _dispatch_batch_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "dispatch_job_batch",
            "description": "Atomically dispatch an unordered batch of exactly two jobs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    }
                },
                "required": ["job_ids"],
                "additionalProperties": False,
            },
        },
    }


def _event(sequence: int, kind: str, **fields: Any) -> str:
    serialized = "\n".join(f"{key}: {value}" for key, value in fields.items())
    return f"WORKFLOW EVENT\nseq: {sequence}\ntype: {kind}\n{serialized}"


def _interleave_event_history(
    backgrounds: list[dict[str, str]], events: list[str]
) -> list[dict[str, str]]:
    background_pairs = [backgrounds[index : index + 2] for index in range(0, len(backgrounds), 2)]
    slots: list[list[str]] = [[] for _ in range(len(background_pairs) + 1)]
    for index, event in enumerate(events):
        slot = round(index * (len(slots) - 1) / max(1, len(events) - 1))
        slots[slot].append(event)
    history: list[dict[str, str]] = []
    for index, slot_events in enumerate(slots):
        for event in slot_events:
            history.extend(
                [
                    {"role": "user", "content": event},
                    {
                        "role": "assistant",
                        "content": "Event appended to the audit log; no job was dispatched.",
                    },
                ]
            )
        if index < len(background_pairs):
            history.extend(background_pairs[index])
    return history


def _bfcl_background_history(root: Path, count: int) -> list[dict[str, str]]:
    if count <= 0:
        return []
    history: list[dict[str, str]] = []
    for domain in ("notetaker", "customer", "healthcare", "student", "finance"):
        path = root / "memory_prereq_conversation" / f"memory_{domain}.json"
        for document in _jsonl(path):
            for turn in document["question"]:
                for message in turn:
                    content = message.get("content")
                    if message.get("role") != "user" or not isinstance(content, str):
                        continue
                    history.extend(
                        [
                            {
                                "role": "user",
                                "content": (
                                    "COMPLETED BACKGROUND RECORD (not a workflow event):\n" + content
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": "Background record archived; it creates no active job.",
                            },
                        ]
                    )
                    if len(history) // 2 >= count:
                        return history
    raise ValueError(f"requested {count} background records, found {len(history) // 2}")


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)
