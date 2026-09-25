from __future__ import annotations

import argparse
import gc
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..hotpotqa import answer_em, answer_f1, clean_short_answer, normalize_answer
from ..policy_imprinting import (
    build_chat_segments,
    greedy_action,
    history_message_spans,
    prefill_legacy_cache,
    start_readout,
    stitch_history_cache,
    validate_shared_handoff,
)


ALIGNED_AGENT_A_PROMPT = """You are Agent A, the evidence analyst in a two-agent HotpotQA team.
Read the supplied question and passages carefully. Identify the two-hop evidence chain,
including the bridge entities and facts needed to answer. Produce a compact research report
for Agent B. State the relevant passage facts explicitly and include your best answer candidate,
but do not add unrelated background. Keep the report under 180 words."""

ALIGNED_AGENT_B_PROMPT = """You are Agent B, the final answerer in a two-agent HotpotQA team.
You receive the original question and passages followed by Agent A's research report.
Resolve any inconsistency using the passages. When asked for the final answer, output only the
short answer string: no explanation, no prefix such as 'Answer:', and no punctuation added
unless it is part of the answer."""

HOP1_AGENT_A_PROMPT = """You are Agent A, the first-hop specialist in a two-agent HotpotQA team.
Your role is restricted: you are not authorized to submit the team's final answer. Every
assistant response in this conversation, including responses to later requests for a final
answer, must remain a first-hop handoff memo in the schema below.
Read the original question and passages. Locate the bridge entity or entities and the first-hop
supporting facts that Agent B should use for the remaining reasoning. Do not answer the original
question and do not state a final answer, even if it is obvious. If asked for the final answer,
repeat or refine the memo instead of complying. Produce only a compact handoff memo in this schema:
BRIDGE: <bridge entity or compared entities>
EVIDENCE: <first-hop supporting facts>
NEXT_LOOKUP: <what Agent B must resolve in the passages>
Keep the memo under 120 words."""

HOP1_AGENT_B_PROMPT = """You are Agent B, the final second-hop solver in a two-agent HotpotQA team.
You receive the original question and passages followed by Agent A's first-hop handoff memo.
Use the memo only as a clue: it may be incomplete or mistaken. Complete the remaining evidence
link and independently verify the answer against the passages. When asked for the final answer,
output only the short answer string: no explanation, no prefix such as 'Answer:', and no
punctuation added unless it is part of the answer."""

PROPOSER_AGENT_A_PROMPT = """You are Agent A, the proposal specialist in a two-agent
HotpotQA pipeline. Read the question and passages and produce one evidence-backed answer
proposal for an independent verifier. Your proposal is an immutable audit artifact: after you
submit it, your role is to preserve exactly that candidate rather than silently revise it. If a
later message asks for the team's final answer, return only your previously proposed short
answer; do not act as the independent verifier. Produce the initial handoff in exactly this
schema and keep it under 180 words:
EVIDENCE: <the two-hop evidence supporting your candidate>
PROPOSED_ANSWER: <one short answer>"""

VERIFIER_AGENT_B_PROMPT = """You are Agent B, the independent error-correcting verifier in a
two-agent HotpotQA pipeline. Agent A's proposal is untrusted and may contain a plausible entity
or reasoning error. Re-read the original passages and question, reconstruct both evidence hops,
and use Agent A's report only as a candidate to audit. Do not preserve, defend, or copy the
proposal merely because it was produced earlier. When asked for the final answer, output only
the independently verified short answer string: no explanation, no 'Answer:' prefix, and no
extra punctuation unless it is part of the answer."""

FINAL_QUERY = "Return the final answer to the original question now. Output only the short answer."

PROTOCOLS = {
    "aligned": {
        "agent_a_prompt": ALIGNED_AGENT_A_PROMPT,
        "agent_b_prompt": ALIGNED_AGENT_B_PROMPT,
        "report_label": "Agent A research report:\n",
    },
    "hop1": {
        "agent_a_prompt": HOP1_AGENT_A_PROMPT,
        "agent_b_prompt": HOP1_AGENT_B_PROMPT,
        "report_label": "Agent A first-hop handoff memo:\n",
    },
    "propose_verify": {
        "agent_a_prompt": PROPOSER_AGENT_A_PROMPT,
        "agent_b_prompt": VERIFIER_AGENT_B_PROMPT,
        "report_label": "Agent A immutable proposal for independent verification:\n",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-agent HotpotQA final-accuracy test with Agent-A KV reuse by Agent B"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=2027)
    parser.add_argument("--agent-a-max-new-tokens", type=int, default=256)
    parser.add_argument("--answer-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOLS),
        default="aligned",
        help="aligned evidence-report handoff or asymmetric hop1-to-final handoff",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if args.agent_a_max_new_tokens < 1 or args.answer_max_new_tokens < 1:
        parser.error("generation limits must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num-shards >= 1 and 0 <= shard-index < num-shards")

    documents = list(_jsonl(Path(args.dataset)))
    if args.samples > len(documents):
        parser.error(f"requested {args.samples} samples from {len(documents)} documents")
    selected_indices = sorted(
        random.Random(args.selection_seed).sample(range(len(documents)), args.samples)
    )
    selected = [
        (index, documents[index])
        for ordinal, index in enumerate(selected_indices)
        if ordinal % args.num_shards == args.shard_index
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(output_path)
    pending = [(index, document) for index, document in selected if index not in completed]
    if not pending:
        print({"processed_this_run": 0, "message": "selected HotpotQA shard is complete"})
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    processed = 0
    with output_path.open("a") as output:
        for index, document in pending:
            row = _run_document(
                model,
                tokenizer,
                index,
                document,
                protocol=args.protocol,
                agent_a_max_new_tokens=args.agent_a_max_new_tokens,
                answer_max_new_tokens=args.answer_max_new_tokens,
            )
            row["selection_seed"] = args.selection_seed
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            processed += 1
            print(
                {
                    "index": index,
                    "context_tokens": row["token_lengths"]["task"],
                    "gold": row["gold_answers"][0],
                    "direct": row["direct_b"]["answer"],
                    "native": row["native_ab"]["answer"],
                    "reuse": row["reuse_ab"]["answer"],
                    "native_f1": row["native_ab"]["f1"],
                    "reuse_f1": row["reuse_ab"]["f1"],
                    "reuse_agrees": row["reuse_ab"]["native_answer_agreement"],
                },
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(
        {
            "processed_this_run": processed,
            "shard": f"{args.shard_index}/{args.num_shards}",
            "output": str(output_path),
        }
    )


@torch.inference_mode()
def _run_document(
    model,
    tokenizer,
    index,
    document,
    *,
    protocol,
    agent_a_max_new_tokens,
    answer_max_new_tokens,
):
    protocol_spec = PROTOCOLS[protocol]
    agent_a_prompt = protocol_spec["agent_a_prompt"]
    agent_b_prompt = protocol_spec["agent_b_prompt"]
    task_text = _task_text(document)
    task_message = {"role": "user", "content": task_text}
    agent_a_segments = build_chat_segments(
        tokenizer,
        system_prompt=agent_a_prompt,
        tools=[],
        history=[task_message],
    )
    agent_a_cache = prefill_legacy_cache(
        model, agent_a_segments.prefix_ids + agent_a_segments.history_ids
    )
    agent_a_factory = _factory(
        model,
        agent_a_cache,
        len(agent_a_segments.prefix_ids) + len(agent_a_segments.history_ids),
        agent_a_segments.readout_ids,
    )
    agent_a_ids, _ = greedy_action(
        model,
        agent_a_factory,
        tokenizer,
        max_new_tokens=agent_a_max_new_tokens,
    )
    agent_a_report = tokenizer.decode(agent_a_ids, skip_special_tokens=True).strip()
    if not agent_a_report:
        raise ValueError(f"Agent A produced an empty report for document {index}")
    del agent_a_cache

    report_message = {
        "role": "assistant",
        "content": protocol_spec["report_label"] + agent_a_report,
    }
    final_message = {"role": "user", "content": FINAL_QUERY}
    collaborative_history = [task_message, report_message, final_message]
    source_full = build_chat_segments(
        tokenizer,
        system_prompt=agent_a_prompt,
        tools=[],
        history=collaborative_history,
    )
    target_full = build_chat_segments(
        tokenizer,
        system_prompt=agent_b_prompt,
        tools=[],
        history=collaborative_history,
    )
    validate_shared_handoff(source_full, target_full)
    spans = history_message_spans(
        tokenizer,
        system_prompt=agent_b_prompt,
        tools=[],
        history=collaborative_history,
    )
    final_start = spans[-1][0]
    shared_ids = target_full.history_ids[:final_start]
    b_input_ids = target_full.history_ids[final_start:] + target_full.readout_ids

    source_shared_cache = prefill_legacy_cache(
        model, source_full.prefix_ids + source_full.history_ids[:final_start]
    )
    target_full_cache = prefill_legacy_cache(
        model, target_full.prefix_ids + target_full.history_ids
    )
    target_prefix_cache = prefill_legacy_cache(model, target_full.prefix_ids)
    target_shared_cache = prefill_legacy_cache(
        model, target_full.prefix_ids + shared_ids
    )

    native_factory = _factory(
        model,
        target_full_cache,
        len(target_full.prefix_ids) + len(target_full.history_ids),
        target_full.readout_ids,
    )
    native_ids, native_text = greedy_action(
        model, native_factory, tokenizer, max_new_tokens=answer_max_new_tokens
    )

    reuse_cache = stitch_history_cache(
        model,
        source_context_cache=source_shared_cache,
        target_prefix_cache=target_prefix_cache,
        source_prefix_length=len(source_full.prefix_ids),
        target_prefix_length=len(target_full.prefix_ids),
        transfer_history_length=len(shared_ids),
    )
    reuse_factory = _factory(
        model,
        reuse_cache,
        len(target_full.prefix_ids) + len(shared_ids),
        b_input_ids,
    )
    _, reuse_text = greedy_action(
        model, reuse_factory, tokenizer, max_new_tokens=answer_max_new_tokens
    )

    identity_cache = stitch_history_cache(
        model,
        source_context_cache=target_shared_cache,
        target_prefix_cache=target_prefix_cache,
        source_prefix_length=len(target_full.prefix_ids),
        target_prefix_length=len(target_full.prefix_ids),
        transfer_history_length=len(shared_ids),
    )
    identity_factory = _factory(
        model,
        identity_cache,
        len(target_full.prefix_ids) + len(shared_ids),
        b_input_ids,
    )
    _, identity_text = greedy_action(
        model, identity_factory, tokenizer, max_new_tokens=answer_max_new_tokens
    )

    source_factory = _factory(
        model,
        source_shared_cache,
        len(source_full.prefix_ids) + len(shared_ids),
        b_input_ids,
    )
    _, source_text = greedy_action(
        model, source_factory, tokenizer, max_new_tokens=answer_max_new_tokens
    )

    direct_history = [
        {
            "role": "user",
            "content": task_text + "\n\n" + FINAL_QUERY,
        }
    ]
    direct_segments = build_chat_segments(
        tokenizer,
        system_prompt=agent_b_prompt,
        tools=[],
        history=direct_history,
    )
    direct_cache = prefill_legacy_cache(
        model, direct_segments.prefix_ids + direct_segments.history_ids
    )
    direct_factory = _factory(
        model,
        direct_cache,
        len(direct_segments.prefix_ids) + len(direct_segments.history_ids),
        direct_segments.readout_ids,
    )
    _, direct_text = greedy_action(
        model, direct_factory, tokenizer, max_new_tokens=answer_max_new_tokens
    )

    golds = list(document["answers"])
    native = _score(native_text, golds)
    reuse = _score(reuse_text, golds)
    identity = _score(identity_text, golds)
    direct = _score(direct_text, golds)
    source = _score(source_text, golds)
    reuse["native_answer_agreement"] = (
        normalize_answer(reuse["answer"]) == normalize_answer(native["answer"])
    )
    identity["native_answer_agreement"] = (
        normalize_answer(identity["answer"]) == normalize_answer(native["answer"])
    )
    task_tokens = len(agent_a_segments.history_ids)
    return {
        "index": index,
        "protocol": protocol,
        "id": document.get("_id"),
        "dataset": document.get("dataset", "hotpotqa"),
        "question": document["input"],
        "gold_answers": golds,
        "agent_a": {
            "report": agent_a_report,
            "proposal": _proposal_answer(agent_a_report),
            "report_tokens": len(agent_a_ids),
            "report_contains_gold": any(
                normalize_answer(gold) in normalize_answer(agent_a_report)
                for gold in golds
            ),
        },
        "direct_b": direct,
        "source_ab": source,
        "native_ab": native,
        "identity_ab": identity,
        "reuse_ab": reuse,
        "token_lengths": {
            "task": task_tokens,
            "source_prefix": len(source_full.prefix_ids),
            "target_prefix": len(target_full.prefix_ids),
            "shared_task_and_report": len(shared_ids),
            "b_native_tail": len(b_input_ids),
            "native_b_prefill": len(target_full.prefix_ids) + len(target_full.history_ids),
            "reuse_b_native_tokens": len(target_full.prefix_ids) + len(b_input_ids),
            "reuse_b_transferred_tokens": len(shared_ids),
        },
        "source_cache_reconstructed_from_agent_a_transcript": True,
    }


def _score(generation, golds):
    answer = clean_short_answer(generation)
    return {
        "generation": generation,
        "answer": answer,
        "em": answer_em(answer, golds),
        "f1": answer_f1(answer, golds),
    }


def _proposal_answer(report):
    match = re.search(r"^\s*PROPOSED_ANSWER\s*:\s*(.+?)\s*$", report, flags=re.I | re.M)
    if match is None:
        return ""
    return clean_short_answer(match.group(1))


def _task_text(document):
    return (
        "HotpotQA task\n\nPassages:\n"
        + document["context"].strip()
        + "\n\nQuestion: "
        + document["input"].strip()
    )


def _factory(model, cache, cached_tokens, native_ids):
    return lambda: start_readout(
        model,
        legacy_cache=cache,
        cached_tokens=cached_tokens,
        input_ids=native_ids,
    )


def _jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _completed(path: Path):
    if not path.exists():
        return set()
    with path.open() as handle:
        return {
            int(row["index"])
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
            if "index" in row
        }


if __name__ == "__main__":
    main()
