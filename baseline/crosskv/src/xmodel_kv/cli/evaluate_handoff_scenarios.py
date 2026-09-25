from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..handoff_scenarios import (
    HandoffCase,
    collaboration_event_cases,
    rag_policy_cases,
    score_arguments,
)
from ..policy_imprinting import (
    build_chat_segments,
    greedy_action,
    parse_tool_call,
    prefill_legacy_cache,
    start_readout,
    stitch_history_cache,
    validate_shared_handoff,
)


NEUTRAL_PROMPT = (
    "You are a neutral shared-context cache host. Preserve the retrieval evidence and workflow "
    "audit text without adopting an evidence namespace, tenant, role, or selection policy. A "
    "private receiving-agent policy may be supplied in a later message; follow that policy when "
    "it appears."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate front/tail plaintext and KV-reuse controls on RAG and multi-agent "
            "handoff scenarios"
        )
    )
    parser.add_argument("--scenario", choices=("rag", "collaboration", "both"), default="both")
    parser.add_argument(
        "--rag-dataset",
        default="datasets/longbench/hotpotqa.jsonl",
        help="Path to the LongBench HotpotQA JSONL used by the RAG scenario",
    )
    parser.add_argument(
        "--bfcl-root",
        default=(
            "datasets/bfcl_official/berkeley-function-call-leaderboard/"
            "bfcl_eval/data"
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--rag-cases", type=int, default=10)
    parser.add_argument("--collaboration-cases", type=int, default=20)
    parser.add_argument("--background-records", type=int, default=36)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--tail-role", choices=("system", "user"), default="system")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Use the model chat template's native reasoning mode before the tool call.",
    )
    args = parser.parse_args()
    if min(args.rag_cases, args.collaboration_cases, args.max_new_tokens) < 1:
        parser.error("case counts and max-new-tokens must be positive")

    cases: list[HandoffCase] = []
    if args.scenario in {"rag", "both"}:
        cases.extend(
            rag_policy_cases(
                Path(args.rag_dataset), max_cases=args.rag_cases, seed=args.seed
            )
        )
    if args.scenario in {"collaboration", "both"}:
        cases.extend(
            collaboration_event_cases(
                Path(args.bfcl_root),
                max_cases=args.collaboration_cases,
                seed=args.seed,
                background_records=args.background_records,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {row["case_id"] for row in _jsonl(output_path)} if output_path.exists() else set()
    pending = [case for case in cases if case.case_id not in completed]
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=getattr(torch, args.dtype),
            device_map={"": args.device},
            attn_implementation="sdpa",
        ).eval()
        with output_path.open("a") as output:
            for case in pending:
                row = evaluate_case(
                    model,
                    tokenizer,
                    case,
                    tail_role=args.tail_role,
                    max_new_tokens=args.max_new_tokens,
                    enable_thinking=args.enable_thinking,
                )
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    {
                        "case_id": case.case_id,
                        "history_tokens": row["token_lengths"]["history"],
                        **{
                            arm: (
                                row["arms"][arm]["target_correct"],
                                row["arms"][arm]["source_correct"],
                            )
                            for arm in (
                                "front_full_reprefill",
                                "front_kv_reuse",
                                "tail_full_reprefill",
                                "tail_kv_reuse",
                            )
                        },
                    },
                    flush=True,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del model

    rows = list(_jsonl(output_path))
    selected_ids = {case.case_id for case in cases}
    rows = [row for row in rows if row["case_id"] in selected_ids]
    summary = summarize(rows)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


@torch.inference_mode()
def evaluate_case(
    model,
    tokenizer,
    case: HandoffCase,
    *,
    tail_role: str = "system",
    max_new_tokens: int = 128,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    source = build_chat_segments(
        tokenizer,
        system_prompt=case.source_prompt,
        tools=case.tools,
        history=case.history,
        enable_thinking=enable_thinking,
    )
    target = build_chat_segments(
        tokenizer,
        system_prompt=case.target_prompt,
        tools=case.tools,
        history=case.history,
        enable_thinking=enable_thinking,
    )
    neutral = build_chat_segments(
        tokenizer,
        system_prompt=NEUTRAL_PROMPT,
        tools=case.tools,
        history=case.history,
        enable_thinking=enable_thinking,
    )
    validate_shared_handoff(source, target)
    validate_shared_handoff(source, neutral)

    source_context = prefill_legacy_cache(model, source.prefix_ids + source.history_ids)
    target_context = prefill_legacy_cache(model, target.prefix_ids + target.history_ids)
    target_prefix = prefill_legacy_cache(model, target.prefix_ids)

    arms: dict[str, dict[str, Any]] = {}
    arms["source_front_native"] = _generate_and_score(
        model,
        tokenizer,
        case,
        _factory(model, source_context, len(source.prefix_ids) + len(source.history_ids), source.readout_ids),
        max_new_tokens=max_new_tokens,
    )
    arms["front_full_reprefill"] = _generate_and_score(
        model,
        tokenizer,
        case,
        _factory(model, target_context, len(target.prefix_ids) + len(target.history_ids), target.readout_ids),
        max_new_tokens=max_new_tokens,
    )

    front_identity_cache = stitch_history_cache(
        model,
        source_context_cache=target_context,
        target_prefix_cache=target_prefix,
        source_prefix_length=len(target.prefix_ids),
        target_prefix_length=len(target.prefix_ids),
        transfer_history_length=len(target.history_ids),
    )
    arms["front_identity"] = _generate_and_score(
        model,
        tokenizer,
        case,
        _factory(
            model,
            front_identity_cache,
            len(target.prefix_ids) + len(target.history_ids),
            target.readout_ids,
        ),
        max_new_tokens=max_new_tokens,
    )
    front_reuse_cache = stitch_history_cache(
        model,
        source_context_cache=source_context,
        target_prefix_cache=target_prefix,
        source_prefix_length=len(source.prefix_ids),
        target_prefix_length=len(target.prefix_ids),
        transfer_history_length=len(source.history_ids),
    )
    arms["front_kv_reuse"] = _generate_and_score(
        model,
        tokenizer,
        case,
        _factory(
            model,
            front_reuse_cache,
            len(target.prefix_ids) + len(target.history_ids),
            target.readout_ids,
        ),
        max_new_tokens=max_new_tokens,
    )
    del target_context, target_prefix, front_identity_cache, front_reuse_cache

    tail_message = {"role": tail_role, "content": case.target_prompt}
    neutral_tail = build_chat_segments(
        tokenizer,
        system_prompt=NEUTRAL_PROMPT,
        tools=case.tools,
        history=[*case.history, tail_message],
        enable_thinking=enable_thinking,
    )
    tail_ids = _appended_history_ids(neutral, neutral_tail)
    neutral_context = prefill_legacy_cache(
        model, neutral.prefix_ids + neutral.history_ids
    )
    neutral_prefix = prefill_legacy_cache(model, neutral.prefix_ids)
    tail_context = prefill_legacy_cache(
        model, neutral_tail.prefix_ids + neutral_tail.history_ids
    )
    arms["tail_full_reprefill"] = _generate_and_score(
        model,
        tokenizer,
        case,
        _factory(
            model,
            tail_context,
            len(neutral_tail.prefix_ids) + len(neutral_tail.history_ids),
            neutral_tail.readout_ids,
        ),
        max_new_tokens=max_new_tokens,
    )

    tail_identity_cache = stitch_history_cache(
        model,
        source_context_cache=neutral_context,
        target_prefix_cache=neutral_prefix,
        source_prefix_length=len(neutral.prefix_ids),
        target_prefix_length=len(neutral.prefix_ids),
        transfer_history_length=len(neutral.history_ids),
    )
    arms["tail_identity"] = _generate_and_score(
        model,
        tokenizer,
        case,
        _factory(
            model,
            tail_identity_cache,
            len(neutral.prefix_ids) + len(neutral.history_ids),
            tail_ids + neutral_tail.readout_ids,
        ),
        max_new_tokens=max_new_tokens,
    )
    tail_reuse_cache = stitch_history_cache(
        model,
        source_context_cache=source_context,
        target_prefix_cache=neutral_prefix,
        source_prefix_length=len(source.prefix_ids),
        target_prefix_length=len(neutral.prefix_ids),
        transfer_history_length=len(source.history_ids),
    )
    arms["tail_kv_reuse"] = _generate_and_score(
        model,
        tokenizer,
        case,
        _factory(
            model,
            tail_reuse_cache,
            len(neutral.prefix_ids) + len(neutral.history_ids),
            tail_ids + neutral_tail.readout_ids,
        ),
        max_new_tokens=max_new_tokens,
    )

    # Strong protocol-redesign control: late-bind both B's complete private policy and the
    # small current handoff request.  The long evidence/event history is still not replayed.
    tail_request_message = {
        "role": tail_role,
        "content": case.target_prompt + "\n\n" + _current_handoff_request(case),
    }
    neutral_tail_request = build_chat_segments(
        tokenizer,
        system_prompt=NEUTRAL_PROMPT,
        tools=case.tools,
        history=[*case.history, tail_request_message],
        enable_thinking=enable_thinking,
    )
    tail_request_ids = _appended_history_ids(neutral, neutral_tail_request)
    tail_request_context = prefill_legacy_cache(
        model,
        neutral_tail_request.prefix_ids + neutral_tail_request.history_ids,
    )
    arms["tail_request_full_reprefill"] = _generate_and_score(
        model,
        tokenizer,
        case,
        _factory(
            model,
            tail_request_context,
            len(neutral_tail_request.prefix_ids)
            + len(neutral_tail_request.history_ids),
            neutral_tail_request.readout_ids,
        ),
        max_new_tokens=max_new_tokens,
    )
    arms["tail_request_identity"] = _generate_and_score(
        model,
        tokenizer,
        case,
        _factory(
            model,
            tail_identity_cache,
            len(neutral.prefix_ids) + len(neutral.history_ids),
            tail_request_ids + neutral_tail_request.readout_ids,
        ),
        max_new_tokens=max_new_tokens,
    )
    arms["tail_request_kv_reuse"] = _generate_and_score(
        model,
        tokenizer,
        case,
        _factory(
            model,
            tail_reuse_cache,
            len(neutral.prefix_ids) + len(neutral.history_ids),
            tail_request_ids + neutral_tail_request.readout_ids,
        ),
        max_new_tokens=max_new_tokens,
    )
    del (
        source_context,
        neutral_context,
        neutral_prefix,
        tail_context,
        tail_identity_cache,
        tail_reuse_cache,
        tail_request_context,
    )

    return {
        "case_id": case.case_id,
        "scenario": case.scenario,
        "tail_role": tail_role,
        "enable_thinking": enable_thinking,
        "source_prompt": case.source_prompt,
        "target_prompt": case.target_prompt,
        "source_expected": case.source_expected,
        "target_expected": case.target_expected,
        "metadata": case.metadata,
        "token_lengths": {
            "source_prefix": len(source.prefix_ids),
            "target_prefix": len(target.prefix_ids),
            "neutral_prefix": len(neutral.prefix_ids),
            "history": len(source.history_ids),
            "target_tail": len(tail_ids),
            "target_tail_with_request": len(tail_request_ids),
            "readout": len(target.readout_ids),
            "front_reuse_fraction": len(source.history_ids)
            / (len(target.prefix_ids) + len(source.history_ids) + len(target.readout_ids)),
            "tail_reuse_fraction": len(source.history_ids)
            / (
                len(neutral.prefix_ids)
                + len(source.history_ids)
                + len(tail_ids)
                + len(neutral_tail.readout_ids)
            ),
        },
        "arms": arms,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios = sorted({row["scenario"] for row in rows})
    return {
        "rows": len(rows),
        "scenarios": {
            scenario: _summarize_group(
                [row for row in rows if row["scenario"] == scenario]
            )
            for scenario in scenarios
        },
    }


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    arm_names = tuple(rows[0]["arms"])
    strict = [
        row
        for row in rows
        if row["arms"]["source_front_native"]["source_correct"]
        and row["arms"]["front_full_reprefill"]["target_correct"]
        and row["arms"]["front_identity"]["target_correct"]
    ]
    return {
        "n": len(rows),
        "history_tokens": {
            "min": min(row["token_lengths"]["history"] for row in rows),
            "max": max(row["token_lengths"]["history"] for row in rows),
            "mean": sum(row["token_lengths"]["history"] for row in rows) / len(rows),
        },
        "all": {
            arm: _arm_counts(rows, arm)
            for arm in arm_names
        },
        "strict": {
            "n": len(strict),
            "definition": "source-native, target-native, and front identity are correct",
            "arms": {arm: _arm_counts(strict, arm) for arm in arm_names},
            "tail_plaintext_correct_but_tail_kv_wrong": sum(
                row["arms"]["tail_full_reprefill"]["target_correct"]
                and not row["arms"]["tail_kv_reuse"]["target_correct"]
                for row in strict
            ),
            "tail_request_plaintext_correct_but_tail_request_kv_wrong": sum(
                row["arms"]["tail_request_full_reprefill"]["target_correct"]
                and not row["arms"]["tail_request_kv_reuse"]["target_correct"]
                for row in strict
            ),
            "front_reuse_follows_source": sum(
                row["arms"]["front_kv_reuse"]["source_correct"] for row in strict
            ),
            "tail_reuse_follows_source": sum(
                row["arms"]["tail_kv_reuse"]["source_correct"] for row in strict
            ),
        },
    }


def _arm_counts(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    n = len(rows)
    target = sum(row["arms"][arm]["target_correct"] for row in rows)
    source = sum(row["arms"][arm]["source_correct"] for row in rows)
    return {
        "target_correct": target,
        "source_correct": source,
        "n": n,
        "target_accuracy": target / n if n else None,
        "source_rate": source / n if n else None,
    }


def _generate_and_score(
    model,
    tokenizer,
    case: HandoffCase,
    state_factory,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    _, text = greedy_action(
        model, state_factory, tokenizer, max_new_tokens=max_new_tokens
    )
    call = parse_tool_call(text)
    target_correct = (
        call.name == case.target_expected["name"]
        and score_arguments(case.scenario, call.arguments, case.target_expected)
    )
    source_correct = (
        call.name == case.source_expected["name"]
        and score_arguments(case.scenario, call.arguments, case.source_expected)
    )
    return {
        "generation": text,
        "tool_call": asdict(call),
        "target_correct": target_correct,
        "source_correct": source_correct,
    }


def _factory(model, cache, cached_tokens: int, input_ids):
    return lambda: start_readout(
        model,
        legacy_cache=cache,
        cached_tokens=cached_tokens,
        input_ids=input_ids,
    )


def _current_handoff_request(case: HandoffCase) -> str:
    if case.scenario == "rag_namespace":
        return (
            "CURRENT HANDOFF REQUEST (repeated, not part of the reusable corpus):\n"
            + str(case.metadata["question"])
            + "\nSubmit the authorized namespace's exact EVID- answer now."
        )
    return (
        "CURRENT HANDOFF REQUEST (repeated, not part of the reusable event log): "
        "select the winning active job in LANE-RED and LANE-BLUE for your tenant, then "
        "call dispatch_job_batch."
    )


def _appended_history_ids(base, appended) -> tuple[int, ...]:
    if base.prefix_ids != appended.prefix_ids:
        raise ValueError("appending target policy changed neutral prefix tokens")
    if appended.history_ids[: len(base.history_ids)] != base.history_ids:
        raise ValueError("appending target policy rewrote shared history tokens")
    tail = appended.history_ids[len(base.history_ids) :]
    if not tail:
        raise ValueError("target policy tail contains no tokens")
    return tail


def _jsonl(path: Path):
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
