from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..coqa import answer_from_generation, coqa_f1, gold_answers_at_turn, primary_answers, questions
from ..data import iter_records
from ..policy_imprinting import (
    build_chat_segments,
    greedy_action,
    history_message_spans,
    prefill_legacy_cache,
    start_readout,
)
from ..semantic_kv_summary import (
    TrainableKVSummary,
    concatenate_caches,
    mean_pool_segment_cache,
    teacher_forced_answer_logits,
)


AGENT_A_SYSTEM = (
    "You are Agent A, a careful reader. Read and internally retain the supplied "
    "source material so that another agent can later use your state."
)
AGENT_B_SYSTEM = (
    "You are Agent B. Answer each user question from the handed-off briefing state. "
    "Return only a concise answer without explanation."
)
ACK_B = "I have read and retained Agent A's briefing."
SELF_STUDY_PROMPTS = (
    "Write a compact factual synopsis of the source material. Preserve names, "
    "attributes, relationships, locations, quantities, and outcomes.",
    "List the important entities or characters in the source material and state "
    "their properties and relationships. Be concise but complete.",
    "Give a chronological outline of the events in the source material, including "
    "causes, actions, reactions, and final outcomes.",
    "Create terse downstream-agent notes containing concrete details that could be "
    "asked later, especially colors, counts, ownership, places, and who did what.",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle existence probe for short receiver-native semantic KV summaries. "
            "The base LLM is frozen; only per-instance KV slots are optimized."
        )
    )
    parser.add_argument("--dataset", required=True, help="CoQA validation parquet")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--conversation-indices", type=_parse_ints, default=(0,))
    parser.add_argument("--budgets", type=_parse_ints, default=(4, 8, 16, 32))
    parser.add_argument("--max-questions", type=int, default=12)
    parser.add_argument(
        "--train-turns",
        type=_parse_ints,
        help=(
            "optional 1-based CoQA turns used for oracle optimization; all remaining "
            "selected turns are held out from gradients"
        ),
    )
    parser.add_argument(
        "--training-mode",
        choices=("qa", "self_study"),
        default="qa",
        help=(
            "qa optimizes on selected benchmark questions; self_study uses only "
            "generic full-context teacher interactions and holds out every CoQA question"
        ),
    )
    parser.add_argument("--self-study-max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--initialization",
        choices=("source_pool", "teacher_pool"),
        default="source_pool",
        help=(
            "source_pool starts from Agent A's full-context KV; teacher_pool is an "
            "optimistic upper-bound initialized from the plaintext briefing KV"
        ),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--drift-weight", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    if not args.conversation_indices or min(args.conversation_indices) < 0:
        parser.error("conversation indices must be non-negative")
    if not args.budgets or min(args.budgets) < 1:
        parser.error("budgets must be positive")
    if args.train_turns and (min(args.train_turns) < 1 or max(args.train_turns) > args.max_questions):
        parser.error("train turns must lie inside [1, max-questions]")
    positive = (
        args.max_questions,
        args.steps,
        args.learning_rate,
        args.temperature,
        args.grad_clip,
        args.log_every,
        args.max_new_tokens,
        args.self_study_max_new_tokens,
    )
    if min(positive) <= 0:
        parser.error("question counts, steps, rates, temperature, and limits must be positive")
    if min(args.ce_weight, args.distill_weight, args.drift_weight, args.weight_decay) < 0:
        parser.error("loss weights and weight decay must be non-negative")
    if args.training_mode == "self_study" and args.train_turns:
        parser.error("--train-turns is only valid with --training-mode=qa")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**vars(args), "conversation_indices": list(args.conversation_indices), "budgets": list(args.budgets)}, indent=2)
        + "\n"
    )

    selected = _load_selected(args.dataset, args.conversation_indices)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    results_path = output_dir / "results.jsonl"
    with results_path.open("w") as result_file:
        for conversation_index, document in selected:
            prepared = _prepare_conversation(
                model,
                tokenizer,
                conversation_index=conversation_index,
                document=document,
                max_questions=args.max_questions,
                initialization=args.initialization,
            )
            prepared["training_mode"] = args.training_mode
            if args.training_mode == "qa":
                prepared["train_turns"] = tuple(
                    args.train_turns
                    if args.train_turns
                    else (example["turn"] for example in prepared["examples"])
                )
                train_turns = set(prepared["train_turns"])
                prepared["optimization_examples"] = [
                    example
                    for example in prepared["examples"]
                    if example["turn"] in train_turns
                ]
                prepared["self_study_rows"] = []
            else:
                prepared["train_turns"] = ()
                (
                    prepared["optimization_examples"],
                    prepared["self_study_rows"],
                ) = _prepare_self_study(
                    model,
                    tokenizer,
                    prepared,
                    document=document,
                    max_new_tokens=args.self_study_max_new_tokens,
                )
            print(
                json.dumps(
                    {
                        "event": "prepared",
                        "conversation_index": conversation_index,
                        "questions": len(prepared["examples"]),
                        "story_tokens": prepared["story_tokens"],
                        "briefing_tokens": prepared["briefing_tokens"],
                        "source_segment_tokens": prepared["source_segment_tokens"],
                        "teacher_segment_tokens": prepared["teacher_segment_tokens"],
                        "train_turns": prepared["train_turns"],
                        "training_mode": prepared["training_mode"],
                        "optimization_examples": len(prepared["optimization_examples"]),
                    }
                ),
                flush=True,
            )
            baselines = _evaluate_baselines(
                model,
                tokenizer,
                prepared,
                max_new_tokens=args.max_new_tokens,
            )
            for budget in args.budgets:
                if budget > prepared["initial_segment_tokens"]:
                    print(
                        json.dumps(
                            {
                                "event": "skip_budget",
                                "conversation_index": conversation_index,
                                "budget": budget,
                                "reason": "budget exceeds initialization segment length",
                            }
                        ),
                        flush=True,
                    )
                    continue
                row = _run_budget(
                    model,
                    tokenizer,
                    prepared,
                    budget=budget,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    ce_weight=args.ce_weight,
                    distill_weight=args.distill_weight,
                    temperature=args.temperature,
                    drift_weight=args.drift_weight,
                    grad_clip=args.grad_clip,
                    log_every=args.log_every,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                )
                row["baselines"] = baselines
                result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                result_file.flush()
                rows.append(row)
                print(json.dumps({"event": "result", **_compact_log(row)}), flush=True)
                gc.collect()
                torch.cuda.empty_cache()

    summary = _summarize(rows, config={**vars(args)})
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"event": "complete", "output": str(output_dir), "summary": summary["by_budget"]}), flush=True)


def _prepare_conversation(
    model,
    tokenizer,
    *,
    conversation_index: int,
    document: dict[str, Any],
    max_questions: int,
    initialization: str,
) -> dict[str, Any]:
    all_questions = questions(document)[:max_questions]
    all_answers = primary_answers(document)[:max_questions]
    if not all_questions or len(all_questions) != len(all_answers):
        raise ValueError(f"conversation {conversation_index} has invalid questions/answers")
    briefing = _gold_briefing(all_questions, all_answers)

    a_history = [
        {"role": "user", "content": f"Source material:\n{document['story']}"},
    ]
    b_history = [
        {"role": "user", "content": f"Agent A briefing:\n{briefing}"},
        {"role": "assistant", "content": ACK_B},
    ]
    a_segments = build_chat_segments(
        tokenizer,
        system_prompt=AGENT_A_SYSTEM,
        tools=[],
        history=a_history,
        enable_thinking=False,
    )
    a_system_ids = list(a_segments.prefix_ids)
    # Agent A's existing state is captured immediately before it would generate
    # a response, so the assistant readout header belongs to the source state.
    a_prefix_ids = [
        *a_segments.prefix_ids,
        *a_segments.history_ids,
        *a_segments.readout_ids,
    ]

    first_teacher = build_chat_segments(
        tokenizer,
        system_prompt=AGENT_B_SYSTEM,
        tools=[],
        history=[*b_history, {"role": "user", "content": all_questions[0]}],
        enable_thinking=False,
    )
    first_spans = history_message_spans(
        tokenizer,
        system_prompt=AGENT_B_SYSTEM,
        tools=[],
        history=[*b_history, {"role": "user", "content": all_questions[0]}],
        enable_thinking=False,
    )
    question_start = first_spans[-1][0]
    b_system_ids = list(first_teacher.prefix_ids)
    b_prefix_ids = [
        *first_teacher.prefix_ids,
        *first_teacher.history_ids[:question_start],
    ]

    a_cache = prefill_legacy_cache(model, a_prefix_ids)
    b_system_cache = prefill_legacy_cache(model, b_system_ids)
    teacher_cache = prefill_legacy_cache(model, b_prefix_ids)
    examples = []
    for turn, (question, answer) in enumerate(zip(all_questions, all_answers, strict=True), 1):
        student_segments = build_chat_segments(
            tokenizer,
            system_prompt=AGENT_B_SYSTEM,
            tools=[],
            history=[{"role": "user", "content": question}],
            enable_thinking=False,
        )
        teacher_segments = build_chat_segments(
            tokenizer,
            system_prompt=AGENT_B_SYSTEM,
            tools=[],
            history=[*b_history, {"role": "user", "content": question}],
            enable_thinking=False,
        )
        teacher_spans = history_message_spans(
            tokenizer,
            system_prompt=AGENT_B_SYSTEM,
            tools=[],
            history=[*b_history, {"role": "user", "content": question}],
            enable_thinking=False,
        )
        current_question_start = teacher_spans[-1][0]
        current_teacher_prefix = [
            *teacher_segments.prefix_ids,
            *teacher_segments.history_ids[:current_question_start],
        ]
        if current_teacher_prefix != b_prefix_ids:
            raise ValueError(f"teacher briefing prefix changed at turn {turn}")
        if list(student_segments.prefix_ids) != b_system_ids:
            raise ValueError(f"student system prefix changed at turn {turn}")
        student_query_ids = [
            *student_segments.history_ids,
            *student_segments.readout_ids,
        ]
        teacher_query_ids = [
            *teacher_segments.history_ids[current_question_start:],
            *teacher_segments.readout_ids,
        ]
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if not answer_ids:
            raise ValueError(f"conversation {conversation_index} turn {turn} has empty answer IDs")
        teacher_logits = teacher_forced_answer_logits(
            model,
            legacy_cache=teacher_cache,
            cached_tokens=len(b_prefix_ids),
            query_ids=teacher_query_ids,
            answer_ids=answer_ids,
        ).detach()
        examples.append(
            {
                "turn": turn,
                "question": question,
                "answer": answer,
                "gold_answers": gold_answers_at_turn(document, turn),
                "answer_ids": tuple(int(token) for token in answer_ids),
                "student_query_ids": tuple(student_query_ids),
                "teacher_query_ids": tuple(teacher_query_ids),
                "teacher_logits": teacher_logits,
            }
        )

    if initialization == "source_pool":
        initial_cache = a_cache
        initial_start = len(a_system_ids)
        initial_end = len(a_prefix_ids)
    elif initialization == "teacher_pool":
        initial_cache = teacher_cache
        initial_start = len(b_system_ids)
        initial_end = len(b_prefix_ids)
    else:
        raise ValueError(f"unknown initialization: {initialization}")

    return {
        "conversation_index": conversation_index,
        "conversation_id": document.get("id"),
        "domain": document.get("source"),
        "briefing": briefing,
        "story_tokens": len(tokenizer.encode(document["story"], add_special_tokens=False)),
        "briefing_tokens": len(tokenizer.encode(briefing, add_special_tokens=False)),
        "a_cache": a_cache,
        "a_system_tokens": len(a_system_ids),
        "a_prefix_tokens": len(a_prefix_ids),
        "b_system_cache": b_system_cache,
        "b_system_ids": tuple(b_system_ids),
        "b_system_tokens": len(b_system_ids),
        "teacher_cache": teacher_cache,
        "teacher_prefix_tokens": len(b_prefix_ids),
        "source_segment_tokens": len(a_prefix_ids) - len(a_system_ids),
        "teacher_segment_tokens": len(b_prefix_ids) - len(b_system_ids),
        "initial_cache": initial_cache,
        "initial_start": initial_start,
        "initial_end": initial_end,
        "initial_segment_tokens": initial_end - initial_start,
        "initialization": initialization,
        "examples": examples,
    }


@torch.inference_mode()
def _prepare_self_study(
    model,
    tokenizer,
    prepared: dict[str, Any],
    *,
    document: dict[str, Any],
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate context-distillation interactions without benchmark Q/A labels."""

    full_story_history = [
        {"role": "user", "content": f"Agent A source material:\n{document['story']}"},
        {"role": "assistant", "content": ACK_B},
    ]
    teacher_prefix_ids = None
    teacher_cache = None
    optimization_examples = []
    audit_rows = []
    stop_ids = {
        token_id
        for token_id in (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        )
        if isinstance(token_id, int) and token_id >= 0
    }
    for index, prompt in enumerate(SELF_STUDY_PROMPTS, 1):
        current_prefix, teacher_query_ids, teacher_system_ids = _split_chat_before_last_user(
            tokenizer,
            system_prompt=AGENT_B_SYSTEM,
            prior_history=full_story_history,
            query=prompt,
        )
        if teacher_prefix_ids is None:
            teacher_prefix_ids = current_prefix
            teacher_cache = prefill_legacy_cache(model, teacher_prefix_ids)
        elif current_prefix != teacher_prefix_ids:
            raise ValueError("self-study teacher prefix changed across prompts")
        if tuple(teacher_system_ids) != prepared["b_system_ids"]:
            raise ValueError("self-study teacher and student system prefixes differ")

        student_segments = build_chat_segments(
            tokenizer,
            system_prompt=AGENT_B_SYSTEM,
            tools=[],
            history=[{"role": "user", "content": prompt}],
            enable_thinking=False,
        )
        if list(student_segments.prefix_ids) != teacher_system_ids:
            raise ValueError("self-study student system prefix changed")
        student_query_ids = [
            *student_segments.history_ids,
            *student_segments.readout_ids,
        ]
        generated_ids, generated_text = _generate_ids(
            model,
            tokenizer,
            legacy_cache=teacher_cache,
            cached_tokens=len(teacher_prefix_ids),
            query_ids=teacher_query_ids,
            max_new_tokens=max_new_tokens,
        )
        answer_ids = list(generated_ids)
        while answer_ids and answer_ids[-1] in stop_ids:
            answer_ids.pop()
        if not answer_ids:
            raise ValueError(f"self-study prompt {index} generated no content tokens")
        response = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        teacher_logits = teacher_forced_answer_logits(
            model,
            legacy_cache=teacher_cache,
            cached_tokens=len(teacher_prefix_ids),
            query_ids=teacher_query_ids,
            answer_ids=answer_ids,
        ).detach()
        optimization_examples.append(
            {
                "name": f"self_study_{index}",
                "prompt": prompt,
                "answer_ids": tuple(int(token) for token in answer_ids),
                "student_query_ids": tuple(student_query_ids),
                "teacher_query_ids": tuple(teacher_query_ids),
                "teacher_logits": teacher_logits,
            }
        )
        audit_rows.append(
            {
                "name": f"self_study_{index}",
                "prompt": prompt,
                "response": response,
                "response_tokens": len(answer_ids),
                "hit_token_cap": len(generated_ids) >= max_new_tokens,
                "raw_generation": generated_text,
            }
        )
    return optimization_examples, audit_rows


def _split_chat_before_last_user(
    tokenizer,
    *,
    system_prompt: str,
    prior_history: Sequence[dict[str, str]],
    query: str,
) -> tuple[list[int], list[int], list[int]]:
    history = [*prior_history, {"role": "user", "content": query}]
    segments = build_chat_segments(
        tokenizer,
        system_prompt=system_prompt,
        tools=[],
        history=history,
        enable_thinking=False,
    )
    spans = history_message_spans(
        tokenizer,
        system_prompt=system_prompt,
        tools=[],
        history=history,
        enable_thinking=False,
    )
    query_start = spans[-1][0]
    prefix_ids = [*segments.prefix_ids, *segments.history_ids[:query_start]]
    query_ids = [*segments.history_ids[query_start:], *segments.readout_ids]
    return prefix_ids, query_ids, list(segments.prefix_ids)


def _run_budget(
    model,
    tokenizer,
    prepared: dict[str, Any],
    *,
    budget: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    ce_weight: float,
    distill_weight: float,
    temperature: float,
    drift_weight: float,
    grad_clip: float,
    log_every: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    model_dtype = model.model.embed_tokens.weight.dtype
    device = model.model.embed_tokens.weight.device
    initial = mean_pool_segment_cache(
        model,
        prepared["initial_cache"],
        segment_start=prepared["initial_start"],
        segment_end=prepared["initial_end"],
        target_start=prepared["b_system_tokens"],
        tokens=budget,
    )
    summary = TrainableKVSummary(initial).to(device=device)
    initial_parameters = tuple(parameter.detach().clone() for parameter in summary.parameters())
    initial_eval = _evaluate_summary(
        model,
        tokenizer,
        prepared,
        summary,
        max_new_tokens=max_new_tokens,
    )
    optimizer = AdamW(summary.parameters(), lr=learning_rate, weight_decay=weight_decay)
    order = list(range(len(prepared["optimization_examples"])))
    if not order:
        raise ValueError("no examples selected for optimization")
    generator = random.Random(seed + prepared["conversation_index"] * 1009 + budget)
    totals = {"loss": 0.0, "ce": 0.0, "distill": 0.0, "drift": 0.0}
    started = time.perf_counter()
    for step in range(1, steps + 1):
        if (step - 1) % len(order) == 0:
            generator.shuffle(order)
        example = prepared["optimization_examples"][order[(step - 1) % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        summary_cache = summary.cache(dtype=model_dtype, device=device)
        assembled = concatenate_caches(prepared["b_system_cache"], summary_cache)
        logits = teacher_forced_answer_logits(
            model,
            legacy_cache=assembled,
            cached_tokens=prepared["b_system_tokens"] + budget,
            query_ids=example["student_query_ids"],
            answer_ids=example["answer_ids"],
        )
        targets = torch.tensor(example["answer_ids"], dtype=torch.long, device=device)
        ce = F.cross_entropy(logits.squeeze(0).float(), targets)
        teacher = example["teacher_logits"].to(device=device, dtype=torch.float32)
        distill = F.kl_div(
            F.log_softmax(logits.float() / temperature, dim=-1),
            F.softmax(teacher / temperature, dim=-1),
            reduction="sum",
        ) * (temperature**2 / targets.numel())
        drift = torch.stack(
            [
                (parameter - origin).square().mean()
                for parameter, origin in zip(summary.parameters(), initial_parameters, strict=True)
            ]
        ).mean()
        loss = ce_weight * ce + distill_weight * distill + drift_weight * drift
        loss.backward()
        torch.nn.utils.clip_grad_norm_(summary.parameters(), grad_clip)
        optimizer.step()
        values = {"loss": loss, "ce": ce, "distill": distill, "drift": drift}
        for name, value in values.items():
            totals[name] += float(value.detach())
        if step == 1 or step % log_every == 0 or step == steps:
            print(
                json.dumps(
                    {
                        "event": "optimize",
                        "conversation_index": prepared["conversation_index"],
                        "budget": budget,
                        "step": step,
                        **{f"mean_{name}": total / step for name, total in totals.items()},
                    }
                ),
                flush=True,
            )
        del logits, assembled, summary_cache, loss, ce, distill, drift
    optimization_seconds = time.perf_counter() - started
    optimized_eval = _evaluate_summary(
        model,
        tokenizer,
        prepared,
        summary,
        max_new_tokens=max_new_tokens,
    )
    cache_elements = sum(parameter.numel() for parameter in summary.parameters())
    return {
        "conversation_index": prepared["conversation_index"],
        "conversation_id": prepared["conversation_id"],
        "domain": prepared["domain"],
        "initialization": prepared["initialization"],
        "training_mode": prepared["training_mode"],
        "budget": budget,
        "questions": len(prepared["examples"]),
        "train_turns": list(prepared["train_turns"]),
        "heldout_turns": [
            example["turn"]
            for example in prepared["examples"]
            if example["turn"] not in set(prepared["train_turns"])
        ],
        "story_tokens": prepared["story_tokens"],
        "briefing_tokens": prepared["briefing_tokens"],
        "source_segment_tokens": prepared["source_segment_tokens"],
        "teacher_segment_tokens": prepared["teacher_segment_tokens"],
        "compression_vs_briefing_segment": prepared["teacher_segment_tokens"] / budget,
        "cache_elements": cache_elements,
        "bf16_cache_bytes": cache_elements * 2,
        "steps": steps,
        "optimization_seconds": optimization_seconds,
        "training_means": {name: total / steps for name, total in totals.items()},
        "self_study_rows": prepared["self_study_rows"],
        "initial": initial_eval,
        "optimized": optimized_eval,
    }


@torch.inference_mode()
def _evaluate_baselines(model, tokenizer, prepared, *, max_new_tokens: int) -> dict[str, Any]:
    teacher_rows = []
    no_summary_rows = []
    for example in prepared["examples"]:
        teacher_generation = _generate(
            model,
            tokenizer,
            legacy_cache=prepared["teacher_cache"],
            cached_tokens=prepared["teacher_prefix_tokens"],
            query_ids=example["teacher_query_ids"],
            max_new_tokens=max_new_tokens,
        )
        no_summary_generation = _generate(
            model,
            tokenizer,
            legacy_cache=prepared["b_system_cache"],
            cached_tokens=prepared["b_system_tokens"],
            query_ids=example["student_query_ids"],
            max_new_tokens=max_new_tokens,
        )
        teacher_rows.append(_generation_row(example, teacher_generation))
        no_summary_rows.append(_generation_row(example, no_summary_generation))
    result = {
        "teacher_text": _aggregate_with_splits(teacher_rows, prepared["train_turns"]),
        "no_summary": _aggregate_with_splits(no_summary_rows, prepared["train_turns"]),
        "teacher_rows": teacher_rows,
        "no_summary_rows": no_summary_rows,
    }
    if prepared["self_study_rows"]:
        self_study_text_rows, self_study_text_tokens = _evaluate_self_study_text(
            model,
            tokenizer,
            prepared,
            max_new_tokens=max_new_tokens,
        )
        result["self_study_text"] = _aggregate_with_splits(
            self_study_text_rows, prepared["train_turns"]
        )
        result["self_study_text_rows"] = self_study_text_rows
        result["self_study_text_tokens"] = self_study_text_tokens
    return result


@torch.inference_mode()
def _evaluate_self_study_text(
    model,
    tokenizer,
    prepared: dict[str, Any],
    *,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    notes = "\n\n".join(
        f"View {index}: {row['response']}"
        for index, row in enumerate(prepared["self_study_rows"], 1)
    )
    prior_history = [
        {"role": "user", "content": f"Agent A self-study notes:\n{notes}"},
        {"role": "assistant", "content": ACK_B},
    ]
    prefix_ids = None
    prefix_cache = None
    rows = []
    for example in prepared["examples"]:
        current_prefix, query_ids, system_ids = _split_chat_before_last_user(
            tokenizer,
            system_prompt=AGENT_B_SYSTEM,
            prior_history=prior_history,
            query=example["question"],
        )
        if tuple(system_ids) != prepared["b_system_ids"]:
            raise ValueError("self-study text and student system prefixes differ")
        if prefix_ids is None:
            prefix_ids = current_prefix
            prefix_cache = prefill_legacy_cache(model, prefix_ids)
        elif current_prefix != prefix_ids:
            raise ValueError("self-study text prefix changed across questions")
        generation = _generate(
            model,
            tokenizer,
            legacy_cache=prefix_cache,
            cached_tokens=len(prefix_ids),
            query_ids=query_ids,
            max_new_tokens=max_new_tokens,
        )
        rows.append(_generation_row(example, generation))
    return rows, len(prefix_ids) - prepared["b_system_tokens"]


@torch.inference_mode()
def _evaluate_summary(model, tokenizer, prepared, summary, *, max_new_tokens: int) -> dict[str, Any]:
    model_dtype = model.model.embed_tokens.weight.dtype
    device = model.model.embed_tokens.weight.device
    summary_cache = summary.detached_cache(dtype=model_dtype, device=device)
    assembled = concatenate_caches(prepared["b_system_cache"], summary_cache)
    rows = []
    total_kl = 0.0
    total_nll = 0.0
    total_top1 = 0
    total_tokens = 0
    for example in prepared["examples"]:
        logits = teacher_forced_answer_logits(
            model,
            legacy_cache=assembled,
            cached_tokens=prepared["b_system_tokens"] + summary.tokens,
            query_ids=example["student_query_ids"],
            answer_ids=example["answer_ids"],
        ).float()
        teacher = example["teacher_logits"].float()
        targets = torch.tensor(example["answer_ids"], dtype=torch.long, device=device)
        token_count = targets.numel()
        total_nll += float(F.cross_entropy(logits.squeeze(0), targets, reduction="sum"))
        total_kl += float(
            F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(teacher, dim=-1),
                reduction="sum",
            )
        )
        total_top1 += int((logits.argmax(-1) == teacher.argmax(-1)).sum())
        total_tokens += token_count
        generation = _generate(
            model,
            tokenizer,
            legacy_cache=assembled,
            cached_tokens=prepared["b_system_tokens"] + summary.tokens,
            query_ids=example["student_query_ids"],
            max_new_tokens=max_new_tokens,
        )
        rows.append(_generation_row(example, generation))
    return {
        **_aggregate_with_splits(rows, prepared["train_turns"]),
        "answer_nll": total_nll / total_tokens,
        "teacher_kl": total_kl / total_tokens,
        "teacher_top1_agreement": total_top1 / total_tokens,
        "answer_tokens": total_tokens,
        "rows": rows,
    }


def _generate(
    model,
    tokenizer,
    *,
    legacy_cache,
    cached_tokens: int,
    query_ids: Sequence[int],
    max_new_tokens: int,
) -> str:
    _, text = _generate_ids(
        model,
        tokenizer,
        legacy_cache=legacy_cache,
        cached_tokens=cached_tokens,
        query_ids=query_ids,
        max_new_tokens=max_new_tokens,
    )
    return text


def _generate_ids(
    model,
    tokenizer,
    *,
    legacy_cache,
    cached_tokens: int,
    query_ids: Sequence[int],
    max_new_tokens: int,
) -> tuple[tuple[int, ...], str]:
    return greedy_action(
        model,
        lambda: start_readout(
            model,
            legacy_cache=legacy_cache,
            cached_tokens=cached_tokens,
            input_ids=query_ids,
        ),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )


def _generation_row(example: dict[str, Any], generation: str) -> dict[str, Any]:
    clean_generation = generation
    for stop_text in ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>"):
        clean_generation = clean_generation.split(stop_text, 1)[0]
    answer = answer_from_generation(clean_generation)
    return {
        "turn": example["turn"],
        "question": example["question"],
        "gold_answers": example["gold_answers"],
        "generation": generation,
        "answer": answer,
        "f1": coqa_f1(answer, example["gold_answers"]),
    }


def _aggregate_generation(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_f1": sum(row["f1"] for row in rows) / len(rows),
        "exact_match": sum(
            any(_normalized(row["answer"]) == _normalized(gold) for gold in row["gold_answers"])
            for row in rows
        )
        / len(rows),
    }


def _aggregate_with_splits(
    rows: Sequence[dict[str, Any]], train_turns: Sequence[int]
) -> dict[str, Any]:
    train_turns = set(train_turns)
    fit = [row for row in rows if row["turn"] in train_turns]
    heldout = [row for row in rows if row["turn"] not in train_turns]
    return {
        **_aggregate_generation(rows),
        "fit": _aggregate_generation(fit) if fit else None,
        "heldout": _aggregate_generation(heldout) if heldout else None,
    }


def _summarize(rows: Sequence[dict[str, Any]], *, config: dict[str, Any]) -> dict[str, Any]:
    by_budget = {}
    for budget in sorted({row["budget"] for row in rows}):
        selected = [row for row in rows if row["budget"] == budget]
        by_budget[str(budget)] = {
            "conversations": len(selected),
            "mean_initial_f1": _mean(row["initial"]["mean_f1"] for row in selected),
            "mean_optimized_f1": _mean(row["optimized"]["mean_f1"] for row in selected),
            "mean_optimized_teacher_kl": _mean(row["optimized"]["teacher_kl"] for row in selected),
            "mean_optimized_top1_agreement": _mean(
                row["optimized"]["teacher_top1_agreement"] for row in selected
            ),
            "mean_teacher_text_f1": _mean(
                row["baselines"]["teacher_text"]["mean_f1"] for row in selected
            ),
            "mean_no_summary_f1": _mean(
                row["baselines"]["no_summary"]["mean_f1"] for row in selected
            ),
            "mean_self_study_text_f1": _optional_mean(
                row["baselines"].get("self_study_text", {}).get("mean_f1")
                for row in selected
            ),
            "mean_heldout_f1": _optional_mean(
                row["optimized"]["heldout"]["mean_f1"]
                if row["optimized"]["heldout"] is not None
                else None
                for row in selected
            ),
            "mean_teacher_heldout_f1": _optional_mean(
                row["baselines"]["teacher_text"]["heldout"]["mean_f1"]
                if row["baselines"]["teacher_text"]["heldout"] is not None
                else None
                for row in selected
            ),
        }
    return {"config": config, "rows": len(rows), "by_budget": by_budget}


def _compact_log(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_index": row["conversation_index"],
        "initialization": row["initialization"],
        "budget": row["budget"],
        "compression": row["compression_vs_briefing_segment"],
        "teacher_f1": row["baselines"]["teacher_text"]["mean_f1"],
        "no_summary_f1": row["baselines"]["no_summary"]["mean_f1"],
        "self_study_text_f1": row["baselines"].get("self_study_text", {}).get("mean_f1"),
        "initial_f1": row["initial"]["mean_f1"],
        "optimized_f1": row["optimized"]["mean_f1"],
        "optimized_heldout_f1": (
            row["optimized"]["heldout"]["mean_f1"]
            if row["optimized"]["heldout"] is not None
            else None
        ),
        "optimized_kl": row["optimized"]["teacher_kl"],
    }


def _gold_briefing(all_questions: Sequence[str], all_answers: Sequence[str]) -> str:
    lines = ["Concise facts retained by Agent A for the downstream questions:"]
    for question, answer in zip(all_questions, all_answers, strict=True):
        lines.append(f"- {question} Answer: {answer}.")
    return "\n".join(lines)


def _load_selected(path: str, indices: Sequence[int]) -> list[tuple[int, dict[str, Any]]]:
    wanted = set(indices)
    selected = []
    for index, document in enumerate(iter_records(path)):
        if index in wanted:
            selected.append((index, document))
        if len(selected) == len(wanted):
            break
    found = {index for index, _ in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"dataset does not contain conversation indices {missing}")
    return sorted(selected)


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("integer list must be non-empty and unique")
    return values


def _normalized(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _optional_mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return _mean(values) if values else None


if __name__ == "__main__":
    main()
