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

from ..coqa import gold_answers_at_turn, primary_answers, questions
from ..policy_imprinting import build_chat_segments, prefill_legacy_cache
from ..semantic_kv_summary import (
    concatenate_caches,
    mean_pool_segment_cache,
    teacher_forced_answer_logits,
)
from ..semantic_kv_transcoder import TailReadoutKVTranscoder, relative_cache_mse
from .probe_semantic_kv_summary import (
    ACK_B,
    AGENT_A_SYSTEM,
    AGENT_B_SYSTEM,
    SELF_STUDY_PROMPTS,
    _aggregate_generation,
    _generate,
    _generate_ids,
    _generation_row,
    _load_selected,
    _parse_ints,
    _split_chat_before_last_user,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gate-2 probe: amortize a joint source KV into a fixed-rate latent and "
            "synthesize receiver-native compact KV for an independently trained model."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--train-indices", type=_parse_ints, required=True)
    parser.add_argument("--test-indices", type=_parse_ints, required=True)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--max-questions", type=int, default=8)
    parser.add_argument("--self-study-max-new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--kv-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-quant-bits", type=int, choices=(4, 8))
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    if set(args.train_indices) & set(args.test_indices):
        parser.error("train and test document indices must be disjoint")
    positive = (
        args.slots,
        args.latent_dim,
        args.max_questions,
        args.self_study_max_new_tokens,
        args.steps,
        args.learning_rate,
        args.temperature,
        args.grad_clip,
        args.max_new_tokens,
        args.log_every,
    )
    if min(positive) <= 0:
        parser.error("budgets, counts, rates, temperature, and limits must be positive")
    if min(args.weight_decay, args.ce_weight, args.distill_weight, args.kv_weight) < 0:
        parser.error("loss weights and weight decay must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "train_indices": list(args.train_indices),
        "test_indices": list(args.test_indices),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    dtype = getattr(torch, args.dtype)
    source_tokenizer = AutoTokenizer.from_pretrained(
        args.source_model
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        args.target_model
    )
    source_model = _load_model(args.source_model, args.device, dtype)
    target_model = _load_model(args.target_model, args.device, dtype)
    transcoder = TailReadoutKVTranscoder(
        source_model.config,
        target_model.config,
        slots=args.slots,
        latent_dim=args.latent_dim,
    ).to(args.device)

    wanted = tuple((*args.train_indices, *args.test_indices))
    selected = dict(_load_selected(args.dataset, wanted))
    documents = []
    for index in wanted:
        split = "train" if index in set(args.train_indices) else "test"
        document = _prepare_document(
            source_model,
            source_tokenizer,
            target_model,
            target_tokenizer,
            index=index,
            record=selected[index],
            split=split,
            slots=args.slots,
            max_questions=args.max_questions,
            self_study_max_new_tokens=args.self_study_max_new_tokens,
        )
        documents.append(document)
        print(
            json.dumps(
                {
                    "event": "prepared",
                    "index": index,
                    "split": split,
                    "source_tokens": document["source_tokens"],
                    "target_story_tokens": document["target_story_tokens"],
                    "optimization_examples": len(document["optimization_examples"]),
                }
            ),
            flush=True,
        )

    train_documents = [document for document in documents if document["split"] == "train"]
    optimizer = AdamW(
        transcoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    schedule = [
        (document, example)
        for document in train_documents
        for example in document["optimization_examples"]
    ]
    if not schedule:
        raise ValueError("training schedule is empty")
    generator = random.Random(args.seed)
    totals = {"loss": 0.0, "ce": 0.0, "distill": 0.0, "kv": 0.0}
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(schedule) == 0:
            generator.shuffle(schedule)
        document, example = schedule[(step - 1) % len(schedule)]
        optimizer.zero_grad(set_to_none=True)
        latent, compact_cache = transcoder(
            source_model,
            target_model,
            source_cache=document["source_cache"],
            source_cached_tokens=document["source_tokens"],
            target_start=document["target_system_tokens"],
            quant_bits=args.train_quant_bits,
            straight_through=args.train_quant_bits is not None,
        )
        assembled = concatenate_caches(document["target_system_cache"], compact_cache)
        logits = teacher_forced_answer_logits(
            target_model,
            legacy_cache=assembled,
            cached_tokens=document["target_system_tokens"] + args.slots,
            query_ids=example["student_query_ids"],
            answer_ids=example["answer_ids"],
        )
        targets = torch.tensor(
            example["answer_ids"], dtype=torch.long, device=args.device
        )
        ce = F.cross_entropy(logits.squeeze(0).float(), targets)
        teacher_logits = example["teacher_logits"].to(
            device=args.device, dtype=torch.float32
        )
        distill = F.kl_div(
            F.log_softmax(logits.float() / args.temperature, dim=-1),
            F.softmax(teacher_logits / args.temperature, dim=-1),
            reduction="sum",
        ) * (args.temperature**2 / targets.numel())
        kv = relative_cache_mse(compact_cache, document["target_pooled_cache"])
        loss = (
            args.ce_weight * ce
            + args.distill_weight * distill
            + args.kv_weight * kv
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transcoder.parameters(), args.grad_clip)
        optimizer.step()
        values = {"loss": loss, "ce": ce, "distill": distill, "kv": kv}
        for name, value in values.items():
            totals[name] += float(value.detach())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "document": document["index"],
                        **{
                            f"mean_{name}": total / step
                            for name, total in totals.items()
                        },
                    }
                ),
                flush=True,
            )
        del latent, compact_cache, assembled, logits, loss, ce, distill, kv

    training_seconds = time.perf_counter() - started
    torch.save(
        {
            "state_dict": transcoder.state_dict(),
            "config": config,
            "training_means": {
                name: total / args.steps for name, total in totals.items()
            },
        },
        output_dir / "transcoder.pt",
    )

    rows = []
    for quant_bits in (None, 8, 4):
        for document in documents:
            row = _evaluate_document(
                source_model,
                target_model,
                target_tokenizer,
                transcoder,
                document,
                slots=args.slots,
                quant_bits=quant_bits,
                max_new_tokens=args.max_new_tokens,
            )
            rows.append(row)
            print(json.dumps({"event": "evaluate", **row["metrics"]}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    with (output_dir / "results.jsonl").open("w") as result_file:
        for row in rows:
            result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = _summarize(
        rows,
        config=config,
        training_seconds=training_seconds,
        training_means={name: total / args.steps for name, total in totals.items()},
        latent_elements=args.slots * args.latent_dim,
        receiver_cache_elements=transcoder.receiver_cache_elements,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def _load_model(path: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def _prepare_document(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    *,
    index: int,
    record: dict[str, Any],
    split: str,
    slots: int,
    max_questions: int,
    self_study_max_new_tokens: int,
) -> dict[str, Any]:
    selected_questions = questions(record)[:max_questions]
    selected_answers = primary_answers(record)[:max_questions]
    if not selected_questions or len(selected_questions) != len(selected_answers):
        raise ValueError(f"conversation {index} has invalid questions/answers")

    source_segments = build_chat_segments(
        source_tokenizer,
        system_prompt=AGENT_A_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": f"Source material:\n{record['story']}"}],
        enable_thinking=False,
    )
    source_ids = [
        *source_segments.prefix_ids,
        *source_segments.history_ids,
        *source_segments.readout_ids,
    ]
    source_cache = prefill_legacy_cache(source_model, source_ids)

    target_probe = build_chat_segments(
        target_tokenizer,
        system_prompt=AGENT_B_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": selected_questions[0]}],
        enable_thinking=False,
    )
    target_system_ids = list(target_probe.prefix_ids)
    target_system_cache = prefill_legacy_cache(target_model, target_system_ids)
    full_story_history = [
        {"role": "user", "content": f"Agent A source material:\n{record['story']}"},
        {"role": "assistant", "content": ACK_B},
    ]
    full_prefix_ids, _, full_system_ids = _split_chat_before_last_user(
        target_tokenizer,
        system_prompt=AGENT_B_SYSTEM,
        prior_history=full_story_history,
        query=selected_questions[0],
    )
    if full_system_ids != target_system_ids:
        raise ValueError("target system prefix changed in full-story prompt")
    target_full_cache = prefill_legacy_cache(target_model, full_prefix_ids)
    target_pooled_cache = mean_pool_segment_cache(
        target_model,
        target_full_cache,
        segment_start=len(target_system_ids),
        segment_end=len(full_prefix_ids),
        target_start=len(target_system_ids),
        tokens=slots,
    )

    qa_examples = []
    for turn, question in enumerate(selected_questions, 1):
        student_segments = build_chat_segments(
            target_tokenizer,
            system_prompt=AGENT_B_SYSTEM,
            tools=[],
            history=[{"role": "user", "content": question}],
            enable_thinking=False,
        )
        teacher_prefix_ids, teacher_query_ids, teacher_system_ids = (
            _split_chat_before_last_user(
                target_tokenizer,
                system_prompt=AGENT_B_SYSTEM,
                prior_history=full_story_history,
                query=question,
            )
        )
        if teacher_prefix_ids != full_prefix_ids:
            raise ValueError(f"full-story target prefix changed at turn {turn}")
        if teacher_system_ids != target_system_ids:
            raise ValueError(f"target system prefix changed at turn {turn}")
        qa_examples.append(
            {
                "turn": turn,
                "question": question,
                "gold_answers": gold_answers_at_turn(record, turn),
                "student_query_ids": tuple(
                    (*student_segments.history_ids, *student_segments.readout_ids)
                ),
                "teacher_query_ids": tuple(teacher_query_ids),
            }
        )

    optimization_examples = []
    self_study_rows = []
    if split == "train":
        stop_ids = {
            token_id
            for token_id in (
                target_tokenizer.eos_token_id,
                target_tokenizer.convert_tokens_to_ids("<|im_end|>"),
                target_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            )
            if isinstance(token_id, int) and token_id >= 0
        }
        for prompt_index, prompt in enumerate(SELF_STUDY_PROMPTS, 1):
            teacher_prefix_ids, teacher_query_ids, _ = _split_chat_before_last_user(
                target_tokenizer,
                system_prompt=AGENT_B_SYSTEM,
                prior_history=full_story_history,
                query=prompt,
            )
            if teacher_prefix_ids != full_prefix_ids:
                raise ValueError("self-study target prefix changed")
            student_segments = build_chat_segments(
                target_tokenizer,
                system_prompt=AGENT_B_SYSTEM,
                tools=[],
                history=[{"role": "user", "content": prompt}],
                enable_thinking=False,
            )
            generated_ids, generated_text = _generate_ids(
                target_model,
                target_tokenizer,
                legacy_cache=target_full_cache,
                cached_tokens=len(full_prefix_ids),
                query_ids=teacher_query_ids,
                max_new_tokens=self_study_max_new_tokens,
            )
            answer_ids = list(generated_ids)
            while answer_ids and answer_ids[-1] in stop_ids:
                answer_ids.pop()
            if not answer_ids:
                raise ValueError(f"self-study prompt {prompt_index} generated no tokens")
            teacher_logits = teacher_forced_answer_logits(
                target_model,
                legacy_cache=target_full_cache,
                cached_tokens=len(full_prefix_ids),
                query_ids=teacher_query_ids,
                answer_ids=answer_ids,
            ).detach().to(dtype=torch.bfloat16)
            optimization_examples.append(
                {
                    "name": f"self_study_{prompt_index}",
                    "answer_ids": tuple(int(token) for token in answer_ids),
                    "student_query_ids": tuple(
                        (*student_segments.history_ids, *student_segments.readout_ids)
                    ),
                    "teacher_logits": teacher_logits,
                }
            )
            self_study_rows.append(
                {
                    "name": f"self_study_{prompt_index}",
                    "prompt": prompt,
                    "response": target_tokenizer.decode(
                        answer_ids, skip_special_tokens=True
                    ).strip(),
                    "tokens": len(answer_ids),
                    "raw_generation": generated_text,
                }
            )

    return {
        "index": index,
        "split": split,
        "conversation_id": record.get("id"),
        "domain": record.get("source"),
        "source_cache": source_cache,
        "source_tokens": len(source_ids),
        "target_system_cache": target_system_cache,
        "target_system_tokens": len(target_system_ids),
        "target_full_cache": target_full_cache,
        "target_full_tokens": len(full_prefix_ids),
        "target_story_tokens": len(full_prefix_ids) - len(target_system_ids),
        "target_pooled_cache": target_pooled_cache,
        "qa_examples": qa_examples,
        "optimization_examples": optimization_examples,
        "self_study_rows": self_study_rows,
    }


@torch.inference_mode()
def _evaluate_document(
    source_model,
    target_model,
    target_tokenizer,
    transcoder,
    document: dict[str, Any],
    *,
    slots: int,
    quant_bits: int | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    latent, compact_cache = transcoder(
        source_model,
        target_model,
        source_cache=document["source_cache"],
        source_cached_tokens=document["source_tokens"],
        target_start=document["target_system_tokens"],
        quant_bits=quant_bits,
    )
    assembled = concatenate_caches(document["target_system_cache"], compact_cache)
    student_rows = []
    teacher_rows = []
    no_summary_rows = []
    for example in document["qa_examples"]:
        student_generation = _generate(
            target_model,
            target_tokenizer,
            legacy_cache=assembled,
            cached_tokens=document["target_system_tokens"] + slots,
            query_ids=example["student_query_ids"],
            max_new_tokens=max_new_tokens,
        )
        teacher_generation = _generate(
            target_model,
            target_tokenizer,
            legacy_cache=document["target_full_cache"],
            cached_tokens=document["target_full_tokens"],
            query_ids=example["teacher_query_ids"],
            max_new_tokens=max_new_tokens,
        )
        no_summary_generation = _generate(
            target_model,
            target_tokenizer,
            legacy_cache=document["target_system_cache"],
            cached_tokens=document["target_system_tokens"],
            query_ids=example["student_query_ids"],
            max_new_tokens=max_new_tokens,
        )
        student_rows.append(_generation_row(example, student_generation))
        teacher_rows.append(_generation_row(example, teacher_generation))
        no_summary_rows.append(_generation_row(example, no_summary_generation))
    student = _aggregate_generation(student_rows)
    teacher = _aggregate_generation(teacher_rows)
    no_summary = _aggregate_generation(no_summary_rows)
    metrics = {
        "index": document["index"],
        "split": document["split"],
        "quant_bits": quant_bits or 16,
        "student_f1": student["mean_f1"],
        "teacher_f1": teacher["mean_f1"],
        "no_summary_f1": no_summary["mean_f1"],
    }
    return {
        "metrics": metrics,
        "latent_abs_mean": float(latent.float().abs().mean()),
        "student": student,
        "teacher": teacher,
        "no_summary": no_summary,
        "student_rows": student_rows,
        "teacher_rows": teacher_rows,
        "no_summary_rows": no_summary_rows,
    }


def _summarize(
    rows: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    training_seconds: float,
    training_means: dict[str, float],
    latent_elements: int,
    receiver_cache_elements: int,
) -> dict[str, Any]:
    by_split_quant = {}
    for split in ("train", "test"):
        for quant_bits in (16, 8, 4):
            selected = [
                row
                for row in rows
                if row["metrics"]["split"] == split
                and row["metrics"]["quant_bits"] == quant_bits
            ]
            if not selected:
                continue
            student = _mean(row["student"]["mean_f1"] for row in selected)
            teacher = _mean(row["teacher"]["mean_f1"] for row in selected)
            no_summary = _mean(row["no_summary"]["mean_f1"] for row in selected)
            denominator = teacher - no_summary
            by_split_quant[f"{split}_int{quant_bits}"] = {
                "documents": len(selected),
                "mean_student_f1": student,
                "mean_teacher_f1": teacher,
                "mean_no_summary_f1": no_summary,
                "incremental_quality_retention": (
                    (student - no_summary) / denominator
                    if abs(denominator) > 1e-12
                    else None
                ),
                "wire_bytes_per_document": latent_elements * quant_bits / 8,
            }
    return {
        "config": config,
        "training_seconds": training_seconds,
        "training_means": training_means,
        "latent_elements": latent_elements,
        "receiver_cache_elements": receiver_cache_elements,
        "receiver_bf16_cache_bytes": receiver_cache_elements * 2,
        "by_split_quant": by_split_quant,
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
