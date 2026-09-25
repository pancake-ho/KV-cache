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

from ..coqa import coqa_f1
from ..policy_imprinting import ReadoutState, build_chat_segments, prefill_legacy_cache, start_readout
from ..semantic_kv_summary import concatenate_caches, teacher_forced_answer_logits
from ..semantic_kv_transcoder import (
    ConditionedMultiLayerKVTranscoder,
    ReceiverManifoldKVTranscoder,
    TailKVContentManifoldTranscoder,
    TailReadoutKVTranscoder,
    fake_quantize_latent,
)
from .evaluate_amortized_musique_handoff import (
    DEPENDENT_B_SYSTEM,
    _clean_generation,
    _format_source,
    _format_target,
)
from .evaluate_musique_agent_a import SOURCE_ANSWER_SYSTEM
from .probe_semantic_kv_summary import ACK_B, AGENT_A_SYSTEM, _generate, _normalized, _split_chat_before_last_user


INTERMEDIATE_B_SYSTEM = (
    "You are the receiving agent. The context before the user request either states "
    "or semantically encodes Agent A's retained intermediate answer. Output that exact "
    "short answer and no explanation."
)
INTERMEDIATE_QUERY = "What exact intermediate answer did Agent A retain?"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a source-KV to receiver-KV handoff compressor on MuSiQue train "
            "cases and evaluate on disjoint MuSiQue dev cases."
        )
    )
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--variant",
        choices=(
            "tail",
            "conditioned_multilayer",
            "receiver_manifold",
            "tailkv_manifold",
        ),
        required=True,
    )
    parser.add_argument(
        "--source-state",
        choices=("prefill", "generated_answer", "gold_answer"),
        default="prefill",
    )
    parser.add_argument("--source-answer-max-new-tokens", type=int, default=24)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=32)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--global-slots", type=int)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage-weight", type=float, default=1.0)
    parser.add_argument("--bridge-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-quant-bits", type=int, choices=(4, 8))
    parser.add_argument("--eval-quant-bits", default="16,4")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    positive = (
        args.train_count,
        args.eval_count,
        args.slots,
        args.latent_dim,
        args.steps,
        args.learning_rate,
        args.temperature,
        args.grad_clip,
        args.max_new_tokens,
        args.source_answer_max_new_tokens,
        args.log_every,
    )
    if min(positive) <= 0 or min(args.train_offset, args.eval_offset) < 0:
        parser.error("counts, budgets, rates, and limits must be positive; offsets non-negative")
    if min(args.weight_decay, args.stage_weight, args.bridge_weight, args.distill_weight) < 0:
        parser.error("loss weights and weight decay must be non-negative")
    eval_quant_bits = tuple(int(value) for value in args.eval_quant_bits.split(","))
    if not eval_quant_bits or any(value not in (4, 8, 16) for value in eval_quant_bits):
        parser.error("--eval-quant-bits must contain only 4, 8, or 16")
    if len(set(eval_quant_bits)) != len(eval_quant_bits):
        parser.error("--eval-quant-bits must be unique")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "eval_quant_bits": list(eval_quant_bits)}
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
    transcoder = _build_transcoder(args, source_model, target_model).to(args.device)

    stage_system_ids, stage_system_cache = _system_cache(
        target_model,
        target_tokenizer,
        DEPENDENT_B_SYSTEM,
        "placeholder",
    )
    bridge_system_ids, bridge_system_cache = _system_cache(
        target_model,
        target_tokenizer,
        INTERMEDIATE_B_SYSTEM,
        INTERMEDIATE_QUERY,
    )

    train_cases = _load_slice(
        args.train_dataset, offset=args.train_offset, count=args.train_count
    )
    eval_cases = _load_slice(
        args.eval_dataset, offset=args.eval_offset, count=args.eval_count
    )
    train_ids = {case["id"] for _, case in train_cases}
    eval_ids = {case["id"] for _, case in eval_cases}
    if train_ids & eval_ids:
        parser.error("train and evaluation case IDs overlap")

    documents = []
    resident_cache_bytes = 0
    for position, (index, case) in enumerate(train_cases, 1):
        document = _prepare_training_case(
            source_model,
            source_tokenizer,
            target_model,
            target_tokenizer,
            transcoder=transcoder,
            variant=args.variant,
            source_state=args.source_state,
            source_answer_max_new_tokens=args.source_answer_max_new_tokens,
            index=index,
            case=case,
            stage_system_ids=stage_system_ids,
            bridge_system_ids=bridge_system_ids,
        )
        documents.append(document)
        resident_cache_bytes += _resident_source_bytes(document)
        print(
            json.dumps(
                {
                    "event": "prepared_train",
                    "position": position,
                    "index": index,
                    "id": case["id"],
                    "source_tokens": document["source_tokens"],
                    "resident_source_cache_gib": resident_cache_bytes / 2**30,
                }
            ),
            flush=True,
        )

    optimizer = AdamW(
        transcoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = random.Random(args.seed)
    schedule = list(documents)
    totals = {
        "loss": 0.0,
        "stage_ce": 0.0,
        "stage_kl": 0.0,
        "bridge_ce": 0.0,
        "bridge_kl": 0.0,
    }
    started = time.perf_counter()
    transcoder.train()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(schedule) == 0:
            generator.shuffle(schedule)
        document = schedule[(step - 1) % len(schedule)]
        optimizer.zero_grad(set_to_none=True)
        latent = _encode(
            transcoder,
            source_model,
            document,
            quant_bits=args.train_quant_bits,
            straight_through=args.train_quant_bits is not None,
        )
        stage_compact = _decode(
            transcoder,
            target_model,
            latent,
            target_start=len(stage_system_ids),
            prefix_cache=stage_system_cache,
        )
        stage_ce, stage_kl = _task_loss(
            target_model,
            prefix_cache=stage_system_cache,
            prefix_tokens=len(stage_system_ids),
            compact_cache=stage_compact,
            slots=args.slots,
            query_ids=document["stage_query_ids"],
            answer_ids=document["stage_answer_ids"],
            teacher_logits=document["stage_teacher_logits"],
            temperature=args.temperature,
        )
        bridge_compact = _decode(
            transcoder,
            target_model,
            latent,
            target_start=len(bridge_system_ids),
            prefix_cache=bridge_system_cache,
        )
        bridge_ce, bridge_kl = _task_loss(
            target_model,
            prefix_cache=bridge_system_cache,
            prefix_tokens=len(bridge_system_ids),
            compact_cache=bridge_compact,
            slots=args.slots,
            query_ids=document["bridge_query_ids"],
            answer_ids=document["bridge_answer_ids"],
            teacher_logits=document["bridge_teacher_logits"],
            temperature=args.temperature,
        )
        loss = (
            args.stage_weight * stage_ce
            + args.bridge_weight * bridge_ce
            + args.distill_weight * (stage_kl + bridge_kl)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transcoder.parameters(), args.grad_clip)
        optimizer.step()
        values = {
            "loss": loss,
            "stage_ce": stage_ce,
            "stage_kl": stage_kl,
            "bridge_ce": bridge_ce,
            "bridge_kl": bridge_kl,
        }
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
        del latent, stage_compact, bridge_compact, stage_ce, stage_kl
        del bridge_ce, bridge_kl, loss

    training_seconds = time.perf_counter() - started
    transcoder.eval()
    checkpoint = {
        "state_dict": transcoder.state_dict(),
        "config": config,
        "training_means": {name: total / args.steps for name, total in totals.items()},
    }
    torch.save(checkpoint, output_dir / "transcoder.pt")

    # Release large resident train caches before streaming held-out evaluation.
    del documents, schedule, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for position, (index, case) in enumerate(eval_cases, 1):
        case_rows = _evaluate_case(
            source_model,
            source_tokenizer,
            target_model,
            target_tokenizer,
            transcoder,
            variant=args.variant,
            source_state=args.source_state,
            source_answer_max_new_tokens=args.source_answer_max_new_tokens,
            index=index,
            case=case,
            stage_system_ids=stage_system_ids,
            stage_system_cache=stage_system_cache,
            bridge_system_ids=bridge_system_ids,
            bridge_system_cache=bridge_system_cache,
            quant_bits_values=eval_quant_bits,
            max_new_tokens=args.max_new_tokens,
        )
        rows.extend(case_rows)
        for row in case_rows:
            print(json.dumps({"event": "evaluate", **row}, ensure_ascii=False), flush=True)
        print(
            json.dumps({"event": "evaluation_progress", "position": position}),
            flush=True,
        )
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
        resident_train_cache_bytes=resident_cache_bytes,
        transcoder=transcoder,
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


def _build_transcoder(args, source_model, target_model):
    common = {
        "slots": args.slots,
        "latent_dim": args.latent_dim,
    }
    if args.variant == "tail":
        if args.global_slots is not None:
            raise ValueError("--global-slots is only valid for conditioned_multilayer")
        return TailReadoutKVTranscoder(
            source_model.config, target_model.config, **common
        )
    if args.variant == "receiver_manifold":
        if args.global_slots is not None:
            raise ValueError("--global-slots is only valid for conditioned_multilayer")
        return ReceiverManifoldKVTranscoder(
            source_model.config, target_model.config, **common
        )
    if args.variant == "tailkv_manifold":
        if args.source_state == "prefill":
            raise ValueError("tailkv_manifold requires generated_answer or gold_answer source state")
        if args.global_slots is not None:
            raise ValueError("--global-slots is only valid for conditioned_multilayer")
        return TailKVContentManifoldTranscoder(
            source_model.config, target_model.config, **common
        )
    return ConditionedMultiLayerKVTranscoder(
        source_model.config,
        target_model.config,
        global_slots=args.global_slots,
        **common,
    )


@torch.inference_mode()
def _system_cache(model, tokenizer, system_prompt: str, user_text: str):
    segments = build_chat_segments(
        tokenizer,
        system_prompt=system_prompt,
        tools=[],
        history=[{"role": "user", "content": user_text}],
        enable_thinking=False,
    )
    ids = list(segments.prefix_ids)
    return ids, prefill_legacy_cache(model, ids)


@torch.inference_mode()
def _prepare_training_case(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    *,
    transcoder,
    variant: str,
    source_state: str,
    source_answer_max_new_tokens: int,
    index: int,
    case: dict[str, Any],
    stage_system_ids: Sequence[int],
    bridge_system_ids: Sequence[int],
) -> dict[str, Any]:
    source_cache, source_tokens, source_answer, source_answer_tokens = _source_cache(
        source_model,
        source_tokenizer,
        case,
        source_state=source_state,
        max_new_tokens=source_answer_max_new_tokens,
    )
    condition_state = (
        _condition_state(source_model, source_tokenizer, case["agent_a"]["question"])
        if variant == "conditioned_multilayer"
        else None
    )
    stage = _training_task(
        target_model,
        target_tokenizer,
        system_prompt=DEPENDENT_B_SYSTEM,
        system_ids=stage_system_ids,
        query=_format_target(case),
        answer=case["agent_b"]["gold_answer"],
        bridge=case["agent_a"]["gold_answer"],
    )
    bridge = _training_task(
        target_model,
        target_tokenizer,
        system_prompt=INTERMEDIATE_B_SYSTEM,
        system_ids=bridge_system_ids,
        query=INTERMEDIATE_QUERY,
        answer=case["agent_a"]["gold_answer"],
        bridge=case["agent_a"]["gold_answer"],
    )
    source_tail_features = None
    if isinstance(transcoder, TailKVContentManifoldTranscoder):
        source_tail_features = transcoder.extract_tail_features(
            source_model,
            source_cache=source_cache,
            source_cached_tokens=source_tokens,
            source_answer_tokens=source_answer_tokens,
        )
        # Tail features are a sufficient statistic for this encoder; retaining
        # the 0.9-GiB full cache per training case would unnecessarily cap the
        # number of training documents.
        source_cache = None
    return {
        "index": index,
        "id": case["id"],
        "source_cache": source_cache,
        "source_tokens": source_tokens,
        "condition_state": condition_state,
        "source_answer": source_answer,
        "source_answer_tokens": source_answer_tokens,
        "source_tail_features": source_tail_features,
        "stage_query_ids": stage["query_ids"],
        "stage_answer_ids": stage["answer_ids"],
        "stage_teacher_logits": stage["teacher_logits"],
        "bridge_query_ids": bridge["query_ids"],
        "bridge_answer_ids": bridge["answer_ids"],
        "bridge_teacher_logits": bridge["teacher_logits"],
    }


@torch.inference_mode()
def _source_cache(
    source_model,
    source_tokenizer,
    case: dict[str, Any],
    *,
    source_state: str,
    max_new_tokens: int,
    return_metadata: bool = False,
):
    if source_state == "prefill":
        source_user = _format_source(case)
        source_system = AGENT_A_SYSTEM
    else:
        documents = "\n\n".join(
            f"[DOC-{number}] {document['title']}\n{document['text']}"
            for number, document in enumerate(case["agent_a"]["documents"], 1)
        )
        source_user = (
            f"<DOSSIER>\n{documents}\n</DOSSIER>\n\n"
            f"Question: {case['agent_a']['question']}\n"
            "Output only the exact short answer."
        )
        source_system = SOURCE_ANSWER_SYSTEM
    segments = build_chat_segments(
        source_tokenizer,
        system_prompt=source_system,
        tools=[],
        history=[{"role": "user", "content": source_user}],
        enable_thinking=False,
    )
    source_ids = [*segments.prefix_ids, *segments.history_ids, *segments.readout_ids]
    if source_state == "prefill":
        result = (prefill_legacy_cache(source_model, source_ids), len(source_ids), None, 0)
        return (*result, {}) if return_metadata else result
    if source_state == "gold_answer":
        answer_ids = source_tokenizer.encode(
            case["agent_a"]["gold_answer"], add_special_tokens=False
        )
        full_ids = [*source_ids, *answer_ids]
        result = (
            prefill_legacy_cache(source_model, full_ids),
            len(full_ids),
            case["agent_a"]["gold_answer"],
            len(answer_ids),
        )
        return (*result, {}) if return_metadata else result
    if source_state != "generated_answer":
        raise ValueError(f"unknown source_state: {source_state}")

    prefix_cache = prefill_legacy_cache(source_model, segments.prefix_ids)
    state = start_readout(
        source_model,
        legacy_cache=prefix_cache,
        cached_tokens=len(segments.prefix_ids),
        input_ids=(*segments.history_ids, *segments.readout_ids),
    )
    generated: list[int] = []
    token_logprobs: list[float] = []
    token_margins: list[float] = []
    for _ in range(max_new_tokens):
        next_id = int(state.logits.argmax(dim=-1).item())
        candidate = source_tokenizer.decode(
            [*generated, next_id], skip_special_tokens=False
        )
        if (
            next_id == source_tokenizer.eos_token_id
            or "<|im_end|>" in candidate
            or "<|eot_id|>" in candidate
            or "<|end_of_text|>" in candidate
        ):
            break
        log_probs = state.logits.float().log_softmax(dim=-1)
        token_logprobs.append(float(log_probs[0, next_id]))
        top_two = torch.topk(state.logits.float(), k=2, dim=-1).values
        token_margins.append(float(top_two[0, 0] - top_two[0, 1]))
        generated.append(next_id)
        state = _advance_source_one(source_model, state, next_id)
    source_answer = _clean_generation(
        source_tokenizer.decode(generated, skip_special_tokens=False)
    )
    if not generated:
        raise ValueError("source model generated an empty intermediate answer")
    result = (state.past_key_values, state.cached_tokens, source_answer, len(generated))
    if not return_metadata:
        return result
    metadata = {
        "source_answer_mean_logprob": sum(token_logprobs) / len(token_logprobs),
        "source_answer_min_logprob": min(token_logprobs),
        "source_answer_mean_margin": sum(token_margins) / len(token_margins),
    }
    return (*result, metadata)


@torch.inference_mode()
def _advance_source_one(source_model, state: ReadoutState, token_id: int) -> ReadoutState:
    device = source_model.model.embed_tokens.weight.device
    current = torch.tensor([[token_id]], dtype=torch.long, device=device)
    output = source_model(
        input_ids=current,
        attention_mask=torch.ones(
            (1, state.cached_tokens + 1), dtype=torch.long, device=device
        ),
        position_ids=torch.tensor([[state.cached_tokens]], dtype=torch.long, device=device),
        cache_position=torch.tensor([state.cached_tokens], dtype=torch.long, device=device),
        past_key_values=state.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    return ReadoutState(
        logits=output.logits[:, -1],
        past_key_values=output.past_key_values,
        cached_tokens=state.cached_tokens + 1,
    )


@torch.inference_mode()
def _condition_state(source_model, source_tokenizer, question: str) -> torch.Tensor:
    ids = source_tokenizer.encode(
        "Agent A extraction question: " + question,
        add_special_tokens=True,
    )
    input_ids = torch.tensor(
        [ids], dtype=torch.long, device=source_model.model.embed_tokens.weight.device
    )
    output = source_model.model(
        input_ids=input_ids,
        use_cache=False,
        return_dict=True,
    )
    return output.last_hidden_state[:, -1].detach().float()


@torch.inference_mode()
def _training_task(
    target_model,
    target_tokenizer,
    *,
    system_prompt: str,
    system_ids: Sequence[int],
    query: str,
    answer: str,
    bridge: str,
) -> dict[str, Any]:
    student = build_chat_segments(
        target_tokenizer,
        system_prompt=system_prompt,
        tools=[],
        history=[{"role": "user", "content": query}],
        enable_thinking=False,
    )
    if tuple(student.prefix_ids) != tuple(system_ids):
        raise ValueError("student target prefix changed across tasks")
    teacher_history = [
        {"role": "user", "content": "Agent A intermediate answer: " + bridge},
        {"role": "assistant", "content": ACK_B},
    ]
    teacher_prefix_ids, teacher_query_ids, teacher_system_ids = _split_chat_before_last_user(
        target_tokenizer,
        system_prompt=system_prompt,
        prior_history=teacher_history,
        query=query,
    )
    if tuple(teacher_system_ids) != tuple(system_ids):
        raise ValueError("teacher and student target prefixes differ")
    answer_ids = target_tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError("target answer tokenization is empty")
    teacher_cache = prefill_legacy_cache(target_model, teacher_prefix_ids)
    teacher_logits = teacher_forced_answer_logits(
        target_model,
        legacy_cache=teacher_cache,
        cached_tokens=len(teacher_prefix_ids),
        query_ids=teacher_query_ids,
        answer_ids=answer_ids,
    ).detach().to(dtype=torch.bfloat16)
    return {
        "query_ids": tuple((*student.history_ids, *student.readout_ids)),
        "answer_ids": tuple(int(value) for value in answer_ids),
        "teacher_logits": teacher_logits,
    }


def _encode(
    transcoder,
    source_model,
    document: dict[str, Any],
    *,
    quant_bits: int | None,
    straight_through: bool,
) -> torch.Tensor:
    common = {
        "source_cache": document["source_cache"],
        "source_cached_tokens": document["source_tokens"],
        "quant_bits": quant_bits,
        "straight_through": straight_through,
    }
    if isinstance(transcoder, ConditionedMultiLayerKVTranscoder):
        return transcoder.encode(
            source_model,
            condition_state=document["condition_state"],
            **common,
        )
    if isinstance(transcoder, TailKVContentManifoldTranscoder):
        if document.get("source_tail_features") is not None:
            return transcoder.encode_tail_features(
                document["source_tail_features"],
                quant_bits=quant_bits,
                straight_through=straight_through,
            )
        return transcoder.encode(
            source_model,
            source_answer_tokens=document["source_answer_tokens"],
            **common,
        )
    return transcoder.encode(source_model, **common)


def _decode(
    transcoder,
    target_model,
    latent: torch.Tensor,
    *,
    target_start: int,
    prefix_cache,
):
    if isinstance(transcoder, ReceiverManifoldKVTranscoder):
        return transcoder.decode(
            target_model,
            latent,
            target_start=target_start,
            prefix_cache=prefix_cache,
        )
    return transcoder.decode(target_model, latent, target_start=target_start)


def _task_loss(
    target_model,
    *,
    prefix_cache,
    prefix_tokens: int,
    compact_cache,
    slots: int,
    query_ids: Sequence[int],
    answer_ids: Sequence[int],
    teacher_logits: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assembled = concatenate_caches(prefix_cache, compact_cache)
    logits = teacher_forced_answer_logits(
        target_model,
        legacy_cache=assembled,
        cached_tokens=prefix_tokens + slots,
        query_ids=query_ids,
        answer_ids=answer_ids,
    )
    targets = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
    ce = F.cross_entropy(logits.squeeze(0).float(), targets)
    teacher_logits = teacher_logits.to(device=logits.device, dtype=torch.float32)
    kl = F.kl_div(
        F.log_softmax(logits.float() / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="sum",
    ) * (temperature**2 / targets.numel())
    return ce, kl


@torch.inference_mode()
def _evaluate_case(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    transcoder,
    *,
    variant: str,
    source_state: str,
    source_answer_max_new_tokens: int,
    index: int,
    case: dict[str, Any],
    stage_system_ids: Sequence[int],
    stage_system_cache,
    bridge_system_ids: Sequence[int],
    bridge_system_cache,
    quant_bits_values: Sequence[int],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    source_cache, source_tokens, source_answer, source_answer_tokens = _source_cache(
        source_model,
        source_tokenizer,
        case,
        source_state=source_state,
        max_new_tokens=source_answer_max_new_tokens,
    )
    condition_state = (
        _condition_state(source_model, source_tokenizer, case["agent_a"]["question"])
        if variant == "conditioned_multilayer"
        else None
    )
    stage_query = build_chat_segments(
        target_tokenizer,
        system_prompt=DEPENDENT_B_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": _format_target(case)}],
        enable_thinking=False,
    )
    bridge_query = build_chat_segments(
        target_tokenizer,
        system_prompt=INTERMEDIATE_B_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": INTERMEDIATE_QUERY}],
        enable_thinking=False,
    )
    if tuple(stage_query.prefix_ids) != tuple(stage_system_ids):
        raise ValueError("stage prefix changed during evaluation")
    if tuple(bridge_query.prefix_ids) != tuple(bridge_system_ids):
        raise ValueError("bridge prefix changed during evaluation")

    teacher_history = [
        {
            "role": "user",
            "content": "Agent A intermediate answer: " + case["agent_a"]["gold_answer"],
        },
        {"role": "assistant", "content": ACK_B},
    ]
    teacher_prefix_ids, teacher_query_ids, _ = _split_chat_before_last_user(
        target_tokenizer,
        system_prompt=DEPENDENT_B_SYSTEM,
        prior_history=teacher_history,
        query=_format_target(case),
    )
    teacher_cache = prefill_legacy_cache(target_model, teacher_prefix_ids)
    teacher_text = _clean_generation(
        _generate(
            target_model,
            target_tokenizer,
            legacy_cache=teacher_cache,
            cached_tokens=len(teacher_prefix_ids),
            query_ids=teacher_query_ids,
            max_new_tokens=max_new_tokens,
        )
    )
    no_summary_text = _clean_generation(
        _generate(
            target_model,
            target_tokenizer,
            legacy_cache=stage_system_cache,
            cached_tokens=len(stage_system_ids),
            query_ids=(*stage_query.history_ids, *stage_query.readout_ids),
            max_new_tokens=max_new_tokens,
        )
    )
    document = {
        "source_cache": source_cache,
        "source_tokens": source_tokens,
        "condition_state": condition_state,
        "source_answer_tokens": source_answer_tokens,
    }
    if isinstance(transcoder, TailKVContentManifoldTranscoder):
        document["source_tail_features"] = transcoder.extract_tail_features(
            source_model,
            source_cache=source_cache,
            source_cached_tokens=source_tokens,
            source_answer_tokens=source_answer_tokens,
        )
    latent = _encode(
        transcoder,
        source_model,
        document,
        quant_bits=None,
        straight_through=False,
    )
    rows = []
    for quant_bits in quant_bits_values:
        transmitted = fake_quantize_latent(
            latent, bits=None if quant_bits == 16 else quant_bits
        )
        stage_compact = _decode(
            transcoder,
            target_model,
            transmitted,
            target_start=len(stage_system_ids),
            prefix_cache=stage_system_cache,
        )
        stage_cache = concatenate_caches(stage_system_cache, stage_compact)
        student_text = _clean_generation(
            _generate(
                target_model,
                target_tokenizer,
                legacy_cache=stage_cache,
                cached_tokens=len(stage_system_ids) + transcoder.slots,
                query_ids=(*stage_query.history_ids, *stage_query.readout_ids),
                max_new_tokens=max_new_tokens,
            )
        )
        bridge_compact = _decode(
            transcoder,
            target_model,
            transmitted,
            target_start=len(bridge_system_ids),
            prefix_cache=bridge_system_cache,
        )
        bridge_cache = concatenate_caches(bridge_system_cache, bridge_compact)
        bridge_text = _clean_generation(
            _generate(
                target_model,
                target_tokenizer,
                legacy_cache=bridge_cache,
                cached_tokens=len(bridge_system_ids) + transcoder.slots,
                query_ids=(*bridge_query.history_ids, *bridge_query.readout_ids),
                max_new_tokens=max_new_tokens,
            )
        )
        final_gold = case["agent_b"]["gold_answer"]
        bridge_gold = case["agent_a"]["gold_answer"]
        rows.append(
            {
                "index": index,
                "id": case["id"],
                "relation_key": case["relation_key"],
                "quant_bits": quant_bits,
                "source_tokens": source_tokens,
                "source_answer": source_answer,
                "source_answer_tokens": source_answer_tokens,
                "student": student_text,
                "bridge_student": bridge_text,
                "teacher": teacher_text,
                "no_summary": no_summary_text,
                "final_gold": final_gold,
                "bridge_gold": bridge_gold,
                "student_f1": coqa_f1(student_text, [final_gold]),
                "bridge_f1": coqa_f1(bridge_text, [bridge_gold]),
                "teacher_f1": coqa_f1(teacher_text, [final_gold]),
                "no_summary_f1": coqa_f1(no_summary_text, [final_gold]),
                "student_exact": float(_normalized(student_text) == _normalized(final_gold)),
                "bridge_exact": float(_normalized(bridge_text) == _normalized(bridge_gold)),
                "teacher_exact": float(_normalized(teacher_text) == _normalized(final_gold)),
                "no_summary_exact": float(
                    _normalized(no_summary_text) == _normalized(final_gold)
                ),
                "source_answer_exact": (
                    float(_normalized(source_answer) == _normalized(bridge_gold))
                    if source_answer is not None
                    else None
                ),
                "source_answer_contains": (
                    float(_normalized(bridge_gold) in _normalized(source_answer))
                    if source_answer is not None
                    else None
                ),
            }
        )
    return rows


def _summarize(
    rows: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    training_seconds: float,
    training_means: dict[str, float],
    resident_train_cache_bytes: int,
    transcoder,
) -> dict[str, Any]:
    by_quant = {}
    for quant_bits in sorted({row["quant_bits"] for row in rows}, reverse=True):
        selected = [row for row in rows if row["quant_bits"] == quant_bits]
        by_quant[f"int{quant_bits}"] = {
            "cases": len(selected),
            "student_accuracy": _mean(row["student_exact"] for row in selected),
            "bridge_accuracy": _mean(row["bridge_exact"] for row in selected),
            "teacher_accuracy": _mean(row["teacher_exact"] for row in selected),
            "no_summary_accuracy": _mean(row["no_summary_exact"] for row in selected),
            "student_mean_f1": _mean(row["student_f1"] for row in selected),
            "bridge_mean_f1": _mean(row["bridge_f1"] for row in selected),
            "wins_vs_no_summary": sum(
                row["student_f1"] > row["no_summary_f1"] for row in selected
            ),
            "mean_source_tokens": _mean(row["source_tokens"] for row in selected),
            "wire_payload_bytes": transcoder.slots
            * transcoder.latent_dim
            * quant_bits
            / 8,
            "source_answer_exact": _optional_mean(
                row["source_answer_exact"] for row in selected
            ),
            "source_answer_contains": _optional_mean(
                row["source_answer_contains"] for row in selected
            ),
        }
    return {
        "config": config,
        "training_seconds": training_seconds,
        "training_means": training_means,
        "resident_train_cache_bytes": resident_train_cache_bytes,
        "latent_elements": transcoder.slots * transcoder.latent_dim,
        "receiver_bf16_cache_bytes": transcoder.receiver_cache_elements * 2,
        "by_quant": by_quant,
    }


def _cache_bytes(cache) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache
        for tensor in layer
    )


def _resident_source_bytes(document: dict[str, Any]) -> int:
    if document.get("source_tail_features") is not None:
        tensor = document["source_tail_features"]
        return tensor.numel() * tensor.element_size()
    return _cache_bytes(document["source_cache"])


def _load_slice(path: str, *, offset: int, count: int) -> list[tuple[int, dict[str, Any]]]:
    selected = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if offset <= index < offset + count:
                selected.append((index, json.loads(line)))
            if index >= offset + count:
                break
    if len(selected) != count:
        raise ValueError(f"dataset {path} has only {len(selected)} requested cases")
    return selected


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _optional_mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return _mean(values) if values else None


if __name__ == "__main__":
    main()
