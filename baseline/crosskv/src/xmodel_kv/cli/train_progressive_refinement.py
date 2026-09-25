from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from xmodel_kv.capsule_codec import pack_int4_capsule, unpack_int4_capsule
from xmodel_kv.differentiable_delta import fake_int4_cache_round_trip
from xmodel_kv.hotpotqa import answer_em, answer_f1
from xmodel_kv.policy_imprinting import build_chat_segments, prefill_legacy_cache, start_readout
from xmodel_kv.soft_tail_capsule import (
    BoundaryKVWriteAdapter,
    ProgressiveSoftTailCapsule,
    concatenate_tail_caches,
    load_soft_tail_capsule,
    move_legacy_cache,
    reposition_tail_cache,
)

from .evaluate_hotpot_summary_transfer import (
    QUESTION_B_SYSTEM,
    format_source_user,
    generate_target,
    load_slice,
    prepare_target_spec,
)
from .evaluate_musique_agent_a import SOURCE_ANSWER_SYSTEM
from .evaluate_musique_tail_transplant import active_layer_indices
from .train_musique_semantic_handoff import _load_model
from .train_musique_soft_tail_capsule import (
    _answer_ce,
    _initial_embeddings,
    _task_logits,
    parameter_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a frozen-core, causally appended KV refinement packet."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--core-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--eval-offset", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=16)
    parser.add_argument("--refinement-slots", type=int, default=4)
    parser.add_argument("--refinement-rank", type=int, default=16)
    parser.add_argument("--boundary-layer", type=int, default=14)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--causal-contrast-weight", type=float, default=0.0)
    parser.add_argument("--causal-margin", type=float, default=0.5)
    parser.add_argument("--layer-pattern", default="first_16")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--source-case-shift", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if min(
        args.train_count,
        args.eval_count,
        args.refinement_slots,
        args.refinement_rank,
        args.steps,
        args.max_new_tokens,
        args.source_case_shift,
        args.log_every,
    ) < 1 or min(args.train_offset, args.eval_offset, args.boundary_layer) < 0:
        parser.error("counts/ranks must be positive and offsets/layer non-negative")
    if not all(
        math.isfinite(value) and value > 0
        for value in (args.learning_rate, args.grad_clip)
    ) or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("optimizer values are invalid")
    if (
        not math.isfinite(args.causal_contrast_weight)
        or args.causal_contrast_weight < 0
        or not math.isfinite(args.causal_margin)
        or args.causal_margin <= 0
    ):
        parser.error("causal contrast weight/margin are invalid")
    if args.causal_contrast_weight and args.train_count < 2:
        parser.error("causal contrast requires at least two training cases")
    if args.source_case_shift % args.eval_count == 0:
        parser.error("source-case-shift must change every evaluation source")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    core, core_config = load_soft_tail_capsule(args.core_checkpoint)
    core = core.to(args.device).eval()
    refinement_boundary = BoundaryKVWriteAdapter(
        hidden_size=model.config.hidden_size,
        layer_index=args.boundary_layer,
        rank=args.refinement_rank,
    )
    progressive = ProgressiveSoftTailCapsule(
        core,
        _initial_embeddings(model, tokenizer, slots=args.refinement_slots),
        refinement_boundary_write_adapter=refinement_boundary,
    ).to(args.device)
    trainable_parameters = [
        parameter for parameter in progressive.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    if not trainable_parameters or any(
        parameter.requires_grad for parameter in progressive.core.parameters()
    ):
        raise RuntimeError("progressive parameter isolation is invalid")
    active_layers = active_layer_indices(
        args.layer_pattern, total_layers=model.config.num_hidden_layers
    )

    target_spec = prepare_target_spec(model, tokenizer, "question_conditioned")
    source_segments = build_chat_segments(
        tokenizer,
        system_prompt=SOURCE_ANSWER_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": "placeholder"}],
        enable_thinking=False,
    )
    source_system_ids = tuple(source_segments.prefix_ids)
    source_system_cache = prefill_legacy_cache(model, source_system_ids)
    train_cases = load_slice(args.dataset, offset=args.train_offset, count=args.train_count)
    eval_cases = load_slice(args.dataset, offset=args.eval_offset, count=args.eval_count)
    if {document.get("_id", str(index)) for index, document in train_cases} & {
        document.get("_id", str(index)) for index, document in eval_cases
    }:
        raise ValueError("training and evaluation IDs overlap")
    train_documents = prepare_documents(
        model,
        tokenizer,
        train_cases,
        source_system_ids=source_system_ids,
        source_system_cache=source_system_cache,
        target_system_ids=target_spec["system_ids"],
        device=args.device,
    )
    eval_documents = prepare_documents(
        model,
        tokenizer,
        eval_cases,
        source_system_ids=source_system_ids,
        source_system_cache=source_system_cache,
        target_system_ids=target_spec["system_ids"],
        device=args.device,
    )
    del source_system_cache
    gc.collect()
    torch.cuda.empty_cache()

    config = {
        **vars(args),
        "core_config": core_config,
        "core_checkpoint_sha256": _file_sha256(args.core_checkpoint),
        "dataset_sha256": _file_sha256(args.dataset),
        "core_slots": progressive.core_slots,
        "total_slots": progressive.slots,
        "trainable_parameter_count": trainable_count,
        "active_layers": sorted(active_layers),
        "train_ids": [document["id"] for document in train_documents],
        "eval_ids": [document["id"] for document in eval_documents],
        "core_parameter_sha256_before": parameter_sha256(
            progressive.core, requires_grad=False
        ),
        "trainable_parameter_sha256_before": parameter_sha256(
            progressive, requires_grad=True
        ),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = random.Random(args.seed)
    schedule = [generator.randrange(len(train_documents)) for _ in range(args.steps)]
    log_path = output_dir / "training_log.jsonl"
    totals = {
        "loss": 0.0,
        "correct_ce": 0.0,
        "negative_ce": 0.0,
        "causal_hinge": 0.0,
        "grad_norm": 0.0,
    }
    start = time.perf_counter()
    with log_path.open("w") as log_file:
        for step, document_index in enumerate(schedule, 1):
            document = train_documents[document_index]
            optimizer.zero_grad(set_to_none=True)
            source_cache = move_legacy_cache(document["source_cache"], args.device)
            core_tail, refinement_tail = progressive.materialize_parts(
                model,
                prefix_cache=source_cache,
                prefix_tokens=document["source_tokens"],
            )
            receiver_tail = differentiable_canonical_receiver_tail(
                model,
                core_tail=core_tail,
                refinement_tail=refinement_tail,
                source_tokens=document["source_tokens"],
                target_start=len(target_spec["system_ids"]),
                active_layers=active_layers,
            )
            logits = _task_logits(
                model,
                prefix_cache=target_spec["system_cache"],
                prefix_tokens=len(target_spec["system_ids"]),
                compact_cache=receiver_tail,
                slots=progressive.slots,
                query_ids=document["query_ids"],
                answer_ids=document["answer_ids"],
            )
            correct_ce = _answer_ce(logits, document["answer_ids"])
            negative_ce = correct_ce * 0
            causal_hinge = correct_ce * 0
            negative_tensors = None
            if args.causal_contrast_weight:
                negative_document = train_documents[
                    (document_index + 1) % len(train_documents)
                ]
                negative_source_cache = move_legacy_cache(
                    negative_document["source_cache"], args.device
                )
                negative_core, negative_refinement = progressive.materialize_parts(
                    model,
                    prefix_cache=negative_source_cache,
                    prefix_tokens=negative_document["source_tokens"],
                )
                negative_receiver = differentiable_canonical_receiver_tail(
                    model,
                    core_tail=negative_core,
                    refinement_tail=negative_refinement,
                    source_tokens=negative_document["source_tokens"],
                    target_start=len(target_spec["system_ids"]),
                    active_layers=active_layers,
                )
                negative_logits = _task_logits(
                    model,
                    prefix_cache=target_spec["system_cache"],
                    prefix_tokens=len(target_spec["system_ids"]),
                    compact_cache=negative_receiver,
                    slots=progressive.slots,
                    query_ids=document["query_ids"],
                    answer_ids=document["answer_ids"],
                )
                negative_ce = _answer_ce(negative_logits, document["answer_ids"])
                causal_hinge = torch.relu(
                    args.causal_margin + correct_ce - negative_ce
                )
                negative_tensors = (
                    negative_source_cache,
                    negative_core,
                    negative_refinement,
                    negative_receiver,
                    negative_logits,
                )
            loss = correct_ce + args.causal_contrast_weight * causal_hinge
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.grad_clip
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient at step {step}")
            optimizer.step()
            row = {
                "step": step,
                "document": document_index,
                "id": document["id"],
                "loss": float(loss.detach()),
                "correct_ce": float(correct_ce.detach()),
                "negative_ce": float(negative_ce.detach()),
                "causal_hinge": float(causal_hinge.detach()),
                "grad_norm": float(grad_norm),
            }
            for field in totals:
                totals[field] += row[field]
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(json.dumps({"event": "train", **row}), flush=True)
            del (
                source_cache,
                core_tail,
                refinement_tail,
                receiver_tail,
                logits,
                correct_ce,
                negative_ce,
                causal_hinge,
                loss,
            )
            if negative_tensors is not None:
                del (
                    negative_source_cache,
                    negative_core,
                    negative_refinement,
                    negative_receiver,
                    negative_logits,
                    negative_tensors,
                )
    training_seconds = time.perf_counter() - start

    config.update(
        {
            "training_seconds": training_seconds,
            **{f"mean_{field}": total / args.steps for field, total in totals.items()},
            "core_parameter_sha256_after": parameter_sha256(
                progressive.core, requires_grad=False
            ),
            "trainable_parameter_sha256_after": parameter_sha256(
                progressive, requires_grad=True
            ),
        }
    )
    config["core_parameters_unchanged"] = (
        config["core_parameter_sha256_before"]
        == config["core_parameter_sha256_after"]
    )
    config["trainable_parameters_changed"] = (
        config["trainable_parameter_sha256_before"]
        != config["trainable_parameter_sha256_after"]
    )
    if not config["core_parameters_unchanged"]:
        raise RuntimeError("frozen core parameters changed")
    if not config["trainable_parameters_changed"]:
        raise RuntimeError("refinement parameters did not change")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.save(
        {
            "refinement_embeddings": progressive.refinement_embeddings.detach().cpu(),
            "refinement_boundary_write_adapter_config": (
                progressive.refinement_boundary_write_adapter.checkpoint_config()
            ),
            "refinement_boundary_write_adapter_state": {
                key: value.detach().cpu()
                for key, value in progressive.refinement_boundary_write_adapter.state_dict().items()
            },
            "config": config,
        },
        output_dir / "refinement.pt",
    )

    rows = evaluate_progressive(
        model,
        tokenizer,
        progressive.eval(),
        eval_documents,
        target_spec=target_spec,
        active_layers=active_layers,
        source_shift=args.source_case_shift,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    summary = summarize_rows(rows, config=config)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def prepare_documents(
    model,
    tokenizer,
    cases,
    *,
    source_system_ids,
    source_system_cache,
    target_system_ids,
    device: str,
) -> list[dict]:
    prepared = []
    for position, (index, document) in enumerate(cases, 1):
        source_segments = build_chat_segments(
            tokenizer,
            system_prompt=SOURCE_ANSWER_SYSTEM,
            tools=[],
            history=[{"role": "user", "content": format_source_user(document)}],
            enable_thinking=False,
        )
        if tuple(source_segments.prefix_ids) != tuple(source_system_ids):
            raise ValueError("source system prefix changed")
        source_query = (*source_segments.history_ids, *source_segments.readout_ids)
        with torch.inference_mode():
            source = start_readout(
                model,
                legacy_cache=source_system_cache,
                cached_tokens=len(source_system_ids),
                input_ids=source_query,
            )
        target_segments = build_chat_segments(
            tokenizer,
            system_prompt=QUESTION_B_SYSTEM,
            tools=[],
            history=[{"role": "user", "content": document["input"]}],
            enable_thinking=False,
        )
        if tuple(target_segments.prefix_ids) != tuple(target_system_ids):
            raise ValueError("target system prefix changed")
        answer_ids = tuple(
            tokenizer.encode(document["answers"][0], add_special_tokens=False)
        )
        if not answer_ids:
            raise ValueError("gold answer tokenized to empty")
        prepared.append(
            {
                "index": index,
                "id": document.get("_id", str(index)),
                "question": document["input"],
                "gold_answers": list(document["answers"]),
                "source_tokens": source.cached_tokens,
                "source_cache": move_legacy_cache(source.past_key_values, "cpu"),
                "query_ids": (*target_segments.history_ids, *target_segments.readout_ids),
                "answer_ids": answer_ids,
            }
        )
        print(
            json.dumps(
                {
                    "event": "prepare",
                    "position": position,
                    "id": prepared[-1]["id"],
                    "source_tokens": source.cached_tokens,
                    "answer_tokens": len(answer_ids),
                }
            ),
            flush=True,
        )
        del source
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return prepared


def differentiable_canonical_receiver_tail(
    model,
    *,
    core_tail,
    refinement_tail,
    source_tokens: int,
    target_start: int,
    active_layers: frozenset[int],
):
    core_slots = core_tail[0][0].shape[-2]
    refinement_slots = refinement_tail[0][0].shape[-2]
    canonical_core = reposition_tail_cache(
        model, core_tail, source_start=source_tokens, target_start=0
    )
    canonical_refinement = reposition_tail_cache(
        model,
        refinement_tail,
        source_start=source_tokens + core_slots,
        target_start=core_slots,
    )
    decoded_core = fake_int4_cache_round_trip(
        canonical_core,
        suffix_tokens=core_slots,
        active_layers=active_layers,
        straight_through=False,
    )
    decoded_refinement = fake_int4_cache_round_trip(
        canonical_refinement,
        suffix_tokens=refinement_slots,
        active_layers=active_layers,
        straight_through=True,
    )
    target_core = reposition_tail_cache(
        model, decoded_core, source_start=0, target_start=target_start
    )
    target_refinement = reposition_tail_cache(
        model,
        decoded_refinement,
        source_start=core_slots,
        target_start=target_start + core_slots,
    )
    return concatenate_tail_caches(target_core, target_refinement)


@torch.inference_mode()
def evaluate_progressive(
    model,
    tokenizer,
    progressive,
    documents,
    *,
    target_spec,
    active_layers,
    source_shift: int,
    max_new_tokens: int,
    device: str,
) -> list[dict]:
    packet_states = []
    for position, document in enumerate(documents, 1):
        source_cache = move_legacy_cache(document["source_cache"], device)
        core_tail, refinement_tail = progressive.materialize_parts(
            model,
            prefix_cache=source_cache,
            prefix_tokens=document["source_tokens"],
        )
        canonical_core = reposition_tail_cache(
            model, core_tail, source_start=document["source_tokens"], target_start=0
        )
        canonical_refinement = reposition_tail_cache(
            model,
            refinement_tail,
            source_start=document["source_tokens"] + progressive.core_slots,
            target_start=progressive.core_slots,
        )
        core_packet = pack_int4_capsule(
            canonical_core,
            suffix_tokens=progressive.core_slots,
            active_layers=active_layers,
        )
        refinement_packet = pack_int4_capsule(
            canonical_refinement,
            suffix_tokens=progressive.refinement_slots,
            active_layers=active_layers,
        )
        packet_states.append(
            {
                "source_id": document["id"],
                "core": move_legacy_cache(
                    unpack_int4_capsule(
                        core_packet, dtype=canonical_core[0][0].dtype, device=device
                    ),
                    "cpu",
                ),
                "refinement": move_legacy_cache(
                    unpack_int4_capsule(
                        refinement_packet,
                        dtype=canonical_refinement[0][0].dtype,
                        device=device,
                    ),
                    "cpu",
                ),
                "core_packet_bytes": len(core_packet),
                "refinement_packet_bytes": len(refinement_packet),
                "core_packet_sha256": hashlib.sha256(core_packet).hexdigest(),
                "refinement_packet_sha256": hashlib.sha256(
                    refinement_packet
                ).hexdigest(),
            }
        )
        print(
            json.dumps({"event": "packet", "position": position, "id": document["id"]}),
            flush=True,
        )
        del source_cache, core_tail, refinement_tail, canonical_core, canonical_refinement

    rows = []
    target_start = len(target_spec["system_ids"])
    for ordinal, document in enumerate(documents):
        correct = packet_states[ordinal]
        shifted = packet_states[(ordinal + source_shift) % len(packet_states)]
        no_state, _ = generate_target(
            model,
            tokenizer,
            cache=target_spec["system_cache"],
            cached_tokens=target_start,
            query_ids=document["query_ids"],
            max_new_tokens=max_new_tokens,
        )
        core_cache = receiver_packet_cache(
            model, correct, target_start=target_start, include_refinement=False, device=device
        )
        full_cache = receiver_packet_cache(
            model, correct, target_start=target_start, include_refinement=True, device=device
        )
        shifted_cache = receiver_packet_cache(
            model, shifted, target_start=target_start, include_refinement=True, device=device
        )
        core_output, _ = generate_target(
            model,
            tokenizer,
            cache=concatenate_tail_caches(target_spec["system_cache"], core_cache),
            cached_tokens=target_start + progressive.core_slots,
            query_ids=document["query_ids"],
            max_new_tokens=max_new_tokens,
        )
        full_output, _ = generate_target(
            model,
            tokenizer,
            cache=concatenate_tail_caches(target_spec["system_cache"], full_cache),
            cached_tokens=target_start + progressive.slots,
            query_ids=document["query_ids"],
            max_new_tokens=max_new_tokens,
        )
        shifted_output, _ = generate_target(
            model,
            tokenizer,
            cache=concatenate_tail_caches(target_spec["system_cache"], shifted_cache),
            cached_tokens=target_start + progressive.slots,
            query_ids=document["query_ids"],
            max_new_tokens=max_new_tokens,
        )
        golds = document["gold_answers"]
        row = {
            "index": document["index"],
            "id": document["id"],
            "question": document["question"],
            "gold_answers": golds,
            "source_id": correct["source_id"],
            "shifted_source_id": shifted["source_id"],
            "core": core_output,
            "full": full_output,
            "shifted_full": shifted_output,
            "no_state": no_state,
            "core_em": answer_em(core_output, golds),
            "core_f1": answer_f1(core_output, golds),
            "full_em": answer_em(full_output, golds),
            "full_f1": answer_f1(full_output, golds),
            "shifted_full_em": answer_em(shifted_output, golds),
            "shifted_full_f1": answer_f1(shifted_output, golds),
            "no_state_em": answer_em(no_state, golds),
            "no_state_f1": answer_f1(no_state, golds),
            "core_packet_bytes": correct["core_packet_bytes"],
            "refinement_packet_bytes": correct["refinement_packet_bytes"],
            "core_packet_sha256": correct["core_packet_sha256"],
            "refinement_packet_sha256": correct["refinement_packet_sha256"],
        }
        rows.append(row)
        print(json.dumps({"event": "evaluate", **row}, ensure_ascii=False), flush=True)
        del core_cache, full_cache, shifted_cache
    return rows


def receiver_packet_cache(
    model,
    state,
    *,
    target_start: int,
    include_refinement: bool,
    device: str,
):
    core = reposition_tail_cache(
        model,
        move_legacy_cache(state["core"], device),
        source_start=0,
        target_start=target_start,
    )
    if not include_refinement:
        return core
    core_slots = core[0][0].shape[-2]
    refinement = reposition_tail_cache(
        model,
        move_legacy_cache(state["refinement"], device),
        source_start=core_slots,
        target_start=target_start + core_slots,
    )
    return concatenate_tail_caches(core, refinement)


def summarize_rows(rows: list[dict], *, config: dict) -> dict:
    mean = lambda field: sum(float(row[field]) for row in rows) / len(rows)
    return {
        "config": config,
        "cases": len(rows),
        "arms": {
            arm: {"em": mean(f"{arm}_em"), "f1": mean(f"{arm}_f1")}
            for arm in ("core", "full", "shifted_full", "no_state")
        },
        "mean_core_packet_bytes": mean("core_packet_bytes"),
        "mean_refinement_packet_bytes": mean("refinement_packet_bytes"),
    }


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
