from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from xmodel_kv.attention_operator import cache_attention_operator_distillation_loss
from xmodel_kv.differentiable_delta import fake_int4_cache_round_trip
from xmodel_kv.gradient_alignment import gradient_alignment, loss_gradient_tuple
from xmodel_kv.native_tail_distillation import cache_prefix_relative_mse
from xmodel_kv.policy_imprinting import prefill_legacy_cache
from xmodel_kv.receiver_query_probes import capture_teacher_forced_query_probes
from xmodel_kv.soft_tail_capsule import (
    load_soft_tail_capsule,
    materialize_token_tail,
    reposition_tail_cache,
)

from .train_musique_semantic_handoff import (
    DEPENDENT_B_SYSTEM,
    INTERMEDIATE_B_SYSTEM,
    INTERMEDIATE_QUERY,
    _load_model,
    _load_slice,
    _system_cache,
)
from .train_musique_soft_tail_capsule import (
    _answer_ce,
    _prepare_document,
    _task_logits,
)


ALIGNMENTS = (
    "coordinate_final",
    "coordinate_bridge",
    "operator_final",
    "operator_bridge",
    "final_bridge",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure native-Tail and downstream gradient alignment."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--offset", type=int, default=96)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--teacher-token-cap", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2098)
    args = parser.parse_args()
    if min(args.count, args.teacher_token_cap, args.bootstrap_replicates) < 1:
        parser.error("counts and teacher cap must be positive")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    capsule, training_config = load_soft_tail_capsule(args.checkpoint)
    capsule = capsule.to(args.device).train()
    if capsule.slots != args.teacher_token_cap:
        parser.error("this equal-cap diagnostic requires teacher cap == capsule slots")
    parameters = tuple(parameter for parameter in capsule.parameters() if parameter.requires_grad)
    if not parameters:
        parser.error("capsule checkpoint has no trainable parameters")

    stage_system_ids, stage_system_cache = _system_cache(
        model, tokenizer, DEPENDENT_B_SYSTEM, "placeholder"
    )
    bridge_system_ids, bridge_system_cache = _system_cache(
        model, tokenizer, INTERMEDIATE_B_SYSTEM, INTERMEDIATE_QUERY
    )
    cases = _load_slice(args.dataset, offset=args.offset, count=args.count)
    training_ids = set(training_config.get("train_case_ids", ()))
    overlap = training_ids & {case["id"] for _, case in cases}
    if overlap:
        parser.error(f"gradient probe overlaps checkpoint training IDs: {len(overlap)}")

    packet_layers = frozenset(range(model.config.num_hidden_layers))
    teacher_layers = frozenset(range(1, model.config.num_hidden_layers))
    rows = []
    for ordinal, (index, case) in enumerate(cases, 1):
        document = _prepare_document(
            tokenizer,
            index=index,
            case=case,
            stage_system_ids=stage_system_ids,
            bridge_system_ids=bridge_system_ids,
        )
        source_cache = prefill_legacy_cache(model, document["source_ids"])
        with torch.no_grad():
            teacher_tail = materialize_token_tail(
                model,
                document["bridge_answer_ids"][: args.teacher_token_cap],
                prefix_cache=source_cache,
                prefix_tokens=len(document["source_ids"]),
            )
            query_probes = capture_teacher_forced_query_probes(
                model,
                legacy_cache=stage_system_cache,
                cached_tokens=len(stage_system_ids),
                query_ids=document["stage_query_ids"],
                answer_ids=document["stage_answer_ids"],
                position_shift=capsule.slots,
            )
        student_tail = capsule.materialize(
            model,
            prefix_cache=source_cache,
            prefix_tokens=len(document["source_ids"]),
        )
        canonical_student = _canonical_int4(
            model,
            student_tail,
            source_start=len(document["source_ids"]),
            active_layers=packet_layers,
            straight_through=True,
        )
        with torch.no_grad():
            canonical_teacher = _canonical_int4(
                model,
                teacher_tail,
                source_start=len(document["source_ids"]),
                active_layers=packet_layers,
                straight_through=False,
            )
        stage_tail = reposition_tail_cache(
            model,
            canonical_student,
            source_start=0,
            target_start=len(stage_system_ids),
        )
        bridge_tail = reposition_tail_cache(
            model,
            canonical_student,
            source_start=0,
            target_start=len(bridge_system_ids),
        )
        teacher_stage_tail = reposition_tail_cache(
            model,
            canonical_teacher,
            source_start=0,
            target_start=len(stage_system_ids),
        )
        stage_logits = _task_logits(
            model,
            prefix_cache=stage_system_cache,
            prefix_tokens=len(stage_system_ids),
            compact_cache=stage_tail,
            slots=capsule.slots,
            query_ids=document["stage_query_ids"],
            answer_ids=document["stage_answer_ids"],
        )
        bridge_logits = _task_logits(
            model,
            prefix_cache=bridge_system_cache,
            prefix_tokens=len(bridge_system_ids),
            compact_cache=bridge_tail,
            slots=capsule.slots,
            query_ids=document["bridge_query_ids"],
            answer_ids=document["bridge_answer_ids"],
        )
        losses = {
            "final": _answer_ce(stage_logits, document["stage_answer_ids"]),
            "bridge": _answer_ce(bridge_logits, document["bridge_answer_ids"]),
            "coordinate": cache_prefix_relative_mse(
                canonical_student,
                canonical_teacher,
                active_layers=teacher_layers,
            ),
            "operator": cache_attention_operator_distillation_loss(
                stage_tail,
                teacher_stage_tail,
                query_probes,
                active_layers=teacher_layers,
                value_weight=1.0,
                log_mass_weight=0.1,
            ).total,
        }
        gradients = {}
        loss_names = tuple(losses)
        for loss_position, loss_name in enumerate(loss_names):
            gradients[loss_name] = loss_gradient_tuple(
                losses[loss_name],
                parameters,
                retain_graph=loss_position < len(loss_names) - 1,
            )
        alignments = {
            "coordinate_final": gradient_alignment(
                gradients["coordinate"], gradients["final"]
            ),
            "coordinate_bridge": gradient_alignment(
                gradients["coordinate"], gradients["bridge"]
            ),
            "operator_final": gradient_alignment(
                gradients["operator"], gradients["final"]
            ),
            "operator_bridge": gradient_alignment(
                gradients["operator"], gradients["bridge"]
            ),
            "final_bridge": gradient_alignment(
                gradients["final"], gradients["bridge"]
            ),
        }
        row = {
            "index": index,
            "id": case["id"],
            **{f"{name}_loss": float(loss.detach()) for name, loss in losses.items()},
        }
        for name, alignment in alignments.items():
            row[f"{name}_dot"] = alignment.dot
            row[f"{name}_cosine"] = alignment.cosine
            row[f"{name}_first_norm"] = alignment.first_norm
            row[f"{name}_second_norm"] = alignment.second_norm
        rows.append(row)
        print(json.dumps({"event": "progress", "position": ordinal, **row}), flush=True)
        del source_cache, teacher_tail, query_probes, student_tail
        del canonical_student, canonical_teacher, stage_tail, bridge_tail
        del teacher_stage_tail, stage_logits, bridge_logits, losses, gradients
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = analyze_gradient_rows(
        rows, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed
    )
    result = {
        "analysis": "native_tail_task_gradient_alignment",
        "cases": args.count,
        "checkpoint": args.checkpoint,
        "teacher_token_cap": args.teacher_token_cap,
        "packet_layers": sorted(packet_layers),
        "teacher_layers": sorted(teacher_layers),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        **summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "complete", **{k: v for k, v in result.items() if k != "rows"}}))


def analyze_gradient_rows(rows, *, bootstrap_replicates: int, seed: int) -> dict:
    if not rows or bootstrap_replicates < 1:
        raise ValueError("rows and bootstrap count must be non-empty")
    rng = np.random.default_rng(seed)
    summaries = {}
    for name in ALIGNMENTS:
        values = np.asarray([row[f"{name}_cosine"] for row in rows], dtype=np.float64)
        samples = values[
            rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
        ].mean(axis=1)
        summaries[name] = {
            "mean_cosine": float(values.mean()),
            "bootstrap_95ci": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
            "negative_fraction": float(np.mean(values < 0)),
            "positive_fraction": float(np.mean(values > 0)),
            "zero_fraction": float(np.mean(values == 0)),
        }
    coordinate_gap = np.asarray(
        [
            row["coordinate_bridge_cosine"] - row["coordinate_final_cosine"]
            for row in rows
        ],
        dtype=np.float64,
    )
    operator_gap = np.asarray(
        [row["operator_bridge_cosine"] - row["operator_final_cosine"] for row in rows],
        dtype=np.float64,
    )
    gaps = {}
    for name, values in (
        ("coordinate_bridge_minus_final", coordinate_gap),
        ("operator_bridge_minus_final", operator_gap),
    ):
        samples = values[
            rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
        ].mean(axis=1)
        gaps[name] = {
            "mean": float(values.mean()),
            "bootstrap_95ci": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
        }
    return {"alignments": summaries, "bridge_minus_final_alignment": gaps}


def _canonical_int4(
    model, tail, *, source_start: int, active_layers, straight_through: bool
):
    tokens = tail[0][0].shape[-2]
    canonical = reposition_tail_cache(
        model, tail, source_start=source_start, target_start=0
    )
    return fake_int4_cache_round_trip(
        canonical,
        suffix_tokens=tokens,
        active_layers=active_layers,
        straight_through=straight_through,
    )


if __name__ == "__main__":
    main()
