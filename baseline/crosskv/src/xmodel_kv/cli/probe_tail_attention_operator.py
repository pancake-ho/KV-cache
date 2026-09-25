from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from xmodel_kv.attention_operator import (
    cache_attention_operator_distillation_loss,
)
from xmodel_kv.differentiable_delta import fake_int4_cache_round_trip
from xmodel_kv.policy_imprinting import prefill_legacy_cache
from xmodel_kv.receiver_query_probes import capture_teacher_forced_query_probes
from xmodel_kv.soft_tail_capsule import (
    load_soft_tail_capsule,
    materialize_token_tail,
    reposition_tail_cache,
)

from .analyze_native_tail_state_distillation import _paired_delta
from .train_musique_semantic_handoff import (
    DEPENDENT_B_SYSTEM,
    INTERMEDIATE_B_SYSTEM,
    INTERMEDIATE_QUERY,
    _load_model,
    _load_slice,
    _system_cache,
)
from .train_musique_soft_tail_capsule import _prepare_document


ARMS = ("state4", "ce8", "state8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe query-conditioned attention operators of Tail-KV students."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    for arm in ARMS:
        parser.add_argument(f"--{arm}-checkpoint", required=True)
        parser.add_argument(f"--{arm}-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--offset", type=int, default=32)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2097)
    args = parser.parse_args()
    if min(args.count, args.bootstrap_replicates) < 1 or args.offset < 0:
        parser.error("count and bootstrap replicates must be positive")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    capsules = {}
    for arm in ARMS:
        capsule, _ = load_soft_tail_capsule(getattr(args, f"{arm}_checkpoint"))
        capsules[arm] = capsule.to(args.device).eval().requires_grad_(False)
    if capsules["state4"].slots != 4 or any(
        capsules[arm].slots != 8 for arm in ("ce8", "state8")
    ):
        parser.error("the probe requires one four-slot and two eight-slot capsules")

    stage_system_ids, stage_system_cache = _system_cache(
        model, tokenizer, DEPENDENT_B_SYSTEM, "placeholder"
    )
    bridge_system_ids, _ = _system_cache(
        model, tokenizer, INTERMEDIATE_B_SYSTEM, INTERMEDIATE_QUERY
    )
    cases = _load_slice(args.dataset, offset=args.offset, count=args.count)
    quality = {
        arm: _load_quality(getattr(args, f"{arm}_results"), expected=args.count)
        for arm in ARMS
    }
    rows = []
    packet_layers = frozenset(range(model.config.num_hidden_layers))
    operator_layers = frozenset(range(1, model.config.num_hidden_layers))
    for ordinal, (index, case) in enumerate(cases, 1):
        document = _prepare_document(
            tokenizer,
            index=index,
            case=case,
            stage_system_ids=stage_system_ids,
            bridge_system_ids=bridge_system_ids,
        )
        with torch.inference_mode():
            source_cache = prefill_legacy_cache(model, document["source_ids"])
            student_tails = {
                arm: capsules[arm].materialize(
                    model,
                    prefix_cache=source_cache,
                    prefix_tokens=len(document["source_ids"]),
                )
                for arm in ARMS
            }
            teacher_tails = {
                slots: materialize_token_tail(
                    model,
                    document["bridge_answer_ids"][:slots],
                    prefix_cache=source_cache,
                    prefix_tokens=len(document["source_ids"]),
                )
                for slots in (4, 8)
            }
            probes = {
                slots: capture_teacher_forced_query_probes(
                    model,
                    legacy_cache=stage_system_cache,
                    cached_tokens=len(stage_system_ids),
                    query_ids=document["stage_query_ids"],
                    answer_ids=document["stage_answer_ids"],
                    position_shift=slots,
                )
                for slots in (4, 8)
            }
            target_students = {
                arm: _canonical_int4_to_target(
                    model,
                    student_tails[arm],
                    source_start=len(document["source_ids"]),
                    target_start=len(stage_system_ids),
                    active_layers=packet_layers,
                )
                for arm in ARMS
            }
            target_teachers = {
                slots: _canonical_int4_to_target(
                    model,
                    teacher_tails[slots],
                    source_start=len(document["source_ids"]),
                    target_start=len(stage_system_ids),
                    active_layers=packet_layers,
                )
                for slots in (4, 8)
            }
            losses = {}
            for arm in ARMS:
                slots = capsules[arm].slots
                losses[arm] = cache_attention_operator_distillation_loss(
                    target_students[arm],
                    target_teachers[slots],
                    probes[slots],
                    active_layers=operator_layers,
                    value_weight=1.0,
                    log_mass_weight=0.1,
                )
        case_id = case["id"]
        row = {"index": index, "id": case_id}
        for arm in ARMS:
            row[f"{arm}_operator_total"] = float(losses[arm].total.item())
            row[f"{arm}_operator_value"] = float(losses[arm].value.item())
            row[f"{arm}_operator_log_mass"] = float(losses[arm].log_mass.item())
            row[f"{arm}_final_f1"] = quality[arm][case_id]["student_f1"]
        rows.append(row)
        print(json.dumps({"event": "progress", "position": ordinal, **row}), flush=True)
        del source_cache, student_tails, teacher_tails, probes, target_students
        del target_teachers, losses
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tables = {
        arm: {
            row["id"]: {
                "operator_total": row[f"{arm}_operator_total"],
                "operator_value": row[f"{arm}_operator_value"],
                "operator_log_mass": row[f"{arm}_operator_log_mass"],
                "final_f1": row[f"{arm}_final_f1"],
            }
            for row in rows
        }
        for arm in ARMS
    }
    comparisons = {}
    for candidate, reference in (("state8", "ce8"), ("state8", "state4")):
        comparisons[f"{candidate}_minus_{reference}"] = {
            field: _paired_delta(
                tables[candidate],
                tables[reference],
                field=field,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            for field in (
                "operator_total",
                "operator_value",
                "operator_log_mass",
                "final_f1",
            )
        }
    operator_delta = np.asarray(
        [row["state8_operator_total"] - row["ce8_operator_total"] for row in rows]
    )
    quality_delta = np.asarray(
        [row["state8_final_f1"] - row["ce8_final_f1"] for row in rows]
    )
    correlation = float(np.corrcoef(operator_delta, quality_delta)[0, 1])
    summary = {
        "analysis": "tail_attention_operator_probe",
        "cases": args.count,
        "query_probe": "no_state_teacher_forced_final_answer_positions",
        "operator_layers": sorted(operator_layers),
        "loss_weights": {"conditional_value": 1.0, "log_mass": 0.1},
        "absolute": {
            arm: {
                field: float(np.mean([item[field] for item in table.values()]))
                for field in (
                    "operator_total",
                    "operator_value",
                    "operator_log_mass",
                    "final_f1",
                )
            }
            for arm, table in tables.items()
        },
        "comparisons": comparisons,
        "pearson_state8_minus_ce8_operator_vs_f1": correlation,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", **{k: v for k, v in summary.items() if k != "rows"}}))


def _canonical_int4_to_target(
    model, tail, *, source_start: int, target_start: int, active_layers
):
    tokens = tail[0][0].shape[-2]
    canonical = reposition_tail_cache(
        model, tail, source_start=source_start, target_start=0
    )
    decoded = fake_int4_cache_round_trip(
        canonical,
        suffix_tokens=tokens,
        active_layers=active_layers,
        straight_through=False,
    )
    return reposition_tail_cache(
        model, decoded, source_start=0, target_start=target_start
    )


def _load_quality(path, *, expected: int):
    path = Path(path)
    if path.is_dir():
        path = path / "results.jsonl"
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("quant_bits") != 4 or row.get("layer_pattern") != "all":
                continue
            rows[row["id"]] = row
    if len(rows) != expected:
        raise ValueError(f"expected {expected} quality cases, found {len(rows)}")
    return rows


if __name__ == "__main__":
    main()
