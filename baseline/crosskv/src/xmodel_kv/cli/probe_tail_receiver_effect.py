from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from xmodel_kv.differentiable_delta import fake_int4_cache_round_trip
from xmodel_kv.policy_imprinting import prefill_legacy_cache
from xmodel_kv.receiver_effect import (
    counterfactual_receiver_effect_loss,
    predictor_hidden_trajectory,
    slice_cache_tokens,
    zero_cache_like,
)
from xmodel_kv.semantic_kv_summary import concatenate_caches
from xmodel_kv.soft_tail_capsule import (
    load_soft_tail_capsule,
    reposition_tail_cache,
)

from .evaluate_amortized_musique_handoff import _format_target
from .probe_semantic_kv_summary import ACK_B, _split_chat_before_last_user
from .train_musique_semantic_handoff import (
    DEPENDENT_B_SYSTEM,
    INTERMEDIATE_B_SYSTEM,
    INTERMEDIATE_QUERY,
    _load_model,
    _load_slice,
    _system_cache,
)
from .train_musique_soft_tail_capsule import _prepare_document


ARM_NAMES = ("state4", "ce8", "state8")
METRICS = ("effect_total", "effect_direction", "effect_log_norm", "final_f1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare compact caches by the marginal effect they induce on a full "
            "receiver trajectory relative to a plaintext handoff."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    for name in ARM_NAMES:
        parser.add_argument(f"--{name}-checkpoint", required=True)
        parser.add_argument(f"--{name}-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--offset", type=int, default=32)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--log-norm-weight", type=float, default=0.1)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2099)
    args = parser.parse_args()
    if min(args.count, args.bootstrap_replicates) < 1 or args.offset < 0:
        parser.error("counts must be positive and offset non-negative")
    if args.log_norm_weight < 0:
        parser.error("log-norm weight must be non-negative")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    capsules = {}
    checkpoint_configs = {}
    result_metrics = {}
    for name in ARM_NAMES:
        checkpoint = getattr(args, f"{name}_checkpoint")
        capsule, config = load_soft_tail_capsule(checkpoint)
        capsules[name] = capsule.to(args.device).eval()
        capsules[name].requires_grad_(False)
        checkpoint_configs[name] = config
        result_metrics[name] = _load_result_f1(getattr(args, f"{name}_results"))

    if capsules["state4"].slots != 4:
        parser.error("state4 checkpoint must contain four slots")
    if capsules["ce8"].slots != 8 or capsules["state8"].slots != 8:
        parser.error("ce8 and state8 checkpoints must contain eight slots")

    stage_system_ids, stage_system_cache = _system_cache(
        model, tokenizer, DEPENDENT_B_SYSTEM, "placeholder"
    )
    bridge_system_ids, _ = _system_cache(
        model, tokenizer, INTERMEDIATE_B_SYSTEM, INTERMEDIATE_QUERY
    )
    cases = _load_slice(args.dataset, offset=args.offset, count=args.count)
    probe_ids = {case["id"] for _, case in cases}
    for name, config in checkpoint_configs.items():
        overlap = set(config.get("train_case_ids", ())) & probe_ids
        if overlap:
            parser.error(f"{name} checkpoint training overlap: {len(overlap)}")
        missing = probe_ids - set(result_metrics[name])
        if missing:
            parser.error(f"{name} results omit {len(missing)} probe IDs")

    packet_layers = frozenset(range(model.config.num_hidden_layers))
    rows = []
    with torch.no_grad():
        for ordinal, (index, case) in enumerate(cases, 1):
            document = _prepare_document(
                tokenizer,
                index=index,
                case=case,
                stage_system_ids=stage_system_ids,
                bridge_system_ids=bridge_system_ids,
            )
            teacher_prefix_ids, teacher_query_ids, teacher_system_ids = (
                _split_chat_before_last_user(
                    tokenizer,
                    system_prompt=DEPENDENT_B_SYSTEM,
                    prior_history=[
                        {
                            "role": "user",
                            "content": (
                                "Agent A intermediate answer: "
                                + case["agent_a"]["gold_answer"]
                            ),
                        },
                        {"role": "assistant", "content": ACK_B},
                    ],
                    query=_format_target(case),
                )
            )
            if tuple(teacher_system_ids) != tuple(stage_system_ids):
                raise ValueError("plaintext and capsule system prefixes differ")
            if tuple(teacher_query_ids) != tuple(document["stage_query_ids"]):
                raise ValueError("plaintext and capsule receiver query tokens differ")

            teacher_prefix_cache = prefill_legacy_cache(model, teacher_prefix_ids)
            teacher_memory = slice_cache_tokens(
                teacher_prefix_cache, len(stage_system_ids)
            )
            teacher_null_prefix = concatenate_caches(
                stage_system_cache, zero_cache_like(teacher_memory)
            )
            teacher_trajectory = predictor_hidden_trajectory(
                model,
                legacy_cache=teacher_prefix_cache,
                cached_tokens=len(teacher_prefix_ids),
                query_ids=teacher_query_ids,
                answer_ids=document["stage_answer_ids"],
            )
            teacher_null_trajectory = predictor_hidden_trajectory(
                model,
                legacy_cache=teacher_null_prefix,
                cached_tokens=len(teacher_prefix_ids),
                query_ids=teacher_query_ids,
                answer_ids=document["stage_answer_ids"],
            )

            source_cache = prefill_legacy_cache(model, document["source_ids"])
            null_trajectories = {}
            arm_values = {}
            for name in ARM_NAMES:
                capsule = capsules[name]
                source_tail = capsule.materialize(
                    model,
                    prefix_cache=source_cache,
                    prefix_tokens=len(document["source_ids"]),
                )
                canonical = _canonical_int4(
                    model,
                    source_tail,
                    source_start=len(document["source_ids"]),
                    active_layers=packet_layers,
                )
                stage_tail = reposition_tail_cache(
                    model,
                    canonical,
                    source_start=0,
                    target_start=len(stage_system_ids),
                )
                student_prefix = concatenate_caches(stage_system_cache, stage_tail)
                student_trajectory = predictor_hidden_trajectory(
                    model,
                    legacy_cache=student_prefix,
                    cached_tokens=len(stage_system_ids) + capsule.slots,
                    query_ids=document["stage_query_ids"],
                    answer_ids=document["stage_answer_ids"],
                )
                if capsule.slots not in null_trajectories:
                    null_prefix = concatenate_caches(
                        stage_system_cache, zero_cache_like(stage_tail)
                    )
                    null_trajectories[capsule.slots] = predictor_hidden_trajectory(
                        model,
                        legacy_cache=null_prefix,
                        cached_tokens=len(stage_system_ids) + capsule.slots,
                        query_ids=document["stage_query_ids"],
                        answer_ids=document["stage_answer_ids"],
                    )
                effect = counterfactual_receiver_effect_loss(
                    student_trajectory,
                    null_trajectories[capsule.slots],
                    teacher_trajectory,
                    teacher_null_trajectory,
                    log_norm_weight=args.log_norm_weight,
                )
                arm_values[name] = {
                    "effect_total": float(effect.total),
                    "effect_direction": float(effect.direction),
                    "effect_log_norm": float(effect.log_norm),
                    "final_f1": result_metrics[name][case["id"]],
                }
                del source_tail, canonical, stage_tail, student_prefix
                del student_trajectory

            row = {
                "index": index,
                "id": case["id"],
                "plaintext_memory_tokens": len(teacher_prefix_ids)
                - len(stage_system_ids),
                "arms": arm_values,
            }
            rows.append(row)
            print(
                json.dumps({"event": "progress", "position": ordinal, **row}),
                flush=True,
            )
            del teacher_prefix_cache, teacher_memory, teacher_null_prefix
            del teacher_trajectory, teacher_null_trajectory, source_cache
            del null_trajectories, arm_values
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    analysis = analyze_receiver_effect_rows(
        rows, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed
    )
    result = {
        "analysis": "counterfactual_full_receiver_effect",
        "cases": args.count,
        "offset": args.offset,
        "packet_layers": sorted(packet_layers),
        "log_norm_weight": args.log_norm_weight,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        **analysis,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"event": "complete", **{k: v for k, v in result.items() if k != "rows"}}),
        flush=True,
    )


def analyze_receiver_effect_rows(
    rows, *, bootstrap_replicates: int, seed: int
) -> dict:
    if not rows or bootstrap_replicates < 1:
        raise ValueError("rows and bootstrap count must be non-empty")
    rng = np.random.default_rng(seed)
    arm_means = {
        arm: {
            metric: float(np.mean([row["arms"][arm][metric] for row in rows]))
            for metric in METRICS
        }
        for arm in ARM_NAMES
    }
    comparisons = {}
    for left, right in (("state8", "ce8"), ("state4", "ce8")):
        fields = {}
        for metric in METRICS:
            values = np.asarray(
                [
                    row["arms"][left][metric] - row["arms"][right][metric]
                    for row in rows
                ],
                dtype=np.float64,
            )
            samples = values[
                rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
            ].mean(axis=1)
            fields[metric] = {
                "mean_delta": float(values.mean()),
                "bootstrap_95ci": [
                    float(np.quantile(samples, 0.025)),
                    float(np.quantile(samples, 0.975)),
                ],
                "positive_fraction": float(np.mean(values > 0)),
                "negative_fraction": float(np.mean(values < 0)),
                "zero_fraction": float(np.mean(values == 0)),
            }
        comparisons[f"{left}_minus_{right}"] = fields

    effect_delta = np.asarray(
        [
            row["arms"]["state8"]["effect_total"]
            - row["arms"]["ce8"]["effect_total"]
            for row in rows
        ],
        dtype=np.float64,
    )
    f1_delta = np.asarray(
        [
            row["arms"]["state8"]["final_f1"]
            - row["arms"]["ce8"]["final_f1"]
            for row in rows
        ],
        dtype=np.float64,
    )
    correlation = (
        float(np.corrcoef(effect_delta, f1_delta)[0, 1])
        if effect_delta.std() > 0 and f1_delta.std() > 0
        else None
    )
    return {
        "arm_means": arm_means,
        "comparisons": comparisons,
        "state8_ce8_effect_delta_vs_f1_delta_pearson": correlation,
    }


def _load_result_f1(path: str) -> dict[str, float]:
    values = {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("quant_bits") == 4 and row.get("layer_pattern") == "all":
                values[row["id"]] = float(row["student_f1"])
    if not values:
        raise ValueError(f"no all-layer K4 rows in {path}")
    return values


def _canonical_int4(model, tail, *, source_start: int, active_layers):
    tokens = tail[0][0].shape[-2]
    canonical = reposition_tail_cache(
        model, tail, source_start=source_start, target_start=0
    )
    return fake_int4_cache_round_trip(
        canonical,
        suffix_tokens=tokens,
        active_layers=active_layers,
        straight_through=False,
    )


if __name__ == "__main__":
    main()
