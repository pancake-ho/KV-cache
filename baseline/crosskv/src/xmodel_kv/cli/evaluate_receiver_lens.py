from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from xmodel_kv.capsule_codec import unpack_int4_capsule
from xmodel_kv.coqa import coqa_f1
from xmodel_kv.receiver_lens import ReceiverLens, cache_with_lens
from xmodel_kv.receiver_lens_handoff import (
    build_route_prefix,
    lens_token_ids,
    materialize_canonical_packet,
    prepare_lens_document,
)
from xmodel_kv.semantic_kv_summary import concatenate_caches
from xmodel_kv.soft_tail_capsule import (
    load_soft_tail_capsule,
    materialize_token_tail,
    reposition_tail_cache,
)

from .evaluate_amortized_musique_handoff import _clean_generation, _format_target
from .evaluate_musique_agent_a import SOURCE_ANSWER_SYSTEM
from .evaluate_musique_soft_tail_capsule import shifted_source_cases
from .probe_semantic_kv_summary import _generate, _normalized
from .train_musique_semantic_handoff import (
    DEPENDENT_B_SYSTEM,
    INTERMEDIATE_B_SYSTEM,
    INTERMEDIATE_QUERY,
    _load_model,
    _load_slice,
    _system_cache,
)


ARM_NAMES = ("base", "hard4", "lens4")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate base, hard-token, and receiver-lens readout arms."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--lens-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--source-case-shift", type=int, default=0)
    parser.add_argument("--arms", default=",".join(ARM_NAMES))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if min(args.count, args.max_new_tokens) <= 0 or args.offset < 0:
        parser.error("count and token limit must be positive; offset non-negative")
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not arms or len(set(arms)) != len(arms) or not set(arms).issubset(ARM_NAMES):
        parser.error(f"--arms must be a unique subset of {','.join(ARM_NAMES)}")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "arms": list(arms), "wire_codec": "canonical_real_k4v4"}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base, base_config = load_soft_tail_capsule(args.base_checkpoint)
    if base.slots != 8 or base.embeddings.shape[1] != model.config.hidden_size:
        raise ValueError("base checkpoint is not the compatible CE8 capsule")
    checkpoint_model = base_config.get("model")
    if checkpoint_model is not None and str(checkpoint_model) != str(args.model):
        raise ValueError("base checkpoint model differs from receiver model")
    base = base.to(args.device).eval()
    base.requires_grad_(False)

    lens_payload = torch.load(
        args.lens_checkpoint, map_location="cpu", weights_only=True
    )
    embeddings = lens_payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (4, model.config.hidden_size):
        raise ValueError("lens checkpoint embeddings are incompatible")
    hard_ids = tuple(int(value) for value in lens_payload.get("hard_token_ids", ()))
    if hard_ids != lens_token_ids(tokenizer, slots=4):
        raise ValueError("lens checkpoint hard-token initialization differs")
    lens_config = lens_payload.get("config", {})
    base_hash = hashlib.sha256(Path(args.base_checkpoint).read_bytes()).hexdigest()
    if lens_config.get("base_checkpoint_sha256") != base_hash:
        raise ValueError("lens was trained against a different semantic base")
    lens = ReceiverLens(embeddings).to(args.device).eval()

    stage_system_ids, stage_system_cache = _system_cache(
        model, tokenizer, DEPENDENT_B_SYSTEM, "placeholder"
    )
    bridge_system_ids, bridge_system_cache = _system_cache(
        model, tokenizer, INTERMEDIATE_B_SYSTEM, INTERMEDIATE_QUERY
    )
    cases = _load_slice(args.dataset, offset=args.offset, count=args.count)
    sources = shifted_source_cases(cases, args.source_case_shift)
    rows = []
    for position, ((index, case), (source_index, source_case)) in enumerate(
        zip(cases, sources, strict=True), 1
    ):
        target_document = prepare_document(
            tokenizer,
            index=index,
            case=case,
            stage_system_ids=stage_system_ids,
            bridge_system_ids=bridge_system_ids,
        )
        source_document = prepare_document(
            tokenizer,
            index=source_index,
            case=source_case,
            stage_system_ids=stage_system_ids,
            bridge_system_ids=bridge_system_ids,
        )
        packet_started = time.perf_counter()
        packet = materialize_canonical_packet(model, base, source_document)
        packet_seconds = time.perf_counter() - packet_started
        if len(packet) != 270426:
            raise RuntimeError(f"unexpected semantic packet size {len(packet)}")
        canonical = unpack_int4_capsule(
            packet,
            dtype=model.model.embed_tokens.weight.dtype,
            device=model.model.embed_tokens.weight.device,
        )
        stage_outputs, stage_times = evaluate_route(
            model,
            tokenizer,
            lens,
            hard_ids=hard_ids,
            arms=arms,
            canonical=canonical,
            system_cache=stage_system_cache,
            system_tokens=len(stage_system_ids),
            semantic_slots=base.slots,
            route=target_document.stage,
            max_new_tokens=args.max_new_tokens,
        )
        bridge_outputs, bridge_times = evaluate_route(
            model,
            tokenizer,
            lens,
            hard_ids=hard_ids,
            arms=arms,
            canonical=canonical,
            system_cache=bridge_system_cache,
            system_tokens=len(bridge_system_ids),
            semantic_slots=base.slots,
            route=target_document.bridge,
            max_new_tokens=args.max_new_tokens,
        )
        no_summary = _clean_generation(
            _generate(
                model,
                tokenizer,
                legacy_cache=stage_system_cache,
                cached_tokens=len(stage_system_ids),
                query_ids=(
                    *target_document.stage.history_ids,
                    *target_document.stage.readout_ids,
                ),
                max_new_tokens=args.max_new_tokens,
            )
        )
        row = {
            "index": index,
            "id": case["id"],
            "source_index": source_index,
            "source_id": source_case["id"],
            "source_case_shift": args.source_case_shift,
            "source_tokens": len(source_document.source_ids),
            "semantic_slots": base.slots,
            "lens_slots": lens.slots,
            "actual_packet_bytes": len(packet),
            "lens_wire_bytes": 0,
            "lens_local_bf16_cache_bytes": 524288,
            "packet_preparation_seconds": packet_seconds,
            "final_gold": case["agent_b"]["gold_answer"],
            "bridge_gold": case["agent_a"]["gold_answer"],
            "no_summary": no_summary,
            "no_summary_f1": coqa_f1(no_summary, [case["agent_b"]["gold_answer"]]),
        }
        for arm in arms:
            final = stage_outputs[arm]
            bridge = bridge_outputs[arm]
            row.update(
                {
                    f"{arm}_student": final,
                    f"{arm}_bridge_student": bridge,
                    f"{arm}_student_f1": coqa_f1(
                        final, [case["agent_b"]["gold_answer"]]
                    ),
                    f"{arm}_bridge_f1": coqa_f1(
                        bridge, [case["agent_a"]["gold_answer"]]
                    ),
                    f"{arm}_student_exact": float(
                        _normalized(final)
                        == _normalized(case["agent_b"]["gold_answer"])
                    ),
                    f"{arm}_bridge_exact": float(
                        _normalized(bridge)
                        == _normalized(case["agent_a"]["gold_answer"])
                    ),
                    f"{arm}_stage_materialization_seconds": stage_times[arm],
                    f"{arm}_bridge_materialization_seconds": bridge_times[arm],
                }
            )
        rows.append(row)
        print(json.dumps({"event": "evaluate", **row}), flush=True)
        print(
            json.dumps(
                {"event": "evaluation_progress", "position": position, "cases": len(cases)}
            ),
            flush=True,
        )
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    summary = summarize(rows, arms)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def prepare_document(tokenizer, *, index, case, stage_system_ids, bridge_system_ids):
    return prepare_lens_document(
        tokenizer,
        index=index,
        case=case,
        source_system_prompt=SOURCE_ANSWER_SYSTEM,
        stage_system_prompt=DEPENDENT_B_SYSTEM,
        stage_system_ids=stage_system_ids,
        stage_query=_format_target(case),
        bridge_system_prompt=INTERMEDIATE_B_SYSTEM,
        bridge_system_ids=bridge_system_ids,
        bridge_query=INTERMEDIATE_QUERY,
    )


@torch.inference_mode()
def evaluate_route(
    model,
    tokenizer,
    lens,
    *,
    hard_ids,
    arms,
    canonical,
    system_cache,
    system_tokens,
    semantic_slots,
    route,
    max_new_tokens,
):
    semantic_tail = reposition_tail_cache(
        model, canonical, source_start=0, target_start=system_tokens
    )
    prefix, cached_tokens = build_route_prefix(
        model,
        system_cache=system_cache,
        system_tokens=system_tokens,
        semantic_tail=semantic_tail,
        semantic_slots=semantic_slots,
        history_ids=route.history_ids,
    )
    outputs = {}
    materialization_times = {}
    for arm in arms:
        started = time.perf_counter()
        if arm == "base":
            cache = prefix
            arm_cached_tokens = cached_tokens
        elif arm == "hard4":
            local_tail = materialize_token_tail(
                model,
                hard_ids,
                prefix_cache=prefix,
                prefix_tokens=cached_tokens,
            )
            cache = concatenate_caches(prefix, local_tail)
            arm_cached_tokens = cached_tokens + len(hard_ids)
        elif arm == "lens4":
            local_tail = lens.materialize(
                model, prefix_cache=prefix, prefix_tokens=cached_tokens
            )
            cache = cache_with_lens(prefix, local_tail)
            arm_cached_tokens = cached_tokens + lens.slots
        else:
            raise ValueError(f"unknown arm {arm}")
        materialization_times[arm] = time.perf_counter() - started
        outputs[arm] = _clean_generation(
            _generate(
                model,
                tokenizer,
                legacy_cache=cache,
                cached_tokens=arm_cached_tokens,
                query_ids=route.readout_ids,
                max_new_tokens=max_new_tokens,
            )
        )
    return outputs, materialization_times


def summarize(rows, arms):
    result = {"cases": len(rows)}
    for arm in arms:
        result[arm] = {
            "student_f1": sum(row[f"{arm}_student_f1"] for row in rows) / len(rows),
            "bridge_f1": sum(row[f"{arm}_bridge_f1"] for row in rows) / len(rows),
            "student_exact": sum(row[f"{arm}_student_exact"] for row in rows) / len(rows),
            "bridge_exact": sum(row[f"{arm}_bridge_exact"] for row in rows) / len(rows),
            "mean_stage_materialization_seconds": sum(
                row[f"{arm}_stage_materialization_seconds"] for row in rows
            )
            / len(rows),
            "mean_bridge_materialization_seconds": sum(
                row[f"{arm}_bridge_materialization_seconds"] for row in rows
            )
            / len(rows),
        }
    result["no_summary_f1"] = sum(row["no_summary_f1"] for row in rows) / len(rows)
    result["mean_packet_preparation_seconds"] = sum(
        row["packet_preparation_seconds"] for row in rows
    ) / len(rows)
    return result


if __name__ == "__main__":
    main()
