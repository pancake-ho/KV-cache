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

from xmodel_kv.capsule_codec import unpack_int4_capsule
from xmodel_kv.receiver_lens import ReceiverLens, cache_with_lens
from xmodel_kv.receiver_lens_handoff import (
    build_route_prefix,
    lens_token_ids,
    materialize_canonical_packet,
    prepare_lens_document,
)
from xmodel_kv.semantic_kv_summary import teacher_forced_answer_logits
from xmodel_kv.soft_tail_capsule import load_soft_tail_capsule, reposition_tail_cache

from .evaluate_amortized_musique_handoff import _format_target
from .evaluate_musique_agent_a import SOURCE_ANSWER_SYSTEM
from .train_musique_semantic_handoff import (
    DEPENDENT_B_SYSTEM,
    INTERMEDIATE_B_SYSTEM,
    INTERMEDIATE_QUERY,
    _load_model,
    _load_slice,
    _system_cache,
)
from .train_musique_soft_tail_capsule import _answer_ce, parameter_sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train four receiver-local readout embeddings over a frozen CE8 packet."
    )
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=568)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--stage-weight", type=float, default=1.0)
    parser.add_argument("--bridge-weight", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2111)
    parser.add_argument(
        "--packet-cache",
        default=None,
        help="Optional previously prepared packet-cache .pt with matching IDs and base hash.",
    )
    args = parser.parse_args()
    positive = (
        args.train_count,
        args.slots,
        args.steps,
        args.learning_rate,
        args.grad_clip,
        args.log_every,
    )
    if min(positive) <= 0 or args.train_offset < 0:
        parser.error("counts, slots, steps, rates, and logging interval must be positive")
    if min(args.stage_weight, args.bridge_weight, args.weight_decay) < 0:
        parser.error("loss weights and weight decay must be non-negative")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint_sha256 = hashlib.sha256(
        Path(args.base_checkpoint).read_bytes()
    ).hexdigest()
    config = {
        **vars(args),
        "method": "receiver_lens4_frozen_ce8",
        "base_checkpoint_sha256": base_checkpoint_sha256,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base, base_config = load_soft_tail_capsule(args.base_checkpoint)
    if base.slots != 8 or base.embeddings.shape[1] != model.config.hidden_size:
        raise ValueError("frozen base must be the compatible eight-slot capsule")
    checkpoint_model = base_config.get("model")
    if checkpoint_model is not None and str(checkpoint_model) != str(args.model):
        raise ValueError("base checkpoint model differs from receiver model")
    base = base.to(args.device).eval()
    base.requires_grad_(False)
    base_parameter_sha256_before = parameter_sha256(base)

    hard_ids = lens_token_ids(tokenizer, slots=args.slots)
    hard_index = torch.tensor(hard_ids, dtype=torch.long, device=args.device)
    initial = model.model.embed_tokens(hard_index).detach().float()
    lens = ReceiverLens(initial).to(args.device)
    if [name for name, _ in lens.named_parameters()] != ["embeddings"]:
        raise RuntimeError("fixed feasibility arm must train only lens embeddings")
    config.update(
        {
            "hard_token_ids": list(hard_ids),
            "base_slots": base.slots,
            "trainable_parameter_count": sum(p.numel() for p in lens.parameters()),
            "base_parameter_sha256_before": base_parameter_sha256_before,
        }
    )

    stage_system_ids, stage_system_cache = _system_cache(
        model, tokenizer, DEPENDENT_B_SYSTEM, "placeholder"
    )
    bridge_system_ids, bridge_system_cache = _system_cache(
        model, tokenizer, INTERMEDIATE_B_SYSTEM, INTERMEDIATE_QUERY
    )
    cases = _load_slice(
        args.train_dataset, offset=args.train_offset, count=args.train_count
    )
    documents = [
        prepare_lens_document(
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
        for index, case in cases
    ]
    config["train_case_ids"] = [document.case_id for document in documents]
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    packet_started = time.perf_counter()
    if args.packet_cache:
        packet_payload = torch.load(
            args.packet_cache, map_location="cpu", weights_only=True
        )
        if packet_payload.get("base_checkpoint_sha256") != base_checkpoint_sha256:
            raise ValueError("packet cache base checkpoint hash differs")
        if packet_payload.get("case_ids") != config["train_case_ids"]:
            raise ValueError("packet cache case IDs differ")
        packets = packet_payload.get("packets")
        if not isinstance(packets, list) or len(packets) != len(documents):
            raise ValueError("packet cache payload is incomplete")
        for document, packet in zip(documents, packets, strict=True):
            if not isinstance(packet, bytes):
                raise ValueError("packet cache contains a non-byte packet")
            document.packet = packet
    else:
        packets = []
        for position, document in enumerate(documents, 1):
            document.packet = materialize_canonical_packet(model, base, document)
            packets.append(document.packet)
            print(
                json.dumps(
                    {
                        "event": "prepared_semantic_packet",
                        "position": position,
                        "cases": len(documents),
                        "packet_bytes": len(document.packet),
                    }
                ),
                flush=True,
            )
            gc.collect()
        packet_path = output_dir / "train_packets.pt"
        torch.save(
            {
                "base_checkpoint_sha256": base_checkpoint_sha256,
                "case_ids": config["train_case_ids"],
                "packets": packets,
            },
            packet_path,
        )
        config["prepared_packet_cache"] = str(packet_path)
    packet_seconds = time.perf_counter() - packet_started
    packet_sizes = {len(document.packet) for document in documents}
    if packet_sizes != {270426}:
        raise RuntimeError(f"unexpected semantic packet sizes: {sorted(packet_sizes)}")
    config["packet_preparation_seconds"] = packet_seconds
    config["semantic_packet_bytes"] = next(iter(packet_sizes))
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    optimizer = AdamW(
        lens.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    schedule = list(documents)
    generator = random.Random(args.seed)
    totals = {"loss": 0.0, "stage_ce": 0.0, "bridge_ce": 0.0, "grad_norm": 0.0}
    rows = []
    started = time.perf_counter()
    lens.train()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(schedule) == 0:
            generator.shuffle(schedule)
        document = schedule[(step - 1) % len(schedule)]
        canonical = unpack_int4_capsule(
            document.packet,
            dtype=model.model.embed_tokens.weight.dtype,
            device=model.model.embed_tokens.weight.device,
        )
        optimizer.zero_grad(set_to_none=True)
        stage_ce = route_ce(
            model,
            lens,
            canonical=canonical,
            system_cache=stage_system_cache,
            system_tokens=len(stage_system_ids),
            semantic_slots=base.slots,
            route=document.stage,
        )
        bridge_ce = route_ce(
            model,
            lens,
            canonical=canonical,
            system_cache=bridge_system_cache,
            system_tokens=len(bridge_system_ids),
            semantic_slots=base.slots,
            route=document.bridge,
        )
        loss = args.stage_weight * stage_ce + args.bridge_weight * bridge_ce
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(lens.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        values = {
            "loss": float(loss.detach()),
            "stage_ce": float(stage_ce.detach()),
            "bridge_ce": float(bridge_ce.detach()),
            "grad_norm": float(grad_norm.detach()),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"non-finite metrics at step {step}")
        for name, value in values.items():
            totals[name] += value
        rows.append({"step": step, "document": document.index, **values})
        if step == 1 or step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "document": document.index,
                        **{f"mean_{name}": total / step for name, total in totals.items()},
                    }
                ),
                flush=True,
            )
        del canonical, stage_ce, bridge_ce, loss, grad_norm

    training_seconds = time.perf_counter() - started
    base_parameter_sha256_after = parameter_sha256(base)
    config.update(
        {
            "training_seconds": training_seconds,
            "training_means": {
                name: total / args.steps for name, total in totals.items()
            },
            "base_parameter_sha256_after": base_parameter_sha256_after,
            "base_parameters_unchanged": (
                base_parameter_sha256_before == base_parameter_sha256_after
            ),
        }
    )
    if not config["base_parameters_unchanged"]:
        raise RuntimeError("frozen semantic base changed during lens training")
    (output_dir / "training_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.save(
        {
            "embeddings": lens.embeddings.detach().cpu(),
            "hard_token_ids": hard_ids,
            "config": config,
        },
        output_dir / "lens.pt",
    )
    print(json.dumps({"event": "complete", "config": config}), flush=True)


def route_ce(
    model,
    lens,
    *,
    canonical,
    system_cache,
    system_tokens: int,
    semantic_slots: int,
    route,
) -> torch.Tensor:
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
    lens_tail = lens.materialize(
        model, prefix_cache=prefix, prefix_tokens=cached_tokens
    )
    logits = teacher_forced_answer_logits(
        model,
        legacy_cache=cache_with_lens(prefix, lens_tail),
        cached_tokens=cached_tokens + lens.slots,
        query_ids=route.readout_ids,
        answer_ids=route.answer_ids,
    )
    return _answer_ce(logits, route.answer_ids)


if __name__ == "__main__":
    main()
