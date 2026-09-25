from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from xmodel_kv.capsule_codec import unpack_int4_capsule
from xmodel_kv.receiver_lens import ReceiverLens
from xmodel_kv.receiver_lens_gradient_structure import (
    analyze_receiver_lens_gradients,
)
from xmodel_kv.receiver_lens_handoff import (
    LensDocument,
    materialize_canonical_packet,
    prepare_lens_document,
)
from xmodel_kv.soft_tail_capsule import load_soft_tail_capsule

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
from .train_musique_soft_tail_capsule import parameter_sha256
from .train_receiver_lens import route_ce


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe frozen Receiver-Lens task-gradient conflict and structure."
    )
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--musique-dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--lens-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--packet-cache", default=None)
    parser.add_argument("--gradient-output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--dev-offset", type=int, default=0)
    parser.add_argument("--dev-count", type=int, default=56)
    parser.add_argument("--musique-offset", type=int, default=96)
    parser.add_argument("--musique-count", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2113)
    args = parser.parse_args()
    if min(args.dev_count, args.musique_count, args.bootstrap_replicates) < 1:
        parser.error("counts and bootstrap_replicates must be positive")
    if min(args.dev_offset, args.musique_offset) < 0:
        parser.error("offsets must be non-negative")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    packet_path = (
        Path(args.packet_cache)
        if args.packet_cache
        else output.with_suffix(".packets.pt")
    )
    gradient_path = (
        Path(args.gradient_output)
        if args.gradient_output
        else output.with_suffix(".gradients.pt")
    )
    base_file_hash = file_sha256(args.base_checkpoint)
    lens_file_hash = file_sha256(args.lens_checkpoint)
    dataset_hashes = {
        "dev": file_sha256(args.dev_dataset),
        "musique": file_sha256(args.musique_dataset),
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base, base_config = load_soft_tail_capsule(args.base_checkpoint)
    base = base.to(args.device).eval().requires_grad_(False)
    if base.slots != 8 or base.embeddings.shape[1] != model.config.hidden_size:
        raise ValueError("frozen base must be a compatible eight-slot capsule")

    lens_payload = torch.load(
        args.lens_checkpoint, map_location="cpu", weights_only=True
    )
    lens_config = lens_payload.get("config", {})
    embeddings = lens_payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("lens checkpoint has no embedding tensor")
    lens = ReceiverLens(embeddings).to(args.device).eval()
    if lens.slots != 4 or lens.embeddings.shape[1] != model.config.hidden_size:
        raise ValueError("diagnostic requires the compatible four-slot lens")
    if lens_config.get("base_checkpoint_sha256") != base_file_hash:
        raise ValueError("lens checkpoint was trained against another base file")
    checkpoint_model = lens_config.get("model", base_config.get("model"))
    if checkpoint_model is not None and str(checkpoint_model) != str(args.model):
        raise ValueError("checkpoint model differs from receiver model")

    base_parameter_hash_before = parameter_sha256(base)
    lens_parameter_hash_before = parameter_sha256(lens)
    stage_system_ids, stage_system_cache = _system_cache(
        model, tokenizer, DEPENDENT_B_SYSTEM, "placeholder"
    )
    bridge_system_ids, bridge_system_cache = _system_cache(
        model, tokenizer, INTERMEDIATE_B_SYSTEM, INTERMEDIATE_QUERY
    )
    cases = {
        "dev": _load_slice(
            args.dev_dataset, offset=args.dev_offset, count=args.dev_count
        ),
        "musique": _load_slice(
            args.musique_dataset,
            offset=args.musique_offset,
            count=args.musique_count,
        ),
    }
    documents = {
        group: prepare_documents(
            tokenizer,
            group_cases,
            stage_system_ids=stage_system_ids,
            bridge_system_ids=bridge_system_ids,
        )
        for group, group_cases in cases.items()
    }
    training_ids = set(lens_config.get("train_case_ids", ()))
    overlap = training_ids & {
        document.case_id
        for group_documents in documents.values()
        for document in group_documents
    }
    if overlap:
        raise ValueError(f"diagnostic cases overlap lens training IDs: {len(overlap)}")

    packet_started = time.perf_counter()
    packets = load_or_materialize_packets(
        packet_path,
        model=model,
        base=base,
        documents=documents,
        base_file_hash=base_file_hash,
        dataset_hashes=dataset_hashes,
    )
    packet_seconds = time.perf_counter() - packet_started
    if {len(packet) for values in packets.values() for packet in values} != {270426}:
        raise RuntimeError("diagnostic packet size differs from frozen real codec")

    matrices: dict[str, dict[str, list[torch.Tensor]]] = {
        group: {route: [] for route in ("final", "bridge", "joint")}
        for group in documents
    }
    rows = []
    probe_started = time.perf_counter()
    ordinal = 0
    total = sum(len(values) for values in documents.values())
    for group in ("dev", "musique"):
        for document, packet in zip(documents[group], packets[group], strict=True):
            ordinal += 1
            canonical = unpack_int4_capsule(
                packet,
                dtype=model.model.embed_tokens.weight.dtype,
                device=model.model.embed_tokens.weight.device,
            )
            final_loss, final_gradient = route_loss_gradient(
                model,
                lens,
                canonical=canonical,
                system_cache=stage_system_cache,
                system_tokens=len(stage_system_ids),
                semantic_slots=base.slots,
                route=document.stage,
            )
            bridge_loss, bridge_gradient = route_loss_gradient(
                model,
                lens,
                canonical=canonical,
                system_cache=bridge_system_cache,
                system_tokens=len(bridge_system_ids),
                semantic_slots=base.slots,
                route=document.bridge,
            )
            joint_gradient = final_gradient + 2.0 * bridge_gradient
            matrices[group]["final"].append(final_gradient)
            matrices[group]["bridge"].append(bridge_gradient)
            matrices[group]["joint"].append(joint_gradient)
            final_norm = float(final_gradient.norm())
            bridge_norm = float(bridge_gradient.norm())
            joint_norm = float(joint_gradient.norm())
            final_bridge_cosine = float(
                torch.dot(final_gradient, bridge_gradient)
                / max(final_norm * bridge_norm, 1e-30)
            )
            row = {
                "group": group,
                "index": document.index,
                "id": document.case_id,
                "final_ce": final_loss,
                "bridge_ce": bridge_loss,
                "final_gradient_norm": final_norm,
                "bridge_gradient_norm": bridge_norm,
                "joint_gradient_norm": joint_norm,
                "final_bridge_gradient_cosine": final_bridge_cosine,
            }
            rows.append(row)
            print(
                json.dumps(
                    {"event": "gradient_probe", "position": ordinal, "cases": total, **row}
                ),
                flush=True,
            )
            del canonical, final_gradient, bridge_gradient, joint_gradient
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tensors = {
        group: {
            route: torch.stack(values).float()
            for route, values in group_matrices.items()
        }
        for group, group_matrices in matrices.items()
    }
    summary = analyze_receiver_lens_gradients(
        tensors["dev"],
        tensors["musique"],
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    base_parameter_hash_after = parameter_sha256(base)
    lens_parameter_hash_after = parameter_sha256(lens)
    immutable = {
        "base": base_parameter_hash_before == base_parameter_hash_after,
        "lens": lens_parameter_hash_before == lens_parameter_hash_after,
    }
    if not all(immutable.values()):
        raise RuntimeError("a frozen diagnostic parameter changed")
    gradient_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "case_ids": {
                group: [document.case_id for document in group_documents]
                for group, group_documents in documents.items()
            },
            "gradients": tensors,
        },
        gradient_path,
    )
    result = {
        "analysis": "receiver_lens_task_gradient_structure",
        "model": args.model,
        "base_checkpoint": args.base_checkpoint,
        "base_checkpoint_sha256": base_file_hash,
        "lens_checkpoint": args.lens_checkpoint,
        "lens_checkpoint_sha256": lens_file_hash,
        "dataset_sha256": dataset_hashes,
        "cases": {group: len(values) for group, values in documents.items()},
        "training_id_overlap": 0,
        "semantic_packet_bytes": 270426,
        "lens_slots": lens.slots,
        "trainable_parameter_dimension": lens.embeddings.numel(),
        "packet_cache": str(packet_path),
        "gradient_artifact": str(gradient_path),
        "packet_preparation_seconds": packet_seconds,
        "gradient_probe_seconds": time.perf_counter() - probe_started,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "parameter_sha256": {
            "base_before": base_parameter_hash_before,
            "base_after": base_parameter_hash_after,
            "lens_before": lens_parameter_hash_before,
            "lens_after": lens_parameter_hash_after,
        },
        "parameters_unchanged": immutable,
        **summary,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"event": "complete", **{key: value for key, value in result.items() if key != "rows"}}
        ),
        flush=True,
    )


def prepare_documents(
    tokenizer,
    cases,
    *,
    stage_system_ids,
    bridge_system_ids,
) -> list[LensDocument]:
    return [
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


def load_or_materialize_packets(
    path: Path,
    *,
    model,
    base,
    documents: dict[str, list[LensDocument]],
    base_file_hash: str,
    dataset_hashes: dict[str, str],
) -> dict[str, list[bytes]]:
    case_ids = {
        group: [document.case_id for document in group_documents]
        for group, group_documents in documents.items()
    }
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("base_checkpoint_sha256") != base_file_hash:
            raise ValueError("packet cache base hash differs")
        if payload.get("dataset_sha256") != dataset_hashes:
            raise ValueError("packet cache dataset hash differs")
        if payload.get("case_ids") != case_ids:
            raise ValueError("packet cache case IDs differ")
        packets = payload.get("packets")
        if not isinstance(packets, dict):
            raise ValueError("packet cache payload is incomplete")
        return packets

    packets: dict[str, list[bytes]] = {}
    total = sum(len(values) for values in documents.values())
    position = 0
    for group in ("dev", "musique"):
        packets[group] = []
        for document in documents[group]:
            position += 1
            packet = materialize_canonical_packet(model, base, document)
            packets[group].append(packet)
            print(
                json.dumps(
                    {
                        "event": "prepared_semantic_packet",
                        "position": position,
                        "cases": total,
                        "group": group,
                        "packet_bytes": len(packet),
                    }
                ),
                flush=True,
            )
            gc.collect()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base_checkpoint_sha256": base_file_hash,
            "dataset_sha256": dataset_hashes,
            "case_ids": case_ids,
            "packets": packets,
        },
        path,
    )
    return packets


def route_loss_gradient(model, lens, **kwargs) -> tuple[float, torch.Tensor]:
    loss = route_ce(model, lens, **kwargs)
    gradient = torch.autograd.grad(loss, lens.embeddings, retain_graph=False)[0]
    gradient = gradient.detach().float().cpu().flatten()
    if not torch.isfinite(gradient).all() or gradient.norm() <= 0:
        raise FloatingPointError("route produced an invalid lens gradient")
    return float(loss.detach()), gradient


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
