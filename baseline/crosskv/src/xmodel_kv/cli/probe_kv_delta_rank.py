from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..kv_delta_rank import (
    DEFAULT_RANKS,
    context_orders,
    matrix_delta_rank_metrics,
    pca_rank_metrics,
    summarize_rank_rows,
)
from ..rope import model_rope_cos_sin, remove_rope


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the intrinsic per-layer/per-KV-head rank of a fixed document's "
            "de-RoPE K/V change under held-out preceding contexts"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--num-documents", type=int, default=6)
    parser.add_argument("--num-contexts", type=int, default=64)
    parser.add_argument("--train-contexts", type=int, default=48)
    parser.add_argument("--document-tokens", type=int, default=128)
    parser.add_argument("--context-chunk-tokens", type=int, default=32)
    parser.add_argument("--chunks-per-context", type=int, default=6)
    parser.add_argument(
        "--regimes", default="permutation,composition", help="comma-separated regimes"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-dataset-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    regimes = tuple(item.strip() for item in args.regimes.split(",") if item.strip())
    if not regimes or any(item not in {"permutation", "composition"} for item in regimes):
        parser.error("regimes must contain permutation and/or composition")
    positive = (
        args.num_documents,
        args.num_contexts,
        args.train_contexts,
        args.document_tokens,
        args.context_chunk_tokens,
        args.chunks_per_context,
        args.batch_size,
        args.max_dataset_rows,
    )
    if any(value < 1 for value in positive):
        parser.error("numeric size arguments must be positive")
    if args.train_contexts >= args.num_contexts:
        parser.error("train-contexts must be smaller than num-contexts")

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    is_main = rank == 0

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    records = _load_records(
        Path(args.dataset),
        tokenizer,
        minimum_tokens=max(args.document_tokens, args.context_chunk_tokens),
        max_rows=args.max_dataset_rows,
    )
    if len(records) < args.num_documents + args.chunks_per_context:
        raise ValueError(f"only found {len(records)} sufficiently long unique paragraphs")
    selection_generator = torch.Generator().manual_seed(args.seed)
    selection = torch.randperm(len(records), generator=selection_generator).tolist()
    targets = [records[index] for index in selection[: args.num_documents]]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=getattr(torch, args.dtype),
        device_map={"": str(device)},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    header_ids = _encode(tokenizer, "Background passages:\n")
    separator_ids = _encode(tokenizer, "\n\n")
    target_header_ids = _encode(tokenizer, "\n\nTarget passage:\n")

    shard_path = output_dir / f"rank_cells.rank{rank}.jsonl"
    local_targets = list(enumerate(targets))[rank::world_size]
    with shard_path.open("w") as output:
        for local_step, (document_index, target) in enumerate(local_targets, 1):
            pool = [
                record["tokens"][: args.context_chunk_tokens]
                for record in records
                if record["record_id"] != target["record_id"]
            ]
            target_tokens = target["tokens"][: args.document_tokens]
            base_prefix = header_ids + target_header_ids
            isolated = _extract_document_caches(
                model,
                [base_prefix + target_tokens],
                document_start=len(base_prefix),
                document_tokens=args.document_tokens,
                device=device,
                batch_size=1,
            )[:, :, :, 0]

            for regime_index, regime in enumerate(regimes):
                started = time.monotonic()
                generator = torch.Generator().manual_seed(
                    args.seed + document_index * 1009 + regime_index * 104729
                )
                orders = context_orders(
                    len(pool),
                    regime=regime,
                    num_contexts=args.num_contexts,
                    chunks_per_context=args.chunks_per_context,
                    generator=generator,
                )
                sequences = [
                    _context_prefix(
                        order,
                        pool=pool,
                        header_ids=header_ids,
                        separator_ids=separator_ids,
                        target_header_ids=target_header_ids,
                    )
                    + target_tokens
                    for order in orders
                ]
                document_start = len(sequences[0]) - args.document_tokens
                if any(len(sequence) != len(sequences[0]) for sequence in sequences):
                    raise AssertionError("fixed-width contexts produced unequal sequence lengths")
                samples = _extract_document_caches(
                    model,
                    sequences,
                    document_start=document_start,
                    document_tokens=args.document_tokens,
                    device=device,
                    batch_size=args.batch_size,
                )
                cells = _analyze_samples(
                    samples,
                    isolated,
                    train_contexts=args.train_contexts,
                    ranks=DEFAULT_RANKS,
                    device=device,
                )
                row = {
                    "document_index": document_index,
                    "record_id": target["record_id"],
                    "title": target["title"],
                    "regime": regime,
                    "num_contexts": args.num_contexts,
                    "train_contexts": args.train_contexts,
                    "test_contexts": args.num_contexts - args.train_contexts,
                    "document_tokens": args.document_tokens,
                    "context_tokens": document_start,
                    "cells": cells,
                    "elapsed_seconds": time.monotonic() - started,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    {
                        "rank": rank,
                        "document": document_index,
                        "regime": regime,
                        "local_step": f"{local_step}/{len(local_targets)}",
                        "seconds": round(row["elapsed_seconds"], 1),
                    },
                    flush=True,
                )
                del samples, cells
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del isolated

    if distributed:
        dist.barrier()
    if is_main:
        all_rows = []
        for shard_rank in range(world_size):
            with (output_dir / f"rank_cells.rank{shard_rank}.jsonl").open() as source:
                all_rows.extend(json.loads(line) for line in source if line.strip())
        all_rows.sort(key=lambda row: (row["document_index"], row["regime"]))
        summary = {
            "experiment": {
                "model": args.model,
                "dataset": args.dataset,
                "num_documents": args.num_documents,
                "regimes": regimes,
                "num_contexts": args.num_contexts,
                "train_contexts": args.train_contexts,
                "test_contexts": args.num_contexts - args.train_contexts,
                "document_tokens": args.document_tokens,
                "context_chunk_tokens": args.context_chunk_tokens,
                "chunks_per_context": args.chunks_per_context,
                "ranks": DEFAULT_RANKS,
                "seed": args.seed,
                "key_preprocessing": "exact inverse model RoPE at target positions",
                "pca_protocol": "fit on train contexts; oracle-project held-out contexts",
            },
            "rank_summary": summarize_rank_rows(all_rows),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        (output_dir / "rank_cells.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows)
        )
        print(json.dumps(summary["rank_summary"], indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def _load_records(path: Path, tokenizer, *, minimum_tokens: int, max_rows: int):
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open() as source:
        for row_index, line in enumerate(source):
            if row_index >= max_rows:
                break
            example = json.loads(line)
            for paragraph in example.get("paragraphs", ()):
                text = str(paragraph.get("paragraph_text", "")).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                tokens = _encode(tokenizer, text)
                if len(tokens) < minimum_tokens:
                    continue
                records.append(
                    {
                        "record_id": f"{example.get('id', row_index)}:{paragraph.get('idx', 0)}",
                        "title": str(paragraph.get("title", "")),
                        "tokens": tokens,
                    }
                )
    return records


def _encode(tokenizer, text: str) -> list[int]:
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def _context_prefix(
    order,
    *,
    pool,
    header_ids: list[int],
    separator_ids: list[int],
    target_header_ids: list[int],
) -> list[int]:
    prefix = list(header_ids)
    for offset, pool_index in enumerate(order):
        if offset:
            prefix.extend(separator_ids)
        prefix.extend(pool[pool_index])
    prefix.extend(target_header_ids)
    return prefix


@torch.inference_mode()
def _extract_document_caches(
    model,
    sequences: list[list[int]],
    *,
    document_start: int,
    document_tokens: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    config = model.config
    storage = torch.empty(
        (
            config.num_hidden_layers,
            2,
            config.num_key_value_heads,
            len(sequences),
            document_tokens,
            config.head_dim,
        ),
        dtype=torch.bfloat16,
        device="cpu",
    )
    positions = torch.arange(document_start, document_start + document_tokens)
    cos, sin = model_rope_cos_sin(
        model, positions, device=device, dtype=torch.float32
    )
    for begin in range(0, len(sequences), batch_size):
        batch = sequences[begin : begin + batch_size]
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        output = model.model(input_ids=input_ids, use_cache=True, return_dict=True)
        legacy = output.past_key_values.to_legacy_cache()
        end = begin + len(batch)
        for layer, (key, value) in enumerate(legacy):
            key = key[:, :, document_start : document_start + document_tokens]
            value = value[:, :, document_start : document_start + document_tokens]
            content_key = remove_rope(
                key.float(), cos.to(key.device), sin.to(key.device), sequence_dim=2
            )
            storage[layer, 0, :, begin:end].copy_(
                content_key.permute(1, 0, 2, 3).to(device="cpu", dtype=torch.bfloat16)
            )
            storage[layer, 1, :, begin:end].copy_(
                value.permute(1, 0, 2, 3).to(device="cpu", dtype=torch.bfloat16)
            )
        del output, legacy, input_ids
    return storage


@torch.inference_mode()
def _analyze_samples(
    samples: torch.Tensor,
    isolated: torch.Tensor,
    *,
    train_contexts: int,
    ranks,
    device: torch.device,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for layer in range(samples.shape[0]):
        for kind_index, kind in enumerate(("key", "value")):
            matrices = samples[layer, kind_index].to(device=device, dtype=torch.float32)
            isolated_matrices = isolated[layer, kind_index].to(
                device=device, dtype=torch.float32
            )
            paired_count = min(train_contexts, matrices.shape[1] - train_contexts)
            paired_matrix_metrics = matrix_delta_rank_metrics(
                matrices[:, train_contexts : train_contexts + paired_count]
                - matrices[:, :paired_count],
                ranks=ranks,
            )
            isolated_matrix_metrics = matrix_delta_rank_metrics(
                matrices[:, train_contexts:] - isolated_matrices.unsqueeze(1), ranks=ranks
            )
            vectors = matrices.flatten(start_dim=2)
            isolated_vectors = isolated_matrices.flatten(start_dim=1)
            train = vectors[:, :train_contexts]
            test = vectors[:, train_contexts:]
            metrics = pca_rank_metrics(train, test, ranks=ranks)

            train_center = train.mean(dim=1)
            centered_energy = (vectors - train_center.unsqueeze(1)).square().sum(dim=2)
            center_energy = train_center.square().sum(dim=1)
            centered_relative = (
                centered_energy.mean(dim=1).sqrt()
                / center_energy.sqrt().clamp_min(torch.finfo(vectors.dtype).tiny)
            )
            isolated_delta = (vectors - isolated_vectors.unsqueeze(1)).square().sum(dim=2)
            isolated_norm = isolated_vectors.square().sum(dim=1).sqrt().clamp_min(
                torch.finfo(vectors.dtype).tiny
            )
            mean_isolated_relative = isolated_delta.sqrt().mean(dim=1) / isolated_norm

            for head, metric in enumerate(metrics):
                cells.append(
                    {
                        "layer": layer,
                        "head": head,
                        "kind": kind,
                        "centered_relative_l2": float(centered_relative[head].item()),
                        "mean_isolated_delta_relative_l2": float(
                            mean_isolated_relative[head].item()
                        ),
                        "paired_context_matrix_delta": paired_matrix_metrics[head],
                        "isolated_matrix_delta": isolated_matrix_metrics[head],
                        **metric,
                    }
                )
            del matrices, isolated_matrices, vectors, isolated_vectors, train, test
    return cells


if __name__ == "__main__":
    main()
