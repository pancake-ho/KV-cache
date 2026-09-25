from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from xmodel_kv.demand_paged_kv import (
    DemandPagedKVConfig,
    DemandPagedKVController,
    register_demand_paged_attention,
)
from xmodel_kv.hotpotqa import answer_em, answer_f1, clean_short_answer
from xmodel_kv.rag_chunk_kv import (
    ComposedChunkKV,
    compose_chunk_bundle,
    generate_from_chunk_controller,
    precompute_chunk_bundle,
)

SYSTEM_PREFIX = (
    "You are a careful question-answering assistant. Use the supplied passages "
    "to answer the question.\n\nPassages:\n"
)
QUESTION_PREFIX = "\n\nQuestion: "
ANSWER_SUFFIX = "\nReturn only the exact short answer without explanation.\nAnswer:"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare standard RAG prefill with order-specific and independently "
            "precomputed document-chunk KV."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--budget-ratio", type=float, default=0.05)
    parser.add_argument("--min-pages", type=int, default=1)
    parser.add_argument("--cache-capacity-ratio", type=float, default=2.0)
    parser.add_argument("--seed-budget-ratio", type=float, default=0.75)
    parser.add_argument("--no-pin-memory", action="store_true")
    args = parser.parse_args()
    if min(args.offset, args.count - 1, args.max_new_tokens - 1) < 0:
        parser.error("offset must be non-negative and counts must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    register_demand_paged_attention()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="demand_paged",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    cases = load_cases(args.dataset, offset=args.offset, count=args.count)
    result_path = output_dir / "results.jsonl"
    with result_path.open("w") as result_file:
        for position, (case_index, case) in enumerate(cases, 1):
            prompt = tokenize_case(
                tokenizer,
                context=str(case["context"]),
                question=str(case["input"]),
                max_prompt_tokens=args.max_prompt_tokens,
            )
            baseline_ids, baseline_ms = full_generate(
                model,
                prompt["full_ids"],
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

            sequential, sequential_producer_ms = timed_precompute(
                model,
                system_ids=prompt["system_ids"],
                chunks=prompt["chunks"],
                mode="sequential",
                pin_memory=not args.no_pin_memory,
                device=device,
            )
            sequential_composed = compose_chunk_bundle(model, sequential)
            sequential_ids, sequential_metrics = run_consumer(
                model,
                sequential_composed,
                suffix_ids=prompt["suffix_ids"],
                max_new_tokens=args.max_new_tokens,
                eos_token_id=model.generation_config.eos_token_id,
                page_size=args.page_size,
                budget_ratio=1.0,
                min_pages=args.min_pages,
                cache_capacity_ratio=args.cache_capacity_ratio,
                pin_memory=not args.no_pin_memory,
                device=device,
                mandatory_page_ids=(),
            )
            del sequential, sequential_composed
            release(device)

            independent, independent_producer_ms = timed_precompute(
                model,
                system_ids=prompt["system_ids"],
                chunks=prompt["chunks"],
                mode="independent",
                pin_memory=not args.no_pin_memory,
                device=device,
            )
            independent_composed = compose_chunk_bundle(model, independent)
            document_ids = [
                token for _, token_ids in prompt["chunks"] for token in token_ids
            ]
            seed_pages = independent_composed.select_seed_pages(
                document_token_ids=document_ids,
                query_token_ids=prompt["question_ids"],
                page_size=args.page_size,
                budget_ratio=args.budget_ratio,
                min_pages=args.min_pages,
                seed_budget_ratio=args.seed_budget_ratio,
            )
            independent_seeded_ids, independent_seeded_metrics = run_consumer(
                model,
                independent_composed,
                suffix_ids=prompt["suffix_ids"],
                max_new_tokens=args.max_new_tokens,
                eos_token_id=model.generation_config.eos_token_id,
                page_size=args.page_size,
                budget_ratio=args.budget_ratio,
                min_pages=args.min_pages,
                cache_capacity_ratio=args.cache_capacity_ratio,
                pin_memory=not args.no_pin_memory,
                device=device,
                mandatory_page_ids=seed_pages,
            )
            independent_full_ids, independent_full_metrics = run_consumer(
                model,
                independent_composed,
                suffix_ids=prompt["suffix_ids"],
                max_new_tokens=args.max_new_tokens,
                eos_token_id=model.generation_config.eos_token_id,
                page_size=args.page_size,
                budget_ratio=1.0,
                min_pages=args.min_pages,
                cache_capacity_ratio=args.cache_capacity_ratio,
                pin_memory=not args.no_pin_memory,
                device=device,
                mandatory_page_ids=(),
            )
            independent_sparse_ids, independent_sparse_metrics = run_consumer(
                model,
                independent_composed,
                suffix_ids=prompt["suffix_ids"],
                max_new_tokens=args.max_new_tokens,
                eos_token_id=model.generation_config.eos_token_id,
                page_size=args.page_size,
                budget_ratio=args.budget_ratio,
                min_pages=args.min_pages,
                cache_capacity_ratio=args.cache_capacity_ratio,
                pin_memory=not args.no_pin_memory,
                device=device,
                mandatory_page_ids=(),
            )

            answers = [str(answer) for answer in case.get("answers", [])]
            texts = {
                "baseline": decode(tokenizer, baseline_ids),
                "sequential_full": decode(tokenizer, sequential_ids),
                "independent_full": decode(tokenizer, independent_full_ids),
                "independent_sparse": decode(tokenizer, independent_sparse_ids),
                "independent_seeded": decode(tokenizer, independent_seeded_ids),
            }
            row = {
                "index": case_index,
                "question": case["input"],
                "answers": answers,
                "system_tokens": len(prompt["system_ids"]),
                "document_tokens": independent_composed.document_tokens,
                "suffix_tokens": len(prompt["suffix_ids"]),
                "chunks": len(prompt["chunks"]),
                "chunk_tokens": [
                    len(token_ids) for _, token_ids in prompt["chunks"]
                ],
                "consumer_document_prefill_tokens": 0,
                "independent_relocated_tokens": (
                    independent_composed.relocated_tokens
                ),
                "seed_pages": list(seed_pages),
                "baseline_ms": baseline_ms,
                "sequential_producer_ms": sequential_producer_ms,
                "independent_producer_ms": independent_producer_ms,
                "texts": texts,
                "token_ids": {
                    "baseline": baseline_ids,
                    "sequential_full": sequential_ids,
                    "independent_full": independent_full_ids,
                    "independent_sparse": independent_sparse_ids,
                    "independent_seeded": independent_seeded_ids,
                },
                "agreement": {
                    "sequential_full_vs_baseline": token_agreement(
                        baseline_ids, sequential_ids
                    ),
                    "independent_full_vs_baseline": token_agreement(
                        baseline_ids, independent_full_ids
                    ),
                    "independent_sparse_vs_independent_full": token_agreement(
                        independent_full_ids, independent_sparse_ids
                    ),
                    "independent_seeded_vs_independent_full": token_agreement(
                        independent_full_ids, independent_seeded_ids
                    ),
                },
                "sequential_full": sequential_metrics,
                "independent_full": independent_full_metrics,
                "independent_sparse": independent_sparse_metrics,
                "independent_seeded": independent_seeded_metrics,
            }
            if answers:
                row["quality"] = {
                    name: {
                        "em": answer_em(text, answers),
                        "f1": answer_f1(text, answers),
                    }
                    for name, text in texts.items()
                }
            result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            result_file.flush()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "position": position,
                        "index": case_index,
                        "texts": texts,
                        "agreement": row["agreement"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del independent, independent_composed
            release(device)

    summary = summarize(rows, config=vars(args))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def load_cases(path: str, *, offset: int, count: int) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with Path(path).open() as source:
        for index, line in enumerate(source):
            if index < offset:
                continue
            if len(rows) == count:
                break
            row = json.loads(line)
            if not all(key in row for key in ("input", "context")):
                raise ValueError("dataset row must contain input and context")
            rows.append((index, row))
    if len(rows) != count:
        raise ValueError(f"requested {count} rows but found {len(rows)}")
    return rows


def tokenize_case(
    tokenizer,
    *,
    context: str,
    question: str,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    system_ids = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=True)
    suffix_ids = tokenizer.encode(
        QUESTION_PREFIX + question + ANSWER_SUFFIX,
        add_special_tokens=False,
    )
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    available = max_prompt_tokens - len(system_ids) - len(suffix_ids)
    if available < 1:
        raise ValueError("system and question exceed max-prompt-tokens")

    chunks = []
    for index, text in enumerate(split_passages(context), 1):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        token_ids = token_ids[:available]
        if token_ids:
            title = text.splitlines()[0].strip() or f"chunk-{index}"
            chunks.append((title, token_ids))
            available -= len(token_ids)
        if available == 0:
            break
    if not chunks:
        raise ValueError("context produced no document chunks")
    full_ids = system_ids + [
        token for _, token_ids in chunks for token in token_ids
    ] + suffix_ids
    return {
        "system_ids": system_ids,
        "suffix_ids": suffix_ids,
        "question_ids": question_ids,
        "chunks": chunks,
        "full_ids": full_ids,
    }


def split_passages(context: str) -> list[str]:
    passages = [
        passage
        for passage in re.split(r"(?=Passage \d+:\n)", context)
        if passage
    ]
    return passages or [context]


@torch.inference_mode()
def full_generate(
    model,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    pad_token_id: int | None,
) -> tuple[list[int], float]:
    device = model.model.embed_tokens.weight.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    synchronize(device)
    start = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
    )
    synchronize(device)
    return output[0, input_ids.shape[1] :].tolist(), (time.perf_counter() - start) * 1000


def timed_precompute(model, *, device: torch.device, **kwargs):
    synchronize(device)
    start = time.perf_counter()
    bundle = precompute_chunk_bundle(model, **kwargs)
    synchronize(device)
    return bundle, (time.perf_counter() - start) * 1000


def run_consumer(
    model,
    composed: ComposedChunkKV,
    *,
    suffix_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
    page_size: int,
    budget_ratio: float,
    min_pages: int,
    cache_capacity_ratio: float,
    pin_memory: bool,
    device: torch.device,
    mandatory_page_ids: tuple[int, ...],
) -> tuple[list[int], dict[str, Any]]:
    config = DemandPagedKVConfig(
        page_size=page_size,
        budget_ratio=budget_ratio,
        min_pages=min_pages,
        cache_capacity_ratio=cache_capacity_ratio,
        pin_cpu_memory=pin_memory,
    )
    synchronize(device)
    build_start = time.perf_counter()
    controller = DemandPagedKVController.from_cache(
        composed.layers,
        cold_start=composed.cold_start,
        cold_end=composed.cold_end,
        config=config,
        device=device,
        cold_page_ranges=composed.page_ranges(page_size),
        mandatory_page_ids=mandatory_page_ids,
    )
    synchronize(device)
    build_ms = (time.perf_counter() - build_start) * 1000
    synchronize(device)
    consumer_start = time.perf_counter()
    token_ids = generate_from_chunk_controller(
        model,
        controller,
        suffix_ids=suffix_ids,
        virtual_prefix_tokens=composed.cold_end,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )
    synchronize(device)
    consumer_ms = (time.perf_counter() - consumer_start) * 1000
    metrics = {
        "directory_build_ms": build_ms,
        "consumer_ms": consumer_ms,
        "paging": controller.summary().as_dict(),
    }
    del controller
    release(device)
    return token_ids, metrics


def decode(tokenizer, token_ids: list[int]) -> str:
    return clean_short_answer(tokenizer.decode(token_ids, skip_special_tokens=True))


def token_agreement(reference: list[int], candidate: list[int]) -> float:
    width = max(len(reference), len(candidate))
    if width == 0:
        return 1.0
    matches = sum(
        left == right for left, right in zip(reference, candidate, strict=False)
    )
    return matches / width


def summarize(rows: list[dict[str, Any]], *, config: dict[str, Any]) -> dict[str, Any]:
    def mean(path: tuple[str, ...]) -> float:
        values = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / len(values)

    summary = {
        "config": config,
        "cases": len(rows),
        "consumer_document_prefill_tokens": 0,
        "agreement": {
            "sequential_full_vs_baseline": mean(
                ("agreement", "sequential_full_vs_baseline")
            ),
            "independent_full_vs_baseline": mean(
                ("agreement", "independent_full_vs_baseline")
            ),
            "independent_sparse_vs_independent_full": mean(
                ("agreement", "independent_sparse_vs_independent_full")
            ),
            "independent_seeded_vs_independent_full": mean(
                ("agreement", "independent_seeded_vs_independent_full")
            ),
        },
        "mean_document_tokens": mean(("document_tokens",)),
        "mean_chunks": mean(("chunks",)),
        "mean_relocated_tokens": mean(("independent_relocated_tokens",)),
        "independent_sparse_selected_token_fraction": mean(
            (
                "independent_sparse",
                "paging",
                "selected_token_fraction",
            )
        ),
        "independent_sparse_loaded_bytes": sum(
            row["independent_sparse"]["paging"]["loaded_bytes"] for row in rows
        ),
        "independent_seeded_loaded_bytes": sum(
            row["independent_seeded"]["paging"]["loaded_bytes"] for row in rows
        ),
    }
    if rows and "quality" in rows[0]:
        summary["quality"] = {
            mode: {
                metric: mean(("quality", mode, metric))
                for metric in ("em", "f1")
            }
            for mode in (
                "baseline",
                "sequential_full",
                "independent_full",
                "independent_sparse",
                "independent_seeded",
            )
        }
    return summary


def release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
