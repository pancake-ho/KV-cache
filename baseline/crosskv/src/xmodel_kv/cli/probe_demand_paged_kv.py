from __future__ import annotations

import argparse
import gc
import json
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


SYSTEM_PREFIX = (
    "You are a careful question-answering assistant. Use the supplied passages "
    "to answer the question.\n\nPassages:\n"
)
QUESTION_PREFIX = "\n\nQuestion: "
ANSWER_SUFFIX = "\nReturn only the exact short answer without explanation.\nAnswer:"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-KV decoding with query-aware demand paging over the "
            "document portion of a RAG prompt."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--budget-ratio", type=float, default=0.05)
    parser.add_argument("--min-pages", type=int, default=1)
    parser.add_argument("--cache-capacity-ratio", type=float, default=2.0)
    parser.add_argument("--no-pin-memory", action="store_true")
    args = parser.parse_args()
    if min(args.offset, args.count - 1, args.max_new_tokens - 1) < 0:
        parser.error("offset must be non-negative and counts must be positive")
    if args.max_prompt_tokens < 4:
        parser.error("max-prompt-tokens must be at least four")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    paging_config = DemandPagedKVConfig(
        page_size=args.page_size,
        budget_ratio=args.budget_ratio,
        min_pages=args.min_pages,
        cache_capacity_ratio=args.cache_capacity_ratio,
        pin_cpu_memory=not args.no_pin_memory,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "paging": vars(paging_config)}
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    cases = load_cases(args.dataset, offset=args.offset, count=args.count)
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
    result_path = output_dir / "results.jsonl"
    with result_path.open("w") as result_file:
        for position, (index, case) in enumerate(cases, 1):
            prompt = build_prompt_ids(
                tokenizer,
                context=str(case["context"]),
                question=str(case["input"]),
                max_tokens=args.max_prompt_tokens,
                device=device,
            )
            baseline_ids, baseline_ms = full_generate(
                model,
                prompt["input_ids"],
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            demand_ids, demand_metrics = demand_generate(
                model,
                prompt["input_ids"],
                cold_start=prompt["cold_start"],
                cold_end=prompt["cold_end"],
                config=paging_config,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            baseline_text = clean_short_answer(
                tokenizer.decode(baseline_ids, skip_special_tokens=True)
            )
            demand_text = clean_short_answer(
                tokenizer.decode(demand_ids, skip_special_tokens=True)
            )
            answers = [str(answer) for answer in case.get("answers", [])]
            row = {
                "index": index,
                "question": case["input"],
                "answers": answers,
                "prompt_tokens": int(prompt["input_ids"].shape[1]),
                "cold_document_tokens": prompt["cold_end"] - prompt["cold_start"],
                "baseline": baseline_text,
                "demand_paged": demand_text,
                "baseline_token_ids": baseline_ids,
                "demand_paged_token_ids": demand_ids,
                "token_agreement": token_agreement(baseline_ids, demand_ids),
                "baseline_ms": baseline_ms,
                **demand_metrics,
            }
            if answers:
                row.update(
                    {
                        "baseline_em": answer_em(baseline_text, answers),
                        "baseline_f1": answer_f1(baseline_text, answers),
                        "demand_paged_em": answer_em(demand_text, answers),
                        "demand_paged_f1": answer_f1(demand_text, answers),
                    }
                )
            result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            result_file.flush()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "position": position,
                        "index": index,
                        "baseline": baseline_text,
                        "demand_paged": demand_text,
                        "token_agreement": row["token_agreement"],
                        "paging": row["paging"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = summarize(rows, config=config)
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
            if len(rows) >= count:
                break
            row = json.loads(line)
            if not all(name in row for name in ("input", "context")):
                raise ValueError("dataset rows must contain input and context")
            rows.append((index, row))
    if len(rows) != count:
        raise ValueError(f"requested {count} rows at offset {offset}, found {len(rows)}")
    return rows


def build_prompt_ids(
    tokenizer,
    *,
    context: str,
    question: str,
    max_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    prefix = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=True)
    document = tokenizer.encode(context, add_special_tokens=False)
    suffix = tokenizer.encode(
        QUESTION_PREFIX + question + ANSWER_SUFFIX,
        add_special_tokens=False,
    )
    available = max_tokens - len(prefix) - len(suffix)
    if available < 1:
        raise ValueError("system and question exceed max-prompt-tokens")
    document = document[:available]
    token_ids = prefix + document + suffix
    if len(token_ids) < 2:
        raise ValueError("prompt must contain at least two tokens")
    return {
        "input_ids": torch.tensor([token_ids], dtype=torch.long, device=device),
        "cold_start": len(prefix),
        "cold_end": len(prefix) + len(document),
    }


@torch.inference_mode()
def full_generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[list[int], float]:
    synchronize(input_ids.device)
    start = time.perf_counter()
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=eos_token_id,
    )
    synchronize(input_ids.device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return generated[0, input_ids.shape[1] :].tolist(), elapsed_ms


@torch.inference_mode()
def demand_generate(
    model,
    input_ids: torch.Tensor,
    *,
    cold_start: int,
    cold_end: int,
    config: DemandPagedKVConfig,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[list[int], dict[str, Any]]:
    prefix_ids = input_ids[:, :-1]
    synchronize(input_ids.device)
    prefill_start = time.perf_counter()
    prefix = model(
        input_ids=prefix_ids,
        attention_mask=torch.ones_like(prefix_ids),
        use_cache=True,
        return_dict=True,
    )
    synchronize(input_ids.device)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000

    build_start = time.perf_counter()
    controller = DemandPagedKVController.from_cache(
        prefix.past_key_values,
        cold_start=cold_start,
        cold_end=cold_end,
        config=config,
        device=input_ids.device,
    )
    del prefix
    synchronize(input_ids.device)
    build_ms = (time.perf_counter() - build_start) * 1000
    controller.attach(model)

    generated: list[int] = []
    current = input_ids[:, -1:]
    prefix_tokens = prefix_ids.shape[1]
    synchronize(input_ids.device)
    decode_start = time.perf_counter()
    try:
        for step in range(max_new_tokens):
            position = prefix_tokens + step
            output = model(
                input_ids=current,
                attention_mask=torch.ones(
                    (1, position + 1), dtype=torch.long, device=input_ids.device
                ),
                position_ids=torch.tensor(
                    [[position]], dtype=torch.long, device=input_ids.device
                ),
                cache_position=torch.tensor(
                    [position], dtype=torch.long, device=input_ids.device
                ),
                use_cache=False,
                return_dict=True,
            )
            next_id = int(output.logits[0, -1].argmax())
            generated.append(next_id)
            controller.next_step()
            if eos_token_id is not None and next_id == eos_token_id:
                break
            current = torch.tensor([[next_id]], dtype=torch.long, device=input_ids.device)
    finally:
        controller.detach(model)
    synchronize(input_ids.device)
    decode_ms = (time.perf_counter() - decode_start) * 1000
    return generated, {
        "demand_prefill_ms": prefill_ms,
        "demand_directory_build_ms": build_ms,
        "demand_decode_ms": decode_ms,
        "paging": controller.summary().as_dict(),
    }


def token_agreement(reference: list[int], candidate: list[int]) -> float:
    width = max(len(reference), len(candidate))
    if width == 0:
        return 1.0
    matches = sum(
        left == right for left, right in zip(reference, candidate, strict=False)
    )
    return matches / width


def summarize(rows: list[dict[str, Any]], *, config: dict[str, Any]) -> dict[str, Any]:
    def mean(name: str) -> float:
        return sum(float(row[name]) for row in rows) / len(rows)

    summary: dict[str, Any] = {
        "config": config,
        "cases": len(rows),
        "token_agreement": mean("token_agreement"),
        "baseline_ms": mean("baseline_ms"),
        "demand_prefill_ms": mean("demand_prefill_ms"),
        "demand_directory_build_ms": mean("demand_directory_build_ms"),
        "demand_decode_ms": mean("demand_decode_ms"),
        "selected_token_fraction": sum(
            row["paging"]["selected_token_fraction"] for row in rows
        )
        / len(rows),
        "page_hit_fraction": sum(
            row["paging"]["page_hit_fraction"] for row in rows
        )
        / len(rows),
        "loaded_bytes": sum(row["paging"]["loaded_bytes"] for row in rows),
        "full_scan_bytes": sum(row["paging"]["full_scan_bytes"] for row in rows),
    }
    if rows and "baseline_f1" in rows[0]:
        summary.update(
            {
                "baseline_em": mean("baseline_em"),
                "baseline_f1": mean("baseline_f1"),
                "demand_paged_em": mean("demand_paged_em"),
                "demand_paged_f1": mean("demand_paged_f1"),
            }
        )
    return summary


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
