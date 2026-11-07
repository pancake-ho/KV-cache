from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..artifact import LinearKVMapper
from ..benchmarks import (
    GSM8K_ANSWER_STOP_PATTERN,
    gsm8k_exact_match,
    gsm8k_extract_flexible,
    gsm8k_extract_strict,
    gsm8k_gold,
    gsm8k_prompt,
    iter_arc_challenge,
    iter_mmlu,
    iter_winogrande,
    load_wikitext_tokens,
)
from ..data import iter_records
from ..generation import GenerationPair, generate_pairs
from ..multiple_choice import score_choices, score_token_continuation


TASKS = ("arc_challenge", "winogrande", "mmlu", "gsm8k", "wikitext2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-protocol standalone/transfer benchmark evaluation")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--dataset", required=True, help="Downloaded dataset repository directory")
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--output", required=True, help="Per-document/chunk JSONL; safely resumable")
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument(
        "--retry-maxed-from",
        nargs="+",
        help="For GSM8K, evaluate only indices that hit the token cap in these JSONL files",
    )
    parser.add_argument("--retry-maxed-threshold", type=int, default=1024)
    parser.add_argument("--chunk-length", type=int, default=2048)
    parser.add_argument("--prefix-length", type=int, default=1024)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0,num-shards)")
    if args.generation_batch_size < 1:
        parser.error("generation-batch-size must be positive")
    if args.retry_maxed_from and args.task != "gsm8k":
        parser.error("retry-maxed-from is only valid for GSM8K")
    if args.task == "wikitext2" and not 1 < args.prefix_length < args.chunk_length:
        parser.error("WikiText prefix-length must be between 2 and chunk-length")

    tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    source = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        dtype=torch.bfloat16,
        device_map={"": args.source_device},
        attn_implementation="sdpa",
    ).eval()
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=torch.bfloat16,
        device_map={"": args.target_device},
        attn_implementation="sdpa",
    ).eval()
    mapper = LinearKVMapper(args.mapper)
    mapper.preload_for_model(target)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_indices(output_path)
    if args.task == "gsm8k":
        retry_indices = maxed_indices(
            args.retry_maxed_from, args.retry_maxed_threshold
        ) if args.retry_maxed_from else None
        processed = _run_gsm8k(
            args=args,
            source=source,
            target=target,
            mapper=mapper,
            tokenizer=tokenizer,
            output_path=output_path,
            completed=completed,
            retry_indices=retry_indices,
        )
        print({"task": args.task, "shard": args.shard_index, "processed_this_run": processed})
        return

    processed = 0
    with output_path.open("a") as output:
        for index, item in enumerate(_items(args.task, args.dataset, tokenizer, args.chunk_length)):
            if index % args.num_shards != args.shard_index or index in completed:
                continue
            if args.task in {"arc_challenge", "winogrande", "mmlu"}:
                row = _score_multiple_choice(item, source, target, mapper, tokenizer)
            else:
                row = _score_wikitext(
                    index, item, source, target, mapper, args.prefix_length
                )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            processed += 1
            if processed % 10 == 0:
                print({"task": args.task, "shard": args.shard_index, "processed_this_run": processed}, flush=True)
            if args.max_documents and processed >= args.max_documents:
                break
    print({"task": args.task, "shard": args.shard_index, "processed_this_run": processed})


def _items(task: str, dataset: str, tokenizer, chunk_length: int):
    if task == "arc_challenge":
        yield from iter_arc_challenge(dataset)
    elif task == "winogrande":
        yield from iter_winogrande(dataset)
    elif task == "mmlu":
        yield from iter_mmlu(dataset)
    elif task == "gsm8k":
        path = Path(dataset) / "main" / "test-00000-of-00001.parquet"
        yield from iter_records(path)
    elif task == "wikitext2":
        tokens = load_wikitext_tokens(dataset, tokenizer)
        for start in range(0, len(tokens) - chunk_length + 1, chunk_length):
            yield tokens[start : start + chunk_length]
    else:
        raise ValueError(task)


def _score_multiple_choice(item, source, target, mapper, tokenizer) -> dict:
    if len(item.contexts) == 1:
        scores = score_choices(
            source_model=source,
            target_model=target,
            mapper=mapper,
            tokenizer=tokenizer,
            context=item.contexts[0],
            choices=item.choices,
        )
        standalone_scores = scores.standalone
        transfer_scores = scores.transfer
        if item.normalization == "character":
            standalone_decision = scores.standalone_normalized
            transfer_decision = scores.transfer_normalized
        else:
            standalone_decision = standalone_scores
            transfer_decision = transfer_scores
        token_counts = scores.token_counts
        character_counts = scores.character_counts
    else:
        standalone_scores = []
        transfer_scores = []
        token_counts = []
        character_counts = []
        for context, choice in zip(item.contexts, item.choices, strict=True):
            scores = score_choices(
                source_model=source,
                target_model=target,
                mapper=mapper,
                tokenizer=tokenizer,
                context=context,
                choices=[choice],
            )
            standalone_scores.append(scores.standalone[0])
            transfer_scores.append(scores.transfer[0])
            token_counts.append(scores.token_counts[0])
            character_counts.append(scores.character_counts[0])
        standalone_decision = standalone_scores
        transfer_decision = transfer_scores

    standalone_prediction = max(range(len(standalone_decision)), key=standalone_decision.__getitem__)
    transfer_prediction = max(range(len(transfer_decision)), key=transfer_decision.__getitem__)
    return {
        "index": item.index,
        "subject": item.subject,
        "label": item.label,
        "standalone_prediction": standalone_prediction,
        "transfer_prediction": transfer_prediction,
        "standalone": standalone_scores,
        "transfer": transfer_scores,
        "token_counts": token_counts,
        "character_counts": character_counts,
        "normalization": item.normalization,
    }


def _run_gsm8k(
    *, args, source, target, mapper, tokenizer, output_path, completed, retry_indices
) -> int:
    pending = []
    for index, doc in enumerate(_items("gsm8k", args.dataset, tokenizer, args.chunk_length)):
        if index % args.num_shards != args.shard_index or index in completed:
            continue
        if retry_indices is not None and index not in retry_indices:
            continue
        pending.append((index, doc))
        if args.max_documents and len(pending) >= args.max_documents:
            break

    processed = 0
    with output_path.open("a") as output:
        for start in range(0, len(pending), args.generation_batch_size):
            batch = pending[start : start + args.generation_batch_size]
            generations = generate_pairs(
                source_model=source,
                target_model=target,
                mapper=mapper,
                tokenizer=tokenizer,
                contexts=[gsm8k_prompt(doc["question"]) for _, doc in batch],
                max_new_tokens=args.max_new_tokens,
                stop=("Q:", "<|im_end|>"),
                # The paper omits its generation cap. Qwen3 can reason for more
                # than 256 tokens, so stop each row once a conventional final
                # answer phrase appears and retain both harness extractors below.
                stop_regex=GSM8K_ANSWER_STOP_PATTERN,
            )
            for (index, doc), generation in zip(batch, generations, strict=True):
                output.write(
                    json.dumps(
                        _gsm8k_row(index, doc, generation, args.max_new_tokens),
                        ensure_ascii=False,
                    ) + "\n"
                )
                output.flush()
                processed += 1
            print(
                {"task": "gsm8k", "shard": args.shard_index, "processed_this_run": processed},
                flush=True,
            )
    return processed


def _gsm8k_row(
    index: int, doc: dict, generation: GenerationPair, max_new_tokens: int
) -> dict:
    gold = gsm8k_gold(doc["answer"])
    standalone_strict = gsm8k_extract_strict(generation.standalone)
    transfer_strict = gsm8k_extract_strict(generation.transfer)
    standalone_flexible = gsm8k_extract_flexible(generation.standalone)
    transfer_flexible = gsm8k_extract_flexible(generation.transfer)
    return {
        "index": index,
        "generation_max_new_tokens": max_new_tokens,
        "gold": gold,
        "standalone_generation": generation.standalone,
        "transfer_generation": generation.transfer,
        "standalone_tokens": generation.standalone_tokens,
        "transfer_tokens": generation.transfer_tokens,
        "standalone_stopped_on_answer_match": generation.standalone_stopped_on_match,
        "transfer_stopped_on_answer_match": generation.transfer_stopped_on_match,
        "standalone_strict_answer": standalone_strict,
        "transfer_strict_answer": transfer_strict,
        "standalone_flexible_answer": standalone_flexible,
        "transfer_flexible_answer": transfer_flexible,
        "standalone_strict_correct": gsm8k_exact_match(standalone_strict, gold),
        "transfer_strict_correct": gsm8k_exact_match(transfer_strict, gold),
        "standalone_flexible_correct": gsm8k_exact_match(standalone_flexible, gold),
        "transfer_flexible_correct": gsm8k_exact_match(transfer_flexible, gold),
    }


def _score_wikitext(index, tokens, source, target, mapper, prefix_length: int) -> dict:
    scores = score_token_continuation(
        source_model=source,
        target_model=target,
        mapper=mapper,
        prefix_ids=tokens[:prefix_length],
        continuation_ids=tokens[prefix_length:],
    )
    return {
        "index": index,
        "tokens": scores.tokens,
        "standalone_log_likelihood": scores.standalone_log_likelihood,
        "transfer_log_likelihood": scores.transfer_log_likelihood,
        "standalone_nll": -scores.standalone_log_likelihood / scores.tokens,
        "transfer_nll": -scores.transfer_log_likelihood / scores.tokens,
        "standalone_ppl": math.exp(-scores.standalone_log_likelihood / scores.tokens),
        "transfer_ppl": math.exp(-scores.transfer_log_likelihood / scores.tokens),
    }


def completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {int(json.loads(line)["index"]) for line in handle if line.strip()}


def maxed_indices(paths: list[str], threshold: int) -> set[int]:
    selected = set()
    for name in paths:
        with Path(name).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    int(row["standalone_tokens"]) >= threshold
                    or int(row["transfer_tokens"]) >= threshold
                ):
                    selected.add(int(row["index"]))
    if not selected:
        raise ValueError("retry-maxed-from did not select any indices")
    return selected


if __name__ == "__main__":
    main()
