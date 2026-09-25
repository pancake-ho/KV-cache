from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..artifact import LinearKVMapper
from ..coqa import (
    COQA_PAPER_TURNS,
    answer_from_generation,
    context_at_turn,
    coqa_f1,
    gold_answers_at_turn,
    questions,
    select_conversations,
)
from ..data import iter_records
from ..generation import generate_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen3 source-to-target KV handoff at CoQA turn depths"
    )
    parser.add_argument("--dataset", required=True, help="EleutherAI CoQA validation Parquet")
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--output", required=True, help="Per-turn JSONL; safely resumable")
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--conversations-per-domain", type=int, default=20)
    parser.add_argument("--selection-strategy", choices=("first", "random"), default="first")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--turns", type=_parse_turns, default=COQA_PAPER_TURNS)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.conversations_per_domain < 1:
        parser.error("conversations-per-domain must be positive")
    if args.generation_batch_size < 1:
        parser.error("generation-batch-size must be positive")
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0,num-shards)")

    documents = select_conversations(
        enumerate(iter_records(args.dataset)),
        per_domain=args.conversations_per_domain,
        strategy=args.selection_strategy,
        seed=args.selection_seed,
    )
    for index, document in documents:
        if any(turn > len(questions(document)) for turn in args.turns):
            parser.error(f"conversation {index} does not contain every requested turn")
    documents = [
        item for ordinal, item in enumerate(documents)
        if ordinal % args.num_shards == args.shard_index
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_handoffs(output_path)
    pending = [
        (turn, index, document)
        for turn in args.turns
        for index, document in documents
        if (index, turn) not in completed
    ]
    if not pending:
        print({"processed_this_run": 0, "message": "all selected handoffs already complete"})
        return

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

    processed = 0
    with output_path.open("a") as output:
        for start in range(0, len(pending), args.generation_batch_size):
            batch = pending[start : start + args.generation_batch_size]
            generations = generate_pairs(
                source_model=source,
                target_model=target,
                mapper=mapper,
                tokenizer=tokenizer,
                contexts=[context_at_turn(document, turn) for turn, _, document in batch],
                max_new_tokens=args.max_new_tokens,
                stop=("\nQ:",),
            )
            for (turn, index, document), generation in zip(batch, generations, strict=True):
                golds = gold_answers_at_turn(document, turn)
                standalone_answer = answer_from_generation(generation.standalone)
                transfer_answer = answer_from_generation(generation.transfer)
                row = {
                    "index": index,
                    "conversation_id": document.get("id"),
                    "domain": document["source"],
                    "turn": turn,
                    "conversation_turns": len(questions(document)),
                    "gold_answers": golds,
                    "standalone_generation": generation.standalone,
                    "transfer_generation": generation.transfer,
                    "standalone_answer": standalone_answer,
                    "transfer_answer": transfer_answer,
                    "standalone_f1": coqa_f1(standalone_answer, golds),
                    "transfer_f1": coqa_f1(transfer_answer, golds),
                    "standalone_tokens": generation.standalone_tokens,
                    "transfer_tokens": generation.transfer_tokens,
                    "standalone_hit_token_cap": generation.standalone_tokens >= args.max_new_tokens,
                    "transfer_hit_token_cap": generation.transfer_tokens >= args.max_new_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "selection_strategy": args.selection_strategy,
                    "selection_seed": args.selection_seed,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                processed += 1
            print(
                {
                    "turn": batch[-1][0],
                    "processed_this_run": processed,
                    "remaining": len(pending) - processed,
                },
                flush=True,
            )
    print({"processed_this_run": processed, "output": str(output_path)})


def completed_handoffs(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {
            (int(row["index"]), int(row["turn"]))
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }


def _parse_turns(value: str) -> tuple[int, ...]:
    try:
        turns = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("turns must be comma-separated integers") from error
    if not turns or any(turn < 1 for turn in turns) or len(set(turns)) != len(turns):
        raise argparse.ArgumentTypeError("turns must be unique positive integers")
    return turns


if __name__ == "__main__":
    main()
