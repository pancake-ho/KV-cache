from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..artifact import LinearKVMapper
from ..data import iter_records
from ..multiple_choice import hellaswag_context_and_choices, score_choices


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate target standalone vs transferred KV on HellaSwag")
    parser.add_argument("--dataset", required=True, help="HellaSwag validation Parquet/JSONL file")
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--output", required=True, help="Per-document JSONL output; safely resumable")
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-documents", type=int)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")

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
    completed = _completed_indices(output_path)
    counters = {"documents": 0, "standalone_correct": 0, "transfer_correct": 0}
    with output_path.open("a") as output:
        for index, document in enumerate(iter_records(args.dataset)):
            if index % args.num_shards != args.shard_index or index in completed:
                continue
            context, choices, label = hellaswag_context_and_choices(document)
            scores = score_choices(
                source_model=source,
                target_model=target,
                mapper=mapper,
                tokenizer=tokenizer,
                context=context,
                choices=choices,
            )
            standalone_prediction = max(range(len(choices)), key=scores.standalone_normalized.__getitem__)
            transfer_prediction = max(range(len(choices)), key=scores.transfer_normalized.__getitem__)
            row = {
                "index": index,
                "label": label,
                "standalone_prediction": standalone_prediction,
                "transfer_prediction": transfer_prediction,
                "standalone": scores.standalone,
                "transfer": scores.transfer,
                "standalone_normalized": scores.standalone_normalized,
                "transfer_normalized": scores.transfer_normalized,
                "token_counts": scores.token_counts,
                "character_counts": scores.character_counts,
            }
            output.write(json.dumps(row) + "\n")
            output.flush()
            counters["documents"] += 1
            counters["standalone_correct"] += int(standalone_prediction == label)
            counters["transfer_correct"] += int(transfer_prediction == label)
            if counters["documents"] % 10 == 0:
                print(_summary(counters), flush=True)
            if args.max_documents and counters["documents"] >= args.max_documents:
                break
    print(_summary(counters))


def _completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {json.loads(line)["index"] for line in handle if line.strip()}


def _summary(counters: dict[str, int]) -> dict[str, float | int]:
    count = counters["documents"]
    if count == 0:
        return {**counters, "standalone_acc_norm": 0.0, "transfer_acc_norm": 0.0}
    standalone = counters["standalone_correct"] / count
    transfer = counters["transfer_correct"] / count
    return {
        **counters,
        "standalone_acc_norm": standalone,
        "transfer_acc_norm": transfer,
        "retention_percent": 100.0 * transfer / standalone if standalone else float("nan"),
    }


if __name__ == "__main__":
    main()
