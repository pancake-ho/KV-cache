from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..artifact import LinearKVMapper
from ..evaluation import compare_continuation


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare standalone and mapped-cache continuation likelihood")
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--prefix-length", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    encoded = tokenizer(args.text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=args.max_length)
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
    result = compare_continuation(
        source_model=source,
        target_model=target,
        mapper=mapper,
        input_ids=encoded.input_ids,
        prefix_length=args.prefix_length,
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
