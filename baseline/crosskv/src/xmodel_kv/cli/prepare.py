from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from ..data import iter_records, packed_token_sequences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare fixed calibration token sequences")
    parser.add_argument("--dataset", required=True, help="Parquet/JSON(L) file, glob, or directory")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Hugging Face tokenizer ID or local source/target-family tokenizer path",
    )
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--text-field", default="text", help="Dotted text field; conversations are auto-detected")
    parser.add_argument("--num-sequences", type=int, default=500)
    parser.add_argument("--sequence-length", type=int, default=1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens = packed_token_sequences(
        iter_records(args.dataset),
        tokenizer,
        count=args.num_sequences,
        sequence_length=args.sequence_length,
        text_field=args.text_field,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, tokens)
    metadata = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "num_sequences": args.num_sequences,
        "sequence_length": args.sequence_length,
        "text_field": args.text_field,
    }
    with output.with_suffix(".json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"saved tokens {tokens.shape} to {output}")


if __name__ == "__main__":
    main()
