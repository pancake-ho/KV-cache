from __future__ import annotations

import argparse

import torch

from ..extraction import extract_activations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract stride-sampled, RoPE-stripped calibration KV")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", required=True, choices=("source", "target"))
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--attn-implementation", default="sdpa", choices=("eager", "sdpa", "flash_attention_2"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = extract_activations(
        model_path=args.model,
        tokens_path=args.tokens,
        output_dir=args.output,
        role=args.role,
        stride=args.stride,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        max_sequences=args.max_sequences,
        attn_implementation=args.attn_implementation,
    )
    print(metadata)


if __name__ == "__main__":
    main()
