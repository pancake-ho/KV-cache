"""User-facing train / check / evaluation commands (invoked only via run/)."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NVIDIA 2026 Neural Cross-Model KV Mapper reproduction")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--source-dir", required=True)
    fit.add_argument("--target-dir", required=True)
    fit.add_argument("--selection", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--layer-start", type=int, default=0)
    fit.add_argument("--layer-end", type=int)
    fit.add_argument("--head-start", type=int, default=0)
    fit.add_argument("--head-end", type=int)
    fit.add_argument("--kinds", default="k,v")
    fit.add_argument("--hidden-dim", type=int, default=1024)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--batch-size", type=int, default=4096)
    fit.add_argument("--lr", type=float, default=0.001)
    fit.add_argument("--seed", type=int, default=2026)
    fit.add_argument("--max-observations", type=int)
    fit.add_argument("--train-dtype", default="bfloat16", choices=("bfloat16", "float32"))
    fit.add_argument("--storage-dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    check = sub.add_parser("check")
    check.add_argument("--mapper-dir", required=True)
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--mapper-dir", required=True)
    evaluate.add_argument("--input-jsonl", required=True)
    evaluate.add_argument("--output-json", required=True)
    evaluate.add_argument("--prefix-length", type=int, default=128)
    evaluate.add_argument("--max-examples", type=int, default=4)
    evaluate.add_argument("--token-chunk", type=int, default=128)
    evaluate.add_argument("--source-model")
    evaluate.add_argument("--target-model")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        from .train import build_spec, fit_all
        spec = build_spec(args.source_dir, args.target_dir, args.selection,
                          hidden_dim=args.hidden_dim, epochs=args.epochs,
                          batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                          max_observations=args.max_observations,
                          train_dtype=args.train_dtype, storage_dtype=args.storage_dtype)
        print(json.dumps(asdict(spec), indent=2), flush=True)
        fit_all(source_dir=args.source_dir, target_dir=args.target_dir,
                spec=spec, output_dir=args.output, target_layer_start=args.layer_start,
                target_layer_end=args.layer_end, head_start=args.head_start,
                head_end=args.head_end, kinds=tuple(args.kinds.split(",")))
    elif args.command == "check":
        from .transfer import NeuralKVMapper
        mapper = NeuralKVMapper(args.mapper_dir)
        mapper.assert_complete()
        print(json.dumps({"complete": True, "source_model": mapper.config.source_model,
                          "target_model": mapper.config.target_model,
                          "target_layers": mapper.config.target_layers}, indent=2))
    elif args.command == "eval":
        from .evaluate import evaluate
        evaluate(mapper_dir=args.mapper_dir, input_jsonl=args.input_jsonl,
                 output_json=args.output_json, prefix_length=args.prefix_length,
                 max_examples=args.max_examples, token_chunk=args.token_chunk,
                 source_model=args.source_model, target_model=args.target_model)


if __name__ == "__main__":
    main()
