from __future__ import annotations

import argparse

from ..fitting import fit_mapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the production per-head cross-layer ridge mapper")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--weight-dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--target-layer-start", type=int, default=0, help="Inclusive target layer index")
    parser.add_argument("--target-layer-end", type=int, help="Exclusive target layer index")
    args = parser.parse_args()
    config = fit_mapper(
        source_dir=args.source,
        target_dir=args.target,
        selection_path=args.selection,
        output_dir=args.output,
        ridge=args.ridge,
        device=args.device,
        max_observations=args.max_observations,
        weight_dtype=args.weight_dtype,
        resume=not args.no_resume,
        target_layer_start=args.target_layer_start,
        target_layer_end=args.target_layer_end,
    )
    print(config)


if __name__ == "__main__":
    main()
