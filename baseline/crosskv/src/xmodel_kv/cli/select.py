from __future__ import annotations

import argparse

from ..selection import save_selection, select_source_layers


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top-k source layers by single-source head-averaged R2")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--ridge", type=float, default=0.0, help="Paper's selection probe is OLS")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-observations", type=int)
    args = parser.parse_args()
    result = select_source_layers(
        args.source,
        args.target,
        top_k=args.top_k,
        ridge=args.ridge,
        device=args.device,
        max_observations=args.max_observations,
    )
    save_selection(result, args.output)
    print(f"saved selection to {args.output}")


if __name__ == "__main__":
    main()
