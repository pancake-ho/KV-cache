from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from ..diverse_handoff import DEFAULT_SPLIT_SIZES, generate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the diverse Agent-KV V2 manifest")
    parser.add_argument("--bfcl-root", required=True)
    parser.add_argument("--fineweb", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    for split, default in DEFAULT_SPLIT_SIZES.items():
        parser.add_argument("--" + split.replace("_", "-") + "-rows", type=int, default=default)
    args = parser.parse_args()
    sizes = {
        split: getattr(args, split + "_rows") for split in DEFAULT_SPLIT_SIZES
    }
    if any(count < 1 for count in sizes.values()):
        parser.error("all split sizes must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    audit = generate_manifest(
        tokenizer,
        bfcl_root=Path(args.bfcl_root),
        fineweb_path=Path(args.fineweb),
        output_path=Path(args.output),
        sizes=sizes,
        seed=args.seed,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
