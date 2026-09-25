from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_paired_transfer import paired_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired comparison of the same arm across two Hotpot transfer runs."
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--arm", default="capsule")
    parser.add_argument(
        "--reference-arm",
        default=None,
        help=(
            "Optional different arm in the reference file, for example "
            "shifted_capsule for a paired causal contrast. Defaults to --arm."
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument(
        "--min-index",
        type=int,
        default=None,
        help="Optional inclusive dataset-index floor for a preregistered holdout.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.min_index is not None and args.min_index < 0:
        parser.error("--min-index must be non-negative")
    candidate = load_rows(args.candidate, min_index=args.min_index)
    reference = load_rows(args.reference, min_index=args.min_index)
    reference_arm = args.reference_arm or args.arm
    if candidate.keys() != reference.keys():
        raise ValueError("candidate and reference contain different (id, protocol) keys")
    output = {
        "candidate": args.candidate,
        "reference": args.reference,
        "arm": args.arm,
        "reference_arm": reference_arm,
        "min_index": args.min_index,
        "by_protocol": {},
    }
    for protocol in ("state_readout", "question_conditioned"):
        keys = sorted(key for key in candidate if key[1] == protocol)
        if not keys:
            raise ValueError(f"missing protocol: {protocol}")
        output["by_protocol"][protocol] = paired_summary(
            [candidate[key][f"{args.arm}_em"] for key in keys],
            [reference[key][f"{reference_arm}_em"] for key in keys],
            [candidate[key][f"{args.arm}_f1"] for key in keys],
            [reference[key][f"{reference_arm}_f1"] for key in keys],
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


def load_rows(
    path: str | Path, *, min_index: int | None = None
) -> dict[tuple[str, str], dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.jsonl"
    selected = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if min_index is not None and int(row.get("index", -1)) < min_index:
                continue
            key = (row["id"], row["protocol"])
            if key in selected:
                raise ValueError(f"duplicate result key: {key}")
            selected[key] = row
    if not selected:
        raise ValueError("result file is empty")
    return selected


if __name__ == "__main__":
    main()
