"""Stage only selected calibration K/V arrays from NAS to compute-node local disk."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_selected(source_root: Path, target_root: Path, selection: Path,
                  staging_root: Path, begin: int, end: int, kinds: tuple[str, ...]) -> None:
    with selection.open() as f:
        selected = json.load(f)["selected_layers"]
    with (source_root/"metadata.json").open() as f:
        source_meta = json.load(f)
    with (target_root/"metadata.json").open() as f:
        target_meta = json.load(f)
    if not 0 <= begin < end <= len(selected) == int(target_meta["num_layers"]):
        raise ValueError("invalid target layer range")
    dest_source = staging_root/"source"
    dest_target = staging_root/"target"
    dest_source.mkdir(parents=True, exist_ok=False)
    dest_target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_root/"metadata.json", dest_source/"metadata.json")
    shutil.copy2(target_root/"metadata.json", dest_target/"metadata.json")
    shutil.copy2(selection, staging_root/"selection.json")
    required_source = sorted(set(i for layer in selected[begin:end] for i in layer))
    for kind in kinds:
        (dest_source/kind).mkdir()
        (dest_target/kind).mkdir()
        for idx in required_source:
            shutil.copy2(source_root/kind/f"layer_{idx:03d}.npy",
                         dest_source/kind/f"layer_{idx:03d}.npy")
        for idx in range(begin, end):
            shutil.copy2(target_root/kind/f"layer_{idx:03d}.npy",
                         dest_target/kind/f"layer_{idx:03d}.npy")
    print(json.dumps({"source_layers": required_source, "target_layers": list(range(begin, end)),
                      "kinds": kinds, "staging": str(staging_root)}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer-start", type=int, required=True)
    p.add_argument("--layer-end", type=int, required=True)
    p.add_argument("--kinds", default="k,v")
    args = p.parse_args()
    copy_selected(args.source, args.target, args.selection, args.output,
                  args.layer_start, args.layer_end, tuple(args.kinds.split(",")))


if __name__ == "__main__":
    main()
