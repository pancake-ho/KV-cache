from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


UNIT_LAYOUT_VERSION = 1


def unit_key(t: int, layer: int, head: int) -> str:
    return f"{t}:{layer}:{head}"


def unit_relative_path(t: int, layer: int, head: int) -> str:
    return f"units/t{t:03d}/l{layer:02d}_h{head:02d}.safetensors"


def _slice_raw(
    tensors: dict[str, torch.Tensor],
    layer: int,
    head: int,
) -> dict[str, torch.Tensor]:
    key_name = f"k_{layer:02d}"
    value_name = f"v_{layer:02d}"

    if key_name not in tensors or value_name not in tensors:
        raise KeyError(
            f"raw tensors are missing layer {layer}: "
            f"{key_name}, {value_name}"
        )

    key = tensors[key_name][:, head : head + 1].contiguous()
    value = tensors[value_name][:, head : head + 1].contiguous()

    if key.shape != value.shape:
        raise RuntimeError(
            f"K/V shape mismatch for layer={layer}, head={head}: "
            f"K={tuple(key.shape)}, V={tuple(value.shape)}"
        )

    return {"k": key, "v": value}


def _slice_q5(
    tensors: dict[str, torch.Tensor],
    layer: int,
    head: int,
) -> dict[str, torch.Tensor]:
    names = {
        "qk": f"qk_{layer:02d}",
        "qv": f"qv_{layer:02d}",
        "sk": f"sk_{layer:02d}",
        "sv": f"sv_{layer:02d}",
    }
    missing = [name for name in names.values() if name not in tensors]
    if missing:
        raise KeyError(
            f"q5 tensors are missing layer {layer}: {missing}"
        )

    qk = tensors[names["qk"]][:, head : head + 1].contiguous()
    qv = tensors[names["qv"]][:, head : head + 1].contiguous()
    sk = tensors[names["sk"]][:, head : head + 1].contiguous()
    sv = tensors[names["sv"]][:, head : head + 1].contiguous()

    if qk.shape != qv.shape:
        raise RuntimeError(
            f"q5 K/V symbol shape mismatch for layer={layer}, head={head}"
        )
    if sk.shape[:2] != (1, 1) or sv.shape[:2] != (1, 1):
        raise RuntimeError(
            f"q5 scale shape mismatch for layer={layer}, head={head}: "
            f"sk={tuple(sk.shape)}, sv={tuple(sv.shape)}"
        )

    return {"qk": qk, "qv": qv, "sk": sk, "sv": sv}


def materialize_sample(
    sample_dir: Path,
    fmt: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    if fmt not in {"raw", "q5"}:
        raise ValueError(f"unsupported format: {fmt}")

    meta_path = sample_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = {
        "num_chunks",
        "layers",
        "kv_heads",
        "chunk_size",
        "seq_len",
        "chunks",
    }
    missing = required - set(meta)
    if missing:
        raise ValueError(f"metadata missing fields: {sorted(missing)}")

    T = int(meta["num_chunks"])
    L = int(meta["layers"])
    H = int(meta["kv_heads"])

    if len(meta["chunks"]) != T:
        raise ValueError("metadata chunk count mismatch")

    unit_files: dict[str, dict[str, Any]] = {}
    total_storage_bytes = 0
    total_wire_bytes = 0

    for t in range(T):
        token_path = sample_dir / f"chunk_{t:03d}.safetensors"
        if not token_path.is_file():
            raise FileNotFoundError(f"token cache file not found: {token_path}")

        token_tensors = load_file(str(token_path), device="cpu")
        chunk_meta = meta["chunks"][t]
        lh_wire = chunk_meta.get("lh_wire_bytes")
        if not isinstance(lh_wire, dict):
            raise ValueError(f"lh_wire_bytes missing for t={t}")

        for layer in range(L):
            for head in range(H):
                key = unit_key(t, layer, head)
                relative = unit_relative_path(t, layer, head)
                path = sample_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)

                logical_wire_bytes = int(lh_wire[f"{layer}:{head}"])
                if logical_wire_bytes < 0:
                    raise ValueError(f"negative wire size for {key}")

                if fmt == "raw":
                    payload = _slice_raw(token_tensors, layer, head)
                else:
                    payload = _slice_q5(token_tensors, layer, head)

                if overwrite or not path.exists():
                    save_file(payload, str(path))

                storage_bytes = path.stat().st_size
                total_storage_bytes += storage_bytes
                total_wire_bytes += logical_wire_bytes

                unit_files[key] = {
                    "path": relative,
                    "wire_bytes": logical_wire_bytes,
                    "storage_bytes": storage_bytes,
                }

        del token_tensors

    meta["unit_layout_version"] = UNIT_LAYOUT_VERSION
    meta["unit_files"] = unit_files
    meta["unit_file_count"] = len(unit_files)
    meta["unit_storage_bytes"] = total_storage_bytes
    meta["unit_wire_bytes"] = total_wire_bytes

    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(meta_path)

    return {
        "sample_dir": str(sample_dir),
        "format": fmt,
        "unit_layout_version": UNIT_LAYOUT_VERSION,
        "unit_files": len(unit_files),
        "unit_storage_bytes": total_storage_bytes,
        "unit_wire_bytes": total_wire_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize true (token, layer, KV-head) cache files while "
            "preserving the existing token-level cache files for baselines."
        )
    )
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--format", choices=["raw", "q5"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = materialize_sample(
        Path(args.sample_dir),
        args.format,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
