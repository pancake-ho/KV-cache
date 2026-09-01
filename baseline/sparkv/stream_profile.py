from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def _decode(
    tensors: dict[str, torch.Tensor],
    fmt: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fmt == "raw":
        return tensors["k"].to(dtype), tensors["v"].to(dtype)

    if fmt != "q5":
        raise ValueError(fmt)

    qk = tensors["qk"].float() - 16.0
    qv = tensors["qv"].float() - 16.0
    sk = tensors["sk"].float()
    sv = tensors["sv"].float()

    return (qk * sk).to(dtype), (qv * sv).to(dtype)


def _copy_to_device(
    pair: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> None:
    if device.type != "cuda":
        _ = pair[0].to(device)
        _ = pair[1].to(device)
        return

    key, value = pair
    try:
        key = key.pin_memory()
    except RuntimeError:
        pass
    try:
        value = value.pin_memory()
    except RuntimeError:
        pass

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        key_gpu = key.to(device, non_blocking=True)
        value_gpu = value.to(device, non_blocking=True)
    stream.synchronize()
    del key_gpu, value_gpu


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def profile_sample(
    sample_dir: Path,
    fmt: str,
    output: Path,
    device_name: str,
    dtype_name: str,
) -> dict[str, Any]:
    meta_path = sample_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if int(meta.get("unit_layout_version", 0)) < 1:
        raise RuntimeError(
            "fine-grained unit layout is missing; run "
            "`python -m baseline.sparkv.unit_cache` first"
        )

    unit_files = meta.get("unit_files")
    if not isinstance(unit_files, dict) or not unit_files:
        raise RuntimeError("metadata has no unit_files mapping")

    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dtype = _dtype_from_name(dtype_name)
    records: dict[str, dict[str, float | int | str]] = {}

    disk_values: list[float] = []
    decode_values: list[float] = []
    h2d_values: list[float] = []
    processing_values: list[float] = []

    # Small warm-up for allocation/copy path, not included in profile.
    first = next(iter(unit_files.values()))
    warm_path = sample_dir / str(first["path"])
    warm = load_file(str(warm_path), device="cpu")
    warm_pair = _decode(warm, fmt, dtype)
    _copy_to_device(warm_pair, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    del warm, warm_pair

    for key, info in unit_files.items():
        path = sample_dir / str(info["path"])

        begin = time.perf_counter()
        tensors = load_file(str(path), device="cpu")
        disk_ms = (time.perf_counter() - begin) * 1000.0

        begin = time.perf_counter()
        pair = _decode(tensors, fmt, dtype)
        decode_ms = (time.perf_counter() - begin) * 1000.0

        begin = time.perf_counter()
        _copy_to_device(pair, device)
        h2d_ms = (time.perf_counter() - begin) * 1000.0

        processing_ms = disk_ms + decode_ms + h2d_ms

        records[key] = {
            "path": str(info["path"]),
            "wire_bytes": int(info["wire_bytes"]),
            "storage_bytes": int(info["storage_bytes"]),
            "disk_ms": float(disk_ms),
            "decode_ms": float(decode_ms),
            "h2d_ms": float(h2d_ms),
            "processing_ms": float(processing_ms),
        }

        disk_values.append(disk_ms)
        decode_values.append(decode_ms)
        h2d_values.append(h2d_ms)
        processing_values.append(processing_ms)

        del tensors, pair

    result = {
        "format": fmt,
        "sample_dir": str(sample_dir),
        "device": str(device),
        "dtype": dtype_name,
        "unit_count": len(records),
        "disk_ms": _summary(disk_values),
        "decode_ms": _summary(decode_values),
        "h2d_ms": _summary(h2d_values),
        "processing_ms": _summary(processing_values),
        "units": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--format", choices=["raw", "q5"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    args = parser.parse_args()

    result = profile_sample(
        Path(args.sample_dir),
        args.format,
        Path(args.output),
        args.device,
        args.dtype,
    )
    print(
        json.dumps(
            {
                "saved": args.output,
                "format": result["format"],
                "unit_count": result["unit_count"],
                "device": result["device"],
                "processing_ms": result["processing_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
