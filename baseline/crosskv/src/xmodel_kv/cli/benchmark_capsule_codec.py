from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
from time import perf_counter

import numpy as np
import torch

from xmodel_kv.capsule_codec import pack_int4_capsule, unpack_int4_capsule


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real capsule wire packing.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if min(args.iterations, args.warmups) < 1:
        parser.error("iterations and warmups must be positive")

    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    cache = tuple(
        (
            torch.randn(
                1, 8, 4, 128, device=args.device, dtype=torch.bfloat16,
                generator=generator,
            ),
            torch.randn(
                1, 8, 4, 128, device=args.device, dtype=torch.bfloat16,
                generator=generator,
            ),
        )
        for _ in range(32)
    )
    active_layers = frozenset(range(16, 32))
    packet = pack_int4_capsule(
        cache, suffix_tokens=4, active_layers=active_layers
    )
    for _ in range(args.warmups):
        packet = pack_int4_capsule(
            cache, suffix_tokens=4, active_layers=active_layers
        )
        unpack_int4_capsule(packet, dtype=torch.bfloat16, device=args.device)
    _synchronize(args.device)

    pack_ms = []
    unpack_ms = []
    for _ in range(args.iterations):
        _synchronize(args.device)
        started = perf_counter()
        packet = pack_int4_capsule(
            cache, suffix_tokens=4, active_layers=active_layers
        )
        _synchronize(args.device)
        pack_ms.append((perf_counter() - started) * 1000)

        started = perf_counter()
        unpack_int4_capsule(packet, dtype=torch.bfloat16, device=args.device)
        _synchronize(args.device)
        unpack_ms.append((perf_counter() - started) * 1000)

    socket_ms = benchmark_socketpair(packet, iterations=args.iterations)
    result = {
        "device": args.device,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "packet_bytes": len(packet),
        "theoretical_unframed_bytes": 67_584,
        "framing_bytes": len(packet) - 67_584,
        "pack_gpu_to_bytes_ms": summarize_times(pack_ms),
        "unpack_bytes_to_gpu_ms": summarize_times(unpack_ms),
        "inprocess_socketpair_transfer_ms": summarize_times(socket_ms),
        "modeled_one_way_serialization_plus_link_ms": {
            "1_gbps": np.mean(pack_ms) + len(packet) * 8 / 1e6 + np.mean(unpack_ms),
            "10_gbps": np.mean(pack_ms) + len(packet) * 8 / 1e7 + np.mean(unpack_ms),
            "100_gbps": np.mean(pack_ms) + len(packet) * 8 / 1e8 + np.mean(unpack_ms),
        },
        "seed": args.seed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def benchmark_socketpair(packet: bytes, *, iterations: int) -> list[float]:
    sender, receiver = socket.socketpair()
    try:
        times = []
        for _ in range(iterations):
            started = perf_counter()
            sender.sendall(packet)
            remaining = len(packet)
            while remaining:
                chunk = receiver.recv(remaining)
                if not chunk:
                    raise RuntimeError("socketpair closed during capsule transfer")
                remaining -= len(chunk)
            times.append((perf_counter() - started) * 1000)
        return times
    finally:
        sender.close()
        receiver.close()


def summarize_times(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


if __name__ == "__main__":
    main()
