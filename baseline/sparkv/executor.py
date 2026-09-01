from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import DynamicCache

from baseline.sparkv.scheduler import Chunk


@dataclass
class ScheduleExecutionStats:
    scheduled_compute_units: int = 0
    scheduled_stream_units: int = 0
    actual_compute_token_forwards: int = 0
    fetched_token_files: int = 0
    wire_bytes: int = 0
    disk_ms: float = 0.0
    wire_ms: float = 0.0
    decode_ms: float = 0.0
    predicted_makespan_ms: float = 0.0
    schedule_stages: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduled_compute_units": self.scheduled_compute_units,
            "scheduled_stream_units": self.scheduled_stream_units,
            "actual_compute_token_forwards": self.actual_compute_token_forwards,
            "fetched_token_files": self.fetched_token_files,
            "schedule_predicted_makespan_ms": self.predicted_makespan_ms,
            "schedule_stages": self.schedule_stages,
            "executor": "scheduler-guided-hf-compat-v1",
            "executor_granularity": "full-token-forward-plus-(t,l,h)-KV-merge",
        }


def _load_schedule(path: str | Path) -> dict[str, Any]:
    schedule = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "makespan_ms",
        "stages",
        "chunks",
        "compute_chunks",
        "stream_chunks",
    }
    missing = required - set(schedule)
    if missing:
        raise ValueError(f"schedule missing fields: {sorted(missing)}")
    return schedule


def _schedule_assignments(
    schedule: dict[str, Any],
) -> tuple[dict[Chunk, str], list[Chunk]]:
    assignments: dict[Chunk, str] = {}
    stream_order: list[Chunk] = []

    for stage in schedule["stages"]:
        for operation in ("compute", "stream"):
            for item in stage.get(operation, []):
                chunk = Chunk(
                    int(item["t"]),
                    int(item["layer"]),
                    int(item["head"]),
                )
                if chunk in assignments:
                    raise ValueError(f"duplicate scheduled chunk: {chunk}")
                assignments[chunk] = operation
                if operation == "stream":
                    stream_order.append(chunk)

    if len(assignments) != int(schedule["chunks"]):
        raise ValueError("schedule chunk count mismatch")

    compute_count = sum(v == "compute" for v in assignments.values())
    stream_count = len(assignments) - compute_count
    if compute_count != int(schedule["compute_chunks"]):
        raise ValueError("schedule compute count mismatch")
    if stream_count != int(schedule["stream_chunks"]):
        raise ValueError("schedule stream count mismatch")

    return assignments, stream_order


def _validate_against_meta(
    schedule: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[dict[Chunk, str], list[Chunk]]:
    assignments, stream_order = _schedule_assignments(schedule)

    T = int(meta["num_chunks"])
    L = int(meta["layers"])
    H = int(meta["kv_heads"])
    expected = {
        Chunk(t, layer, head)
        for t in range(T)
        for layer in range(L)
        for head in range(H)
    }

    if set(assignments) != expected:
        missing = sorted(expected - set(assignments))
        extra = sorted(set(assignments) - expected)
        raise ValueError(
            "schedule/cache geometry mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return assignments, stream_order


@dataclass
class _StreamStore:
    values: dict[Chunk, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    error: BaseException | None = None
    finished: bool = False

    def __post_init__(self) -> None:
        self.cond = threading.Condition()

    def publish(
        self,
        chunk: Chunk,
        pair: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        with self.cond:
            self.values[chunk] = pair
            self.cond.notify_all()

    def fail(self, exc: BaseException) -> None:
        with self.cond:
            self.error = exc
            self.finished = True
            self.cond.notify_all()

    def finish(self) -> None:
        with self.cond:
            self.finished = True
            self.cond.notify_all()

    def wait_for(
        self,
        chunks: set[Chunk],
    ) -> dict[Chunk, tuple[torch.Tensor, torch.Tensor]]:
        if not chunks:
            return {}

        with self.cond:
            while not chunks.issubset(self.values):
                if self.error is not None:
                    raise self.error
                if self.finished:
                    missing = sorted(chunks - set(self.values))
                    raise RuntimeError(
                        f"stream worker ended early: {missing[:5]}"
                    )
                self.cond.wait()

            return {chunk: self.values[chunk] for chunk in chunks}


class _TokenFileReader:
    def __init__(self, sample_dir: Path) -> None:
        self.sample_dir = sample_dir
        self.cache: dict[int, dict[str, torch.Tensor]] = {}
        self.disk_ms = 0.0
        self.loads = 0

    def get(self, t: int) -> dict[str, torch.Tensor]:
        if t not in self.cache:
            begin = time.perf_counter()
            self.cache[t] = load_file(
                str(self.sample_dir / f"chunk_{t:03d}.safetensors"),
                device="cpu",
            )
            self.disk_ms += (time.perf_counter() - begin) * 1000.0
            self.loads += 1
        return self.cache[t]


def _effective_bandwidth(
    mean_mbps: float,
    cv: float,
    rng: np.random.Generator,
) -> float:
    if mean_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive")
    if cv <= 0:
        return mean_mbps

    sigma2 = math.log1p(cv * cv)
    factor = rng.lognormal(
        mean=-0.5 * sigma2,
        sigma=math.sqrt(sigma2),
    )
    return max(1e-3, mean_mbps * factor)


def _decode_unit(
    cpu: dict[str, torch.Tensor],
    fmt: str,
    chunk: Chunk,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer, head = chunk.layer, chunk.head

    if fmt == "raw":
        return (
            cpu[f"k_{layer:02d}"][:, head : head + 1].to(target_dtype),
            cpu[f"v_{layer:02d}"][:, head : head + 1].to(target_dtype),
        )

    if fmt != "q5":
        raise ValueError(fmt)

    qk = cpu[f"qk_{layer:02d}"][:, head : head + 1].float() - 16.0
    qv = cpu[f"qv_{layer:02d}"][:, head : head + 1].float() - 16.0
    sk = cpu[f"sk_{layer:02d}"][:, head : head + 1].float()
    sv = cpu[f"sv_{layer:02d}"][:, head : head + 1].float()

    return (
        (qk * sk).to(target_dtype),
        (qv * sv).to(target_dtype),
    )


def _to_device(
    pair: tuple[torch.Tensor, torch.Tensor],
    runtime: Any,
    copy_stream: torch.cuda.Stream | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    key, value = pair
    if not runtime.is_cuda:
        return key.to(runtime.device), value.to(runtime.device)

    assert copy_stream is not None
    try:
        key = key.pin_memory()
    except RuntimeError:
        pass
    try:
        value = value.pin_memory()
    except RuntimeError:
        pass

    with torch.cuda.stream(copy_stream):
        key_gpu = key.to(runtime.device, non_blocking=True)
        value_gpu = value.to(runtime.device, non_blocking=True)
    copy_stream.synchronize()
    return key_gpu, value_gpu


def _clone_cache(
    cache: DynamicCache | None,
    to_legacy: Callable[[Any], Any],
) -> DynamicCache | None:
    if cache is None:
        return None
    return DynamicCache.from_legacy_cache(tuple(to_legacy(cache)))


def _current_units(
    cache: Any,
    chunk_size: int,
    layers: int,
    heads: int,
    to_legacy: Callable[[Any], Any],
) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    legacy = to_legacy(cache)
    result = {}

    for layer in range(layers):
        key, value = legacy[layer]
        key = key[..., -chunk_size:, :]
        value = value[..., -chunk_size:, :]
        if key.shape[1] != heads:
            raise RuntimeError("KV head count mismatch")

        for head in range(heads):
            result[(layer, head)] = (
                key[:, head : head + 1].contiguous(),
                value[:, head : head + 1].contiguous(),
            )
    return result


def _append_token(
    prefix: DynamicCache | None,
    units: dict[Chunk, tuple[torch.Tensor, torch.Tensor]],
    t: int,
    layers: int,
    heads: int,
    chunk_size: int,
    to_legacy: Callable[[Any], Any],
) -> DynamicCache:
    prefix_legacy = None if prefix is None else to_legacy(prefix)
    legacy = []

    for layer in range(layers):
        keys = []
        values = []
        for head in range(heads):
            chunk = Chunk(t, layer, head)
            if chunk not in units:
                raise RuntimeError(f"missing KV unit: {chunk}")
            key, value = units[chunk]
            keys.append(key)
            values.append(value)

        current_key = torch.cat(keys, dim=1)
        current_value = torch.cat(values, dim=1)

        if prefix_legacy is not None:
            current_key = torch.cat(
                [prefix_legacy[layer][0], current_key],
                dim=-2,
            )
            current_value = torch.cat(
                [prefix_legacy[layer][1], current_value],
                dim=-2,
            )

        legacy.append((current_key, current_value))

    cache = DynamicCache.from_legacy_cache(tuple(legacy))
    expected = (t + 1) * chunk_size
    if int(cache.get_seq_length()) != expected:
        raise RuntimeError(
            f"cache length mismatch: expected={expected}, "
            f"got={cache.get_seq_length()}"
        )
    return cache


def execute_sparkv_schedule(
    *,
    model: Any,
    record: dict[str, Any],
    sample_dir: Path,
    fmt: str,
    schedule_path: str | Path,
    bandwidth_mbps: float,
    jitter_cv: float,
    rng_seed: int,
    runtime: Any,
    compute_one_chunk: Callable[..., Any],
    to_legacy: Callable[[Any], Any],
) -> tuple[DynamicCache, ScheduleExecutionStats]:
    meta = json.loads(
        (sample_dir / "meta.json").read_text(encoding="utf-8")
    )
    schedule = _load_schedule(schedule_path)
    assignments, stream_order = _validate_against_meta(schedule, meta)

    T = int(meta["num_chunks"])
    L = int(meta["layers"])
    H = int(meta["kv_heads"])
    chunk_size = int(meta["chunk_size"])

    compute_by_t = {t: set() for t in range(T)}
    stream_by_t = {t: set() for t in range(T)}
    for chunk, route in assignments.items():
        target = compute_by_t if route == "compute" else stream_by_t
        target[chunk.t].add(chunk)

    stats = ScheduleExecutionStats(
        scheduled_compute_units=sum(map(len, compute_by_t.values())),
        scheduled_stream_units=sum(map(len, stream_by_t.values())),
        predicted_makespan_ms=float(schedule["makespan_ms"]),
        schedule_stages=len(schedule["stages"]),
    )

    store = _StreamStore()
    reader = _TokenFileReader(sample_dir)
    rng = np.random.default_rng(rng_seed)
    copy_stream = torch.cuda.Stream() if runtime.is_cuda else None

    def stream_worker() -> None:
        try:
            if runtime.is_cuda:
                torch.cuda.set_device(runtime.device)

            for chunk in stream_order:
                cpu = reader.get(chunk.t)
                wire_bytes = int(
                    meta["chunks"][chunk.t]["lh_wire_bytes"][
                        f"{chunk.layer}:{chunk.head}"
                    ]
                )
                bw = _effective_bandwidth(
                    bandwidth_mbps,
                    jitter_cv,
                    rng,
                )
                delay = wire_bytes * 8 / (bw * 1e6)
                time.sleep(delay)

                begin = time.perf_counter()
                pair = _decode_unit(cpu, fmt, chunk, runtime.dtype)
                stats.decode_ms += (time.perf_counter() - begin) * 1000.0
                pair = _to_device(pair, runtime, copy_stream)

                stats.wire_bytes += wire_bytes
                stats.wire_ms += delay * 1000.0
                store.publish(chunk, pair)

            stats.disk_ms = reader.disk_ms
            stats.fetched_token_files = reader.loads
            store.finish()
        except BaseException as exc:
            stats.disk_ms = reader.disk_ms
            stats.fetched_token_files = reader.loads
            store.fail(exc)

    thread = threading.Thread(target=stream_worker, daemon=True)
    thread.start()

    prefix: DynamicCache | None = None

    try:
        for t in range(T):
            computed = {}

            if compute_by_t[t]:
                compute_prefix = _clone_cache(prefix, to_legacy)
                computed_cache = compute_one_chunk(
                    model,
                    record["prefill_ids"],
                    chunk_size,
                    t,
                    compute_prefix,
                    runtime,
                )
                stats.actual_compute_token_forwards += 1

                current = _current_units(
                    computed_cache,
                    chunk_size,
                    L,
                    H,
                    to_legacy,
                )
                for chunk in compute_by_t[t]:
                    computed[chunk] = current[
                        (chunk.layer, chunk.head)
                    ]

            streamed = store.wait_for(stream_by_t[t])
            units = {**computed, **streamed}

            expected = {
                Chunk(t, layer, head)
                for layer in range(L)
                for head in range(H)
            }
            if set(units) != expected:
                raise RuntimeError(
                    f"incomplete KV units at token chunk {t}"
                )

            prefix = _append_token(
                prefix,
                units,
                t,
                L,
                H,
                chunk_size,
                to_legacy,
            )

        thread.join()
        if store.error is not None:
            raise store.error
    finally:
        if thread.is_alive():
            thread.join()

    if prefix is None:
        raise RuntimeError("empty SparKV cache")

    if int(prefix.get_seq_length()) != int(meta["seq_len"]):
        raise RuntimeError("final SparKV cache length mismatch")

    return prefix, stats