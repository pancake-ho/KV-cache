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
    compute_materialized_units: int = 0
    discarded_compute_units: int = 0

    fetched_token_files: int = 0
    unit_files_loaded: int = 0

    wire_bytes: int = 0
    disk_ms: float = 0.0
    decode_ms: float = 0.0
    h2d_ms: float = 0.0
    wire_ms: float = 0.0
    compute_ms: float = 0.0
    wait_ms: float = 0.0
    merge_ms: float = 0.0

    predicted_makespan_ms: float = 0.0
    schedule_stages: int = 0

    peak_buffered_stream_units: int = 0

    @property
    def compute_amplification_ratio(self) -> float:
        if self.scheduled_compute_units <= 0:
            return 0.0
        return (
            self.compute_materialized_units
            / self.scheduled_compute_units
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduled_compute_units": self.scheduled_compute_units,
            "scheduled_stream_units": self.scheduled_stream_units,
            "actual_compute_token_forwards": self.actual_compute_token_forwards,
            "compute_materialized_units": self.compute_materialized_units,
            "discarded_compute_units": self.discarded_compute_units,
            "compute_amplification_ratio": self.compute_amplification_ratio,
            "fetched_token_files": self.fetched_token_files,
            "unit_files_loaded": self.unit_files_loaded,
            "stream_disk_ms": self.disk_ms,
            "stream_decode_ms": self.decode_ms,
            "stream_h2d_ms": self.h2d_ms,
            "stream_wire_ms": self.wire_ms,
            "actual_compute_ms": self.compute_ms,
            "stream_wait_ms": self.wait_ms,
            "cache_merge_ms": self.merge_ms,
            "peak_buffered_stream_units": self.peak_buffered_stream_units,
            "schedule_predicted_makespan_ms": self.predicted_makespan_ms,
            "schedule_stages": self.schedule_stages,
            "executor": "scheduler-guided-unit-stream-v3",
            "executor_granularity": (
                "stream-(t,l,h)-exact/"
                "compute-full-token-compat"
            ),
            "stream_granularity_exact": True,
            "stage_execution_exact": False,
            "fine_grained_compute_exact": False,
        }


def _load_schedule(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"schedule does not exist: {path}")

    schedule = json.loads(path.read_text(encoding="utf-8"))
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
    if not isinstance(schedule["stages"], list):
        raise ValueError("schedule['stages'] must be a list")
    if int(schedule["chunks"]) <= 0:
        raise ValueError("schedule must contain at least one unit")
    if (
        int(schedule["compute_chunks"])
        + int(schedule["stream_chunks"])
        != int(schedule["chunks"])
    ):
        raise ValueError("schedule route counts do not sum to total chunks")
    if float(schedule["makespan_ms"]) < 0:
        raise ValueError("schedule makespan must be non-negative")
    return schedule


def _schedule_assignments(
    schedule: dict[str, Any],
) -> tuple[dict[Chunk, str], list[Chunk]]:
    assignments: dict[Chunk, str] = {}
    stream_order: list[Chunk] = []
    expected_stage = 1

    for stage in schedule["stages"]:
        if not isinstance(stage, dict):
            raise ValueError("each schedule stage must be a dict")

        stage_id = int(stage.get("stage", expected_stage))
        if stage_id != expected_stage:
            raise ValueError(
                "schedule stage ids must be contiguous starting at 1; "
                f"expected={expected_stage}, got={stage_id}"
            )
        expected_stage += 1

        for operation in ("compute", "stream"):
            items = stage.get(operation, [])
            if not isinstance(items, list):
                raise ValueError(
                    f"stage {stage_id}: {operation} must be a list"
                )

            for item in items:
                chunk = Chunk(
                    int(item["t"]),
                    int(item["layer"]),
                    int(item["head"]),
                )
                if chunk in assignments:
                    raise ValueError(f"duplicate scheduled unit: {chunk}")
                assignments[chunk] = operation
                if operation == "stream":
                    stream_order.append(chunk)

    if len(assignments) != int(schedule["chunks"]):
        raise ValueError(
            "schedule unit count mismatch: "
            f"declared={schedule['chunks']}, parsed={len(assignments)}"
        )

    compute_count = sum(
        route == "compute"
        for route in assignments.values()
    )
    stream_count = sum(
        route == "stream"
        for route in assignments.values()
    )
    if compute_count != int(schedule["compute_chunks"]):
        raise ValueError("schedule compute count mismatch")
    if stream_count != int(schedule["stream_chunks"]):
        raise ValueError("schedule stream count mismatch")

    return assignments, stream_order


def _validate_against_meta(
    schedule: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[
    dict[Chunk, str],
    list[Chunk],
]:
    required_meta = {
        "num_chunks",
        "layers",
        "kv_heads",
        "chunk_size",
        "seq_len",
        "chunks",
        "unit_layout_version",
        "unit_files",
    }

    missing = (
        required_meta
        - set(meta)
    )

    if missing:
        raise ValueError(
            "cache metadata missing fields: "
            f"{sorted(missing)}"
        )

    if int(
        meta["unit_layout_version"]
    ) < 1:
        raise ValueError(
            "fine-grained unit layout "
            "version is invalid"
        )

    assignments, stream_order = (
        _schedule_assignments(
            schedule
        )
    )

    T = int(
        meta["num_chunks"]
    )

    L = int(
        meta["layers"]
    )

    H = int(
        meta["kv_heads"]
    )

    chunk_size = int(
        meta["chunk_size"]
    )

    seq_len = int(
        meta["seq_len"]
    )

    if (
        T <= 0
        or L <= 0
        or H <= 0
        or chunk_size <= 0
    ):
        raise ValueError(
            "invalid cache geometry: "
            f"T={T}, "
            f"L={L}, "
            f"H={H}, "
            f"chunk_size={chunk_size}"
        )

    if (
        T * chunk_size
        != seq_len
    ):
        raise ValueError(
            "cache sequence geometry "
            "mismatch: "
            f"T*chunk_size="
            f"{T * chunk_size}, "
            f"seq_len={seq_len}"
        )

    expected = {
        Chunk(
            t,
            layer,
            head,
        )
        for t in range(T)
        for layer in range(L)
        for head in range(H)
    }

    actual = set(
        assignments
    )

    # Check scheduler geometry first.
    # This preserves the intended failure
    # contract of geometry tests.
    if actual != expected:
        missing_units = sorted(
            expected - actual
        )

        extra_units = sorted(
            actual - expected
        )

        raise ValueError(
            "schedule/cache geometry "
            "mismatch; "
            f"missing="
            f"{missing_units[:5]}, "
            f"extra="
            f"{extra_units[:5]}"
        )

    if int(
        schedule["chunks"]
    ) != (
        T * L * H
    ):
        raise ValueError(
            "schedule unit count "
            "does not match "
            "cache geometry"
        )

    chunks_meta = (
        meta["chunks"]
    )

    if not isinstance(
        chunks_meta,
        list,
    ):
        raise ValueError(
            "meta['chunks'] "
            "must be a list"
        )

    if len(
        chunks_meta
    ) != T:
        raise ValueError(
            "metadata token-chunk "
            "count mismatch: "
            f"expected={T}, "
            f"got={len(chunks_meta)}"
        )

    # ----------------------------------------------------------
    # Validate legacy/logical wire metadata.
    #
    # scheduler.py still relies on:
    # meta["chunks"][t]["lh_wire_bytes"]["layer:head"]
    # ----------------------------------------------------------
    for t in range(T):
        chunk_meta = (
            chunks_meta[t]
        )

        index = int(
            chunk_meta.get(
                "index",
                t,
            )
        )

        if index != t:
            raise ValueError(
                "metadata chunk index "
                "mismatch: "
                f"expected={t}, "
                f"got={index}"
            )

        lh_wire_bytes = (
            chunk_meta.get(
                "lh_wire_bytes"
            )
        )

        if not isinstance(
            lh_wire_bytes,
            dict,
        ):
            raise ValueError(
                "missing wire metadata "
                f"for token chunk {t}"
            )

        for layer in range(L):
            for head in range(H):
                wire_key = (
                    f"{layer}:"
                    f"{head}"
                )

                if (
                    wire_key
                    not in
                    lh_wire_bytes
                ):
                    raise ValueError(
                        "missing wire size "
                        f"for "
                        f"t={t}, "
                        f"layer={layer}, "
                        f"head={head}"
                    )

                wire_bytes = int(
                    lh_wire_bytes[
                        wire_key
                    ]
                )

                if wire_bytes < 0:
                    raise ValueError(
                        "negative wire size "
                        f"for "
                        f"t={t}, "
                        f"layer={layer}, "
                        f"head={head}"
                    )

    # ----------------------------------------------------------
    # Validate physical fine-grained unit-file metadata.
    # ----------------------------------------------------------
    unit_files = (
        meta["unit_files"]
    )

    if not isinstance(
        unit_files,
        dict,
    ):
        raise ValueError(
            "meta['unit_files'] "
            "must be a dict"
        )

    for chunk in expected:
        unit_key = (
            f"{chunk.t}:"
            f"{chunk.layer}:"
            f"{chunk.head}"
        )

        if (
            unit_key
            not in unit_files
        ):
            raise ValueError(
                "unit file metadata "
                "missing for "
                f"{unit_key}"
            )

        info = (
            unit_files[
                unit_key
            ]
        )

        if not isinstance(
            info,
            dict,
        ):
            raise ValueError(
                "invalid unit file "
                "metadata for "
                f"{unit_key}"
            )

        required_unit_fields = {
            "path",
            "wire_bytes",
            "storage_bytes",
        }

        missing_fields = (
            required_unit_fields
            - set(info)
        )

        if missing_fields:
            raise ValueError(
                "unit file metadata "
                f"missing fields for "
                f"{unit_key}: "
                f"{sorted(missing_fields)}"
            )

        path = str(
            info["path"]
        )

        if not path:
            raise ValueError(
                "empty unit file path "
                f"for {unit_key}"
            )

        physical_wire_bytes = int(
            info[
                "wire_bytes"
            ]
        )

        storage_bytes = int(
            info[
                "storage_bytes"
            ]
        )

        if (
            physical_wire_bytes
            < 0
        ):
            raise ValueError(
                "negative unit wire "
                "size for "
                f"{unit_key}"
            )

        if storage_bytes < 0:
            raise ValueError(
                "negative unit storage "
                "size for "
                f"{unit_key}"
            )

        # Logical scheduler wire size and
        # unit-file wire size must agree.
        logical_wire_bytes = int(
            chunks_meta[
                chunk.t
            ][
                "lh_wire_bytes"
            ][
                f"{chunk.layer}:"
                f"{chunk.head}"
            ]
        )

        if (
            physical_wire_bytes
            != logical_wire_bytes
        ):
            raise ValueError(
                "wire-size metadata "
                "mismatch for "
                f"{unit_key}: "
                f"logical="
                f"{logical_wire_bytes}, "
                f"unit="
                f"{physical_wire_bytes}"
            )

    if (
        "unit_file_count"
        in meta
        and int(
            meta[
                "unit_file_count"
            ]
        )
        != len(expected)
    ):
        raise ValueError(
            "unit_file_count "
            "mismatch: "
            f"expected={len(expected)}, "
            f"got="
            f"{meta['unit_file_count']}"
        )

    return (
        assignments,
        stream_order,
    )


@dataclass
class _StreamStore:
    values: dict[
        Chunk,
        tuple[torch.Tensor, torch.Tensor],
    ] = field(default_factory=dict)
    error: BaseException | None = None
    finished: bool = False
    peak_size: int = 0

    def __post_init__(self) -> None:
        self.cond = threading.Condition()

    def publish(
        self,
        chunk: Chunk,
        pair: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        with self.cond:
            if chunk in self.values:
                raise RuntimeError(f"duplicate streamed unit: {chunk}")
            self.values[chunk] = pair
            self.peak_size = max(self.peak_size, len(self.values))
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

    def take(
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
                        "stream worker ended before all requested units "
                        f"were available: {missing[:5]}"
                    )
                self.cond.wait()

            result = {
                chunk: self.values.pop(chunk)
                for chunk in chunks
            }
            self.cond.notify_all()
            return result


class _UnitFileReader:
    def __init__(
        self,
        sample_dir: Path,
        meta: dict[str, Any],
    ) -> None:
        self.sample_dir = sample_dir
        self.unit_files = meta["unit_files"]
        self.loads = 0
        self.disk_ms = 0.0
        self.touched_token_chunks: set[int] = set()

    def get(
        self,
        chunk: Chunk,
    ) -> dict[str, torch.Tensor]:
        key = f"{chunk.t}:{chunk.layer}:{chunk.head}"
        info = self.unit_files[key]
        path = self.sample_dir / str(info["path"])
        if not path.is_file():
            raise FileNotFoundError(f"unit cache file not found: {path}")

        begin = time.perf_counter()
        tensors = load_file(str(path), device="cpu")
        self.disk_ms += (time.perf_counter() - begin) * 1000.0
        self.loads += 1
        self.touched_token_chunks.add(chunk.t)
        return tensors


def _effective_bandwidth(
    mean_mbps: float,
    cv: float,
    rng: np.random.Generator,
) -> float:
    if mean_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive")
    if cv < 0:
        raise ValueError("jitter_cv must be non-negative")
    if cv == 0:
        return mean_mbps

    sigma2 = math.log1p(cv * cv)
    factor = rng.lognormal(
        mean=-0.5 * sigma2,
        sigma=math.sqrt(sigma2),
    )
    return max(1e-3, mean_mbps * factor)


def _decode_unit(
    tensors: dict[str, torch.Tensor],
    fmt: str,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fmt == "raw":
        return (
            tensors["k"].to(target_dtype),
            tensors["v"].to(target_dtype),
        )

    if fmt != "q5":
        raise ValueError(f"unsupported cache format: {fmt}")

    qk = tensors["qk"].float() - 16.0
    qv = tensors["qv"].float() - 16.0
    sk = tensors["sk"].float()
    sv = tensors["sv"].float()

    return (
        (qk * sk).to(target_dtype),
        (qv * sv).to(target_dtype),
    )


def _to_device(
    pair: tuple[torch.Tensor, torch.Tensor],
    runtime: Any,
    copy_stream: torch.cuda.Stream | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
    key, value = pair

    if not runtime.is_cuda:
        begin = time.perf_counter()
        result = (
            key.to(runtime.device),
            value.to(runtime.device),
        )
        return result, (time.perf_counter() - begin) * 1000.0

    assert copy_stream is not None

    try:
        key = key.pin_memory()
    except RuntimeError:
        pass
    try:
        value = value.pin_memory()
    except RuntimeError:
        pass

    begin = time.perf_counter()
    with torch.cuda.stream(copy_stream):
        key_gpu = key.to(runtime.device, non_blocking=True)
        value_gpu = value.to(runtime.device, non_blocking=True)
    copy_stream.synchronize()
    elapsed = (time.perf_counter() - begin) * 1000.0
    return (key_gpu, value_gpu), elapsed


def _clone_cache_container(
    cache: DynamicCache | None,
    to_legacy: Callable[[Any], Any],
) -> DynamicCache | None:
    if cache is None:
        return None

    legacy = tuple(
        (key, value)
        for key, value in to_legacy(cache)
    )
    return DynamicCache.from_legacy_cache(legacy)


def _current_units(
    cache: Any,
    chunk_size: int,
    layers: int,
    heads: int,
    to_legacy: Callable[[Any], Any],
) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    legacy = to_legacy(cache)
    if len(legacy) != layers:
        raise RuntimeError("computed layer count mismatch")

    result = {}
    for layer in range(layers):
        key, value = legacy[layer]
        if key.shape[1] != heads or value.shape[1] != heads:
            raise RuntimeError(f"KV head count mismatch at layer={layer}")

        key = key[..., -chunk_size:, :]
        value = value[..., -chunk_size:, :]

        for head in range(heads):
            result[(layer, head)] = (
                key[:, head : head + 1].contiguous(),
                value[:, head : head + 1].contiguous(),
            )
    return result


def _validate_unit_pair(
    chunk: Chunk,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int,
) -> None:
    if key.ndim != 4 or value.ndim != 4:
        raise RuntimeError(f"{chunk}: KV tensors must be rank-4")
    if key.shape != value.shape:
        raise RuntimeError(f"{chunk}: K/V shape mismatch")
    if key.shape[0] != 1 or key.shape[1] != 1:
        raise RuntimeError(f"{chunk}: expected batch=1 and one KV head")
    if key.shape[-2] != chunk_size:
        raise RuntimeError(
            f"{chunk}: expected sequence length {chunk_size}, "
            f"got {key.shape[-2]}"
        )


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

    if prefix is not None:
        expected_prefix = t * chunk_size
        actual_prefix = int(prefix.get_seq_length())
        if actual_prefix != expected_prefix:
            raise RuntimeError(
                "prefix length mismatch: "
                f"expected={expected_prefix}, got={actual_prefix}"
            )

    legacy = []
    for layer in range(layers):
        keys = []
        values = []

        for head in range(heads):
            chunk = Chunk(t, layer, head)
            if chunk not in units:
                raise RuntimeError(f"missing KV unit: {chunk}")

            key, value = units[chunk]
            _validate_unit_pair(chunk, key, value, chunk_size)
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
            f"cache length mismatch after t={t}: "
            f"expected={expected}, got={cache.get_seq_length()}"
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
    seq_len = int(meta["seq_len"])

    if T * chunk_size != seq_len:
        raise ValueError("metadata sequence geometry is inconsistent")
    if len(record["prefill_ids"]) != seq_len:
        raise ValueError("prepared sample/cache sequence length mismatch")

    compute_by_t = {t: set() for t in range(T)}
    stream_by_t = {t: set() for t in range(T)}
    for chunk, route in assignments.items():
        if route == "compute":
            compute_by_t[chunk.t].add(chunk)
        elif route == "stream":
            stream_by_t[chunk.t].add(chunk)
        else:
            raise RuntimeError(f"unexpected route: {route}")

    stats = ScheduleExecutionStats(
        scheduled_compute_units=sum(map(len, compute_by_t.values())),
        scheduled_stream_units=sum(map(len, stream_by_t.values())),
        predicted_makespan_ms=float(schedule["makespan_ms"]),
        schedule_stages=len(schedule["stages"]),
    )

    store = _StreamStore()
    reader = _UnitFileReader(sample_dir, meta)
    rng = np.random.default_rng(rng_seed)
    copy_stream = torch.cuda.Stream() if runtime.is_cuda else None
    cancel = threading.Event()

    def stream_worker() -> None:
        try:
            if runtime.is_cuda:
                torch.cuda.set_device(runtime.device)

            for chunk in stream_order:
                if cancel.is_set():
                    store.finish()
                    return

                tensors = reader.get(chunk)

                key = f"{chunk.t}:{chunk.layer}:{chunk.head}"
                wire_bytes = int(meta["unit_files"][key]["wire_bytes"])
                bw = _effective_bandwidth(
                    bandwidth_mbps,
                    jitter_cv,
                    rng,
                )
                delay = wire_bytes * 8.0 / (bw * 1e6)

                if cancel.wait(delay):
                    store.finish()
                    return

                begin = time.perf_counter()
                pair = _decode_unit(tensors, fmt, runtime.dtype)
                stats.decode_ms += (
                    time.perf_counter() - begin
                ) * 1000.0

                pair, h2d_ms = _to_device(
                    pair,
                    runtime,
                    copy_stream,
                )
                stats.h2d_ms += h2d_ms
                stats.wire_bytes += wire_bytes
                stats.wire_ms += delay * 1000.0

                store.publish(chunk, pair)
                del tensors

            store.finish()

        except BaseException as exc:
            store.fail(exc)

    thread = threading.Thread(
        target=stream_worker,
        daemon=True,
        name="sparkv-unit-stream-worker",
    )
    thread.start()

    prefix: DynamicCache | None = None

    try:
        for t in range(T):
            computed: dict[
                Chunk,
                tuple[torch.Tensor, torch.Tensor],
            ] = {}

            compute_units = compute_by_t[t]
            if compute_units:
                compute_prefix = _clone_cache_container(
                    prefix,
                    to_legacy,
                )

                begin = time.perf_counter()
                computed_cache = compute_one_chunk(
                    model,
                    record["prefill_ids"],
                    chunk_size,
                    t,
                    compute_prefix,
                    runtime,
                )
                if runtime.is_cuda:
                    torch.cuda.synchronize(runtime.device)
                stats.compute_ms += (
                    time.perf_counter() - begin
                ) * 1000.0

                stats.actual_compute_token_forwards += 1
                stats.compute_materialized_units += L * H

                current = _current_units(
                    computed_cache,
                    chunk_size,
                    L,
                    H,
                    to_legacy,
                )
                for chunk in compute_units:
                    computed[chunk] = current[
                        (chunk.layer, chunk.head)
                    ]

                del computed_cache, current

            begin = time.perf_counter()
            streamed = store.take(stream_by_t[t])
            stats.wait_ms += (
                time.perf_counter() - begin
            ) * 1000.0

            units = {**computed, **streamed}
            expected = {
                Chunk(t, layer, head)
                for layer in range(L)
                for head in range(H)
            }
            if set(units) != expected:
                missing = sorted(expected - set(units))
                extra = sorted(set(units) - expected)
                raise RuntimeError(
                    f"incomplete KV ownership at t={t}; "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )

            begin = time.perf_counter()
            prefix = _append_token(
                prefix,
                units,
                t,
                L,
                H,
                chunk_size,
                to_legacy,
            )
            stats.merge_ms += (
                time.perf_counter() - begin
            ) * 1000.0

            # Drop temporary references immediately after they have been
            # incorporated into the prefix cache.
            del units, streamed, computed

        thread.join()
        if store.error is not None:
            raise store.error

    except BaseException:
        cancel.set()
        raise

    finally:
        if thread.is_alive():
            cancel.set()
            thread.join()

    if prefix is None:
        raise RuntimeError("empty SparKV cache")
    if int(prefix.get_seq_length()) != seq_len:
        raise RuntimeError("final SparKV cache length mismatch")

    stats.unit_files_loaded = reader.loads
    stats.fetched_token_files = len(reader.touched_token_chunks)
    stats.disk_ms = reader.disk_ms
    stats.peak_buffered_stream_units = store.peak_size
    stats.discarded_compute_units = (
        stats.compute_materialized_units
        - stats.scheduled_compute_units
    )

    if stats.discarded_compute_units < 0:
        raise RuntimeError(
            "executor materialized fewer units than requested by schedule"
        )

    return prefix, stats
