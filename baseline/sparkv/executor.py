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

    wire_bytes: int = 0
    disk_ms: float = 0.0
    wire_ms: float = 0.0
    decode_ms: float = 0.0

    predicted_makespan_ms: float = 0.0
    schedule_stages: int = 0

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
            "scheduled_compute_units":
                self.scheduled_compute_units,

            "scheduled_stream_units":
                self.scheduled_stream_units,

            "actual_compute_token_forwards":
                self.actual_compute_token_forwards,

            "compute_materialized_units":
                self.compute_materialized_units,

            "discarded_compute_units":
                self.discarded_compute_units,

            "compute_amplification_ratio":
                self.compute_amplification_ratio,

            "fetched_token_files":
                self.fetched_token_files,

            "schedule_predicted_makespan_ms":
                self.predicted_makespan_ms,

            "schedule_stages":
                self.schedule_stages,

            # P0 executor state.
            #
            # Scheduler decisions are honored at (t,l,h) KV ownership
            # granularity, but Hugging Face still recomputes an entire
            # token chunk whenever at least one unit of that token chunk
            # is assigned to compute.
            "executor":
                "scheduler-guided-hf-compat-v2",

            "executor_granularity":
                "assignment-(t,l,h)-merge/"
                "full-token-compute",

            "stage_execution_exact": False,

            "fine_grained_compute_exact": False,
        }


def _load_schedule(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"schedule does not exist: {path}"
        )

    schedule = json.loads(
        path.read_text(encoding="utf-8")
    )

    required = {
        "makespan_ms",
        "stages",
        "chunks",
        "compute_chunks",
        "stream_chunks",
    }

    missing = required - set(schedule)

    if missing:
        raise ValueError(
            "schedule missing fields: "
            f"{sorted(missing)}"
        )

    if not isinstance(
        schedule["stages"],
        list,
    ):
        raise ValueError(
            "schedule['stages'] must be a list"
        )

    if int(schedule["chunks"]) <= 0:
        raise ValueError(
            "schedule must contain at least one unit"
        )

    if (
        int(schedule["compute_chunks"])
        + int(schedule["stream_chunks"])
        != int(schedule["chunks"])
    ):
        raise ValueError(
            "schedule route counts do not sum "
            "to total chunks"
        )

    if float(schedule["makespan_ms"]) < 0:
        raise ValueError(
            "schedule makespan must be non-negative"
        )

    return schedule


def _schedule_assignments(
    schedule: dict[str, Any],
) -> tuple[
    dict[Chunk, str],
    list[Chunk],
]:
    assignments: dict[Chunk, str] = {}
    stream_order: list[Chunk] = []

    expected_stage = 1

    for stage in schedule["stages"]:
        if not isinstance(stage, dict):
            raise ValueError(
                "each schedule stage must be a dict"
            )

        stage_id = int(
            stage.get(
                "stage",
                expected_stage,
            )
        )

        if stage_id != expected_stage:
            raise ValueError(
                "schedule stage ids must be "
                "contiguous starting at 1; "
                f"expected={expected_stage}, "
                f"got={stage_id}"
            )

        expected_stage += 1

        for operation in (
            "compute",
            "stream",
        ):
            items = stage.get(
                operation,
                [],
            )

            if not isinstance(items, list):
                raise ValueError(
                    f"stage {stage_id}: "
                    f"{operation} must be a list"
                )

            for item in items:
                try:
                    chunk = Chunk(
                        int(item["t"]),
                        int(item["layer"]),
                        int(item["head"]),
                    )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise ValueError(
                        "invalid scheduled unit "
                        f"in stage {stage_id}: {item}"
                    ) from exc

                if chunk in assignments:
                    raise ValueError(
                        "duplicate scheduled unit: "
                        f"{chunk}"
                    )

                assignments[chunk] = operation

                if operation == "stream":
                    stream_order.append(chunk)

    if len(assignments) != int(
        schedule["chunks"]
    ):
        raise ValueError(
            "schedule unit count mismatch: "
            f"declared={schedule['chunks']}, "
            f"parsed={len(assignments)}"
        )

    compute_count = sum(
        route == "compute"
        for route in assignments.values()
    )

    stream_count = sum(
        route == "stream"
        for route in assignments.values()
    )

    if compute_count != int(
        schedule["compute_chunks"]
    ):
        raise ValueError(
            "schedule compute count mismatch"
        )

    if stream_count != int(
        schedule["stream_chunks"]
    ):
        raise ValueError(
            "schedule stream count mismatch"
        )

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
    }

    missing_meta = (
        required_meta
        - set(meta)
    )

    if missing_meta:
        raise ValueError(
            "cache metadata missing fields: "
            f"{sorted(missing_meta)}"
        )

    assignments, stream_order = (
        _schedule_assignments(schedule)
    )

    T = int(meta["num_chunks"])
    L = int(meta["layers"])
    H = int(meta["kv_heads"])

    if T <= 0 or L <= 0 or H <= 0:
        raise ValueError(
            "invalid cache geometry: "
            f"T={T}, L={L}, H={H}"
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

    actual = set(assignments)

    if actual != expected:
        missing = sorted(
            expected - actual
        )

        extra = sorted(
            actual - expected
        )

        raise ValueError(
            "schedule/cache geometry mismatch; "
            f"missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )

    expected_count = T * L * H

    if int(schedule["chunks"]) != expected_count:
        raise ValueError(
            "schedule total unit count does not "
            "match T*L*H: "
            f"schedule={schedule['chunks']}, "
            f"T*L*H={expected_count}"
        )

    if len(meta["chunks"]) != T:
        raise ValueError(
            "metadata token-chunk count mismatch"
        )

    # Every streamable (layer, head) unit must have
    # an explicit wire-size entry.
    for t in range(T):
        chunk_meta = meta["chunks"][t]

        if int(
            chunk_meta.get(
                "index",
                t,
            )
        ) != t:
            raise ValueError(
                "metadata chunk index mismatch: "
                f"expected={t}"
            )

        lh_wire_bytes = chunk_meta.get(
            "lh_wire_bytes"
        )

        if not isinstance(
            lh_wire_bytes,
            dict,
        ):
            raise ValueError(
                "metadata missing "
                f"lh_wire_bytes at t={t}"
            )

        for layer in range(L):
            for head in range(H):
                key = f"{layer}:{head}"

                if key not in lh_wire_bytes:
                    raise ValueError(
                        "metadata missing wire size "
                        f"for t={t}, {key}"
                    )

                if int(
                    lh_wire_bytes[key]
                ) < 0:
                    raise ValueError(
                        "negative wire size for "
                        f"t={t}, {key}"
                    )

    return assignments, stream_order


@dataclass
class _StreamStore:
    values: dict[
        Chunk,
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ] = field(
        default_factory=dict
    )

    error: BaseException | None = None
    finished: bool = False

    def __post_init__(self) -> None:
        self.cond = (
            threading.Condition()
        )

    def publish(
        self,
        chunk: Chunk,
        pair: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        with self.cond:
            if chunk in self.values:
                raise RuntimeError(
                    "duplicate streamed unit: "
                    f"{chunk}"
                )

            self.values[chunk] = pair
            self.cond.notify_all()

    def fail(
        self,
        exc: BaseException,
    ) -> None:
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
    ) -> dict[
        Chunk,
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        if not chunks:
            return {}

        with self.cond:
            while not chunks.issubset(
                self.values
            ):
                if self.error is not None:
                    raise self.error

                if self.finished:
                    missing = sorted(
                        chunks
                        - set(self.values)
                    )

                    raise RuntimeError(
                        "stream worker ended "
                        "before all requested units "
                        "were available: "
                        f"{missing[:5]}"
                    )

                self.cond.wait()

            return {
                chunk:
                    self.values[chunk]
                for chunk in chunks
            }


class _TokenFileReader:
    def __init__(
        self,
        sample_dir: Path,
    ) -> None:
        self.sample_dir = sample_dir

        self.cache: dict[
            int,
            dict[
                str,
                torch.Tensor,
            ],
        ] = {}

        self.disk_ms = 0.0
        self.loads = 0

    def get(
        self,
        t: int,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        if t in self.cache:
            return self.cache[t]

        path = (
            self.sample_dir
            / f"chunk_{t:03d}.safetensors"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"cache chunk not found: {path}"
            )

        begin = time.perf_counter()

        tensors = load_file(
            str(path),
            device="cpu",
        )

        self.disk_ms += (
            time.perf_counter()
            - begin
        ) * 1000.0

        self.loads += 1
        self.cache[t] = tensors

        return tensors


def _effective_bandwidth(
    mean_mbps: float,
    cv: float,
    rng: np.random.Generator,
) -> float:
    if mean_mbps <= 0:
        raise ValueError(
            "bandwidth_mbps must be positive"
        )

    if cv < 0:
        raise ValueError(
            "jitter_cv must be non-negative"
        )

    if cv == 0:
        return mean_mbps

    sigma2 = math.log1p(
        cv * cv
    )

    factor = rng.lognormal(
        mean=-0.5 * sigma2,
        sigma=math.sqrt(sigma2),
    )

    return max(
        1e-3,
        mean_mbps * factor,
    )


def _decode_unit(
    cpu: dict[
        str,
        torch.Tensor,
    ],
    fmt: str,
    chunk: Chunk,
    target_dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    layer = chunk.layer
    head = chunk.head

    if fmt == "raw":
        key_name = (
            f"k_{layer:02d}"
        )

        value_name = (
            f"v_{layer:02d}"
        )

        if (
            key_name not in cpu
            or value_name not in cpu
        ):
            raise KeyError(
                "raw cache tensors missing "
                f"for layer={layer}"
            )

        key = cpu[
            key_name
        ][
            :,
            head:head + 1,
        ]

        value = cpu[
            value_name
        ][
            :,
            head:head + 1,
        ]

        return (
            key.to(target_dtype),
            value.to(target_dtype),
        )

    if fmt != "q5":
        raise ValueError(
            f"unsupported cache format: {fmt}"
        )

    required = [
        f"qk_{layer:02d}",
        f"qv_{layer:02d}",
        f"sk_{layer:02d}",
        f"sv_{layer:02d}",
    ]

    missing = [
        name
        for name in required
        if name not in cpu
    ]

    if missing:
        raise KeyError(
            "q5 cache tensors missing: "
            f"{missing}"
        )

    qk = (
        cpu[f"qk_{layer:02d}"]
        [:, head:head + 1]
        .float()
        - 16.0
    )

    qv = (
        cpu[f"qv_{layer:02d}"]
        [:, head:head + 1]
        .float()
        - 16.0
    )

    sk = (
        cpu[f"sk_{layer:02d}"]
        [:, head:head + 1]
        .float()
    )

    sv = (
        cpu[f"sv_{layer:02d}"]
        [:, head:head + 1]
        .float()
    )

    key = (
        qk * sk
    ).to(
        target_dtype
    )

    value = (
        qv * sv
    ).to(
        target_dtype
    )

    return key, value


def _to_device(
    pair: tuple[
        torch.Tensor,
        torch.Tensor,
    ],
    runtime: Any,
    copy_stream: (
        torch.cuda.Stream
        | None
    ),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    key, value = pair

    if not runtime.is_cuda:
        return (
            key.to(runtime.device),
            value.to(runtime.device),
        )

    assert (
        copy_stream is not None
    )

    try:
        key = key.pin_memory()
    except RuntimeError:
        pass

    try:
        value = value.pin_memory()
    except RuntimeError:
        pass

    with torch.cuda.stream(
        copy_stream
    ):
        key_gpu = key.to(
            runtime.device,
            non_blocking=True,
        )

        value_gpu = value.to(
            runtime.device,
            non_blocking=True,
        )

    # P0 correctness first:
    # publish only a fully resident GPU unit.
    copy_stream.synchronize()

    return key_gpu, value_gpu


def _clone_cache_container(
    cache: DynamicCache | None,
    to_legacy: Callable[
        [Any],
        Any,
    ],
) -> DynamicCache | None:
    if cache is None:
        return None

    # New cache container prevents Hugging Face's
    # DynamicCache update from changing the prefix
    # object's list structure.
    #
    # Tensor cloning is intentionally avoided here:
    # DynamicCache appends through new torch.cat
    # outputs, so duplicating the entire prefix would
    # introduce a large artificial TTFT penalty.
    legacy = tuple(
        (
            key,
            value,
        )
        for key, value
        in to_legacy(cache)
    )

    return (
        DynamicCache
        .from_legacy_cache(
            legacy
        )
    )


def _current_units(
    cache: Any,
    chunk_size: int,
    layers: int,
    heads: int,
    to_legacy: Callable[
        [Any],
        Any,
    ],
) -> dict[
    tuple[int, int],
    tuple[
        torch.Tensor,
        torch.Tensor,
    ],
]:
    legacy = to_legacy(cache)

    if len(legacy) != layers:
        raise RuntimeError(
            "computed layer count mismatch: "
            f"expected={layers}, "
            f"got={len(legacy)}"
        )

    result = {}

    for layer in range(layers):
        key, value = legacy[layer]

        if (
            key.shape[1] != heads
            or value.shape[1] != heads
        ):
            raise RuntimeError(
                "KV head count mismatch "
                f"at layer={layer}"
            )

        if (
            key.shape[-2] < chunk_size
            or value.shape[-2]
            < chunk_size
        ):
            raise RuntimeError(
                "computed cache shorter "
                "than one token chunk"
            )

        key = key[
            ...,
            -chunk_size:,
            :,
        ]

        value = value[
            ...,
            -chunk_size:,
            :,
        ]

        for head in range(heads):
            result[
                (
                    layer,
                    head,
                )
            ] = (
                key[
                    :,
                    head:head + 1,
                ].contiguous(),

                value[
                    :,
                    head:head + 1,
                ].contiguous(),
            )

    return result


def _validate_unit_pair(
    *,
    chunk: Chunk,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int,
) -> None:
    if key.ndim != 4:
        raise RuntimeError(
            f"{chunk}: key must be rank-4, "
            f"got shape={tuple(key.shape)}"
        )

    if value.ndim != 4:
        raise RuntimeError(
            f"{chunk}: value must be rank-4, "
            f"got shape={tuple(value.shape)}"
        )

    if key.shape != value.shape:
        raise RuntimeError(
            f"{chunk}: K/V shape mismatch: "
            f"K={tuple(key.shape)}, "
            f"V={tuple(value.shape)}"
        )

    if key.shape[0] != 1:
        raise RuntimeError(
            f"{chunk}: only batch size 1 "
            "is supported in P0"
        )

    if key.shape[1] != 1:
        raise RuntimeError(
            f"{chunk}: expected one KV head"
        )

    if key.shape[-2] != chunk_size:
        raise RuntimeError(
            f"{chunk}: expected seq length "
            f"{chunk_size}, "
            f"got={key.shape[-2]}"
        )


def _append_token(
    prefix: DynamicCache | None,
    units: dict[
        Chunk,
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ],
    t: int,
    layers: int,
    heads: int,
    chunk_size: int,
    to_legacy: Callable[
        [Any],
        Any,
    ],
) -> DynamicCache:
    prefix_legacy = (
        None
        if prefix is None
        else to_legacy(prefix)
    )

    expected_prefix_len = (
        t * chunk_size
    )

    if prefix is not None:
        actual_prefix_len = int(
            prefix.get_seq_length()
        )

        if (
            actual_prefix_len
            != expected_prefix_len
        ):
            raise RuntimeError(
                "prefix cache length mismatch "
                f"before t={t}: "
                f"expected={expected_prefix_len}, "
                f"got={actual_prefix_len}"
            )

    legacy = []

    for layer in range(layers):
        keys = []
        values = []

        for head in range(heads):
            chunk = Chunk(
                t,
                layer,
                head,
            )

            if chunk not in units:
                raise RuntimeError(
                    "missing KV unit: "
                    f"{chunk}"
                )

            key, value = units[chunk]

            _validate_unit_pair(
                chunk=chunk,
                key=key,
                value=value,
                chunk_size=chunk_size,
            )

            keys.append(key)
            values.append(value)

        current_key = torch.cat(
            keys,
            dim=1,
        )

        current_value = torch.cat(
            values,
            dim=1,
        )

        if prefix_legacy is not None:
            prefix_key = (
                prefix_legacy[layer][0]
            )

            prefix_value = (
                prefix_legacy[layer][1]
            )

            current_key = torch.cat(
                [
                    prefix_key,
                    current_key,
                ],
                dim=-2,
            )

            current_value = torch.cat(
                [
                    prefix_value,
                    current_value,
                ],
                dim=-2,
            )

        legacy.append(
            (
                current_key,
                current_value,
            )
        )

    cache = (
        DynamicCache
        .from_legacy_cache(
            tuple(legacy)
        )
    )

    expected = (
        (t + 1)
        * chunk_size
    )

    actual = int(
        cache.get_seq_length()
    )

    if actual != expected:
        raise RuntimeError(
            "cache length mismatch "
            f"after t={t}: "
            f"expected={expected}, "
            f"got={actual}"
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
    compute_one_chunk: Callable[
        ...,
        Any,
    ],
    to_legacy: Callable[
        [Any],
        Any,
    ],
) -> tuple[
    DynamicCache,
    ScheduleExecutionStats,
]:
    meta_path = (
        sample_dir
        / "meta.json"
    )

    if not meta_path.is_file():
        raise FileNotFoundError(
            f"cache metadata not found: "
            f"{meta_path}"
        )

    meta = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )

    schedule = _load_schedule(
        schedule_path
    )

    assignments, stream_order = (
        _validate_against_meta(
            schedule,
            meta,
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
        T * chunk_size
        != seq_len
    ):
        raise ValueError(
            "metadata sequence geometry "
            "is inconsistent: "
            f"T={T}, "
            f"chunk_size={chunk_size}, "
            f"seq_len={seq_len}"
        )

    if len(
        record["prefill_ids"]
    ) != seq_len:
        raise ValueError(
            "prepared sample/cache "
            "sequence length mismatch"
        )

    compute_by_t: dict[
        int,
        set[Chunk],
    ] = {
        t: set()
        for t in range(T)
    }

    stream_by_t: dict[
        int,
        set[Chunk],
    ] = {
        t: set()
        for t in range(T)
    }

    for chunk, route in (
        assignments.items()
    ):
        if route == "compute":
            compute_by_t[
                chunk.t
            ].add(chunk)

        elif route == "stream":
            stream_by_t[
                chunk.t
            ].add(chunk)

        else:
            raise RuntimeError(
                "unexpected schedule route: "
                f"{route}"
            )

    scheduled_compute = sum(
        len(items)
        for items
        in compute_by_t.values()
    )

    scheduled_stream = sum(
        len(items)
        for items
        in stream_by_t.values()
    )

    stats = (
        ScheduleExecutionStats(
            scheduled_compute_units=(
                scheduled_compute
            ),
            scheduled_stream_units=(
                scheduled_stream
            ),
            predicted_makespan_ms=float(
                schedule["makespan_ms"]
            ),
            schedule_stages=len(
                schedule["stages"]
            ),
        )
    )

    store = _StreamStore()

    reader = _TokenFileReader(
        sample_dir
    )

    rng = (
        np.random.default_rng(
            rng_seed
        )
    )

    copy_stream = (
        torch.cuda.Stream()
        if runtime.is_cuda
        else None
    )

    cancel_event = (
        threading.Event()
    )

    def update_reader_stats() -> None:
        stats.disk_ms = (
            reader.disk_ms
        )

        stats.fetched_token_files = (
            reader.loads
        )

    def stream_worker() -> None:
        try:
            if runtime.is_cuda:
                torch.cuda.set_device(
                    runtime.device
                )

            for chunk in stream_order:
                if cancel_event.is_set():
                    store.finish()
                    return

                cpu = reader.get(
                    chunk.t
                )

                wire_key = (
                    f"{chunk.layer}:"
                    f"{chunk.head}"
                )

                wire_bytes = int(
                    meta["chunks"]
                    [chunk.t]
                    ["lh_wire_bytes"]
                    [wire_key]
                )

                bw = (
                    _effective_bandwidth(
                        bandwidth_mbps,
                        jitter_cv,
                        rng,
                    )
                )

                delay = (
                    wire_bytes
                    * 8.0
                    / (
                        bw
                        * 1e6
                    )
                )

                # event.wait(timeout) gives the same
                # simulated transfer delay while
                # allowing executor cancellation.
                cancelled = (
                    cancel_event.wait(
                        delay
                    )
                )

                if cancelled:
                    store.finish()
                    return

                decode_begin = (
                    time.perf_counter()
                )

                pair = _decode_unit(
                    cpu,
                    fmt,
                    chunk,
                    runtime.dtype,
                )

                stats.decode_ms += (
                    time.perf_counter()
                    - decode_begin
                ) * 1000.0

                pair = _to_device(
                    pair,
                    runtime,
                    copy_stream,
                )

                stats.wire_bytes += (
                    wire_bytes
                )

                stats.wire_ms += (
                    delay
                    * 1000.0
                )

                store.publish(
                    chunk,
                    pair,
                )

            update_reader_stats()
            store.finish()

        except BaseException as exc:
            update_reader_stats()
            store.fail(exc)

    thread = threading.Thread(
        target=stream_worker,
        daemon=True,
        name="sparkv-stream-worker",
    )

    thread.start()

    prefix: DynamicCache | None = None

    try:
        for t in range(T):
            computed: dict[
                Chunk,
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                ],
            ] = {}

            compute_units = (
                compute_by_t[t]
            )

            if compute_units:
                compute_prefix = (
                    _clone_cache_container(
                        prefix,
                        to_legacy,
                    )
                )

                computed_cache = (
                    compute_one_chunk(
                        model,
                        record[
                            "prefill_ids"
                        ],
                        chunk_size,
                        t,
                        compute_prefix,
                        runtime,
                    )
                )

                stats.actual_compute_token_forwards += 1

                # HF computes every layer/head for
                # this token chunk even though only
                # scheduler-assigned units are kept.
                stats.compute_materialized_units += (
                    L * H
                )

                current = (
                    _current_units(
                        computed_cache,
                        chunk_size,
                        L,
                        H,
                        to_legacy,
                    )
                )

                for chunk in compute_units:
                    computed[chunk] = (
                        current[
                            (
                                chunk.layer,
                                chunk.head,
                            )
                        ]
                    )

            streamed = (
                store.wait_for(
                    stream_by_t[t]
                )
            )

            units = {
                **computed,
                **streamed,
            }

            expected = {
                Chunk(
                    t,
                    layer,
                    head,
                )
                for layer in range(L)
                for head in range(H)
            }

            if set(units) != expected:
                missing = sorted(
                    expected
                    - set(units)
                )

                extra = sorted(
                    set(units)
                    - expected
                )

                raise RuntimeError(
                    "incomplete KV ownership "
                    f"at token chunk t={t}; "
                    f"missing={missing[:5]}, "
                    f"extra={extra[:5]}"
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

    except BaseException:
        cancel_event.set()
        raise

    finally:
        if thread.is_alive():
            cancel_event.set()
            thread.join()

        update_reader_stats()

    if prefix is None:
        raise RuntimeError(
            "empty SparKV cache"
        )

    final_len = int(
        prefix.get_seq_length()
    )

    if final_len != seq_len:
        raise RuntimeError(
            "final SparKV cache length "
            "mismatch: "
            f"expected={seq_len}, "
            f"got={final_len}"
        )

    stats.discarded_compute_units = (
        stats.compute_materialized_units
        - stats.scheduled_compute_units
    )

    if stats.discarded_compute_units < 0:
        raise RuntimeError(
            "executor materialized fewer "
            "compute units than the schedule "
            "requested"
        )

    return prefix, stats