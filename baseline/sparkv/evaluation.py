from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from baseline.sparkv.executor import (
    CloudMemorySource,
    ExecutionStats,
    execute_sparkv,
)
from baseline.sparkv.experiment import (
    _generate_first_token,
)
from baseline.sparkv.runtime import (
    MODEL_ID,
    PowerSampler,
    clear_device_cache,
    continue_greedy,
    device_synchronize,
    load_model,
    qa_f1,
    seed_everything,
)
from baseline.sparkv.runtime_controller import (
    RuntimeControllerConfig,
)
from baseline.sparkv.scheduler import Chunk


def _load_records(
    path: Path,
) -> list[dict[str, Any]]:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def _load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def _chunk_from_key(
    key: str,
) -> dict[str, int]:
    t, layer, head = (
        int(x)
        for x in key.split(":")
    )
    return {
        "t": t,
        "layer": layer,
        "head": head,
    }


def build_all_stream_schedule(
    *,
    sparkv_schedule_path: Path,
    output: Path,
) -> dict[str, Any]:
    """
    Build a control baseline that sends every KV unit from cloud.

    This deliberately reuses the exact per-unit stream cost table produced for
    the SparKV schedule.  No local compute and no runtime migration are allowed
    in the all-stream run, so differences against SparKV isolate the routing
    decision rather than codec/network implementation changes.
    """
    source = _load_json(
        sparkv_schedule_path
    )
    unit_costs = source[
        "unit_costs"
    ]

    ordered_keys = sorted(
        unit_costs,
        key=lambda key: tuple(
            int(x)
            for x in key.split(":")
        ),
    )

    stream_items = [
        _chunk_from_key(key)
        for key in ordered_keys
    ]

    stream_ms = float(
        sum(
            float(
                unit_costs[key][
                    "stream_ms"
                ]
            )
            for key in ordered_keys
        )
    )

    result = {
        "scheduler":
            "all-stream-control",
        "indexing":
            "zero-based-python",
        "delta_ms":
            stream_ms,
        "makespan_ms":
            stream_ms,
        "chunks":
            len(ordered_keys),
        "compute_chunks":
            0,
        "stream_chunks":
            len(ordered_keys),
        "stages": [
            {
                "stage": 1,
                "duration_ms":
                    stream_ms,
                "compute_ms":
                    0.0,
                "stream_ms":
                    stream_ms,
                "compute": [],
                "stream":
                    stream_items,
            }
        ],
        "assignments": {
            key: "stream"
            for key in ordered_keys
        },
        "unit_costs":
            unit_costs,
        "control_baseline":
            True,
        "source_sparkv_schedule":
            str(
                sparkv_schedule_path
            ),
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def _schedule_geometry(
    unit_costs: dict[str, Any],
) -> tuple[list[Chunk], int, int, int]:
    chunks = sorted(
        (
            Chunk(**_chunk_from_key(key))
            for key in unit_costs
        ),
    )
    if not chunks:
        raise ValueError("empty unit cost table")

    token_chunks = max(c.t for c in chunks) + 1
    layers = max(c.layer for c in chunks) + 1
    heads = max(c.head for c in chunks) + 1
    expected = token_chunks * layers * heads
    if len(chunks) != expected:
        raise ValueError(
            "unit cost table is not a complete (token, layer, head) grid: "
            f"expected={expected}, got={len(chunks)}"
        )
    return chunks, token_chunks, layers, heads


def _fixed_compute_ready(
    c: Chunk,
    done: dict[Chunk, str],
    layers: int,
) -> bool:
    token_ready = (
        c.t == 0
        or c.layer == layers - 1
        or Chunk(c.t - 1, c.layer, c.head) in done
    )
    layer_ready = (
        c.layer == 0
        or done.get(Chunk(c.t, c.layer - 1, c.head))
        == "compute"
    )
    return c not in done and token_ready and layer_ready


def _build_fixed_pipeline_schedule(
    *,
    sparkv_schedule_path: Path,
    output: Path,
    compute_tokens: int,
    scheduler_name: str,
    baseline_name: str,
    partition_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a dependency-valid fixed positional baseline.

    Early token chunks are recomputed and later token chunks are streamed.
    Unlike SparKV, the assignment never uses per-unit overhead and runtime
    migration is disabled by the evaluator.  This is the paper's Strong
    Hybrid structure, while ``compute_tokens == T`` is Local Prefill using
    the same SpargeAttention executor as SparKV.
    """
    source = _load_json(sparkv_schedule_path)
    unit_costs = source["unit_costs"]
    chunks, token_chunks, layers, heads = _schedule_geometry(unit_costs)
    if not 0 <= compute_tokens <= token_chunks:
        raise ValueError(
            f"compute_tokens must be in [0, {token_chunks}]"
        )

    compute_pending = {
        c for c in chunks if c.t < compute_tokens
    }
    stream_pending = set(chunks) - compute_pending

    requested_delta = float(source.get("delta_ms", 0.0))
    largest_compute = max(
        (
            float(unit_costs[c.key]["comp_ms"])
            for c in compute_pending
        ),
        default=0.0,
    )
    largest_stream = max(
        (
            float(unit_costs[c.key]["stream_ms"])
            for c in stream_pending
        ),
        default=0.0,
    )
    delta_ms = max(requested_delta, largest_compute, largest_stream)
    if delta_ms <= 0:
        raise ValueError("fixed pipeline requires positive unit costs")

    done: dict[Chunk, str] = {}
    stages: list[dict[str, Any]] = []
    makespan_ms = 0.0

    while len(done) < len(chunks):
        selected_compute: list[Chunk] = []
        compute_ms = 0.0
        while True:
            ready = sorted(
                c
                for c in compute_pending
                if _fixed_compute_ready(c, done, layers)
            )
            fitting = [
                c
                for c in ready
                if compute_ms
                + float(unit_costs[c.key]["comp_ms"])
                <= delta_ms + 1e-9
            ]
            if not fitting:
                break
            c = fitting[0]
            compute_pending.remove(c)
            done[c] = "compute"
            selected_compute.append(c)
            compute_ms += float(unit_costs[c.key]["comp_ms"])

        selected_stream: list[Chunk] = []
        stream_ms = 0.0
        # The fixed hybrid baseline streams later context first while the
        # early prefix advances through local Transformer dependencies.
        for c in sorted(
            stream_pending,
            key=lambda item: (-item.t, item.layer, item.head),
        ):
            cost = float(unit_costs[c.key]["stream_ms"])
            if stream_ms + cost <= delta_ms + 1e-9:
                selected_stream.append(c)
                stream_ms += cost

        for c in selected_stream:
            stream_pending.remove(c)
            done[c] = "stream"

        if not selected_compute and not selected_stream:
            raise RuntimeError(
                "fixed pipeline made no progress; the positional compute "
                "partition is dependency-infeasible"
            )

        duration_ms = max(compute_ms, stream_ms)
        makespan_ms += duration_ms
        stages.append(
            {
                "stage": len(stages) + 1,
                "duration_ms": duration_ms,
                "compute_ms": compute_ms,
                "stream_ms": stream_ms,
                "compute": [c.__dict__ for c in selected_compute],
                "stream": [c.__dict__ for c in selected_stream],
            }
        )

    result = {
        "scheduler": scheduler_name,
        "baseline": baseline_name,
        "paper_baseline": True,
        "indexing": "zero-based-python",
        "delta_ms": delta_ms,
        "requested_delta_ms": requested_delta,
        "delta_adjusted_for_unit_feasibility": not math.isclose(
            delta_ms,
            requested_delta,
        ),
        "makespan_ms": makespan_ms,
        "chunks": len(chunks),
        "compute_chunks": sum(route == "compute" for route in done.values()),
        "stream_chunks": sum(route == "stream" for route in done.values()),
        "geometry": {
            "token_chunks": token_chunks,
            "layers": layers,
            "heads": heads,
        },
        "stages": stages,
        "assignments": {
            c.key: done[c]
            for c in chunks
        },
        "unit_costs": unit_costs,
        "source_sparkv_schedule": str(sparkv_schedule_path),
        **partition_metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_local_sparse_schedule(
    *,
    sparkv_schedule_path: Path,
    output: Path,
) -> dict[str, Any]:
    source = _load_json(sparkv_schedule_path)
    _, token_chunks, _, _ = _schedule_geometry(source["unit_costs"])
    return _build_fixed_pipeline_schedule(
        sparkv_schedule_path=sparkv_schedule_path,
        output=output,
        compute_tokens=token_chunks,
        scheduler_name="local-prefill-sparge-control",
        baseline_name="Local Prefill",
        partition_metadata={
            "fixed_partition": "all token chunks computed locally",
            "same_spargeattention_as_sparkv": True,
        },
    )


def build_strong_hybrid_schedule(
    *,
    sparkv_schedule_path: Path,
    output: Path,
    compute_fraction: float = 0.5,
) -> dict[str, Any]:
    if not 0.0 < compute_fraction < 1.0:
        raise ValueError("compute_fraction must be in (0, 1)")
    source = _load_json(sparkv_schedule_path)
    _, token_chunks, _, _ = _schedule_geometry(source["unit_costs"])
    compute_tokens = max(
        1,
        min(
            token_chunks,
            int(math.ceil(token_chunks * compute_fraction)),
        ),
    )
    return _build_fixed_pipeline_schedule(
        sparkv_schedule_path=sparkv_schedule_path,
        output=output,
        compute_tokens=compute_tokens,
        scheduler_name="strong-hybrid-fixed-positional",
        baseline_name="Strong Hybrid",
        partition_metadata={
            "fixed_partition": "early compute / later stream",
            "compute_fraction": float(compute_fraction),
            "compute_token_chunks": compute_tokens,
            "partition_parameter_paper_disclosed": False,
            "same_codec_as_sparkv": True,
            "same_spargeattention_as_sparkv": True,
        },
    )


def _warm_model(
    model: Any,
    runtime: Any,
) -> None:
    warm = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
        device=runtime.device,
    )
    with torch.inference_mode():
        model(
            input_ids=warm,
            use_cache=True,
            logits_to_keep=1,
        )
    device_synchronize(
        runtime
    )
    del warm
    clear_device_cache(
        runtime
    )


def _quality(
    *,
    model: Any,
    tokenizer: Any,
    first_token: int,
    decode_cache: Any,
    record: dict[str, Any],
    quality_tokens: int,
    runtime: Any,
) -> tuple[str, float]:
    generated = continue_greedy(
        model,
        first_token,
        decode_cache,
        max_new_tokens=max(
            1,
            quality_tokens,
        ),
        eos=tokenizer.eos_token_id,
        runtime=runtime,
    )

    prediction = (
        tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )
        .strip()
    )
    f1 = qa_f1(
        prediction,
        record[
            "answers"
        ],
    )
    return (
        prediction,
        float(f1),
    )


def _local_first_token(
    *,
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    runtime: Any,
) -> tuple[
    int,
    Any,
]:
    logical_ids = [
        int(x)
        for x in record[
            "prefill_ids"
        ]
    ]
    logical_ids.append(
        int(
            record[
                "seed_id"
            ]
        )
    )

    input_ids = torch.tensor(
        [logical_ids],
        dtype=torch.long,
        device=runtime.device,
    )
    attention_mask = (
        torch.ones_like(
            input_ids,
            dtype=torch.long,
        )
    )

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=
                attention_mask,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id
                is not None
                else tokenizer.eos_token_id
            ),
        )

    first_token = int(
        generated.sequences[
            0,
            -1,
        ].item()
    )
    cache = getattr(
        generated,
        "past_key_values",
        None,
    )
    if cache is None:
        raise RuntimeError(
            "local baseline generate() "
            "did not return past_key_values"
        )

    expected = len(
        logical_ids
    )
    actual = int(
        cache.get_seq_length()
    )
    if actual != expected:
        raise RuntimeError(
            "local baseline cache length "
            "mismatch: "
            f"expected={expected}, "
            f"got={actual}"
        )

    return (
        first_token,
        cache,
    )


def _zero_stats() -> dict[str, Any]:
    stats = ExecutionStats()
    data = stats.to_dict()
    data[
        "runtime_adaptation"
    ] = False
    data[
        "actual_huffman_bitstream"
    ] = False
    data[
        "executor"
    ] = "hf-full-local-prefill"
    return data


def _run_local(
    *,
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    sample_idx: int,
    repeat: int,
    quality_tokens: int,
    runtime: Any,
) -> dict[str, Any]:
    gc.collect()
    clear_device_cache(
        runtime
    )
    device_synchronize(
        runtime
    )
    torch.cuda.reset_peak_memory_stats(
        runtime.device
    )

    power = PowerSampler(
        enabled=True
    )
    power.start()

    begin = time.perf_counter()
    (
        first_token,
        decode_cache,
    ) = _local_first_token(
        model=model,
        tokenizer=tokenizer,
        record=record,
        runtime=runtime,
    )
    device_synchronize(
        runtime
    )
    ttft_ms = (
        time.perf_counter()
        - begin
    ) * 1000.0
    energy_j = power.stop()

    if repeat == 0:
        prediction, f1 = _quality(
            model=model,
            tokenizer=tokenizer,
            first_token=first_token,
            decode_cache=decode_cache,
            record=record,
            quality_tokens=
                quality_tokens,
            runtime=runtime,
        )
    else:
        prediction = ""
        f1 = None

    result = {
        "record_type":
            "sparkv-lab-eval-v2",
        "protocol_version":
            "paper-oriented-v2",
        "strategy":
            "local_full",
        "sample_index":
            sample_idx,
        "sample_id":
            record[
                "sample_id"
            ],
        "repeat":
            repeat,
        "ttft_ms":
            float(ttft_ms),
        "ttft_energy_j":
            energy_j,
        "prediction":
            prediction,
        "f1":
            f1,
        "peak_vram_mib":
            float(
                torch.cuda
                .max_memory_allocated(
                    runtime.device
                )
                / 2**20
            ),
        "rss_mib":
            float(
                psutil.Process()
                .memory_info()
                .rss
                / 2**20
            ),
        "runtime_device":
            str(
                runtime.device
            ),
        "runtime_dtype":
            str(
                runtime.dtype
            ).removeprefix(
                "torch."
            ),
        "runtime_backend":
            runtime.backend,
        "measurement_scope":
            (
                "full local prompt prefill "
                "+ first generated token"
            ),
        "network_mode":
            "none",
        "energy_scope":
            "NVML GPU power integrated over TTFT",
        "cloud_preload_ms":
            0.0,
    }
    result.update(
        _zero_stats()
    )

    del decode_cache
    clear_device_cache(
        runtime
    )
    return result


def _run_cache_strategy(
    *,
    strategy: str,
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    sample_idx: int,
    repeat: int,
    quality_tokens: int,
    runtime: Any,
    sample_dir: Path,
    schedule_path: Path,
    cloud_source: CloudMemorySource,
    cloud_preload_ms: float,
    bandwidth_mbps: float,
    jitter_cv: float,
    seed: int,
    runtime_window: int,
    imbalance_margin: float,
    max_migrations_per_stage: int,
) -> dict[str, Any]:
    if strategy not in {
        "all_stream",
        "local_sparse",
        "strong_hybrid",
        "sparkv",
    }:
        raise ValueError(
            f"invalid cache strategy: {strategy}"
        )

    gc.collect()
    clear_device_cache(
        runtime
    )
    device_synchronize(
        runtime
    )
    torch.cuda.reset_peak_memory_stats(
        runtime.device
    )

    if strategy != "sparkv":
        controller_config = (
            RuntimeControllerConfig(
                window=runtime_window,
                imbalance_margin=
                    imbalance_margin,
                max_migrations_per_stage=0,
            )
        )
    else:
        controller_config = (
            RuntimeControllerConfig(
                window=runtime_window,
                imbalance_margin=
                    imbalance_margin,
                max_migrations_per_stage=
                    max_migrations_per_stage,
            )
        )

    power = PowerSampler(
        enabled=True
    )
    power.start()

    begin = time.perf_counter()

    cache, exec_stats = (
        execute_sparkv(
            model=model,
            record=record,
            runtime=runtime,
            sample_dir=sample_dir,
            schedule_path=
                schedule_path,
            bandwidth_mbps=
                bandwidth_mbps,
            jitter_cv=
                jitter_cv,
            seed=seed,
            controller_config=
                controller_config,
            cloud_source=
                cloud_source,
        )
    )

    (
        first_token,
        decode_cache,
    ) = _generate_first_token(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        prefill_ids=[
            int(x)
            for x in record[
                "prefill_ids"
            ]
        ],
        seed_id=int(
            record[
                "seed_id"
            ]
        ),
        runtime=runtime,
    )

    device_synchronize(
        runtime
    )
    ttft_ms = (
        time.perf_counter()
        - begin
    ) * 1000.0
    energy_j = power.stop()

    if repeat == 0:
        prediction, f1 = _quality(
            model=model,
            tokenizer=tokenizer,
            first_token=first_token,
            decode_cache=decode_cache,
            record=record,
            quality_tokens=
                quality_tokens,
            runtime=runtime,
        )
    else:
        prediction = ""
        f1 = None

    schedule_document = _load_json(schedule_path)

    result = {
        "record_type":
            "sparkv-lab-eval-v2",
        "protocol_version":
            "paper-oriented-v2",
        "strategy":
            strategy,
        "sample_index":
            sample_idx,
        "sample_id":
            record[
                "sample_id"
            ],
        "repeat":
            repeat,
        "ttft_ms":
            float(ttft_ms),
        "ttft_energy_j":
            energy_j,
        "prediction":
            prediction,
        "f1":
            f1,
        "peak_vram_mib":
            float(
                torch.cuda
                .max_memory_allocated(
                    runtime.device
                )
                / 2**20
            ),
        "rss_mib":
            float(
                psutil.Process()
                .memory_info()
                .rss
                / 2**20
            ),
        "runtime_device":
            str(
                runtime.device
            ),
        "runtime_dtype":
            str(
                runtime.dtype
            ).removeprefix(
                "torch."
            ),
        "runtime_backend":
            runtime.backend,
        "bandwidth_mbps":
            float(
                bandwidth_mbps
            ),
        "jitter_cv":
            float(
                jitter_cv
            ),
        "runtime_window":
            runtime_window,
        "imbalance_margin":
            imbalance_margin,
        "max_migrations_per_stage":
            (
                0
                if strategy != "sparkv"
                else max_migrations_per_stage
            ),
        "schedule_path":
            str(
                schedule_path
            ),
        "schedule_scheduler":
            schedule_document.get(
                "scheduler"
            ),
        "schedule_delta_ms":
            schedule_document.get(
                "delta_ms"
            ),
        "schedule_delta_mode":
            schedule_document.get(
                "delta_selection",
                {},
            ).get(
                "mode",
                "fixed-baseline",
            ),
        "cloud_preload_ms":
            float(
                cloud_preload_ms
            ),
        "measurement_scope":
            (
                "preloaded cloud bytes; "
                "simulated wireless + Huffman "
                "decode + H2D + cache rebuild "
                "+ first generated token"
            ),
        "network_mode":
            (
                "none"
                if strategy == "local_sparse"
                else "preloaded-artifact bandwidth simulation"
            ),
        "energy_scope":
            "NVML GPU power integrated over TTFT",
    }
    result.update(
        exec_stats.to_dict()
    )
    if strategy != "sparkv":
        result[
            "runtime_adaptation"
        ] = False
    result["paper_baseline"] = strategy in {
        "local_sparse",
        "strong_hybrid",
    }
    result["paper_method"] = strategy == "sparkv"
    if strategy == "strong_hybrid":
        result["strong_hybrid_compute_fraction"] = schedule_document.get(
            "compute_fraction"
        )
        result["strong_hybrid_partition_parameter_paper_disclosed"] = False

    del (
        cache,
        decode_cache,
    )
    clear_device_cache(
        runtime
    )
    return result


def make_parser() -> (
    argparse.ArgumentParser
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=MODEL_ID,
    )
    parser.add_argument(
        "--prepared",
        required=True,
    )
    parser.add_argument(
        "--cloud-root",
        required=True,
    )
    parser.add_argument(
        "--schedule-root",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--all-stream-schedule-root",
        default=None,
    )
    parser.add_argument(
        "--local-sparse-schedule-root",
        default=None,
    )
    parser.add_argument(
        "--strong-hybrid-schedule-root",
        default=None,
    )
    parser.add_argument(
        "--strategies",
        default="local_sparse,strong_hybrid,sparkv",
        help=(
            "Comma-separated subset of local_full, local_sparse, "
            "all_stream, strong_hybrid, sparkv. The paper-aligned "
            "default excludes local_full and the all-stream control."
        ),
    )
    parser.add_argument(
        "--strong-hybrid-compute-fraction",
        type=float,
        default=0.5,
        help=(
            "Early-token compute fraction for the fixed Strong Hybrid "
            "baseline. The SparKV paper does not disclose this value."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--quality-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--bandwidth-mbps",
        type=float,
        default=640.0,
    )
    parser.add_argument(
        "--jitter-cv",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--runtime-window",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--imbalance-margin",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max-migrations-per-stage",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        choices=[
            "cuda",
        ],
        default="cuda",
    )
    parser.add_argument(
        "--cpu-dtype",
        choices=[
            "float32",
            "bfloat16",
        ],
        default="float32",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    records = _load_records(
        Path(
            args.prepared
        )
    )
    if len(records) < (
        args.samples
    ):
        raise ValueError(
            "not enough prepared samples: "
            f"{len(records)} < "
            f"{args.samples}"
        )

    model, tokenizer, runtime = (
        load_model(
            args.model,
            args.device,
            args.cpu_dtype,
        )
    )
    if not runtime.is_cuda:
        raise RuntimeError(
            "lab evaluation requires CUDA"
        )

    _warm_model(
        model,
        runtime,
    )

    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    allowed_strategies = {
        "local_full",
        "local_sparse",
        "all_stream",
        "strong_hybrid",
        "sparkv",
    }
    strategies = [
        item.strip()
        for item in args.strategies.split(",")
        if item.strip()
    ]
    if not strategies:
        raise ValueError("at least one strategy is required")
    if len(strategies) != len(set(strategies)):
        raise ValueError("strategies must not contain duplicates")
    invalid = set(strategies) - allowed_strategies
    if invalid:
        raise ValueError(f"invalid strategies: {sorted(invalid)}")

    schedule_roots = {
        "all_stream": Path(
            args.all_stream_schedule_root
            or output.parent / "all_stream_schedules"
        ),
        "local_sparse": Path(
            args.local_sparse_schedule_root
            or output.parent / "local_sparse_schedules"
        ),
        "strong_hybrid": Path(
            args.strong_hybrid_schedule_root
            or output.parent / "strong_hybrid_schedules"
        ),
    }
    for strategy, root in schedule_roots.items():
        if strategy in strategies:
            root.mkdir(parents=True, exist_ok=True)

    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for sample_idx in range(
            args.samples
        ):
            record = records[
                sample_idx
            ]
            sample_name = (
                f"sample_"
                f"{sample_idx:03d}"
            )
            sample_dir = (
                Path(
                    args.cloud_root
                )
                / sample_name
            )
            sparkv_schedule = (
                Path(
                    args.schedule_root
                )
                / (
                    sample_name
                    + ".json"
                )
            )
            if not (
                sparkv_schedule
                .is_file()
            ):
                raise FileNotFoundError(
                    sparkv_schedule
                )

            meta_path = (
                sample_dir
                / "meta.json"
            )
            meta = _load_json(
                meta_path
            )

            preload_begin = (
                time.perf_counter()
            )
            cloud_source = (
                CloudMemorySource(
                    sample_dir,
                    meta,
                )
            )
            cloud_preload_ms = (
                time.perf_counter()
                - preload_begin
            ) * 1000.0

            control_schedules: dict[str, Path] = {}
            if "all_stream" in strategies:
                path = schedule_roots["all_stream"] / f"{sample_name}.json"
                build_all_stream_schedule(
                    sparkv_schedule_path=sparkv_schedule,
                    output=path,
                )
                control_schedules["all_stream"] = path
            if "local_sparse" in strategies:
                path = schedule_roots["local_sparse"] / f"{sample_name}.json"
                build_local_sparse_schedule(
                    sparkv_schedule_path=sparkv_schedule,
                    output=path,
                )
                control_schedules["local_sparse"] = path
            if "strong_hybrid" in strategies:
                path = schedule_roots["strong_hybrid"] / f"{sample_name}.json"
                build_strong_hybrid_schedule(
                    sparkv_schedule_path=sparkv_schedule,
                    output=path,
                    compute_fraction=args.strong_hybrid_compute_fraction,
                )
                control_schedules["strong_hybrid"] = path

            for repeat in range(
                args.repeats
            ):
                # Rotate strategy order to reduce systematic thermal/order
                # bias.  Network-based strategies receive the same RNG seed
                # for a paired common-random-number comparison.
                base_order = list(strategies)
                shift = (
                    sample_idx
                    + repeat
                ) % len(
                    base_order
                )
                order = (
                    base_order[
                        shift:
                    ]
                    + base_order[
                        :shift
                    ]
                )

                paired_seed = (
                    args.seed
                    + sample_idx
                    * 1000
                    + repeat
                )

                for strategy in order:
                    if strategy == (
                        "local_full"
                    ):
                        result = (
                            _run_local(
                                model=model,
                                tokenizer=
                                    tokenizer,
                                record=record,
                                sample_idx=
                                    sample_idx,
                                repeat=repeat,
                                quality_tokens=
                                    args.quality_tokens,
                                runtime=runtime,
                            )
                        )
                        result[
                            "bandwidth_mbps"
                        ] = float(
                            args.bandwidth_mbps
                        )
                        result[
                            "jitter_cv"
                        ] = float(
                            args.jitter_cv
                        )
                    else:
                        schedule_path = (
                            sparkv_schedule
                            if strategy == "sparkv"
                            else control_schedules[strategy]
                        )
                        result = (
                            _run_cache_strategy(
                                strategy=
                                    strategy,
                                model=model,
                                tokenizer=
                                    tokenizer,
                                record=record,
                                sample_idx=
                                    sample_idx,
                                repeat=repeat,
                                quality_tokens=
                                    args.quality_tokens,
                                runtime=runtime,
                                sample_dir=
                                    sample_dir,
                                schedule_path=
                                    schedule_path,
                                cloud_source=
                                    cloud_source,
                                cloud_preload_ms=
                                    cloud_preload_ms,
                                bandwidth_mbps=
                                    args.bandwidth_mbps,
                                jitter_cv=
                                    args.jitter_cv,
                                seed=
                                    paired_seed,
                                runtime_window=
                                    args.runtime_window,
                                imbalance_margin=
                                    args.imbalance_margin,
                                max_migrations_per_stage=
                                    args.max_migrations_per_stage,
                            )
                        )

                    handle.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                    print(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        )
                    )

    print(
        json.dumps(
            {
                "saved":
                    str(output),
                "samples":
                    args.samples,
                "repeats":
                    args.repeats,
                "strategies": [
                    *strategies,
                ],
                "strong_hybrid_compute_fraction":
                    args.strong_hybrid_compute_fraction,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
