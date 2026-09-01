from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from baseline.sparkv.runtime import (
    MODEL_ID,
    PowerSampler,
    clear_device_cache,
    continue_greedy,
    device_synchronize,
    load_model,
    prepare_command,
    qa_f1,
    seed_everything,
)
from baseline.sparkv.overhead_model import (
    PredictorTrainConfig,
    train_predictor,
)
from baseline.sparkv.artifacts import (
    build_cloud_cache,
    build_scheduler_profile,
    collect_predictor_profile,
    profile_stream_processing,
)
from baseline.sparkv.executor import (
    execute_sparkv,
)
from baseline.sparkv.runtime_controller import (
    RuntimeControllerConfig,
)


def _load_records(
    path: str,
) -> list[dict[str, Any]]:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def build_cloud_command(
    args: argparse.Namespace,
) -> None:
    records = _load_records(
        args.prepared
    )

    result = build_cloud_cache(
        records=records,
        model_id=args.model,
        output_root=Path(
            args.output_root
        ),
        samples=args.samples,
        chunk_size=args.chunk_size,
        layer_bits_spec=(
            args.layer_bits
        ),
        device=args.device,
        cpu_dtype=args.cpu_dtype,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


def profile_stream_command(
    args: argparse.Namespace,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    device = torch.device(
        "cuda:0"
    )

    result = (
        profile_stream_processing(
            sample_dir=Path(
                args.sample_dir
            ),
            output=Path(
                args.output
            ),
            dtype=(
                torch.bfloat16
                if args.dtype
                == "bfloat16"
                else torch.float16
            ),
            device=device,
        )
    )

    print(
        json.dumps(
            {
                "saved":
                    args.output,
                "median_processing_ms":
                    result[
                        "median_processing_ms"
                    ],
                "units":
                    len(
                        result[
                            "units"
                        ]
                    ),
            },
            indent=2,
        )
    )


def collect_overhead_command(
    args: argparse.Namespace,
) -> None:
    records = _load_records(
        args.prepared
    )

    result = (
        collect_predictor_profile(
            records=records,
            model_id=args.model,
            output=Path(
                args.output
            ),
            chunk_size=(
                args.chunk_size
            ),
            target_records=(
                args.target_records
            ),
            device=args.device,
            cpu_dtype=args.cpu_dtype,
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


def train_predictor_command(
    args: argparse.Namespace,
) -> None:
    records = []

    for profile in (
        args.profiles
    ):
        with Path(
            profile
        ).open(
            encoding="utf-8"
        ) as handle:
            records.extend(
                json.loads(
                    line
                )
                for line in handle
                if line.strip()
            )

    meta = json.loads(
        Path(
            args.profile_meta
        ).read_text(
            encoding="utf-8"
        )
    )

    result = train_predictor(
        records=records,
        output=Path(
            args.output
        ),
        config=(
            PredictorTrainConfig(
                seed=args.seed,
                samples=6000,
                epochs=args.epochs,
                learning_rate=(
                    args.learning_rate
                ),
                momentum=(
                    args.momentum
                ),
            )
        ),
        dense_ms=float(
            meta[
                "dense_ms"
            ]
        ),
        final_projection_ms=float(
            meta[
                "final_projection_ms"
            ]
        ),
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


def scheduler_profile_command(
    args: argparse.Namespace,
) -> None:
    records = _load_records(
        args.prepared
    )
    record = records[
        args.sample_index
    ]

    result = (
        build_scheduler_profile(
            record=record,
            sample_dir=Path(
                args.sample_dir
            ),
            stream_profile_path=Path(
                args.stream_profile
            ),
            predictor_path=Path(
                args.predictor
            ),
            model_id=args.model,
            output=Path(
                args.output
            ),
            chunk_size=(
                args.chunk_size
            ),
            device=args.device,
            cpu_dtype=args.cpu_dtype,
        )
    )

    print(
        json.dumps(
            {
                "saved":
                    args.output,
                "units":
                    len(
                        result[
                            "units"
                        ]
                    ),
            },
            indent=2,
        )
    )


def _generate_first_token(
    *,
    model: Any,
    tokenizer: Any,
    cache: Any,
    seed_id: int,
    runtime: Any,
) -> tuple[
    int,
    Any,
]:
    past_len = int(
        cache.get_seq_length()
    )

    current = torch.tensor(
        [[seed_id]],
        dtype=torch.long,
        device=runtime.device,
    )

    attention_mask = torch.ones(
        (
            1,
            past_len + 1,
        ),
        dtype=torch.long,
        device=runtime.device,
    )

    with torch.inference_mode():
        generated = model.generate(
            input_ids=current,
            attention_mask=(
                attention_mask
            ),
            past_key_values=cache,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=(
                tokenizer.pad_token_id
                if (
                    tokenizer
                    .pad_token_id
                    is not None
                )
                else (
                    tokenizer
                    .eos_token_id
                )
            ),
        )

    first_token = int(
        generated.sequences[
            0,
            -1,
        ].item()
    )

    decode_cache = getattr(
        generated,
        "past_key_values",
        None,
    )

    if decode_cache is None:
        raise RuntimeError(
            "transformers generate() did not "
            "return past_key_values; the "
            "SparKV integration requires "
            "cache injection into generate()."
        )

    return (
        first_token,
        decode_cache,
    )


def run_command(
    args: argparse.Namespace,
) -> None:
    records = _load_records(
        args.prepared
    )

    if len(records) < (
        args.samples
    ):
        raise ValueError(
            "prepared sample count is "
            f"{len(records)}, but "
            f"--samples={args.samples}"
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
            "SparKV paper run must not "
            "fall back to CPU."
        )

    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Warm-up is outside the TTFT measurement.
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

            sample_dir = (
                Path(
                    args.cloud_root
                )
                / (
                    f"sample_"
                    f"{sample_idx:03d}"
                )
            )

            schedule_path = (
                Path(
                    args.schedule_root
                )
                / (
                    f"sample_"
                    f"{sample_idx:03d}.json"
                )
            )

            if not (
                sample_dir
                / "meta.json"
            ).is_file():
                raise FileNotFoundError(
                    "missing cloud metadata: "
                    f"{sample_dir / 'meta.json'}"
                )

            if not (
                schedule_path
                .is_file()
            ):
                raise FileNotFoundError(
                    "missing schedule: "
                    f"{schedule_path}"
                )

            for repeat in range(
                args.repeats
            ):
                torch.cuda.reset_peak_memory_stats(
                    runtime.device
                )

                power = PowerSampler(
                    enabled=True
                )
                power.start()

                request_begin = (
                    time.perf_counter()
                )

                cache, exec_stats = (
                    execute_sparkv(
                        model=model,
                        record=record,
                        runtime=runtime,
                        sample_dir=(
                            sample_dir
                        ),
                        schedule_path=(
                            schedule_path
                        ),
                        bandwidth_mbps=(
                            args.bandwidth_mbps
                        ),
                        jitter_cv=(
                            args.jitter_cv
                        ),
                        seed=(
                            args.seed
                            + sample_idx
                            * 1000
                            + repeat
                        ),
                        controller_config=(
                            RuntimeControllerConfig(
                                window=(
                                    args.runtime_window
                                ),
                                imbalance_margin=(
                                    args.imbalance_margin
                                ),
                                max_migrations_per_stage=(
                                    args.max_migrations_per_stage
                                ),
                            )
                        ),
                    )
                )

                (
                    first_token,
                    decode_cache,
                ) = _generate_first_token(
                    model=model,
                    tokenizer=tokenizer,
                    cache=cache,
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
                    (
                        time.perf_counter()
                        - request_begin
                    )
                    * 1000.0
                )

                ttft_energy_j = (
                    power.stop()
                )

                if repeat == 0:
                    generated = (
                        continue_greedy(
                            model,
                            first_token,
                            decode_cache,
                            max_new_tokens=max(
                                1,
                                args.quality_tokens,
                            ),
                            eos=(
                                tokenizer
                                .eos_token_id
                            ),
                            runtime=runtime,
                        )
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

                else:
                    prediction = ""
                    f1 = None

                result = {
                    "record_type":
                        "sparkv-direct",
                    "strategy":
                        "sparkv",
                    "paper_algorithm":
                        (
                            "potential-aware-greedy"
                            "+mlp"
                            "+runtime-controller"
                        ),
                    "sample_index":
                        sample_idx,
                    "sample_id":
                        record[
                            "sample_id"
                        ],
                    "repeat":
                        repeat,
                    "ttft_ms":
                        float(
                            ttft_ms
                        ),
                    "ttft_energy_j":
                        ttft_energy_j,
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
                        args.bandwidth_mbps,
                    "jitter_cv":
                        args.jitter_cv,
                    "runtime_window":
                        args.runtime_window,
                    "imbalance_margin":
                        args.imbalance_margin,
                    "max_migrations_per_stage":
                        (
                            args
                            .max_migrations_per_stage
                        ),
                    "schedule_path":
                        str(
                            schedule_path
                        ),
                }

                result.update(
                    exec_stats.to_dict()
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

                del (
                    cache,
                    decode_cache,
                )

                clear_device_cache(
                    runtime
                )


def add_runtime_args(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--device",
        choices=[
            "cuda",
            "auto",
            "cpu",
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


def make_parser() -> argparse.ArgumentParser:
    parser = (
        argparse.ArgumentParser()
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    p = sub.add_parser(
        "prepare"
    )
    p.add_argument(
        "--model",
        default=MODEL_ID,
    )
    p.add_argument(
        "--samples",
        type=int,
        default=4,
    )
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=8193,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.set_defaults(
        func=prepare_command
    )

    p = sub.add_parser(
        "build-cloud"
    )
    p.add_argument(
        "--model",
        default=MODEL_ID,
    )
    p.add_argument(
        "--prepared",
        required=True,
    )
    p.add_argument(
        "--output-root",
        required=True,
    )
    p.add_argument(
        "--samples",
        type=int,
        default=1,
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--layer-bits",
        default="5",
    )
    add_runtime_args(p)
    p.set_defaults(
        func=build_cloud_command
    )

    p = sub.add_parser(
        "profile-stream"
    )
    p.add_argument(
        "--sample-dir",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--dtype",
        choices=[
            "bfloat16",
            "float16",
        ],
        default="bfloat16",
    )
    p.set_defaults(
        func=profile_stream_command
    )

    p = sub.add_parser(
        "collect-overhead"
    )
    p.add_argument(
        "--model",
        default=MODEL_ID,
    )
    p.add_argument(
        "--prepared",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--target-records",
        type=int,
        default=6000,
    )
    add_runtime_args(p)
    p.set_defaults(
        func=collect_overhead_command
    )

    p = sub.add_parser(
        "train-predictor"
    )
    p.add_argument(
        "--profiles",
        nargs="+",
        required=True,
    )
    p.add_argument(
        "--profile-meta",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=400,
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
    )
    p.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    p.set_defaults(
        func=train_predictor_command
    )

    p = sub.add_parser(
        "scheduler-profile"
    )
    p.add_argument(
        "--model",
        default=MODEL_ID,
    )
    p.add_argument(
        "--prepared",
        required=True,
    )
    p.add_argument(
        "--sample-index",
        type=int,
        required=True,
    )
    p.add_argument(
        "--sample-dir",
        required=True,
    )
    p.add_argument(
        "--stream-profile",
        required=True,
    )
    p.add_argument(
        "--predictor",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
    )
    add_runtime_args(p)
    p.set_defaults(
        func=scheduler_profile_command
    )

    p = sub.add_parser(
        "run"
    )
    p.add_argument(
        "--model",
        default=MODEL_ID,
    )
    p.add_argument(
        "--prepared",
        required=True,
    )
    p.add_argument(
        "--cloud-root",
        required=True,
    )
    p.add_argument(
        "--schedule-root",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--samples",
        type=int,
        default=1,
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
    )
    p.add_argument(
        "--quality-tokens",
        type=int,
        default=32,
    )
    p.add_argument(
        "--bandwidth-mbps",
        type=float,
        default=640.0,
    )
    p.add_argument(
        "--jitter-cv",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    # Paper does not disclose these numerical controller values.
    p.add_argument(
        "--runtime-window",
        type=int,
        default=4,
    )
    p.add_argument(
        "--imbalance-margin",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--max-migrations-per-stage",
        type=int,
        default=4,
    )

    add_runtime_args(p)
    p.set_defaults(
        func=run_command
    )

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    seed_everything(
        getattr(
            args,
            "seed",
            2026,
        )
    )

    args.func(
        args
    )
