from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
)

from baseline.sparkv.runtime import (
    clear_device_cache,
    describe_runtime,
    load_model,
    to_legacy,
)
from baseline.sparkv.overhead_model import (
    ComputationLatencyPredictor,
)
from baseline.sparkv.codec import (
    encode_kv_unit,
    read_encoded_unit,
    write_encoded_unit,
)
from baseline.sparkv.sparge import (
    attention_mask_active_blocks,
    gpu_utilization_percent,
    sparse_attention_current_chunk,
)


def parse_layer_bits(
    spec: str,
    layers: int,
) -> list[int]:
    """
    Reproducible codec configuration.

    `5` -> all layers use 5-bit quantization (explicitly evaluated in the
    paper's motivation experiment).
    `5,5,4,...` -> layer-specific bit widths for experiments with the paper's
    implementation-level "layer-wise non-uniform" statement.
    """
    parts = [
        int(x.strip())
        for x in spec.split(",")
        if x.strip()
    ]

    if len(parts) == 1:
        parts = (
            parts
            * layers
        )

    if len(parts) != layers:
        raise ValueError(
            "layer bit specification must contain either one bit width "
            f"or exactly {layers} entries"
        )

    if any(
        bit < 2 or bit > 8
        for bit in parts
    ):
        raise ValueError(
            "layer bit widths must be in [2, 8]"
        )

    return parts


def _unit_relpath(
    t: int,
    layer: int,
    head: int,
) -> str:
    return (
        f"units/t{t:03d}/"
        f"l{layer:02d}_h{head:02d}.skv"
    )


def build_cloud_cache(
    *,
    records: list[dict[str, Any]],
    model_id: str,
    output_root: Path,
    samples: int,
    chunk_size: int,
    layer_bits_spec: str,
    device: str,
    cpu_dtype: str,
) -> list[dict[str, Any]]:
    model, _, runtime = load_model(
        model_id,
        device,
        cpu_dtype,
    )

    if not runtime.is_cuda:
        raise RuntimeError(
            "Direct SparKV paper reproduction requires CUDA."
        )

    results = []

    for sample_idx, record in enumerate(
        records[:samples]
    ):
        prefill_ids = record[
            "prefill_ids"
        ]
        seq_len = len(
            prefill_ids
        )

        if seq_len % chunk_size:
            raise ValueError(
                "prefill sequence length must be divisible by chunk size"
            )

        ids = torch.tensor(
            prefill_ids,
            dtype=torch.long,
            device=runtime.device,
        )[None]

        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=
                    torch.ones_like(ids),
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )

        legacy = to_legacy(
            output.past_key_values
        )

        layers = len(
            legacy
        )
        heads = int(
            legacy[0][0]
            .shape[1]
        )
        head_dim = int(
            legacy[0][0]
            .shape[-1]
        )
        token_chunks = (
            seq_len
            // chunk_size
        )

        layer_bits = (
            parse_layer_bits(
                layer_bits_spec,
                layers,
            )
        )

        sample_dir = (
            output_root
            / f"sample_{sample_idx:03d}"
        )
        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        unit_files: dict[
            str,
            dict[str, Any],
        ] = {}

        total_wire_bytes = 0

        for t in range(
            token_chunks
        ):
            start = (
                t * chunk_size
            )
            end = (
                start
                + chunk_size
            )

            for layer, (
                key,
                value,
            ) in enumerate(
                legacy
            ):
                k = (
                    key[
                        :,
                        :,
                        start:end,
                        :,
                    ]
                    .detach()
                    .cpu()
                    .contiguous()
                )
                v = (
                    value[
                        :,
                        :,
                        start:end,
                        :,
                    ]
                    .detach()
                    .cpu()
                    .contiguous()
                )

                for head in range(
                    heads
                ):
                    unit_key = (
                        f"{t}:"
                        f"{layer}:"
                        f"{head}"
                    )
                    relative = (
                        _unit_relpath(
                            t,
                            layer,
                            head,
                        )
                    )
                    path = (
                        sample_dir
                        / relative
                    )

                    encoded = (
                        encode_kv_unit(
                            k[
                                :,
                                head:
                                head + 1,
                            ],
                            v[
                                :,
                                head:
                                head + 1,
                            ],
                            bits=
                                layer_bits[
                                    layer
                                ],
                        )
                    )
                    wire_bytes = (
                        write_encoded_unit(
                            encoded,
                            path,
                        )
                    )

                    unit_files[
                        unit_key
                    ] = {
                        "path":
                            relative,
                        "wire_bytes":
                            wire_bytes,
                        "bits":
                            layer_bits[
                                layer
                            ],
                    }
                    total_wire_bytes += (
                        wire_bytes
                    )

        meta = {
            "version":
                "sparkv-cloud-v1",
            "sample_index":
                sample_idx,
            "sample_id":
                record[
                    "sample_id"
                ],
            "model":
                model_id,
            "runtime":
                describe_runtime(
                    runtime
                ),
            "seq_len":
                seq_len,
            "chunk_size":
                chunk_size,
            "token_chunks":
                token_chunks,
            "layers":
                layers,
            "kv_heads":
                heads,
            "head_dim":
                head_dim,
            "layer_bits":
                layer_bits,
            "codec":
                (
                    "symmetric-layer-bitwidth"
                    "+canonical-huffman"
                ),
            "unit_files":
                unit_files,
            "unit_count":
                len(
                    unit_files
                ),
            "wire_bytes":
                total_wire_bytes,
        }

        (
            sample_dir
            / "meta.json"
        ).write_text(
            json.dumps(
                meta,
                indent=2,
            ),
            encoding="utf-8",
        )

        results.append(
            {
                "sample_dir":
                    str(
                        sample_dir
                    ),
                "unit_count":
                    len(
                        unit_files
                    ),
                "wire_bytes":
                    total_wire_bytes,
            }
        )

        del (
            ids,
            output,
            legacy,
        )
        gc.collect()
        clear_device_cache(
            runtime
        )

    return results


def profile_stream_processing(
    *,
    sample_dir: Path,
    output: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    meta = json.loads(
        (
            sample_dir
            / "meta.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    units = meta[
        "unit_files"
    ]

    processing: dict[
        str,
        dict[str, float],
    ] = {}

    all_ms: list[
        float
    ] = []

    # Warm-up codec + H2D.
    first = next(
        iter(
            units.values()
        )
    )
    warm_path = (
        sample_dir
        / first["path"]
    )
    key, value, _ = (
        read_encoded_unit(
            warm_path,
            dtype=dtype,
        )
    )
    key = key.pin_memory()
    value = value.pin_memory()
    _ = key.to(
        device,
        non_blocking=True,
    )
    _ = value.to(
        device,
        non_blocking=True,
    )
    torch.cuda.synchronize(
        device
    )
    del key, value

    for unit_key, info in (
        units.items()
    ):
        path = (
            sample_dir
            / info["path"]
        )

        begin = (
            time.perf_counter()
        )
        key_cpu, value_cpu, _ = (
            read_encoded_unit(
                path,
                dtype=dtype,
            )
        )
        decode_ms = (
            time.perf_counter()
            - begin
        ) * 1000.0

        try:
            key_cpu = (
                key_cpu.pin_memory()
            )
            value_cpu = (
                value_cpu.pin_memory()
            )
        except RuntimeError:
            pass

        begin = (
            time.perf_counter()
        )
        key_gpu = key_cpu.to(
            device,
            non_blocking=True,
        )
        value_gpu = value_cpu.to(
            device,
            non_blocking=True,
        )
        torch.cuda.synchronize(
            device
        )
        h2d_ms = (
            time.perf_counter()
            - begin
        ) * 1000.0

        processing_ms = (
            decode_ms
            + h2d_ms
        )
        all_ms.append(
            processing_ms
        )

        processing[
            unit_key
        ] = {
            "decode_ms":
                float(
                    decode_ms
                ),
            "h2d_ms":
                float(
                    h2d_ms
                ),
            "processing_ms":
                float(
                    processing_ms
                ),
            "wire_bytes":
                int(
                    info[
                        "wire_bytes"
                    ]
                ),
        }

        del (
            key_cpu,
            value_cpu,
            key_gpu,
            value_gpu,
        )

    result = {
        "sample_dir":
            str(
                sample_dir
            ),
        "dtype":
            str(dtype)
            .removeprefix(
                "torch."
            ),
        "device":
            str(device),
        "median_processing_ms":
            float(
                statistics.median(
                    all_ms
                )
            ),
        "units":
            processing,
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


def _qkv_for_hidden(
    *,
    layer: Any,
    hidden: torch.Tensor,
    position_embeddings: tuple[
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    attn = layer.self_attn

    normalized = (
        layer.input_layernorm(
            hidden
        )
    )

    input_shape = (
        normalized.shape[:-1]
    )
    head_dim = int(
        attn.head_dim
    )

    q = (
        attn.q_proj(
            normalized
        )
        .view(
            *input_shape,
            -1,
            head_dim,
        )
        .transpose(
            1,
            2,
        )
    )
    k = (
        attn.k_proj(
            normalized
        )
        .view(
            *input_shape,
            -1,
            head_dim,
        )
        .transpose(
            1,
            2,
        )
    )
    v = (
        attn.v_proj(
            normalized
        )
        .view(
            *input_shape,
            -1,
            head_dim,
        )
        .transpose(
            1,
            2,
        )
    )

    q = attn.q_norm(
        q.transpose(
            1,
            2,
        )
    ).transpose(
        1,
        2,
    )
    k = attn.k_norm(
        k.transpose(
            1,
            2,
        )
    ).transpose(
        1,
        2,
    )

    cos, sin = (
        position_embeddings
    )

    q, k = (
        apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
        )
    )

    return (
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
    )


def _profile_dense_and_projection(
    *,
    model: Any,
    hidden_states: tuple[
        torch.Tensor,
        ...,
    ],
    chunk_size: int,
    repeats: int = 16,
) -> tuple[
    float,
    float,
]:
    layers = list(
        model.model.layers
    )

    dense_values = []
    proj_values = []

    representative_layers = (
        list(
            range(
                min(
                    len(layers),
                    8,
                )
            )
        )
    )

    for layer_idx in (
        representative_layers
    ):
        layer = layers[
            layer_idx
        ]
        hidden = hidden_states[
            layer_idx
        ][
            :,
            :chunk_size,
            :,
        ].contiguous()

        for _ in range(
            max(
                1,
                repeats
                // len(
                    representative_layers
                ),
            )
        ):
            torch.cuda.synchronize()
            start = torch.cuda.Event(
                enable_timing=True
            )
            end = torch.cuda.Event(
                enable_timing=True
            )

            start.record()

            residual = hidden
            normed = (
                layer.input_layernorm(
                    hidden
                )
            )
            # t_qkv
            _ = layer.self_attn.q_proj(
                normed
            )
            _ = layer.self_attn.k_proj(
                normed
            )
            _ = layer.self_attn.v_proj(
                normed
            )

            # t_o + residual + norm + FFN.
            #
            # Qwen3 does not require the concatenated attention width before o_proj
            # to equal hidden_size.  In Qwen3-4B, for example:
            #
            #   hidden_size             = 2560
            #   num_attention_heads     = 32
            #   head_dim                = 128
            #   o_proj input width      = 32 * 128 = 4096
            #
            # Therefore zeros_like(hidden) is invalid here.  Build a synthetic
            # attention output using the actual o_proj input geometry instead.
            attention_width = int(
                layer.self_attn.o_proj.in_features
            )

            if (
                int(
                    layer.self_attn.o_proj.out_features
                )
                != int(
                    hidden.shape[-1]
                )
            ):
                raise RuntimeError(
                    "Qwen3 o_proj output width does not match hidden size: "
                    f"o_proj.out_features="
                    f"{layer.self_attn.o_proj.out_features}, "
                    f"hidden_size={hidden.shape[-1]}"
                )

            dummy_attention = hidden.new_zeros(
                (
                    *hidden.shape[:-1],
                    attention_width,
                )
            )

            after_attn = (
                residual
                + layer.self_attn.o_proj(
                    dummy_attention
                )
            )
            residual2 = after_attn
            ffn_in = (
                layer.post_attention_layernorm(
                    after_attn
                )
            )
            after_ffn = (
                residual2
                + layer.mlp(
                    ffn_in
                )
            )

            end.record()
            end.synchronize()

            dense_values.append(
                float(
                    start.elapsed_time(
                        end
                    )
                )
            )

            del (
                normed,
                dummy_attention,
                after_attn,
                ffn_in,
                after_ffn,
            )

    final_layer = layers[
        -1
    ]
    final_hidden = hidden_states[
        -2
    ][
        :,
        :chunk_size,
        :,
    ].contiguous()

    for _ in range(
        repeats
    ):
        torch.cuda.synchronize()
        start = torch.cuda.Event(
            enable_timing=True
        )
        end = torch.cuda.Event(
            enable_timing=True
        )
        start.record()

        normed = (
            final_layer
            .input_layernorm(
                final_hidden
            )
        )
        _ = (
            final_layer
            .self_attn
            .k_proj(
                normed
            )
        )
        _ = (
            final_layer
            .self_attn
            .v_proj(
                normed
            )
        )

        end.record()
        end.synchronize()

        proj_values.append(
            float(
                start.elapsed_time(
                    end
                )
            )
        )
        del normed

    return (
        float(
            statistics.median(
                dense_values
            )
        ),
        float(
            statistics.median(
                proj_values
            )
        ),
    )


def collect_predictor_profile(
    *,
    records: list[dict[str, Any]],
    model_id: str,
    output: Path,
    chunk_size: int,
    target_records: int,
    device: str,
    cpu_dtype: str,
) -> dict[str, Any]:
    model, _, runtime = load_model(
        model_id,
        device,
        cpu_dtype,
    )
    if not runtime.is_cuda:
        raise RuntimeError(
            "predictor profiling requires CUDA"
        )

    model.config.output_hidden_states = (
        True
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0
    dense_samples = []
    projection_samples = []

    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for sample_idx, record in enumerate(
            records
        ):
            if count >= target_records:
                break

            ids = torch.tensor(
                record[
                    "prefill_ids"
                ],
                dtype=torch.long,
                device=runtime.device,
            )[None]
            seq_len = int(
                ids.shape[1]
            )

            if seq_len % chunk_size:
                continue

            with torch.inference_mode():
                full = model(
                    input_ids=ids,
                    attention_mask=
                        torch.ones_like(
                            ids
                        ),
                    use_cache=False,
                    output_hidden_states=
                        True,
                    logits_to_keep=1,
                    return_dict=True,
                )

            hidden_states = (
                full.hidden_states
            )
            layers = list(
                model.model.layers
            )

            dense_ms, proj_ms = (
                _profile_dense_and_projection(
                    model=model,
                    hidden_states=
                        hidden_states,
                    chunk_size=
                        chunk_size,
                )
            )
            dense_samples.append(
                dense_ms
            )
            projection_samples.append(
                proj_ms
            )

            position_ids = (
                torch.arange(
                    seq_len,
                    dtype=torch.long,
                    device=runtime.device,
                )[None]
            )
            token_chunks = (
                seq_len
                // chunk_size
            )

            for layer_idx, layer in enumerate(
                layers[:-1]
            ):
                if count >= target_records:
                    break

                hidden = hidden_states[
                    layer_idx
                ]

                position_embeddings = (
                    model.model.rotary_emb(
                        hidden,
                        position_ids,
                    )
                )

                with torch.inference_mode():
                    q, k, v = (
                        _qkv_for_hidden(
                            layer=layer,
                            hidden=hidden,
                            position_embeddings=
                                position_embeddings,
                        )
                    )

                groups = int(
                    layer.self_attn
                    .num_key_value_groups
                )
                kv_heads = int(
                    k.shape[1]
                )

                for t in range(
                    token_chunks
                ):
                    if count >= target_records:
                        break

                    start = (
                        t
                        * chunk_size
                    )
                    end = (
                        start
                        + chunk_size
                    )

                    for head in range(
                        kv_heads
                    ):
                        if count >= target_records:
                            break

                        q_start = (
                            head
                            * groups
                        )
                        q_end = (
                            q_start
                            + groups
                        )

                        q_current = (
                            q[
                                :,
                                q_start:
                                q_end,
                                start:end,
                                :,
                            ]
                            .contiguous()
                        )
                        k_prefix = (
                            k[
                                :,
                                head:
                                head + 1,
                                :end,
                                :,
                            ]
                            .contiguous()
                        )
                        v_prefix = (
                            v[
                                :,
                                head:
                                head + 1,
                                :end,
                                :,
                            ]
                            .contiguous()
                        )

                        gpu_util = (
                            gpu_utilization_percent(
                                runtime.device.index
                                or 0
                            )
                        )

                        begin = (
                            time.perf_counter()
                        )

                        result = (
                            sparse_attention_current_chunk(
                                query_current=
                                    q_current,
                                key_history_and_current=
                                    k_prefix,
                                value_history_and_current=
                                    v_prefix,
                                current_chunk_tokens=
                                    chunk_size,
                                num_key_value_groups=
                                    groups,
                                scale=float(
                                    layer.self_attn.scaling
                                ),
                            )
                        )

                        wall_ms = (
                            time.perf_counter()
                            - begin
                        ) * 1000.0

                        row = {
                            "sample_index":
                                sample_idx,
                            "token_block":
                                t + 1,
                            "layer":
                                layer_idx,
                            "head":
                                head,
                            "active_blocks":
                                result.active_blocks,
                            "sparsity":
                                result.sparsity,
                            "gpu_util":
                                gpu_util,
                            # Include mask construction + sparse CUDA call
                            # because both are part of the actual attention
                            # path at runtime.
                            "attention_ms":
                                float(
                                    wall_ms
                                ),
                            "kernel_ms":
                                float(
                                    result.elapsed_ms
                                ),
                        }

                        handle.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        handle.flush()
                        count += 1

                        del (
                            q_current,
                            k_prefix,
                            v_prefix,
                            result,
                        )

                del (
                    q,
                    k,
                    v,
                    position_embeddings,
                )
                clear_device_cache(
                    runtime
                )

            del (
                ids,
                full,
                hidden_states,
            )
            gc.collect()
            clear_device_cache(
                runtime
            )

    if count < target_records:
        raise RuntimeError(
            "not enough profiling records were generated: "
            f"{count} < {target_records}"
        )

    sidecar = {
        "records":
            count,
        "dense_ms":
            float(
                statistics.median(
                    dense_samples
                )
            ),
        "final_projection_ms":
            float(
                statistics.median(
                    projection_samples
                )
            ),
        "chunk_size":
            chunk_size,
        "model":
            model_id,
    }

    output.with_suffix(
        ".meta.json"
    ).write_text(
        json.dumps(
            sidecar,
            indent=2,
        ),
        encoding="utf-8",
    )

    return sidecar


def build_scheduler_profile(
    *,
    record: dict[str, Any],
    sample_dir: Path,
    stream_profile_path: Path,
    predictor_path: Path,
    model_id: str,
    output: Path,
    chunk_size: int,
    device: str,
    cpu_dtype: str,
) -> dict[str, Any]:
    model, _, runtime = load_model(
        model_id,
        device,
        cpu_dtype,
    )
    if not runtime.is_cuda:
        raise RuntimeError(
            "scheduler profiling requires CUDA"
        )

    meta = json.loads(
        (
            sample_dir
            / "meta.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    stream_profile = json.loads(
        stream_profile_path.read_text(
            encoding="utf-8"
        )
    )

    predictor = (
        ComputationLatencyPredictor(
            predictor_path
        )
    )

    ids = torch.tensor(
        record[
            "prefill_ids"
        ],
        dtype=torch.long,
        device=runtime.device,
    )[None]
    seq_len = int(
        ids.shape[1]
    )

    with torch.inference_mode():
        full = model(
            input_ids=ids,
            attention_mask=
                torch.ones_like(ids),
            use_cache=False,
            output_hidden_states=
                True,
            logits_to_keep=1,
            return_dict=True,
        )

    hidden_states = (
        full.hidden_states
    )
    layers = list(
        model.model.layers
    )
    position_ids = (
        torch.arange(
            seq_len,
            dtype=torch.long,
            device=runtime.device,
        )[None]
    )

    token_chunks = int(
        meta[
            "token_chunks"
        ]
    )
    kv_heads = int(
        meta[
            "kv_heads"
        ]
    )

    units: dict[
        str,
        dict[str, Any],
    ] = {}

    # Scheduler is cloud-side/offline.  Use a snapshot of the edge load as
    # the initial estimate; the runtime controller corrects transient drift.
    initial_gpu_util = (
        gpu_utilization_percent(
            runtime.device.index
            or 0
        )
    )

    for layer_idx, layer in enumerate(
        layers
    ):
        hidden = hidden_states[
            layer_idx
        ]

        if (
            layer_idx
            == len(layers) - 1
        ):
            for t in range(
                token_chunks
            ):
                for head in range(
                    kv_heads
                ):
                    key = (
                        f"{t}:"
                        f"{layer_idx}:"
                        f"{head}"
                    )
                    units[key] = {
                        "wire_bytes":
                            int(
                                meta[
                                    "unit_files"
                                ][key][
                                    "wire_bytes"
                                ]
                            ),
                        "processing_ms":
                            float(
                                stream_profile[
                                    "units"
                                ][key][
                                    "processing_ms"
                                ]
                            ),
                        "active_blocks":
                            0,
                        "sparsity":
                            1.0,
                        "initial_gpu_util":
                            initial_gpu_util,
                        "predicted_comp_ms":
                            predictor.chunk_ms(
                                token_block=
                                    t + 1,
                                active_blocks=
                                    0,
                                gpu_util=
                                    initial_gpu_util,
                                final_layer=
                                    True,
                            ),
                    }
            continue

        position_embeddings = (
            model.model.rotary_emb(
                hidden,
                position_ids,
            )
        )

        with torch.inference_mode():
            q, k, _ = (
                _qkv_for_hidden(
                    layer=layer,
                    hidden=hidden,
                    position_embeddings=
                        position_embeddings,
                )
            )

        groups = int(
            layer.self_attn
            .num_key_value_groups
        )

        for t in range(
            token_chunks
        ):
            start = (
                t
                * chunk_size
            )
            end = (
                start
                + chunk_size
            )

            for head in range(
                kv_heads
            ):
                q_start = (
                    head
                    * groups
                )
                q_end = (
                    q_start
                    + groups
                )

                active, sparsity = (
                    attention_mask_active_blocks(
                        query_current=
                            q[
                                :,
                                q_start:
                                q_end,
                                start:end,
                                :,
                            ].contiguous(),
                        key_history_and_current=
                            k[
                                :,
                                head:
                                head + 1,
                                :end,
                                :,
                            ].contiguous(),
                        current_chunk_tokens=
                            chunk_size,
                        num_key_value_groups=
                            groups,
                    )
                )

                key = (
                    f"{t}:"
                    f"{layer_idx}:"
                    f"{head}"
                )

                units[key] = {
                    "wire_bytes":
                        int(
                            meta[
                                "unit_files"
                            ][key][
                                "wire_bytes"
                            ]
                        ),
                    "processing_ms":
                        float(
                            stream_profile[
                                "units"
                            ][key][
                                "processing_ms"
                            ]
                        ),
                    "active_blocks":
                        int(active),
                    "sparsity":
                        float(
                            sparsity
                        ),
                    "initial_gpu_util":
                        initial_gpu_util,
                    "predicted_comp_ms":
                        predictor.chunk_ms(
                            token_block=
                                t + 1,
                            active_blocks=
                                active,
                            gpu_util=
                                initial_gpu_util,
                            final_layer=
                                False,
                        ),
                }

        del (
            q,
            k,
            position_embeddings,
        )
        clear_device_cache(
            runtime
        )

    profile = {
        "version":
            "sparkv-scheduler-profile-v1",
        "sample_dir":
            str(
                sample_dir
            ),
        "model":
            model_id,
        "predictor":
            str(
                predictor_path
            ),
        "geometry": {
            "token_chunks":
                token_chunks,
            "layers":
                len(
                    layers
                ),
            "heads":
                kv_heads,
            "chunk_size":
                chunk_size,
        },
        "units":
            units,
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            profile,
            indent=2,
        ),
        encoding="utf-8",
    )

    del (
        ids,
        full,
        hidden_states,
    )
    clear_device_cache(
        runtime
    )

    return profile
