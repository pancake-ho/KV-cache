from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel

from .rope import model_rope_cos_sin, remove_rope
from .store import ActivationMetadata, ActivationStore


def extract_activations(
    *,
    model_path: str,
    tokens_path: str | Path,
    output_dir: str | Path,
    role: str,
    stride: int,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    max_sequences: int | None = None,
    attn_implementation: str = "sdpa",
) -> ActivationMetadata:
    tokens = np.load(tokens_path, mmap_mode="r")
    if tokens.ndim != 2:
        raise ValueError(f"tokens must have [sequences,length], got {tokens.shape}")
    num_sequences = min(tokens.shape[0], max_sequences or tokens.shape[0])
    sequence_length = int(tokens.shape[1])
    sampled_positions = np.arange(0, sequence_length, stride, dtype=np.int64)
    config = AutoConfig.from_pretrained(model_path)
    metadata = ActivationMetadata(
        role=role,
        model_path=str(model_path),
        model_type=config.model_type,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        ),
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        stride=stride,
        num_observations=num_sequences * sampled_positions.size,
    )
    metadata.save(output_dir)
    store = ActivationStore(output_dir)
    writers = {
        (kind, layer): store.create(kind, layer)
        for kind in ("k", "v")
        for layer in range(config.num_hidden_layers)
    }

    model = AutoModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval()
    input_device = model.embed_tokens.weight.device
    position_tensor = torch.from_numpy(sampled_positions)
    full_positions = torch.arange(sequence_length)
    cos, sin = model_rope_cos_sin(
        model,
        full_positions,
        device=input_device,
        dtype=torch.float32,
    )
    cos = cos.index_select(0, position_tensor.to(cos.device))
    sin = sin.index_select(0, position_tensor.to(sin.device))

    samples_per_sequence = sampled_positions.size
    for sequence_index in tqdm(range(num_sequences), desc=f"extract {role}"):
        input_ids = torch.from_numpy(np.asarray(tokens[sequence_index], dtype=np.int64)).unsqueeze(0).to(input_device)
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        legacy = output.past_key_values.to_legacy_cache()
        begin = sequence_index * samples_per_sequence
        end = begin + samples_per_sequence
        for layer, (key, value) in enumerate(legacy):
            index = position_tensor.to(key.device)
            sampled_key = key.index_select(2, index).to(dtype=torch.float32)
            sampled_key = remove_rope(
                sampled_key,
                cos.to(key.device),
                sin.to(key.device),
                sequence_dim=2,
            )
            sampled_value = value.index_select(2, index)
            writers[("k", layer)][begin:end] = (
                sampled_key[0].permute(1, 0, 2).to(dtype=torch.float16).cpu().numpy()
            )
            writers[("v", layer)][begin:end] = (
                sampled_value[0].permute(1, 0, 2).to(dtype=torch.float16).cpu().numpy()
            )
        del output, legacy, input_ids

    for writer in writers.values():
        writer.flush()
    del writers, model
    gc.collect()
    torch.cuda.empty_cache()
    return metadata
