from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import numpy as np

from ..rope import apply_rope, model_rope_cos_sin, remove_rope
from .common import KINDS, digest, lock, provenance, read_json, sha256, write_json


def cache_pairs(cache):
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple(cache[layer] for layer in range(len(cache)))


def extract_with_model(model, tokens_path, root, *, role, prepared, device):
    """Also used by the offline tiny-model integration test; production requires CUDA."""
    import torch
    root = Path(root)
    tokens = np.load(tokens_path, mmap_mode="r")
    cfg = prepared["config"]
    positions = np.arange(0, tokens.shape[1], cfg["stride"], dtype=np.int64)
    count = len(positions)
    mc = model.config
    shape = (len(tokens) * count, mc.num_key_value_heads, mc.head_dim)
    identity = {
        "schema_version": 1, "role": role, "config": cfg,
        "model": prepared["models"][role], "prepare_digest": prepared["prepare_digest"],
        "tokens_sha256": sha256(tokens_path), "num_layers": mc.num_hidden_layers,
        "num_kv_heads": mc.num_key_value_heads, "head_dim": mc.head_dim,
        "num_observations": shape[0], "sequence_length": int(tokens.shape[1]),
        "stride": cfg["stride"], "train_observations": prepared["train_observations"],
        "heldout_observations": prepared["heldout_observations"],
        "kinds": list(KINDS), "storage_dtype": "float16",
    }
    if identity["tokens_sha256"] != prepared["tokens_sha256"]:
        raise ValueError("token file digest does not match prepared data")
    identity_hash = digest(identity)
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / ".extract.lock"):
        metadata_path = root / "metadata.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata["identity_hash"] != identity_hash:
                raise ValueError("existing activation store has different model/data/settings")
            if metadata.get("complete"):
                from .store import ProbeStore
                existing = ProbeStore(root)
                for kind in KINDS:
                    for layer in range(mc.num_hidden_layers):
                        existing.open(kind, layer)
                print(f"verified complete {role} extraction", flush=True)
                return metadata
            start = metadata["completed_sequences"]
            file_mode = "r+"
        else:
            required = np.prod(shape) * 2 * mc.num_hidden_layers * len(KINDS)
            free = shutil.disk_usage(root).free
            if free < required + 1024**3:
                raise RuntimeError(f"not enough disk: extraction needs {required / 1024**3:.1f} GiB + reserve")
            start, file_mode = 0, "w+"
            metadata = {**identity, "identity_hash": identity_hash, "complete": False,
                        "completed_sequences": 0, "provenance": provenance()}
        writers = {}
        for kind in KINDS:
            folder = root / kind
            folder.mkdir(exist_ok=True)
            for layer in range(mc.num_hidden_layers):
                path = folder / f"layer_{layer:03d}.npy"
                if file_mode == "w+":
                    writer = np.lib.format.open_memmap(path, mode=file_mode, dtype="float16", shape=shape)
                else:
                    writer = np.load(path, mmap_mode="r+")
                if writer.shape != shape or writer.dtype != np.dtype("float16"):
                    raise ValueError(f"invalid partial extraction file: {path}")
                writers[kind, layer] = writer
        write_json(metadata_path, metadata)
        model.eval()
        index = torch.tensor(positions, device=device, dtype=torch.long)
        full_positions = torch.arange(tokens.shape[1], device=device)
        with torch.inference_mode():
            cos, sin = model_rope_cos_sin(model, full_positions, device=device, dtype=torch.float32)
            cos, sin = cos.index_select(0, index), sin.index_select(0, index)
            max_roundtrip_error = metadata.get("rope_roundtrip_max_abs", 0.0)
            for sequence in range(start, len(tokens)):
                ids = torch.tensor(np.array(tokens[sequence], copy=True), dtype=torch.long, device=device)[None]
                output = model(input_ids=ids, use_cache=True, return_dict=True)
                pairs = cache_pairs(output.past_key_values)
                if len(pairs) != mc.num_hidden_layers:
                    raise ValueError("unexpected number of cache layers")
                for layer, pair in enumerate(pairs):
                    key, value = pair[:2]
                    expected = (1, mc.num_key_value_heads, tokens.shape[1], mc.head_dim)
                    if tuple(key.shape) != expected or tuple(value.shape) != expected:
                        raise ValueError(f"unexpected cache layout at layer {layer}: {key.shape}/{value.shape}")
                    raw_key = key.index_select(2, index).float()
                    stripped = remove_rope(raw_key, cos, sin, sequence_dim=2)
                    if sequence == start:
                        roundtrip = apply_rope(stripped, cos, sin, sequence_dim=2)
                        error = float((roundtrip - raw_key).abs().max())
                        if error > 2e-5 * max(1.0, float(raw_key.abs().max())):
                            raise ValueError("inverse RoPE roundtrip check failed")
                        max_roundtrip_error = max(max_roundtrip_error, error)
                    sampled = (raw_key, stripped, value.index_select(2, index))
                    begin, end = sequence * count, (sequence + 1) * count
                    for kind, tensor in zip(KINDS, sampled):
                        array = tensor[0].permute(1, 0, 2).to(torch.float16).cpu().numpy()
                        if not np.isfinite(array).all():
                            raise ValueError(f"NaN/Inf in {role}, {kind}, layer {layer}")
                        writers[kind, layer][begin:end] = array
                del output, pairs, ids, sampled, raw_key, stripped, key, value, tensor, array
                if (sequence + 1) % cfg["flush_sequences"] == 0 or sequence + 1 == len(tokens):
                    for writer in writers.values():
                        writer.flush()
                    metadata.update(completed_sequences=sequence + 1, rope_roundtrip_max_abs=max_roundtrip_error)
                    write_json(metadata_path, metadata)
                    print(f"{role}: {sequence + 1}/{len(tokens)} sequences persisted", flush=True)
        metadata["complete"] = True
        if str(device).startswith("cuda"):
            metadata.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(metadata_path, metadata)
        return metadata


def main():
    import torch
    from transformers import AutoModel
    parser = argparse.ArgumentParser(description="Extract raw K, inverse-RoPE K and V sequentially")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--role", choices=("source", "target"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; CPU fallback is disabled")
    root = Path(args.run_dir)
    prepared = read_json(root / "prepare.json")
    cfg = prepared["config"]
    dtype = getattr(torch, cfg["forward_dtype"])
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("configured bfloat16 forward is unsupported by this GPU")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.cuda.reset_peak_memory_stats(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    info = prepared["models"][args.role]
    model = AutoModel.from_pretrained(info["model_id"], revision=info["revision"],
                                     torch_dtype=dtype, device_map="cuda:0",
                                     attn_implementation="sdpa", low_cpu_mem_usage=True).eval()
    expected = (cfg[f"expected_{args.role}_layers"], cfg["expected_kv_heads"], cfg["expected_head_dim"])
    actual = (model.config.num_hidden_layers, model.config.num_key_value_heads, model.config.head_dim)
    if actual != expected:
        raise ValueError("loaded model architecture differs from prepared configuration")
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise ValueError("model was offloaded; entire model must be on the assigned GPU")
    extract_with_model(model, root / "tokens.npy", root / args.role, role=args.role,
                       prepared=prepared, device=torch.device("cuda:0"))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
