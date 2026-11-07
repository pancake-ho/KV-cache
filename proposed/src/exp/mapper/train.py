"""One-GPU, per-(target-layer, head, K/V) MLP fitting with durable resumable artifacts."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .data import ActivationStore, check_calibration_pair, minibatch
from .model import HeadMLP
from .spec import MapperSpec, checkpoint_path, file_hash


def build_spec(source_dir: str, target_dir: str, selection_path: str, *,
               hidden_dim: int, lr: float, epochs: int, batch_size: int, seed: int,
               max_observations: int | None, train_dtype: str, storage_dtype: str) -> MapperSpec:
    source, target = ActivationStore(source_dir), ActivationStore(target_dir)
    check_calibration_pair(source, target)
    with Path(selection_path).open() as f:
        selection = json.load(f)
    selected = [[int(i) for i in layer] for layer in selection["selected_layers"]]
    available = int(source.meta["num_observations"])
    if max_observations is not None and not 2 <= max_observations <= available:
        raise ValueError("max_observations must be between 2 and available observations")
    spec = MapperSpec(
        source_model=str(source.meta["model_path"]), target_model=str(target.meta["model_path"]),
        source_layers=int(source.meta["num_layers"]), target_layers=int(target.meta["num_layers"]),
        source_kv_heads=int(source.meta["num_kv_heads"]), target_kv_heads=int(target.meta["num_kv_heads"]),
        source_head_dim=int(source.meta["head_dim"]), target_head_dim=int(target.meta["head_dim"]),
        selected_layers=selected, num_observations=max_observations or available,
        hidden_dim=hidden_dim, lr=lr, epochs=epochs, batch_size=batch_size, seed=seed,
        train_dtype=train_dtype, storage_dtype=storage_dtype,
        source_metadata_hash=file_hash(Path(source_dir)/"metadata.json"),
        target_metadata_hash=file_hash(Path(target_dir)/"metadata.json"),
        selection_hash=file_hash(selection_path),
    )
    spec.validate()
    return spec


def fit_all(*, source_dir: str, target_dir: str, spec: MapperSpec, output_dir: str,
            target_layer_start: int = 0, target_layer_end: int | None = None,
            head_start: int = 0, head_end: int | None = None,
            kinds: tuple[str, ...] = ("k", "v")) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: refusing silent CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    spec.validate()
    last_layer = spec.target_layers if target_layer_end is None else target_layer_end
    last_head = spec.target_kv_heads if head_end is None else head_end
    if not 0 <= target_layer_start < last_layer <= spec.target_layers:
        raise ValueError("invalid target layer range")
    if not 0 <= head_start < last_head <= spec.target_kv_heads:
        raise ValueError("invalid target head range")
    if not kinds or any(k not in {"k", "v"} for k in kinds):
        raise ValueError("kinds must be k/v")
    root = Path(output_dir)
    spec.save_or_validate(root)
    source, target = ActivationStore(source_dir), ActivationStore(target_dir)
    check_calibration_pair(source, target)
    n = spec.num_observations
    if n > int(source.meta["num_observations"]):
        raise ValueError("not enough source samples")
    for target_layer in range(target_layer_start, last_layer):
        selected = spec.selected_layers[target_layer]
        for kind in kinds:
            x_layers = [source.layer(kind, i) for i in selected]
            y_layer = target.layer(kind, target_layer)
            for head in range(head_start, last_head):
                path = checkpoint_path(root, target_layer, head, kind)
                if path.exists():
                    print(f"SKIP completed {path}", flush=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                metrics = _fit_head(spec, x_layers, y_layer, head, target_layer, kind, device)
                state, stats = metrics
                # Write one complete head atomically. Interrupted head is retried from scratch.
                temp = path.with_name(f".{path.stem}.{os.getpid()}.safetensors")
                save_file(state, str(temp))
                os.replace(temp, path)
                metrics_path = path.with_suffix(".json")
                metric_temp = metrics_path.with_name(f".{metrics_path.name}.{os.getpid()}.tmp")
                with metric_temp.open("w") as f:
                    json.dump(stats, f, indent=2)
                    f.write("\n")
                os.replace(metric_temp, metrics_path)
                print("DONE", json.dumps(stats), flush=True)
            target.close()
        source.close()


def _fit_head(spec: MapperSpec, x_layers: list[np.ndarray], y_layer: np.ndarray,
              head: int, target_layer: int, kind: str, device: torch.device):
    nonce = (spec.seed + target_layer*1024 + head*4 + (0 if kind == "k" else 1))
    random.seed(nonce)
    np.random.seed(nonce % (2**32 - 1))
    torch.manual_seed(nonce)
    torch.cuda.manual_seed_all(nonce)
    model = HeadMLP(spec.input_dim, spec.target_head_dim, spec.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.lr)
    generator = np.random.default_rng(nonce)
    n = spec.num_observations
    use_bf16 = spec.train_dtype == "bfloat16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 training requested but GPU does not support BF16")
    tic = time.monotonic()
    losses = []
    for epoch in range(spec.epochs):
        model.train()
        perm = generator.permutation(n)
        weighted_loss = 0.0
        for begin in range(0, n, spec.batch_size):
            rows = perm[begin:begin+spec.batch_size]
            x, y = minibatch(x_layers, y_layer, rows, head, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output = model(x)
            loss = F.mse_loss(output.float(), y.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite MSE at layer={target_layer}, head={head}, kind={kind}")
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item())*len(rows)
        losses.append(weighted_loss/n)
        print(f"layer={target_layer} head={head} kind={kind} epoch={epoch+1}/{spec.epochs} mse={losses[-1]:.7g}", flush=True)
    storage = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[spec.storage_dtype]
    state = {name: value.detach().to(dtype=storage, device="cpu").contiguous()
             for name, value in model.state_dict().items()}
    stats = {"target_layer": target_layer, "head": head, "kind": kind,
             "observations": n, "epochs": spec.epochs, "train_mse": losses[-1],
             "epoch_mse": losses, "elapsed_seconds": time.monotonic()-tic,
             "parameter_count": sum(p.numel() for p in model.parameters()),
             "seed": nonce, "device": torch.cuda.get_device_name(device)}
    return state, stats
