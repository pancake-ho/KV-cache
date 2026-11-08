"""Read saved training metrics and score frozen neural mappers on calibration KV."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .spec import MapperSpec, checkpoint_path, file_hash

if TYPE_CHECKING:
    import numpy as np
    import torch

    from .model import HeadMLP


@dataclass(frozen=True)
class HeadTrainingMetric:
    target_layer: int
    head: int
    kind: str
    train_mse: float
    observations: int
    epochs: int


def load_training_metrics(root: Path) -> tuple[MapperSpec, list[HeadTrainingMetric]]:
    spec = MapperSpec.load(root)
    metrics = []
    for layer in range(spec.target_layers):
        for head in range(spec.target_kv_heads):
            for kind in ("k", "v"):
                path = checkpoint_path(root, layer, head, kind).with_suffix(".json")
                try:
                    raw = path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    continue  # Partial training runs are valid inputs for plotting.
                try:
                    row = json.loads(raw)
                    if (row["target_layer"], row["head"], row["kind"]) != (layer, head, kind):
                        raise ValueError("layer, head or kind does not match its path")
                    if (row["observations"], row["epochs"]) != (spec.num_observations, spec.epochs):
                        raise ValueError("observations or epochs do not match config.json")
                    mse = float(row["train_mse"])
                    if not math.isfinite(mse) or mse < 0:
                        raise ValueError(f"invalid train_mse {mse}")
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid training metric {path}: {error}") from error
                metrics.append(HeadTrainingMetric(layer, head, kind, mse, row["observations"], row["epochs"]))
    if not metrics:
        raise ValueError(f"No per-head metric JSONs found under {root / 'layers'}")
    return spec, metrics


def write_training_metrics_csv(path: Path, metrics: list[HeadTrainingMetric]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("target_layer", "target_kv_head", "kind", "train_mse", "observations", "epochs"))
        for item in metrics:
            writer.writerow((item.target_layer, item.head, item.kind, item.train_mse,
                             item.observations, item.epochs))


def _score_head(network: HeadMLP, inputs: torch.Tensor, target: torch.Tensor,
                block_width: int) -> list[float]:
    """Vary one source block, fixing all other blocks to their sample means."""
    import torch

    batch_size = 256
    blocks = inputs.shape[1] // block_width
    template = inputs.mean(dim=0).expand(batch_size, -1)
    target = target.double()
    sst = (target - target.mean(dim=0)).square().sum().item()
    if not math.isfinite(sst):
        raise ValueError("Nonfinite target activations")
    if sst == 0:
        return [math.nan] * blocks

    scores = []
    for slot in range(blocks):
        block = slice(slot * block_width, (slot + 1) * block_width)
        sse = torch.zeros((), dtype=torch.float64, device=target.device)
        for start in range(0, len(target), batch_size):
            chunk = slice(start, start + batch_size)
            observed = inputs[chunk, block]
            features = template[:len(observed)].clone()
            features[:, block] = observed
            sse += (network(features).double() - target[chunk]).square().sum()
        score = 1 - sse.item() / sst
        if not math.isfinite(score):
            raise ValueError("Nonfinite mapper score")
        scores.append(score)
    return scores


def score_source_layers(mapper_dir: Path, source_dir: Path, target_dir: Path,
                        samples: int, seed: int) -> tuple[MapperSpec, np.ndarray, np.ndarray]:
    """Return head-averaged scores indexed by (K/V, source layer, target layer)."""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    from .data import ActivationStore, check_calibration_pair
    from .model import HeadMLP

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run the R² probe on an allocated GPU node.")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    spec = MapperSpec.load(mapper_dir)
    source, target = ActivationStore(source_dir), ActivationStore(target_dir)
    check_calibration_pair(source, target)
    for store, fingerprint in ((source, spec.source_metadata_hash), (target, spec.target_metadata_hash)):
        if not fingerprint or file_hash(store.root / "metadata.json") != fingerprint:
            raise ValueError(f"Calibration metadata differs from mapper training: {store.root}")
    if not 2 <= samples <= spec.num_observations:
        raise ValueError(f"samples must be between 2 and {spec.num_observations}")

    rows = np.sort(np.random.default_rng(seed).choice(spec.num_observations, samples, replace=False))
    scores = np.full((2, spec.source_layers, spec.target_layers), np.nan)
    block_width = spec.source_kv_heads * spec.source_head_dim
    with torch.inference_mode():
        for layer, selected in enumerate(spec.selected_layers):
            for panel, kind in enumerate(("k", "v")):
                paths = [checkpoint_path(mapper_dir, layer, head, kind) for head in range(spec.target_kv_heads)]
                if not all(path.is_file() for path in paths):
                    continue
                features = [np.asarray(source.layer(kind, index)[rows], dtype=np.float32).reshape(samples, -1)
                            for index in selected]
                inputs = torch.from_numpy(np.concatenate(features, axis=1)).to(device)
                targets = torch.from_numpy(np.asarray(target.layer(kind, layer)[rows], dtype=np.float32)).to(device)
                head_scores = []
                for head, path in enumerate(paths):
                    network = HeadMLP(spec.input_dim, spec.target_head_dim, spec.hidden_dim)
                    network.load_state_dict(load_file(str(path), device="cpu"), strict=True)
                    network.to(device).eval()
                    head_scores.append(_score_head(network, inputs, targets[:, head], block_width))
                    del network
                if np.isfinite(head_scores).all():
                    scores[panel, selected, layer] = np.mean(head_scores, axis=0)
                    print(f"Scored target layer {layer} {kind.upper()} ({len(selected)} source layers)", flush=True)
                else:
                    print(f"Skipped target layer {layer} {kind.upper()}: a target head has zero variance", flush=True)
                del features, inputs, targets
                source.close()
                target.close()
    if not np.isfinite(scores).any():
        raise ValueError("No R² scores. At least one target layer and K/V kind needs every head checkpoint.")
    return spec, scores, rows
