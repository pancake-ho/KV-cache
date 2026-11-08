"""Read and export the neural mapper's existing per-head training metrics."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .spec import MapperSpec, checkpoint_path


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
