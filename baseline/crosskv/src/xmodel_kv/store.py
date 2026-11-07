from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ActivationMetadata:
    role: str
    model_path: str
    model_type: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_sequences: int
    sequence_length: int
    stride: int
    num_observations: int
    dtype: str = "float16"
    keys_are_rope_stripped: bool = True

    @classmethod
    def load(cls, root: str | Path) -> "ActivationMetadata":
        with (Path(root) / "metadata.json").open() as handle:
            return cls(**json.load(handle))

    def save(self, root: str | Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "metadata.json").open("w") as handle:
            json.dump(asdict(self), handle, indent=2)
            handle.write("\n")


class ActivationStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.metadata = ActivationMetadata.load(root)

    def path(self, kind: str, layer: int) -> Path:
        if kind not in {"k", "v"}:
            raise ValueError("kind must be k or v")
        return self.root / kind / f"layer_{layer:03d}.npy"

    def open(self, kind: str, layer: int, mode: str = "r") -> np.memmap:
        return np.load(self.path(kind, layer), mmap_mode=mode)

    def create(self, kind: str, layer: int) -> np.memmap:
        path = self.path(kind, layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        shape = (
            self.metadata.num_observations,
            self.metadata.num_kv_heads,
            self.metadata.head_dim,
        )
        return np.lib.format.open_memmap(path, mode="w+", dtype=self.metadata.dtype, shape=shape)


def validate_pair(source: ActivationStore, target: ActivationStore) -> None:
    sm, tm = source.metadata, target.metadata
    for field in ("num_kv_heads", "head_dim", "num_sequences", "sequence_length", "stride", "num_observations"):
        if getattr(sm, field) != getattr(tm, field):
            raise ValueError(f"source/target {field} mismatch: {getattr(sm, field)} != {getattr(tm, field)}")
    if not sm.keys_are_rope_stripped or not tm.keys_are_rope_stripped:
        raise ValueError("selection/fitting require RoPE-stripped activation stores")
