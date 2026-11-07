"""Explicit, immutable mapper training/runtime contract."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MapperSpec:
    source_model: str
    target_model: str
    source_layers: int
    target_layers: int
    source_kv_heads: int
    target_kv_heads: int
    source_head_dim: int
    target_head_dim: int
    selected_layers: list[list[int]]
    num_observations: int
    hidden_dim: int = 1024
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 4096
    seed: int = 2026
    train_dtype: str = "bfloat16"
    storage_dtype: str = "float32"
    source_metadata_hash: str = ""
    target_metadata_hash: str = ""
    selection_hash: str = ""
    format_version: int = 1

    @property
    def input_dim(self) -> int:
        return len(self.selected_layers[0]) * self.source_kv_heads * self.source_head_dim

    def validate(self) -> None:
        if min(self.source_layers, self.target_layers, self.source_kv_heads,
               self.target_kv_heads, self.source_head_dim, self.target_head_dim,
               self.num_observations, self.hidden_dim, self.batch_size, self.epochs) <= 0:
            raise ValueError("invalid nonpositive spec field")
        if not 0 < self.lr < 1:
            raise ValueError("learning rate must be between zero and one")
        if self.train_dtype not in {"bfloat16", "float32"}:
            raise ValueError("train_dtype must be bfloat16 or float32")
        if self.storage_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("invalid storage_dtype")
        if len(self.selected_layers) != self.target_layers:
            raise ValueError("selected_layers must have an entry per target layer")
        k = len(self.selected_layers[0])
        if not k or any(len(layers) != k or len(set(layers)) != k or
                        any(not 0 <= v < self.source_layers for v in layers)
                        for layers in self.selected_layers):
            raise ValueError("invalid selected source layer lists")
        if self.source_kv_heads != self.target_kv_heads or self.source_head_dim != self.target_head_dim:
            raise ValueError("matched-KV model pairs only in this first implementation")

    @classmethod
    def load(cls, root: str | Path) -> "MapperSpec":
        with (Path(root) / "config.json").open() as f:
            obj = cls(**json.load(f))
        obj.validate()
        return obj

    def save_or_validate(self, root: str | Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "config.json"
        if path.exists():
            if MapperSpec.load(root) != self:
                raise ValueError("existing mapper config conflicts with requested training; use a new output")
            return
        # This file is created once. Never silently replace a previous artifact.
        with path.open("x") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
            f.write("\n")


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_path(root: str | Path, target_layer: int, head: int, kind: str) -> Path:
    if kind not in {"k", "v"}:
        raise ValueError("kind must be k or v")
    return Path(root) / "layers" / f"layer_{target_layer:03d}" / f"head_{head:02d}_{kind}.safetensors"


def expected_checkpoints(spec: MapperSpec, root: str | Path) -> list[Path]:
    return [checkpoint_path(root, layer, head, kind)
            for layer in range(spec.target_layers)
            for head in range(spec.target_kv_heads)
            for kind in ("k", "v")]
