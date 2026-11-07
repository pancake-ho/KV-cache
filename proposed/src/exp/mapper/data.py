"""Read CrossKV's existing RoPE-stripped activation stores, without modifying them."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch


class ActivationStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        with (self.root / "metadata.json").open() as f:
            self.meta = json.load(f)
        if not self.meta.get("keys_are_rope_stripped", True):
            raise ValueError("keys must be RoPE-stripped in the stored activations")
        self._arrays: dict[tuple[str, int], np.ndarray] = {}

    def layer(self, kind: str, layer: int) -> np.ndarray:
        if kind not in {"k", "v"}:
            raise ValueError("kind must be k or v")
        if not 0 <= layer < int(self.meta["num_layers"]):
            raise ValueError(f"out-of-range source/target layer {layer}")
        key = (kind, layer)
        if key not in self._arrays:
            array = np.load(self.root / kind / f"layer_{layer:03d}.npy", mmap_mode="r")
            expected = (int(self.meta["num_observations"]), int(self.meta["num_kv_heads"]),
                        int(self.meta["head_dim"]))
            if array.shape != expected:
                raise ValueError(f"{key} has shape {array.shape}; expected {expected}")
            self._arrays[key] = array
        return self._arrays[key]

    def close(self) -> None:
        self._arrays.clear()


def check_calibration_pair(source: ActivationStore, target: ActivationStore) -> None:
    for field in ("num_kv_heads", "head_dim", "num_observations", "num_sequences",
                  "sequence_length", "stride"):
        if source.meta[field] != target.meta[field]:
            raise ValueError(f"calibration mismatch in {field}: {source.meta[field]} / {target.meta[field]}")


def minibatch(source_arrays: list[np.ndarray], target_array: np.ndarray,
              indices: np.ndarray, head: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample rows only from disk-backed arrays; flatten *all* source KV heads."""
    features = [np.asarray(part[indices].reshape(len(indices), -1), dtype=np.float32)
                for part in source_arrays]
    x = torch.from_numpy(np.ascontiguousarray(np.concatenate(features, axis=1))).to(device)
    y = torch.from_numpy(np.array(target_array[indices, head, :], dtype=np.float32, copy=True)).to(device)
    return x, y
