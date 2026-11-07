from __future__ import annotations

from pathlib import Path
import numpy as np

from .common import KINDS, read_json


class ProbeStore:
    def __init__(self, root, *, complete=True):
        self.root = Path(root)
        self.metadata = read_json(self.root / "metadata.json")
        if self.metadata["schema_version"] != 1:
            raise ValueError("unsupported Figure 2 store schema")
        if complete and not self.metadata.get("complete", False):
            raise ValueError(f"incomplete extraction: {self.root}")
        m = self.metadata
        self.shape = (m["num_observations"], m["num_kv_heads"], m["head_dim"])

    def path(self, kind, layer):
        if kind not in KINDS or not 0 <= layer < self.metadata["num_layers"]:
            raise ValueError(f"invalid activation index: {kind}, {layer}")
        return self.root / kind / f"layer_{layer:03d}.npy"

    def open(self, kind, layer):
        data = np.load(self.path(kind, layer), mmap_mode="r")
        if data.shape != self.shape or data.dtype != np.dtype("float16"):
            raise ValueError(f"unexpected activation shape/dtype: {data.shape}/{data.dtype}")
        return data


def validate_pair(source, target):
    sm, tm = source.metadata, target.metadata
    for field in ("num_observations", "num_kv_heads", "head_dim", "prepare_digest", "tokens_sha256",
                  "train_observations", "heldout_observations", "sequence_length", "stride"):
        if sm[field] != tm[field]:
            raise ValueError(f"source/target {field} mismatch")
    if sm["role"] != "source" or tm["role"] != "target":
        raise ValueError("incorrect store roles")
    if sm["config"] != tm["config"]:
        raise ValueError("source/target experiment configs differ")
