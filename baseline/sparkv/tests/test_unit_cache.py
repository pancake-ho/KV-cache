import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from baseline.sparkv.unit_cache import (
    materialize_sample,
    unit_key,
)


def test_materialize_raw_units(tmp_path: Path):
    sample_dir = tmp_path / "raw" / "sample_000"
    sample_dir.mkdir(parents=True)

    chunk_size = 3
    layers = 2
    heads = 2

    tensors = {}
    for layer in range(layers):
        key = torch.arange(
            1 * heads * chunk_size * 4,
            dtype=torch.bfloat16,
        ).reshape(1, heads, chunk_size, 4)
        value = key + 1
        tensors[f"k_{layer:02d}"] = key
        tensors[f"v_{layer:02d}"] = value

    save_file(tensors, str(sample_dir / "chunk_000.safetensors"))

    meta = {
        "seq_len": chunk_size,
        "chunk_size": chunk_size,
        "num_chunks": 1,
        "layers": layers,
        "kv_heads": heads,
        "chunks": [
            {
                "index": 0,
                "wire_bytes": 128,
                "lh_wire_bytes": {
                    f"{layer}:{head}": 32
                    for layer in range(layers)
                    for head in range(heads)
                },
            }
        ],
    }
    (sample_dir / "meta.json").write_text(
        json.dumps(meta),
        encoding="utf-8",
    )

    result = materialize_sample(sample_dir, "raw")
    assert result["unit_files"] == layers * heads

    updated = json.loads(
        (sample_dir / "meta.json").read_text(encoding="utf-8")
    )
    assert updated["unit_layout_version"] == 1
    assert unit_key(0, 1, 1) in updated["unit_files"]

    path = (
        sample_dir
        / updated["unit_files"][unit_key(0, 1, 1)]["path"]
    )
    assert path.is_file()
