"""CPU-only structural tests. GPU/cache injection tests require Seraph."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from proposed.src.exp.mapper.data import ActivationStore, check_calibration_pair, minibatch
from proposed.src.exp.mapper.model import HeadMLP
from proposed.src.exp.mapper.spec import MapperSpec, expected_checkpoints
from proposed.src.exp.mapper.stage import copy_selected
from proposed.src.exp.mapper.train import build_spec


def make_stores(root: Path, *, source_layers: int = 3, target_layers: int = 2):
    source_dir, target_dir = root / "source", root / "target"
    for path, count, role in ((source_dir, source_layers, "source"),
                              (target_dir, target_layers, "target")):
        path.mkdir()
        (path / "metadata.json").write_text(json.dumps({
            "role": role, "model_path": f"model-{role}", "num_layers": count,
            "num_kv_heads": 2, "head_dim": 4, "num_sequences": 4,
            "sequence_length": 8, "stride": 1, "num_observations": 32,
            "keys_are_rope_stripped": True,
        }))
        for kind in ("k", "v"):
            (path / kind).mkdir()
            for layer in range(count):
                rng = np.random.default_rng(layer + (1 if role == "source" else 10))
                np.save(path / kind / f"layer_{layer:03d}.npy",
                        rng.normal(size=(32, 2, 4)).astype(np.float16))
    selection = root / "selection.json"
    selection.write_text(json.dumps({"selected_layers": [[2, 0], [0, 1]]}))
    return source_dir, target_dir, selection


def test_mlp_shape_and_backward():
    net = HeadMLP(16, 4, hidden_dim=8)
    x = torch.randn(7, 16)
    y = net(x)
    assert y.shape == (7, 4)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_stage_metadata_and_training_contract(tmp_path):
    source, target, select = make_stores(tmp_path)
    stage = tmp_path/"staged"
    copy_selected(source, target, select, stage, begin=0, end=1, kinds=("k", "v"))
    assert sorted((stage/"source"/"k").glob("*.npy")) == [
        stage/"source"/"k"/"layer_000.npy", stage/"source"/"k"/"layer_002.npy"]
    assert len(list((stage/"target"/"v").glob("*.npy"))) == 1
    ss, tt = ActivationStore(stage/"source"), ActivationStore(stage/"target")
    check_calibration_pair(ss, tt)
    x, y = minibatch([ss.layer("k",2),ss.layer("k",0)], tt.layer("k",0),
                     np.array([0,2,1,3]), 1, torch.device("cpu"))
    assert x.shape == (4, 16) and y.shape == (4, 4)
    spec = build_spec(str(stage/"source"), str(stage/"target"), str(stage/"selection.json"),
                      hidden_dim=8, lr=.001, epochs=1, batch_size=4,
                      seed=2026, max_observations=16,
                      train_dtype="float32", storage_dtype="float32")
    assert spec.input_dim == 16
    artifact = tmp_path/"artifact"
    spec.save_or_validate(artifact)
    spec.save_or_validate(artifact)
    assert MapperSpec.load(artifact) == spec
    assert len(expected_checkpoints(spec, artifact)) == 8


def test_config_guard(tmp_path):
    source, target, select = make_stores(tmp_path)
    spec = build_spec(str(source), str(target), str(select), hidden_dim=8, lr=.001,
                      epochs=1, batch_size=4, seed=2026,
                      max_observations=16, train_dtype="float32", storage_dtype="float32")
    out = tmp_path/"artifacts"
    spec.save_or_validate(out)
    spec2 = build_spec(str(source), str(target), str(select), hidden_dim=16, lr=.001,
                       epochs=1, batch_size=4, seed=2026, max_observations=16,
                       train_dtype="float32", storage_dtype="float32")
    import pytest
    with pytest.raises(ValueError, match="conflicts"):
        spec2.save_or_validate(out)
