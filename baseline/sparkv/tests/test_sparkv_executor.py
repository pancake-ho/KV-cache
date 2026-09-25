import json
from pathlib import Path

import numpy as np
import pytest
import torch

from baseline.sparkv.codec import (
    encode_kv_unit,
    write_encoded_unit,
)
from baseline.sparkv.executor import (
    CloudMemorySource,
    HybridQwen3Engine,
    UnitStore,
    _effective_bandwidth,
    _permanently_layer_blocked,
)
from baseline.sparkv.scheduler import (
    Chunk,
)


def test_unit_store_rejects_duplicate_ownership():
    store = UnitStore()
    c = Chunk(
        0,
        0,
        0,
    )

    pair = (
        torch.zeros(
            1,
            1,
            2,
            4,
        ),
        torch.zeros(
            1,
            1,
            2,
            4,
        ),
    )

    store.put(
        c,
        pair,
        "compute",
    )

    assert store.has(c)
    assert store.route(c) == "compute"
    assert store.count() == 1

    with pytest.raises(
        RuntimeError,
        match="duplicate KV ownership",
    ):
        store.put(
            c,
            pair,
            "stream",
        )


def test_permanently_layer_blocked_only_after_parent_stream():
    c = Chunk(
        0,
        1,
        0,
    )
    parent = Chunk(
        0,
        0,
        0,
    )

    assert not (
        _permanently_layer_blocked(
            c,
            {},
        )
    )

    assert not (
        _permanently_layer_blocked(
            c,
            {
                parent:
                    "compute"
            },
        )
    )

    assert (
        _permanently_layer_blocked(
            c,
            {
                parent:
                    "stream"
            },
        )
    )


def test_effective_bandwidth_is_exact_without_jitter():
    rng = (
        np.random.default_rng(
            2026
        )
    )

    assert (
        _effective_bandwidth(
            640.0,
            0.0,
            rng,
        )
        == 640.0
    )


def test_cloud_memory_source_uses_actual_encoded_bytes(
    tmp_path: Path,
):
    sample_dir = (
        tmp_path
        / "sample_000"
    )
    path = (
        sample_dir
        / "units"
        / "t000"
        / "l00_h00.skv"
    )

    key = torch.zeros(
        1,
        1,
        8,
        4,
    )
    value = torch.zeros_like(
        key
    )

    encoded = encode_kv_unit(
        key,
        value,
        bits=5,
    )
    wire_bytes = (
        write_encoded_unit(
            encoded,
            path,
        )
    )

    meta = {
        "unit_files": {
            "0:0:0": {
                "path":
                    "units/t000/l00_h00.skv",
                "wire_bytes":
                    wire_bytes,
            }
        }
    }

    source = CloudMemorySource(
        sample_dir,
        meta,
    )

    blob = source.get(
        Chunk(
            0,
            0,
            0,
        )
    )

    assert isinstance(
        blob,
        bytes,
    )
    assert len(blob) == (
        wire_bytes
    )



def test_finalized_layer_releases_dead_intermediates():
    engine = (
        HybridQwen3Engine
        .__new__(
            HybridQwen3Engine
        )
    )
    engine.H = 2

    target = (
        0,
        0,
    )
    keep = (
        0,
        1,
    )

    engine.hidden_inputs = {
        target:
            torch.zeros(1),
        keep:
            torch.ones(1),
    }
    engine.projection_cache = {
        target: (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        ),
        keep: (
            torch.ones(1),
            torch.ones(1),
            torch.ones(1),
        ),
    }
    engine.local_heads = {
        target: {0, 1},
        keep: {0},
    }
    engine.attention_parts = {
        Chunk(0, 0, 0):
            torch.zeros(1),
        Chunk(0, 0, 1):
            torch.zeros(1),
        Chunk(0, 1, 0):
            torch.ones(1),
    }

    engine._release_finalized_state(
        0,
        0,
    )

    assert target not in (
        engine.hidden_inputs
    )
    assert target not in (
        engine.projection_cache
    )
    assert target not in (
        engine.local_heads
    )
    assert (
        Chunk(0, 0, 0)
        not in engine.attention_parts
    )
    assert (
        Chunk(0, 0, 1)
        not in engine.attention_parts
    )

    # State for the next layer must remain intact.
    assert keep in (
        engine.hidden_inputs
    )
    assert keep in (
        engine.projection_cache
    )
    assert keep in (
        engine.local_heads
    )
    assert (
        Chunk(0, 1, 0)
        in engine.attention_parts
    )
