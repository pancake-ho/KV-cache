import copy

import torch

from xmodel_kv.cli.analyze_capsule_checkpoint_drift import analyze_checkpoints


def _checkpoint(*, layers=5, hidden=8, rank=2):
    generator = torch.Generator().manual_seed(1)
    state = {}
    for index in range(layers - 1):
        state[f"blocks.{index}.down.weight"] = torch.randn(
            rank, hidden, generator=generator
        )
        state[f"blocks.{index}.up.weight"] = torch.randn(
            hidden, rank, generator=generator
        )
    return {
        "embeddings": torch.randn(2, hidden, generator=generator),
        "write_adapter_config": {
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "rank": rank,
            "scale": 1.0,
        },
        "write_adapter_state": state,
    }


def test_drift_analysis_detects_embedding_dominance():
    parent = _checkpoint()
    candidate = copy.deepcopy(parent)
    candidate["embeddings"] += 10

    result = analyze_checkpoints(parent, candidate, probes=16, seed=2)

    assert result["classification"] == "embedding_dominant"
    assert result["parameter_drift"]["embedding_share"] == 1.0
    assert result["functional_drift"]["total_energy"] == 0.0


def test_drift_analysis_detects_localized_writer_change():
    parent = _checkpoint(layers=9)
    candidate = copy.deepcopy(parent)
    candidate["write_adapter_state"]["blocks.3.up.weight"] += 1

    result = analyze_checkpoints(parent, candidate, probes=32, seed=2)

    assert result["classification"] == "layer_localized"
    assert result["functional_drift"]["layers"][3]["functional_drift_share"] == 1.0
    assert result["parameter_drift"]["embedding_share"] == 0.0


def test_drift_analysis_rejects_incompatible_writer_configs():
    parent = _checkpoint()
    candidate = _checkpoint(rank=3)

    try:
        analyze_checkpoints(parent, candidate, probes=8, seed=2)
    except ValueError as error:
        assert "configs differ" in str(error)
    else:
        raise AssertionError("expected incompatible checkpoints to fail")
