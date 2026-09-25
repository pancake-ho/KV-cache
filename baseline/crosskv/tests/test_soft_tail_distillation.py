import pytest
import torch
from types import SimpleNamespace

from xmodel_kv.cli.train_musique_soft_tail_capsule import (
    _validate_initialized_capsule,
    behavioral_kl,
    expand_soft_tail_capsule_slots,
    resolve_train_layer_bridge_weights,
    resolve_train_layer_patterns,
)
from xmodel_kv.soft_tail_capsule import LayerwiseKVWriteAdapter, SoftTailCapsule


def test_behavioral_kl_is_zero_for_identical_logits_and_positive_otherwise():
    teacher = torch.tensor([[[2.0, -1.0, 0.5], [0.0, 1.0, -2.0]]])

    identical = behavioral_kl(teacher, teacher, temperature=2.0)
    shifted = behavioral_kl(-teacher, teacher, temperature=2.0)

    assert identical.item() == pytest.approx(0.0, abs=1e-6)
    assert shifted.item() > 0


def test_behavioral_kl_rejects_shape_and_temperature_errors():
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="identical shapes"):
        behavioral_kl(logits, torch.zeros(1, 1, 3), temperature=1.0)
    with pytest.raises(ValueError, match="positive"):
        behavioral_kl(logits, logits, temperature=0.0)


def test_expand_soft_tail_capsule_slots_cycles_embeddings_only():
    capsule = SoftTailCapsule(torch.arange(12).reshape(3, 4).float())
    before = capsule.embeddings.detach().clone()

    expand_soft_tail_capsule_slots(capsule, slots=8)

    assert capsule.slots == 8
    assert torch.equal(capsule.embeddings[:3], before)
    assert torch.equal(capsule.embeddings[3:6], before)
    assert torch.equal(capsule.embeddings[6:], before[:2])


def test_behavioral_kl_is_mean_normalized_over_answer_positions():
    student = torch.tensor([[[0.0, 1.0, -1.0]]])
    teacher = torch.tensor([[[1.0, -1.0, 0.0]]])
    single = behavioral_kl(student, teacher, temperature=1.0)
    repeated = behavioral_kl(
        student.repeat(1, 5, 1), teacher.repeat(1, 5, 1), temperature=1.0
    )

    assert repeated.item() == pytest.approx(single.item())


def test_train_layer_pattern_schedule_parsing():
    assert resolve_train_layer_patterns("all", None) == ("all",)
    assert resolve_train_layer_patterns("all", "first_16, all") == (
        "first_16",
        "all",
    )
    with pytest.raises(ValueError, match="conflicts"):
        resolve_train_layer_patterns("last_16", "first_16,all")
    with pytest.raises(ValueError, match="unique"):
        resolve_train_layer_patterns("all", "first_16,first_16")


def test_train_layer_bridge_weight_parsing():
    patterns = ("first_16", "all")
    assert resolve_train_layer_bridge_weights(
        patterns, None, default_weight=2
    ) == {"first_16": 2.0, "all": 2.0}
    assert resolve_train_layer_bridge_weights(
        patterns, "first_16:0,all:4", default_weight=2
    ) == {"first_16": 0.0, "all": 4.0}
    with pytest.raises(ValueError, match="exactly match"):
        resolve_train_layer_bridge_weights(
            patterns, "first_16:0", default_weight=2
        )
    with pytest.raises(ValueError, match="non-negative"):
        resolve_train_layer_bridge_weights(
            patterns, "first_16:-1,all:4", default_weight=2
        )


def test_initialized_capsule_validation_freezes_model_and_writer_contract():
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16, num_hidden_layers=3)
    )
    capsule = SoftTailCapsule(
        torch.randn(2, 16),
        write_adapter=LayerwiseKVWriteAdapter(
            hidden_size=16, num_hidden_layers=3, rank=2, scale=0.5
        ),
    )

    _validate_initialized_capsule(
        capsule,
        initialization_config={"model": "model-a"},
        model=model,
        model_path="model-a",
        slots=2,
        writer_rank=2,
        writer_scale=0.5,
    )
    with pytest.raises(ValueError, match="writer configuration"):
        _validate_initialized_capsule(
            capsule,
            initialization_config={"model": "model-a"},
            model=model,
            model_path="model-a",
            slots=2,
            writer_rank=4,
            writer_scale=0.5,
        )
    with pytest.raises(ValueError, match="model does not match"):
        _validate_initialized_capsule(
            capsule,
            initialization_config={"model": "model-b"},
            model=model,
            model_path="model-a",
            slots=2,
            writer_rank=2,
            writer_scale=0.5,
        )
