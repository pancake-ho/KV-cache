import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from xmodel_kv.cli.train_musique_soft_tail_capsule import (
    freeze_initialized_base,
    parameter_sha256,
    positional_orbit_caches,
)
from xmodel_kv.composable_delta import compose_int4_base_delta
from xmodel_kv.soft_tail_capsule import (
    BoundaryKVWriteAdapter,
    SoftTailCapsule,
    reposition_tail_cache,
)


def _tiny_qwen():
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        )
    ).eval()


def _base_and_task_caches():
    generator = torch.Generator().manual_seed(71)
    base = tuple(
        (
            torch.randn(1, 2, 4, 8, generator=generator).to(torch.bfloat16),
            torch.randn(1, 2, 4, 8, generator=generator).to(torch.bfloat16),
        )
        for _ in range(4)
    )
    task = []
    for layer_index, layer in enumerate(base):
        task_layer = []
        for tensor in layer:
            value = tensor.clone()
            if layer_index == 2:
                value = (value.float() + 0.125).to(torch.bfloat16)
            task_layer.append(value.detach().requires_grad_(True))
        task.append(tuple(task_layer))
    return base, tuple(task)


def test_positional_orbit_teacher_is_exact_target_frame_production_composition():
    model = _tiny_qwen()
    base, task = _base_and_task_caches()
    source_start = 19
    target_start = 5
    base_layers = frozenset({0, 1, 2})
    delta_layers = frozenset({2})

    student, teacher = positional_orbit_caches(
        model,
        base_source_tail=base,
        task_source_tail=task,
        source_start=source_start,
        target_start=target_start,
        slots=4,
        base_layers=base_layers,
        delta_layers=delta_layers,
    )
    expected = compose_int4_base_delta(
        reposition_tail_cache(
            model, base, source_start=source_start, target_start=target_start
        ),
        reposition_tail_cache(
            model, task, source_start=source_start, target_start=target_start
        ),
        suffix_tokens=4,
        base_layers=base_layers,
        delta_layers=delta_layers,
    ).cache

    for actual_layer, expected_layer in zip(teacher, expected, strict=True):
        for actual_tensor, expected_tensor in zip(
            actual_layer, expected_layer, strict=True
        ):
            assert torch.equal(actual_tensor, expected_tensor)

    sum(tensor.float().sum() for layer in student for tensor in layer).backward()
    for layer_index, layer in enumerate(task):
        for tensor in layer:
            if layer_index == 2:
                assert tensor.grad is not None
                assert torch.isfinite(tensor.grad).all()
                assert torch.count_nonzero(tensor.grad).item() > 0
            else:
                assert tensor.grad is None


def test_freeze_initialized_base_exposes_only_rank16_boundary_parameters():
    capsule = SoftTailCapsule(
        torch.randn(4, 4096),
        boundary_write_adapter=BoundaryKVWriteAdapter(
            hidden_size=4096,
            layer_index=14,
            rank=16,
        ),
    )

    freeze_initialized_base(capsule)

    trainable = [
        parameter for parameter in capsule.parameters() if parameter.requires_grad
    ]
    assert sum(parameter.numel() for parameter in trainable) == 131_072
    assert all(
        name.startswith("boundary_write_adapter.")
        for name, parameter in capsule.named_parameters()
        if parameter.requires_grad
    )


def test_parameter_hash_separates_frozen_and_trainable_state():
    capsule = SoftTailCapsule(
        torch.randn(2, 16),
        boundary_write_adapter=BoundaryKVWriteAdapter(
            hidden_size=16,
            layer_index=1,
            rank=2,
        ),
    )
    freeze_initialized_base(capsule)
    frozen_before = parameter_sha256(capsule, requires_grad=False)
    trainable_before = parameter_sha256(capsule, requires_grad=True)

    with torch.no_grad():
        capsule.boundary_write_adapter.block.up.weight.add_(0.25)

    assert parameter_sha256(capsule, requires_grad=False) == frozen_before
    assert parameter_sha256(capsule, requires_grad=True) != trainable_before
