import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from xmodel_kv.policy_imprinting import prefill_legacy_cache
from xmodel_kv.semantic_kv_summary import concatenate_caches, teacher_forced_answer_logits
from xmodel_kv.soft_tail_capsule import (
    BoundaryKVWriteAdapter,
    LayerwiseKVWriteAdapter,
    ProgressiveSoftTailCapsule,
    SoftTailCapsule,
    detach_cache_layer_prefix,
    load_soft_tail_capsule,
    mask_source_prefix_attention,
    materialize_token_tail,
    move_legacy_cache,
    reposition_tail_cache,
)


def _tiny_qwen(num_hidden_layers=2):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        )
    ).eval()


def _normal_cache(cache):
    return tuple((key.clone(), value.clone()) for key, value in cache)


def test_soft_tail_capsule_materializes_native_kv_and_receives_task_gradient():
    torch.manual_seed(2)
    model = _tiny_qwen()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source_ids = [1, 2, 3, 4, 5]
    target_ids = [6, 7, 8]
    source_cache = _normal_cache(prefill_legacy_cache(model, source_ids))
    target_cache = _normal_cache(prefill_legacy_cache(model, target_ids))
    initial = model.model.embed_tokens.weight[[9, 10]].detach()
    capsule = SoftTailCapsule(initial)

    source_tail = capsule.materialize(
        model, prefix_cache=source_cache, prefix_tokens=len(source_ids)
    )
    assert len(source_tail) == model.config.num_hidden_layers
    assert source_tail[0][0].shape == (1, 2, 2, 8)
    target_tail = reposition_tail_cache(
        model,
        source_tail,
        source_start=len(source_ids),
        target_start=len(target_ids),
    )
    assembled = concatenate_caches(target_cache, target_tail)
    logits = teacher_forced_answer_logits(
        model,
        legacy_cache=assembled,
        cached_tokens=len(target_ids) + capsule.slots,
        query_ids=[11, 12],
        answer_ids=[13, 14],
    )
    logits.float().square().mean().backward()
    assert capsule.embeddings.grad is not None
    assert torch.isfinite(capsule.embeddings.grad).all()


def test_reposition_tail_cache_is_identity_at_same_positions():
    torch.manual_seed(3)
    model = _tiny_qwen()
    cache = _normal_cache(prefill_legacy_cache(model, [1, 2, 3, 4]))
    tail = tuple((key[:, :, -2:], value[:, :, -2:]) for key, value in cache)
    repositioned = reposition_tail_cache(
        model, tail, source_start=2, target_start=2
    )
    for (key, value), (new_key, new_value) in zip(tail, repositioned, strict=True):
        torch.testing.assert_close(new_key, key, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(new_value, value)


def test_load_soft_tail_capsule_validates_and_restores_checkpoint(tmp_path):
    embeddings = torch.randn(3, 16)
    checkpoint = tmp_path / "capsule.pt"
    torch.save({"embeddings": embeddings, "config": {"slots": 3}}, checkpoint)

    capsule, config = load_soft_tail_capsule(checkpoint)

    assert capsule.slots == 3
    assert config == {"slots": 3}
    torch.testing.assert_close(capsule.embeddings, embeddings)


def test_load_soft_tail_capsule_rejects_slot_mismatch(tmp_path):
    checkpoint = tmp_path / "capsule.pt"
    torch.save(
        {"embeddings": torch.randn(2, 16), "config": {"slots": 3}}, checkpoint
    )

    try:
        load_soft_tail_capsule(checkpoint)
    except ValueError as error:
        assert "slot count" in str(error)
    else:
        raise AssertionError("expected a mismatched slot count to be rejected")


def test_move_legacy_cache_detaches_and_preserves_structure():
    source = (
        (torch.randn(1, 2, 3, 4, requires_grad=True), torch.randn(1, 2, 3, 4)),
        (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),
    )

    moved = move_legacy_cache(source, "cpu")

    assert len(moved) == 2
    assert not moved[0][0].requires_grad
    for source_layer, moved_layer in zip(source, moved, strict=True):
        for source_tensor, moved_tensor in zip(source_layer, moved_layer, strict=True):
            torch.testing.assert_close(source_tensor, moved_tensor)


def test_materialize_token_tail_matches_full_native_prefill():
    torch.manual_seed(4)
    model = _tiny_qwen()
    prefix = [1, 2, 3, 4]
    suffix = [5, 6, 7]
    prefix_cache = _normal_cache(prefill_legacy_cache(model, prefix))

    tail = materialize_token_tail(
        model, suffix, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )
    full = _normal_cache(prefill_legacy_cache(model, [*prefix, *suffix]))

    for (tail_key, tail_value), (full_key, full_value) in zip(tail, full, strict=True):
        torch.testing.assert_close(tail_key, full_key[:, :, -len(suffix) :])
        torch.testing.assert_close(tail_value, full_value[:, :, -len(suffix) :])


def test_sender_only_write_adapter_gets_gradients_and_removes_hooks():
    torch.manual_seed(5)
    model = _tiny_qwen()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefix = [1, 2, 3, 4]
    prefix_cache = _normal_cache(prefill_legacy_cache(model, prefix))
    writer = LayerwiseKVWriteAdapter(
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        rank=4,
    )
    capsule = SoftTailCapsule(
        model.model.embed_tokens.weight[[5, 6]].detach(), write_adapter=writer
    )

    tail = capsule.materialize(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )
    sum(tensor.float().square().mean() for layer in tail for tensor in layer).backward()

    assert writer.blocks[0].up.weight.grad is not None
    assert torch.count_nonzero(writer.blocks[0].up.weight.grad).item() > 0
    assert not model.model.layers[0]._forward_hooks


def test_depth_gradient_boundary_isolates_early_base_from_deep_residual():
    torch.manual_seed(51)
    model = _tiny_qwen(num_hidden_layers=4)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefix = [1, 2, 3, 4]
    prefix_cache = _normal_cache(prefill_legacy_cache(model, prefix))
    writer = LayerwiseKVWriteAdapter(
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        rank=4,
    )
    capsule = SoftTailCapsule(
        model.model.embed_tokens.weight[[5, 6]].detach(), write_adapter=writer
    )

    tail = capsule.materialize(
        model,
        prefix_cache=prefix_cache,
        prefix_tokens=len(prefix),
        gradient_boundary_layers=2,
    )
    isolated = detach_cache_layer_prefix(tail, 2)
    sum(
        tensor.float().square().mean()
        for layer in isolated[2:]
        for tensor in layer
    ).backward()

    assert capsule.embeddings.grad is None
    assert writer.blocks[0].up.weight.grad is None
    for block in writer.blocks[1:]:
        assert block.up.weight.grad is not None
        assert torch.count_nonzero(block.up.weight.grad).item() > 0
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_source_read_bottleneck_preserves_early_kv_and_full_gradient_path():
    torch.manual_seed(52)
    model = _tiny_qwen(num_hidden_layers=4)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefix = [1, 2, 3, 4]
    prefix_cache = _normal_cache(prefill_legacy_cache(model, prefix))
    writer = LayerwiseKVWriteAdapter(
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        rank=4,
    )
    capsule = SoftTailCapsule(
        model.model.embed_tokens.weight[[5, 6]].detach(), write_adapter=writer
    )

    unrestricted = capsule.materialize(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )
    capsule.source_read_layers = 2
    bottlenecked = capsule.materialize(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )

    for layer_index in range(2):
        for expected, actual in zip(
            unrestricted[layer_index], bottlenecked[layer_index], strict=True
        ):
            torch.testing.assert_close(actual, expected)
    assert any(
        not torch.equal(expected, actual)
        for layer_index in range(2, 4)
        for expected, actual in zip(
            unrestricted[layer_index], bottlenecked[layer_index], strict=True
        )
    )

    sum(
        tensor.float().square().mean()
        for layer in bottlenecked[2:]
        for tensor in layer
    ).backward()
    assert capsule.embeddings.grad is not None
    assert torch.count_nonzero(capsule.embeddings.grad).item() > 0
    for block in writer.blocks:
        assert block.up.weight.grad is not None
        assert torch.count_nonzero(block.up.weight.grad).item() > 0
    assert all(not layer._forward_pre_hooks for layer in model.model.layers)
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_mask_source_prefix_attention_preserves_capsule_causal_mask():
    minimum = torch.finfo(torch.float32).min
    attention_mask = torch.zeros(1, 1, 2, 6)
    attention_mask[..., 0, 5] = minimum

    masked = mask_source_prefix_attention(attention_mask, prefix_tokens=4)

    assert torch.all(masked[..., :4] == minimum)
    torch.testing.assert_close(masked[..., 4:], attention_mask[..., 4:])
    torch.testing.assert_close(attention_mask[..., :4], torch.zeros(1, 1, 2, 4))


def test_detach_cache_layer_prefix_preserves_values_and_validates_split():
    cache = tuple(
        (
            torch.randn(1, 2, 3, 4, requires_grad=True),
            torch.randn(1, 2, 3, 4, requires_grad=True),
        )
        for _ in range(3)
    )

    isolated = detach_cache_layer_prefix(cache, 2)

    for index, (source_layer, isolated_layer) in enumerate(
        zip(cache, isolated, strict=True)
    ):
        for source, result in zip(source_layer, isolated_layer, strict=True):
            torch.testing.assert_close(source, result)
            assert result.requires_grad is (index >= 2)
    for invalid in (0, 3):
        try:
            detach_cache_layer_prefix(cache, invalid)
        except ValueError as error:
            assert "leave at least one early and one deep" in str(error)
        else:
            raise AssertionError("expected invalid cache split to be rejected")


def test_zero_initialized_write_adapter_preserves_plain_capsule_output():
    torch.manual_seed(6)
    model = _tiny_qwen()
    prefix = [1, 2, 3, 4]
    prefix_cache = _normal_cache(prefill_legacy_cache(model, prefix))
    embeddings = model.model.embed_tokens.weight[[5, 6]].detach()
    plain = SoftTailCapsule(embeddings)
    adapted = SoftTailCapsule(
        embeddings,
        write_adapter=LayerwiseKVWriteAdapter(
            hidden_size=model.config.hidden_size,
            num_hidden_layers=model.config.num_hidden_layers,
            rank=4,
        ),
    )

    plain_tail = plain.materialize(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )
    adapted_tail = adapted.materialize(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )

    for plain_layer, adapted_layer in zip(plain_tail, adapted_tail, strict=True):
        for plain_tensor, adapted_tensor in zip(plain_layer, adapted_layer, strict=True):
            torch.testing.assert_close(plain_tensor, adapted_tensor)


def test_load_soft_tail_capsule_restores_write_adapter(tmp_path):
    writer = LayerwiseKVWriteAdapter(
        hidden_size=16, num_hidden_layers=3, rank=2, scale=0.5
    )
    with torch.no_grad():
        writer.blocks[0].up.weight.fill_(0.25)
    checkpoint = tmp_path / "writer_capsule.pt"
    torch.save(
        {
            "embeddings": torch.randn(2, 16),
            "config": {"slots": 2},
            "write_adapter_config": writer.checkpoint_config(),
            "write_adapter_state": writer.state_dict(),
        },
        checkpoint,
    )

    capsule, _ = load_soft_tail_capsule(checkpoint)

    assert capsule.write_adapter is not None
    assert capsule.write_adapter.rank == 2
    torch.testing.assert_close(
        capsule.write_adapter.blocks[0].up.weight,
        torch.full_like(capsule.write_adapter.blocks[0].up.weight, 0.25),
    )


def test_load_soft_tail_capsule_restores_source_read_bottleneck(tmp_path):
    checkpoint = tmp_path / "bottleneck_capsule.pt"
    torch.save(
        {
            "embeddings": torch.randn(2, 16),
            "config": {"slots": 2},
            "materialization_config": {"source_read_layers": 2},
        },
        checkpoint,
    )

    capsule, _ = load_soft_tail_capsule(checkpoint)

    assert capsule.source_read_layers == 2


def test_zero_initialized_boundary_writer_preserves_base_and_receives_gradient():
    torch.manual_seed(53)
    model = _tiny_qwen(num_hidden_layers=4)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefix = [1, 2, 3, 4]
    prefix_cache = _normal_cache(prefill_legacy_cache(model, prefix))
    embeddings = model.model.embed_tokens.weight[[5, 6]].detach()
    base_writer = LayerwiseKVWriteAdapter(
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        rank=4,
    )
    base = SoftTailCapsule(embeddings, write_adapter=base_writer)
    boundary = BoundaryKVWriteAdapter(
        hidden_size=model.config.hidden_size,
        layer_index=1,
        rank=4,
    )
    adapted = SoftTailCapsule(
        embeddings,
        write_adapter=LayerwiseKVWriteAdapter(
            hidden_size=model.config.hidden_size,
            num_hidden_layers=model.config.num_hidden_layers,
            rank=4,
        ),
        boundary_write_adapter=boundary,
    )
    adapted.write_adapter.load_state_dict(base_writer.state_dict())
    adapted.embeddings.requires_grad_(False)
    adapted.write_adapter.requires_grad_(False)

    expected = base.materialize(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )
    actual = adapted.materialize(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )
    for expected_layer, actual_layer in zip(expected, actual, strict=True):
        for expected_tensor, actual_tensor in zip(
            expected_layer, actual_layer, strict=True
        ):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    sum(
        tensor.float().square().mean()
        for layer in actual[2:]
        for tensor in layer
    ).backward()
    assert adapted.embeddings.grad is None
    assert all(
        parameter.grad is None for parameter in adapted.write_adapter.parameters()
    )
    assert boundary.block.up.weight.grad is not None
    assert torch.count_nonzero(boundary.block.up.weight.grad).item() > 0
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_zero_boundary_and_source_mask_preserve_transmitted_base_layers():
    torch.manual_seed(59)
    model = _tiny_qwen(num_hidden_layers=4)
    prefix = [1, 2, 3, 4]
    prefix_cache = _normal_cache(prefill_legacy_cache(model, prefix))
    embeddings = model.model.embed_tokens.weight[[5, 6]].detach()
    writer = LayerwiseKVWriteAdapter(
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        rank=4,
    )
    base = SoftTailCapsule(embeddings, write_adapter=writer)
    decomposed = SoftTailCapsule(
        embeddings,
        write_adapter=LayerwiseKVWriteAdapter(
            hidden_size=model.config.hidden_size,
            num_hidden_layers=model.config.num_hidden_layers,
            rank=4,
        ),
        boundary_write_adapter=BoundaryKVWriteAdapter(
            hidden_size=model.config.hidden_size,
            layer_index=1,
            rank=4,
        ),
        source_read_layers=2,
    )
    decomposed.write_adapter.load_state_dict(writer.state_dict())

    expected = base.materialize(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )
    actual = decomposed.materialize(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )

    # The source mask begins at block 2, after K/V layer 1 is projected, and
    # the boundary residual is exactly zero.  The transmitted first-two-layer
    # packet must therefore be bitwise identical to the unrestricted base.
    for expected_layer, actual_layer in zip(expected[:2], actual[:2], strict=True):
        for expected_tensor, actual_tensor in zip(
            expected_layer, actual_layer, strict=True
        ):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert any(
        not torch.equal(expected_tensor, actual_tensor)
        for expected_layer, actual_layer in zip(expected[2:], actual[2:], strict=True)
        for expected_tensor, actual_tensor in zip(
            expected_layer, actual_layer, strict=True
        )
    )


def test_load_soft_tail_capsule_restores_boundary_writer(tmp_path):
    boundary = BoundaryKVWriteAdapter(
        hidden_size=16, layer_index=1, rank=2, scale=0.5
    )
    with torch.no_grad():
        boundary.block.up.weight.fill_(0.125)
    checkpoint = tmp_path / "boundary_capsule.pt"
    torch.save(
        {
            "embeddings": torch.randn(2, 16),
            "config": {"slots": 2},
            "boundary_write_adapter_config": boundary.checkpoint_config(),
            "boundary_write_adapter_state": boundary.state_dict(),
        },
        checkpoint,
    )

    capsule, _ = load_soft_tail_capsule(checkpoint)

    assert capsule.boundary_write_adapter is not None
    assert capsule.boundary_write_adapter.layer_index == 1
    torch.testing.assert_close(
        capsule.boundary_write_adapter.block.up.weight,
        torch.full_like(
            capsule.boundary_write_adapter.block.up.weight, 0.125
        ),
    )


def test_progressive_capsule_reuses_exact_core_and_isolates_gradients():
    torch.manual_seed(61)
    model = _tiny_qwen(num_hidden_layers=4)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefix = [1, 2, 3, 4]
    prefix_cache = _normal_cache(prefill_legacy_cache(model, prefix))
    core = SoftTailCapsule(
        model.model.embed_tokens.weight[[5, 6]].detach(),
        write_adapter=LayerwiseKVWriteAdapter(
            hidden_size=model.config.hidden_size,
            num_hidden_layers=model.config.num_hidden_layers,
            rank=4,
        ),
        boundary_write_adapter=BoundaryKVWriteAdapter(
            hidden_size=model.config.hidden_size,
            layer_index=1,
            rank=4,
        ),
        source_read_layers=2,
    )
    progressive = ProgressiveSoftTailCapsule(
        core,
        model.model.embed_tokens.weight[[7, 8]].detach(),
        refinement_boundary_write_adapter=BoundaryKVWriteAdapter(
            hidden_size=model.config.hidden_size,
            layer_index=1,
            rank=4,
        ),
    )

    standalone_core = progressive.materialize_core(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )
    reused_core, refinement = progressive.materialize_parts(
        model,
        prefix_cache=prefix_cache,
        prefix_tokens=len(prefix),
        core_tail=standalone_core,
    )
    full = progressive.materialize(
        model, prefix_cache=prefix_cache, prefix_tokens=len(prefix)
    )

    for standalone_layer, reused_layer, full_layer in zip(
        standalone_core, reused_core, full, strict=True
    ):
        for standalone, reused, full_tensor in zip(
            standalone_layer, reused_layer, full_layer, strict=True
        ):
            assert reused.data_ptr() == standalone.data_ptr()
            torch.testing.assert_close(reused, standalone, rtol=0, atol=0)
            torch.testing.assert_close(
                full_tensor[:, :, : progressive.core_slots],
                standalone,
                rtol=0,
                atol=0,
            )
    assert all(layer[0].shape[-2] == progressive.refinement_slots for layer in refinement)

    sum(
        tensor.float().square().mean()
        for layer in refinement
        for tensor in layer
    ).backward()
    assert progressive.refinement_embeddings.grad is not None
    assert torch.count_nonzero(progressive.refinement_embeddings.grad).item() > 0
    assert all(parameter.grad is None for parameter in progressive.core.parameters())
    boundary = progressive.refinement_boundary_write_adapter
    assert boundary.block.up.weight.grad is not None
    assert torch.count_nonzero(boundary.block.up.weight.grad).item() > 0
    assert all(not layer._forward_hooks for layer in model.model.layers)
    assert all(not layer._forward_pre_hooks for layer in model.model.layers)
