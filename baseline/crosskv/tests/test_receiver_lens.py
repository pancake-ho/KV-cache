import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from xmodel_kv.policy_imprinting import prefill_legacy_cache
from xmodel_kv.receiver_lens import (
    ReceiverLens,
    cache_after_memory_eviction,
    cache_with_lens,
    extend_cache_with_tokens,
)
from xmodel_kv.rope import model_rope_cos_sin, remove_rope
from xmodel_kv.soft_tail_capsule import LayerwiseKVWriteAdapter


def _tiny_qwen(num_hidden_layers=4):
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


def _cache(model, ids):
    return tuple(
        (key.clone(), value.clone())
        for key, value in prefill_legacy_cache(model, ids)
    )


def test_receiver_lens_matches_native_suffix_and_preserves_prefix_bits():
    torch.manual_seed(71)
    model = _tiny_qwen()
    prefix_ids = [1, 2, 3, 4, 5]
    lens_ids = [6, 7]
    prefix = _cache(model, prefix_ids)
    before = tuple((key.clone(), value.clone()) for key, value in prefix)
    lens = ReceiverLens(model.model.embed_tokens.weight[lens_ids].detach())

    tail = lens.materialize(
        model, prefix_cache=prefix, prefix_tokens=len(prefix_ids)
    )
    native = _cache(model, [*prefix_ids, *lens_ids])

    for prefix_layer, before_layer in zip(prefix, before, strict=True):
        for actual, expected in zip(prefix_layer, before_layer, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for tail_layer, native_layer in zip(tail, native, strict=True):
        for actual, expected in zip(tail_layer, native_layer, strict=True):
            torch.testing.assert_close(actual, expected[:, :, -len(lens_ids) :])


def test_receiver_lens_detaches_semantic_memory_but_trains_local_writer():
    torch.manual_seed(72)
    model = _tiny_qwen()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    raw_prefix = _cache(model, [1, 2, 3, 4, 5])
    prefix = tuple(
        (
            key.detach().requires_grad_(True),
            value.detach().requires_grad_(True),
        )
        for key, value in raw_prefix
    )
    writer = LayerwiseKVWriteAdapter(
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        rank=4,
    )
    lens = ReceiverLens(
        model.model.embed_tokens.weight[[6, 7]].detach(),
        write_adapter=writer,
    )

    tail = lens.materialize(model, prefix_cache=prefix, prefix_tokens=5)
    sum(
        tensor.float().square().mean()
        for layer in tail
        for tensor in layer
    ).backward()

    assert lens.embeddings.grad is not None
    assert torch.count_nonzero(lens.embeddings.grad).item() > 0
    assert all(key.grad is None and value.grad is None for key, value in prefix)
    assert writer.blocks[0].up.weight.grad is not None
    assert torch.count_nonzero(writer.blocks[0].up.weight.grad).item() > 0
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_receiver_lens_can_be_relocated_after_memory_eviction():
    torch.manual_seed(73)
    model = _tiny_qwen()
    prompt_ids = [1, 2, 3]
    memory_ids = [4, 5]
    lens_ids = [6, 7]
    prompt = _cache(model, prompt_ids)
    prefix = _cache(model, [*prompt_ids, *memory_ids])
    lens = ReceiverLens(model.model.embed_tokens.weight[lens_ids].detach())
    tail = lens.materialize(model, prefix_cache=prefix, prefix_tokens=5)

    retained = cache_after_memory_eviction(
        model,
        receiver_prompt_cache=prompt,
        receiver_prompt_tokens=3,
        lens_cache=tail,
        lens_source_start=5,
    )

    assert all(key.shape[-2] == 5 for key, _ in retained)
    for layer_index, ((old_key, old_value), (new_key, new_value)) in enumerate(
        zip(tail, retained, strict=True)
    ):
        new_key = new_key[:, :, -2:]
        new_value = new_value[:, :, -2:]
        source_positions = torch.arange(5, 7)
        target_positions = torch.arange(3, 5)
        source_cos, source_sin = model_rope_cos_sin(
            model, source_positions, device=old_key.device, dtype=torch.float32
        )
        target_cos, target_sin = model_rope_cos_sin(
            model, target_positions, device=new_key.device, dtype=torch.float32
        )
        old_content = remove_rope(
            old_key.float(), source_cos, source_sin, sequence_dim=2
        )
        new_content = remove_rope(
            new_key.float(), target_cos, target_sin, sequence_dim=2
        )
        torch.testing.assert_close(new_content, old_content, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(new_value, old_value)


def test_cache_with_lens_leaves_inputs_unmodified():
    torch.manual_seed(74)
    model = _tiny_qwen()
    prefix = _cache(model, [1, 2, 3])
    lens = _cache(model, [4, 5])
    prefix_before = tuple((key.clone(), value.clone()) for key, value in prefix)
    lens_before = tuple((key.clone(), value.clone()) for key, value in lens)

    combined = cache_with_lens(prefix, lens)

    assert all(key.shape[-2] == 5 for key, _ in combined)
    for actual_layer, expected_layer in zip(prefix, prefix_before, strict=True):
        for actual, expected in zip(actual_layer, expected_layer, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_layer, expected_layer in zip(lens, lens_before, strict=True):
        for actual, expected in zip(actual_layer, expected_layer, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_receiver_lens_rejects_inconsistent_prefix_length():
    model = _tiny_qwen()
    prefix = _cache(model, [1, 2, 3])
    lens = ReceiverLens(model.model.embed_tokens.weight[[4, 5]].detach())

    try:
        lens.materialize(model, prefix_cache=prefix, prefix_tokens=4)
    except ValueError as error:
        assert "prefix_tokens" in str(error)
    else:
        raise AssertionError("expected mismatched prefix length to fail")


def test_extend_cache_with_tokens_matches_full_prefill():
    torch.manual_seed(75)
    model = _tiny_qwen()
    prefix = _cache(model, [1, 2, 3])

    extended = extend_cache_with_tokens(
        model, prefix_cache=prefix, prefix_tokens=3, token_ids=[4, 5]
    )
    expected = _cache(model, [1, 2, 3, 4, 5])

    for actual_layer, expected_layer in zip(extended, expected, strict=True):
        for actual, target in zip(actual_layer, expected_layer, strict=True):
            torch.testing.assert_close(actual, target)
