import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from xmodel_kv.policy_imprinting import prefill_legacy_cache
from xmodel_kv.semantic_kv_summary import (
    TrainableKVSummary,
    concatenate_caches,
    even_bin_mean,
    mean_pool_segment_cache,
    teacher_forced_answer_logits,
)


def _tiny_qwen() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=1_000_000,
        attention_dropout=0.0,
        use_cache=True,
    )
    return Qwen3ForCausalLM(config).eval()


def test_even_bin_mean_preserves_axis_and_expected_values():
    tensor = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1)
    pooled = even_bin_mean(tensor, 4, sequence_dim=2)
    assert pooled.shape == (1, 1, 4, 1)
    torch.testing.assert_close(
        pooled.flatten(), torch.tensor([0.5, 2.5, 4.5, 6.5])
    )


def test_full_length_segment_rebuild_preserves_continuation_logits():
    torch.manual_seed(41)
    model = _tiny_qwen()
    prefix = [4, 5, 6]
    segment = [7, 8, 9, 10]
    query = [11, 12, 13]
    answer = [14, 15]
    prefix_cache = prefill_legacy_cache(model, prefix)
    full_cache = prefill_legacy_cache(model, prefix + segment)
    rebuilt_segment = mean_pool_segment_cache(
        model,
        full_cache,
        segment_start=len(prefix),
        segment_end=len(prefix) + len(segment),
        target_start=len(prefix),
        tokens=len(segment),
    )
    rebuilt = concatenate_caches(prefix_cache, rebuilt_segment)
    original_logits = teacher_forced_answer_logits(
        model,
        legacy_cache=full_cache,
        cached_tokens=len(prefix) + len(segment),
        query_ids=query,
        answer_ids=answer,
    )
    rebuilt_logits = teacher_forced_answer_logits(
        model,
        legacy_cache=rebuilt,
        cached_tokens=len(prefix) + len(segment),
        query_ids=query,
        answer_ids=answer,
    )
    torch.testing.assert_close(rebuilt_logits, original_logits, atol=3e-5, rtol=3e-5)


def test_trainable_summary_receives_gradients_through_frozen_model():
    torch.manual_seed(43)
    model = _tiny_qwen()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefix = [4, 5, 6]
    segment = [7, 8, 9, 10, 11, 12]
    prefix_cache = prefill_legacy_cache(model, prefix)
    full_cache = prefill_legacy_cache(model, prefix + segment)
    initial = mean_pool_segment_cache(
        model,
        full_cache,
        segment_start=len(prefix),
        segment_end=len(prefix) + len(segment),
        target_start=len(prefix),
        tokens=3,
    )
    summary = TrainableKVSummary(initial)
    cache = concatenate_caches(
        prefix_cache,
        summary.cache(dtype=prefix_cache[0][0].dtype, device="cpu"),
    )
    logits = teacher_forced_answer_logits(
        model,
        legacy_cache=cache,
        cached_tokens=len(prefix) + summary.tokens,
        query_ids=[13, 14],
        answer_ids=[15, 16],
    )
    logits.square().mean().backward()
    assert all(parameter.grad is not None for parameter in summary.parameters())
    assert sum(float(parameter.grad.abs().sum()) for parameter in summary.parameters()) > 0
