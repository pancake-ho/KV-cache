import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from xmodel_kv.policy_imprinting import prefill_legacy_cache
from xmodel_kv.semantic_kv_summary import concatenate_caches, teacher_forced_answer_logits
from xmodel_kv.semantic_kv_transcoder import (
    ConditionedMultiLayerKVTranscoder,
    ReceiverManifoldKVTranscoder,
    TailKVContentManifoldTranscoder,
    TailReadoutKVTranscoder,
    fake_quantize_latent,
)


def _tiny_qwen() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    return Qwen3ForCausalLM(config).eval()


def test_transcoder_builds_target_native_cache_and_receives_gradients():
    torch.manual_seed(0)
    source = _tiny_qwen()
    target = _tiny_qwen()
    for model in (source, target):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    source_ids = [1, 2, 3, 4, 5]
    target_prefix_ids = [6, 7, 8]
    source_cache = prefill_legacy_cache(source, source_ids)
    target_prefix = prefill_legacy_cache(target, target_prefix_ids)
    transcoder = TailReadoutKVTranscoder(
        source.config, target.config, slots=2, latent_dim=16
    )

    latent, compact = transcoder(
        source,
        target,
        source_cache=source_cache,
        source_cached_tokens=len(source_ids),
        target_start=len(target_prefix_ids),
    )
    assert latent.shape == (1, 2, 16)
    assert len(compact) == target.config.num_hidden_layers
    assert compact[0][0].shape == (1, 2, 2, 8)

    assembled = concatenate_caches(target_prefix, compact)
    logits = teacher_forced_answer_logits(
        target,
        legacy_cache=assembled,
        cached_tokens=len(target_prefix_ids) + 2,
        query_ids=[9, 10],
        answer_ids=[11, 12],
    )
    logits.float().square().mean().backward()
    assert transcoder.readout_embeddings.grad is not None
    assert transcoder.encoder_in.weight.grad is not None
    assert transcoder.decoders[0].weight.grad is not None


def test_fake_quantization_preserves_shape_and_uses_straight_through_gradient():
    latent = torch.randn(1, 3, 5, requires_grad=True)
    quantized = fake_quantize_latent(latent, bits=4, straight_through=True)
    assert quantized.shape == latent.shape
    quantized.sum().backward()
    assert torch.equal(latent.grad, torch.ones_like(latent))


def test_conditioned_multilayer_transcoder_separates_global_and_evidence_slots():
    torch.manual_seed(0)
    source = _tiny_qwen()
    target = _tiny_qwen()
    for model in (source, target):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    source_ids = [1, 2, 3, 4, 5]
    source_cache = prefill_legacy_cache(source, source_ids)
    transcoder = ConditionedMultiLayerKVTranscoder(
        source.config,
        target.config,
        slots=4,
        latent_dim=16,
        global_slots=2,
    )
    condition = torch.randn(1, source.config.hidden_size)

    latent, compact = transcoder(
        source,
        target,
        source_cache=source_cache,
        source_cached_tokens=len(source_ids),
        target_start=3,
        condition_state=condition,
    )
    assert latent.shape == (1, 4, 16)
    assert compact[0][0].shape == (1, 2, 4, 8)
    assert torch.equal(
        transcoder.evidence_mask.squeeze(-1), torch.tensor([0.0, 0.0, 1.0, 1.0])
    )

    sum(tensor.float().square().mean() for layer in compact for tensor in layer).backward()
    assert transcoder.condition_in.weight.grad is not None
    assert transcoder.layer_mix_logits.grad is not None
    assert transcoder.readout_embeddings.grad is not None


def test_receiver_manifold_transcoder_materializes_cache_through_target_model():
    torch.manual_seed(0)
    source = _tiny_qwen()
    target = _tiny_qwen()
    for model in (source, target):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    source_ids = [1, 2, 3, 4, 5]
    target_ids = [6, 7, 8]
    source_cache = prefill_legacy_cache(source, source_ids)
    target_cache = prefill_legacy_cache(target, target_ids)
    transcoder = ReceiverManifoldKVTranscoder(
        source.config, target.config, slots=2, latent_dim=16
    )

    latent, compact = transcoder(
        source,
        target,
        source_cache=source_cache,
        source_cached_tokens=len(source_ids),
        target_start=len(target_ids),
        prefix_cache=target_cache,
    )
    assert latent.shape == (1, 2, 16)
    assert compact[0][0].shape == (1, 2, 2, 8)
    assert not hasattr(transcoder, "decoders")

    assembled = concatenate_caches(target_cache, compact)
    logits = teacher_forced_answer_logits(
        target,
        legacy_cache=assembled,
        cached_tokens=len(target_ids) + 2,
        query_ids=[9, 10],
        answer_ids=[11, 12],
    )
    logits.float().square().mean().backward()
    assert transcoder.target_embedding_decoder.weight.grad is not None
    assert transcoder.target_base_embeddings.grad is not None
    assert transcoder.readout_embeddings.grad is not None


def test_tail_kv_content_transcoder_uses_real_answer_cache_positions():
    torch.manual_seed(0)
    source = _tiny_qwen()
    target = _tiny_qwen()
    for model in (source, target):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    source_ids = [1, 2, 3, 4, 5, 6]
    target_ids = [7, 8]
    source_cache = prefill_legacy_cache(source, source_ids)
    target_cache = prefill_legacy_cache(target, target_ids)
    transcoder = TailKVContentManifoldTranscoder(
        source.config, target.config, slots=3, latent_dim=16
    )

    latent, compact = transcoder(
        source,
        target,
        source_cache=source_cache,
        source_cached_tokens=len(source_ids),
        source_answer_tokens=2,
        target_start=len(target_ids),
        prefix_cache=target_cache,
    )
    assert latent.shape == (1, 3, 16)
    assert compact[0][0].shape == (1, 2, 3, 8)
    assert not hasattr(transcoder, "readout_embeddings")
    features = transcoder.extract_tail_features(
        source,
        source_cache=source_cache,
        source_cached_tokens=len(source_ids),
        source_answer_tokens=2,
    )
    assert features.shape == (1, 3, transcoder.tail_feature_dim)
    assert transcoder.encode_tail_features(features).shape == (1, 3, 16)

    assembled = concatenate_caches(target_cache, compact)
    logits = teacher_forced_answer_logits(
        target,
        legacy_cache=assembled,
        cached_tokens=len(target_ids) + 3,
        query_ids=[9, 10],
        answer_ids=[11, 12],
    )
    logits.float().square().mean().backward()
    assert transcoder.tail_encoder_in.weight.grad is not None
    assert transcoder.target_embedding_decoder.weight.grad is not None
