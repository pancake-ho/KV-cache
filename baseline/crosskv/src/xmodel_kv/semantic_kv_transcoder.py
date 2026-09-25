from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import DynamicCache

from .rope import apply_rope, model_rope_cos_sin, remove_rope
from .semantic_kv_summary import LegacyCache, _legacy_cache


class TailReadoutKVTranscoder(nn.Module):
    """Amortized source-cache to receiver-native compact-KV transcoder.

    Learned readout embeddings are appended to an already materialized source
    cache.  Their final hidden states form a fixed-rate semantic code, and small
    per-layer decoders synthesize a target model's native K/V tensors.  The base
    models remain frozen.
    """

    def __init__(
        self,
        source_config,
        target_config,
        *,
        slots: int,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()
        if slots < 1 or latent_dim < 1:
            raise ValueError("slots and latent_dim must be positive")
        self.slots = int(slots)
        self.latent_dim = int(latent_dim)
        self.source_hidden_size = int(source_config.hidden_size)
        self.target_layers = int(target_config.num_hidden_layers)
        self.target_kv_heads = int(target_config.num_key_value_heads)
        self.target_head_dim = int(
            getattr(
                target_config,
                "head_dim",
                target_config.hidden_size // target_config.num_attention_heads,
            )
        )

        self.readout_embeddings = nn.Parameter(
            torch.empty(self.slots, self.source_hidden_size)
        )
        self.encoder_norm = nn.LayerNorm(self.source_hidden_size)
        self.encoder_in = nn.Linear(self.source_hidden_size, self.latent_dim)
        self.encoder_out = nn.Linear(self.latent_dim, self.latent_dim)
        self.decoders = nn.ModuleList(
            nn.Linear(
                self.latent_dim,
                2 * self.target_kv_heads * self.target_head_dim,
            )
            for _ in range(self.target_layers)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.readout_embeddings, mean=0.0, std=0.02)
        for module in (self.encoder_in, self.encoder_out, *self.decoders):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)

    def encode(
        self,
        source_model,
        *,
        source_cache: Any,
        source_cached_tokens: int,
        quant_bits: int | None = None,
        straight_through: bool = False,
    ) -> torch.Tensor:
        device = source_model.model.embed_tokens.weight.device
        dtype = source_model.model.embed_tokens.weight.dtype
        positions = torch.arange(
            source_cached_tokens,
            source_cached_tokens + self.slots,
            device=device,
            dtype=torch.long,
        )
        cache = DynamicCache(
            ddp_cache_data=_legacy_cache(source_cache), config=source_model.config
        )
        output = source_model.model(
            inputs_embeds=self.readout_embeddings.to(device=device, dtype=dtype).unsqueeze(0),
            attention_mask=torch.ones(
                (1, source_cached_tokens + self.slots),
                dtype=torch.long,
                device=device,
            ),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state.float()
        latent = self.encoder_out(
            torch.nn.functional.gelu(self.encoder_in(self.encoder_norm(hidden)))
        )
        return fake_quantize_latent(
            latent, bits=quant_bits, straight_through=straight_through
        )

    def decode(
        self,
        target_model,
        latent: torch.Tensor,
        *,
        target_start: int,
        dtype: torch.dtype | None = None,
    ) -> LegacyCache:
        if latent.shape != (1, self.slots, self.latent_dim):
            raise ValueError(
                "latent must have shape "
                f"(1, {self.slots}, {self.latent_dim}), got {tuple(latent.shape)}"
            )
        device = target_model.model.embed_tokens.weight.device
        dtype = dtype or target_model.model.embed_tokens.weight.dtype
        positions = torch.arange(
            target_start,
            target_start + self.slots,
            device=device,
            dtype=torch.long,
        )
        cos, sin = model_rope_cos_sin(
            target_model, positions, device=device, dtype=torch.float32
        )
        layers = []
        for decoder in self.decoders:
            decoded = decoder(latent.to(device=device, dtype=decoder.weight.dtype))
            decoded = decoded.view(
                1,
                self.slots,
                2,
                self.target_kv_heads,
                self.target_head_dim,
            )
            content_key = decoded[:, :, 0].transpose(1, 2)
            value = decoded[:, :, 1].transpose(1, 2)
            key = apply_rope(content_key, cos, sin, sequence_dim=2)
            layers.append((key.to(dtype=dtype), value.to(dtype=dtype)))
        return tuple(layers)

    def forward(
        self,
        source_model,
        target_model,
        *,
        source_cache: Any,
        source_cached_tokens: int,
        target_start: int,
        quant_bits: int | None = None,
        straight_through: bool = False,
    ) -> tuple[torch.Tensor, LegacyCache]:
        latent = self.encode(
            source_model,
            source_cache=source_cache,
            source_cached_tokens=source_cached_tokens,
            quant_bits=quant_bits,
            straight_through=straight_through,
        )
        cache = self.decode(target_model, latent, target_start=target_start)
        return latent, cache

    @property
    def receiver_cache_elements(self) -> int:
        return (
            self.slots
            * self.target_layers
            * 2
            * self.target_kv_heads
            * self.target_head_dim
        )


class ConditionedMultiLayerKVTranscoder(TailReadoutKVTranscoder):
    """Task-conditioned readout with global/evidence slots and layer fusion.

    The first ``global_slots`` use document-only learned readout embeddings.  The
    remaining slots additionally receive a contextual embedding of Agent A's
    assigned question.  Readout states from early, middle, and final source
    layers are mixed independently for every slot before target-cache decoding.

    This keeps the transmitted representation identical to
    :class:`TailReadoutKVTranscoder` (``slots * latent_dim`` scalars); only the
    source-side extraction function changes.
    """

    def __init__(
        self,
        source_config,
        target_config,
        *,
        slots: int,
        latent_dim: int = 256,
        global_slots: int | None = None,
    ) -> None:
        super().__init__(
            source_config,
            target_config,
            slots=slots,
            latent_dim=latent_dim,
        )
        if global_slots is None:
            global_slots = slots // 2
        if not 0 <= global_slots < slots:
            raise ValueError("global_slots must be in [0, slots)")
        self.global_slots = int(global_slots)
        source_layers = int(source_config.num_hidden_layers)
        self.source_layer_indices = tuple(
            sorted({max(1, source_layers // 3), max(1, 2 * source_layers // 3), source_layers})
        )
        self.condition_norm = nn.LayerNorm(self.source_hidden_size)
        self.condition_in = nn.Linear(self.source_hidden_size, self.latent_dim)
        self.condition_out = nn.Linear(self.latent_dim, self.source_hidden_size)
        self.condition_slot_scale = nn.Parameter(torch.ones(self.slots, 1))
        evidence_mask = torch.zeros(self.slots, 1)
        evidence_mask[self.global_slots :] = 1.0
        self.register_buffer("evidence_mask", evidence_mask, persistent=False)
        self.layer_mix_logits = nn.Parameter(
            torch.zeros(self.slots, len(self.source_layer_indices))
        )
        self.reset_conditioned_parameters()

    def reset_conditioned_parameters(self) -> None:
        for module in (self.condition_in, self.condition_out):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.condition_slot_scale.fill_(1.0)
            # Start close to the original final-layer readout while leaving
            # non-zero probability and gradients for earlier layers.
            self.layer_mix_logits.zero_()
            self.layer_mix_logits[:, -1] = 2.0

    def encode(
        self,
        source_model,
        *,
        source_cache: Any,
        source_cached_tokens: int,
        condition_state: torch.Tensor,
        quant_bits: int | None = None,
        straight_through: bool = False,
    ) -> torch.Tensor:
        if condition_state.shape not in (
            (1, self.source_hidden_size),
            (1, 1, self.source_hidden_size),
        ):
            raise ValueError(
                "condition_state must have shape "
                f"(1, {self.source_hidden_size}) or (1, 1, {self.source_hidden_size})"
            )
        device = source_model.model.embed_tokens.weight.device
        dtype = source_model.model.embed_tokens.weight.dtype
        # Dataset preparation may cache this frozen vector under
        # ``torch.inference_mode``.  Clone it in the training context so
        # LayerNorm can safely save the input for parameter gradients.
        condition_state = (
            condition_state.reshape(1, self.source_hidden_size).float().clone()
        )
        conditioned = self.condition_out(
            torch.nn.functional.gelu(
                self.condition_in(self.condition_norm(condition_state))
            )
        )
        conditioned = conditioned.unsqueeze(1) * (
            self.condition_slot_scale * self.evidence_mask
        ).unsqueeze(0)
        readout = self.readout_embeddings.unsqueeze(0) + conditioned
        positions = torch.arange(
            source_cached_tokens,
            source_cached_tokens + self.slots,
            device=device,
            dtype=torch.long,
        )
        cache = DynamicCache(
            ddp_cache_data=_legacy_cache(source_cache), config=source_model.config
        )
        output = source_model.model(
            inputs_embeds=readout.to(device=device, dtype=dtype),
            attention_mask=torch.ones(
                (1, source_cached_tokens + self.slots),
                dtype=torch.long,
                device=device,
            ),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=cache,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        selected = torch.stack(
            [output.hidden_states[index].float() for index in self.source_layer_indices],
            dim=2,
        )
        mix = self.layer_mix_logits.softmax(dim=-1).view(
            1, self.slots, len(self.source_layer_indices), 1
        )
        hidden = (selected * mix).sum(dim=2)
        latent = self.encoder_out(
            torch.nn.functional.gelu(self.encoder_in(self.encoder_norm(hidden)))
        )
        return fake_quantize_latent(
            latent, bits=quant_bits, straight_through=straight_through
        )

    def forward(
        self,
        source_model,
        target_model,
        *,
        source_cache: Any,
        source_cached_tokens: int,
        target_start: int,
        condition_state: torch.Tensor,
        quant_bits: int | None = None,
        straight_through: bool = False,
    ) -> tuple[torch.Tensor, LegacyCache]:
        latent = self.encode(
            source_model,
            source_cache=source_cache,
            source_cached_tokens=source_cached_tokens,
            condition_state=condition_state,
            quant_bits=quant_bits,
            straight_through=straight_through,
        )
        cache = self.decode(target_model, latent, target_start=target_start)
        return latent, cache


class ReceiverManifoldKVTranscoder(TailReadoutKVTranscoder):
    """Decode the packet through frozen receiver layers instead of free K/V.

    A learned projection maps each source latent slot to one receiver soft-token
    embedding.  Running those embeddings after the receiver's real system cache
    materializes a compact K/V cache constrained to the receiver model's native
    activation manifold.  Wire size and sequence-slot count are unchanged.
    """

    def __init__(
        self,
        source_config,
        target_config,
        *,
        slots: int,
        latent_dim: int = 256,
    ) -> None:
        super().__init__(
            source_config,
            target_config,
            slots=slots,
            latent_dim=latent_dim,
        )
        # The parent free-KV decoders are deliberately removed: leaving them in
        # the optimizer would make the comparison and parameter count misleading.
        del self.decoders
        self.target_hidden_size = int(target_config.hidden_size)
        self.target_input_norm = nn.LayerNorm(self.latent_dim)
        self.target_embedding_decoder = nn.Linear(
            self.latent_dim, self.target_hidden_size
        )
        self.target_base_embeddings = nn.Parameter(
            torch.empty(self.slots, self.target_hidden_size)
        )
        initializer_range = float(getattr(target_config, "initializer_range", 0.02))
        nn.init.normal_(
            self.target_embedding_decoder.weight, mean=0.0, std=0.01
        )
        nn.init.zeros_(self.target_embedding_decoder.bias)
        nn.init.normal_(
            self.target_base_embeddings, mean=0.0, std=initializer_range
        )

    def decode(
        self,
        target_model,
        latent: torch.Tensor,
        *,
        target_start: int,
        prefix_cache: Any,
        dtype: torch.dtype | None = None,
    ) -> LegacyCache:
        if latent.shape != (1, self.slots, self.latent_dim):
            raise ValueError(
                "latent must have shape "
                f"(1, {self.slots}, {self.latent_dim}), got {tuple(latent.shape)}"
            )
        device = target_model.model.embed_tokens.weight.device
        dtype = dtype or target_model.model.embed_tokens.weight.dtype
        decoded = self.target_embedding_decoder(
            self.target_input_norm(latent.float())
        ) + self.target_base_embeddings.unsqueeze(0)
        positions = torch.arange(
            target_start,
            target_start + self.slots,
            device=device,
            dtype=torch.long,
        )
        cache = DynamicCache(
            ddp_cache_data=_legacy_cache(prefix_cache), config=target_model.config
        )
        output = target_model.model(
            inputs_embeds=decoded.to(device=device, dtype=dtype),
            attention_mask=torch.ones(
                (1, target_start + self.slots), dtype=torch.long, device=device
            ),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return tuple(
            (
                key[:, :, -self.slots :],
                value[:, :, -self.slots :],
            )
            for key, value in _legacy_cache(output.past_key_values)
        )

    def forward(
        self,
        source_model,
        target_model,
        *,
        source_cache: Any,
        source_cached_tokens: int,
        target_start: int,
        prefix_cache: Any,
        quant_bits: int | None = None,
        straight_through: bool = False,
    ) -> tuple[torch.Tensor, LegacyCache]:
        latent = self.encode(
            source_model,
            source_cache=source_cache,
            source_cached_tokens=source_cached_tokens,
            quant_bits=quant_bits,
            straight_through=straight_through,
        )
        cache = self.decode(
            target_model,
            latent,
            target_start=target_start,
            prefix_cache=prefix_cache,
        )
        return latent, cache


class TailKVContentManifoldTranscoder(ReceiverManifoldKVTranscoder):
    """Encode actual post-answer tail K/V content into receiver soft slots.

    Unlike learned readout queries, this encoder consumes the K/V tensors at
    Agent A's real generated-answer positions.  Keys are de-rotated before
    early/middle/final-layer features are concatenated and adaptively pooled to
    a fixed slot budget.  The receiver side remains manifold-constrained.
    """

    def __init__(
        self,
        source_config,
        target_config,
        *,
        slots: int,
        latent_dim: int = 256,
    ) -> None:
        super().__init__(
            source_config,
            target_config,
            slots=slots,
            latent_dim=latent_dim,
        )
        del self.readout_embeddings
        del self.encoder_norm
        del self.encoder_in
        del self.encoder_out
        source_layers = int(source_config.num_hidden_layers)
        self.source_layer_indices = tuple(
            sorted(
                {
                    max(0, source_layers // 3 - 1),
                    max(0, 2 * source_layers // 3 - 1),
                    source_layers - 1,
                }
            )
        )
        self.source_kv_heads = int(source_config.num_key_value_heads)
        self.source_head_dim = int(
            getattr(
                source_config,
                "head_dim",
                source_config.hidden_size // source_config.num_attention_heads,
            )
        )
        feature_dim = (
            len(self.source_layer_indices)
            * 2
            * self.source_kv_heads
            * self.source_head_dim
        )
        self.tail_feature_dim = feature_dim
        self.tail_norm = nn.LayerNorm(feature_dim)
        self.tail_encoder_in = nn.Linear(feature_dim, self.latent_dim)
        self.tail_encoder_out = nn.Linear(self.latent_dim, self.latent_dim)
        for module in (self.tail_encoder_in, self.tail_encoder_out):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)

    def encode(
        self,
        source_model,
        *,
        source_cache: Any,
        source_cached_tokens: int,
        source_answer_tokens: int,
        quant_bits: int | None = None,
        straight_through: bool = False,
    ) -> torch.Tensor:
        pooled = self.extract_tail_features(
            source_model,
            source_cache=source_cache,
            source_cached_tokens=source_cached_tokens,
            source_answer_tokens=source_answer_tokens,
        )
        return self.encode_tail_features(
            pooled,
            quant_bits=quant_bits,
            straight_through=straight_through,
        )

    @torch.no_grad()
    def extract_tail_features(
        self,
        source_model,
        *,
        source_cache: Any,
        source_cached_tokens: int,
        source_answer_tokens: int,
    ) -> torch.Tensor:
        if not 1 <= source_answer_tokens <= source_cached_tokens:
            raise ValueError("source_answer_tokens must lie inside the source cache")
        layers = _legacy_cache(source_cache)
        start = source_cached_tokens - source_answer_tokens
        positions = torch.arange(
            start,
            source_cached_tokens,
            device=layers[0][0].device,
            dtype=torch.long,
        )
        cos, sin = model_rope_cos_sin(
            source_model,
            positions,
            device=layers[0][0].device,
            dtype=torch.float32,
        )
        features = []
        for layer_index in self.source_layer_indices:
            key, value = layers[layer_index]
            content_key = remove_rope(
                key[:, :, -source_answer_tokens:].float(),
                cos,
                sin,
                sequence_dim=2,
            )
            value = value[:, :, -source_answer_tokens:].float()
            features.extend(
                (
                    content_key.transpose(1, 2).flatten(start_dim=2),
                    value.transpose(1, 2).flatten(start_dim=2),
                )
            )
        tail = torch.cat(features, dim=-1)
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            tail.transpose(1, 2), self.slots
        ).transpose(1, 2)
        return pooled

    def encode_tail_features(
        self,
        pooled: torch.Tensor,
        *,
        quant_bits: int | None = None,
        straight_through: bool = False,
    ) -> torch.Tensor:
        if pooled.shape != (1, self.slots, self.tail_feature_dim):
            raise ValueError(
                "pooled tail features must have shape "
                f"(1, {self.slots}, {self.tail_feature_dim})"
            )
        # Cached features are usually materialized under inference mode.
        pooled = pooled.float().clone()
        latent = self.tail_encoder_out(
            torch.nn.functional.gelu(
                self.tail_encoder_in(self.tail_norm(pooled))
            )
        )
        return fake_quantize_latent(
            latent, bits=quant_bits, straight_through=straight_through
        )

    def forward(
        self,
        source_model,
        target_model,
        *,
        source_cache: Any,
        source_cached_tokens: int,
        source_answer_tokens: int,
        target_start: int,
        prefix_cache: Any,
        quant_bits: int | None = None,
        straight_through: bool = False,
    ) -> tuple[torch.Tensor, LegacyCache]:
        latent = self.encode(
            source_model,
            source_cache=source_cache,
            source_cached_tokens=source_cached_tokens,
            source_answer_tokens=source_answer_tokens,
            quant_bits=quant_bits,
            straight_through=straight_through,
        )
        cache = self.decode(
            target_model,
            latent,
            target_start=target_start,
            prefix_cache=prefix_cache,
        )
        return latent, cache


def fake_quantize_latent(
    latent: torch.Tensor,
    *,
    bits: int | None,
    straight_through: bool = False,
) -> torch.Tensor:
    if bits is None:
        return latent
    if bits not in (4, 8):
        raise ValueError("only 4-bit and 8-bit latent quantization are supported")
    qmax = float((1 << (bits - 1)) - 1)
    scale = latent.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = (latent / scale).round().clamp(-qmax, qmax) * scale
    if straight_through:
        return latent + (quantized - latent).detach()
    return quantized


def relative_cache_mse(candidate: Any, reference: Any) -> torch.Tensor:
    candidate_layers = _legacy_cache(candidate)
    reference_layers = _legacy_cache(reference)
    if len(candidate_layers) != len(reference_layers):
        raise ValueError("candidate and reference cache layer counts differ")
    squared_error = []
    squared_reference = []
    for (candidate_key, candidate_value), (reference_key, reference_value) in zip(
        candidate_layers, reference_layers, strict=True
    ):
        for predicted, expected in (
            (candidate_key, reference_key),
            (candidate_value, reference_value),
        ):
            squared_error.append((predicted.float() - expected.float()).square().mean())
            squared_reference.append(expected.float().square().mean())
    return torch.stack(squared_error).mean() / torch.stack(squared_reference).mean().clamp_min(1e-8)
