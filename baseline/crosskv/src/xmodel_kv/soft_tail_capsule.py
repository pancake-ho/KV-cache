from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache

from .rope import apply_rope, model_rope_cos_sin, remove_rope


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


class SoftTailCapsule(nn.Module):
    """Global soft state-emission tokens materialized as contextual native KV.

    The learned embeddings are shared across examples. Their per-example state
    comes entirely from attending to the frozen source prefix. Unlike a dense
    transcoder, the emitted K/V follows the base model's real layer-by-layer
    computation path and can therefore be transplanted without decoding an
    arbitrary latent into every receiver layer.
    """

    def __init__(
        self,
        initial_embeddings: torch.Tensor,
        *,
        write_adapter: LayerwiseKVWriteAdapter | None = None,
        boundary_write_adapter: BoundaryKVWriteAdapter | None = None,
        source_read_layers: int | None = None,
    ) -> None:
        super().__init__()
        if initial_embeddings.ndim != 2 or min(initial_embeddings.shape) < 1:
            raise ValueError("initial_embeddings must have shape [slots, hidden]")
        if source_read_layers is not None and source_read_layers < 1:
            raise ValueError("source_read_layers must be positive when specified")
        self.embeddings = nn.Parameter(initial_embeddings.detach().float().clone())
        self.write_adapter = write_adapter
        self.boundary_write_adapter = boundary_write_adapter
        self.source_read_layers = source_read_layers

    @property
    def slots(self) -> int:
        return int(self.embeddings.shape[0])

    def materialize(
        self,
        model,
        *,
        prefix_cache: Any,
        prefix_tokens: int,
        gradient_boundary_layers: int | None = None,
    ) -> LegacyCache:
        if prefix_tokens < 1:
            raise ValueError("prefix_tokens must be positive")
        if gradient_boundary_layers is not None and self.write_adapter is None:
            raise ValueError("gradient boundary requires a write adapter")
        if self.source_read_layers is not None and not (
            1 <= self.source_read_layers < model.config.num_hidden_layers
        ):
            raise ValueError(
                "source-read bottleneck must leave at least one source-reading "
                "and one source-masked layer"
            )
        layers = _legacy_cache(prefix_cache)
        device = model.model.embed_tokens.weight.device
        dtype = model.model.embed_tokens.weight.dtype
        positions = torch.arange(
            prefix_tokens,
            prefix_tokens + self.slots,
            dtype=torch.long,
            device=device,
        )
        cache = DynamicCache(ddp_cache_data=layers, config=model.config)
        active_writer = (
            nullcontext()
            if self.write_adapter is None
            else self.write_adapter.activate(
                model, gradient_boundary_layers=gradient_boundary_layers
            )
        )
        active_boundary_writer = (
            nullcontext()
            if self.boundary_write_adapter is None
            else self.boundary_write_adapter.activate(model)
        )
        source_attention = (
            nullcontext()
            if self.source_read_layers is None
            else restrict_capsule_source_attention(
                model,
                prefix_tokens=prefix_tokens,
                source_read_layers=self.source_read_layers,
            )
        )
        # Enter the base writer first so a boundary residual registered on the
        # same block observes the already-adapted transferable base state.
        with active_writer, active_boundary_writer, source_attention:
            output = model(
                inputs_embeds=self.embeddings.to(device=device, dtype=dtype).unsqueeze(0),
                attention_mask=torch.ones(
                    (1, prefix_tokens + self.slots), dtype=torch.long, device=device
                ),
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        full = _legacy_cache(output.past_key_values)
        return tuple(
            (key[:, :, -self.slots :], value[:, :, -self.slots :])
            for key, value in full
        )

    def materialization_config(self) -> dict:
        """Return inference-critical settings stored with the checkpoint."""

        return {"source_read_layers": self.source_read_layers}


class ProgressiveSoftTailCapsule(nn.Module):
    """A frozen core packet followed by independently trainable refinement slots.

    Core slots are materialized once with an existing :class:`SoftTailCapsule`.
    Refinement slots are then run as a second causal suffix with the source and
    exact core K/V as their prefix.  Consequently the low-budget packet is not
    recomputed or approximated when the high-budget path is selected: the full
    cache literally concatenates the original core tensors with a refinement
    tail.  Frozen core writers still transform refinement hidden states, while
    an optional refinement boundary writer supplies new trainable capacity.
    """

    def __init__(
        self,
        core: SoftTailCapsule,
        refinement_initial_embeddings: torch.Tensor,
        *,
        refinement_boundary_write_adapter: BoundaryKVWriteAdapter | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(core, SoftTailCapsule):
            raise TypeError("core must be a SoftTailCapsule")
        if (
            refinement_initial_embeddings.ndim != 2
            or min(refinement_initial_embeddings.shape) < 1
            or refinement_initial_embeddings.shape[1] != core.embeddings.shape[1]
        ):
            raise ValueError(
                "refinement embeddings must have shape [slots, core hidden size]"
            )
        self.core = core
        self.core.requires_grad_(False)
        self.refinement_embeddings = nn.Parameter(
            refinement_initial_embeddings.detach().float().clone()
        )
        self.refinement_boundary_write_adapter = refinement_boundary_write_adapter

    @property
    def core_slots(self) -> int:
        return self.core.slots

    @property
    def refinement_slots(self) -> int:
        return int(self.refinement_embeddings.shape[0])

    @property
    def slots(self) -> int:
        return self.core_slots + self.refinement_slots

    def materialize_core(
        self,
        model,
        *,
        prefix_cache: Any,
        prefix_tokens: int,
    ) -> LegacyCache:
        return self.core.materialize(
            model,
            prefix_cache=prefix_cache,
            prefix_tokens=prefix_tokens,
        )

    def materialize_parts(
        self,
        model,
        *,
        prefix_cache: Any,
        prefix_tokens: int,
        core_tail: Any | None = None,
    ) -> tuple[LegacyCache, LegacyCache]:
        if prefix_tokens < 1:
            raise ValueError("prefix_tokens must be positive")
        if core_tail is None:
            core_tail = self.materialize_core(
                model,
                prefix_cache=prefix_cache,
                prefix_tokens=prefix_tokens,
            )
        core_layers = _legacy_cache(core_tail)
        if len(core_layers) != model.config.num_hidden_layers or any(
            key.shape[-2] != self.core_slots or value.shape != key.shape
            for key, value in core_layers
        ):
            raise ValueError("core tail has incompatible layer or slot structure")
        source_layers = _legacy_cache(prefix_cache)
        if len(source_layers) != len(core_layers):
            raise ValueError("source and core caches have different layer counts")
        combined_prefix = tuple(
            (
                torch.cat((source_key, core_key.detach()), dim=-2),
                torch.cat((source_value, core_value.detach()), dim=-2),
            )
            for (source_key, source_value), (core_key, core_value) in zip(
                source_layers, core_layers, strict=True
            )
        )
        total_prefix_tokens = prefix_tokens + self.core_slots
        device = model.model.embed_tokens.weight.device
        dtype = model.model.embed_tokens.weight.dtype
        positions = torch.arange(
            total_prefix_tokens,
            total_prefix_tokens + self.refinement_slots,
            dtype=torch.long,
            device=device,
        )
        cache = DynamicCache(ddp_cache_data=combined_prefix, config=model.config)
        active_writer = (
            nullcontext()
            if self.core.write_adapter is None
            else self.core.write_adapter.activate(model)
        )
        active_core_boundary = (
            nullcontext()
            if self.core.boundary_write_adapter is None
            else self.core.boundary_write_adapter.activate(model)
        )
        active_refinement_boundary = (
            nullcontext()
            if self.refinement_boundary_write_adapter is None
            else self.refinement_boundary_write_adapter.activate(model)
        )
        source_attention = (
            nullcontext()
            if self.core.source_read_layers is None
            else restrict_capsule_source_attention(
                model,
                # Mask only the long source.  The cached core positions remain
                # visible to deep refinement layers.
                prefix_tokens=prefix_tokens,
                source_read_layers=self.core.source_read_layers,
            )
        )
        with (
            active_writer,
            active_core_boundary,
            active_refinement_boundary,
            source_attention,
        ):
            output = model(
                inputs_embeds=self.refinement_embeddings.to(
                    device=device, dtype=dtype
                ).unsqueeze(0),
                attention_mask=torch.ones(
                    (1, total_prefix_tokens + self.refinement_slots),
                    dtype=torch.long,
                    device=device,
                ),
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        full = _legacy_cache(output.past_key_values)
        refinement_tail = tuple(
            (
                key[:, :, -self.refinement_slots :],
                value[:, :, -self.refinement_slots :],
            )
            for key, value in full
        )
        return tuple(core_layers), refinement_tail

    def materialize(
        self,
        model,
        *,
        prefix_cache: Any,
        prefix_tokens: int,
    ) -> LegacyCache:
        core_tail, refinement_tail = self.materialize_parts(
            model,
            prefix_cache=prefix_cache,
            prefix_tokens=prefix_tokens,
        )
        return concatenate_tail_caches(core_tail, refinement_tail)


def concatenate_tail_caches(first: Any, second: Any) -> LegacyCache:
    """Concatenate two native tails without altering either component."""

    first_layers = _legacy_cache(first)
    second_layers = _legacy_cache(second)
    if len(first_layers) != len(second_layers) or not first_layers:
        raise ValueError("tail caches must have the same non-zero layer count")
    result = []
    for (first_key, first_value), (second_key, second_value) in zip(
        first_layers, second_layers, strict=True
    ):
        if (
            first_key.shape[:-2] != second_key.shape[:-2]
            or first_key.shape[-1] != second_key.shape[-1]
            or first_value.shape != first_key.shape
            or second_value.shape != second_key.shape
        ):
            raise ValueError("tail cache tensor shapes are incompatible")
        result.append(
            (
                torch.cat((first_key, second_key), dim=-2),
                torch.cat((first_value, second_value), dim=-2),
            )
        )
    return tuple(result)


class _ResidualWriteBlock(nn.Module):
    def __init__(self, hidden_size: int, rank: int, scale: float) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        self.scale = float(scale)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        normalized = F.rms_norm(hidden_states.float(), (hidden_states.shape[-1],))
        delta = self.up(F.silu(self.down(normalized))) * self.scale
        return hidden_states + delta.to(dtype=dtype)


class LayerwiseKVWriteAdapter(nn.Module):
    """Low-rank sender-only residuals that preserve native K/V projections.

    A block is attached after each decoder layer except the last while capsule
    slots are materialized. Its output affects subsequent layers' native K/V,
    but hooks are removed before any receiver computation.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        rank: int,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if min(hidden_size, num_hidden_layers, rank) < 1 or scale <= 0:
            raise ValueError("writer dimensions, rank, and scale must be positive")
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.rank = int(rank)
        self.scale = float(scale)
        self.blocks = nn.ModuleList(
            _ResidualWriteBlock(hidden_size, rank, scale)
            for _ in range(max(0, num_hidden_layers - 1))
        )

    @contextmanager
    def activate(
        self,
        model,
        *,
        gradient_boundary_layers: int | None = None,
    ):
        layers = model.model.layers
        if len(layers) != self.num_hidden_layers:
            raise ValueError("writer layer count does not match the model")
        if model.config.hidden_size != self.hidden_size:
            raise ValueError("writer hidden size does not match the model")
        if gradient_boundary_layers is not None and not (
            1 <= gradient_boundary_layers < self.num_hidden_layers
        ):
            raise ValueError(
                "gradient boundary must leave at least one early and one deep layer"
            )
        handles = []
        try:
            for layer_index, (layer, block) in enumerate(
                zip(layers[:-1], self.blocks, strict=True)
            ):
                handles.append(
                    layer.register_forward_hook(
                        _writer_hook(
                            block,
                            detach_block_input=(
                                gradient_boundary_layers is not None
                                and layer_index == gradient_boundary_layers - 1
                            ),
                        )
                    )
                )
            yield
        finally:
            for handle in handles:
                handle.remove()

    def checkpoint_config(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "rank": self.rank,
            "scale": self.scale,
        }


class BoundaryKVWriteAdapter(nn.Module):
    """A zero-initialized residual written at one sender layer boundary.

    Attaching after decoder block ``layer_index`` changes the hidden state that
    is projected into K/V at ``layer_index + 1``.  With layer 14, the residual
    is therefore serialized in K/V layer 15, the final layer of a first16
    packet.  A frozen layerwise writer can serve as a transferable base while
    only this small residual is optimized for the compact task channel.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        layer_index: int,
        rank: int,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if min(hidden_size, rank) < 1 or layer_index < 0 or scale <= 0:
            raise ValueError("boundary writer dimensions and scale are invalid")
        self.hidden_size = int(hidden_size)
        self.layer_index = int(layer_index)
        self.rank = int(rank)
        self.scale = float(scale)
        self.block = _ResidualWriteBlock(hidden_size, rank, scale)

    @contextmanager
    def activate(self, model):
        layers = model.model.layers
        if self.hidden_size != model.config.hidden_size:
            raise ValueError("boundary writer hidden size does not match the model")
        if self.layer_index >= len(layers) - 1:
            raise ValueError(
                "boundary writer must leave a following layer for K/V emission"
            )
        handle = layers[self.layer_index].register_forward_hook(
            _writer_hook(self.block)
        )
        try:
            yield
        finally:
            handle.remove()

    def checkpoint_config(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "layer_index": self.layer_index,
            "rank": self.rank,
            "scale": self.scale,
        }


def _writer_hook(block: nn.Module, *, detach_block_input: bool = False):
    def hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            return block(output.detach() if detach_block_input else output)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            hidden_states = output[0].detach() if detach_block_input else output[0]
            return (block(hidden_states), *output[1:])
        raise TypeError(f"unsupported decoder-layer output: {type(output)!r}")

    return hook


@contextmanager
def restrict_capsule_source_attention(
    model,
    *,
    prefix_tokens: int,
    source_read_layers: int,
):
    """Prevent deep emission layers from rereading the long source cache.

    The first ``source_read_layers`` decoder blocks can attend the complete
    source prefix while contextualizing the capsule slots.  Later blocks can
    attend only the capsule positions.  Unlike a gradient boundary, this keeps
    the hidden-state and autograd paths continuous: a full-depth receiver loss
    still trains the embeddings and early writer blocks, but every bit used by
    the deep capsule must first pass through the early hidden-state bottleneck.
    """

    layers = model.model.layers
    if prefix_tokens < 1:
        raise ValueError("prefix_tokens must be positive")
    if not 1 <= source_read_layers < len(layers):
        raise ValueError(
            "source-read bottleneck must leave at least one source-reading "
            "and one source-masked layer"
        )
    handles = []
    try:
        for layer in layers[source_read_layers:]:
            handles.append(
                layer.register_forward_pre_hook(
                    _source_attention_mask_hook(prefix_tokens), with_kwargs=True
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


def mask_source_prefix_attention(
    attention_mask: torch.Tensor,
    *,
    prefix_tokens: int,
) -> torch.Tensor:
    """Mask cached source keys while retaining the capsule causal submatrix."""

    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a tensor")
    if attention_mask.ndim < 2:
        raise ValueError("attention_mask must expose a key dimension")
    if prefix_tokens < 1 or prefix_tokens >= attention_mask.shape[-1]:
        raise ValueError("source prefix must leave at least one capsule key")
    masked = attention_mask.clone()
    if masked.ndim == 2:
        masked[..., :prefix_tokens] = 0
    elif masked.dtype == torch.bool:
        # Transformers' four-dimensional causal masks use True for allowed
        # attention positions when a Boolean representation is selected.
        masked[..., :prefix_tokens] = False
    elif masked.is_floating_point():
        masked[..., :prefix_tokens] = torch.finfo(masked.dtype).min
    else:
        raise TypeError("four-dimensional attention masks must be bool or floating")
    return masked


def _source_attention_mask_hook(prefix_tokens: int):
    def hook(_module, args, kwargs):
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError(
                "source-read bottleneck requires an explicit decoder attention mask"
            )
        updated = dict(kwargs)
        updated["attention_mask"] = mask_source_prefix_attention(
            attention_mask, prefix_tokens=prefix_tokens
        )
        return args, updated

    return hook


def detach_cache_layer_prefix(cache: Any, prefix_layers: int) -> LegacyCache:
    """Stop receiver losses on deep packets from updating their early base.

    This operation does not alter cache values.  It only cuts autograd edges
    for the first ``prefix_layers`` K/V pairs, allowing a larger packet to be
    trained as a deep residual on top of a separately optimized early packet.
    """

    layers = _legacy_cache(cache)
    if not 1 <= prefix_layers < len(layers):
        raise ValueError(
            "detached cache prefix must leave at least one early and one deep layer"
        )
    return tuple(
        (key.detach(), value.detach()) if index < prefix_layers else (key, value)
        for index, (key, value) in enumerate(layers)
    )


def load_soft_tail_capsule(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SoftTailCapsule, dict]:
    """Load a saved capsule while validating its portable checkpoint contract."""

    payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("capsule checkpoint must contain a mapping")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("capsule checkpoint embeddings must have shape [slots, hidden]")
    if min(embeddings.shape) < 1 or not torch.isfinite(embeddings).all():
        raise ValueError("capsule checkpoint embeddings must be non-empty and finite")
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("capsule checkpoint config must be a mapping")
    configured_slots = config.get("slots")
    if configured_slots is not None and int(configured_slots) != embeddings.shape[0]:
        raise ValueError("capsule checkpoint slot count disagrees with its config")
    writer = None
    writer_config = payload.get("write_adapter_config")
    writer_state = payload.get("write_adapter_state")
    if writer_config is not None or writer_state is not None:
        if not isinstance(writer_config, dict) or not isinstance(writer_state, dict):
            raise ValueError("writer checkpoint requires config and state mappings")
        writer = LayerwiseKVWriteAdapter(**writer_config)
        writer.load_state_dict(writer_state, strict=True)
    boundary_writer = None
    boundary_writer_config = payload.get("boundary_write_adapter_config")
    boundary_writer_state = payload.get("boundary_write_adapter_state")
    if boundary_writer_config is not None or boundary_writer_state is not None:
        if not isinstance(boundary_writer_config, dict) or not isinstance(
            boundary_writer_state, dict
        ):
            raise ValueError(
                "boundary writer checkpoint requires config and state mappings"
            )
        boundary_writer = BoundaryKVWriteAdapter(**boundary_writer_config)
        boundary_writer.load_state_dict(boundary_writer_state, strict=True)
    materialization_config = payload.get("materialization_config", {})
    if not isinstance(materialization_config, dict):
        raise ValueError("capsule materialization config must be a mapping")
    unknown_materialization = set(materialization_config) - {"source_read_layers"}
    if unknown_materialization:
        raise ValueError("capsule materialization config contains unknown fields")
    source_read_layers = materialization_config.get("source_read_layers")
    if source_read_layers is not None:
        if not isinstance(source_read_layers, int) or isinstance(
            source_read_layers, bool
        ):
            raise ValueError("source_read_layers must be an integer or null")
    return SoftTailCapsule(
        embeddings,
        write_adapter=writer,
        boundary_write_adapter=boundary_writer,
        source_read_layers=source_read_layers,
    ), config


def move_legacy_cache(
    cache: Any,
    device: str | torch.device,
    *,
    non_blocking: bool = False,
) -> LegacyCache:
    """Move a detached legacy cache without changing its layer structure."""

    return tuple(
        (
            key.detach().to(device=device, non_blocking=non_blocking),
            value.detach().to(device=device, non_blocking=non_blocking),
        )
        for key, value in _legacy_cache(cache)
    )


def materialize_token_tail(
    model,
    token_ids,
    *,
    prefix_cache: Any,
    prefix_tokens: int,
) -> LegacyCache:
    """Append hard tokens to a detached prefix and return only their native KV."""

    ids = tuple(int(token_id) for token_id in token_ids)
    if prefix_tokens < 1 or not ids:
        raise ValueError("prefix and appended token sequence must be non-empty")
    layers = _legacy_cache(prefix_cache)
    device = model.model.embed_tokens.weight.device
    positions = torch.arange(
        prefix_tokens, prefix_tokens + len(ids), dtype=torch.long, device=device
    )
    cache = DynamicCache(ddp_cache_data=layers, config=model.config)
    output = model(
        input_ids=torch.tensor([ids], dtype=torch.long, device=device),
        attention_mask=torch.ones(
            (1, prefix_tokens + len(ids)), dtype=torch.long, device=device
        ),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    full = _legacy_cache(output.past_key_values)
    return tuple(
        (key[:, :, -len(ids) :], value[:, :, -len(ids) :])
        for key, value in full
    )


def reposition_tail_cache(
    model,
    tail_cache: Any,
    *,
    source_start: int,
    target_start: int,
) -> LegacyCache:
    """Differentiably move a native tail cache between absolute RoPE positions."""

    layers = _legacy_cache(tail_cache)
    if not layers:
        raise ValueError("tail_cache must be non-empty")
    tokens = int(layers[0][0].shape[-2])
    if tokens < 1 or min(source_start, target_start) < 0:
        raise ValueError("tail length must be positive and positions non-negative")
    device = layers[0][0].device
    source_positions = torch.arange(source_start, source_start + tokens, device=device)
    target_positions = torch.arange(target_start, target_start + tokens, device=device)
    source_cos, source_sin = model_rope_cos_sin(
        model, source_positions, device=device, dtype=torch.float32
    )
    target_cos, target_sin = model_rope_cos_sin(
        model, target_positions, device=device, dtype=torch.float32
    )
    result = []
    for key, value in layers:
        if key.shape != value.shape or key.shape[-2] != tokens:
            raise ValueError("tail cache layers must have matching shapes")
        content_key = remove_rope(
            key.float(), source_cos, source_sin, sequence_dim=2
        )
        positioned_key = apply_rope(
            content_key, target_cos, target_sin, sequence_dim=2
        ).to(dtype=key.dtype)
        result.append((positioned_key, value))
    return tuple(result)


def _legacy_cache(cache: Any) -> LegacyCache:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise TypeError(f"unsupported cache type: {type(cache)!r}")
