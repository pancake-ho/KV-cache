from __future__ import annotations

import copy
import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .rope import apply_rope, model_rope_cos_sin, remove_rope


@dataclass(frozen=True)
class ChatSegments:
    """Token-exact decomposition of a tool-use chat prompt.

    ``prefix_ids`` contains the private system prompt and tool schema,
    ``history_ids`` is the shared transcript, and ``readout_ids`` is the
    assistant-generation header processed natively by the receiving model.
    """

    prefix_ids: tuple[int, ...]
    history_ids: tuple[int, ...]
    readout_ids: tuple[int, ...]


@dataclass(frozen=True)
class ToolCall:
    name: str | None
    arguments: dict[str, Any] | None
    valid_json: bool


@dataclass
class ReadoutState:
    logits: torch.Tensor
    past_key_values: Any
    cached_tokens: int


@dataclass(frozen=True)
class DistributionComparison:
    mean_kl: float
    max_kl: float
    reference_nll: float
    candidate_nll: float
    nll_delta: float
    top1_agreement: float
    tokens: int


def build_chat_segments(
    tokenizer,
    *,
    system_prompt: str,
    tools: list[dict[str, Any]],
    history: list[dict[str, Any]],
    enable_thinking: bool = False,
) -> ChatSegments:
    """Render prompt pieces through one chat template and verify exact boundaries."""

    system = {"role": "system", "content": system_prompt}
    context = _chat_ids(
        tokenizer,
        [system, *history],
        tools=tools,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    full = _chat_ids(
        tokenizer,
        [system, *history],
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    try:
        prefix = _chat_ids(
            tokenizer,
            [system],
            tools=tools,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
    except Exception as prefix_error:
        try:
            prefix_length = _history_content_boundary(
                tokenizer,
                system=system,
                tools=tools,
                history=history,
                context=context,
                enable_thinking=enable_thinking,
            )
        except Exception as boundary_error:
            raise ValueError(
                "chat template cannot render a private prefix or expose an exact "
                "first-history-content boundary"
            ) from boundary_error
        prefix = context[:prefix_length]
    if context[: len(prefix)] != prefix:
        raise ValueError(
            "the rendered system/tool prefix is not a token-exact prefix of the full chat"
        )
    if full[: len(context)] != context:
        raise ValueError(
            "adding the assistant readout changed already-rendered context tokens"
        )
    history_ids = context[len(prefix) :]
    readout_ids = full[len(context) :]
    if not prefix or not history_ids or not readout_ids:
        raise ValueError("prefix, history, and readout must each contain at least one token")
    return ChatSegments(tuple(prefix), tuple(history_ids), tuple(readout_ids))


def _history_content_boundary(
    tokenizer,
    *,
    system: dict[str, str],
    tools: list[dict[str, Any]],
    history: list[dict[str, Any]],
    context: Sequence[int],
    enable_thinking: bool,
) -> int:
    """Find the exact first-history-content boundary for templates like Llama 3.1.

    Some tool templates require a first user message because they serialize the
    tool schema immediately before that message's content. A reserved single-token
    marker lets us locate the content start without approximating text offsets.
    Everything before it (including tools and the user header) remains private.
    """

    if not history or not isinstance(history[0].get("content"), str):
        raise ValueError("first history message must contain text")
    candidates = (
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|unused0|>",
    )
    marker = None
    marker_id = None
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in context:
            marker = candidate
            marker_id = ids[0]
            break
    if marker is None or marker_id is None:
        raise ValueError("tokenizer has no unused single-token boundary marker")

    marked_history = copy.deepcopy(history)
    marked_history[0]["content"] = marker + marked_history[0]["content"]
    marked = _chat_ids(
        tokenizer,
        [system, *marked_history],
        tools=tools,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    positions = [index for index, token_id in enumerate(marked) if token_id == marker_id]
    if len(positions) != 1:
        raise ValueError("boundary marker was not serialized exactly once")
    boundary = positions[0]
    if tuple(marked[:boundary]) != tuple(context[:boundary]):
        raise ValueError("boundary marker changed tokens before first history content")
    if boundary <= 0 or boundary >= len(context):
        raise ValueError("invalid first history content boundary")
    return boundary


def validate_shared_handoff(source: ChatSegments, target: ChatSegments) -> None:
    """Require source and target to differ only in their private prefix."""

    if source.history_ids != target.history_ids:
        raise ValueError("source and target histories do not have identical token IDs")
    if source.readout_ids != target.readout_ids:
        raise ValueError("source and target assistant readouts do not have identical token IDs")


def history_message_spans(
    tokenizer,
    *,
    system_prompt: str,
    tools: list[dict[str, Any]],
    history: list[dict[str, Any]],
    enable_thinking: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Return token spans for each serialized history message.

    The function validates that rendering progressively longer message lists
    does not rewrite earlier tokens. This is true for the controlled Qwen tool
    prompts and prevents approximate string/token boundary matching.
    """

    segments = build_chat_segments(
        tokenizer,
        system_prompt=system_prompt,
        tools=tools,
        history=history,
        enable_thinking=enable_thinking,
    )
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>") if hasattr(
        tokenizer, "convert_tokens_to_ids"
    ) else None
    if isinstance(start_id, int) and start_id >= 0:
        starts = [
            index for index, token in enumerate(segments.history_ids) if token == start_id
        ]
        if len(starts) == len(history) and starts[0] == 0:
            ends = [*starts[1:], len(segments.history_ids)]
            return tuple(zip(starts, ends, strict=True))

    system = {"role": "system", "content": system_prompt}
    prefix = list(segments.prefix_ids)
    contexts = [prefix]
    for stop in range(1, len(history) + 1):
        context = _chat_ids(
            tokenizer,
            [system, *history[:stop]],
            tools=tools,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        if context[: len(contexts[-1])] != contexts[-1]:
            raise ValueError(
                "chat template rewrote earlier history tokens; message spans are ambiguous"
            )
        contexts.append(context)
    offsets = [len(context) - len(prefix) for context in contexts]
    return tuple(zip(offsets[:-1], offsets[1:], strict=True))


@torch.inference_mode()
def prefill_legacy_cache(model, token_ids: Sequence[int]) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if not token_ids:
        raise ValueError("cannot prefill an empty token sequence")
    device = _model_input_device(model)
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    output = model.model(input_ids=inputs, use_cache=True, return_dict=True)
    return _legacy_cache(output.past_key_values)


@torch.inference_mode()
def stitch_history_cache(
    model,
    *,
    source_context_cache: Any,
    target_prefix_cache: Any,
    source_prefix_length: int,
    target_prefix_length: int,
    transfer_history_length: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Attach source-imprinted history KV behind a native target-private prefix.

    Source RoPE is removed at its original absolute positions and target RoPE is
    applied at the positions that the same history tokens occupy after the
    target prefix. Values are copied because RoPE is not applied to V.
    """

    source_layers = _legacy_cache(source_context_cache)
    target_layers = _legacy_cache(target_prefix_cache)
    if len(source_layers) != len(target_layers):
        raise ValueError("source and target caches have different layer counts")
    if transfer_history_length < 0:
        raise ValueError("transfer_history_length must be non-negative")
    source_total = source_layers[0][0].shape[-2]
    if source_prefix_length + transfer_history_length > source_total:
        raise ValueError("requested history slice exceeds the source cache")
    if target_layers[0][0].shape[-2] != target_prefix_length:
        raise ValueError("target_prefix_length does not match target cache")

    if transfer_history_length:
        source_positions = torch.arange(
            source_prefix_length,
            source_prefix_length + transfer_history_length,
            dtype=torch.long,
        )
        target_positions = torch.arange(
            target_prefix_length,
            target_prefix_length + transfer_history_length,
            dtype=torch.long,
        )
        rope_device = _model_input_device(model)
        source_cos, source_sin = model_rope_cos_sin(
            model,
            source_positions,
            device=rope_device,
            dtype=torch.float32,
        )
        target_cos, target_sin = model_rope_cos_sin(
            model,
            target_positions,
            device=rope_device,
            dtype=torch.float32,
        )

    stitched = []
    same_positions = source_prefix_length == target_prefix_length
    for source_layer, target_layer in zip(source_layers, target_layers, strict=True):
        source_key, source_value = source_layer
        target_key, target_value = target_layer
        history_key = source_key[
            :, :, source_prefix_length : source_prefix_length + transfer_history_length
        ]
        history_value = source_value[
            :, :, source_prefix_length : source_prefix_length + transfer_history_length
        ]
        if transfer_history_length and not same_positions:
            device = source_key.device
            content_key = remove_rope(
                history_key.float(),
                source_cos.to(device=device),
                source_sin.to(device=device),
                sequence_dim=2,
            )
            history_key = apply_rope(
                content_key,
                target_cos.to(device=device),
                target_sin.to(device=device),
                sequence_dim=2,
            ).to(dtype=source_key.dtype)
        history_key = history_key.to(device=target_key.device, dtype=target_key.dtype)
        history_value = history_value.to(device=target_value.device, dtype=target_value.dtype)
        stitched.append(
            (
                torch.cat((target_key, history_key), dim=2),
                torch.cat((target_value, history_value), dim=2),
            )
        )
    return tuple(stitched)


@torch.inference_mode()
def patch_history_cache(
    model,
    *,
    source_context_cache: Any,
    target_context_cache: Any,
    target_prefix_cache: Any,
    source_prefix_length: int,
    target_prefix_length: int,
    history_length: int,
    source_layers: Collection[int] | None = None,
    source_keys: bool = True,
    source_values: bool = True,
    source_token_mask: Sequence[bool] | torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Causally patch selected source H-KV components into target-native H-KV.

    This constructs deliberately hybrid per-layer caches for mechanistic
    localization. It can patch K only, V only, selected layers, and selected
    history-token regions while all unselected components remain target-native.
    """

    source_context = _legacy_cache(source_context_cache)
    target_context = _legacy_cache(target_context_cache)
    target_prefix = _legacy_cache(target_prefix_cache)
    layer_count = len(target_context)
    if len(source_context) != layer_count or len(target_prefix) != layer_count:
        raise ValueError("cache layer counts differ")
    selected_layers = set(range(layer_count)) if source_layers is None else set(source_layers)
    if any(layer < 0 or layer >= layer_count for layer in selected_layers):
        raise ValueError("source_layers contains an out-of-range layer")
    if source_token_mask is None:
        token_mask = torch.ones(history_length, dtype=torch.bool)
    else:
        token_mask = torch.as_tensor(source_token_mask, dtype=torch.bool)
        if token_mask.ndim != 1 or token_mask.numel() != history_length:
            raise ValueError("source_token_mask must have shape [history_length]")

    source_positions = torch.arange(
        source_prefix_length, source_prefix_length + history_length, dtype=torch.long
    )
    target_positions = torch.arange(
        target_prefix_length, target_prefix_length + history_length, dtype=torch.long
    )
    same_positions = source_prefix_length == target_prefix_length
    if not same_positions:
        rope_device = _model_input_device(model)
        source_cos, source_sin = model_rope_cos_sin(
            model, source_positions, device=rope_device, dtype=torch.float32
        )
        target_cos, target_sin = model_rope_cos_sin(
            model, target_positions, device=rope_device, dtype=torch.float32
        )

    patched = []
    for layer, (source_pair, target_pair, prefix_pair) in enumerate(
        zip(source_context, target_context, target_prefix, strict=True)
    ):
        source_key, source_value = source_pair
        target_key, target_value = target_pair
        prefix_key, prefix_value = prefix_pair
        source_key = source_key[
            :, :, source_prefix_length : source_prefix_length + history_length
        ]
        source_value = source_value[
            :, :, source_prefix_length : source_prefix_length + history_length
        ]
        target_key = target_key[
            :, :, target_prefix_length : target_prefix_length + history_length
        ]
        target_value = target_value[
            :, :, target_prefix_length : target_prefix_length + history_length
        ]
        if not same_positions:
            device = source_key.device
            source_key = apply_rope(
                remove_rope(
                    source_key.float(),
                    source_cos.to(device=device),
                    source_sin.to(device=device),
                    sequence_dim=2,
                ),
                target_cos.to(device=device),
                target_sin.to(device=device),
                sequence_dim=2,
            ).to(dtype=target_key.dtype)
        source_key = source_key.to(device=target_key.device, dtype=target_key.dtype)
        source_value = source_value.to(device=target_value.device, dtype=target_value.dtype)

        use_source_layer = layer in selected_layers
        mask_key = token_mask.to(device=target_key.device).view(1, 1, history_length, 1)
        mask_value = token_mask.to(device=target_value.device).view(1, 1, history_length, 1)
        if use_source_layer and source_keys:
            history_key = torch.where(mask_key, source_key, target_key)
        else:
            history_key = target_key
        if use_source_layer and source_values:
            history_value = torch.where(mask_value, source_value, target_value)
        else:
            history_value = target_value
        patched.append(
            (
                torch.cat((prefix_key, history_key), dim=2),
                torch.cat((prefix_value, history_value), dim=2),
            )
        )
    return tuple(patched)


@torch.inference_mode()
def start_readout(
    model,
    *,
    legacy_cache: Any,
    cached_tokens: int,
    input_ids: Sequence[int],
) -> ReadoutState:
    """Process replay/readout tokens natively after an externally assembled cache."""

    if not input_ids:
        raise ValueError("a readout requires at least one native token")
    device = _model_input_device(model)
    current = torch.tensor([input_ids], dtype=torch.long, device=device)
    final_length = cached_tokens + current.shape[1]
    attention_mask = torch.ones((1, final_length), dtype=torch.long, device=device)
    positions = torch.arange(cached_tokens, final_length, device=device).unsqueeze(0)
    cache_positions = torch.arange(cached_tokens, final_length, device=device)
    cache = DynamicCache(ddp_cache_data=_legacy_cache(legacy_cache), config=model.config)
    output = model(
        input_ids=current,
        attention_mask=attention_mask,
        position_ids=positions,
        cache_position=cache_positions,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    return ReadoutState(output.logits[:, -1], output.past_key_values, final_length)


@torch.inference_mode()
def greedy_action(
    model,
    state_factory: Callable[[], ReadoutState],
    tokenizer,
    *,
    max_new_tokens: int = 96,
) -> tuple[tuple[int, ...], str]:
    state = state_factory()
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        next_id = int(state.logits.argmax(dim=-1).item())
        generated.append(next_id)
        decoded = tokenizer.decode(generated, skip_special_tokens=False)
        if (
            next_id == eos_id
            or "</tool_call>" in decoded
            or "<|im_end|>" in decoded
            or "<|eom_id|>" in decoded
            or "<|eot_id|>" in decoded
        ):
            break
        state = _advance_one(model, state, next_id)
    return tuple(generated), tokenizer.decode(generated, skip_special_tokens=False)


@torch.inference_mode()
def action_logits(
    model,
    state_factory: Callable[[], ReadoutState],
    action_ids: Sequence[int],
) -> torch.Tensor:
    """Teacher-forced logits predicting each token in ``action_ids``."""

    if not action_ids:
        raise ValueError("action_ids must be non-empty")
    state = state_factory()
    logits = [state.logits.unsqueeze(1)]
    if len(action_ids) > 1:
        device = state.logits.device
        inputs = torch.tensor([action_ids[:-1]], dtype=torch.long, device=device)
        final_length = state.cached_tokens + inputs.shape[1]
        output = model(
            input_ids=inputs,
            attention_mask=torch.ones((1, final_length), dtype=torch.long, device=device),
            position_ids=torch.arange(
                state.cached_tokens, final_length, device=device
            ).unsqueeze(0),
            cache_position=torch.arange(state.cached_tokens, final_length, device=device),
            past_key_values=state.past_key_values,
            use_cache=False,
            return_dict=True,
        )
        logits.append(output.logits)
    return torch.cat(logits, dim=1)


def compare_action_distributions(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    action_ids: Sequence[int],
) -> DistributionComparison:
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("reference and candidate logits have different shapes")
    if reference_logits.shape[1] != len(action_ids):
        raise ValueError("logit sequence length does not match action_ids")
    labels = torch.tensor(action_ids, dtype=torch.long, device=reference_logits.device).unsqueeze(0)
    reference_log_probs = reference_logits.float().log_softmax(dim=-1)
    candidate_log_probs = candidate_logits.float().log_softmax(dim=-1)
    reference_probs = reference_log_probs.exp()
    token_kl = (
        reference_probs * (reference_log_probs - candidate_log_probs)
    ).sum(dim=-1).clamp_min(0)
    reference_nll = F.nll_loss(
        reference_log_probs.flatten(0, 1), labels.flatten()
    ).item()
    candidate_nll = F.nll_loss(
        candidate_log_probs.flatten(0, 1), labels.flatten()
    ).item()
    agreement = (
        reference_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1)
    ).float().mean().item()
    return DistributionComparison(
        mean_kl=token_kl.mean().item(),
        max_kl=token_kl.max().item(),
        reference_nll=reference_nll,
        candidate_nll=candidate_nll,
        nll_delta=candidate_nll - reference_nll,
        top1_agreement=agreement,
        tokens=len(action_ids),
    )


def action_mean_logprob(logits: torch.Tensor, action_ids: Sequence[int]) -> float:
    if logits.shape[1] != len(action_ids):
        raise ValueError("logit sequence length does not match action_ids")
    labels = torch.tensor(action_ids, dtype=torch.long, device=logits.device).view(1, -1, 1)
    return logits.float().log_softmax(dim=-1).gather(-1, labels).mean().item()


def parse_tool_call(text: str) -> ToolCall:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    payload_text = match.group(1) if match is not None else _first_json_object(text)
    if payload_text is None:
        return ToolCall(None, None, False)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return ToolCall(None, None, False)
    name = payload.get("name")
    arguments = payload.get("arguments", payload.get("parameters"))
    valid = isinstance(name, str) and isinstance(arguments, dict)
    return ToolCall(name if isinstance(name, str) else None, arguments if isinstance(arguments, dict) else None, valid)


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        _, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return text[start : start + end]


@torch.inference_mode()
def layerwise_history_distance(
    model,
    *,
    source_context_cache: Any,
    target_context_cache: Any,
    source_prefix_length: int,
    target_prefix_length: int,
    history_length: int,
) -> list[dict[str, float | int]]:
    """Compare same-token H caches after removing their respective RoPE."""

    source_layers = _legacy_cache(source_context_cache)
    target_layers = _legacy_cache(target_context_cache)
    if len(source_layers) != len(target_layers):
        raise ValueError("cache layer counts differ")
    rope_device = _model_input_device(model)
    source_positions = torch.arange(source_prefix_length, source_prefix_length + history_length)
    target_positions = torch.arange(target_prefix_length, target_prefix_length + history_length)
    source_cos, source_sin = model_rope_cos_sin(
        model, source_positions, device=rope_device, dtype=torch.float32
    )
    target_cos, target_sin = model_rope_cos_sin(
        model, target_positions, device=rope_device, dtype=torch.float32
    )
    rows = []
    for layer, ((source_key, source_value), (target_key, target_value)) in enumerate(
        zip(source_layers, target_layers, strict=True)
    ):
        source_key = source_key[:, :, source_prefix_length : source_prefix_length + history_length]
        target_key = target_key[:, :, target_prefix_length : target_prefix_length + history_length]
        source_value = source_value[
            :, :, source_prefix_length : source_prefix_length + history_length
        ].float()
        target_value = target_value[
            :, :, target_prefix_length : target_prefix_length + history_length
        ].float()
        source_content_key = remove_rope(
            source_key.float(),
            source_cos.to(source_key.device),
            source_sin.to(source_key.device),
            sequence_dim=2,
        )
        target_content_key = remove_rope(
            target_key.float(),
            target_cos.to(target_key.device),
            target_sin.to(target_key.device),
            sequence_dim=2,
        )
        rows.append(
            {
                "layer": layer,
                "key_relative_l2": _relative_l2(source_content_key, target_content_key),
                "value_relative_l2": _relative_l2(source_value, target_value),
                "key_cosine": _cosine(source_content_key, target_content_key),
                "value_cosine": _cosine(source_value, target_value),
            }
        )
    return rows


def _chat_ids(tokenizer, messages, *, tools, add_generation_prompt, enable_thinking) -> list[int]:
    template_kwargs = {
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "tokenize": True,
    }
    # Llama 3.1's template distinguishes an absent tool argument from an empty
    # list: ``tools=[]`` still enables its function-calling protocol and rewrites
    # ordinary user prompts to request JSON calls.  An empty collection means
    # that the caller has no tools, so normalize it to ``None``.
    template_kwargs["tools"] = tools or None
    ids = tokenizer.apply_chat_template(messages, **template_kwargs)
    if isinstance(ids, torch.Tensor):
        ids = ids.flatten().tolist()
    return [int(token) for token in ids]


@torch.inference_mode()
def _advance_one(model, state: ReadoutState, token_id: int) -> ReadoutState:
    device = state.logits.device
    output = model(
        input_ids=torch.tensor([[token_id]], dtype=torch.long, device=device),
        attention_mask=torch.ones(
            (1, state.cached_tokens + 1), dtype=torch.long, device=device
        ),
        position_ids=torch.tensor([[state.cached_tokens]], dtype=torch.long, device=device),
        cache_position=torch.tensor([state.cached_tokens], dtype=torch.long, device=device),
        past_key_values=state.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    return ReadoutState(output.logits[:, -1], output.past_key_values, state.cached_tokens + 1)


def _legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise TypeError(f"unsupported cache type: {type(cache)!r}")


def _model_input_device(model) -> torch.device:
    backbone = getattr(model, "model", model)
    return backbone.embed_tokens.weight.device


def _relative_l2(source: torch.Tensor, target: torch.Tensor) -> float:
    denominator = target.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return ((source.float() - target.float()).norm() / denominator).item()


def _cosine(source: torch.Tensor, target: torch.Tensor) -> float:
    return F.cosine_similarity(source.float().flatten(), target.float().flatten(), dim=0).item()
