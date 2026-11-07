from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import DynamicCache

from .policy_imprinting import ReadoutState


CAPSULE_INIT_TEXT = (
    "Recompute now using only the active receiver rule above. Suppress the upstream "
    "choice and execute the receiver-selected action."
)


class SoftPolicyCapsule(nn.Module):
    """A short shared suffix whose contextual states are conditioned by a B-policy tail."""

    def __init__(self, tokens: int, hidden_size: int):
        super().__init__()
        if tokens < 1 or hidden_size < 1:
            raise ValueError("tokens and hidden-size must be positive")
        self.tokens = int(tokens)
        self.hidden_size = int(hidden_size)
        self.embeddings = nn.Parameter(torch.empty(tokens, hidden_size))
        self.register_buffer("initial_embeddings", torch.empty(tokens, hidden_size))

    @torch.no_grad()
    def initialize_from_text(self, model, tokenizer, text: str = CAPSULE_INIT_TEXT) -> None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError("capsule initialization text produced no tokens")
        device = model.model.embed_tokens.weight.device
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        source = model.model.embed_tokens(ids).squeeze(0).float()
        initialized = _resample_rows(source, self.tokens)
        self.embeddings.copy_(initialized.to(self.embeddings.device))
        self.initial_embeddings.copy_(initialized.to(self.initial_embeddings.device))

    def drift_regularization(self) -> torch.Tensor:
        return (self.embeddings - self.initial_embeddings).float().square().mean()


class PromptPolicyCompiler(nn.Module):
    """Compile a natural-language receiver prompt into contextual tail slots.

    The compiler only consumes frozen input embeddings from the base model.  Its
    output is inserted after the reused history and is subsequently contextualized
    by the frozen LLM.  It therefore avoids a second full-model pass over the
    receiver prompt while retaining a causal dependency on the reused history.
    """

    def __init__(
        self,
        *,
        tokens: int,
        model_hidden_size: int,
        compiler_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        max_prompt_tokens: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        positive = (
            tokens,
            model_hidden_size,
            compiler_dim,
            layers,
            heads,
            max_prompt_tokens,
        )
        if min(positive) < 1:
            raise ValueError("compiler dimensions and layer counts must be positive")
        if compiler_dim % heads:
            raise ValueError("compiler-dim must be divisible by heads")
        self.tokens = int(tokens)
        self.model_hidden_size = int(model_hidden_size)
        self.compiler_dim = int(compiler_dim)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.input_projection = nn.Linear(
            model_hidden_size, compiler_dim, bias=False
        )
        self.position_embeddings = nn.Embedding(max_prompt_tokens, compiler_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=compiler_dim,
            nhead=heads,
            dim_feedforward=compiler_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.queries = nn.Parameter(torch.empty(tokens, compiler_dim))
        self.cross_attention = nn.MultiheadAttention(
            compiler_dim, heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(compiler_dim)
        self.output_projection = nn.Linear(compiler_dim, model_hidden_size)
        self.register_buffer(
            "initial_embeddings", torch.empty(tokens, model_hidden_size)
        )
        nn.init.normal_(self.queries, std=compiler_dim**-0.5)
        # Start from a readable natural-language steering suffix.  The first
        # optimization steps then learn a prompt-dependent residual instead of
        # injecting an arbitrary random prefix into the frozen model.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @torch.no_grad()
    def initialize_from_text(
        self, model, tokenizer, text: str = CAPSULE_INIT_TEXT
    ) -> None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError("capsule initialization text produced no tokens")
        device = model.model.embed_tokens.weight.device
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        source = model.model.embed_tokens(ids).squeeze(0).float()
        initialized = _resample_rows(source, self.tokens)
        self.initial_embeddings.copy_(
            initialized.to(
                device=self.initial_embeddings.device,
                dtype=self.initial_embeddings.dtype,
            )
        )

    def forward(self, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        squeeze = prompt_embeddings.ndim == 2
        if squeeze:
            prompt_embeddings = prompt_embeddings.unsqueeze(0)
        if prompt_embeddings.ndim != 3 or prompt_embeddings.shape[-1] != self.model_hidden_size:
            raise ValueError(
                "prompt embeddings must have shape [batch, tokens, model-hidden-size]"
            )
        prompt_length = prompt_embeddings.shape[1]
        if not 0 < prompt_length <= self.max_prompt_tokens:
            raise ValueError(
                f"prompt length {prompt_length} is outside [1, {self.max_prompt_tokens}]"
            )
        positions = torch.arange(prompt_length, device=prompt_embeddings.device)
        encoded = self.input_projection(prompt_embeddings.float())
        encoded = encoded + self.position_embeddings(positions).unsqueeze(0)
        encoded = self.encoder(encoded)
        queries = self.queries.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            queries, encoded, encoded, need_weights=False
        )
        residual = self.output_projection(self.output_norm(attended))
        compiled = self.initial_embeddings.unsqueeze(0) + residual
        return compiled.squeeze(0) if squeeze else compiled

    def drift_regularization(self, compiled: torch.Tensor) -> torch.Tensor:
        initial = self.initial_embeddings
        if compiled.ndim == 3:
            initial = initial.unsqueeze(0)
        return (compiled.float() - initial.float()).square().mean()


def capsule_action_logits(
    model,
    legacy_cache,
    capsule: SoftPolicyCapsule,
    *,
    cached_tokens: int,
    tail_ids: Sequence[int],
    readout_ids: Sequence[int],
    action_ids: Sequence[int],
) -> torch.Tensor:
    if not tail_ids or not readout_ids or not action_ids:
        raise ValueError("tail, readout, and action IDs must be non-empty")
    embedding = model.model.embed_tokens
    device = embedding.weight.device
    before = embedding(
        torch.tensor([tail_ids], dtype=torch.long, device=device)
    ).detach()
    teacher_ids = tuple(readout_ids) + tuple(action_ids[:-1])
    after = embedding(
        torch.tensor([teacher_ids], dtype=torch.long, device=device)
    ).detach()
    soft = capsule.embeddings.to(dtype=before.dtype).unsqueeze(0)
    inputs = torch.cat((before, soft, after), dim=1)
    final_length = cached_tokens + inputs.shape[1]
    positions = torch.arange(cached_tokens, final_length, device=device)
    cache = DynamicCache(ddp_cache_data=legacy_cache, config=model.config)
    output = model(
        inputs_embeds=inputs,
        attention_mask=torch.ones((1, final_length), dtype=torch.long, device=device),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    start = len(tail_ids) + capsule.tokens + len(readout_ids) - 1
    return output.logits[:, start : start + len(action_ids)]


def compiled_capsule_action_logits(
    model,
    legacy_cache,
    compiler: PromptPolicyCompiler,
    *,
    cached_tokens: int,
    prompt_ids: Sequence[int] | None = None,
    prompt_states: torch.Tensor | None = None,
    readout_ids: Sequence[int],
    action_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not readout_ids or not action_ids:
        raise ValueError("readout and action IDs must be non-empty")
    embedding = model.model.embed_tokens
    device = embedding.weight.device
    prompt = _resolve_prompt_states(
        embedding,
        prompt_ids=prompt_ids,
        prompt_states=prompt_states,
        device=device,
    )
    soft = compiler(prompt)
    teacher_ids = tuple(readout_ids) + tuple(action_ids[:-1])
    after = embedding(
        torch.tensor([teacher_ids], dtype=torch.long, device=device)
    ).detach()
    inputs = torch.cat(
        (soft.to(dtype=after.dtype).unsqueeze(0), after), dim=1
    )
    final_length = cached_tokens + inputs.shape[1]
    positions = torch.arange(cached_tokens, final_length, device=device)
    cache = DynamicCache(ddp_cache_data=legacy_cache, config=model.config)
    output = model(
        inputs_embeds=inputs,
        attention_mask=torch.ones(
            (1, final_length), dtype=torch.long, device=device
        ),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    start = compiler.tokens + len(readout_ids) - 1
    return output.logits[:, start : start + len(action_ids)], soft


@torch.inference_mode()
def start_capsule_readout(
    model,
    *,
    legacy_cache,
    capsule: SoftPolicyCapsule,
    cached_tokens: int,
    tail_ids: Sequence[int],
    readout_ids: Sequence[int],
) -> ReadoutState:
    if not tail_ids or not readout_ids:
        raise ValueError("tail and readout IDs must be non-empty")
    embedding = model.model.embed_tokens
    device = embedding.weight.device
    before = embedding(
        torch.tensor([tail_ids], dtype=torch.long, device=device)
    )
    after = embedding(
        torch.tensor([readout_ids], dtype=torch.long, device=device)
    )
    soft = capsule.embeddings.to(dtype=before.dtype).unsqueeze(0)
    inputs = torch.cat((before, soft, after), dim=1)
    final_length = cached_tokens + inputs.shape[1]
    positions = torch.arange(cached_tokens, final_length, device=device)
    cache = DynamicCache(ddp_cache_data=legacy_cache, config=model.config)
    output = model(
        inputs_embeds=inputs,
        attention_mask=torch.ones((1, final_length), dtype=torch.long, device=device),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    return ReadoutState(output.logits[:, -1], output.past_key_values, final_length)


@torch.inference_mode()
def start_compiled_capsule_readout(
    model,
    *,
    legacy_cache,
    compiler: PromptPolicyCompiler,
    cached_tokens: int,
    prompt_ids: Sequence[int] | None = None,
    prompt_states: torch.Tensor | None = None,
    readout_ids: Sequence[int],
) -> ReadoutState:
    embedding = model.model.embed_tokens
    device = embedding.weight.device
    prompt = _resolve_prompt_states(
        embedding,
        prompt_ids=prompt_ids,
        prompt_states=prompt_states,
        device=device,
    )
    soft = compiler(prompt)
    return start_embedding_capsule_readout(
        model,
        legacy_cache=legacy_cache,
        soft_embeddings=soft,
        cached_tokens=cached_tokens,
        readout_ids=readout_ids,
    )


@torch.inference_mode()
def start_embedding_capsule_readout(
    model,
    *,
    legacy_cache,
    soft_embeddings: torch.Tensor,
    cached_tokens: int,
    readout_ids: Sequence[int],
) -> ReadoutState:
    if not readout_ids:
        raise ValueError("readout IDs must be non-empty")
    if soft_embeddings.ndim != 2:
        raise ValueError("soft embeddings must have shape [tokens, hidden-size]")
    embedding = model.model.embed_tokens
    device = embedding.weight.device
    after = embedding(
        torch.tensor([readout_ids], dtype=torch.long, device=device)
    )
    soft = soft_embeddings.to(device=device, dtype=after.dtype).unsqueeze(0)
    inputs = torch.cat((soft, after), dim=1)
    final_length = cached_tokens + inputs.shape[1]
    positions = torch.arange(cached_tokens, final_length, device=device)
    cache = DynamicCache(ddp_cache_data=legacy_cache, config=model.config)
    output = model(
        inputs_embeds=inputs,
        attention_mask=torch.ones(
            (1, final_length), dtype=torch.long, device=device
        ),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    return ReadoutState(output.logits[:, -1], output.past_key_values, final_length)


def _resample_rows(source: torch.Tensor, rows: int) -> torch.Tensor:
    if source.ndim != 2 or source.shape[0] < 1:
        raise ValueError("source must be a non-empty rank-two tensor")
    boundaries = torch.linspace(
        0, source.shape[0], rows + 1, device=source.device
    ).round().long()
    values = []
    for index in range(rows):
        start = min(int(boundaries[index]), source.shape[0] - 1)
        end = max(start + 1, int(boundaries[index + 1]))
        values.append(source[start:end].mean(dim=0))
    return torch.stack(values)


def _resolve_prompt_states(
    embedding,
    *,
    prompt_ids: Sequence[int] | None,
    prompt_states: torch.Tensor | None,
    device,
) -> torch.Tensor:
    if prompt_states is not None and prompt_ids is not None:
        raise ValueError("provide prompt IDs or prompt states, not both")
    if prompt_states is not None:
        if prompt_states.ndim == 3 and prompt_states.shape[0] == 1:
            prompt_states = prompt_states.squeeze(0)
        if prompt_states.ndim != 2:
            raise ValueError("prompt states must have shape [tokens, hidden-size]")
        if prompt_states.shape[0] < 1:
            raise ValueError("prompt states must be non-empty")
        return prompt_states.detach().to(device=device, dtype=torch.float32)
    if not prompt_ids:
        raise ValueError("prompt IDs must be non-empty when prompt states are absent")
    return (
        embedding(torch.tensor([prompt_ids], dtype=torch.long, device=device))
        .squeeze(0)
        .detach()
        .float()
    )
