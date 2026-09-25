from __future__ import annotations

import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GenerationPair:
    standalone: str
    transfer: str
    standalone_tokens: int
    transfer_tokens: int
    standalone_stopped_on_match: bool
    transfer_stopped_on_match: bool


@dataclass(frozen=True)
class _Generated:
    text: str
    tokens: int
    stopped_on_match: bool


@torch.inference_mode()
def generate_pair(
    *,
    source_model,
    target_model,
    mapper,
    tokenizer,
    context: str,
    max_new_tokens: int = 256,
    stop: tuple[str, ...] = ("Q:",),
    stop_regex: str | None = None,
) -> GenerationPair:
    return generate_pairs(
        source_model=source_model,
        target_model=target_model,
        mapper=mapper,
        tokenizer=tokenizer,
        contexts=[context],
        max_new_tokens=max_new_tokens,
        stop=stop,
        stop_regex=stop_regex,
    )[0]


@torch.inference_mode()
def generate_pairs(
    *,
    source_model,
    target_model,
    mapper,
    tokenizer,
    contexts: list[str],
    max_new_tokens: int = 256,
    stop: tuple[str, ...] = ("Q:",),
    stop_regex: str | None = None,
) -> list[GenerationPair]:
    """Greedily generate standalone and transferred continuations in one batch.

    Contexts are left padded. ``position_ids`` retain each example's logical
    positions, while ``cache_position`` tracks the shared physical cache axis.
    """
    if not contexts:
        return []
    context_rows = [
        tokenizer(context, add_special_tokens=False)["input_ids"] for context in contexts
    ]
    if any(len(row) < 2 for row in context_rows):
        raise ValueError("generation contexts must tokenize to at least two tokens")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    target_device = target_model.model.embed_tokens.weight.device
    standalone_ids, standalone_mask, standalone_positions = _left_pad(
        context_rows, device=target_device, pad_id=pad_id
    )
    standalone = _greedy_generate_batch(
        target_model,
        tokenizer,
        initial_ids=standalone_ids,
        initial_attention_mask=standalone_mask,
        initial_position_ids=standalone_positions,
        past_key_values=None,
        initial_cache_position=0,
        max_new_tokens=max_new_tokens,
        stop=stop,
        stop_regex=stop_regex,
        pad_id=pad_id,
    )

    source_device = source_model.model.embed_tokens.weight.device
    prefix_rows = [row[:-1] for row in context_rows]
    source_prefix, source_mask, source_positions = _left_pad(
        prefix_rows, device=source_device, pad_id=pad_id
    )
    source_output = source_model.model(
        input_ids=source_prefix,
        attention_mask=source_mask,
        position_ids=source_positions,
        use_cache=True,
        return_dict=True,
    )
    mapped = mapper.map_cache(
        source_output.past_key_values,
        source_model=source_model,
        target_model=target_model,
        positions=source_positions,
    )
    transfer_initial = torch.tensor(
        [[row[-1]] for row in context_rows], dtype=torch.long, device=target_device
    )
    transfer_mask = torch.cat(
        [
            source_mask.to(device=target_device),
            torch.ones((len(context_rows), 1), dtype=torch.long, device=target_device),
        ],
        dim=1,
    )
    transfer_positions = source_mask.sum(dim=1, keepdim=True).to(device=target_device)
    transfer = _greedy_generate_batch(
        target_model,
        tokenizer,
        initial_ids=transfer_initial,
        initial_attention_mask=transfer_mask,
        initial_position_ids=transfer_positions,
        past_key_values=mapped,
        initial_cache_position=source_prefix.shape[1],
        max_new_tokens=max_new_tokens,
        stop=stop,
        stop_regex=stop_regex,
        pad_id=pad_id,
    )

    return [
        GenerationPair(
            standalone=base.text,
            transfer=mapped_result.text,
            standalone_tokens=base.tokens,
            transfer_tokens=mapped_result.tokens,
            standalone_stopped_on_match=base.stopped_on_match,
            transfer_stopped_on_match=mapped_result.stopped_on_match,
        )
        for base, mapped_result in zip(standalone, transfer, strict=True)
    ]


def _left_pad(
    rows: list[list[int]], *, device: torch.device, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, rows))
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, -len(row) :] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[index, -len(row) :] = 1
    position_ids = attention_mask.cumsum(dim=1).sub(1).clamp_min(0)
    return input_ids, attention_mask, position_ids


def _greedy_generate_batch(
    model,
    tokenizer,
    *,
    initial_ids: torch.Tensor,
    initial_attention_mask: torch.Tensor,
    initial_position_ids: torch.Tensor,
    past_key_values,
    initial_cache_position: int,
    max_new_tokens: int,
    stop: tuple[str, ...],
    stop_regex: str | None,
    pad_id: int,
) -> list[_Generated]:
    batch_size = initial_ids.shape[0]
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    stopped_on_match = [False] * batch_size
    batch_indices = torch.arange(batch_size, device=initial_ids.device)
    current = initial_ids
    attention_mask = initial_attention_mask
    position_ids = initial_position_ids
    past = past_key_values
    physical_position = initial_cache_position
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        cache_position = torch.arange(
            physical_position,
            physical_position + current.shape[1],
            dtype=torch.long,
            device=current.device,
        )
        output = model(
            input_ids=current,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        next_ids = output.logits[:, -1].argmax(dim=-1)
        past = output.past_key_values
        physical_position += current.shape[1]

        finished = torch.zeros(len(batch_indices), dtype=torch.bool, device=current.device)
        for local_index, original_index in enumerate(batch_indices.tolist()):
            next_id = int(next_ids[local_index])
            generated[original_index].append(next_id)
            if eos_id is not None and next_id == eos_id:
                finished[local_index] = True
                continue
            decoded = tokenizer.decode(generated[original_index], skip_special_tokens=True)
            if stop_regex is not None and re.search(stop_regex, decoded):
                stopped_on_match[original_index] = True
                finished[local_index] = True
            elif any(term and term in decoded for term in stop):
                finished[local_index] = True

        keep = (~finished).nonzero(as_tuple=False).flatten()
        if keep.numel() == 0:
            break
        next_positions = attention_mask.sum(dim=1, keepdim=True)[keep]
        past.batch_select_indices(keep)
        current = next_ids[keep].unsqueeze(1)
        attention_mask = torch.cat(
            [
                attention_mask[keep],
                torch.ones((keep.numel(), 1), dtype=attention_mask.dtype, device=current.device),
            ],
            dim=1,
        )
        position_ids = next_positions
        batch_indices = batch_indices[keep]

    results = []
    for token_ids, matched in zip(generated, stopped_on_match, strict=True):
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        for term in stop:
            if term:
                decoded = decoded.split(term)[0]
        results.append(_Generated(decoded, len(token_ids), matched))
    return results
