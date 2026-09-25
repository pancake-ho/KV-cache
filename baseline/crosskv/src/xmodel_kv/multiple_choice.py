from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import DynamicCache


@dataclass
class ChoiceScores:
    standalone: list[float]
    transfer: list[float]
    standalone_normalized: list[float]
    transfer_normalized: list[float]
    token_counts: list[int]
    character_counts: list[int]


@dataclass
class TokenContinuationScores:
    standalone_log_likelihood: float
    transfer_log_likelihood: float
    tokens: int


def hellaswag_preprocess(text: str) -> str:
    """Match lm-evaluation-harness's HellaSwag text cleanup."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hellaswag_context_and_choices(document: dict) -> tuple[str, list[str], int]:
    context = hellaswag_preprocess(
        document["activity_label"]
        + ": "
        + document["ctx_a"]
        + " "
        + document["ctx_b"].capitalize()
    )
    choices = [hellaswag_preprocess(ending) for ending in document["endings"]]
    return context, choices, int(document["label"])


@torch.inference_mode()
def score_choices(
    *,
    source_model,
    target_model,
    mapper,
    tokenizer,
    context: str,
    choices: list[str],
    target_delimiter: str = " ",
) -> ChoiceScores:
    if not choices:
        raise ValueError("at least one choice is required")
    continuations_text = [target_delimiter + choice for choice in choices]
    pairs = [_encode_pair(tokenizer, context, continuation) for continuation in continuations_text]
    context_ids = pairs[0][0]
    if len(context_ids) < 2:
        raise ValueError("context must tokenize to at least two tokens")
    if any(pair_context != context_ids for pair_context, _ in pairs):
        raise ValueError("choice-dependent context tokenization; boundary handling is inconsistent")
    continuations = [continuation for _, continuation in pairs]
    if any(not continuation for continuation in continuations):
        raise ValueError("an empty continuation cannot be scored")

    target_device = target_model.model.embed_tokens.weight.device
    standalone_inputs, standalone_mask = _padded_standalone_batch(
        context_ids,
        continuations,
        pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        device=target_device,
    )
    standalone_logits = target_model(
        input_ids=standalone_inputs,
        attention_mask=standalone_mask,
        use_cache=False,
        return_dict=True,
    ).logits
    standalone_scores = []
    for row, continuation in enumerate(continuations):
        start = len(context_ids) - 1
        logits = standalone_logits[row, start : start + len(continuation)]
        labels = torch.tensor(continuation, device=logits.device)
        standalone_scores.append(float(_token_log_likelihood(logits, labels).sum().cpu()))

    source_device = source_model.model.embed_tokens.weight.device
    source_prefix = torch.tensor(context_ids[:-1], device=source_device).unsqueeze(0)
    source_output = source_model.model(input_ids=source_prefix, use_cache=True, return_dict=True)
    mapped = mapper.map_cache(
        source_output.past_key_values,
        source_model=source_model,
        target_model=target_model,
    )
    batched_cache = _repeat_cache(mapped, len(choices))
    transfer_inputs, transfer_mask = _padded_transfer_batch(
        context_ids[-1],
        continuations,
        past_length=len(context_ids) - 1,
        pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        device=target_device,
    )
    cache_position = torch.arange(
        len(context_ids) - 1,
        len(context_ids) - 1 + transfer_inputs.shape[1],
        device=target_device,
    )
    transfer_logits = target_model(
        input_ids=transfer_inputs,
        attention_mask=transfer_mask,
        past_key_values=batched_cache,
        cache_position=cache_position,
        use_cache=False,
        return_dict=True,
    ).logits
    transfer_scores = []
    for row, continuation in enumerate(continuations):
        logits = transfer_logits[row, : len(continuation)]
        labels = torch.tensor(continuation, device=logits.device)
        transfer_scores.append(float(_token_log_likelihood(logits, labels).sum().cpu()))

    token_counts = [len(continuation) for continuation in continuations]
    # This deliberately follows lm-evaluation-harness: HellaSwag acc_norm divides
    # by Python len(choice), i.e. characters in the preprocessed choice, while the
    # delimiter is excluded. It does not normalize by tokenizer-token count.
    character_counts = [len(choice) for choice in choices]
    if any(count == 0 for count in character_counts):
        raise ValueError("an empty choice cannot be length-normalized")
    return ChoiceScores(
        standalone=standalone_scores,
        transfer=transfer_scores,
        standalone_normalized=[
            score / count for score, count in zip(standalone_scores, character_counts)
        ],
        transfer_normalized=[score / count for score, count in zip(transfer_scores, character_counts)],
        token_counts=token_counts,
        character_counts=character_counts,
    )


@torch.inference_mode()
def score_token_continuation(
    *,
    source_model,
    target_model,
    mapper,
    prefix_ids: list[int],
    continuation_ids: list[int],
) -> TokenContinuationScores:
    """Score fixed tokens after a prefix, replacing only the prefix KV in transfer mode."""
    if len(prefix_ids) < 2:
        raise ValueError("prefix must contain at least two tokens")
    if len(continuation_ids) < 2:
        raise ValueError("continuation must contain at least two tokens")

    target_device = target_model.model.embed_tokens.weight.device
    standalone_input = torch.tensor(
        prefix_ids + continuation_ids[:-1], dtype=torch.long, device=target_device
    ).unsqueeze(0)
    standalone_logits = target_model(
        input_ids=standalone_input,
        use_cache=False,
        return_dict=True,
    ).logits[0, len(prefix_ids) :]
    # A cache cannot provide the target LM-head logits for the last prefix token.
    # Map the complete 1,024-token prefix as specified by the paper and score
    # continuation[1:] in both modes; continuation[0] is the bridge input.
    labels = torch.tensor(continuation_ids[1:], dtype=torch.long, device=standalone_logits.device)
    standalone_score = float(_token_log_likelihood(standalone_logits, labels).sum().cpu())

    source_device = source_model.model.embed_tokens.weight.device
    source_prefix = torch.tensor(prefix_ids, dtype=torch.long, device=source_device).unsqueeze(0)
    source_output = source_model.model(input_ids=source_prefix, use_cache=True, return_dict=True)
    mapped = mapper.map_cache(
        source_output.past_key_values,
        source_model=source_model,
        target_model=target_model,
    )
    transfer_input = torch.tensor(
        continuation_ids[:-1], dtype=torch.long, device=target_device
    ).unsqueeze(0)
    past_length = len(prefix_ids)
    attention_mask = torch.ones(
        (1, past_length + transfer_input.shape[1]), dtype=torch.long, device=target_device
    )
    cache_position = torch.arange(
        past_length,
        past_length + transfer_input.shape[1],
        dtype=torch.long,
        device=target_device,
    )
    transfer_logits = target_model(
        input_ids=transfer_input,
        attention_mask=attention_mask,
        past_key_values=mapped,
        cache_position=cache_position,
        use_cache=False,
        return_dict=True,
    ).logits[0]
    transfer_score = float(_token_log_likelihood(transfer_logits, labels).sum().cpu())
    return TokenContinuationScores(
        standalone_log_likelihood=standalone_score,
        transfer_log_likelihood=transfer_score,
        tokens=len(continuation_ids) - 1,
    )


def _encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    # This is the boundary treatment used by lm-evaluation-harness: trailing context
    # spaces belong to the continuation before joint tokenization.
    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    whole = tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    return context_ids, whole[len(context_ids) :]


def _padded_standalone_batch(context, continuations, *, pad_id, device):
    rows = [context + continuation[:-1] for continuation in continuations]
    return _right_pad(rows, pad_id=pad_id, prefix_length=0, device=device)


def _padded_transfer_batch(last_context_token, continuations, *, past_length, pad_id, device):
    rows = [[last_context_token] + continuation[:-1] for continuation in continuations]
    return _right_pad(rows, pad_id=pad_id, prefix_length=past_length, device=device)


def _right_pad(rows, *, pad_id, prefix_length, device):
    if pad_id is None:
        raise ValueError("tokenizer must define eos_token_id or pad_token_id")
    max_length = max(map(len, rows))
    input_ids = torch.full((len(rows), max_length), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), prefix_length + max_length), dtype=torch.long, device=device)
    if prefix_length:
        attention_mask[:, :prefix_length] = 1
    for row_index, row in enumerate(rows):
        input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[row_index, prefix_length : prefix_length + len(row)] = 1
    return input_ids, attention_mask


def _repeat_cache(cache, batch_size: int) -> DynamicCache:
    layers = []
    for key, value in cache.to_legacy_cache():
        repeats = (batch_size,) + (1,) * (key.ndim - 1)
        layers.append((key.repeat(repeats), value.repeat(repeats)))
    return DynamicCache.from_legacy_cache(tuple(layers))


def _token_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return logits.float().log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
