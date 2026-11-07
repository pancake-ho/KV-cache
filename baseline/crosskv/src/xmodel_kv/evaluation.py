from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ContinuationComparison:
    standalone_nll: float
    transfer_nll: float
    nll_delta: float
    top1_agreement: float
    logprob_mae: float
    continuation_tokens: int


@torch.inference_mode()
def compare_continuation(
    *,
    source_model,
    target_model,
    mapper,
    input_ids: torch.Tensor,
    prefix_length: int,
) -> ContinuationComparison:
    """Compare standalone target prefill with mapped-prefix continuation scoring.

    The mapped cache ends one token before the prefix boundary. The target processes the
    last prefix token itself, which supplies its own hidden state/logit for the first scored
    continuation token. This avoids an easy off-by-one error in prefill-free evaluation.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("current verifier expects input_ids=[1,total_tokens]")
    total_length = input_ids.shape[1]
    if not 2 <= prefix_length < total_length:
        raise ValueError("prefix_length must leave at least one continuation token")

    target_device = target_model.model.embed_tokens.weight.device
    standalone_inputs = input_ids[:, :-1].to(target_device)
    standalone = target_model(input_ids=standalone_inputs, use_cache=False, return_dict=True).logits
    standalone_logits = standalone[:, prefix_length - 1 :]
    labels = input_ids[:, prefix_length:].to(standalone_logits.device)

    source_device = source_model.model.embed_tokens.weight.device
    source_prefix = input_ids[:, : prefix_length - 1].to(source_device)
    source_output = source_model.model(input_ids=source_prefix, use_cache=True, return_dict=True)
    mapped_cache = mapper.map_cache(
        source_output.past_key_values,
        source_model=source_model,
        target_model=target_model,
    )
    transfer_inputs = input_ids[:, prefix_length - 1 : -1].to(target_device)
    full_attention_mask = torch.ones((1, total_length - 1), dtype=torch.long, device=target_device)
    cache_position = torch.arange(prefix_length - 1, total_length - 1, device=target_device)
    transfer_logits = target_model(
        input_ids=transfer_inputs,
        attention_mask=full_attention_mask,
        past_key_values=mapped_cache,
        cache_position=cache_position,
        use_cache=False,
        return_dict=True,
    ).logits

    standalone_log_probs = standalone_logits.float().log_softmax(dim=-1)
    transfer_log_probs = transfer_logits.float().log_softmax(dim=-1)
    standalone_nll = F.nll_loss(standalone_log_probs.flatten(0, 1), labels.flatten()).item()
    transfer_nll = F.nll_loss(transfer_log_probs.flatten(0, 1), labels.flatten()).item()
    top1_agreement = (
        standalone_logits.argmax(dim=-1) == transfer_logits.argmax(dim=-1)
    ).float().mean().item()
    label_index = labels.unsqueeze(-1)
    logprob_mae = (
        standalone_log_probs.gather(-1, label_index) - transfer_log_probs.gather(-1, label_index)
    ).abs().mean().item()
    return ContinuationComparison(
        standalone_nll=standalone_nll,
        transfer_nll=transfer_nll,
        nll_delta=transfer_nll - standalone_nll,
        top1_agreement=top1_agreement,
        logprob_mae=logprob_mae,
        continuation_tokens=labels.numel(),
    )
