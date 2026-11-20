"""Paired continuation quality and single-GPU handoff measurements."""
from __future__ import annotations

from statistics import median
from time import perf_counter

import torch
import torch.nn.functional as F

from .cache import prefill


def _timed(device: torch.device, fn):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = perf_counter()
    value = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, (perf_counter() - t0) * 1000.


def _append_target(target, ids, cache, start: int):
    stop = start + ids.shape[1]
    return target(input_ids=ids, past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(start, stop, device=ids.device), return_dict=True)


def _transfer(mapper, source, target, cache, prefix, budget, replay):
    mapped, score = mapper.map(source, target, cache, collect_scores=bool(budget))
    if budget:
        choose, recompute = replay
        positions = choose(score, budget)
        return recompute(target, mapped, prefix, positions), positions
    return mapped, None


@torch.inference_mode()
def continuation(mapper, source, target, ids, prefix_length, budget, replay):
    start = prefix_length - 1
    native = target(input_ids=ids[:, :-1], use_cache=False,
                    logits_to_keep=ids.shape[1] - prefix_length, return_dict=True).logits
    prefix = ids[:, :start]
    mapped, positions = _transfer(mapper, source, target, prefill(source, prefix), prefix,
                                  budget, replay)
    transferred = _append_target(target, ids[:, start:-1], mapped, start).logits
    labels = ids[:, prefix_length:].flatten()
    nll_native = F.cross_entropy(native.float().flatten(0, 1), labels).item()
    nll_map = F.cross_entropy(transferred.float().flatten(0, 1), labels).item()
    return {
        "native_nll": nll_native, "nll": nll_map, "nll_delta": nll_map - nll_native,
        "top1_agreement": (native.argmax(-1) == transferred.argmax(-1)).float().mean().item(),
        "scored_tokens": labels.numel(), "recomputed_positions": [] if positions is None else positions.tolist(),
    }


@torch.inference_mode()
def benchmark(mapper, source, target, prompt, *, path: str, budget: float, replay,
              warmup: int, repeats: int, decode_tokens: int) -> dict:
    """Measure steady-state stages with per-stage CUDA synchronization."""
    device = prompt.device
    source_prefix, boundary = prompt[:, :-1], prompt.shape[1] - 1
    rows = []
    for index in range(warmup + repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = torch.cuda.memory_allocated(device)
            reserved_before = torch.cuda.memory_reserved(device)
        source_ms = map_ms = replay_ms = 0.
        if path == "native":
            initial, first_ms = _timed(device, lambda: target(
                input_ids=prompt, use_cache=True, logits_to_keep=1, return_dict=True))
        else:
            cache, source_ms = _timed(device, lambda: prefill(source, source_prefix))
            (mapped, score), map_ms = _timed(device, lambda: mapper.map(
                source, target, cache, collect_scores=bool(budget)))
            del cache
            if budget:
                choose, recompute = replay
                def repair():
                    return recompute(target, mapped, source_prefix, choose(score, budget))
                mapped, replay_ms = _timed(device, repair)
            initial, first_ms = _timed(device, lambda: _append_target(
                target, prompt[:, -1:], mapped, boundary))
            del mapped, score

        def first_token():
            return initial.logits[:, -1].argmax(dim=-1, keepdim=True)
        token, first_select_ms = _timed(device, first_token)

        def decode():
            generated = [token]
            cache = initial.past_key_values
            next_id = token
            for offset in range(decode_tokens - 1):
                output = _append_target(target, next_id, cache, prompt.shape[1] + offset)
                cache = output.past_key_values
                next_id = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                generated.append(next_id)
            return torch.cat(generated, dim=1)
        generated, decode_ms = _timed(device, decode)
        first_ms += first_select_ms
        transfer_ms = map_ms + replay_ms + first_ms
        row = {"source_prefill_ms": source_ms, "map_and_score_ms": map_ms,
               "select_and_replay_ms": replay_ms, "target_first_token_ms": first_ms,
               "handoff_to_first_token_ms": transfer_ms,
               "with_source_first_token_ms": source_ms + transfer_ms,
               "further_decode_ms": decode_ms,
               "staged_total_with_source_ms": source_ms + transfer_ms + decode_ms}
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            row.update(peak_allocated_bytes=peak, peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                       peak_extra_allocated_bytes=peak - allocated_before,
                       reserved_before_bytes=reserved_before, allocated_before_bytes=allocated_before)
        if index >= warmup:
            rows.append(row)
        del initial, token
    return {"median": {key: median(row[key] for row in rows) for key in rows[0]},
            "samples": rows, "greedy_token_ids": generated[0].tolist()}
