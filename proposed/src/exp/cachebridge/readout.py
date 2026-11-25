"""Closed-form layer-ordered Value calibration for dense, aligned Qwen3 heads."""

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from xmodel_kv.rope import model_rope_cos_sin, remove_rope

from ..mapper_optim.cache import prefill, prepare_key, prepare_value


def value_features(source_cache, support, stride=1):
    return torch.cat([source_cache[index][1][0, :, ::stride] for index in support], dim=-1).float()


def augment(x):
    return torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)


def causal_readout_features(layer, model, hidden, keys, values, native, features):
    """Compute the readout design and residual with only history substituted."""
    history, query_length = features.shape[1], hidden.shape[1]
    attn = layer.self_attn
    heads, _, width = features.shape
    normalized = layer.input_layernorm(hidden)
    shape = (1, query_length, -1, attn.head_dim)
    q = attn.q_norm(attn.q_proj(normalized).view(shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(normalized).view(shape)).transpose(1, 2)
    v = attn.v_proj(normalized).view(shape).transpose(1, 2)
    positions = torch.arange(history, history + query_length, device=hidden.device)
    cos, sin = model.model.rotary_emb(hidden, positions[None, :])
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    q = q.float().reshape(heads, attn.num_key_value_groups, query_length, attn.head_dim)
    allowed = torch.arange(history + query_length, device=hidden.device) <= positions[:, None]

    def weights(prefix):
        all_keys = torch.cat((prefix, k), dim=-2).float().squeeze(0)
        logits = (q @ all_keys[:, None].transpose(-1, -2)) * attn.scaling
        return logits.masked_fill(~allowed, -torch.inf).softmax(dim=-1)

    mapped_attn, native_attn = weights(keys), weights(native[0])
    design = mapped_attn[..., :history] @ augment(features)[:, None]
    mapped_readout = mapped_attn @ torch.cat((values, v), dim=-2)[0, :, None].float()
    native_readout = native_attn @ torch.cat((native[1], v), dim=-2)[0, :, None].float()
    mask = torch.zeros_like(allowed, dtype=hidden.dtype).masked_fill(~allowed, torch.finfo(hidden.dtype).min)
    geometry = positions, (cos, sin), mask[None, None]
    return design.reshape(heads, -1, width + 1), (native_readout - mapped_readout).flatten(1, 2), geometry


def constrained_correction(gram, cross, moment, residual_energy):
    """Solve a batched FP64 ridge system with the paper's update-energy constraint."""
    ridge = (0.01 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)).clamp_min(1e-8)
    gram.diagonal(dim1=-2, dim2=-1).add_(ridge[:, None])
    moment.diagonal(dim1=-2, dim2=-1).add_(1e-4)
    budget = 0.01 * residual_energy

    def solve(multiplier):
        delta = torch.linalg.solve(gram + multiplier[:, None, None] * moment, cross)
        return delta, (delta * (moment @ delta)).sum(dim=(-2, -1))

    zero = torch.zeros_like(budget)
    delta, energy = solve(zero)
    constrained = (energy > budget) & (budget > 0)
    if constrained.any():
        # The norm bound from the M-inverse form gives a feasible upper bracket.
        upper = (cross * torch.linalg.solve(moment, cross)).sum(dim=(-2, -1)).clamp_min(0)
        upper = torch.where(constrained, (upper / budget.clamp_min(1e-30)).sqrt(), zero)
        lower = zero
        for _ in range(24):
            midpoint = (lower + upper) * 0.5
            _, test_energy = solve(midpoint)
            over_budget = constrained & (test_energy > budget)
            lower = torch.where(over_budget, midpoint, lower)
            upper = torch.where(over_budget, upper, midpoint)
        delta, _ = solve(upper)
    delta = torch.where((budget > 0)[:, None, None], delta, 0)
    energy = (delta * (moment @ delta)).sum(dim=(-2, -1))
    if not torch.isfinite(delta).all() or torch.any(energy > budget * 1.00001 + 1e-12):
        raise RuntimeError("Invalid CacheBridge readout correction or update energy")
    return delta, energy


def native_history(model, ids):
    """Keep the native, immutable prefix states off GPU between fitted layers."""
    cache = prefill(model, ids)
    return tuple(tuple(t.cpu() for t in layer) for layer in cache.to_legacy_cache())


def advance(layer, layer_index, hidden, keys, values, geometry):
    positions, rope, mask = geometry
    cache = DynamicCache()
    cache.update(keys, values, layer_index)
    return layer(hidden, attention_mask=mask, position_ids=positions[None, :],
                 past_key_values=cache, use_cache=True, cache_position=positions,
                 position_embeddings=rope)


@torch.inference_mode()
def fit_readout(source_model, target_model, initializer, examples, readout_cases, output: Path):
    device = target_model.device
    dtype = target_model.dtype
    init_source, init_target = [], []
    for ids in examples:
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        init_source.append(native_history(source_model, tokens))
        init_target.append(native_history(target_model, tokens))

    cases = []
    for ids, boundary in readout_cases:
        prefix = torch.tensor([ids[:boundary]], dtype=torch.long, device=device)
        suffix = torch.tensor([ids[boundary:]], dtype=torch.long, device=device)
        cases.append((native_history(source_model, prefix), native_history(target_model, prefix),
                      target_model.model.embed_tokens(suffix).cpu()))
    spec = initializer.spec
    stride = spec["stride"]
    support_by_layer = spec["selected_layers"]["v"]
    layer_metrics = []
    for layer_index, (layer, state, support) in enumerate(
            zip(target_model.model.layers, initializer.layers, support_by_layer, strict=True)):
        weight, bias = state["v_weight"], state["v_bias"]
        head_count, width, target_width = weight.shape
        zeros = lambda a, b: torch.zeros((head_count, a, b), device=device, dtype=torch.float64)
        moment, gram, cross = zeros(width + 1, width + 1), zeros(width + 1, width + 1), zeros(width + 1, target_width)
        residual_energy = torch.zeros(head_count, device=device, dtype=torch.float64)
        readout_energy = torch.zeros_like(residual_energy)
        observations = 0
        for source, target in zip(init_source, init_target, strict=True):
            x = value_features(source, support, stride).to(device)
            y = target[layer_index][1][0, :, ::stride].to(device).float()
            residual = (x @ weight + bias[:, None, :] - y).double()
            x = augment(x).double()
            moment += x.transpose(1, 2) @ x
            residual_energy += residual.square().sum(dim=(1, 2))
            observations += x.shape[1]

        steps = []
        for source, target, hidden in cases:
            features = value_features(source, support).to(device)
            length = features.shape[1]
            positions = torch.arange(length, device=device)
            src_rope = model_rope_cos_sin(source_model, positions, device=device, dtype=torch.float32)
            tgt_rope = model_rope_cos_sin(target_model, positions, device=device, dtype=torch.float32)
            key_x = torch.cat([remove_rope(source[i][0].to(device).float(), *src_rope)[0]
                               for i in support], dim=-1)
            keys = prepare_key((key_x @ state["k_weight"] + state["k_bias"][:, None, :]).transpose(0, 1),
                               *tgt_rope, dtype)
            values = prepare_value((features @ weight + bias[:, None, :]).transpose(0, 1), dtype)
            hidden = hidden.to(device)
            native = tuple(t.to(device) for t in target[layer_index])
            design, residual, geometry = causal_readout_features(layer, target_model, hidden, keys, values,
                                                                native, features)
            x, y = design.double(), residual.double()
            gram += (x.transpose(1, 2) @ x) / x.shape[1]
            cross += (x.transpose(1, 2) @ y) / x.shape[1]
            readout_energy += y.square().sum(dim=(1, 2)) / x.shape[1]
            steps.append((keys.cpu(), geometry))
        delta, energy = constrained_correction(gram.clone(), cross, moment / observations,
                                               residual_energy / observations)
        estimated_loss = readout_energy + (delta * (gram @ delta - 2 * cross)).sum(dim=(1, 2))
        corrected_weight = (weight.double() + delta[:, :-1]).float()
        corrected_bias = (bias.double() + delta[:, -1]).float()
        for i, ((source, target, hidden), (keys, geometry)) in enumerate(zip(cases, steps, strict=True)):
            features = value_features(source, support).to(device)
            values = prepare_value((features @ corrected_weight + corrected_bias[:, None, :]).transpose(0, 1), dtype)
            propagated = advance(layer, layer_index, hidden.to(device), keys.to(device), values, geometry)
            cases[i] = (source, target, propagated.cpu())
        states = {**state, "v_weight": corrected_weight, "v_bias": corrected_bias}
        save_file({k: t.cpu().contiguous() for k, t in states.items()}, str(output / f"layer_{layer_index:03d}.safetensors"))
        layer_metrics.append({"layer": layer_index, "update_energy_max": float(energy.max()),
                              "readout_squared_error_initial": float(readout_energy.sum()),
                              "readout_squared_error_after_fp64_update": float(estimated_loss.sum()),
                              "changed_value_heads": int((energy > 0).sum())})
        print(f"Readout calibrated {layer_index + 1}/{len(initializer.layers)}", flush=True)
    return layer_metrics
