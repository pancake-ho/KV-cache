"""Measure mapping and bitpack restoration error after the latency trials."""

import math

import torch

from .record import write_row


def relative_error(estimate, reference, *, chunk_tokens=256):
    """Compute a layer-relative L2 and the distribution of token-relative L2."""
    if (estimate.ndim != 4 or estimate.shape[0] != 1 or not estimate.shape[2]
            or estimate.shape != reference.shape or chunk_tokens < 1
            or estimate.dtype != reference.dtype or estimate.device != reference.device):
        raise ValueError("KV error comparison requires aligned [1, heads, tokens, head_dim] tensors")

    dims = (0, 1, 3)
    ref_energy = []
    diff_energy = []
    for start in range(0, reference.shape[2], chunk_tokens):
        stop = start + chunk_tokens
        base = reference[:, :, start:stop, :].float()
        delta = estimate[:, :, start:stop, :].float() - base
        ref_energy.append(base.square().sum(dim=dims))
        diff_energy.append(delta.square_().sum(dim=dims))
    ref = torch.cat(ref_energy)
    diff = torch.cat(diff_energy)
    rel = diff.sqrt() / ref.sqrt().clamp_min(1e-12)
    p50, p95, p99 = torch.quantile(rel, rel.new_tensor([0.50, 0.95, 0.99])).tolist()
    upper, positions = rel.topk(min(3, rel.numel()))
    layer_diff, layer_ref = diff.sum(), ref.sum()
    layer_l2 = (layer_diff.sqrt() / layer_ref.sqrt().clamp_min(1e-12)).item()
    upper = upper.tolist()
    if not all(math.isfinite(value) for value in
               (layer_diff.item(), layer_ref.item(), layer_l2, p50, p95, p99, *upper)):
        raise ValueError("Nonfinite KV error metric")
    return {
        "relative_l2": layer_l2,
        "token_relative_l2": {"p50": p50, "p95": p95, "p99": p99, "max": upper[0]},
        "largest_token_errors": [
            {"position": pos, "relative_l2": value}
            for pos, value in zip(positions.tolist(), upper, strict=True)
        ],
    }


@torch.inference_mode()
def diagnose_prefix(runtime, mapper, codec, source, target):
    """Compare the same absolute prefix positions under the verified LoRA route."""
    ids = source.token_ids
    if ids != target.token_ids[:len(ids)]:
        raise ValueError("Source tokens do not match the Target-native prefix")
    with runtime.adapters.activate(source.model_id, source.adapter_id) as (model, source_sha):
        source_cache = runtime.forward(model, ids, 0).past_key_values
        source_layers = runtime.layers(model, source_cache, len(ids))
        del source_cache
    with runtime.adapters.activate(target.model_id, target.adapter_id) as (model, target_sha):
        projected, _ = mapper.project_prefix(
            source_layers, source_model=runtime.models[source.model_id], target_model=model,
            collect_scores=False,
        )
        mapped = runtime.layers(model, projected, len(ids))
        del source_layers, projected
        exact = runtime.forward(model, ids, 0).past_key_values
        native = runtime.layers(model, exact, len(ids))
        del exact

    if len(mapped) != len(native):
        raise ValueError("Mapped and Target-native layer counts differ")
    layers = []
    for index, (mapped_layer, native_layer) in enumerate(zip(mapped, native, strict=True)):
        # Round-trip one layer with the same GPU codec. This pass sends no TCP traffic.
        restored_layer, = codec.decode(codec.encode((mapped_layer,)))
        layer = {"layer": index}
        for kind, projected, exact, restored in zip(
            ("k", "v"), mapped_layer, native_layer, restored_layer, strict=True,
        ):
            layer[kind] = {
                "mapping_vs_native": relative_error(projected, exact),
                "bitpack_vs_mapped": relative_error(restored, projected),
            }
        layers.append(layer)
        del restored_layer, restored
    return {
        "source_adapter_sha256": source_sha, "target_adapter_sha256": target_sha,
        "prefix_tokens": len(ids), "layers": layers,
    }


def write_diagnostics(path, runtime, mapper, codec, producers, targets):
    with path.open("x", encoding="utf-8") as file:
        write_row(file, {
            "kind": "settings", "wire_codec": "bitpack5",
            "scope": "after TTFT trials; prefix positions start at zero; "
                     "local codec encode/decode before repair; one pass per question",
            "token_error": "L2(estimate - reference) / max(L2(reference), 1e-12)",
        })
        for event in targets:
            request = event.request
            source = producers[event.prefix_request_id].request
            metrics = diagnose_prefix(runtime, mapper, codec, source, request)
            write_row(file, {
                "kind": "kv_diagnostics", "request_id": request.request_id,
                "token_sha256": event.document["token_sha256"], **metrics,
            })
        write_row(file, {"kind": "summary", "questions": len(targets)})
    print(f"KV diagnostics {path}", flush=True)
