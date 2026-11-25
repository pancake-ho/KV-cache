"""Bind LoRA identities to the shared E2E trace and mapper implementations."""

from dataclasses import replace
from pathlib import Path

from ..e2e_inputs import artifact_manifest
from ...src.kv_transfer.adapter_pair import AdapterPairKVTransfer


def activate_adapters(runtime, identities, routes):
    source = next(iter(routes.values()))["source"]
    endpoints = [source, *(route["target"] for route in routes.values())]
    for identity in identities:
        runtime.models[identity.name].config._commit_hash = identity.revision
    for endpoint in endpoints:
        info = runtime.adapters.load(endpoint["model_id"], endpoint["adapter_id"], endpoint["path"])
        if info["adapter_sha256"] != endpoint["adapter_sha256"]:
            raise ValueError(f"LoRA checkpoint changed: {endpoint['adapter_id']}")
        warm_tokens = runtime.tokenizers[endpoint["model_id"]].encode(
            "Warmup.", add_special_tokens=False,
        )
        with runtime.adapters.activate(endpoint["model_id"], endpoint["adapter_id"]) as (model, _):
            runtime.forward(model, warm_tokens, 0)


def assign_adapters(trace, route):
    return [
        replace(event, request=replace(
            event.request, adapter_id=route["source" if event.kind == "producer" else "target"]["adapter_id"],
        )) for event in trace
    ]


def transfer_for(method, settings, route_name, identities, snapshots, device, *, cachebridge_fit=None):
    from ...src.kv_transfer.affine import AffineKVMapper

    if method in {"full", "exact"}:
        return None, None
    route = settings["lora_routes"][route_name]
    route_pair = {role: route[role] for role in ("source", "target")}
    mapper_method = "rarecache"
    if method == "lora_base":
        path = Path(settings["artifacts"]["rarecache"])
        manifest = settings["artifact_manifest"]["rarecache"]
    elif method == "lora_fitted":
        path = Path(route["artifact"])
        manifest = route["manifest"]
    elif method == "lora_cachebridge_v2" and cachebridge_fit is not None:
        mapper_method = "cachebridge_v2"
        path = Path(cachebridge_fit["artifact"])
        manifest = cachebridge_fit["manifest"]
    else:
        raise ValueError(f"Unknown or unfitted LoRA serving method {method}")
    if artifact_manifest(mapper_method, path) != manifest:
        raise ValueError(f"{method} mapper checkpoint differs from the fitted artifact")
    mapper = AffineKVMapper(path, mapper_method, identities, snapshots, device,
                            repair_fraction=settings["repair_fraction"])
    expected_route = None if method == "lora_base" else route_pair
    if mapper.metadata.get("lora_route") != expected_route:
        raise ValueError("Mapper was calibrated for a different adapter route")
    return AdapterPairKVTransfer(mapper, route["source"], route["target"]), {
        "artifact": str(path), "lora_route": route_pair, "storage": "GPU",
        "repair_fraction": mapper.repair_fraction,
    }


def verify_routes(rows, method):
    expected = {"full": "full_recompute", "exact": "miss_recompute"}.get(method, "cross_model_kv_hit")
    if any(row["path"] != expected for row in rows):
        raise RuntimeError(f"{method} did not consistently take its expected LoRA QA path: {expected}")
