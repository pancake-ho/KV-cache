"""Fit and calibrate CacheBridge v2 on the same frozen LoRA routes as Proposed."""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from ..cachebridge_v2_fit import readout_cases
from ..e2e_inputs import ROLES, artifact_manifest, sha, write
from ...src.exp.cachebridge.readout import fit_readout
from ...src.exp.mapper_optim.calibration import fit_layers
from ...src.exp.mapper_optim.transfer import AffineMapper
from .fit import adapter_trace, calibration_inputs


def fit_initializers(settings, report, output, device, prompts, cohort):
    import torch

    _, spec, examples_path, examples, models = calibration_inputs(settings, prompts, cohort)
    source, target = (models[role] for role in ROLES)
    if (source.heads, source.width) != (target.heads, target.width):
        raise ValueError("CacheBridge v2 requires aligned source and target KV heads")
    routes = settings["lora_routes"]
    root = output.parent / "cachebridge_v2_lora"
    root.mkdir()
    examples_digest = sha(examples_path)

    def collect(role, endpoint, directory):
        return adapter_trace(models[role], examples, directory, role=role, endpoint=endpoint,
                             stride=spec["stride"], device=device)

    report["cachebridge_v2_lora_initializers"] = initializers = {}
    with TemporaryDirectory(dir=output.parent, prefix="cachebridge-lora-traces-") as scratch:
        scratch = Path(scratch)
        source_trace = collect("source", next(iter(routes.values()))["source"], scratch / "source")
        for name, route in routes.items():
            print(f"{name}: CacheBridge v2 head-local fit", flush=True)
            started = perf_counter()
            artifact = root / name / "initializer"
            artifact.mkdir(parents=True)
            route_pair = {role: route[role] for role in ROLES}
            with TemporaryDirectory(dir=scratch, prefix="target-") as trace:
                target_trace = collect("target", route["target"], Path(trace) / "trace")
                fit_layers(
                    source_trace, target_trace, artifact,
                    settings={**spec, "method": "cachebridge", "variant": "head-local", "rank": None,
                              "fit_examples_sha256": examples_digest, "lora_route": route_pair},
                    device=device, fit_dtype=getattr(torch, spec["fit_dtype"]),
                )
            initializers[name] = {
                "artifact": str(artifact), "manifest": artifact_manifest("cachebridge", artifact),
                "seconds": perf_counter() - started,
            }
            write(output, report)
    return initializers, [row["ids"] for row in examples]


def calibrate_readouts(settings, report, output, reference, runtime, device, initialization):
    import torch

    initializers, examples = initialization
    routes = settings["lora_routes"]
    model_ids = tuple(settings["models"][role]["name"] for role in ROLES)
    tokenizers = tuple(runtime.tokenizers[model_id] for model_id in model_ids)
    history_limit = settings["profile"]["lora_cachebridge_v2_history"]
    selected, cases = readout_cases(reference, tokenizers, history_limit)
    context_limit = min(settings["models"][role]["context"] for role in ROLES)
    if any(len(ids) > context_limit for ids, _ in cases):
        raise ValueError("LoRA readout calibration example exceeds model context")
    report["cachebridge_v2_lora_fits"] = fits = {}
    for name, route in routes.items():
        print(f"{name}: CacheBridge v2 LoRA readout fit", flush=True)
        started = perf_counter()
        source, target = (route[role] for role in ROLES)
        initial = Path(initializers[name]["artifact"])
        if artifact_manifest("cachebridge", initial) != initializers[name]["manifest"]:
            raise ValueError(f"CacheBridge initializer changed for {name}")
        mapper = AffineMapper(initial, "cachebridge", device)
        spec = mapper.spec
        if (spec["lora_route"] != {role: route[role] for role in ROLES}
                or spec["variant"] != "head-local"
                or spec["selected_layers"]["k"] != spec["selected_layers"]["v"]):
            raise ValueError(f"Invalid joint K/V support or adapter identity for {name}")
        artifact = initial.parent / "calibrated"
        artifact.mkdir()
        with (runtime.adapters.activate(source["model_id"], source["adapter_id"]) as (src_model, src_sha),
              runtime.adapters.activate(target["model_id"], target["adapter_id"]) as (tgt_model, tgt_sha)):
            if (src_sha, tgt_sha) != (source["adapter_sha256"], target["adapter_sha256"]):
                raise ValueError(f"Unexpected LoRA checkpoint in readout calibration for {name}")
            layer_metrics = fit_readout(src_model, tgt_model, mapper, examples, cases, artifact)
        if not any(row["changed_value_heads"] for row in layer_metrics):
            raise RuntimeError(f"CacheBridge v2 fitted no Value corrections for {name}")
        write(artifact / "calibration.json", {
            **spec, "method": "cachebridge_v2", "variant": "layer-ordered-readout",
            "initializer_metadata_sha256": initializers[name]["manifest"]["metadata_sha256"],
            "qa_report_sha256": settings["qa_report_sha256"], "dataset_sha256": settings["dataset_sha256"],
            "readout_documents": selected, "max_readout_history": history_limit,
        })
        fits[name] = {
            "artifact": str(artifact), "manifest": artifact_manifest("cachebridge_v2", artifact),
            "seconds": perf_counter() - started, "layers": layer_metrics,
        }
        write(output, report)
        del mapper
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
