"""Refit the existing rank-disagreement mapper on LoRA-active paired KV traces."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from ..e2e_inputs import ROLES, artifact_manifest, read, sha, write
from ...src.exp.mapper_optim.calibration import collect, fit_layers
from ...src.exp.mapper_optim.inputs import ModelIdentity, check_split, token_hash
from ...src.exp.rarecache.rank import fit_basis
from ...src.models.lora import LoRARegistry


@contextmanager
def active_model(model, endpoint, revision):
    registry = LoRARegistry({endpoint["model_id"]: model})
    model.config._commit_hash = revision  # Caller verified the local snapshot revision.
    info = registry.load(endpoint["model_id"], endpoint["adapter_id"], endpoint["path"])
    if info["adapter_sha256"] != endpoint["adapter_sha256"]:
        raise ValueError("LoRA checkpoint changed after settings were resolved")
    with registry.activate(endpoint["model_id"], endpoint["adapter_id"]) as (active, _):
        yield active


def calibration_inputs(settings, prompts, cohort):
    base = Path(settings["artifacts"]["rarecache"])
    spec = read(base / "calibration.json")
    examples_file = base / "fit_examples.json"
    examples = read(examples_file)
    if (spec["method"] != "rarecache" or spec["variant"] != "rank-disagreement" or spec["dtype"] != "bfloat16"
            or not examples or any(len(row["ids"]) != spec["sequence_length"]
                                   or token_hash(row["ids"]) != row["sha256"] for row in examples)):
        raise ValueError("Base mapper's BF16 fit inputs are missing or incompatible")
    held_out = [dict(ids=ids, line=row["line"], text_sha256=row["document_sha256"],
                     sha256=row["token_sha256"]) for row, (ids, _) in zip(cohort, prompts, strict=True)]
    check_split(held_out, examples, spec["sequence_length"])
    local_models = {}
    for role in ROLES:
        expected = ModelIdentity(**settings["models"][role])
        snapshot = settings["model_snapshots"][role]
        local = ModelIdentity.inspect(snapshot, expected.revision)
        fitted = ModelIdentity(**spec[role])
        if (replace(local, name=expected.name) != expected or fitted.name not in {expected.name, snapshot}
                or replace(fitted, name=expected.name) != expected):
            raise ValueError(f"Base mapper or local snapshot differs for {role}")
        local_models[role] = local

    return base, spec, examples_file, examples, local_models


def adapter_trace(identity, examples, scratch, *, role, endpoint, stride, device):
    import torch

    return collect(identity, examples, scratch, role=role, stride=stride,
                   device=device, dtype=torch.bfloat16,
                   model_context=lambda model: active_model(model, endpoint, identity.revision))


def fit(settings, report, output, device, prompts, cohort):
    import torch

    base, spec, examples_file, examples, local_models = calibration_inputs(settings, prompts, cohort)

    def trace(role, endpoint, scratch):
        return adapter_trace(local_models[role], examples, scratch, role=role, endpoint=endpoint,
                             stride=spec["stride"], device=device)

    report["fitted_routes"] = {}
    output_root = output.parent / "lora_artifacts"
    output_root.mkdir()
    source_endpoint = next(iter(settings["lora_routes"].values()))["source"]
    with TemporaryDirectory(dir=output.parent, prefix="lora-calibration-") as scratch:
        source = trace("source", source_endpoint, Path(scratch) / "source")
        for name, route in settings["lora_routes"].items():
            print(f"Fitting LoRA route {route['source']['adapter_id']} -> {name}", flush=True)
            artifact = output_root / name
            artifact.mkdir()
            with TemporaryDirectory(dir=scratch, prefix="target-") as target_scratch:
                target = trace("target", route["target"], Path(target_scratch) / "trace")
                fit_layers(source, target, artifact,
                           settings={**spec, "lora_route": route,
                                     "fit_examples_sha256": sha(examples_file),
                                     "base_metadata_sha256": sha(base / "calibration.json")},
                           device=device, fit_dtype=getattr(torch, spec["fit_dtype"]),
                           basis_fn=fit_basis)
            report["fitted_routes"][name] = {
                "route": route, "manifest": artifact_manifest("rarecache", artifact),
            }
            write(output, report)
