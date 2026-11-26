"""Resolve LoRA endpoints and bind finished calibration jobs to serving traces."""

from pathlib import Path

from ..e2e_inputs import QASPER_DATASET, REPORT, artifact_manifest, locate_e2e_fit, read
from ...src.models.lora import adapter_fingerprint


def endpoint(config, expected, adapter_id):
    if not adapter_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid adapter ID: {adapter_id}")
    model, dataset, directory = config.MULTI_USER_ADAPTERS[adapter_id]
    if model != expected["name"] or dataset not in config.ADAPTER_DATASETS:
        raise ValueError(f"Expected {adapter_id} on {expected['name']} with a registered training dataset")
    directory = Path(directory).resolve()
    adapter = read(directory / "adapter_config.json")
    if (adapter["base_model_name_or_path"] != model
            or adapter.get("revision") not in {None, expected["revision"]}
            or adapter["peft_type"] != "LORA"):
        raise ValueError(f"Adapter {adapter_id} has an incompatible checkpoint base")
    return {
        "model_id": model, "adapter_id": adapter_id, "dataset": dataset,
        "path": str(directory), "adapter_sha256": adapter_fingerprint(directory),
    }


def resolve_lora(config, settings):
    source_id = config.E2E_LORA_SOURCE_ADAPTER
    target_ids = tuple(config.E2E_LORA_TARGET_ADAPTERS)
    if not target_ids or len(set(target_ids)) != len(target_ids):
        raise ValueError("Configure distinct target LoRA adapter IDs")
    source = endpoint(config, settings["models"]["source"], source_id)
    routes = {
        target_id: {"source": source, "target": endpoint(config, settings["models"]["target"], target_id)}
        for target_id in target_ids
    }
    profile = settings["profile"]
    case = profile["scenarios"][0]
    datasets = settings.get("evaluation_datasets")
    if datasets:
        scenarios = [
            {**case, "id": f"{route}__{dataset}", "lora_route": route, "eval_dataset": dataset,
             **({"reuse_count": profile["questions_per_document"]} if dataset == QASPER_DATASET else {})}
            for route in routes for dataset in datasets
        ]
    else:
        scenarios = [{**case, "id": name} for name in routes]
    profile = {**profile, "scenarios": scenarios}
    settings.update(profile=profile, lora_routes=routes, methods=["full", "exact", "lora_base", "lora_fitted"])
    if profile["lora_stage"] == "fit":
        return settings

    job = str(config.E2E_LORA_FIT_JOB)
    if not job.isdecimal():
        raise ValueError("Set E2E_LORA_FIT_JOB to the completed LoRA calibration job")
    directory = locate_e2e_fit(config, job)
    fit = read(directory / REPORT)
    if (fit["status"] != "complete" or fit["settings"]["profile"]["lora_stage"] != "fit"
            or any(fit["settings"][key] != settings[key] for key in
                   ("pair", "models", "model_snapshots", "qa_report_sha256", "artifact_manifest"))):
        raise ValueError("LoRA calibration job used different model/checkpoint or QA settings")
    for route_name, route in routes.items():
        artifact = directory / "lora_artifacts" / route_name
        result = fit["fitted_routes"][route_name]
        if result["route"] != route or artifact_manifest("rarecache", artifact) != result["manifest"]:
            raise ValueError(f"LoRA calibration artifact or route differs for {route_name}")
        route.update(artifact=str(artifact), manifest=result["manifest"])
    if profile.get("lora_cachebridge_v2_history"):
        settings["methods"].append("lora_cachebridge_v2")
    settings["lora_fit_job"] = job
    return settings
