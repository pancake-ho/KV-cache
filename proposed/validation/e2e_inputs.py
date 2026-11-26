"""Read pinned LongBench cohort and fitted mapper inputs for the serving evaluator."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


REPORT = "report-top-system-e2e.json"
PAIR_REPORT = "report-longbench-qa-pairs.json"
ROLES = ("source", "target")
LEGACY = {"ridge": "legacy_ridge_cached", "mlp": "legacy_mlp"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_qa(results, job):
    if not str(job).isdigit():
        raise ValueError("Use a numeric QA Slurm job ID")
    reports = [log.with_name(PAIR_REPORT) for log in
               (results / "comparison").glob(f"longbench-qa-pairs-*/slurm-{job}.out")
               if log.with_name(PAIR_REPORT).is_file()]
    if len(reports) != 1:
        raise ValueError(f"Expected one report for QA job {job}; set E2E_QA_REPORT if moved")
    return reports[0]


def locate_e2e_fit(config, job):
    """Find a fit report in the current or historical E2E output root."""
    directory = config.E2E_OUTPUT_ROOT / f"e2e_{job}"
    if (directory / REPORT).is_file():
        return directory
    historical = config.OUTPUT_ROOT / f"e2e_{job}"
    if (historical / REPORT).is_file():
        return historical
    paths = ", ".join(str(path / REPORT) for path in dict.fromkeys((directory, historical)))
    raise FileNotFoundError(f"E2E fit job {job} report is missing. Checked {paths}")


def qa_inputs(path, pair):
    qa = read(path)
    result = qa.get("results", {}).get(pair, {})
    if qa["status"] not in {"complete", "partial", "failed"} or result.get("status") != "complete":
        raise ValueError(f"The {pair} QA result has not completed")
    reference = Path(qa["reference_report"])
    if sha(reference) != qa["reference_report_sha256"]:
        raise ValueError("QA reference report fingerprint differs")
    old = read(reference)
    if old["status"] != "complete" or old["cohort"] != qa["cohort"] or old["commit"] != qa["commit"]:
        raise ValueError("QA reference cohort or experimental commit differs")
    if tuple(result["models"][role]["name"] for role in ROLES) != tuple(qa["pairs"][pair]):
        raise ValueError("Reported model identities do not match the selected direction")
    return qa, result, old


def artifact_manifest(method, directory):
    path = Path(directory)
    name = "calibration.json" if method in {"rarecache", "cachebridge", "cachebridge_v2"} else "config.json"
    weights = list(path.rglob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No {method} checkpoints in {path}")
    return {"metadata_sha256": sha(path / name), "safetensors_files": len(weights),
            "safetensors_bytes": sum(file.stat().st_size for file in weights)}


def artifacts_for(pair, result, old, cachebridge, profile):
    paths = {"rarecache": result["fit"]["artifact"]}
    expected = {"rarecache": result["fit"]["manifest"]}
    original_pair = all(result["models"][role] == old["geometry"][role] for role in ROLES)
    if original_pair:
        paths["ridge"] = old["artifacts"][LEGACY["ridge"]]
        expected["ridge"] = old["artifact_manifest"][LEGACY["ridge"]]
    if profile == "legacy":
        if not original_pair:
            raise ValueError("Legacy MLP is only available for the original QA model pair")
        legacy_path = old["baseline_report"]
        if sha(legacy_path) != old["baseline_report_sha256"]:
            raise ValueError("Legacy MLP report fingerprint differs")
        legacy = read(legacy_path)
        paths["mlp"] = legacy["artifacts"][LEGACY["mlp"]]
        expected["mlp"] = legacy["artifact_manifest"][LEGACY["mlp"]]
    if pair in cachebridge:
        paths["cachebridge"] = str(cachebridge[pair])
    methods = ["full", "exact", *(name for name in ("ridge", "cachebridge", "rarecache", "mlp")
                                  if name in paths)]
    manifests = {name: artifact_manifest(name, path) for name, path in paths.items()}
    for name, previous in expected.items():
        if manifests[name] != previous:
            raise ValueError(f"Fitted {name} files differ from the completed QA run")
    return methods, paths, manifests


def resolve_settings(config):
    """Snapshot evaluator settings once inside the Slurm allocation."""
    from .semcache_smoke import git_state
    from .semcache_runner import CHECKOUT_ENV, d5_git_state

    root = config.REPO_ROOT
    path = Path(config.E2E_QA_REPORT or locate_qa(config.E2E_RESULTS, config.E2E_QA_JOB)).resolve()
    pair = config.E2E_PAIR
    profile_name = os.environ.get("KV_TOP_E2E_PROFILE", config.E2E_PROFILE)
    profile = config.E2E_PROFILES[profile_name]
    input_digest = sha(path)
    qa, result, old = qa_inputs(path, pair)
    if sha(path) != input_digest:
        raise ValueError("Completed QA report changed while reading it")
    reuse = None
    if profile.get("reuse_experiment"):
        dataset = Path(config.E2E_REUSE_DATASET).expanduser().resolve()
        if not dataset.is_file():
            raise FileNotFoundError(f"Place the QASPER test archive at {dataset}")
        digest = sha(dataset)
        if digest != config.E2E_REUSE_DATASET_SHA256:
            raise ValueError(f"QASPER test archive fingerprint differs at {dataset}")
        reuse = {"path": str(dataset), "sha256": digest, "dataset": "QASPER v0.3 test"}
        if not profile["scenarios"]:
            raise ValueError("Reuse profiles require at least one scenario")
    elif not profile.get("fit_only") and not 3 <= profile["documents"] <= len(old["cohort"]):
        raise ValueError("Profile document count is outside the completed QA cohort")
    if not Path(old["dataset"]).is_file():
        raise FileNotFoundError(old["dataset"])
    for snapshot in result["model_snapshots"].values():
        if not (Path(snapshot) / "config.json").is_file():
            raise FileNotFoundError(f"Model snapshot is missing at {snapshot}")
    common = {
        "repo": {"path": str(root), "git": git_state(root)},
        "pair": pair, "profile_name": profile_name, "profile": profile,
        "qa_report": str(path), "qa_report_sha256": input_digest,
        "qa_commit": qa["commit"], "dataset_sha256": result["dataset_sha256"],
        "models": result["models"], "model_snapshots": result["model_snapshots"],
        "minimum_free_gib": config.E2E_MIN_FREE_GIB[result["models"]["target"]["name"]],
        "seed": config.E2E_SEED,
    }
    if profile.get("fit_only"):
        initializer = config.E2E_CACHEBRIDGE_V2_INITIALIZERS.get(pair)
        if initializer is None:
            raise ValueError(f"No CacheBridge v2 initializer configured for {pair}")
        artifact_manifest("cachebridge", initializer)
        return {
            **common, "reference_report": qa["reference_report"],
            "reference_report_sha256": qa["reference_report_sha256"],
            "cachebridge_v2_fit": {"initializer": str(initializer),
                                   "max_history": config.E2E_CACHEBRIDGE_V2_MAX_HISTORY},
        }
    external_checkout = os.environ.get(CHECKOUT_ENV)
    semcache = Path(external_checkout or config.E2E_SEMCACHE_ROOT).expanduser().resolve()
    if not (semcache / "src/semcache/edgelora/r5_semantic_index.py").is_file():
        raise FileNotFoundError(f"R5 semantic index missing in {semcache}; set E2E_SEMCACHE_ROOT")
    semcache_git = None
    if external_checkout or any("cache_policy" in case for case in profile["scenarios"]):
        semcache_git = d5_git_state(semcache)
    if profile.get("lora_stage"):
        base = result["fit"]["artifact"]
        manifest = artifact_manifest("rarecache", base)
        if manifest != result["fit"]["manifest"]:
            raise ValueError("Base RaReCache artifact differs from the completed QA run")
        methods, artifacts, manifests = [], {"rarecache": base}, {"rarecache": manifest}
    else:
        methods, artifacts, manifests = artifacts_for(pair, result, old, config.E2E_CACHEBRIDGE, profile_name)
    if profile.get("compare_cachebridge_v2"):
        job = str(config.E2E_CACHEBRIDGE_V2_FIT_JOB)
        if not job.isdecimal():
            raise ValueError("Set E2E_CACHEBRIDGE_V2_FIT_JOB to the completed fit job ID")
        fit_dir = locate_e2e_fit(config, job)
        fit = read(fit_dir / REPORT)
        artifact = fit_dir / "artifact"
        if (fit["status"] != "complete" or fit["settings"]["profile_name"] != "cachebridge_v2_fit"
                or fit["settings"]["pair"] != pair or fit["settings"]["models"] != common["models"]
                or fit["settings"]["reference_report_sha256"] != qa["reference_report_sha256"]
                or fit["settings"]["dataset_sha256"] != common["dataset_sha256"]
                or "cachebridge" not in methods):
            raise ValueError("CacheBridge v2 fit and comparison inputs differ")
        manifest = artifact_manifest("cachebridge_v2", artifact)
        if fit.get("artifact_manifest") != manifest:
            raise ValueError("CacheBridge v2 checkpoint differs from the completed fit")
        methods.insert(methods.index("cachebridge") + 1, "cachebridge_v2")
        artifacts["cachebridge_v2"] = str(artifact)
        manifests["cachebridge_v2"] = manifest
    hub_cache = getattr(config, "E2E_HF_HUB_CACHE", None)
    settings = {
        **({"reuse_dataset": reuse} if reuse is not None else {}),
        **common,
        "artifacts": artifacts, "artifact_manifest": manifests, "methods": methods,
        "semcache_root": str(semcache),
        **({"semcache_git": semcache_git} if semcache_git is not None else {}),
        "tinybert_dir": str(config.E2E_TINYBERT_DIR) if config.E2E_TINYBERT_DIR else None,
        "hf_hub_cache": str(Path(hub_cache).expanduser().resolve()) if hub_cache else None,
        "repair_fraction": config.E2E_REPAIR_FRACTION,
    }

    if profile.get("lora_stage"):
        from .lora_transfer.settings import resolve_lora

        return resolve_lora(config, settings)
    return settings


def longbench_rows(archive, datasets):
    rows = {}
    with ZipFile(archive) as bundle:
        for dataset in datasets:
            paths = [p for p in bundle.namelist() if PurePosixPath(p).name == dataset + ".jsonl"]
            if len(paths) != 1:
                raise ValueError(f"Expected exactly one {dataset}.jsonl archive entry")
            with bundle.open(paths[0]) as stream:
                rows[dataset] = {i: json.loads(line) for i, line in enumerate(stream, 1) if line.strip()}

    return rows


def check_tokenizers(template, tokenizers):
    vocab = template.get_vocab()
    if any(t.get_vocab() != vocab or t.special_tokens_map != template.special_tokens_map for t in tokenizers):
        raise ValueError("Qwen3 tokenizer vocabularies or specials differ")


def render_qa_prompt(context, question, template, tokenizers):
    marker = "__KV_REUSE_BOUNDARY_4459__"
    if marker in context or marker in question:
        raise ValueError("Unexpected QA boundary marker in input")
    content = (
        "Read the following document and answer the question using the document. "
        "Provide only the short answer. If the document does not answer the question, "
        "respond with unanswerable.\n\nDocument:\n"
        f"{context}{marker}\n\nQuestion: {question}\nAnswer:"
    )
    rendered = template.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    prefix, suffix = rendered.split(marker)
    text = prefix + suffix
    options = {"add_special_tokens": False, "return_attention_mask": False, "verbose": False}
    encoding = template(text, return_offsets_mapping=True, **options)
    ids = encoding["input_ids"]
    boundary = next((i for i, (_, end) in enumerate(encoding["offset_mapping"])
                     if end > len(prefix)), len(ids))
    if not 0 < boundary < len(ids) or any(tokenizer(text, **options)["input_ids"] != ids
                                           for tokenizer in tokenizers):
        raise ValueError("QA prompt has an invalid boundary or different token IDs across models")
    return ids, boundary


def frozen_prompts(archive, cohort, template, tokenizers):
    """Reconstruct the original Qwen QA template and verify each frozen item."""
    from ..src.exp.mapper_optim.inputs import token_hash

    check_tokenizers(template, tokenizers)
    rows = longbench_rows(archive, {item["dataset"] for item in cohort})
    prompts = []
    seen = set()
    for item in cohort:
        row = rows[item["dataset"]][item["line"]]
        context, question = row["context"], row["input"].strip()
        ids, boundary = render_qa_prompt(context, question, template, tokenizers)
        actual = (hashlib.sha256(context.strip().encode()).hexdigest(), token_hash(ids),
                  len(ids), boundary, row["answers"])
        expected = tuple(item[name] for name in ("document_sha256", "token_sha256", "prompt_tokens",
                                                "source_prefix_tokens", "answers"))
        if actual != expected:
            raise ValueError(f"Frozen QA prompt differs at {item['dataset']}:{item['line']}")
        canonical = " ".join(context.split()).casefold()
        if canonical in seen:
            raise ValueError("Repeated held-out document")
        seen.add(canonical)
        prompts.append((ids, boundary))
    return prompts
