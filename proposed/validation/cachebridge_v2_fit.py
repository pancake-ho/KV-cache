"""Fit a CacheBridge v2 mapper from local, disjoint calibration examples."""

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys
from time import perf_counter

from .e2e_inputs import (ROLES, artifact_manifest, check_tokenizers, longbench_rows,
                         read, render_qa_prompt, sha, write)


def readout_cases(reference, tokenizers, max_history):
    if max_history < 2:
        raise ValueError("CacheBridge readout requires at least two history tokens")
    docs = reference["calibration_documents"]
    evaluation = {row["document_sha256"] for row in reference["cohort"]}
    rows = longbench_rows(reference["dataset"], {row["dataset"] for row in docs})
    source, target = tokenizers
    check_tokenizers(target, (source,))
    selected, cases = [], []
    for doc in docs:
        record = rows[doc["dataset"]][doc["line"]]
        context, question = record["context"], record["input"].strip()
        fingerprint = hashlib.sha256(context.strip().encode()).hexdigest()
        if fingerprint != doc["document_sha256"] or fingerprint in evaluation:
            raise ValueError("CacheBridge readout calibration document fingerprint or split differs")
        ids, boundary = render_qa_prompt(context, question, target, (source,))
        capped = min(boundary, max_history)
        cases.append((ids[:capped] + ids[boundary:], capped))
        selected.append({"dataset": doc["dataset"], "line": doc["line"],
                         "document_sha256": fingerprint, "prefix_tokens": capped})
    if len({row["document_sha256"] for row in selected}) != len(selected) or len(selected) < 2:
        raise ValueError("Readout calibration requires distinct held-out-from-evaluation documents")
    return selected, cases


def fit(settings, report, directory):
    import torch
    import transformers

    sys.path.insert(0, str(Path(settings["repo"]["path"]) / "baseline/crosskv/src"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from ..src.exp.cachebridge.readout import fit_readout
    from ..src.exp.mapper_optim.inputs import ModelIdentity, token_hash
    from ..src.exp.mapper_optim.transfer import AffineMapper

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CacheBridge readout fitting requires one BF16-capable GPU")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free < settings["minimum_free_gib"] * (1 << 30):
        raise RuntimeError(f"CacheBridge readout fitting requires at least {settings['minimum_free_gib']} GiB free")
    report["environment"] = {"gpu": torch.cuda.get_device_name(device), "gpu_total_bytes": total,
                             "torch": str(torch.__version__), "transformers": transformers.__version__,
                             "cuda": torch.version.cuda, "dtype": "bfloat16"}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(settings["seed"])
    root = Path(settings["cachebridge_v2_fit"]["initializer"])
    source, target = tuple(ModelIdentity(**settings["models"][r]) for r in ROLES)
    snapshots = tuple(settings["model_snapshots"][r] for r in ROLES)
    if sha(settings["reference_report"]) != settings["reference_report_sha256"]:
        raise ValueError("Reference report changed before readout calibration")
    reference = read(settings["reference_report"])
    expected_versions = reference["options"]["expected_torch"], reference["options"]["expected_transformers"]
    if (str(torch.__version__), transformers.__version__) != expected_versions:
        raise RuntimeError(f"Use the completed QA job environment: {expected_versions}")
    if sha(reference["dataset"]) != settings["dataset_sha256"]:
        raise ValueError("Calibration archive fingerprint differs")
    initializer = AffineMapper(root, "cachebridge", device)
    spec = initializer.spec
    if (spec["variant"] != "head-local" or spec["dtype"] != "bfloat16" or source.heads != target.heads
            or spec["selected_layers"]["k"] != spec["selected_layers"]["v"]):
        raise ValueError("Expected a BF16 head-local CacheBridge initializer with shared K/V support")
    for actual, expected, snapshot in zip((initializer.source, initializer.target), (source, target),
                                          snapshots, strict=True):
        if actual.name not in {expected.name, snapshot} or replace(actual, name=expected.name) != expected:
            raise ValueError(f"Initializer is for a different model or revision: {expected.name}")
    fit_examples = root / "fit_examples.json"
    examples = read(fit_examples)
    if not examples or any(len(row["ids"]) != spec["sequence_length"] or token_hash(row["ids"]) != row["sha256"]
                           for row in examples):
        raise ValueError("Initializer token data is missing or its fingerprint is invalid")
    tokenizers = tuple(transformers.AutoTokenizer.from_pretrained(p, local_files_only=True) for p in snapshots)
    selected, cases = readout_cases(reference, tokenizers, settings["cachebridge_v2_fit"]["max_history"])
    if not all(len(ids) <= min(source.context, target.context) for ids, _ in cases):
        raise ValueError("CacheBridge readout examples exceed a model context limit")
    local = tuple(ModelIdentity.inspect(path, info.revision)
                  for path, info in zip(snapshots, (source, target), strict=True))
    for identity, expected in zip(local, (source, target), strict=True):
        if replace(identity, name=expected.name) != expected:
            raise ValueError(f"Local snapshot no longer matches pinned geometry: {expected.name}")
    source_model, target_model = (info.load(device, torch.bfloat16) for info in local)
    artifact = directory / "artifact"
    artifact.mkdir()
    started = perf_counter()
    report["fit_layers"] = fit_readout(source_model, target_model, initializer,
                                       [item["ids"] for item in examples], cases, artifact)
    if not any(layer["changed_value_heads"] for layer in report["fit_layers"]):
        raise RuntimeError("CacheBridge v2 fitted no Value corrections")
    write(artifact / "calibration.json", {
        **spec, "method": "cachebridge_v2", "variant": "layer-ordered-readout",
        "initializer_metadata_sha256": sha(root / "calibration.json"),
        "initializer_examples_sha256": sha(fit_examples),
        "reference_report_sha256": sha(settings["reference_report"]),
        "dataset_sha256": settings["dataset_sha256"], "readout_documents": selected,
        "max_readout_history": settings["cachebridge_v2_fit"]["max_history"],
    })
    report.update(artifact=str(artifact), artifact_manifest=artifact_manifest("cachebridge_v2", artifact),
                  fit_seconds=perf_counter() - started)
