"""Replay a completed LongBench cohort through the existing serving engine."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
import gc
import os
from pathlib import Path
from statistics import median
import sys

from . import config
from .e2e_inputs import (
    REPORT, ROLES, artifact_manifest, frozen_prompts, qa_inputs, resolve_settings, sha, write,
)
from .e2e_workload import documents, make_trace, paired_deltas, route_check, score_answer, summarize
from .semcache_smoke import TINYBERT_REVISION, git_state
from .serving_metrics import measure_replays
from ..src.cache.prefix import PrefixCache
from ..src.serving.engine import ServingEngine


@contextmanager
def load_transfer(method, settings, identities, snapshots, target, device):
    from ..src.kv_transfer.base_model import BaseModelKVTransfer

    if method in {"full", "exact"}:
        yield None, None
        return
    artifact = Path(settings["artifacts"][method])
    if method in {"rarecache", "cachebridge", "cachebridge_v2"}:
        from ..src.kv_transfer.affine import AffineKVMapper

        mapper = AffineKVMapper(artifact, method, identities, snapshots, device,
                                repair_fraction=settings["repair_fraction"])
        yield BaseModelKVTransfer(prefix_mappers=(mapper,)), {
            "artifact": mapper.metadata, "repair_fraction": mapper.repair_fraction, "storage": "GPU",
        }
        return
    if method == "ridge":
        from xmodel_kv.artifact import LinearKVMapper

        mapper = LinearKVMapper(artifact)
        mapper.preload_for_model(target)
        storage = "GPU"
    else:
        from ..src.exp.mapper.transfer import NeuralKVMapper

        mapper = NeuralKVMapper(artifact)
        mapper.assert_complete()
        storage = "streamed from checkpoint files"
    expected_pair = tuple(identity.name for identity in identities)
    if (mapper.config.source_model, mapper.config.target_model) != expected_pair:
        raise ValueError(f"Fitted {method} is for a different directed pair")
    yield BaseModelKVTransfer(mapper), {"storage": storage}


def run_trace(runtime, method, transfer, make_index, trace, case, budget, codec):
    policy_name = case.get("cache_policy", "lru")
    policy = None
    if policy_name != "lru" and method != "full":
        from ..src.cache.semcache_policy import SemCachePrefixPolicy

        if policy_name != SemCachePrefixPolicy.name:
            raise ValueError(f"Unknown native prefix cache policy: {policy_name}")
        policy = SemCachePrefixPolicy()
    cache = None if method == "full" else PrefixCache(
        make_index(), case["cache_mib"] << 20, codec=codec, policy=policy)
    engine = ServingEngine(runtime, cache=cache, transfer=transfer,
                           collect_cache_metrics="cache_policy" in case)
    rows = []
    for event in trace:
        if "reuse_count" in case:
            before_bytes = cache.bytes if cache is not None else 0
            before_evictions = cache.evictions if cache is not None else 0
        row = engine.serve(event.request, 1 if event.kind == "producer" else budget)
        route_check(event, row, method)
        if "reuse_count" in case:
            row["resident_kv_bytes_before"] = before_bytes
            row["cache_evictions"] = cache.evictions - before_evictions if cache is not None else 0
        rows.append(row)
    return rows


def add_quality(rows, trace, tokenizer, eos):
    for row, event in zip(rows, trace, strict=True):
        row.update(kind=event.kind, group=event.document["group"],
                   document_sha256=event.document["document_sha256"],
                   token_sha256=event.document["token_sha256"],
                   source_prefix_tokens=event.document["source_prefix_tokens"])
        if "paper_id" in event.document:
            row.update(paper_id=event.document["paper_id"], question_id=event.document["question_id"])
        if event.kind != "producer":
            ids = row["generated_ids"]
            length = next((i for i, token in enumerate(ids) if token in eos), len(ids))
            text = tokenizer.decode(ids[:length], skip_special_tokens=True)
            row.update(answer=text, **score_answer(text, event.document["answers"]))


def pilot_check(rows, method):
    required = {"full_recompute"} if method == "full" else {"native_kv_hit", "miss_recompute"}
    if method not in {"full", "exact"}:
        required.add("cross_model_kv_hit")
    if absent := required - {row["path"] for row in rows}:
        raise RuntimeError(f"{method} pilot did not exercise {sorted(absent)}")


def models_and_prompts(settings, qa, old, device, *, load_runtime=True):
    import torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    from transformers import AutoTokenizer

    from ..src.exp.mapper_optim.inputs import ModelIdentity
    from ..src.models.qwen3 import QwenRuntime

    identities = tuple(ModelIdentity(**settings["models"][role]) for role in ROLES)
    snapshots = tuple(Path(settings["model_snapshots"][role]) for role in ROLES)
    models = tuple(ModelIdentity.inspect(str(path), info.revision)
                   for path, info in zip(snapshots, identities, strict=True))
    for model, identity in zip(models, identities, strict=True):
        if replace(model, name=identity.name) != identity:
            raise ValueError(f"Local checkpoint geometry differs for {identity.name}")
    tokenizers = tuple(AutoTokenizer.from_pretrained(str(path), local_files_only=True) for path in snapshots)
    reference = old["geometry"]["target"]
    saved = {identity.name: (str(path), identity.revision)
             for identity, path in zip(identities, snapshots, strict=True)}
    model_id, revision = reference["name"], reference["revision"]
    if model_id in saved:
        template_path, template_revision = saved[model_id]
    elif model_id in qa.get("local_models", {}):
        local = qa["local_models"][model_id]
        template_path, template_revision = local["path"], local["revision"]
    else:
        hub_cache = settings.get("hf_hub_cache")
        try:
            config_path = hf_hub_download(
                model_id, "config.json", revision=revision,
                cache_dir=hub_cache, local_files_only=True,
            )
        except LocalEntryNotFoundError as exc:
            raise FileNotFoundError(
                f"Frozen QA tokenizer snapshot {model_id}@{revision} is missing in "
                f"{hub_cache or 'the default Hugging Face cache'}. "
                "Set E2E_HF_HUB_CACHE to the hub directory containing this snapshot."
            ) from exc
        template_path = str(Path(config_path).parent)
        template_revision = Path(template_path).name
    if template_revision != revision:
        raise ValueError("Pinned reference Qwen tokenizer revision differs from the QA template")
    template = AutoTokenizer.from_pretrained(template_path, local_files_only=True)
    if reuse_dataset := settings.get("reuse_dataset"):
        from .reuse_experiment.qasper_cohort import load_reuse_cohort

        cohort, prompts = load_reuse_cohort(
            reuse_dataset, old, settings["profile"], template, tokenizers, settings["seed"],
        )
    else:
        cohort = old["cohort"]
        prompts = frozen_prompts(old["dataset"], cohort, template, tokenizers)
    if not load_runtime:
        return None, identities, snapshots, cohort, prompts
    keys = [identity.name for identity in identities]
    runtime = QwenRuntime(
        {name: model.load(device, torch.bfloat16) for name, model in zip(keys, models, strict=True)},
        dict(zip(keys, tokenizers, strict=True)), device,
    )
    return runtime, identities, snapshots, cohort, prompts


def evaluate(settings, report, output):
    import socket

    semcache_root = Path(settings["semcache_root"])
    if expected_git := settings.get("semcache_git"):
        from .semcache_d5 import d5_git_state

        if d5_git_state(semcache_root) != expected_git:
            raise ValueError("SemCache source changed after resolving settings")
    semcache = semcache_root / "src"
    root = Path(settings["repo"]["path"])
    sys.path[:0] = [str(semcache), str(root / "baseline/crosskv/src")]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import torch
    import transformers
    from semcache.edgelora.r5_semantic_index import SemanticIndex
    from semcache.semantic.encoder import TinyBERTSemanticEncoder

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA initialization failed or the Slurm job does not expose exactly one GPU")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    report["environment"] = {"gpu": torch.cuda.get_device_name(device), "hostname": socket.gethostname(),
                             "gpu_free_before_load_bytes": free, "gpu_total_bytes": total,
                             "torch": str(torch.__version__), "cuda": torch.version.cuda,
                             "transformers": transformers.__version__, "dtype": "bfloat16"}
    write(output, report)
    if torch.cuda.get_device_capability(device)[0] < 8 or free < settings["minimum_free_gib"] * (1 << 30):
        raise RuntimeError(f"Expected native BF16 and {settings['minimum_free_gib']} GiB free VRAM, "
                           f"found {free / (1 << 30):.2f} GiB")
    if git_state(root) != settings["repo"]["git"]:
        raise ValueError("Repository state changed after starting the job")
    if sha(settings["qa_report"]) != settings["qa_report_sha256"]:
        raise ValueError("QA report changed after starting the job")
    for method, path in settings["artifacts"].items():
        if artifact_manifest(method, path) != settings["artifact_manifest"][method]:
            raise ValueError(f"{method} checkpoint metadata or sizes changed after submission")
    qa, _, old = qa_inputs(settings["qa_report"], settings["pair"])
    if sha(old["dataset"]) != settings["dataset_sha256"]:
        raise ValueError("The LongBench archive differs from the mapper QA experiment")
    if (str(torch.__version__), transformers.__version__) != (
            old["options"]["expected_torch"], old["options"]["expected_transformers"]):
        raise RuntimeError("Use the PyTorch and Transformers versions of the completed QA job")
    torch.manual_seed(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    lora_stage = settings["profile"].get("lora_stage")
    if lora_stage == "fit":
        from .lora_transfer.fit import fit

        _, _, _, cohort, prompts = models_and_prompts(settings, qa, old, device, load_runtime=False)
        fit(settings, report, output, device, prompts, cohort)
        return
    cachebridge_initializers = None
    if "lora_cachebridge_v2" in settings["methods"]:
        from .lora_transfer.cachebridge_v2 import fit_initializers

        _, _, _, cohort, prompts = models_and_prompts(settings, qa, old, device, load_runtime=False)
        cachebridge_initializers = fit_initializers(settings, report, output, device, prompts, cohort)
    encoder_model = settings["tinybert_dir"] or TinyBERTSemanticEncoder.DEFAULT_MODEL_ID
    try:
        encoder = TinyBERTSemanticEncoder(
            model_id=encoder_model, revision=TINYBERT_REVISION,
            local_files_only=True, device="cpu", dtype="float32",
        )
    except OSError as exc:
        raise OSError(
            f"TinyBERT encoder could not be loaded offline from {encoder_model!r} "
            f"(revision {TINYBERT_REVISION}): {exc}. "
            "Check E2E_TINYBERT_DIR for the local encoder files."
        ) from exc
    encoder.encode(["Warmup."])

    runtime, identities, snapshots, cohort, prompts = models_and_prompts(settings, qa, old, device)
    runtime.warmup()
    if lora_stage:
        from .lora_transfer.replay import (
            activate_adapters, assign_adapters, transfer_for, verify_routes,
        )

        activate_adapters(runtime, identities, settings["lora_routes"])
        runtime.sync()
    if cachebridge_initializers is not None:
        from .lora_transfer.cachebridge_v2 import calibrate_readouts

        calibrate_readouts(settings, report, output, old, runtime, device, cachebridge_initializers)
    pair = tuple(item.name for item in identities)
    target = runtime.models[pair[1]]
    end = target.generation_config.eos_token_id
    eos = {end} if end is None or isinstance(end, int) else set(end)
    profile = settings["profile"]
    reuse = "reuse_dataset" in settings
    selected = (list(zip(cohort, prompts, strict=True)) if reuse
                else documents(cohort, prompts, profile["documents"], settings["seed"]))
    report["cohort"] = [item for item, _ in selected]
    if max(len(ids) for _, (ids, _) in selected) + profile["new_tokens"] > min(x.context for x in identities):
        raise ValueError("Selected profile exceeds model context length")
    report["measurement_scope"] = (
        "one GPU, both models resident, sequential ServingEngine requests; TTFT and E2E on QA requests; "
        "memory and throughput include producers; fixed decode budget, score before first EOS"
    )
    if reuse or lora_stage:
        report["measurement_scope"] += "; elapsed per target QA includes all producer requests"
    report["semantic_encoder"] = encoder.metadata
    base_bytes = torch.cuda.memory_allocated(device)

    traces = {}
    for case in profile["scenarios"]:
        name = case["id"]
        report["results"][name] = {"scenario": case, "methods": {}}
        tokenizer = runtime.tokenizers[pair[1]]
        if reuse:
            from .reuse_experiment.request_schedule import make_reuse_trace

            trace = make_reuse_trace(selected, tokenizer, pair, case, profile, settings["seed"])
        else:
            trace = make_trace(selected, tokenizer, pair, case, profile["window"], settings["seed"])
        if lora_stage:
            trace = assign_adapters(trace, settings["lora_routes"][name])
        warm_ids = set()
        for kind in ("same", "cross", "cold"):
            candidates = [event for event in trace if event.kind == kind]
            if candidates:
                longest = max(candidates, key=lambda event: len(event.request.token_ids))
                warm_ids.update((longest.request.request_id, longest.prefix_request_id))
        warm_trace = [event for event in trace if event.request.request_id in warm_ids]
        codec = None
        if case["codec"] == "bitpack5":
            from ..src.cache.kv_bitpack import KVBitpackCodec

            codec = KVBitpackCodec()
        traces[name] = trace, warm_trace, codec

    def make_index():
        return SemanticIndex(encoder, clusters=1)

    for method in settings["methods"]:
        context = (nullcontext((None, None)) if lora_stage
                   else load_transfer(method, settings, identities, snapshots, target, device))
        with context as (transfer, metadata):
            runtime.sync()
            for case in profile["scenarios"]:
                name = case["id"]
                print(f"{name}: {method}", flush=True)
                if lora_stage:
                    transfer, metadata = transfer_for(
                        method, settings, name, identities, snapshots, device,
                        cachebridge_fit=report.get("cachebridge_v2_lora_fits", {}).get(name),
                    )
                trace, warm_trace, codec = traces[name]

                def replay(events, budget):
                    return run_trace(runtime, method, transfer, make_index, events, case, budget, codec)

                samples = measure_replays(
                    lambda: replay(trace, profile["new_tokens"]), device=device, repeats=profile["repeats"],
                    warmup=lambda: replay(warm_trace, 2),
                )
                replays = []
                for rows, latency in samples:
                    add_quality(rows, trace, runtime.tokenizers[pair[1]], eos)
                    replays.append({
                        "requests": rows, **latency,
                        "requests_per_second": len(rows) / latency["wall_seconds"],
                        "peak_allocated_bytes": latency["peak_gpu_allocated_bytes"],
                        "peak_reserved_bytes": latency["peak_gpu_reserved_bytes"],
                        "model_allocated_bytes": base_bytes,
                        "mapper_allocated_bytes": latency["start_gpu_allocated_bytes"] - base_bytes,
                    })
                summary, per_doc = summarize(
                    replays, trace=trace if reuse else None, cache_capacity_bytes=case["cache_mib"] << 20,
                )
                if settings["profile_name"] in {"pilot", "reuse_pilot"}:
                    pilot_check(per_doc, method)
                if lora_stage:
                    verify_routes((row for replay in replays for row in replay["requests"]
                                   if row["kind"] != "producer"), method)
                    summary["elapsed_ms_per_target_qa"] = median(
                        1000 * replay["wall_seconds"] / len(per_doc) for replay in replays)
                methods = report["results"][name]["methods"]
                delta = {} if method == "full" else paired_deltas(
                    per_doc, methods["full"]["documents"], settings["seed"],
                )
                methods[method] = {"summary": summary, "documents": per_doc, "replays": replays,
                                   "mapper": metadata, "paired_vs_full": delta}
                if case.get("cache_policy") == "semcache_frequency_age_size":
                    baseline = report["results"]["lru"]["methods"][method]
                    for sample, reference in zip(replays, baseline["replays"], strict=True):
                        ids = [row["request_id"] for row in sample["requests"]]
                        if ids != [row["request_id"] for row in reference["requests"]]:
                            raise ValueError("LRU and SemCache used different ordered requests")
                    methods[method]["paired_vs_lru"] = {
                        **paired_deltas(per_doc, baseline["documents"], settings["seed"]),
                        "trace_wall_ms_by_replay": [
                            1000 * (sample["wall_seconds"] - reference["wall_seconds"])
                            for sample, reference in zip(replays, baseline["replays"], strict=True)
                        ],
                    }
                write(output, report)
                if lora_stage:
                    transfer = None
                    torch.cuda.empty_cache()
        del transfer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    job = os.environ.get("SLURM_JOB_ID")
    if job is None or not job.isdecimal():
        raise RuntimeError("Submit run/seraph/qwen3_serving_e2e.sh through Slurm")
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    directory = config.E2E_OUTPUT_ROOT / (f"e2e_{job}_r{restart}" if restart else f"e2e_{job}")
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / REPORT
    report = {"schema_version": 1, "status": "running", "slurm_job": job,
              "slurm_nodes": os.environ.get("SLURM_JOB_NODELIST"),
              "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "results": {}}
    write(output, report)
    print(f"Report {output}", flush=True)
    try:
        settings = resolve_settings(config)
        report["settings"] = settings
        write(directory / "settings.json", settings)
        write(output, report)
        print(f"Pair {settings['pair']} | Profile {settings['profile_name']}", flush=True)
        if settings["profile"].get("fit_only"):
            from .cachebridge_v2_fit import fit

            fit(settings, report, directory)
        else:
            print(f"Methods {', '.join(settings['methods'])}", flush=True)
            evaluate(settings, report, output)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(output, report)
        raise
    report["status"] = "complete"
    write(output, report)
    if settings["profile"].get("fit_only") or settings["profile"].get("lora_stage") == "fit":
        print(f"Report {output}", flush=True)
        return
    try:
        from .e2e_figures import render

        render(output, output.parent / "figures")
    except Exception as error:
        report["figures"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        print(f"Metrics saved. Plotting failed: {error}", flush=True)
    else:
        report["figures"] = {"status": "complete", "directory": str(output.parent / "figures")}
    write(output, report)
    print(f"Report {output}", flush=True)


if __name__ == "__main__":
    main()
