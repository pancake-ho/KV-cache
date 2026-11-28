"""Compare paired Target UD TTFT over the existing Seraph namespace/veth link."""

import argparse
import json
import os
from pathlib import Path
from dataclasses import replace
import sys

from .link import connect_ud
from .metrics.record import measure_request, summarize_trials, write_row
from .rx.transport import accept_es


def roundtrip_smoke(link, size):
    """Check both directions by comparing the bytes returned by each receiver."""
    seed = bytes(range(251))
    payload = (seed * (size // len(seed) + 1))[:size]
    (received,), uplink = link.exchange([payload], from_ud=True)
    if received != payload:
        raise RuntimeError("TCP uplink corrupted the payload")
    (returned,), downlink = link.exchange([received], from_ud=False)
    if returned != payload:
        raise RuntimeError("TCP downlink corrupted the payload")
    return {"network": link.metadata, "up": uplink, "down": downlink}


def _load_settings(args):
    from ..validation import config
    from ..validation.e2e_inputs import resolve_settings

    config.E2E_PAIR = args.pair
    config.E2E_QA_JOB = args.qa_job
    config.E2E_QA_REPORT = args.qa_report
    config.E2E_REPAIR_FRACTION = args.repair
    if args.lora_route:
        config.E2E_LORA_TARGET_ADAPTERS = (args.lora_route,)
    os.environ.update(
        KV_TOP_E2E_PROFILE="lora_transfer_pilot" if args.lora_route else "raw_sweep",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
    )
    settings = resolve_settings(config)
    sys.path[:0] = [
        str(Path(settings["semcache_root"]) / "src"),
        str(Path(settings["repo"]["path"]) / "baseline/crosskv/src"),
    ]
    return settings


def _mapper(args, settings, runtime, identities, snapshots, device):
    from ..src.kv_transfer.affine import AffineKVMapper
    from ..src.kv_transfer.base_model import BaseModelKVTransfer

    if args.lora_route:
        from ..validation.lora_transfer.replay import activate_adapters, transfer_for

        activate_adapters(runtime, identities, settings["lora_routes"])
        guard, _ = transfer_for(
            "lora_fitted", settings, args.lora_route, identities, snapshots, device,
        )
        return guard, guard.mapper

    mapper = AffineKVMapper(
        Path(settings["artifacts"]["rarecache"]), "rarecache",
        identities, snapshots, device, repair_fraction=args.repair,
    )
    return BaseModelKVTransfer(prefix_mappers=(mapper,)), mapper


def run(args, link, file):
    import torch
    import transformers

    settings = _load_settings(args)
    from semcache.edgelora.r5_semantic_index import SemanticIndex
    from semcache.semantic.encoder import TinyBERTSemanticEncoder

    from ..src.serving.engine import ServingEngine
    from ..validation.e2e_inputs import qa_inputs, sha
    from ..validation.e2e_workload import documents, make_trace, route_check
    from ..validation.lora_transfer.replay import assign_adapters
    from ..validation.semcache_smoke import TINYBERT_REVISION
    from ..validation.serving_e2e import add_quality, models_and_prompts
    from .cache_path import NetworkPrefixCache, NetworkRaReCache

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU")
    device = torch.device("cuda:0")
    free, _ = torch.cuda.mem_get_info(device)
    if (torch.cuda.get_device_capability(device)[0] < 8
            or free / (1 << 30) < settings["minimum_free_gib"]):
        raise RuntimeError(f"Need BF16 and {settings['minimum_free_gib']} GiB GPU memory free")
    torch.manual_seed(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False

    qa, _, reference = qa_inputs(settings["qa_report"], args.pair)
    if sha(reference["dataset"]) != settings["dataset_sha256"]:
        raise ValueError("The LongBench archive differs from the QA calibration input")
    if (str(torch.__version__), transformers.__version__) != (
            reference["options"]["expected_torch"], reference["options"]["expected_transformers"]):
        raise RuntimeError("Use the PyTorch and Transformers versions from the completed QA job")
    runtime, identities, snapshots, cohort, prompts = models_and_prompts(
        settings, qa, reference, device,
    )
    runtime.warmup()
    guard, mapper = _mapper(args, settings, runtime, identities, snapshots, device)
    pair = tuple(identity.name for identity in identities)
    tokenizer = runtime.tokenizers[pair[1]]
    encoder = TinyBERTSemanticEncoder(
        model_id=settings["tinybert_dir"] or TinyBERTSemanticEncoder.DEFAULT_MODEL_ID,
        revision=TINYBERT_REVISION, local_files_only=True, device="cpu", dtype="float32",
    )
    encoder.encode(["Warmup."])
    if args.samples > len(cohort):
        raise ValueError(f"Requested {args.samples} QA rows from a cohort of {len(cohort)}")
    selected = documents(cohort, prompts, args.samples, settings["seed"])
    if max(len(ids) for _, (ids, _) in selected) + args.new_tokens > min(
            identity.context for identity in identities):
        raise ValueError("Selected QA requests exceed the target context")
    case = {"id": "ud_es", "same_share": 0, "cross_share": 1, "cache_mib": 1024, "codec": "raw"}
    events = make_trace(selected, tokenizer, pair, case, 1, settings["seed"])
    if args.lora_route:
        events = assign_adapters(events, settings["lora_routes"][args.lora_route])
    producers = {event.request.request_id: event for event in events if event.kind == "producer"}
    targets = [event for event in events if event.kind == "cross"]
    end = runtime.models[pair[1]].generation_config.eos_token_id
    eos = {end} if end is None or isinstance(end, int) else set(end)
    codec = None
    if args.wire_codec == "bitpack5":
        from ..src.cache.kv_bitpack import KVBitpackCodec

        codec = KVBitpackCodec()
    transfer = NetworkRaReCache(guard, mapper, link, codec, args.source_placement)
    full = ServingEngine(runtime)

    def prepare(event, *, miss=False):
        cache = NetworkPrefixCache(
            SemanticIndex(encoder, clusters=1), case["cache_mib"] << 20, link, pair[0],
            args.source_placement,
        )
        engine = ServingEngine(runtime, cache=cache, transfer=transfer)
        source = producers[event.prefix_request_id].request
        if miss:
            source = replace(source, share_group=f"{source.share_group}:miss")
        row = engine.serve(source, 1)
        if source.request_id not in cache.entries:
            raise RuntimeError("ES index has no eligible source prefix")
        return engine, {
            "source_application_ms": row["application_ms"],
            "source_prefill_ms": row["prefill_ms"],
            "setup_upload": cache.setup_upload,
        }

    def measure(engine, event, budget, *, expected_path):
        row = measure_request(engine, event.request, budget, device)
        if row["path"] != expected_path:
            raise RuntimeError(
                f"{event.request.request_id}: expected {expected_path}, got {row['path']}"
            )
        route_check(event, row, "full" if expected_path == "full_recompute" else "rarecache")
        if expected_path != "full_recompute":
            row.update(engine.cache.query_metrics)
        if expected_path == "cross_model_kv_hit":
            row.update(transfer.metrics)
        add_quality([row], [event], tokenizer, eos)
        return row

    def measure_pair(engine, event, expected_path, *, reuse_first):
        if reuse_first:
            reused = measure(engine, event, args.new_tokens, expected_path=expected_path)
            baseline = measure(full, event, args.new_tokens, expected_path="full_recompute")
        else:
            baseline = measure(full, event, args.new_tokens, expected_path="full_recompute")
            reused = measure(engine, event, args.new_tokens, expected_path=expected_path)
        return baseline, reused

    write_row(file, {"kind": "settings", "network": link.metadata, "serving": settings,
                      "args": vars(args), "gpu": torch.cuda.get_device_name(device),
                      "torch": str(torch.__version__), "cuda": torch.version.cuda,
                      "transformers": transformers.__version__, "dtype": "bfloat16",
                      "measurement_scope": "one GPU and two resident models; serial requests; "
                      "source generation outside target TTFT; TCP connection already open; "
                      "GPU memory includes the same resident source/mapper in both paths; "
                      "source and target UD are logical roles on a single UD link"})

    with torch.inference_mode():
        warmup = max(targets, key=lambda event: len(event.request.token_ids))
        warmup_tokens = min(args.new_tokens, 2)
        engine, _ = prepare(warmup)
        measure(full, warmup, warmup_tokens, expected_path="full_recompute")
        measure(engine, warmup, warmup_tokens, expected_path="cross_model_kv_hit")
        del engine
        if args.measure_cache_miss:
            engine, _ = prepare(warmup, miss=True)
            measure(engine, warmup, warmup_tokens, expected_path="miss_recompute")
            del engine

        paired_by_question = []
        for idx, event in enumerate(targets):
            trials = []
            for replay in range(args.repeats):
                reuse_first = bool((idx + replay) % 2)
                engine, setup = prepare(event)
                baseline, reuse = measure_pair(
                    engine, event, "cross_model_kv_hit", reuse_first=reuse_first,
                )
                del engine
                trial = {
                    "kind": "trial", "replay": replay, "request_id": event.request.request_id,
                    "prompt_tokens": len(event.request.token_ids),
                    "source_prefix_tokens": len(producers[event.prefix_request_id].request.token_ids),
                    "source_setup": setup, "full": baseline, "reuse": reuse,
                    "saved_ms": baseline["ttft_ms"] - reuse["ttft_ms"],
                }
                if args.measure_cache_miss:
                    engine, _ = prepare(event, miss=True)
                    miss_full, miss = measure_pair(
                        engine, event, "miss_recompute", reuse_first=not reuse_first,
                    )
                    del engine
                    trial.update(
                        miss=miss, miss_full_ttft_ms=miss_full["ttft_ms"],
                        miss_extra_ms=miss["ttft_ms"] - miss_full["ttft_ms"],
                    )
                write_row(file, trial)
                print(f"{event.request.request_id}, replay {replay}: {trial['saved_ms']:+.1f} ms",
                      flush=True)
                trials.append(trial)
            paired_by_question.append(trials)

    summary = summarize_trials(paired_by_question, args.repeats)
    write_row(file, summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("es", "smoke", "bench"))
    parser.add_argument("--es-ip", required=True)
    parser.add_argument("--ud-ip", required=True)
    parser.add_argument("--port", type=int, default=38197)
    parser.add_argument("--rendezvous", required=True)
    parser.add_argument("--socket-timeout", type=float, default=300)
    parser.add_argument("--smoke-mib", type=int, default=4)
    parser.add_argument("--pair", default="1.7b-8b")
    parser.add_argument("--qa-job", default="223131")
    parser.add_argument("--qa-report")
    parser.add_argument("--lora-route")
    parser.add_argument("--repair", type=float, default=0.30)
    parser.add_argument("--source-placement", choices=("es", "ud"), default="es")
    parser.add_argument("--wire-codec", choices=("raw", "bitpack5"), default="raw")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--out")
    parser.add_argument("--measure-cache-miss", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 < args.socket_timeout < float("inf"):
        parser.error("Invalid port or socket timeout")
    if args.role == "es":
        accept_es(args.es_ip, args.ud_ip, args.port, args.rendezvous)
        return
    if (not args.out or not 0 < args.repair <= 1 or not 1 <= args.smoke_mib <= 128
            or min(args.samples, args.repeats, args.new_tokens) < 1):
        parser.error("Specify --out and valid repair, sample, replay and token settings")

    with Path(args.out).open("x", encoding="utf-8") as file:
        with connect_ud(args.es_ip, args.ud_ip, args.port, args.rendezvous,
                        args.socket_timeout) as link:
            if args.role == "smoke":
                result = roundtrip_smoke(link, args.smoke_mib << 20)
                write_row(file, result)
                print(json.dumps(result), flush=True)
            else:
                run(args, link, file)


if __name__ == "__main__":
    main()
