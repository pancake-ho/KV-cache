"""Replay controlled Qwen3 and optional LoRA requests through the serving system."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from proposed.src.cache.prefix import PrefixCache
from proposed.src.kv_transfer.base_model import BaseModelKVTransfer
from proposed.src.serving.engine import ServingEngine

from . import config
from .workload import add_lora_requests, controlled_requests, multi_user_requests

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "baseline/crosskv/configs/figure2_qwen3_1p7b_4b.json"
TINYBERT_REVISION = "34707a33cd59a94ecde241ac209bf35103691b43"


def replay(runtime, requests, budget, *, encoder=None, mapper=None, capacity_bytes=256 << 20, codec=None, additional_mappers=()):
    cache = None
    if encoder is not None:
        from semcache.edgelora.r5_semantic_index import SemanticIndex

        cache = PrefixCache(SemanticIndex(encoder, clusters=1), capacity_bytes, codec=codec)
    transfer = BaseModelKVTransfer(mapper, *additional_mappers) if mapper is not None else None
    engine = ServingEngine(runtime, cache=cache, transfer=transfer)
    return [engine.serve(req, budget) for req in requests]


def validate_reuse(rows, prefix_tokens):
    source, hit, miss = rows[:3]
    if (source["path"] != "miss_recompute" or hit["path"] != "cross_model_kv_hit"
            or hit["reused_from"] != source["request_id"]
            or hit["reused_prefix_tokens"] != prefix_tokens
            or hit["forwarded_prefill_tokens"] != hit["prompt_tokens"] - prefix_tokens
            or miss["path"] != "miss_recompute" or miss["reused_prefix_tokens"]
            or not miss["rejections"].get("exact_prefix_mismatch")):
        raise RuntimeError(f"controlled reuse check failed: {[(r['request_id'], r['path']) for r in rows]}")
    if len(rows) > 3:
        source, same, other = rows[3:]
        if (source["path"] != "miss_recompute"
                or same["path"] != "native_kv_hit"
                or same["reused_from"] != source["request_id"]
                or same["reused_prefix_tokens"] != prefix_tokens
                or other["path"] != "miss_recompute"
                or other["reused_prefix_tokens"]
                or not other["rejections"].get("adapter_mismatch")):
            raise RuntimeError("controlled LoRA reuse check failed")


def validate_multi_user(rows, prefix_tokens, expected):
    if len(rows) != len(expected) or {row["request_id"] for row in rows} != set(expected):
        raise RuntimeError("multi-user replay returned unexpected request IDs")
    for row in rows:
        path, source_id, rejection = expected[row["request_id"]]
        prefix = prefix_tokens if source_id else 0
        if (row["path"] != path or row["reused_from"] != source_id
                or row["reused_prefix_tokens"] != prefix
                or row["forwarded_prefill_tokens"] != row["prompt_tokens"] - prefix
                or (rejection and not row["rejections"].get(rejection))):
            raise RuntimeError(f"multi-user reuse check failed for {row['request_id']}: {row['path']}")


def load_mapper(name, mapper_class, path, pair):
    mapper = mapper_class(path)
    if (mapper.config.source_model, mapper.config.target_model) != pair:
        raise ValueError(f"{name} checkpoint does not match the serving model pair {pair}: {path}")
    if name == "ridge":
        if not all(mapper.layer_path(i).is_file() for i in range(mapper.config.target_layers)):
            raise FileNotFoundError(f"incomplete ridge checkpoints in {path}")
    else:
        mapper.assert_complete()
    return mapper


def git_state(root):
    top = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(top).resolve() != root.resolve():
        raise ValueError(f"expected an independent Git checkout at {root}")
    return {
        "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "status": subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True).splitlines(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semcache-root", type=Path, default=ROOT / "baseline/GP_semcache")
    parser.add_argument("--mlp-dir", type=Path)
    parser.add_argument("--ridge-dir", type=Path)
    parser.add_argument("--adapter", action="append", nargs=2, metavar=("ID", "DIRECTORY"),
                        default=[], help="optional existing 4B LoRA adapter; pass twice for the LoRA smoke")
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--suffix-tokens", type=int, default=24)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--cache-mib", type=int, default=512)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--multi-user", action="store_true", help="load the expanded profile from config.py")
    parser.add_argument("--compare-kv-bitpack", action="store_true",
                        help="also replay each mapper with a 5-bit prefix cache")
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.multi_user:
        if not config.MULTI_USER_ADAPTERS:
            parser.error("configure at least one entry in MULTI_USER_ADAPTERS")
        if args.mlp_dir is not None or args.ridge_dir is not None:
            parser.error("mapper arguments apply to the base-model smoke")
    elif args.mlp_dir is None and args.ridge_dir is None:
        parser.error("provide at least one existing mapper directory")
    if (not 4 <= args.prefix_tokens <= 4096 or not 1 <= args.suffix_tokens <= 4096
            or not 1 <= args.new_tokens <= 64 or args.cache_mib < 1):
        parser.error("require 4–4096 prefix tokens, 1–4096 suffix tokens, 1–64 new tokens and positive cache capacity")
    if args.multi_user and args.adapter:
        parser.error("--multi-user uses MULTI_USER_ADAPTERS from config.py")
    if args.adapter and (len(args.adapter) != 2 or len({name for name, _ in args.adapter}) != 2):
        parser.error("provide either zero or two distinct 4B LoRA adapters")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    src = (args.semcache_root / "src").resolve()
    if not (src / "semcache/edgelora/r5_semantic_index.py").is_file():
        parser.error(f"R5 semantic index missing under {src}")
    provenance = {"project": git_state(ROOT), "semcache": git_state(args.semcache_root)}
    sys.path[:0] = [str(src), str(ROOT / "baseline/crosskv/src")]

    import torch
    import transformers
    from packaging.version import Version
    from semcache.semantic.encoder import TinyBERTSemanticEncoder
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xmodel_kv.artifact import LinearKVMapper

    from proposed.src.exp.mapper.transfer import NeuralKVMapper
    from proposed.src.models.qwen3 import QwenRuntime

    if Version(torch.__version__) < Version("2.6"):
        raise RuntimeError("PyTorch >= 2.6 is required to load the pinned TinyBERT .bin checkpoint")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("allocate exactly one CUDA GPU for the serving pilot")

    codec = None
    codec_info = None
    if args.compare_kv_bitpack:
        import triton

        from proposed.src.cache.kv_bitpack import CODEC_ID, KVBitpackCodec

        codec = KVBitpackCodec()
        codec_info = {"id": CODEC_ID, "triton": triton.__version__}

    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    source_id, target_id = cfg["source_model"], cfg["target_model"]
    primary_pair = (source_id, target_id)
    revisions = {source_id: cfg["source_revision"], target_id: cfg["target_revision"]}
    extra_models = config.MULTI_USER_EXTRA_MODELS if args.multi_user else {}
    adapter_specs = (
        config.MULTI_USER_ADAPTERS if args.multi_user
        else {name: (target_id, None, path) for name, path in args.adapter}
    )
    if revisions.keys() & extra_models.keys():
        parser.error("extra models must differ from the primary pair")
    revisions.update(extra_models)

    for name, (model_id, dataset, path) in adapter_specs.items():
        if model_id not in revisions or (args.multi_user and dataset not in config.ADAPTER_DATASETS):
            parser.error(f"unknown model or dataset for LoRA {name}")
        if not all((Path(path) / file).is_file() for file in ("adapter_config.json", "adapter_model.safetensors")):
            parser.error(f"missing fitted LoRA checkpoint for {name}: {path}")

    if args.multi_user:
        adapter_models = {model_id for model_id, _, _ in adapter_specs.values()}
        revisions = {name: rev for name, rev in revisions.items() if name in adapter_models}

    local = not args.allow_download
    encoder = TinyBERTSemanticEncoder(
        revision=TINYBERT_REVISION, device="cpu", dtype="float32", local_files_only=local,
    )
    encoder.encode(["Warmup."])
    tokenizers = {
        name: AutoTokenizer.from_pretrained(name, revision=rev, local_files_only=local)
        for name, rev in revisions.items()
    }
    models = {
        name: AutoModelForCausalLM.from_pretrained(
            name, revision=rev, local_files_only=local, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()
        for name, rev in revisions.items()
    }
    
    runtime = QwenRuntime(models, tokenizers, device)
    adapters = []
    for name, (model_id, dataset, path) in adapter_specs.items():
        row = runtime.adapters.load(model_id, name, path)
        if dataset is not None:
            repo_id, subset, train_split, heldout_split = config.ADAPTER_DATASETS[dataset]
            row["training_dataset"] = {
                "name": dataset, "repo_id": repo_id, "subset": subset,
                "train_split": train_split, "heldout_split": heldout_split,
            }
        adapters.append(row)

    expected = {}
    if args.multi_user:
        requests, expected = multi_user_requests(
            (), tokenizers, args.prefix_tokens, args.suffix_tokens,
            extra_models=(), routes=(), adapters=adapter_specs,
        )
    else:
        requests = controlled_requests(
            source_id, target_id, tokenizers, args.prefix_tokens, args.suffix_tokens,
        )
        if adapters:
            requests = add_lora_requests(requests, [row["adapter_id"] for row in adapters])
    runtime.warmup()
    modes = {"full": replay(runtime, requests, args.new_tokens)}

    mapper_paths = {}
    codec_comparison = {}
    replay_modes = (("semcache", None, None),) if args.multi_user else (
        ("ridge", args.ridge_dir, LinearKVMapper),
        ("mlp", args.mlp_dir, NeuralKVMapper),
    )
    for name, path, mapper_class in replay_modes:
        mapper = None
        if path is not None:
            mapper = load_mapper(name, mapper_class, path, primary_pair)
            mapper_paths[name] = str(path.resolve())
        elif not args.multi_user:
            continue
        replay_options = {
            "encoder": encoder,
            "mapper": mapper,
            "capacity_bytes": args.cache_mib << 20,
        }
        rows = replay(runtime, requests, args.new_tokens, **replay_options)
        modes[name] = rows
        if codec is not None:
            # Warm the codec with the actual Qwen cache shapes before the measured replay.
            replay(runtime, requests, args.new_tokens, codec=codec, **replay_options)
            packed = replay(runtime, requests, args.new_tokens, codec=codec, **replay_options)
            modes[f"{name}_bitpack"] = packed
            codec_comparison[name] = {
                "source_raw_bytes": rows[0]["resident_kv_bytes"],
                "source_bitpack_bytes": packed[0]["resident_kv_bytes"],
                "source_encode_ms": packed[0]["cache_encode_ms"],
                "target_decode_ms": packed[1]["cache_decode_ms"],
                "target_ttft_delta_ms": packed[1]["ttft_ms"] - rows[1]["ttft_ms"],
                "generated_ids_agree_by_request": {
                    raw["request_id"]: raw["generated_ids"] == compressed["generated_ids"]
                    for raw, compressed in zip(rows, packed, strict=True)
                },
            }
        del mapper, replay_options

    full_hit = modes["full"][1]
    comparison = {
        name: {
            "target_ttft_delta_ms": rows[1]["ttft_ms"] - full_hit["ttft_ms"],
            "target_first_token_agrees": rows[1]["generated_ids"][0] == full_hit["generated_ids"][0],
            "target_all_tokens_agree": rows[1]["generated_ids"] == full_hit["generated_ids"],
        }
        for name, rows in modes.items() if name != "full"
    }
    if adapters and not args.multi_user:
        for name, metrics in comparison.items():
            metrics["same_adapter_tokens_agree"] = modes[name][4]["generated_ids"] == modes["full"][4]["generated_ids"]
    report = {
        "scope": "controlled sequential smoke; fixed generation budget; one trace per mode",
        "provenance": provenance, "model_revisions": revisions, "mapper_paths": mapper_paths,
        "profile": "multi_user" if args.multi_user else "smoke",
        "request_count": len(requests), "user_count": len({request.user_id for request in requests}),
        "extra_ridge_routes": [],
        "comparison_request_id": full_hit["request_id"],
        "semantic_encoder": encoder.metadata, "adapters": adapters,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16",
        "prefix_tokens": args.prefix_tokens, "suffix_tokens": args.suffix_tokens,
        "new_tokens": args.new_tokens, "cache_mib": args.cache_mib,
        "modes": modes, "comparison": comparison,
        "kv_codec": codec_info, "codec_comparison": codec_comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for name, rows in modes.items():
        if name == "full":
            continue
        if args.multi_user:
            validate_multi_user(rows, args.prefix_tokens, expected)
        else:
            validate_reuse(rows, args.prefix_tokens)
    print(f"Controlled replay complete: {args.output}")


if __name__ == "__main__":
    main()
