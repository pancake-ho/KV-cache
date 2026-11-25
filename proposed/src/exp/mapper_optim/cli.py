"""Fit and evaluate cross-model KV mappers in an isolated run directory."""
from __future__ import annotations

import argparse
import json
import math
import platform
import tempfile
from pathlib import Path
from time import perf_counter

import torch
import transformers

from .calibration import collect, fit_layers
from .evaluate import benchmark, continuation
from .inputs import ModelIdentity, check_split, read_examples
from .transfer import AffineMapper


def _positive(raw: str) -> int:
    number = int(raw)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{method}: one-GPU Qwen3 mapper experiments")
    sub = parser.add_subparsers(dest="stage", required=True)
    fit = sub.add_parser("fit", help="collect paired traces and fit mapper layers")
    fit.add_argument("--source-model", required=True)
    fit.add_argument("--target-model", required=True)
    fit.add_argument("--source-revision")
    fit.add_argument("--target-revision")
    fit.add_argument("--calibration-jsonl", type=Path, required=True)
    fit.add_argument("--sequence-length", type=_positive, required=True)
    fit.add_argument("--sequences", type=_positive, required=True)
    fit.add_argument("--stride", type=_positive, default=4)
    fit.add_argument("--top-k", type=_positive, required=True)
    fit.add_argument("--ridge", type=float, default=0.01,
                     help="sum-loss regularization, as written in the paper equations")
    fit.add_argument("--fit-dtype", choices=("float32", "float64"), default="float64")
    fit.add_argument("--probe-rows", type=_positive, default=2048)
    fit.add_argument("--chunk-rows", type=_positive, default=256)
    if method == "rarecache":
        fit.add_argument("--rank", type=_positive, default=128)
    else:
        fit.add_argument("--variant", choices=("full-head", "head-local", "attn-repair"), required=True)
        fit.add_argument("--boundaries", type=_positive, default=32)
        fit.add_argument("--first-boundary", type=_positive, default=12)

    evaluate = sub.add_parser("eval", help="evaluate one saved mapper against native prefill")
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--eval-jsonl", type=Path, required=True)
    evaluate.add_argument("--prefix-length", type=_positive, required=True)
    evaluate.add_argument("--continuation-tokens", type=_positive, required=True)
    evaluate.add_argument("--examples", type=_positive, required=True)
    evaluate.add_argument("--warmup", type=int, default=2)
    evaluate.add_argument("--repeats", type=_positive, default=3)
    evaluate.add_argument("--decode-tokens", type=_positive, default=4)
    if method == "rarecache":
        evaluate.add_argument("--recompute-ratios", type=float, nargs="+", required=True)
    for command in (fit, evaluate):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
        command.add_argument("--seed", type=int, default=2026)
    return parser


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fit(args, method, device, dtype, basis, attention) -> None:
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    src = ModelIdentity.inspect(args.source_model, args.source_revision)
    tgt = ModelIdentity.inspect(args.target_model, args.target_revision)
    if method == "cachebridge" and src.heads != tgt.heads:
        raise ValueError("CacheBridge v1 uses identity assignment for equal KV-head counts")
    if args.top_k > src.layers:
        raise ValueError("Layer budget exceeds source layer count")
    width = src.width * (src.heads if method == "rarecache" else 1)
    samples = args.sequences * math.ceil(args.sequence_length / args.stride)
    if min(samples, args.probe_rows) <= width:
        raise ValueError(f"Need more than {width} sampled rows for OLS source-layer selection")
    if method == "rarecache" and args.rank >= tgt.heads * tgt.width:
        raise ValueError("Requested probe rank must be less than target KV width")
    examples = read_examples(src, tgt, args.calibration_jsonl,
                             length=args.sequence_length, limit=args.sequences)
    variant = getattr(args, "variant", "rank-disagreement")
    scores = None
    weight_shrinkage = None
    if variant == "attn-repair":
        query_capture, build_boundaries, weight_shrinkage = attention
        steps = build_boundaries(args.sequence_length, args.boundaries, args.first_boundary)
        scores = lambda model: query_capture(model, torch.tensor(steps, device=device))
    config = {
        "format": 1, "method": method, "variant": variant, "source": src.metadata(),
        "target": tgt.metadata(), "dtype": args.dtype, "ridge": args.ridge,
        "seed": args.seed, "sequence_length": args.sequence_length, "stride": args.stride,
        "top_k": args.top_k, "probe_rows": args.probe_rows, "chunk_rows": args.chunk_rows,
        "fit_dtype": args.fit_dtype, "rank": getattr(args, "rank", None),
        "boundaries": steps.tolist() if scores else [],
    }
    artifact = args.output / "artifact"
    artifact.mkdir()
    _write(artifact / "fit_examples.json", examples)
    tic = perf_counter()
    with tempfile.TemporaryDirectory(prefix="paired_traces_", dir=args.output) as root:
        source = collect(src, examples, Path(root) / "source", role="source",
                         stride=args.stride, device=device, dtype=dtype)
        target = collect(tgt, examples, Path(root) / "target", role="target",
                         stride=args.stride, device=device, dtype=dtype, capture_factory=scores)
        trace_seconds = perf_counter() - tic
        stats = fit_layers(source, target, artifact, settings=config, device=device,
                           fit_dtype=getattr(torch, args.fit_dtype), basis_fn=basis,
                           weight_shrinkage=weight_shrinkage)
    _write(args.output / "fit_summary.json",
           {"trace_seconds": trace_seconds, "selection_and_fit_seconds": perf_counter() - tic - trace_seconds,
            "safetensors_bytes": sum(f.stat().st_size for f in artifact.glob("*.safetensors")),
            "weight_shrinkage": stats})


def _eval(args, method, device, dtype, rank_score, replay) -> None:
    ratios = sorted(set([0.0, *getattr(args, "recompute_ratios", [])]))
    if args.warmup < 0 or args.prefix_length < 2 or any(not 0 <= r <= 1 for r in ratios):
        raise ValueError("Invalid warmup, prefix length, or recompute ratio")
    mapper = AffineMapper(args.artifact, method, device, rank_score)
    if args.dtype != mapper.spec["dtype"]:
        raise ValueError(f"Use the mapper calibration dtype {mapper.spec['dtype']}")
    for identity in (mapper.source, mapper.target):
        if ModelIdentity.inspect(identity.name, identity.revision) != identity:
            raise ValueError(f"Model revision or geometry changed: {identity.name}")
    if args.prefix_length + args.decode_tokens - 1 > mapper.target.context:
        raise ValueError("Generated sequence would exceed target model context")
    examples = read_examples(mapper.source, mapper.target, args.eval_jsonl,
                             length=args.prefix_length + args.continuation_tokens, limit=args.examples)
    fit_examples = json.loads((args.artifact / "fit_examples.json").read_text())
    check_split(examples, fit_examples, mapper.spec["sequence_length"])
    source = mapper.source.load(device, dtype)
    target = mapper.target.load(device, dtype)
    report = {"method": method, "variant": mapper.spec["variant"], "artifact": str(args.artifact),
              "mapper_tensor_bytes": mapper.bytes, "examples": []}
    for index, example in enumerate(examples, start=1):
        ids = torch.tensor([example["ids"]], dtype=torch.long, device=device)
        prompt = ids[:, :args.prefix_length]
        timing = dict(replay=replay, warmup=args.warmup, repeats=args.repeats,
                      decode_tokens=args.decode_tokens)
        native = benchmark(mapper, source, target, prompt, path="native", budget=0., **timing)
        row = {"line": example["line"], "token_sha256": example["sha256"], "native": native, "mapped": []}
        for ratio in ratios:
            quality = continuation(mapper, source, target, ids, args.prefix_length, ratio, replay)
            perf = benchmark(mapper, source, target, prompt, path="mapped", budget=ratio, **timing)
            row["mapped"].append({"recompute_ratio": ratio, "quality": quality, "benchmark": perf})
        report["examples"].append(row)
        print(f"Evaluated {index}/{len(examples)}", flush=True)
    _write(args.output / "evaluation.json", report)


def run(method, *, basis=None, rank_score=None, replay=None, attention=None, device_name="cuda:0"):
    args = _parser(method).parse_args()
    if transformers.__version__ != "4.57.3":
        raise RuntimeError(f"Requires Transformers 4.57.3 from the CrossKV environment, got {transformers.__version__}")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. Use the Seraph Slurm launcher")
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Assigned GPU does not support BF16")
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    _write(args.output / "run.json",
           {"options": {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()},
            "method": method, "device": str(device), "batch_size": 1,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda})
    with torch.inference_mode():
        if args.stage == "fit":
            _fit(args, method, device, dtype, basis, attention)
        else:
            _eval(args, method, device, dtype, rank_score, replay)
