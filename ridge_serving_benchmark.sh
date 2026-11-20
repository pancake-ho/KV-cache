#!/usr/bin/env bash
#SBATCH --job-name=kv_ridge_serving
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the KV-cache repository root}"
test -f proposed/validation/semcache_smoke.py
if [[ $# -gt 1 ]]; then
  echo 'Usage: sbatch ridge_serving_benchmark.sh [ridge_checkpoint_directory]' >&2
  exit 2
fi
conda_base=$(conda info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate lab
set -u

out="$PWD/results/semcache/ridge_serving_${SLURM_JOB_ID:?submit through Slurm}"
mkdir -p "$(dirname "$out")"
mkdir "$out"
status=FAILED
trap 'printf "status=%s\nexit_code=%s\njob_id=%s\n" "$status" "$?" "$SLURM_JOB_ID" > "$out/status.txt"' EXIT
echo "Results: $out"

unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
export HF_HUB_CACHE="/data/$USER/models/semcache/hf_hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SEMCACHE_ROOT="${SEMCACHE_ROOT:-/data/$USER/repos/lab/GP_semcache-r5-d38edf8}"
export SEMCACHE_TINYBERT_DIR="${SEMCACHE_TINYBERT_DIR:-/data/$USER/models/semcache/tinybert-34707a3}"

python -u - "$out" "${1:-}" <<'PY'
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from proposed.validation import config, semcache_smoke as smoke
from proposed.validation.serving_metrics import _compare_modes, _summarize_mode, measure_replays
from proposed.validation.workload import controlled_requests, multi_user_request_cases, multi_user_requests


def find_ridge_checkpoint(override, pair):
    def compatible(path):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(meta, dict):
            return False
        if (meta.get("source_model"), meta.get("target_model")) != pair:
            return False
        if (meta.get("ridge") != 0.01 or meta.get("target_layers") != 36
                or meta.get("num_observations") not in (None, 128000)):
            return False
        layers = meta.get("selected_layers")
        if (not isinstance(layers, list) or len(layers) != 36
                or any(not isinstance(row, list) or len(row) != 12 for row in layers)):
            return False
        return all((path.parent / "layers" / f"layer_{i:03d}.safetensors").is_file()
                   for i in range(36))

    if override:
        path = Path(override).expanduser().resolve()
        if not compatible(path / "config.json"):
            raise RuntimeError(f"Incomplete or incompatible Top-12 Ridge checkpoint: {path}")
        return path
    roots = (Path("results"), Path("baseline/crosskv/work"))
    candidates = sorted({path.parent.resolve() for root in roots
                         for path in root.rglob("config.json") if compatible(path)})
    paired = [path for path in candidates if "217697" in str(path)]
    candidates = paired or candidates
    if len(candidates) == 1:
        return candidates[0]
    listing = "\n".join(str(path) for path in candidates) or "(none)"
    raise RuntimeError(
        "Expected one complete Qwen3-1.7B→4B Top-12 Ridge checkpoint. "
        "Pass the checkpoint directory as the first sbatch script argument.\n" + listing
    )


out = Path(sys.argv[1])
root = smoke.ROOT
r5_root = Path(os.environ["SEMCACHE_ROOT"]).resolve()
tinybert = Path(os.environ["SEMCACHE_TINYBERT_DIR"]).resolve()
if not (r5_root / "src/semcache/edgelora/r5_semantic_index.py").is_file():
    raise FileNotFoundError(f"R5 semantic index missing under {r5_root}")
if not (tinybert / "config.json").is_file():
    raise FileNotFoundError(f"Local TinyBERT config missing under {tinybert}")
sys.path[:0] = [str(r5_root / "src"), str(root / "baseline/crosskv/src")]

import torch
import transformers
import triton
from semcache.semantic.encoder import TinyBERTSemanticEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.hub import cached_file
from xmodel_kv.artifact import LinearKVMapper

from proposed.src.cache.kv_bitpack import CODEC_ID, KVBitpackCodec
from proposed.src.models.qwen3 import QwenRuntime

if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(including_emulation=False):
    raise RuntimeError("Allocate one GPU with native CUDA BF16 support.")
device = "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = False

spec = json.loads(smoke.MODEL_CONFIG.read_text(encoding="utf-8"))
pair = (spec["source_model"], spec["target_model"])
ridge_dir = find_ridge_checkpoint(sys.argv[2], pair)
ridge = smoke.load_mapper("ridge", LinearKVMapper, ridge_dir, pair)
print(f"Ridge checkpoint: {ridge_dir}")

prefix, suffix, new_tokens, cache_mib, repeats = 512, 24, 8, 512, 3
adapter_specs = config.MULTI_USER_ADAPTERS
revisions = {
    **config.MULTI_USER_EXTRA_MODELS,
    pair[0]: spec["source_revision"],
    pair[1]: spec["target_revision"],
}
required_models = set(pair) | {model for model, _, _ in adapter_specs.values()}
revisions = {name: revisions[name] for name in sorted(required_models)}
tokenizers = {}
for name, rev in revisions.items():
    config_file = cached_file(name, "config.json", revision=rev, local_files_only=True)
    tokenizers[name] = AutoTokenizer.from_pretrained(Path(config_file).parent, local_files_only=True)

requests, expected = multi_user_requests(
    (), tokenizers, prefix, suffix, extra_models=(), routes=(), adapters=adapter_specs,
    cross_model_probes=config.MULTI_USER_CROSS_MODEL_PROBES,
)
if len(requests) != 16:
    raise RuntimeError(f"Expected the 16-request LoRA workload from job 217835, got {len(requests)}.")
prior = root / "results/semcache/serving_217835/summary.json"
if prior.is_file():
    rows = json.loads(prior.read_text(encoding="utf-8"))["modes"]["semcache"]
    current = [(r.request_id, r.model_id, r.adapter_id, len(r.token_ids)) for r in requests]
    recorded = [(r["request_id"], r["model_id"], r["adapter_id"], r["prompt_tokens"]) for r in rows]
    if current != recorded:
        raise RuntimeError("The configured LoRA workload differs from job 217835.")

source, target, _ = controlled_requests(*pair, tokenizers, prefix, suffix)
source = replace(source, request_id="ridge-base-source", user_id="ridge-base-user-a",
                 share_group="ridge-base")
target = replace(target, request_id="ridge-base-target", user_id="ridge-base-user-b",
                 share_group="ridge-base")
requests += (source, target)
expected[source.request_id] = ("miss_recompute", None, None)
expected[target.request_id] = ("miss_recompute", None, "unsupported_model_pair")
expected_ridge = {**expected, target.request_id: ("cross_model_kv_hit", source.request_id, None)}
request_cases = multi_user_request_cases(requests)
request_cases[target.request_id] = "base_cross_model"

models = {
    name: AutoModelForCausalLM.from_pretrained(
        name, revision=rev, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    for name, rev in revisions.items()
}
runtime = QwenRuntime(models, tokenizers, device)
adapters = [
    runtime.adapters.load(model_id, adapter_id, path)
    for adapter_id, (model_id, _, path) in adapter_specs.items()
]
runtime.warmup()
encoder = TinyBERTSemanticEncoder(
    model_id=str(tinybert), revision=smoke.TINYBERT_REVISION,
    device="cpu", dtype="float32", local_files_only=True,
)
encoder.encode(["Warmup."])

samples = {}
summary = {}
for name, mapper, codec, checks in (
    ("semcache", None, None, expected),
    ("semcache_ridge", ridge, None, expected_ridge),
    ("semcache_ridge_5bit", ridge, KVBitpackCodec(), expected_ridge),
):
    print(f"Measuring {name}: {len(requests)} requests × {repeats} repeats")
    measured = measure_replays(
        lambda: smoke.replay(runtime, requests, new_tokens, encoder=encoder, mapper=mapper,
                             capacity_bytes=cache_mib << 20, codec=codec),
        device=device, repeats=repeats,
    )
    for rows, _ in measured:
        smoke.validate_multi_user(rows, prefix, checks)
    samples[name] = measured
    summary[name] = _summarize_mode(measured)
    summary[name]["path_counts"] = dict(Counter(
        row["path"] for rows, _ in measured for row in rows
    ))

comparisons = {}
for baseline, candidate in (
    ("semcache", "semcache_ridge"),
    ("semcache", "semcache_ridge_5bit"),
    ("semcache_ridge", "semcache_ridge_5bit"),
):
    key = f"{candidate}_vs_{baseline}"
    comparisons[key] = _compare_modes(
        samples[baseline], samples[candidate], summary[baseline], summary[candidate], request_cases,
    )

cross_model = {
    name: [next(row for row in rows if row["request_id"] == target.request_id)
           for rows, _ in measured]
    for name, measured in samples.items()
}
report = {
    "scope": "Sequential warm serving replays. One isolated, non-LoRA cross-model pair.",
    "job_id": os.environ["SLURM_JOB_ID"],
    "provenance": {"project": smoke.git_state(root), "semcache": smoke.git_state(r5_root)},
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__, "transformers": transformers.__version__, "triton": triton.__version__,
    "model_revisions": revisions, "adapters": adapters,
    "semantic_encoder": encoder.metadata,
    "ridge_dir": str(ridge_dir), "ridge_config": asdict(ridge.config), "codec": CODEC_ID,
    "prefix_tokens": prefix, "suffix_tokens": suffix, "new_tokens": new_tokens,
    "cache_mib": cache_mib, "benchmark_repeats": repeats,
    "request_count": len(requests), "request_cases": request_cases,
    "performance": summary, "comparisons": comparisons, "cross_model": cross_model,
    "first_replay": {name: measured[0][0] for name, measured in samples.items()},
}
with (out / "summary.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")

for name, rows in cross_model.items():
    mean_ttft = sum(row["ttft_ms"] for row in rows) / len(rows)
    path = rows[0]["path"]
    prefill = rows[0]["forwarded_prefill_tokens"]
    print(f"{name}: {path}, target prefill={prefill}, mean target TTFT={mean_ttft:.2f} ms")
print(f"Ridge serving benchmark complete: {out / 'summary.json'}")
PY
status=COMPLETE
