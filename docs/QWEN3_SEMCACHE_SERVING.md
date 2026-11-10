# Qwen3 KV reuse serving system

| Location | Responsibility |
|---|---|
| `proposed/src/serving/` | Request contract and sequential serving engine |
| `proposed/src/cache/` | Prefix KV ownership, R5 lookup, LRU eviction, and optional Triton bitpacking |
| `proposed/src/models/` | Qwen3 execution and PEFT LoRA loading and selection |
| `proposed/src/kv_transfer/` | Policy and interface for the existing base-model Ridge and MLP mappers |
| `proposed/validation/` | Controlled request trace and metric reporting |
| `proposed/validation/config.py` | Mapper paths, LoRA checkpoints, token lengths, and output location |
| `run/seraph/qwen3_serving.sh` | Slurm job entry point, environment activation, and node-local mapper staging |

`ServingEngine.serve` accepts a tokenized request with an explicit model and an
optional adapter. The caller supplies a trusted sharing group. The validation
trace uses one public sharing group and controlled prompts. Model and adapter
selection, prefill, generation, and cache admission run under the runtime lock.
Queue scheduling and tenant authentication are outside the scope of this trace.

The existing `baseline/GP_semcache` R5 checkout supplies TinyBERT and
`SemanticIndex`. The source model's native KV cache is frozen before decode and
published to the index after successful generation. Reuse requires an exact
zero-based token prefix and a matching sharing group. Same-model reuse also
requires the same adapter identity, including the SHA-256 fingerprint of its
configuration and weights.

The Ridge and MLP checkpoints apply to their configured base-model transfer
direction with both adapters disabled. An incompatible adapter or model pair
takes the recomputation path. Mapped target caches are request-local. The LoRA
trace uses two PEFT checkpoints with a fixed Qwen3-4B base revision.

## Environment

Use the existing `baseline/GP_semcache` R5 checkout prepared by its setup ZIP.
The two Qwen snapshots and the pinned TinyBERT snapshot load from the Hugging
Face cache. To let a job fetch missing snapshots, set `ALLOW_DOWNLOAD = True`
in `proposed/validation/config.py`.

If `semcache-qwen` is unavailable, create it once. PyTorch 2.6 or newer is
required to load the pinned TinyBERT PyTorch checkpoint with Transformers 4.57.1.

```bash
conda create -n semcache-qwen python=3.11 pip -y
conda activate semcache-qwen
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'transformers==4.57.1' safetensors 'peft==0.18.1'
python -m pip check
```

## Moana

Edit `proposed/validation/config.py` before submitting.

| Setting | Use |
|---|---|
| `MLP_DIR` | Existing MLP checkpoint directory. Enabled by default. |
| `RIDGE_DIR` | Optional Ridge checkpoint directory from the paired run. |
| `LORA_ADAPTERS` | Leave empty for the base-model trace or enable both `medqa` and `fpb` for the LoRA trace. |
| `PREFIX_TOKENS`, `SUFFIX_TOKENS`, `NEW_TOKENS` | Shared prefix length, request suffix length, and generation budget. |
| `COMPARE_KV_BITPACK` | Set to `True` to replay both raw and compressed cache modes with each enabled mapper. Defaults to `False`. |
| `OUTPUT_ROOT` | Parent directory for per-job results. Defaults to `results/semcache/`. |

Configure at least one mapper. For the LoRA trace, use Qwen3-4B adapter
checkpoints trained with the pinned base revision if their metadata omits the
revision.

The batch script activates `semcache-qwen` on the compute node. Slurm resource
defaults are defined in the script's `#SBATCH` directives.

```bash
cd /data/surt321/repos/lab/kv_cache
mkdir -p logs
bash -n run/seraph/qwen3_serving.sh
sbatch run/seraph/qwen3_serving.sh
```

Once `logs/` exists, submit subsequent jobs from the repository root with
`sbatch run/seraph/qwen3_serving.sh`.

Slurm writes stdout and stderr to `logs/qwen3_serving-<job_id>.out` and
`logs/qwen3_serving-<job_id>.err`. With the default output root, the job writes
`job_id.txt`, `status.txt`, and `summary.json` under
`results/semcache/serving_<job_id>/`.

The three base requests exercise full prefill, cross-model reuse, and an
exact-prefix miss. With default lengths, a cross-model hit reuses 512 prefix
tokens and prefills the remaining 24 tokens on the target. The optional LoRA
requests exercise adapter A cache admission, adapter A exact reuse, and adapter
B recomputation for the same prefix.

Model loading, checkpoint staging, and warmup are outside request timing.
TTFT includes routing, semantic search, mapper application, and target prefill
when applicable. The report captures routing and timing for one controlled
trace. A break-even estimate or a generation-quality comparison requires a
separate benchmark.

## Optional 5-bit prefix KV compression

The uploaded QKV Bitpack V2.1 implements 5-bit packing for OPT-350m LoRA Q/K/V
projection deltas in a UD-to-ES setting. The serving codec in
`proposed/src/cache/kv_bitpack.py` adapts its packing and per-token scaling to
Qwen3's resident K/V tensors after prefill. Each layer uses the native
`[1, heads, tokens, head_dim]` shape, with one FP32 scale per token and per K/V
tensor. The codec restores the original shape and dtype before same-model reuse
or the existing Ridge/MLP mapping. The codec runs locally on the serving GPU.

Use the same `sbatch run/seraph/qwen3_serving.sh` command with
`COMPARE_KV_BITPACK = True`. This mode requires NVIDIA CUDA and Triton 3.2.0
with the documented PyTorch 2.6.0 environment. The first packed replay warms up
the kernels. The report retains the full-prefill and raw cache rows and adds a
packed replay per mapper. Inspect `codec_comparison` for stored source tensor
bytes, codec encode/decode latency, target TTFT delta, and generated-token
agreement with the same mapper's raw replay.

`resident_kv_bytes` counts cached tensor payload and scales. GPU allocator
overhead and temporary dense K/V require a separate memory measurement.
`cache_encode_ms` is the cache-admission snapshot cost after the first token
is computed. `cache_decode_ms` is the restoration cost on a cache hit and is
included in TTFT. Token agreement and timings cover the controlled trace.

## Local tests

From the repository root, run:

```bash
PYTHONPATH="$PWD/baseline/GP_semcache/src:$PWD/baseline/crosskv/src" \
  python -m unittest discover -s proposed/tests -p 'test_semcache_serving.py' -v
```

The tests use tiny Qwen3 models on CPU, the real R5 semantic index with a
controlled encoder, an identity mapper fixture, and two PEFT safetensors
checkpoints. When CUDA is available, they also compare the Triton codec
against the reference quantizer. Use the Slurm job to verify fitted mappers
and trained LoRA checkpoints on the GPU.
