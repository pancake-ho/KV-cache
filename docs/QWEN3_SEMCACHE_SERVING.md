# Qwen3 KV reuse serving system

| Location | Responsibility |
|---|---|
| `proposed/src/serving/` | Request contract and the sequential serving engine |
| `proposed/src/cache/` | Prefix K/V ownership, R5 candidate lookup, eligibility and LRU |
| `proposed/src/models/` | Qwen3 execution and PEFT LoRA loading and selection |
| `proposed/src/kv_transfer/` | Policy and interface for the existing base-model Ridge and neural mappers |
| `proposed/validation/` | Controlled trace, composition of the dependencies and metric reporting |
| `run/seraph/semcache_smoke.sh` | Slurm job entry point and node-local mapper staging |

`ServingEngine.serve` accepts a tokenized request with an explicit model and an
optional adapter. The caller supplies a trusted sharing group. The smoke uses a
single public group with controlled prompts. Model and adapter selection,
prefill, generation and admission for a request run under the runtime lock.
Queue scheduling and tenant authentication are outside this smoke.

The existing `baseline/GP_semcache` R5 checkout supplies TinyBERT and
`SemanticIndex`. The source model's native Qwen K/V is frozen before decode and
published to the index after successful generation. A candidate must have an
exact zero-based token prefix and share group. Same-model reuse also requires
the same adapter ID and SHA256 of its config and weights. The base-model Ridge
and MLP checkpoints apply only to their configured direction with both adapters
disabled. An incompatible adapter or model pair takes the recompute path.
Mapped target caches are request-local. The LoRA integration uses the existing
PEFT checkpoint load and selection pattern, with a fixed Qwen3-4B revision for
the smoke. Independently trained adapters have separate cache identities.

## Environment

The pilot imports the R5 checkout directly from its source directory. Use the
existing `baseline/GP_semcache` R5 checkout prepared by its setup ZIP. The two
Qwen snapshots and the pinned TinyBERT snapshot are loaded from the Hugging Face
cache. Set `SEMCACHE_ALLOW_DOWNLOAD=1` for a run that should fetch missing models.

If a compatible environment is unavailable, prepare a separate one once.
PyTorch 2.6 or newer is required by Transformers 4.57.1 to load the pinned
TinyBERT PyTorch checkpoint.

```bash
conda create -n semcache-qwen python=3.11 pip -y
conda activate semcache-qwen
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'transformers==4.57.1' safetensors 'peft==0.18.1'
python -m pip check
```

## Moana

For a base-model smoke, export the MLP checkpoint and submit. To include the
LoRA trace, also export the two trained adapter directories before submitting.
The smoke uses the two adapters on Qwen3-4B. Use checkpoints trained with its
pinned revision when the adapter metadata does not record a base revision.

```bash
cd /data/surt321/repos/lab/kv_cache
conda activate semcache-qwen
export SEMCACHE_PYTHON="$(command -v python)"
export SEMCACHE_MLP="$PWD/results/neural_mapper_qwen3_17b_4b_k12_fp32_v1"
export SEMCACHE_OUT="$PWD/results/semcache/serving_$(date +%Y%m%d_%H%M%S)"

# Optional Ridge checkpoint from the completed paired run.
# export SEMCACHE_RIDGE="$PAIR_RUN/ridge"

# Optional Phase 3C LoRA checkpoints. Both are required for this controlled trace.
# export SEMCACHE_LORA_A="/data/$USER/models/lora_exp/phase3c_cross_domain_seed2026/medqa/seed2026/adapter_final"
# export SEMCACHE_LORA_B="/data/$USER/models/lora_exp/phase3c_cross_domain_seed2026/fpb/seed2026/adapter_final"
# export SEMCACHE_LORA_A_ID=medqa SEMCACHE_LORA_B_ID=fpb

mkdir -p "$SEMCACHE_OUT"
JOB_ID=$(sbatch --parsable --partition=batch_eebme_ugrad --nodelist=moana-y5 \
  --gres=gpu:1 --cpus-per-task=8 --mem=29G --time=01:00:00 \
  --chdir="$PWD" --export=ALL --job-name=kv-semcache-smoke \
  --output="$SEMCACHE_OUT/slurm-%j.out" --error="$SEMCACHE_OUT/slurm-%j.err" \
  run/seraph/semcache_smoke.sh)
printf '%s\n' "${JOB_ID%%;*}" > "$SEMCACHE_OUT/job_id.txt"
echo "Results in $SEMCACHE_OUT"
```

The three base requests exercise full prefill, mapped reuse of 512 tokens with
24 target prefill tokens and an exact-prefix miss. The optional LoRA requests
add adapter A admission, adapter A exact reuse, and adapter B recomputation
for the same prefix. The base and adapter traces are replayed on each enabled
mode. Model load, checkpoint staging and warmup are outside request timing.
TTFT includes routing, semantic search, mapper application and target prefill.
The report is a one-trace smoke. It records match diagnostics, not benchmark
quality or a statistically validated break-even point.

## Local tests

```bash
PYTHONPATH="$PWD/baseline/GP_semcache/src:$PWD/baseline/crosskv/src" \
  python -m unittest discover -s proposed/tests -p 'test_semcache_serving.py' -v
```

The tests use tiny Qwen3 models on CPU, the real R5 semantic index with a
controlled encoder, an identity mapper fixture and two PEFT safetensors
checkpoints. Run the Slurm smoke to verify the fitted mappers and trained
LoRA checkpoints on the actual GPU.
