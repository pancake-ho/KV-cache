# Qwen3 mapper optimization experiments

Run commands from the KV-cache repository root on Seraph. This workflow uses one Slurm GPU, the existing `lab` environment, and Transformers 4.57.3. The Python entry points expect NumPy, PyTorch, Transformers, and safetensors. They reuse the RoPE and activation-store APIs in `baseline/crosskv/src/xmodel_kv`.

| Path | Responsibility |
| --- | --- |
| `proposed/src/exp/rarecache/` | Rank disagreement and in-place sparse replay with absolute positions |
| `proposed/src/exp/cachebridge/` | CacheBridge v1 receiver attention sensitivity and ESS shrinkage |
| `proposed/src/exp/mapper_optim/` | Pair validation, calibration traces, OLS support, ridge fit, artifact loading, quality and latency evaluation |
| `run/seraph/{rarecache,cachebridge}/run.sh` | Method-specific launch commands |
| `run/seraph/mapper_optim/` | Shared Slurm submission, status and output handling |

## Check before submitting

```bash
bash run/seraph/rarecache/run.sh --help
bash run/seraph/cachebridge/run.sh fit --help
```

The second command uses `/data/$USER/anaconda3/envs/lab/bin/python` by default. Set `KV_OPT_PYTHON` to the absolute Python path if your existing environment lives elsewhere. The launcher defaults to `batch_eebme_ugrad`, 1 GPU, 8 CPUs, 29G RAM and four hours. It accepts `KV_OPT_CPUS`, `KV_OPT_MEM`, `KV_OPT_TIME` and `KV_OPT_PARTITION` overrides. Use `--dry-run` before `fit` or `eval` to print the submission command without creating files or submitting.

## Fit

Use separate calibration and evaluation JSONL files with one `{"text":"…"}` object per line. The reader uses the first requested number of distinct, sufficiently long rows and takes their first requested number of tokens. The source and target must be dense Qwen3 with full causal attention and default RoPE. Both tokenizers must agree on the vocabulary, special tokens and input token IDs. CacheBridge v1 also requires equal KV-head counts for identity head assignment. Keep both models and the mapper in the assigned GPU's memory for evaluation.

Set the variables below to the agreed experimental settings before running. Reuse the same calibration rows, layer budget and ridge value across CacheBridge variants. The code uses sum-loss ridge as written in the papers; the existing CrossKV baseline uses mean-normalized ridge.

```bash
fit_args=(
  --source-model "$SOURCE_MODEL" --target-model "$TARGET_MODEL"
  --calibration-jsonl "$CALIB_JSONL"
  --sequence-length "$CALIB_LENGTH" --sequences "$CALIB_SEQUENCES"
  --top-k "$TOP_K" --stride "$CALIB_STRIDE" --dtype "$MODEL_DTYPE"
)
bash run/seraph/rarecache/run.sh fit "$RARECACHE_FIT_TAG" "${fit_args[@]}" --rank "$RANK"
bash run/seraph/cachebridge/run.sh fit "$CACHEBRIDGE_FIT_TAG" "${fit_args[@]}" --variant attn-repair
```

The CacheBridge fit variants are `full-head`, `head-local` and `attn-repair`. All three use the same matched-head layer selection on the same inputs. RaReCache uses separate full-head K and V support and derives the rank basis from the covariance of mapped calibration outputs. The OLS probe uses at most `--probe-rows` (default 2048) and requires more rows than the single-source feature width. Head-local fitting uses one feature block per selected source layer and head. Attn-Repair weights are shrunk toward uniform to satisfy the paper's effective sample size floor. When calibration rows are no more than twice the head-local width, this floor yields uniform weights. Inspect the recorded `alpha` values before interpreting weighted-fit results.

## Evaluate

Each completed fit publishes an `output/artifact` directory. The default fit results root is `/data/$USER/kv_mapper_optim_runs/<method>/<tag>/`. Pass its artifact path as `RARECACHE_ARTIFACT` or `CACHEBRIDGE_ARTIFACT` below, and use the same model dtype as calibration. Results are keyed by method and unique tag. Compute-node calibration traces use `/local_datasets/$USER/kv_mapper_optim`, configurable via `KV_OPT_SCRATCH`.

```bash
eval_args=(
  --eval-jsonl "$EVAL_JSONL" --prefix-length "$PREFIX_LENGTH"
  --continuation-tokens "$CONTINUATION_TOKENS" --examples "$EVAL_EXAMPLES"
  --dtype "$MODEL_DTYPE"
)
bash run/seraph/rarecache/run.sh eval "$RARECACHE_EVAL_TAG" \
  --artifact "$RARECACHE_ARTIFACT" "${eval_args[@]}" --recompute-ratios 0 "$RHO" 1
bash run/seraph/cachebridge/run.sh eval "$CACHEBRIDGE_EVAL_TAG" \
  --artifact "$CACHEBRIDGE_ARTIFACT" "${eval_args[@]}"
```

Evaluation compares native target prefill, pure transfer, and the requested RaReCache replay ratios. The mapper receives the first `prefix_length - 1` tokens. The target processes the final prompt token to get the first continuation logit. RaReCache recomputes `ceil(rho * (prefix_length - 1))` positions within the mapped cache. The output includes continuation NLL, top-1 agreement, selected positions, fixed-length greedy IDs, mapper bytes, synchronized stage latencies, and allocated/reserved memory. The native and mapped paths use the same resident source/target models and mapper. Handoff time assumes source KV is available. The report also gives staged totals with source prefill included. These measurements use a local single-GPU runtime.

CacheBridge v1 uses bounded two-pass PyTorch ridge panels and head-batched mapping. The paper's custom Fused-Fit kernel and multi-GPU serving are outside this implementation. The current runner measures standalone handoff and continuation on one GPU. Published task accuracy, distributed TTFT and throughput require separate evaluations.

Paper references: [RaReCache](https://arxiv.org/abs/2610.11358v1) · [CacheBridge v1](https://arxiv.org/abs/2609.00891v1)
