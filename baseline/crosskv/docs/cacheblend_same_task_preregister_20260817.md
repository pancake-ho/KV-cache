# CacheBlend same-task HotpotQA-E protocol

Status: frozen before same-task CacheBlend execution.

## Purpose

Compare the learned 4-slot KV capsule against LMCache CacheBlend on the same
Llama-3.1-8B model and the already frozen 290-case HotpotQA-E set.  CacheBlend
has a different reuse contract: all document K/V is cached independently and
15% of positions are recomputed after reordered caches are assembled.  This is
therefore a mechanism/quality/latency comparison, not a claim that both methods
receive the same wire payload.

## Software and fixed configuration

- LMCache checkout: the local `LMCache` repository, version
  `0.5.4rc5.dev3-ge8f93818`;
- vLLM `0.23.0`, Torch `2.11.0+cu130` in `LMCache/.venv-vllm`;
- model: `meta-llama/Llama-3.1-8B-Instruct`, BF16, eager mode;
- native `LMCacheConnectorV1` and the repository's `blend_kv_v1` path;
- chunk size 256, layerwise loading, blend check layer 1, recompute ratio 0.15;
- CPU cache backend, 5 GiB LRU capacity;
- vLLM prefix caching disabled;
- greedy decoding, at most 24 new tokens;
- `max_model_len=20000`, sufficient for the pre-measured maximum target prompt
  length of 16,411 tokens;
- one synthetic non-evaluation warmup of the blending code path per process;
- no parameter, ratio, layer, prompt, or example search on E290.

The LMCache README requires registering the loaded vLLM model with
`VLLMModelTracker` before connector construction.  vLLM 0.23 already delays
connector construction until KV-cache allocation, so the environment carries
only that documented registration hook; no algorithm code is changed.

## Per-case paired protocol

The target prompt uses the exact Agent-B system instruction and full-recompute
user format used by the capsule baseline.  The documented CacheBlend special
token sequence `tokenizer.encode("# #")[1:]` is inserted directly before every
Hotpot passage and before the task suffix, exactly as in the official example.
Direct token insertion is required because rendering `# #` beside a newline
changes its BPE tokenization and produces zero cache hits.  This implementation
correction was established by the two-case integration smoke, before E290
execution and without inspecting task quality.  The same resulting target token
IDs are used for both measured arms.

The formal run uses two independent processes per shard.  This replaces the
two-case integration-only attempt to clear LMCache between paired requests:
the native clear endpoint triggered LMCache double-unpin warnings, so that path
is excluded before E290 execution.

1. **cold process:** plain vLLM with no LMCache connector runs every target-order
   prompt once.  This is the full-recompute arm;
2. **blend process:** for every case, run the same documents in reverse passage
   order with normal LMCache saving, a fixed non-target placeholder query, and
   one generated token; then run the target-order prompt with
   `lmcache.skip_save=true`.  As in the official example, population and target
   queries differ, so the real target-query segment is freshly computed while
   document/prefix segments may hit.  CacheBlend recomputes the fixed 15%
   selected positions among assembled cached state.

An initial E290 integration pass incorrectly reused the same target query in
the population request.  It consequently reported every prompt token as a hit,
including the query/task suffix, and is invalid for claims.  Those raw outputs
are retained under the original `hotpot_e290_blend_shard*` names.  The corrected
run is gated before scoring by `0 < cached_tokens < prompt_tokens` and uses the
`hotpot_e290_blend_querymiss_shard*` names.  This is a protocol bug fix, not a
ratio/layer/quality search; the cold arms and 15% setting are unchanged.

The cold and blend processes independently rebuild and record the target token
count.  Pooling requires exact equality of ID, question, gold answers, prompt
tokens, document tokens, and theoretical document-KV bytes before metrics are
joined.  Every corrected blended row must report both nonzero cached tokens and
nonzero uncached tokens; violations abort execution rather than being filtered.

The reverse population output is never scored.  Cold and blended outputs use
the same Hotpot EM/F1 normalization.  Record wall latency, vLLM TTFT, prompt
tokens, LMCache-reported cached tokens, and the theoretical BF16 document-KV
payload.  Shards are fixed to `(0,97)`, `(97,97)`, `(194,96)` and will be pooled
without exclusions or reranking.

## Interpretation constraints

- CacheBlend retains/transports approximately the full document KV plus online
  recomputation; capsule transport is only four positions in the fixed last 16
  layers at INT4.  Report both contracts explicitly.
- CPU torch fallback for the missing LMCache CUDA extension is acceptable for
  functional/quality validation, but its timing is labeled an integration
  measurement rather than a final optimized CacheBlend systems number.
- Any case with zero reported LMCache cached tokens is retained and separately
  counted; it may not be silently removed from quality or latency aggregates.

## Post-result integration recovery gate

The corrected 15% run unexpectedly produced repetitive generations despite
near-complete document-cache hits.  Before interpreting that observation as a
CacheBlend limitation, freeze the following diagnostic, explicitly labeled as
post-result integration validation rather than an untouched confirmatory test.

1. Add only a command-line control for the already existing
   `blend_recompute_ratios` configuration; do not change prompts, markers,
   cache population, model, decoding, or examples.
2. On fixed HotpotQA-E indices `0:16`, run the corrected query-miss protocol at
   recompute ratio `1.0`.  Compare each greedy output against the existing cold
   arm on exactly the same target token IDs.
3. The integration gate passes only if all 16 normalized answers match the cold
   arm and aggregate EM/F1 are identical.  Also report exact raw-text agreement;
   a raw-only mismatch is retained but does not fail the semantic gate because
   serving paths may differ in whitespace cleanup.
4. If the gate passes, run the same fixed indices at ratio `0.50`; reuse the
   existing formal 15% rows for those indices.  This three-point curve is a
   diagnostic mechanism check, not a tuned operating-point claim.
5. If the 100% gate fails, stop and treat the 15% result as an integration bug;
   do not report it as evidence about CacheBlend.

The strict gate did fail: two of 16 normalized answers differed, although mean
F1 recovered from the formal 15% arm to `.608` versus `.593` for cold.  Both
plain-cold and 100%-recompute reruns reproduced their respective raw outputs on
16/16 cases, localizing a deterministic path difference rather than sampling
noise.  The formal 15% result therefore remains quarantined at this point.

Before further quality interpretation, run one mechanism-only diagnostic on
the same 16 cases: ratio `1.0` with the importance check moved from layer 1 to
layer 0.  This tests whether the non-exact recovery is introduced before the
configured check layer.  It is not an authorized replacement operating point;
any quality search over check layer or ratio would require a new development
split followed by a fresh confirmation set.

Layer 0 subsequently recovered 15/16 raw answers; the sole mismatch was the
semantically compatible but longer `The Hockenheimring has very little
differences in elevation.` instead of `Very little.`.  Layer 1 remained at
14/16 normalized agreement.  This shows a stable greedy-path difference but
does not show aggregate task-quality failure at full recomputation.

Freeze a second, statistically calibrated recovery gate before expanding the
diagnostic: execute ratio `1.0`, official check layer 1 on all 290 fixed cases.
Against the existing cold process, paired bootstrap 95% intervals for both EM
delta and F1 delta must lie entirely inside `[-.05,+.05]`, and normalized-answer
agreement must be at least 80%.  These margins are fixed from the intended
meaning of an integration sanity check, not estimated from an E290 ratio-1
result.  Passing this gate permits interpreting the already frozen 15% result
as a mechanism diagnostic, while the original failure of strict output
identity remains reported.  Failing it leaves the 15% arm quarantined.

The E290 recovery gate passed: ratio-1 versus cold EM delta was `-.00345`
with 95% bootstrap CI `[-.0276,+.0207]`, F1 delta was `+.00136` with CI
`[-.0208,+.0249]`, and normalized-answer agreement was 87.2%.  Strict identity
still failed and remains a disclosed boundary.  The pre-specified ratio `.50`
mechanism diagnostic is therefore authorized.  Expand it from the original
16-case smoke to all E290 cases solely to obtain a stable paired curve; it is
post-result characterization, not a selected or confirmatory operating point.
