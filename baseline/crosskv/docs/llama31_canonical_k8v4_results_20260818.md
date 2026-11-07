# Canonical Cartesian K8/V4 packet results (2026-08-18)

## Decision

The fixed K8/V4 canonical candidate **passes all five preregistered gates** on
checkpoint-training-disjoint LongBench MuSiQue-200.  It provides exact
receiver-independent bytes, is non-inferior to both the K4/V4 target-frame and
K4/V4 canonical representations, and retains a barely positive correct-source
causal lower bound.

It does **not** establish that extra Key bits improve answer quality.  K8/V4
minus K4/V4 canonical is only `+.00243` question F1, CI
`[-.01710,+.02227]`, despite reducing layer-15 tensor error by about 85.6%.
Cold bytes increase 48.43%.  The result is therefore a safe higher-rate
same-family point, not a new default or a significant rate--quality win.

## Protocol and data audit

- Candidate: signed Cartesian INT8 Key, signed Cartesian INT4 Value, one
  transmitted FP16 scale per layer/head/slot vector.
- Canonical positions 0--3 on the wire; receiver-side RoPE after decode.
- Frozen Llama-3.1-8B, rank-16 task checkpoint, unrestricted base, four slots,
  first16 base, layer-15 delta, greedy 24-token decode.
- Confirmation: all 200 rows of `/data/datasets/longbench/musique.jsonl`,
  SHA256 `4ac69b91281c4ec6b21316cb7282e83fb6b4dda04fc68480bb8d8ed1e19ff7bd`.
- Exact normalized-question overlap is zero with all checkpoint-training
  manifests and recent frozen MuSiQue confirmation sets.  Thirty-two questions
  overlap the earliest `dev197` exploration with different LongBench context;
  the candidate and comparison were frozen before that metadata audit.  This
  is checkpoint-training-disjoint same-family evidence, not a wholly unseen
  task claim.
- Rows 0--15 of opened 2Wiki were a disclosed screen and are absent from the
  confirmation numbers.

## Wire audit

| component | K4/V4 | K8/V4 |
|---|---:|---:|
| first16 base | 67,642 B | 100,410 B |
| layer-15 delta | 4,252 B | 6,300 B |
| cold base + delta | 71,894 B | 106,710 B |

All 200 K8/V4 base and delta hashes are identical across state-readout and
question-conditioned receiver prefixes.  Layers 0--14 remain the exact decoded
base and only layer 15 receives the independent delta.  The 6,300-B base-hit
transfer is 6.274% of the base, passing the 6.30% gate.

K8/V4 layer-15 delta-vs-task NRMS is `.02968`, versus `.20551` for K4/V4
canonical.  This 85.6% tensor-error reduction is real, but it is not itself a
behavioral result.

## Question-conditioned confirmation (`n=200`)

| representation | EM | F1 |
|---|---:|---:|
| K4/V4 target-frame sidecar | .050 | .11376 |
| K4/V4 canonical sidecar | .055 | .11914 |
| **K8/V4 canonical sidecar** | **.055** | **.12158** |
| shifted K8/V4 canonical | .035 | .09353 |

Paired comparisons:

- K8/V4 canonical minus K4/V4 target: `+.00782` F1, CI
  `[-.00708,+.02488]`.  The point is above `-.01` and lower bound above `-.03`;
  target-frame safety passes.
- K8/V4 canonical minus K4/V4 canonical: `+.00243`, CI
  `[-.01710,+.02227]`.  The point is above `-.005` and lower bound above
  `-.03`; canonical non-inferiority passes.  The CI is not positive, so
  significant Key-rate benefit is not supported.
- Correct K8/V4 minus shifted K8/V4: `+.02805`, CI
  `[+.00019,+.05782]`; 20 wins, 168 ties, 12 losses.  The causal gate passes,
  narrowly.

K4/V4 canonical itself is safe versus target on this set (`+.00538`, CI
`[-.00452,+.01873]`) but its correct-vs-shift CI is
`[-.00247,+.05202]`.  Extra Key precision changes enough discrete outputs to
cross the causal lower-bound threshold, but not enough to demonstrate an
overall quality improvement.

## State-readout diagnostic

Generic state readout remains weak: K8/V4 F1 is `.05187`, K4/V4 canonical
`.05544`, and target-frame `.05426`.  K8/V4 versus target is `-.00239`, CI
`[-.00822,+.00100]`.  Correct K8/V4 versus shift is `+.05187`, CI
`[+.02988,+.07642]`, so the small readout signal is source-causal.

## Gate table

| preregistered gate | result |
|---|---|
| exact bytes and receiver-independent hashes | pass |
| target-frame safety | pass |
| canonical non-inferiority | pass |
| significant K8 quality benefit | not established (stronger optional claim) |
| correct-source causality | pass narrowly |
| base-hit delta no more than 6.30% of base | pass |

Overall fixed-protocol decision: pass.  Paper interpretation: K8/V4 is a
same-family robustness point and a precision control.  It is not evidence that
canonical RoPE is universally safe or that K8 is Pareto-preferred.

## Artifacts

- Preregistration: `docs/llama31_canonical_k8v4_preregister_20260818.md`
- K8/V4 candidate:
  `outputs/amortized_semantic_kv/llama31_r16_canonical_k8v4_sidecar_longbench_musique200_20260818/`
- K4/V4 target/canonical references:
  `outputs/amortized_semantic_kv/llama31_r16_target_k4v4_sidecar_longbench_musique200_20260818/`,
  `outputs/amortized_semantic_kv/llama31_r16_canonical_k4v4_sidecar_longbench_musique200_20260818/`
- Paired analyses:
  `outputs/amortized_semantic_kv/llama31_r16_canonical_k8v4_analysis_20260818/`
