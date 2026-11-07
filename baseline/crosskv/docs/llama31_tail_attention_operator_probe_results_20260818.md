# Tail-KV attention-operator probe results

Date: 2026-08-18.  This resolves
`llama31_tail_attention_operator_probe_preregister_20260818.md`.

## Result

The fixed no-summary query probe does **not** explain the cap8 quality
reversal.  Mean layer-1--31 operator losses are:

| arm | conditional-value error | log-mass error | weighted total | final F1 |
|---|---:|---:|---:|---:|
| state4 | 1.02486 | 4.89000 | 1.51386 | .88281 |
| CE8 | 1.76290 | 11.16517 | 2.87942 | .92188 |
| state8 | 1.18091 | 8.55139 | 2.03605 | .87240 |

State8-minus-CE8 total operator error is `-.84337`, CI
`[-.88296,-.80330]`; all 64 cases improve.  Conditional-value and log-mass
components also improve on all 64 cases, with fully negative CIs.  Nevertheless
final F1 is `-.04948`, CI `[-.10417,-.00781]`.  The Pearson correlation between
per-case operator-error delta and F1 delta is only `.1298`.

State4 has lower operator error than state8 relative to its own four-token
teacher, but this cross-length absolute comparison is descriptive because the
teacher memory and receiver query positions differ.

## Interpretation

The preregistered second branch applies.  A tail-only operator evaluated on
fixed no-summary receiver queries is not a sufficient functional target.  Two
explanations remain and are not separated by this probe:

1. the gold lexical answer Tail is a privileged intermediate/readout state,
   not the optimal compressed state for the downstream transformation;
2. the relevant operator is on-policy and includes competition with the full
   system/query prefix, whose K/V and query trajectory change after inserting
   the capsule.

This also explains the metric split: state8 raises bridge F1 relative to CE8
while lowering final F1.  It becomes more like Agent A's lexical answer state
but loses some task-optimized information needed by Agent B.

No operator-training run is authorized from this diagnostic.  The next
mechanism test should measure the capsule's *marginal effect on full receiver
attention* using on-policy query trajectories and the common-prefix
normalizer.  If the native teacher is still anti-aligned with final-answer
gradients there, the correct method is a constrained/task-tangent projection,
not stronger imitation.

Machine-readable results:
`outputs/amortized_semantic_kv/llama31_native_tail_scaleup_state8_800_train128_eval64_20260818/attention_operator_probe.json`.
