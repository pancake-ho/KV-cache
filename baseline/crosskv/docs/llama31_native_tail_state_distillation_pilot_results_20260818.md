# Equal-rate native Tail-KV state-distillation pilot results

Date: 2026-08-18.  This resolves the frozen protocol in
`llama31_native_tail_state_distillation_preregister_20260818.md`.

## Intervention

Three four-slot/all32 continuations share the same checkpoint, 32 MuSiQue
training cases, 400 steps, and canonical K4/V4 rate.  CE is the compute-matched
control; behavior-KL matches receiver logits from a gold-answer Tail-KV;
direct-state matches canonical quantized K/V at layers 1--31 to the first four
gold-answer positions.  All final quality numbers below use the real framed
codec, not fake-quant accounting.

## Optimization and held-out state geometry

Direct-state training MSE falls from 1.3952 over steps 1--20 to .5359 over
steps 381--400, a 61.59% reduction.  On the 32 disjoint evaluation cases:

| arm | canonical K4/V4 Tail-state relative MSE |
|---|---:|
| CE | 1.42518 |
| behavior-KL | 1.43523 |
| direct-state | .62401 |

Direct-state is 43.78% of the CE error.  Its paired delta is `-.80117`, CI
`[-.87059,-.72903]`, and it is lower on all 32 cases.  Behavioral KL does not
align the native state geometry.

## Real canonical packet behavior

Every packet is target-invariant and exactly 135,258 framed bytes.

| arm | correct final F1 | shifted final F1 | correct bridge F1 |
|---|---:|---:|---:|
| CE | .85938 | .20312 | .95833 |
| behavior-KL | .70312 | .23438 | .82812 |
| direct-state | .92188 | .21563 | .94271 |

Direct-state minus CE final F1 is `+.06250`, CI `[.00000,+.15625]`: two cases
improve and none regress.  Bridge F1 is `-.01562`, CI
`[-.06771,+.02604]`, so the final-answer improvement does not come with a
resolved bridge improvement.  Behavior-KL loses `-.15625` final F1 and
`-.13021` bridge F1; the latter CI is fully negative.

Direct-state correct minus shifted-source final F1 is `+.70625`, CI
`[+.56250,+.84375]`; bridge causality is `+.86458`, CI
`[+.75000,+.95833]`.  Thus the state arm is not merely a learned answer prior.

## Frozen decision

All four feasibility gates pass:

1. training state loss reduction exceeds 30%;
2. held-out state MSE improves by more than 10% with a negative CI upper bound;
3. final F1 is non-negative versus CE and its CI lower is above `-.05`;
4. correct-source causality has a positive CI lower bound.

A cap8 scale-up is authorized.  This is still opened, same-family development
evidence.  It does not establish external generalization or superiority to
generated Tail-KV.  The next run must keep a cap8 CE control, test whether
capacity and direct-state supervision interact, and then evaluate on a
cross-task set without selecting a weight on that set.

Machine-readable analysis:
`outputs/amortized_semantic_kv/llama31_native_tail_distill_state010_400_train32_eval32_20260818/frozen_analysis.json`.
