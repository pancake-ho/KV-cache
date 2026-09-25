# Native Tail-state distillation cap8 scale-up results

Date: 2026-08-18.  This resolves the frozen protocol in
`llama31_native_tail_state_cap8_scaleup_preregister_20260818.md`.

## State fit succeeds

State8 training loss falls 67.93%, from 1.3994 over the first 20 steps to
.4488 over the final 20.  On 64 disjoint development cases, canonical K4/V4
state relative MSE is 1.28293 for CE8 and .45603 for state8.  The ratio is
35.55%; paired state8-minus-CE8 is `-.82690`, CI
`[-.85522,-.79782]`, with all 64 cases improving.

## Functional quality does not follow coordinate error

All behavioral numbers use real framed canonical packets.

| arm | slots | bytes | correct final F1 | shifted final F1 | bridge F1 |
|---|---:|---:|---:|---:|---:|
| state4 | 4 | 135,258 | .88281 | .19115 | .90372 |
| CE8 | 8 | 270,426 | .92188 | .17292 | .88229 |
| state8 | 8 | 270,426 | .87240 | .13646 | .91563 |

State8 is worse than the equal-capacity CE8 control by `-.04948` final F1,
CI `[-.10417,-.00781]`: four cases regress and none improve.  It is also
`-.01042` below state4, CI `[-.07031,+.05208]`, instead of the preregistered
positive capacity gain.  Bridge state8-minus-CE8 is `+.03333`, but its CI
`[-.00833,+.08229]` crosses zero.

The state remains strongly source-causal: state8 correct-minus-shift final F1
is `+.73594`, CI `[+.64323,+.82552]`.  Packet identity is invariant across
receiver targets for every case.

## Frozen decision

Three of five gates pass: optimization, held-out coordinate error, and source
causality.  Equal-capacity final quality and state8-over-state4 capacity gain
fail.  The cross-task run is therefore **not authorized** and the state-loss
weight must not be tuned on these 64 cases.

This is a useful representation result rather than a generic negative.  Raw
per-position K/V coordinates are not the semantic object that downstream
attention consumes.  A cache denotes a query-dependent operator

\[
q \mapsto \operatorname{softmax}(qK^\top/\sqrt d)V,
\]

and different token counts or K/V coordinates can implement similar
operators.  Coordinate MSE can therefore improve dramatically while forcing
irrelevant or unreachable teacher degrees of freedom and reducing final
quality.  The next objective should match the attention operator on real
receiver-query probes, including its log-normalizer, rather than scan the
coordinate-loss weight or add more slots.

Machine-readable analysis:
`outputs/amortized_semantic_kv/llama31_native_tail_scaleup_state8_800_train128_eval64_20260818/frozen_analysis.json`.
