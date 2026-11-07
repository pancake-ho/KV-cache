# Tail-KV native-objective/task-gradient alignment results

## Decision

Neither native objective passes the frozen conflict gate.  Task-tangent or
PCGrad-style native distillation is not authorized.  The cap8 quality reversal
cannot be attributed to a systematic local opposition between native-state
and downstream gradients at the CE8 solution.

## Frozen diagnostic result

The probe used rows 96--127 of the MuSiQue confirm128 file, the frozen CE8
checkpoint, canonical all-layer K4/V4, 20,000 paired bootstrap resamples, and
seed 2098.

| gradient pair | mean cosine | bootstrap 95% CI | negative cases |
|---|---:|---:|---:|
| coordinate--final | -.00185 | [-.04367, +.04375] | 46.88% |
| coordinate--bridge | +.02941 | [-.01671, +.07659] | 37.50% |
| operator--final | +.00337 | [-.04908, +.05479] | 46.88% |
| operator--bridge | +.01517 | [-.04442, +.07498] | 40.63% |
| final--bridge | +.00419 | [-.03677, +.04466] | 50.00% |

The coordinate bridge-minus-final alignment gap is `+.03126`, CI
`[-.04193,+.10024]`; the operator gap is `+.01181`, CI
`[-.07721,+.09576]`.  Both cross zero.  The native/final negative fractions
are also far below the frozen 65% alternative condition.

## Interpretation

At this checkpoint, all five gradient pairs are approximately orthogonal in
the aggregate and heterogeneous across cases.  This rules out the narrow
mechanism that direct native-state training fails because its instantaneous
gradient consistently reverses final-answer progress.  It does not prove that
the objectives are globally compatible: a privileged post-answer token Tail
can still constrain the optimization trajectory toward the wrong
representation manifold.

The next diagnostic therefore follows the preregistered fallback: measure a
memory's marginal effect on the complete receiver trajectory using on-policy
queries and full prefix competition, with an explicit plaintext handoff rather
than a lexical answer-Tail as the teacher.

## Artifact

`outputs/amortized_semantic_kv/llama31_native_tail_scaleup_ce8_800_train128_eval64_20260818/task_gradient_alignment_confirm32.json`
