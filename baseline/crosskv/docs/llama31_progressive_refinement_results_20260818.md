# Causally nested 4+4 capsule: feasibility results

Date: 2026-08-18.  This reports the two frozen opened-development pilots in
`llama31_progressive_refinement_feasibility_preregister_20260818.md` and
`llama31_progressive_causal_contrast_preregister_20260818.md`.

## Representation and wire mechanism

The representation mechanism succeeds.  A frozen four-slot core is
materialized once, then reused as part of the cached prefix for four causal
refinement slots.  Only refinement embeddings and a rank-16 block-14 boundary
adapter are trainable (147,456 parameters).

The real Llama/H200 probe verifies:

- reused core tensors have identical storage and bitwise-identical values;
- the core prefix of the eight-slot cache is bitwise identical;
- standalone and full-prefix core packet SHA256 hashes are identical;
- separately decoded core+refinement equals the decoded monolithic eight-slot
  packet tensor-for-tensor;
- core and incremental refinement are each 67,642 B; split cold is 135,284 B
  versus 135,226 B monolithic, only 58 B framing overhead.

This is a valid progressive wire abstraction.  It does not by itself establish
that the refinement state is useful.

## CE-only pilot: failed quality and causality

On opened MultifieldQA rows 0--31 train and 32--47 eval, 200 updates reduce
mean loss from `3.09886` over the first 20 steps to `1.31209` over the final
20, a 57.66% reduction.  Exactly 147,456 parameters change; the core hash is
unchanged; all packet checks pass.

| arm | mean F1 |
|---|---:|
| core-4 | `.13299` |
| full-8 | `.14690` |
| shifted full-8 | `.20765` |
| no state | `.24441` |

Full minus core is `+.01392`, CI `[-.04131,+.06784]`, below the frozen
`+.02` gate.  Correct full minus shifted is `-.06075`, CI
`[-.18791,+.03932]`.  Full minus no-state is `-.09751`, with a negative upper
confidence boundary.  Utility and causality fail; no larger CE-only run is
authorized.

## Counterfactual hinge follow-up: failed

The one mechanism-specific follow-up adds a fixed margin loss requiring the
gold answer NLL under correct source to beat a shift-1 source by `.5` nats per
token.  It starts again from the frozen core, not the failed refinement.

Total loss falls 55.94%, all integrity/wire gates pass, but quality worsens:

| arm | mean F1 |
|---|---:|
| core-4 | `.13299` |
| full-8 | `.09223` |
| shifted full-8 | `.14383` |
| no state | `.24441` |

Full minus core is `-.04076`, CI `[-.13520,+.03297]`; correct minus shifted
is `-.05160`, CI `[-.13941,+.02598]`; and full minus no-state is `-.15218`,
CI `[-.27343,-.05200]`.  Both required quality gates fail.  Margin, weight,
rank, and training length will not be scanned.

## Conclusion

The causal nesting idea solves low-budget invariance and incremental transport
exactly, but the minimal learned refinement does not reproduce the capacity
gain of generated-answer Tail-KV.  Falling training loss alongside reversed
source controls shows that question-conditioned CE and a simple negative-NLL
hinge can optimize task/format effects without learning a transferable
positive state channel.

The generated-Tail capacity result should therefore be interpreted narrowly:
additional *native answer-token* positions help long answers, but additional
generic soft positions are not an interchangeable substitute.  The next
paper-facing experiment should characterize an adaptive generated Tail-KV
token-by-layer Pareto frontier.  The progressive learned representation is
retained as a systems mechanism and negative quality result, not promoted as
the current method.
