# Tail-KV attention-operator probe

Status: frozen on 2026-08-18 after the cap8 coordinate-distillation scale-up
failed its two quality gates, before computing any operator metric.

## Motivation

State8 is much closer to its answer-Tail teacher in per-position K/V MSE than
CE8, yet is significantly worse in final F1.  This probe asks whether raw
coordinate matching failed because it did not match the functional memory
operator seen by receiver queries.

For each layer and query probe, a K/V memory is summarized by both

\[
\mu(q)=\operatorname{softmax}(qK^\top/\sqrt d)V
\]

and `logsumexp(qK^T/sqrt(d))`.  The second term is necessary because a tail
memory competes with common receiver-prefix tokens in the full softmax; its
conditional value alone is insufficient.  This comparison is invariant to a
joint permutation of K/V slots and permits teacher and student lengths to
differ.

## Frozen probe

- Same 64 MuSiQue confirm cases and state4/CE8/state8 checkpoints used by the
  cap8 scale-up decision.
- Student and gold-answer Tail teacher are canonical K4/V4 decoded and moved
  to the same receiver-system position.
- Queries are captured from the common no-summary receiver trajectory at all
  teacher-forced final-answer prediction positions.  Their pre-RoPE content is
  fixed across arms and rotated after a hypothetical 4- or 8-slot memory.
- Operator layers 1--31; layer 0 remains the structural negative control.
- Loss: teacher-normalized conditional-value MSE plus `.1` times log-mass MSE.
- 20,000 paired bootstrap replicates, seed 2097.

## Interpretation fixed before observation

1. If state8 has higher operator error than CE8 despite lower coordinate MSE,
   query-conditioned operator distillation is directly supported as the next
   objective.
2. If state8 also has lower operator error but worse F1, this fixed no-summary
   probe is insufficient or the privileged gold-answer Tail operator is itself
   the wrong target.  The next design must include causal prefix competition
   and/or on-policy receiver queries; it cannot claim that operator distance
   already explains the failure.
3. Correlation is diagnostic only.  No new training run is selected from these
   64 cases.
