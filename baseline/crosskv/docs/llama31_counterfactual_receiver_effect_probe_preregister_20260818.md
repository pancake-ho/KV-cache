# Counterfactual full-receiver effect probe

Status: frozen on 2026-08-18 before any probe output was observed.

## Mechanism question

Coordinate and fixed-query operator similarity both rank state8 above CE8,
although equal-capacity CE8 is significantly better on the downstream task.
This probe tests a different equivalence criterion: whether a compact cache
induces the same *marginal causal computation* in the receiver as a normal
plaintext agent handoff.

For memory `M`, define its layerwise receiver effect at teacher-forced answer
prediction positions as

`E(M) = H(system, M, query, answer-prefix) - H(system, zero(|M|), query, answer-prefix)`.

The zero-K/V counterfactual has exactly the same number of positions as its
memory.  Thus each subtraction holds query positions fixed and removes the
common system/query trajectory and the normalization effect of merely adding
slots.  Teacher and student may have different memory lengths because their
effects are compared only after their own length-matched subtraction.

## Frozen arms and cases

- Llama-3.1-8B-Instruct, BF16 receiver execution.
- Three existing checkpoints: state4, CE8, and state8 from the frozen native
  Tail scale-up.
- Canonical, decoded all-layer K4/V4 packets are moved after the dependent
  receiver system prefix.
- Same 64 MuSiQue cases at confirm128 rows 32--95 used for the cap8 reversal.
  They do not overlap any checkpoint's training IDs.  This is a mechanistic
  explanation set, not a new confirmation set.
- Final F1 is joined from each arm's existing real-canonical correct-packet
  evaluation; it is not regenerated or selected here.
- 20,000 paired bootstrap resamples, seed 2099.

## Frozen plaintext teacher and trajectory

The privileged teacher receives the ordinary explicit history:

1. user: `Agent A intermediate answer: <gold bridge answer>`;
2. assistant: `I have read and retained Agent A's briefing.`;
3. the same final user query and teacher-forced final answer as the capsule
   receiver.

The student path never receives this text.  The teacher and student final-query
token IDs must match exactly or the probe aborts.  Block-output hidden states
are collected after every one of the 32 decoder blocks at the final query
state predicting answer token one and at all subsequent teacher-forced answer
predictors.  Because each complete forward is executed under its own memory,
all later-layer queries are on-policy and include prefix competition.

For each layer, flatten token and hidden dimensions.  The loss is

`1 - cosine(E_student, E_teacher) + 0.1 * log(||E_student|| / ||E_teacher||)^2`,

then averaged equally across 32 layers.  Direction, log-norm, and total terms
are all reported; the total with fixed coefficient 0.1 is primary.

## Frozen decision gate

The probe explains the known cap8 reversal only if state8-minus-CE8 total
effect loss is positive with a paired bootstrap 95% CI lower bound above zero.
The paired positive fraction, component losses, state4 ordering, and
effect-delta/F1-delta correlation are descriptive only.

If the primary gate passes, exactly one continuation using this
counterfactual receiver-effect auxiliary is authorized.  Its weight and update
budget must be frozen in a separate training preregistration before observing
held-out candidate quality.  If the gate fails, no effect-distillation
training is authorized: another loss scan over coordinates, operators, hidden
states, or weights would reuse the same failed proxy-selection logic.

This probe cannot establish cross-task generalization, plaintext dominance,
or an end-to-end transfer advantage.
