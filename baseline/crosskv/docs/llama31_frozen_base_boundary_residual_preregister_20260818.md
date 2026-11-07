# Frozen-base boundary-residual preregistration (2026-08-18)

## Question

Can a four-slot KV summary retain the unrestricted capsule's transferable
state while adding a task-specific, emission-aligned residual, without
rewriting the transferable base?

This experiment follows two observed failures.  A source-read-15 bottleneck
improved the opened MuSiQue confirmation set but failed on 2WikiMQA, while
behavioral KL from the unrestricted capsule recovered only part of the
cross-task state and damaged the in-domain result.  The present intervention
changes the parameterization rather than tuning the KL weight.

## Fixed mechanism

- Base: the unrestricted continued capsule
  `llama31_increment256_allonly_cont3200_confirm128_20260818/capsule.pt`.
- Freeze all four slot embeddings and the complete rank-4 layerwise write
  adapter from that checkpoint.
- Permit source-prefix attention only in decoder blocks 0--14
  (`source_read_layers=15`).
- Add one zero-initialized rank-4 residual after decoder block 14.  It changes
  `h15`, whose first serialized effect is K/V layer 15, the final layer in a
  `first_16` packet.
- Train only this residual: 32,768 parameters.  There is no teacher KL and no
  added wire payload.
- Four slots, straight-through INT4, all-layer receiver loss, rank 4, scale
  1.0, AdamW, learning rate 5e-4, weight decay 0.01, 3,200 steps, seed 2027,
  stage weight 1 and bridge weight 2.  These values are inherited from the
  preceding fixed run; no rank, learning-rate, weight-decay, step, or
  checkpoint scan is allowed.

The zero initialization provides a structural invariant: before training,
the transmitted layers 0--15 are exactly the unrestricted base packet.  The
source mask begins changing hidden states only after K/V layer 15 has been
projected.  Unit tests additionally require exact zero-residual equality,
gradient isolation to the boundary residual, hook cleanup, and checkpoint
round-trip recovery.

## Data and controls

- Training: the previously fixed 256-example MuSiQue increment split.
- Opened in-domain development: the disjoint 96-example alignment confirmation
  split (96 unique IDs, 64 bridge-answer clusters).
- Opened cross-task development: LongBench 2WikiMQA examples 0--199, using the
  already fixed question-conditioned protocol.
- Controls are frozen existing outputs: unrestricted base, source-read-15
  full-parameter training, pre-answer-KL student, no-summary, and circular
  source shift by one.
- The sealed `2wikimqa_e` set is not reopened for this design.

All primary comparisons use paired answer F1.  Ordinary case bootstrap and
bridge-cluster bootstrap use the existing analysis implementation and fixed
seeds.  Exact match is secondary.

## Gates fixed before the full run

1. **In-domain task increment:** on MuSiQue first16/INT4, residual minus the
   unrestricted base has mean F1 at least +0.04 and the ordinary paired 95%
   bootstrap interval is positive.
2. **In-domain safety:** residual minus the full-parameter source-read-15
   model has F1 at least -0.05 and both ordinary and cluster lower confidence
   bounds exceed -0.08.
3. **Cross-task base retention:** on 2WikiMQA question-conditioned
   first16/INT4, residual minus unrestricted base has mean F1 at least -0.03,
   with the lower paired bound above -0.08.
4. **Cross-task improvement over task-only bottleneck:** residual minus the
   full-parameter source-read-15 model is positive and its ordinary paired
   interval is positive.
5. **Causal state use:** correct minus shifted source on both opened datasets
   is positive with positive ordinary paired intervals.
6. **Full-packet safety:** on each opened dataset, all-layer residual minus
   unrestricted base has a lower paired bound above -0.10.  This is a safety
   diagnostic, not permission to replace the primary first16 endpoint.

The design is considered a successful decomposition only if gates 1--5 all
pass.  Gate 6 is separately reported.  A failed gate will be reported as a
negative result; no post-hoc hyperparameter rescue is part of this run.

## Configuration-audit note

The first launch accidentally inherited the CLI default weight decay of zero
instead of the preceding run's 0.01.  The mismatch was detected during the
run, before any checkpoint or evaluation output existed, and that process was
aborted at step 1,100.  Its partial directory is retained and labeled as
aborted.  It is not a candidate and none of its outcomes were used to choose
the corrected configuration.  The sole valid full run starts from step zero
with the explicitly recorded weight decay above.
