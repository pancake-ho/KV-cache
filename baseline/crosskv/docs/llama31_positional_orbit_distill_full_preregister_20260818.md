# Positional-orbit distillation: full continuation and external protocol

Status: frozen on 2026-08-18 after the fixed 200-step feasibility pilot and
before any model was run on `multifieldqa_en`.

## Decision carried forward from the pilot

The development-only pilot used exactly the configuration in
`llama31_positional_orbit_distill_feasibility_preregister_20260818.md`.
All authorization gates passed without a hyperparameter scan:

- orbit KL: `.236345` over steps 1--20 and `.106953` over steps 181--200,
  a 54.75% reduction;
- exactly 131,072 rank-16 boundary parameters changed;
- all frozen capsule and independent-base hashes were unchanged;
- on opened MuSiQue alignment rows 0--15, canonical F1 was `.84375` versus
  `.87500` for the frozen parent and `.18750` under a one-case source shift;
- real packets remained 67,642-B base plus 4,252-B delta and were exactly
  receiver-target invariant.

The pilot checkpoint, its 16 rows, and its stopping point cannot be used as a
paper result.  They authorize one full continuation with the fixed choices
below.

## Frozen training run

Train an independent final candidate from the pre-pilot rank-16 checkpoint,
not from the pilot checkpoint:

- model: `meta-llama/Llama-3.1-8B-Instruct`;
- initialization:
  `llama31_frozen_base_boundary_r16_read15_cont3200_alignment96_wd001_20260818/capsule.pt`,
  SHA256 `94368e440e26f38c6152479afa2a5b8c624a48f8c16f9b91d1bc877d7bf45a14`;
- independent reusable base:
  `llama31_increment256_allonly_cont3200_confirm128_20260818/capsule.pt`,
  SHA256 `c488ccefca03cb3dbdfa89740aa2962cfe23979f0da28f9d1abc9594c1debefc`;
- train set: `train256_increment_seed2081_exclude_used1152.jsonl`, SHA256
  `c0c5ec89ffe569fef6b94e387c1b4ceb42e271787cc0686a0cc249d6dc4ffa1b`;
- exactly 1,000 updates from the initialization, final checkpoint only;
- AdamW, learning rate `1e-4`, weight decay `.01`, gradient clip `1`;
- stage/bridge CE weights `1/2`, orbit KL weight `1`, temperature `2`;
- four slots, source-read-15, first16, Cartesian K4/V4, delta layer 15;
- seed 2027 and the existing deterministic document schedule;
- only the existing rank-16 block-14 boundary adapter is trainable.

There is no early stopping, checkpoint selection, second seed, or change to
loss weights, rate, slots, layer, rank, precision, or topology.  The trainer's
mandatory evaluation uses the already opened alignment rows 0--15 only as a
completion smoke test and cannot select a checkpoint.

## Evaluation strata

### Opened in-domain characterization

MuSiQue `alignment_confirm96` rows 16--95 may be used to characterize target
versus canonical behavior, but they have been used by earlier codec studies
and are not confirmation evidence for this method.

### New external RAG confirmation

Use every one of the 150 rows in
`/data/datasets/longbench/multifieldqa_en.jsonl`, SHA256
`0aac182fd317dcf6d74f8e1e0f3e61029407435346c2e0b3ff9fb45ae49c5c3f`.
This long-document QA dataset has not previously been evaluated anywhere in
the project and is disjoint from capsule training by benchmark construction.
Before freezing this protocol, only schema, row count, context-length
statistics, ID equality, and file hashes were inspected.  One raw row was
accidentally printed by an over-broad structural `rg`; no model output or
aggregate label metric was observed.  Therefore this is a preregistered new
external evaluation, not advertised as a perfectly sealed dataset.

The `_e` file contains the same 150 ID/question/answer/context tuples in a
different line order.  It is not an independent dataset and will not be run or
counted as replication.

## Frozen arms

For both receiver protocols (`question_conditioned` primary and
`state_readout` secondary), run all 150 cases with the real packet codec:

1. parent checkpoint, target-frame K4/V4 base+delta;
2. parent checkpoint, canonical-frame K4/V4 base+delta;
3. final orbit candidate, target-frame K4/V4 base+delta;
4. final orbit candidate, canonical-frame K4/V4 base+delta.

Each evaluator invocation also produces no-state, one-case shifted-source, and
generated-answer Tail-KV controls.  The primary capsule comparisons use
`question_conditioned`; no arm may be selected using `state_readout`.

## Statistics and decision gates

Use paired 20,000-replicate bootstrap intervals with seed 2027.  F1 is primary
and EM is reported.  Let `P_t`, `P_c`, `O_t`, and `O_c` denote parent target,
parent canonical, orbit target, and orbit canonical per-case F1.

The final method is safe for external use only if all of the following pass:

1. **Optimization/integrity:** mean orbit KL over steps 981--1000 is at least
   20% below steps 1--20; all values are finite; exactly 131,072 trainable
   parameters change; every frozen-state hash is unchanged.
2. **Target quality retention:** `mean(O_t - P_t) >= -.01` and its paired 95%
   lower bound is greater than `-.03`.
3. **Canonical safety:** `mean(O_c - O_t) >= -.01` and its paired 95% lower
   bound is greater than `-.03`.
4. **Source causality:** the paired 95% lower bound of orbit-canonical correct
   source minus its shifted-source arm is positive.
5. **Wire invariants:** every row reports 67,642-B base, 4,252-B incremental
   delta, identical canonical base/delta hashes across the two receiver
   protocols, and no target-dependent packet bytes.

The stronger mechanistic claim that orbit distillation *repairs* canonical
behavior additionally requires the paired difference-in-differences

\[
  D = (O_c-O_t) - (P_c-P_t)
\]

to have a positive mean and positive 95% lower bound.  If the safety gates pass
but this difference-in-differences gate fails, the result is only a safe
fixed-rate candidate, not evidence that the orbit objective caused a general
repair.  No failed gate may be rescued by the opened MuSiQue rows or by an
additional training/evaluation variant.
