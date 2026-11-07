# Causally nested 4+4 KV capsule: feasibility protocol

Status: frozen on 2026-08-18 after the generated-Tail capacity gates and a
2-step plumbing smoke test, before the 200-step pilot below is run.

This is an opened-development feasibility experiment.  MultifieldQA has
already been fully evaluated and diagnosed, so none of these rows can become
external evidence.  Qasper and NarrativeQA remain untouched.

## Fixed representation

- frozen four-slot core:
  `llama31_r16_orbitdistill_full1000_20260818/capsule.pt`;
- append four trainable refinement slots in a second causal materialization;
- refinement attends to the source and exact core cache;
- reuse all frozen core writers and add one rank-16 boundary adapter after
  block 14 for refinement tokens only;
- first16 layers, canonical positions, production-equivalent Cartesian K4/V4;
- core and refinement are separately framed 67,642-B packets.

The core materialization is reused as the refinement prefix, rather than
recomputed.  A completed real H200 probe already establishes: identical core
tensor storage when passed into the full path, bitwise-identical core values
and packet hash, separately decoded core+refinement exactly equal to a decoded
monolithic eight-slot packet, 135,284-B split cold payload versus 135,226 B
monolithic, and 67,642-B incremental refinement.

## Fixed pilot

- model: Llama-3.1-8B-Instruct;
- opened development data: MultifieldQA rows 0--31 for training and 32--47 for
  final evaluation;
- exactly 200 AdamW updates, seed 2027, LR `1e-4`, weight decay `.01`, clip 1;
- teacher-forced CE on the first gold answer under the question-conditioned
  receiver;
- final checkpoint only, 24 generated evaluation tokens;
- only 4x4096 refinement embeddings plus the rank-16 boundary adapter are
  trainable: exactly 147,456 parameters;
- no checkpoint selection, early stop, second seed, rank scan, slot scan, or
  loss-weight scan.

Evaluation arms are frozen core-4, full core+refinement-8, shift-1 full-8, and
no state.  Statistics use paired 20,000-replicate bootstrap intervals with
seed 2027.

## Authorization gates

The pilot authorizes one larger opened-development run only if all gates pass:

1. exactly 200 finite updates; mean loss over steps 181--200 is at least 20%
   below steps 1--20;
2. exactly 147,456 trainable parameters change and the complete frozen-core
   parameter hash is unchanged;
3. full-8 minus core-4 mean F1 is at least `+.02` and its paired 95% lower
   bound is greater than `-.05`;
4. correct full-8 minus shift-1 full-8 has a positive mean and positive paired
   95% lower bound;
5. every evaluation case carries exactly 67,642-B core and 67,642-B
   refinement packets, with different source IDs under the shift control.

Failure is retained as evidence against this minimal refinement architecture.
It cannot be rescued by changing the threshold or by the previously opened
generated-Tail oracle.  Passing is only evidence that learned refinement is
feasible on opened data; it does not authorize an external claim.
