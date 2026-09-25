# Native Tail-state distillation cap8 scale-up

Status: frozen on 2026-08-18 after the equal-rate four-slot pilot passed all
four gates, before any cap8 training output is observed.

## Objective

Test whether the positive direct-state objective scales with native-state
capacity rather than merely regularizing the existing four-slot writer.

## Frozen arms

All arms start from the same four-slot unrestricted all-layer checkpoint and
preserve its complete rank-4 layerwise writer.  Eight-slot arms deterministically
cycle the four learned embeddings to eight positions before training.  The
different absolute positions and causal mask break equality during the first
forward pass; no random expansion noise is selected.

1. `state4`: four slots, direct canonical K4/V4 state loss weight .1, teacher
   cap 4.
2. `ce8`: eight slots, CE only, teacher cap 8 measured at evaluation.
3. `state8`: eight slots, direct state loss weight .1, teacher cap 8.

Common setup: Llama-3.1-8B-Instruct, MuSiQue increment-train rows 0--127,
800 steps, seed 2095, learning rate `1e-4`, stage/bridge CE weights 1/2,
all32 K4/V4, state layers 1--31, no source-read bottleneck.  Evaluation uses
confirm128 rows 32--95, which are disjoint from training and were not used in
the four-slot pilot decision (but are not globally untouched).

## Evaluation and gates

After training, all three checkpoints receive real framed canonical INT4
correct-source and shift-1 evaluation.  Use 20,000 paired bootstrap replicates,
seed 2096.  A cross-task MultifieldQA run is authorized only if:

1. `state8` held-out state MSE is at most 75% of `ce8`, with the paired
   `state8-ce8` CI upper below zero;
2. `state8-ce8` correct-source final F1 is non-negative and its CI lower is
   greater than `-.03`;
3. `state8-state4` final F1 is at least `+.02` and its CI lower is greater than
   `-.02`, establishing a capacity interaction rather than state-loss-only
   regularization;
4. `state8` correct-minus-shift final F1 has a positive CI lower bound;
5. state8 training state loss over the final 20 steps is at most 70% of the
   first 20-step mean.

Bridge F1 and the fixed eight-slot packet size are reported without selection.
Failure is not rescued by changing the state-loss weight on these cases.
