# Positional-orbit distillation feasibility preregistration (2026-08-18)

## Motivation

Canonical packet bytes are exactly receiver-independent, but behavioral
equivalence is distribution-sensitive.  Two direct codec repairs failed:

- equal-byte polar Keys regress on untouched confirmation rows;
- Cartesian K8/V4 reduces layer-15 tensor NRMS from about `.2056` to `.0300`
  but still fails the external 2Wiki safety gates while adding 48.43% bytes.

The next hypothesis is therefore not insufficient tensor precision.  The
writer was trained under target-frame fake quantization and never optimized
the non-commuting path “quantize once in a canonical frame, then rotate to the
receiver.”  This feasibility study tests whether a fixed-rate layer-15 task
delta can learn to compensate that positional orbit at the **decoder-logit
level**.

## Fixed mechanism

For each training source, materialize the frozen base cache `B` and current
task cache `T`.  For each of the two existing receiver routes `r` (stage and
bridge), construct:

\[
  C_r^{teacher} = Q_4(R_r B) + Q_4(R_r(T-B)),
\]

\[
  C_r^{student} = R_r\left[Q_4(R_0 B) + Q_4(R_0(T-B))\right].
\]

Base layers 0--15 are encoded; the independent delta exists only at layer 15.
Every other runtime layer is zero for the four logical slots.  `Q4` must match
the production codec forward exactly, including transmitted FP16 scales and
BF16 decoded tensors.  Its straight-through backward is identity only through
the layer-15 task delta.

The student receives the ordinary stage/bridge answer CE.  A detached
target-frame teacher supplies:

\[
  L_{orbit}=KL(p(C_r^{teacher})\;||\;p(C_r^{student})).
\]

Only the existing rank-16 block-14 boundary adapter is trainable.  The model,
slot embeddings, layerwise base writer, and independent base checkpoint remain
frozen.  Wire format stays Cartesian K4/V4: 67,642-B base and 4,252-B delta.

## Implementation gates before GPU training

1. The differentiable forward must be bitwise equal to
   `compose_int4_base_delta` for BF16 Cartesian K4/V4 inputs.
2. Zero delta must reconstruct the decoded base exactly.
3. Backpropagation must give gradients only to task tensors in declared delta
   layers; base and non-delta task tensors must have no gradient.
4. Continuing an existing boundary checkpoint with frozen initialized base
   must expose exactly 131,072 trainable parameters.
5. Existing production packet tests and the full repository suite must remain
   green.

## Fixed pilot (development evidence only)

- Initialization: existing frozen-base rank-16 checkpoint.
- Base: unrestricted `llama31_increment256_allonly_cont3200_confirm128`.
- Train data: the same opened `train256_increment` used by rank 16.
- Training: 200 continuation steps, AdamW, learning rate `1e-4`, weight decay
  `.01`, stage/bridge CE weights `1/2`, orbit-KL weight `1`, temperature `2`,
  first16 only, real-scale K4/V4 STE, seed 2027.
- Source caches: CPU-resident, as in the parent run.
- Development evaluation: alignment-confirm rows 0--15 only, which are already
  opened; target-frame parent and canonical parent outputs are fixed
  references.

Pilot feasibility gates:

1. Mean orbit KL over steps 181--200 is at least 20% lower than steps 1--20.
2. All losses and gradients remain finite; the final boundary parameters
   differ from initialization, while frozen state hashes do not change.
3. On the 16-row development screen, canonical first16 F1 is no more than
   `.05` below the parent canonical F1 and correct-source performance remains
   above shifted source.
4. Actual evaluation packets remain 67,642/4,252 B and target-invariant.

Passing this pilot authorizes one separately preregistered full continuation
and confirmation.  It is not itself evidence for an ICLR claim.  Failure is
reported without an orbit-weight, learning-rate, bit-width, layer, or slot
scan.
