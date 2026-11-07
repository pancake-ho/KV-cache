# Canonical-RoPE packet preregistration (2026-08-18)

## Motivation

The target-frame sidecar experiment proves that a task delta can be encoded
independently, but it rotates Keys to the receiver's absolute positions before
quantization.  Two agents with different system-prompt lengths therefore do
not receive byte-identical base packets, weakening the base-cache-hit claim.

This follow-up tests a receiver-independent wire frame.  Sender Keys are moved
from their source positions to canonical positions `0..slots-1`, base and delta
are independently encoded there, and the receiver applies RoPE from canonical
positions to its own target positions after decoding:

\[
 C_s \xrightarrow{R(0)R(p_s)^{-1}} C_0
 \xrightarrow{Q_4,\ wire,\ Q_4^{-1}} \widehat C_0
 \xrightarrow{R(p_t)R(0)^{-1}} \widehat C_t.
\]

No checkpoint is retrained and no packet field depends on `p_t`.

## Frozen experiment

- Primary checkpoint: frozen-base rank 16; base and layer-15 delta are exactly
  those of the target-frame sidecar experiment.
- Llama-3.1-8B-Instruct, four slots, first16, real INT4/FP16-scale codec.
- MuSiQue alignment-confirm96 and opened 2WikiMQA-200, identical prompts,
  greedy decoding, controls, and 24-token limit.
- Direct-task canonical packets and base-plus-delta canonical packets are both
  evaluated.  Their references are the corresponding already frozen
  target-frame real-codec runs.
- 20,000 paired bootstrap replicates; fixed bridge clusters on MuSiQue.

## Gates

1. **Target invariance.**  For a fixed document and layer set, canonical base
   and delta packet bytes must be identical when consumed after the two
   different MuSiQue receiver system prefixes.  Decoded canonical tensors must
   also be identical before receiver-side repositioning.
2. **Direct-packet safety.**  Canonical direct minus target-frame direct F1
   must have point delta at least `-.01` and 95% lower bound above `-.03` on
   MuSiQue and 2Wiki question-conditioned.
3. **Composable-packet safety.**  Canonical base-plus-delta minus target-frame
   base-plus-delta must satisfy the same point and lower-bound margins on both
   datasets.
4. **Causality.**  Correct-source canonical composition must remain above the
   shifted-source arm on both datasets; the MuSiQue ordinary and cluster lower
   bounds and the 2Wiki paired lower bound must be positive.

State-readout, exact-match, NRMS, real framed bytes, and packet preparation time
are mandatory diagnostics but do not replace the gates.

## Decision

Passing establishes that the reusable base is a document packet rather than a
receiver-position-specific cache fragment.  Failure limits base hits to agents
sharing one target position and forbids a general multi-agent fan-out claim.
