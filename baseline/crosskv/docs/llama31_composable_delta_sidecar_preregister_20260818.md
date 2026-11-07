# Composable KV delta sidecar preregistration (2026-08-18)

## Question

The frozen-base boundary experiments established that the task intervention is
serialized only in layer 15 of a first16 packet, but all reported evaluations
quantized the already-merged task packet.  This experiment asks whether the
same behavior survives the operationally useful representation

\[
  Q_4(C_{base}^{0:15}) + Q_4(C_{task}^{15}-C_{base}^{15}),
\]

where the base and task delta are independently quantized and independently
framed.  No model is retrained.

## Frozen intervention

- Model: Llama-3.1-8B-Instruct, BF16 execution.
- Candidate: the already frozen rank-16 boundary checkpoint.  Rank 4 is a
  secondary operating-point audit only; it cannot replace a failed primary.
- Base: the exact unrestricted checkpoint used to initialize the boundary run.
- Four slots, active layers 0--15, delta layer 15 only.
- Both packets use the real signed-INT4/FP16-scale codec, including headers,
  layer indices, and checksums.  Direct-task controls also use that real codec.
- Delta is computed after source materialization and RoPE repositioning, in
  FP32 from the BF16 task/base tensors, then independently quantized.
- MuSiQue alignment-confirm96 and opened 2WikiMQA-200 use their already fixed
  slices, prompts, greedy decoding, and 24-token limit.
- Statistical comparisons use 20,000 paired bootstrap replicates.  MuSiQue
  additionally uses the fixed bridge-cluster bootstrap.

## Two primary gates

1. **Composable fidelity.**  On MuSiQue first16, composed rank-16 minus the
   directly encoded rank-16 packet must have F1 point delta at least `-.01` and
   ordinary and bridge-cluster 95% lower bounds above `-.03`.  On 2Wiki
   question-conditioned, the point delta must be at least `-.01` and the paired
   95% lower bound above `-.03`.  Both datasets must pass.  This compares two
   wire representations of the same frozen model, not against an FP baseline.
2. **Exact fallback and bounded sidecar.**  Omitting the sidecar must feed the
   byte-identical serialized base packet used by composition.  A packet-level
   test must prove identical bytes and decoded tensors for base-only versus the
   composition's base component.  For four Llama slots, the framed sidecar must
   be at most 4,300 B and at most 6.5% of the framed first16 base packet.  The
   cold composed transmission is reported honestly and is not claimed to beat
   a monolithic packet; the systems claim is incremental transfer after a base
   cache hit.

## Mandatory diagnostics

- Layers 0--14 of the composed decoded cache must be bitwise identical to the
  decoded base packet, and inactive layers 16--31 must be exactly zero.
- Report layer-15 normalized RMS error versus direct-task INT4 and versus the
  unquantized task tensor on at least the complete MuSiQue slice.
- Report actual framed bytes separately for base, delta, cold composition, and
  base-hit incremental transmission.
- Preserve the shifted-source causal control.  A fidelity pass cannot be used
  to claim semantic state if correct-minus-shift ceases to be positive.

## Decision rule

Passing both primary gates supports a composable base-plus-task-state interface
and permits latency/cache-hit experiments.  Failure is evidence that the
merged boundary result is not independently packetizable; no alternate rank,
layer, bit width, or delta definition may silently replace the primary result.
