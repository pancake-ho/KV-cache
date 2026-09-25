# Llama-3.1 source-read bottleneck development protocol

Status: frozen after implementation/unit/2-step smoke tests and before the
full candidate run.  This is a development follow-up on the already opened
MuSiQue increment256/confirm128 split, not a new sealed test.

## Motivation and intervention

Three ways of optimizing a shared first16/all packet have failed: stochastic
receiver-depth training, depth-conditioned loss weights, and a layer-16 hard
sender/receiver gradient boundary.  The sealed 2WikiMQA-E result additionally
shows that ordinary full-depth data scaling improves all-layer transfer more
reliably than first16 transfer.

The new candidate changes where the sender may acquire information, not which
receiver loss supplies gradients.  Capsule slots may attend the complete long
source cache in decoder layers 0--15.  In layers 16--31, source-key columns are
masked and the slots may attend only their own causal four-position submatrix.
Hidden states are not detached.  All-layer stage and bridge losses therefore
backpropagate through every deep block, every early block, the writer adapter,
and the slot embeddings.  Any information present in deep emitted K/V must
first cross the layer-16 four-slot hidden-state bottleneck.

This is called the **source-read-16 bottleneck**.  It is not layer dropping at
the receiver and does not change the first16 wire format.

## Fixed comparison

Candidate and control use exactly:

- parent checkpoint:
  `outputs/amortized_semantic_kv/llama31_n512_allonly_int4_budgetdev128_fixed1600_20260817/capsule.pt`,
  SHA256 `b5848635fac01561b8f694aeca79d5d02d340a5c15eeb1e3967d21c90cb0b620`;
- train256 SHA256
  `c0c5ec89ffe569fef6b94e387c1b4ceb42e271787cc0686a0cc249d6dc4ffa1b`;
- confirm128 SHA256
  `1900657b9479fcb080145e8c489b5fd478234d69d1d62f5a5797a73b5e44abef`;
- Llama-3.1-8B-Instruct, four slots, rank-4 writer, seed 2027;
- 3,200 updates, learning rate `5e-4`, weight decay `.01`, gradient clip 1;
- all-layer-only receiver training, stage weight 1, bridge weight 2;
- INT4 straight-through packet during training;
- first16 and all-layer INT4 evaluation on all 128 confirmation cases.

The existing ordinary all-only continuation is the compute-matched control:
`outputs/amortized_semantic_kv/llama31_increment256_allonly_cont3200_confirm128_20260818/`.
The candidate changes only `source_read_layers` from unrestricted to 16.

## Development gates

Use paired 20,000-replicate ordinary and bridge-answer-cluster bootstrap
intervals, seed 2027.

1. **Compact concentration:** candidate first16 stage F1 minus control first16
   is at least `+.03`, with both ordinary and cluster CI lower bounds above 0.
2. **Full-state safety:** candidate all-layer stage F1 minus control all-layer
   is at least `-.03`, and both CI lower bounds are above `-.08`.
3. **Source causality:** candidate correct-source first16 exceeds its circular
   shift-1 first16 by at least `+.10` F1 with a positive ordinary CI lower
   bound.

The mechanism is promising only if all three gates pass.  If compact
concentration fails, masking deep source access does not force the optimizer to
place more usable state in early emitted K/V.  If full-state safety fails, four
hidden positions at layer 16 are too narrow or the transition is too abrupt.
No threshold or layer boundary will be changed after seeing this run.

One secondary, non-selective intervention diagnostic is fixed before candidate
evaluation: apply `source_read_layers=16` to the unchanged parent checkpoint on
the same confirm128.  Its first16 rows must match the ordinary parent exactly,
because the intervention begins after layer 15.  Its all-layer result measures
the immediate cost of removing deep source reads before adaptation.  This
diagnostic cannot satisfy or alter any gate.

## Pre-run implementation evidence

- focused unit suite: `20 passed`;
- exact early invariant on a four-layer Qwen fixture: layers before the mask
  match unrestricted materialization elementwise;
- deep-loss gradient test: embeddings and every early/deep writer block receive
  finite nonzero gradients;
- real Llama-3.1 two-step train/evaluate smoke completed with first16/all INT4
  outputs and a checkpoint that restores `source_read_layers=16`.
