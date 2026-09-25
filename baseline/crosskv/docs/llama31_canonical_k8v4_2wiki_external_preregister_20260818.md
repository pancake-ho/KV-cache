# K8/V4 canonical external 2Wiki diagnostic preregistration (2026-08-18)

## Authorization

The fixed K8/V4 canonical candidate passed all five gates on the
checkpoint-training-disjoint LongBench MuSiQue-200 confirmation.  It was not
significantly better than K4/V4 canonical, but unlike K4/V4 it retained a
strictly positive correct-source causal lower bound.  No code, checkpoint,
prompt, precision, or threshold changes are authorized here.

This follow-up asks whether the same frozen candidate repairs the exact 2Wiki
failure that motivated it.  It is an external diagnostic on opened data, not a
second sealed confirmation.

## Fixed data and arms

- LongBench 2WikiMQA rows 16--199 (`n=184`).
- Rows 0--15 were the disclosed mechanism screen and are excluded.
- Existing K4/V4 target-frame sidecar and K4/V4 canonical sidecar outputs are
  immutable references.
- Run one K8/V4 canonical sidecar job, including the shift-by-one control and
  both receiver protocols.
- Same rank-16 task checkpoint, unrestricted base, four slots, first16 base,
  layer-15 delta, BF16 model, greedy 24-token decoding, and real packet codec.

## Frozen gates

All comparisons use rows 16--199 and 20,000 paired bootstrap replicates.

1. Actual base/delta bytes are 100,410/6,300 and hashes are identical across
   receiver protocols for all documents.
2. K8/V4 canonical minus K4/V4 target-frame question F1 has point delta at
   least `-.01` and lower bound above `-.03`.
3. K8/V4 canonical minus K4/V4 canonical has point delta at least `-.005` and
   lower bound above `-.03`.  A positive lower bound is separately required to
   claim significant Key-rate benefit.
4. Correct K8/V4 source minus shifted K8/V4 source has a positive
   question-F1 lower bound.

Failure remains a reported task-specific boundary.  Rows 0--15, state-readout,
or the passed MuSiQue result cannot replace a failed question-conditioned
2Wiki gate.
