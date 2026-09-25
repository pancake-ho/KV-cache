# Equal-byte RoPE-polar Key codec preregistration (2026-08-18)

## Fixed mechanism

Cartesian INT4 canonical packets missed the 2Wiki safety lower bound by about
`.0033`.  The fixed repair changes only Key encoding.  Each rotate-half pair
`(x_i, x_{i+d/2})` still occupies one byte:

- 3 unsigned bits encode radius, using one transmitted FP16 max-radius scale
  per layer/head/token vector;
- 5 bits encode phase on 32 uniform bins;
- Values retain the existing symmetric signed INT4 codec;
- headers, layer indices, checksums, active layers, slots, and total packet
  bytes are unchanged.

The 3/5 split was selected analytically before behavioral screening: radius has
at most about 7.1% half-step relative error at the vector maximum and phase has
at most 5.625 degrees (about 9.8% tangential error), balancing the two axes.
No alternate split will be tested in this experiment.

## Disclosed development screen

Rows 0--15 of the already opened MuSiQue alignment set and 2Wiki set are a
mechanism screen and excluded from confirmatory statistics.  The screen showed
no MuSiQue change relative to Cartesian canonical and a `+.0028` 2Wiki
question-F1 change.  It only authorized the fixed follow-up; it is not evidence
for a paper claim.

## Frozen confirmation sets

- MuSiQue alignment rows 16--95 (`n=80`).
- 2WikiMQA rows 16--199 (`n=184`).
- Same rank-16 checkpoint, base checkpoint, layer-15 delta, first16 packet,
  four slots, BF16 execution, greedy decoding, prompts, and 24-token limit.
- Real packet codec only.  Cartesian target-frame and Cartesian canonical
  results already exist for every confirmation row.
- 20,000 paired bootstrap replicates; MuSiQue also uses fixed bridge clusters.

## Gates

1. **Equal bytes and target invariance.**  Polar base/delta framed sizes must
   remain 67,642/4,252 B, and per-document packet hashes must remain identical
   across receiver protocols/positions.
2. **2Wiki repair versus target frame.**  On rows 16--199,
   polar-canonical sidecar minus Cartesian target-frame sidecar must have point
   F1 at least `-.01` and paired 95% lower bound above `-.03` for
   question-conditioned evaluation.
3. **No regression versus Cartesian canonical.**  On both confirmation sets,
   polar minus Cartesian canonical sidecar must have point F1 at least `-.01`
   and 95% lower bound above `-.03`; the MuSiQue cluster lower bound must also
   exceed `-.03`.
4. **Causality.**  Correct-source polar sidecar must exceed shifted-source
   polar sidecar with positive ordinary/cluster lower bounds on MuSiQue and a
   positive paired lower bound on 2Wiki question-conditioned.

Direct packets and state-readout are mandatory diagnostics.  A failure cannot
be replaced by another bit split, wider Key payload, threshold change, or use
of the 16 disclosed screen rows.
