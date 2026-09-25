# Generated Tail-KV token-by-depth factorial

Status: frozen on 2026-08-18 after the first16 capacity diagnostic and learned
refinement negatives, before any all-layer cap-4/8/16 output is observed.

This is an opened MultifieldQA mechanism/system characterization, not external
confirmation.  It completes the Cartesian product of generated answer caps
`{4,8,16}` and transmitted layers `{first16,all32}` under canonical real K4/V4.
The existing cap-24/all32 run is a fixed high-budget reference.

Primary endpoint: question-conditioned generated-tail F1 on all 150 paired
cases.  State-readout is secondary.  Use 20,000 paired bootstrap replicates,
seed 2027.  Report realized tail positions and framed bytes.

The cap-8/all32 operating point is authorized for the paper frontier only if:

1. versus cap-24/all32, mean F1 delta is at least `-.03` and paired CI lower is
   greater than `-.06`;
2. versus its own no-state arm, mean F1 delta and paired CI lower are positive;
3. mean packet bytes are at most 60% of cap-24/all32.

All32-versus-first16 effects at each cap, and cap4/8/16 scaling within all32,
are reported without selection.  Passing does not make the state pre-answer:
the sender still decodes a bounded internal answer prefix.  Failure prevents
using cap-8/all32 as a preferred operating point and cannot be rescued by a
new token cap.
