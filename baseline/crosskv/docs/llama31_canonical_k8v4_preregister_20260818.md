# Canonical Cartesian K8/V4 packet preregistration (2026-08-18)

## Motivation and fixed intervention

Canonical Cartesian K4/V4 packets already make each document's bytes exactly
independent of receiver-prefix length.  Their MuSiQue safety gates pass, but
the two opened 2Wiki safety lower bounds (`-.03323` and `-.03333`) narrowly miss
the frozen `-.03` margin.  The equal-byte polar repair failed on untouched
rows, so no further polar bit split is permitted.

This experiment tests one mechanistic hypothesis: quantizing a canonical Key
before its receiver-specific RoPE rotation needs more angular resolution than
quantizing a Value.  The sole candidate is therefore:

- Cartesian Key: symmetric signed INT8;
- Value: the existing symmetric signed INT4;
- one transmitted FP16 max-absolute scale per layer/head/slot vector;
- canonical Key positions 0--3 on the wire, followed by receiver-side RoPE;
- unchanged four slots, first16 layer set, layer-15 delta topology, headers,
  layer indices, CRC, model, checkpoints, prompts, and greedy 24-token decode.

There is no K-bit scan, Value change, layer/slot change, polar variant, or
checkpoint selection.  The candidate is called **K8/V4 canonical**.

## Frozen rate

For Llama-3.1-8B (8 KV heads, head dimension 128, four slots):

| component | K4/V4 framed bytes | K8/V4 framed bytes |
|---|---:|---:|
| first16 base | 67,642 | 100,410 |
| layer-15 delta | 4,252 | 6,300 |
| cold base + delta | 71,894 | 106,710 |

K8/V4 increases cold bytes by 48.43%, but its 6,300-B delta remains 6.27% of
the base.  A behavioral pass would define a receiver-independent operating
point near 100 KB; it would not be described as an equal-rate repair.

## Data boundary

### Disclosed mechanism screen

Rows 0--15 of the already opened LongBench 2WikiMQA file may be used only to
check that K8/V4 moves canonical behavior toward the target-frame reference.
Those rows are excluded from every confirmatory interval and paper number.

### New confirmation set

- all 200 rows of `/data/datasets/longbench/musique.jsonl`;
- SHA256
  `4ac69b91281c4ec6b21316cb7282e83fb6b4dda04fc68480bb8d8ed1e19ff7bd`;
- the path has no reference in existing experiment configs, outputs, or docs
  before this preregistration;
- the file contents and model outputs were not inspected before freezing this
  protocol.

After freezing, IDs will be audited against all local checkpoint-training and
previous MuSiQue confirmation manifests.  Exact overlap with checkpoint
training IDs invalidates the set as confirmation.  Same benchmark family is
intentional: this set tests a wire codec, not a new cross-task-generalization
claim.

### Post-freeze metadata audit (before implementation or model execution)

Normalized exact-question overlap is zero with train128, train512,
increment256, and the recent 96/128/256 confirmation manifests.  It is also
zero with the older test256/budgetdev128 manifests.  Thirty-two of 200 questions
match the earliest `dev197` exploratory file, although LongBench supplies a
different full-context representation and K8/V4 has never been run.  All 200
rows remain fixed because the candidate was chosen before this audit and the
primary inference is a within-row paired codec contrast; the set will be
described as checkpoint-training-disjoint, not as wholly unseen research data.

## Fixed arms

Every confirmation row uses the same frozen rank-16 task checkpoint and
unrestricted base checkpoint.

1. Cartesian K4/V4 target-frame base+delta (behavioral reference).
2. Cartesian K4/V4 canonical base+delta (rate-matched canonical reference).
3. Cartesian K8/V4 canonical base+delta (sole candidate).
4. Shift-by-one K8/V4 canonical base+delta (causal control).

Both `state_readout` and `question_conditioned` receiver protocols are run.
Direct rank-16 K8/V4 is a mandatory diagnostic on the 16-row disclosed screen,
not an additional selectable confirmation arm.

## Statistics and gates

All 200 confirmation rows use paired 20,000-replicate bootstrap intervals,
seed 2027.  Because this LongBench file has no predefined bridge clusters,
ordinary case bootstrap is primary.  Question-conditioned F1 is the primary
behavioral metric; EM and state-readout are mandatory diagnostics.

1. **Wire correctness.** Actual K8/V4 sizes must be exactly 100,410-B base and
   6,300-B delta; all candidate base/delta SHA-256 hashes must be identical
   across the two receiver protocols for each source document.
2. **Target-frame safety.** Candidate K8/V4 canonical minus K4/V4 target-frame
   sidecar must have F1 point delta at least `-.01` and paired 95% lower bound
   above `-.03`.
3. **Canonical rate benefit.** Candidate minus K4/V4 canonical must have F1
   point delta at least `-.005` and paired lower bound above `-.03`.  A positive
   lower bound is required for the stronger claim that extra Key bits improve
   quality; non-inferiority alone supports only a safe higher-rate point.
4. **Causality.** Candidate correct-source minus shifted-source F1 must have a
   positive paired lower bound.
5. **Base-hit semantics.** Runtime checks must prove layers 0--14 are the exact
   decoded base, only layer 15 receives a delta, and the incremental packet is
   no more than 6.30% of the base packet.

Failure is reported without widening the margin, adding cases from opened
sets, changing the precision, or promoting the 16-row screen.  If gate 2
fails, canonical transport remains a mechanism result rather than a safe paper
operating point.
