# Frozen-base boundary rank-16 follow-up preregistration (2026-08-18)

## Motivation and single permitted change

The rank-4 frozen-base residual passed cross-task retention and significantly
repaired the full-parameter aligned model, but missed the in-domain
significance/safety gates.  Its final task loss (`.9123`) remained well above
the full-parameter aligned run (`.5724`).  This fixes the next hypothesis to
insufficient task-residual capacity, not base forgetting or boundary placement.

One development follow-up is permitted.  The sole change is boundary residual
rank 4 to rank 16 (32,768 to 131,072 trainable sender parameters).  The frozen
base, four slots, block-14 boundary, source-read-15 topology, first16 INT4
67,584-B payload, train/eval data, all-layer loss, weights, optimizer, learning
rate 5e-4, weight decay 0.01, 3,200 steps, seed, and final-checkpoint policy are
identical.  No rank scan or intermediate checkpoint selection is allowed.

The rank affects only sender-side training/materialization parameters.  It adds
no KV slots, receiver operation, or wire bytes.  K/V layers 0--14 must remain
bitwise equal to the unrestricted base after training.

## Opened development endpoints

- MuSiQue alignment-confirm96, first16 and all INT4, plus source shift by one.
- LongBench 2WikiMQA examples 0--199 under the fixed question-conditioned
  first16/all protocol, plus the within-run shifted state.
- Existing unrestricted, rank-4 frozen-base, full-parameter aligned, and
  pre-answer-KL outputs are frozen references.
- No sealed `_e` data may be opened for this follow-up.

All comparisons use 20,000 paired bootstrap replicates.  MuSiQue additionally
uses the fixed 64 bridge-answer clusters.

## Gates

1. Rank-16 minus unrestricted MuSiQue first16 F1 is at least +.04 and has
   positive ordinary and cluster confidence bounds.
2. Rank-16 minus full-parameter aligned MuSiQue first16 is at least -.05 and
   both lower bounds exceed -.08.
3. Rank-16 minus rank-4 MuSiQue first16 is at least +.02 with a positive
   ordinary confidence bound, directly confirming the capacity hypothesis.
4. On 2Wiki first16, rank-16 minus unrestricted is at least -.03 with lower
   bound above -.08, and rank-16 minus full-parameter aligned has a positive
   lower bound.
5. Correct minus shifted first16 has positive ordinary confidence bounds on
   both opened datasets.
6. Full-packet rank-16 minus unrestricted has lower confidence bound above
   -.10 on each dataset; this remains a separate safety diagnostic.

The follow-up is a successful compact optimizer only if gates 1--5 pass.  Gate
3 specifically decides whether increasing the isolated residual capacity was
the right diagnosis.  Failure is reported without another rank or learning-rate
attempt.
