# Multi-domain decision-supervised CE8 writer results

Date: 2026-08-18.  This report applies the unchanged gates in
`llama31_multidomain_decision_writer_preregister_20260818.md`.

## Decision

The candidate fails three of four frozen development gates.  No HotpotQA-E,
2WikiMQA-E, Qasper, or NarrativeQA evaluation is authorized.

The eight-slot all32 K4/V4 candidate was initialized from CE8 and continued for
1,600 steps on 312 standard-LongBench-derived HotpotQA/2WikiMQA handoffs plus
256 replayed MuSiQue handoffs.  Its wire format remains 270,426 bytes.

## Frozen results

On the 56-case standard-LongBench development set:

| arm | final F1 | bridge F1 |
|---|---:|---:|
| parent, correct source | .11224 | .23082 |
| candidate, correct source | .16071 | .13449 |
| candidate, shift-1 source | .10714 | .00468 |

Candidate-minus-parent final F1 is `+.04847`, CI
`[-.07143,+.16071]`; bridge F1 is `-.09633`, CI
`[-.18225,-.01792]`.  Candidate correct-minus-shift bridge F1 is
`+.12982`, CI `[+.08717,+.17614]`, but final F1 is only `+.05357`, CI
`[-.05357,+.16071]`.

On the 32-case MuSiQue safety slice:

| arm | final F1 | bridge F1 |
|---|---:|---:|
| parent | .92188 | .89063 |
| candidate | .88902 | .29424 |

Candidate-minus-parent final F1 is `-.03286`, CI
`[-.11155,+.03125]`.  Bridge F1 falls by `-.59639`, CI
`[-.66063,-.52895]`; all 32 cases are losses.

| frozen gate | decision |
|---|---|
| dev bridge improvement | fail |
| dev final non-inferiority | pass |
| dev correct-source causality for bridge and final | fail |
| MuSiQue final/bridge safety | fail |

## Interpretation

The continuation learns some source-dependent bridge information on the new
handoff distribution, but its final-task gain over the parent is unresolved and
its reusable lexical/bridge channel is overwritten.  Replaying 256 original
MuSiQue cases is insufficient: task diversity plus direct stage/bridge CE does
not yield a task-invariant eight-slot semantic state.

This is not evidence that eight slots lack capacity.  The unchanged parent is
already strong on MuSiQue.  It instead shows that updating a single shared
embedding-plus-layerwise writer makes semantic retention and new decision
learning interfere globally.  A further mixture, step, or loss-weight scan is
not justified by this result.

## Artifact

`outputs/amortized_semantic_kv/llama31_ce8_multidomain_decision_cont1600_train568_dev56_20260818/frozen_development_analysis.json`
