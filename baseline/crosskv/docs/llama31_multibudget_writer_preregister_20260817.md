# Llama-3.1 multi-budget writer preregistration

Status: frozen before constructing the new development set and before either
new full training run.

## Question

N=512 improves the all-layer state but makes a post-hoc last16 packet worse,
while fresh evidence identifies first16 as the useful compact functional
route.  Test whether exposing one writer to multiple packet budgets during
training improves first16 INT4 utility without destroying its all-layer state.

This is a method-development experiment, not a new sealed paper test.  Its
purpose is to decide whether stochastic budget training deserves a later
confirmatory run.

## Fresh development set

Construct exactly 128 MuSiQue handoff cases from the official train split with
seed 2061.  Explicitly exclude the union of:

- the N=512 training set;
- the old ID-disjoint test256;
- the fresh first16 confirm256.

These inputs contain 1,024 unique case IDs by construction.  Require 128 unique
new IDs, zero overlap with their union, and record the output and ordered-ID
SHA256 values before any model evaluation.  The untouched `2wikimqa_e` file
remains sealed and cannot be inspected in this experiment.

## Fixed candidates

Use frozen BF16 Llama-3.1-8B and the exact N=512 writer setup: four slots,
rank-4 scale-1 sender adapter, 1,600 AdamW updates, learning rate `5e-4`, weight
decay `.01`, stage weight 1, bridge weight 2, seed 2027, resident detached
source caches on CPU, and final checkpoint only.

Evaluate three writers:

1. existing N512 baseline, trained with all layers and BF16 receiver losses;
2. quant-aware control, newly trained with all-layer INT4 STE on every update;
3. multi-budget candidate, newly trained with INT4 STE and a uniform independent
   per-update draw from `first_16` and `all`.

Candidates 2 and 3 use identical initialization, case order, optimizer, and
compute.  Record the realized multi-budget draw counts.  Do not tune mixture
weights, add slots, alter rank, stop early, or select checkpoints after seeing
the development set.

## Evaluation and gates

For all three final checkpoints, evaluate correct-source `first_16 + INT4` and
`all + INT4` on all 128 new cases.  Also run shift-1 source assignment for the
multi-budget candidate.  Stage F1 is primary; report EM, bridge readout, no
summary, payload, and both ordinary and bridge-cluster bootstrap intervals.

The multi-budget method passes only if all three conditions hold:

1. first16 F1 versus the quant-aware all-only control improves by at least
   `.03` and the paired 95% CI lower bound is positive;
2. all-layer F1 versus the quant-aware all-only control is non-inferior with a
   `-.05` margin, meaning the paired 95% CI lower bound is greater than `-.05`;
3. multi-budget correct first16 beats its shift-1 control with a positive paired
   F1 CI lower bound.

Comparing the quant-aware all-only control with the historical BF16-trained
baseline is secondary and isolates quantization-aware training from
multi-budget exposure.  If gate 1 fails, do not proceed to a sealed dataset. If
gate 1 passes but gate 2 fails, report a compact/full quality tradeoff rather
than a generally improved writer.

## Frozen construction audit

The builder produced 128 unique cases from 615 eligible rows remaining after
excluding exactly 1,024 unique used IDs.  The frozen dataset is
`datasets/musique_handoff/budgetdev128_seed2061_exclude_used1024.jsonl`.

- ordered case-ID SHA256:
  `c5387a996a2aa277a2742361b0d63c2e4d90b5bdccbfc97fd3c6483a92e18d45`;
- file SHA256:
  `1d0c6e99abd163ce9e1223ecbc43fddf1a9f392e15ea14f8b1221d441c68785d`;
- mean Agent-A dossier length: 24,299.70 characters;
- relation clusters: 13.

This audit was recorded before launching either candidate or evaluating the
historical checkpoint on the new development set.

## Frozen post-result mechanism follow-up

The naive multi-budget candidate fails: relative to the quant-aware all-only
control, first16 F1 changes by `-.100` and all-layer F1 by `-.124`, with ordinary
bootstrap CIs excluding zero.  Correct first16 still beats shift by `+.334`, so
the failure is representation quality rather than loss of source causality.
Training traces expose a specific conflict: first16 is forced to minimize the
same lexical bridge CE as all layers even though prior depth interventions show
that lexical readout requires deeper KV.

Before changing code or starting another run, authorize exactly one exploratory
mechanism correction on this already-open development set:

- retain uniform `first_16`/`all`, INT4 STE, seed, initialization, 1,600 steps,
  and every other setting;
- use bridge weight 0 on first16 updates and bridge weight 4 on all-layer
  updates; the expected bridge weight remains 2 because each budget has
  probability one half;
- retain stage weight 1 for both budgets;
- evaluate first16/all correct-source and shift-1 exactly as above.

Use the original three gates against the same quant-aware all-only control.  If
this functional/lexical objective still fails, perform no further writer
variant on `budgetdev128`.  If it passes, the result remains exploratory and
must be frozen on a new dataset before becoming paper evidence.
