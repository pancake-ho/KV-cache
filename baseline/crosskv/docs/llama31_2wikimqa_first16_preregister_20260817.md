# Llama-3.1 first16 capsule on 2WikiMQA

Status: frozen before any model execution on 2WikiMQA.

## Purpose

Test whether the MuSiQue-trained four-slot state and the confirmed first-half
route transfer to a different multi-hop dataset.  Neither
`/data/datasets/longbench/2wikimqa.jsonl` nor `2wikimqa_e.jsonl` has been used
for writer training, layer selection, validation, or prior model runs in this
workspace.

The selected method is frozen from MuSiQue confirm256: the final unconstrained
N=512 Llama-3.1-8B writer, four slots, rank 4, first 16 of 32 layers, INT4.
No 2Wiki result may change layer pattern, precision, writer, checkpoint, prompt,
or decoding.

## Data and audit

- dataset: all 200 rows of `/data/datasets/longbench/2wikimqa.jsonl` in file
  order;
- file SHA256:
  `dda279cf93a99e1e5bfa3291fb199fd55978d10a1feb31822953cf77a1742e37`;
- 200 unique `_id` values and 200 unique normalized questions;
- mean context length: 29,615 characters.

For parallel execution, first16 may be split into rows 0--99 and 100--199.
Shift-1 is applied within each frozen shard; every source ID must differ from
its target ID.  Merge by original index and require exactly 200 cases for both
receiver protocols.

## Arms and protocols

For each case, Agent A prefills the complete 2Wiki context and question once.
Evaluate:

- no-summary receiver;
- correct first16 INT4 capsule (67,584 B accounting; 67,642 B framed codec);
- shift-1 first16 capsule;
- generated-answer Tail-KV at all 32 layers INT4.

Run both generic state readout and question-conditioned answering.  Generated
Tail-KV is an upper bound/control, not part of the capsule gate.  Record Agent
A source-answer quality, payload, source prefill, four-slot emission, answer
decode, relocation, and receiver generation timings.

In a separate secondary run, evaluate the all-layer INT4 capsule (135,168 B)
under the same protocol to place the first16 and all-layer points on the
cross-dataset quality/payload frontier.  The all-layer result cannot replace a
failed first16 gate.

## Confirmatory gates

Question-conditioned F1 is primary.  First16 passes cross-dataset transfer only
if both paired bootstrap 95% CIs have positive lower bounds:

1. correct capsule versus no-summary;
2. correct capsule versus shifted capsule.

Report EM McNemar statistics and the state-readout contrasts regardless of
outcome.  Also report generated-tail gaps, source-answer-conditioned results,
and the all-layer versus first16 quality/payload tradeoff.  Do not inspect
`2wikimqa_e` under this protocol; it remains available for a later sealed test.

## Post-result reproducibility amendment

The first16 confirmatory gates compare arms materialized within one execution
and are unaffected by cross-run variation.  The secondary first16/all-layer
comparison used separate executions.  A post-result audit found six of 200
source-answer strings differed in surface form across those executions even
though source EM and F1 were identical.

Before interpreting first16 versus all-layer quality, freeze one exact,
full-file repeat of the selected first16 command.  Reproducibility passes only
if both absolute question-conditioned EM and F1 differ from the original by
less than 0.02 and the paired bootstrap 95% CI for repeat-minus-original F1
contains zero.  Report the repeat regardless of outcome.  If this check fails,
make no comparative first16/all-layer quality claim; retain only the original
within-run first16 versus no-summary and shifted-capsule conclusions.
