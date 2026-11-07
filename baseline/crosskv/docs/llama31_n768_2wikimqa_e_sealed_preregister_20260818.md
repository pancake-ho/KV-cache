# Llama-3.1 continued-writer sealed 2WikiMQA-E protocol

Status: frozen before reading, hashing, counting, or executing on
`/data/datasets/longbench/2wikimqa_e.jsonl`.

## Purpose

The continued all-only writer was selected by disjoint MuSiQue
train256/confirm128.  It improves parent first16 F1 by `.098` on sealed
MuSiQue confirm128.  On the already-open 2WikiMQA-200 set, first16 improves by
`.051` but its F1 CI narrowly crosses zero, while all-layer improves by `.136`
with a positive CI.  Use the previously untouched 2WikiMQA-E file for one
sealed test of compact scaling and source causality.

No result from 2WikiMQA-E may alter the checkpoint, packet depth, precision,
prompt, decode length, metric, or gate.  After the three runs below, the file
is exhausted for model selection.

## Frozen data handling

Use every non-empty row of
`/data/datasets/longbench/2wikimqa_e.jsonl` exactly once in file order.  After
this protocol is saved and before loading a model, record file SHA256, row
count, unique ID count, and unique normalized-question count without examining
question or answer text.  Require unique IDs and questions.

## Fixed model arms

Frozen model: BF16 Llama-3.1-8B-Instruct.  All runs use four slots, INT4,
greedy decoding, source shift 1, at most 24 source-answer tokens, at most 24
receiver tokens, and both `state_readout` and `question_conditioned` protocols.
The evaluator also reports no-summary and generated-answer all-layer Tail-KV.

Run exactly:

1. new continued checkpoint, first16 packet (67,584 B);
2. parent N512 checkpoint, first16 packet (67,584 B);
3. new continued checkpoint, all-layer packet (135,168 B), secondary.

New checkpoint:
`outputs/amortized_semantic_kv/llama31_increment256_allonly_cont3200_confirm128_20260818/capsule.pt`.

Parent checkpoint:
`outputs/amortized_semantic_kv/llama31_n512_allonly_int4_budgetdev128_fixed1600_20260817/capsule.pt`.

## Confirmatory gates

Question-conditioned F1 is primary.  Use paired 20,000-replicate bootstrap
intervals over all rows.

1. **Useful compact state:** new first16 beats no-summary with a positive CI
   lower bound.
2. **Source causality:** new first16 beats its shift-1 capsule with a positive
   CI lower bound.
3. **Compact data scaling:** new first16 beats parent first16 by at least `.02`
   F1 and has a positive CI lower bound.

All three gates are required for a sealed compact-scaling claim.  Report EM
McNemar statistics, state-readout F1, payload, source-answer quality,
generated Tail, and new all versus new first16 regardless of outcome.  The
all-layer arm is a secondary quality--payload point and cannot rescue a failed
first16 gate.

If any gate fails, state precisely which narrower claims survive.  Do not run
another checkpoint, split point, quantization level, prompt, or decode setting
on 2WikiMQA-E.

## Frozen pre-execution audit

After saving this protocol and before loading a model, metadata-only audit
found:

- file SHA256:
  `525b5b182089a4012cc7429c33f4208358778615173c4a09349429fc80c89641`;
- 300 non-empty rows;
- 300 unique `_id` values;
- 300 unique normalized questions.

No question, context, answer, or model output was printed or examined during
this audit.
