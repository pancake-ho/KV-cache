# Llama-3.1 continued-writer 2WikiMQA follow-up

Status: frozen before executing the new checkpoint on 2WikiMQA.

## Scope

The new all-only continuation checkpoint was selected solely by a disjoint
MuSiQue train256/confirm128 protocol.  On confirm128 it improves its parent by
`.0984` first16 F1 with ordinary and cluster intervals above zero.  Test whether
that compact-packet gain transfers without target-data training to the already
opened LongBench 2WikiMQA-200 workload.

This is a frozen external-distribution follow-up, not a new sealed test,
because the 200 questions and parent results were previously inspected.  It
may support or reject the data-scaling mechanism but cannot replace a run on
the still-untouched `2wikimqa_e` file.  That file remains sealed.

## Fixed protocol

- dataset: all 200 rows of `/data/datasets/longbench/2wikimqa.jsonl`, SHA256
  `dda279cf93a99e1e5bfa3291fb199fd55978d10a1feb31822953cf77a1742e37`;
- model: frozen BF16 Llama-3.1-8B-Instruct;
- new checkpoint:
  `outputs/amortized_semantic_kv/llama31_increment256_allonly_cont3200_confirm128_20260818/capsule.pt`;
- precision: INT4; four slots; greedy decode; 24 source and receiver tokens;
- protocols: `state_readout,question_conditioned`;
- controls: no summary, shift-1 source, and generated-answer all-layer Tail-KV;
- run first16 and all-layer capsule variants without changing any prompt or
  evaluator setting from the prior complete protocol.

Use the exact repeated parent first16 run and original parent all-layer run as
references.  Pair by case ID and protocol with 20,000 bootstrap replicates.

## Frozen interpretation gates

Question-conditioned F1 is primary.

1. **Compact scaling transfer:** new first16 minus repeated parent first16 must
   have delta at least `+.02` and a paired 95% CI lower bound above zero.
2. **Full-state safety:** new all minus parent all is non-inferior at margin
   `-.03`, requiring its CI lower bound above `-.03`.
3. **Causality retention:** within the new first16 run, correct capsule must
   beat shift-1 with a positive paired F1 CI lower bound.

Also report EM, state-readout F1, exact output agreement, payload, no-summary,
and generated Tail.  If gate 1 fails, the MuSiQue compact gain is
distribution-specific and the new checkpoint must not replace the parent in
the cross-task paper table.  No checkpoint, layer, precision, or prompt variant
may be tried on these 200 rows after inspection.
