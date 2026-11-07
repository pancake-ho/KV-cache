# Emission-aligned bottleneck 2WikiMQA cross-task protocol

Status: frozen after the MuSiQue alignment-confirm96 result and before running
either bottleneck checkpoint on LongBench 2WikiMQA-200.

## Scope

The source-read-15 candidate was trained only on the same MuSiQue increment256
as the unrestricted and source-read-16 references.  It was selected by a new
MuSiQue confirmation set.  Test whether emission alignment improves compact KV
without any 2Wiki training, prompt adaptation, or checkpoint selection.

This is an external-distribution development follow-up, not a sealed test: the
200-row dataset and unrestricted outputs were previously opened.  The separate
2WikiMQA-E300 file is exhausted and will not be touched again.

## Fixed protocol

- all 200 rows of `/data/datasets/longbench/2wikimqa.jsonl`, SHA256
  `dda279cf93a99e1e5bfa3291fb199fd55978d10a1feb31822953cf77a1742e37`;
- frozen BF16 Llama-3.1-8B-Instruct;
- four slots, actual INT4 packet accounting, greedy decoding, 24 source-answer
  and receiver tokens;
- `state_readout` and `question_conditioned` protocols;
- no-summary, shift-1 source capsule, and generated-answer all-layer Tail-KV
  controls emitted by the existing evaluator;
- first16 (67,584 B) and all-layer (135,168 B) runs for aligned read-15 and
  unaligned read-16;
- existing unrestricted continued-writer first16/all results are immutable
  compute-matched references.

Use paired 20,000-replicate bootstrap over all 200 IDs, seed 2027.

## Cross-task gates

Question-conditioned F1 is primary.  All gates are required:

1. **Alignment transfer:** aligned minus unaligned first16 is at least `+.02`
   with a positive paired CI lower bound.
2. **Compact superiority transfer:** aligned minus unrestricted first16 is at
   least `+.02` with a positive paired CI lower bound.
3. **Full-state safety:** aligned minus unrestricted all is at least `-.03`,
   with CI lower bound above `-.08`.
4. **Source causality:** aligned correct minus aligned shift-1 first16 is at
   least `+.02` with a positive CI lower bound.

Report EM, state-readout F1, payload, generated Tail, and aligned-vs-unaligned
all regardless of outcome.  If only the in-domain result passes, emission
alignment remains a MuSiQue-specific optimizer and cannot support the paper's
cross-task improvement claim.  No alternative boundary, checkpoint, precision,
prompt, or decode length will be tried on these rows after inspection.
