# Confirmatory replication of the late-layer KV capsule

Status before execution: **dataset materialized and protocol frozen; no result
from this 290-case run has been inspected**.

## Why this replication is necessary

The MuSiQue-trained 4-slot writer was first tested on LongBench HotpotQA
standard-200 at the frozen `first30 + INT4` operating point.  A clearly labeled
post-hoc 2×2/layer diagnostic on those same 200 cases found that:

- INT4 and BF16 are statistically indistinguishable at a fixed layer set;
- removing the last six layers harms cross-dataset transfer;
- the exploratory `last18 + INT4` subset reaches 27.0% EM / 0.418 F1 in the
  question-conditioned protocol with only 76,032 bytes.

Because `last18` was selected after observing standard-200, none of those
numbers can confirm a new operating point.  This protocol freezes it on a new,
question-disjoint split.

## Sealed data

- candidate: `/data/datasets/longbench/hotpotqa_e.jsonl` (300 rows);
- exclusion reference: `/data/datasets/longbench/hotpotqa.jsonl` (the consumed
  standard-200);
- normalization: whitespace collapse, strip, Unicode-preserving casefold;
- excluded: all 10 exact normalized question overlaps, plus any within-E
  duplicate under the same key;
- sealed output: `datasets/hotpot_summary_transfer/hotpotqa_e_disjoint_from_standard.jsonl`;
- final rows: 290;
- SHA-256:
  `7a1efc6a74791348b8aa5e3ec85f3ad231d3f5901802335f202abe3d5a3e8f51`.

The builder and complete exclusion list are recorded in the adjacent summary
JSON.  No standard-200 question is present in this replication.

## Frozen model and execution

- Qwen3-8B, BF16 model computation, SDPA, greedy decoding;
- unchanged MuSiQue-only checkpoint:
  `musique_softcapsule_writer_r4_lr5e4_bridge2_s4_n128_val64_95_steps1600_20260817/capsule.pt`;
- correct/shift capsule: 4 slots, rank-4 writer, **last 18 layers**, INT4,
  76,032-byte payload;
- generated-answer Tail-KV upper bound: all 36 layers, INT4;
- source and receiver limits: 24 new tokens;
- protocols: `state_readout` and `question_conditioned`;
- arms: no summary, correct capsule, circular shift-1 capsule, correct generated
  Tail-KV;
- execution shards: offsets/counts `0/97`, `97/97`, `194/96`; shift is inside
  each shard and never maps an example to itself.

Only execution parallelism differs between shards.  All 290 cases and both
protocols must be retained.

## Confirmatory gates

Primary RAG gate (`question_conditioned`):

1. correct capsule mean F1 exceeds no-summary with paired-bootstrap 95% CI
   excluding zero;
2. correct capsule mean F1 exceeds shifted capsule with CI excluding zero;
3. normalized EM is reported with exact two-sided McNemar, but F1 is the
   primary decision metric.

Source-conditioned state gate (`state_readout`): correct capsule must exceed
shifted capsule in mean F1 with CI excluding zero.  This checks causal state
transport even if the receiver cannot solve every full task.

Secondary outcomes: source Agent-A EM/F1, generated Tail-KV quality, payload,
correct/incorrect-source diagnostic, per-shard consistency, and synchronized
single-request timings.  There is no requirement that `last18` beat all-36 on
this replication; non-inferiority was not powered or margined in advance.

If both primary RAG comparisons pass, `last18 + INT4` becomes the validated
cross-Hotpot operating point.  If it fails, the standard-200 last18 result is
treated as post-hoc split overfitting.  No layer count or quantization setting
may be changed on these 290 cases and still be called confirmatory.
