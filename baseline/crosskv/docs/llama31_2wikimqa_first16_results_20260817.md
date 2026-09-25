# Llama-3.1 first16 capsule: 2WikiMQA results

## Decision

Both preregistered cross-dataset gates pass on all 200 untouched LongBench
2WikiMQA cases.  The MuSiQue-trained four-slot writer transfers without any
2Wiki training or configuration change.

The 67.6 KB first16 packet improves question-conditioned F1 over both
no-summary and a shift-1 source control.  It also has significantly higher EM
than generated-answer Tail-KV on this benchmark.  The all-layer packet restores
generic state readout but does not improve question-conditioned F1 over
first16, exposing a task-state versus lexical-state depth tradeoff.

## Audit and source difficulty

- dataset: 200/200 rows of `/data/datasets/longbench/2wikimqa.jsonl`;
- 200 unique IDs and questions;
- file SHA256:
  `dda279cf93a99e1e5bfa3291fb199fd55978d10a1feb31822953cf77a1742e37`;
- mean source length: 7,197 Llama tokens;
- Agent-A generated answer: `.430 EM / .523 F1`.

The source task is materially harder and more compositionally varied than the
MuSiQue lookup construction: it includes comparisons, yes/no questions, dates,
and multi-entity relations.  No 2Wiki gradient update was performed.

## Question-conditioned answering

| arm | EM | F1 | payload |
|---|---:|---:|---:|
| no-summary | .045 | .141 | 0 |
| shifted first16 | .150 | .207 | 67,584 B |
| generated Tail-KV | .115 | .208 | 184,673 B mean |
| all-layer capsule | .130 | .256 | 135,168 B |
| **first16 capsule** | **.200** | **.267** | **67,584 B** |

First16 versus no-summary:

- F1 delta `+.1259`, 95% CI `[+.0786,+.1762]`;
- EM delta `+.155`, CI `[+.105,+.210]`;
- 32 capsule-only versus 1 no-summary-only exact case,
  McNemar `p=7.92e-9`.

First16 versus shifted first16:

- F1 delta `+.0596`, CI `[+.0134,+.1071]`;
- EM delta `+.050`, CI `[+.005,+.100]`;
- 17 correct-only versus 7 shift-only exact cases.

Both F1 lower bounds are positive, so the preregistered transfer gate passes.
The nontrivial shifted score shows that first-half cache positions can alter
the receiver even with the wrong source; paired correct-vs-shift evidence is
therefore essential and cannot be replaced by comparison to no-summary alone.

Generated Tail-KV versus first16 has F1 delta `-.0591`, CI
`[-.1260,+.0050]`, so F1 is statistically unresolved; its EM is lower by
`.085`, CI `[-.150,-.020]`.  On the 86 cases where Agent A's answer is exact,
first16 reaches `.349 EM / .424 F1`, versus `.174/.248` for generated Tail-KV;
Tail-minus-capsule F1 is `-.176`, CI `[-.296,-.053]`.  Thus capsule utility is
not explained by merely preserving errors from the decoded short answer.

## First16 versus all-layer

First16 has EM `.200` versus `.130` for all layers; paired delta `+.070`, CI
`[+.010,+.130]`, McNemar `p=.0385`.  F1 differs by only `+.0107`, CI
`[-.0508,+.0727]`, and is unresolved.  Therefore first16 halves payload with no
detected F1 loss on 2Wiki and has higher exact short-answer accuracy in this
secondary paired comparison.  This comparison was not a preregistered primary
gate and should not be presented as universal first16 dominance.

This is not universal dominance.  On MuSiQue confirm256, all-layer F1 `.880`
was significantly higher than first16 `.677`.  Depth is a dataset-dependent
quality/payload control.

## Generic state readout

| arm | EM | F1 |
|---|---:|---:|
| no-summary | .000 | .005 |
| shifted first16 | .000 | .005 |
| first16 capsule | .005 | .022 |
| generated Tail-KV | .020 | .115 |
| **all-layer capsule** | **.060** | **.125** |

All-layer capsule versus its all-layer shift control has F1 delta `+.122`, CI
`[+.0848,+.1626]`; first16 versus shift is only `+.0164`, CI
`[+.0013,+.0342]`.  Directly comparing first16 with all layers gives F1 delta
`-.1036`, CI `[-.1429,-.0674]`.

The same all-layer state that strongly restores generic readout is not better
for question-conditioned answering.  This reinforces the earlier distinction:
deep-layer KV carries lexical/readout state, while early-layer KV is a compact
functional injection channel that the downstream query can transform itself.

## Frozen repeatability audit

The first16 and all-layer variants were materialized in separate executions.
A post-result audit found that source-answer generation changed surface form on
six cases even though source EM and F1 were identical.  This does not affect the
within-run first16 gates, and the capsule is emitted before source-answer
decoding, but a full-file first16 repeat was frozen before further comparison.

The repeat passes its predeclared gate exactly:

- question-conditioned first16 is again `.200 EM / .266680819 F1`;
- state-readout first16 is again `.005 EM / .021519841 F1`;
- all 200 capsule strings, EM values, and F1 values match the original for both
  protocols;
- repeat-minus-original EM and F1 deltas are exactly zero, with bootstrap CIs
  `[0,0]`;
- no-summary also matches across the first16/all-layer executions on 200/200
  strings, while generated Tail changes on 25/200 strings.

Thus the selected KV path is exactly reproducible under the frozen greedy
protocol, whereas the decoded-answer Tail path has small surface-form
non-determinism.  The first16/all-layer F1 tradeoff can be retained; its EM gap
remains a secondary layer-routing observation rather than a primary claim.

## Payload and latency

The generated answer averages 5.47 Tail-KV tokens, or 184.7 KB at all-layer
INT4.  First16 is 67.6 KB, a 63.4% reduction; all-layer capsule is 135.2 KB, a
26.9% reduction.

On synchronized single-request H200 runs after the common source prefill:

- question-conditioned first16 path: 193 ms;
- generated Tail-KV path: 485 ms;
- question-conditioned all-layer path: 244 ms;
- all-layer generated Tail-KV path: 482 ms.

First16 avoids Agent A answer decoding and reduces the observed post-prefill
path by 291 ms.  These are HF reference timings, not a serving-engine or
concurrent-throughput claim.

## Paper implication

The supported formulation is now stronger than “Tail-KV contains an answer”:

> A source-conditioned writer can emit four native KV slots before decoding
> any answer.  The resulting depth-indexed state transfers from MuSiQue to
> 2WikiMQA, can outperform the decoded-answer Tail-KV path, and exposes
> separate functional and lexical channels across receiver depth.

This does not solve plaintext dominance or cross-model transfer.  But it gives
the ICLR story a reproducible representation phenomenon, a 67/135 KB Pareto
frontier, causal shift controls, two model families, and now a genuinely new
multi-hop dataset.

Artifacts:

- first16 pooled:
  `outputs/amortized_semantic_kv/llama31_2wikimqa_first16_int4_pooled_20260817/`;
- all-layer:
  `outputs/amortized_semantic_kv/llama31_2wikimqa_all_int4_analysis_20260817/`;
- paired first16-vs-all:
  `outputs/amortized_semantic_kv/llama31_2wikimqa_first16_vs_all_repeat_confirmed_20260817/`;
- full first16 repeat:
  `outputs/amortized_semantic_kv/llama31_2wikimqa_first16_int4_repeat_full_20260817/`;
- repeatability analysis:
  `outputs/amortized_semantic_kv/llama31_2wikimqa_first16_repeatability_20260817/`.
