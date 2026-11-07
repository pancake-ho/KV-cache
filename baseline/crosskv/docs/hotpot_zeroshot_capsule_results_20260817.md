# MuSiQue-trained KV capsule → HotpotQA zero-shot transfer

## Result in one sentence

The frozen 4-slot writer carries a statistically strong, source-conditioned
signal across datasets, but it does **not** yet close the quality gap to
post-answer Tail-KV and its net F1 gain over a question-only receiver remains
inconclusive.

The protocol was frozen before the 200-case run in
`docs/hotpot_zeroshot_capsule_preregister_20260817.md`.  The two implementation
smoke cases are excluded.  No HotpotQA example updated the MuSiQue-trained
checkpoint.

## Frozen setup

- Data: all 200 rows of local LongBench `hotpotqa.jsonl`, average 13,458 source
  tokens.
- Model: Qwen3-8B, same-model transfer.
- Capsule: 4 slots, rank-4 sender writer, first 30 layers, INT4, 126,720 bytes.
- Tail upper bound: greedily generated Agent-A answer tail, all 36 layers, INT4;
  mean 5.435 tokens and 206,617 bytes.
- Controls: no summary and a different example's circularly shifted 4-slot
  capsule.  Shift is performed inside each fixed execution shard.
- Metrics: HotpotQA normalized EM/F1; 20,000 paired bootstrap replicates and
  exact two-sided McNemar tests.

Agent A itself reaches 42.0% EM / 0.570 F1 from the full context.  This source
error ceiling is reported rather than silently conditioning the primary result
on successful source computation.

## Primary results

| Protocol / arm | EM | mean F1 |
|---|---:|---:|
| state readout: no summary | 0.0% | 0.002 |
| state readout: shifted capsule | 0.0% | 0.003 |
| **state readout: correct capsule** | **10.5%** | **0.169** |
| state readout: generated Tail-KV | 34.5% | 0.503 |
| question only: no summary | 16.5% | 0.274 |
| question only: shifted capsule | 13.5% | 0.213 |
| **question + correct capsule** | **21.5%** | **0.309** |
| question + generated Tail-KV | 41.5% | 0.560 |

The source-conditioned existence gate passes:

- state readout, capsule versus shift: 21 EM improvements and 0 regressions,
  exact `p=9.54e-7`; mean-F1 delta `+0.166`, 95% CI
  `[+0.122,+0.212]`;
- question-conditioned, capsule versus shift: 19 improvements and 3
  regressions, exact `p=8.55e-4`; mean-F1 delta `+0.096`, CI
  `[+0.051,+0.144]`.

The stricter net-utility gate is mixed rather than passed:

- question-conditioned capsule versus no summary improves EM by 5.0 points
  (14 improvements, 4 regressions; `p=0.0309`);
- mean F1 improves by `+0.035`, but its 95% CI
  `[-0.004,+0.077]` crosses zero.

Therefore the correct claim is not “the capsule improves HotpotQA” without a
qualifier.  It is: **a MuSiQue-trained 4-slot capsule transfers causal semantic
state zero-shot to HotpotQA; evidence for net question-answering utility is
positive in EM but inconclusive in mean F1.**

## Source-correct diagnostic

Among the 84 cases where full-context Agent A is exactly correct:

- state-readout capsule is 15.5% EM versus shifted 0%; generated Tail-KV is
  82.1%;
- question-conditioned capsule is 41.7% versus shifted 26.2% and no-summary
  34.5%; generated Tail-KV is 90.5%.

Capsule versus shift remains significant in this diagnostic subset, but capsule
versus no-summary does not (`p=0.180`, F1 CI crosses zero).  This shows the main
quality gap is state extraction/generalization, not merely Agent-A answer
errors.

## Payload and system profile

The fixed capsule is 38.7% smaller than the mean generated-answer Tail-KV
(126,720 versus 206,617 bytes), while avoiding 5.435 serial answer tokens.

The three-GPU accuracy run initially recorded synchronized wall times, but the
worker code synchronized PyTorch's default CUDA device rather than the model's
explicit `cuda:1/2` device.  Accuracy and cache contents are unaffected; pooled
three-GPU latency fields are invalid and must not be cited.  The evaluator now
sets the process-local CUDA device explicitly.

A corrected, otherwise idle single-H200 profile on 50 fixed cases gives:

| Component | Mean latency |
|---|---:|
| common source prefill (13,959 tokens) | 1,107.0 ms |
| 4-slot capsule emission | 86.1 ms |
| capsule RoPE/INT4 preparation | 12.5 ms |
| capsule receiver generation | 156.5 ms |
| Agent-A answer decode | 279.4 ms |
| Tail-KV RoPE/INT4 preparation | 13.4 ms |
| Tail-KV receiver generation | 193.4 ms |

After the common source prefill, capsule handoff takes 255.1 ms versus 486.2 ms
for generated Tail-KV, a 231.1 ms / 47.5% reduction.  This is a
speed--quality operating point, not equal-quality acceleration.  It also does
not beat the network size of plaintext short-answer handoff, which remains the
strong systems baseline for this particular task.

## Interpretation and next experiment

Three conclusions survive all controls:

1. Four contextual slots are not merely a MuSiQue label prior: a wrong Hotpot
   source almost eliminates direct state readout.
2. The current writer is only a weak universal extractor.  It preserves enough
   information for a causal signal, but most open-domain Hotpot entities are
   not recoverable.
3. Post-answer Tail-KV generalizes much better because normal autoregressive
   generation writes a strong lexical state.  The research target is therefore
   to train a query-diverse state writer that approaches this natural write
   operation before answer emission.

The next diagnostic is frozen as post-hoc, not confirmatory: evaluate the same
200 cases at full-36/BF16, full-36/INT4, and first-30/BF16.  If all remain near
the current result, the bottleneck is writer generalization; if full-36/BF16
jumps, MuSiQue-selected layer/quantization compression is not distribution
robust.  Any subsequent Hotpot-trained writer must use a new held-out source
pool and cannot reuse these 200 cases as confirmation.

## Completed post-hoc layer diagnosis

The frozen 2×2 and late-layer curve produced:

| Capsule layers / precision | Payload | state readout EM/F1 | question EM/F1 |
|---|---:|---:|---:|
| first30 / INT4 (confirmatory original) | 126,720 B | 10.5% / .169 | 21.5% / .309 |
| first30 / BF16 | 491,520 B | 12.5% / .183 | 23.0% / .309 |
| all36 / INT4 | 152,064 B | 16.5% / .306 | 26.5% / .389 |
| all36 / BF16 | 589,824 B | 18.0% / .308 | 25.5% / .380 |
| last6 / INT4 | 25,344 B | 0.0% / .002 | 22.5% / .351 |
| last12 / INT4 | 50,688 B | 0.5% / .033 | 25.0% / .396 |
| **last18 / INT4** | **76,032 B** | **14.5% / .288** | **27.0% / .418** |

At fixed layer sets, BF16 versus INT4 paired F1 intervals cross zero.  All36
INT4 versus first30 INT4 improves state-readout F1 by .136, CI
`[.087,.187]`, and question F1 by .080, CI `[.034,.126]`.  Last18 versus
all36 has question-F1 delta `+.029`, CI `[+.005,+.056]`, but this is selected
post hoc and not an unbiased superiority estimate.

The late-layer hypothesis was subsequently frozen and **confirmed** on the
question-disjoint HotpotQA-E 290-case split.  See
`docs/hotpot_last18_confirm_results_20260817.md`; these standard-200 diagnostics
remain labeled exploratory.

## Artifacts

- preregistration: `docs/hotpot_zeroshot_capsule_preregister_20260817.md`
- evaluator: `src/xmodel_kv/cli/evaluate_hotpot_summary_transfer.py`
- pooled analyzer: `src/xmodel_kv/cli/analyze_hotpot_summary_transfer.py`
- three frozen shards:
  `outputs/amortized_semantic_kv/hotpot_zeroshot_writer_frozen200_shard{0,1,2}_20260817/`
- merged rows/statistics:
  `outputs/amortized_semantic_kv/hotpot_zeroshot_writer_frozen200_pooled_20260817/`
- corrected system profile:
  `outputs/amortized_semantic_kv/hotpot_writer_system_profile_n50_gpu0_20260817/`
