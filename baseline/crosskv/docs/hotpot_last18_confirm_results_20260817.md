# Confirmed late-layer semantic KV packet on HotpotQA-E

## Confirmatory verdict

Both pre-registered gates pass on the 290-question split that excludes every
normalized question overlap with the consumed HotpotQA standard-200 set.

The confirmed operating point is:

> 4 contextual slots + rank-4 sender-only writer + last 18 Qwen layers + INT4
> = 76,032 bytes, emitted before any answer token.

The writer checkpoint was trained only on 128 MuSiQue cases.  HotpotQA
standard-200 selected the late-layer hypothesis post hoc; HotpotQA-E-disjoint290
is the independent confirmation.  The data SHA-256 and decision rules were
frozen in `docs/hotpot_last18_confirm_preregister_20260817.md` before execution.

## Main results

The source context averages 9,913 tokens.  Full-context Agent A itself obtains
46.2% EM / 0.618 F1.

| Protocol / arm | EM | mean F1 |
|---|---:|---:|
| state readout: no summary | 0.0% | 0.000 |
| state readout: shifted capsule | 0.34% | 0.009 |
| **state readout: correct last18 capsule** | **11.7%** | **0.234** |
| state readout: generated Tail-KV | 35.2% | 0.524 |
| question only: no summary | 19.7% | 0.282 |
| question only: shifted capsule | 15.2% | 0.211 |
| **question + correct last18 capsule** | **31.0%** | **0.437** |
| question + generated Tail-KV | 42.8% | 0.581 |

### Pre-registered gates

State-conditioned readout:

- correct versus shifted F1 delta `+0.225`, paired-bootstrap 95% CI
  `[+0.184,+0.266]`;
- EM discordance 34 improvements versus 1 regression, exact two-sided
  McNemar `p=2.10e-9`.

Question-conditioned RAG utility:

- correct versus no-summary F1 delta `+0.156`, CI
  `[+0.109,+0.203]`; EM 41 improvements versus 8 regressions,
  `p=1.96e-6`;
- correct versus shifted F1 delta `+0.226`, CI
  `[+0.177,+0.275]`; EM 52 improvements versus 6 regressions,
  `p=3.16e-10`.

Thus this replication turns the standard-200 post-hoc late-layer observation
into a confirmed cross-question result.  All three execution shards have the
same F1 direction:

| Shard | n | no summary | shifted | correct capsule | generated Tail-KV |
|---|---:|---:|---:|---:|---:|
| 0 | 97 | .367 | .272 | **.504** | .618 |
| 1 | 97 | .258 | .191 | **.379** | .572 |
| 2 | 96 | .219 | .170 | **.429** | .553 |

## Source-correct diagnostic

On the 134 cases where full-context Agent A is exactly correct:

| Arm | EM | F1 |
|---|---:|---:|
| no summary | 38.1% | .438 |
| shifted capsule | 26.9% | .320 |
| correct last18 capsule | **57.5%** | **.656** |
| generated Tail-KV | 91.8% | .925 |

Correct capsule versus no-summary has F1 delta `+0.217`, CI
`[+0.140,+0.295]`; versus shifted it is `+0.336`, CI
`[+0.256,+0.416]`.  The remaining Tail-KV gap is therefore not explained only
by Agent-A source errors: pre-answer extraction still loses information that
normal answer generation writes into the lexical tail state.

## Transport and latency

- last18 capsule: fixed 76,032 bytes;
- generated Tail-KV: mean 5.272 tokens / 200,436 bytes;
- capsule payload reduction: 62.1%.

All workers explicitly set and synchronize their model CUDA device.  Across
the 290 single-request measurements:

| Component | Mean latency |
|---|---:|
| common source prefill | 716.5 ms |
| capsule emission | 76.3 ms |
| capsule relocation/INT4 preparation | 8.9 ms |
| capsule question-conditioned receiver | 137.2 ms |
| Agent-A answer decode | 260.9 ms |
| Tail relocation/INT4 preparation | 12.7 ms |
| Tail question-conditioned receiver | 167.0 ms |

After common source prefill, capsule handoff is 222.4 ms versus 440.5 ms for
generated Tail-KV, a 218.2 ms / 49.5% reduction.  Including source prefill it is
938.9 ms versus 1,157.1 ms, an 18.9% latency reduction at lower answer quality.
This is a measured quality--latency--payload tradeoff, not an equal-quality
speedup claim.

## What is new mechanistically

The standard-200 post-hoc layer curve separates two receiver behaviors:

- last 6/12 layers alone are almost unreadable under a generic state-readout
  query (0.0/0.5% EM), but support question-conditioned answers at 22.5/25.0%
  EM;
- last 18 layers recover direct readout (14.5% EM) and reach 27.0% EM / .418
  F1 on question-conditioned standard-200;
- first 30 layers lose substantial OOD F1, while BF16 versus INT4 at a fixed
  layer set is statistically indistinguishable.

This is not ordinary layer dropping.  It indicates that a continuation query
can interpret a late-layer KV packet that a generic decoder prompt cannot
lexically read out.  In other words, **task-usable state and directly
verbalizable state have different layer requirements**.  The independent E290
result validates the utility of the selected last18 packet, though a new
benchmark is still required before claiming a universal last-half rule.

## Remaining gap

Generated Tail-KV still exceeds the confirmed capsule by 11.7 EM points and
0.144 F1 on all 290 cases.  It also matches the source answer much more closely.
The current result is strong evidence for an amortized pre-answer semantic
state packet, not evidence that four slots preserve every computed fact.

The next paper-critical experiments are:

1. a second model family, trained and sealed with the same protocol;
2. a task whose handoff state is not reducible to a 3--6-token final answer;
3. plaintext summary, full-recompute, full-KV, CacheBlend/KVPacket-style and
   token-pruning baselines under the same serving stack;
4. a serving-engine measurement including serialization and link transfer;
5. a pre-registered late-layer curve on a third dataset, not further selection
   on these 290 cases.

## Artifacts

- sealed dataset and audit:
  `datasets/hotpot_summary_transfer/hotpotqa_e_disjoint_from_standard.{jsonl,summary.json}`;
- preregistration: `docs/hotpot_last18_confirm_preregister_20260817.md`;
- execution shards:
  `outputs/amortized_semantic_kv/hotpot_e290_last18_confirm_shard{0,1,2}_20260817/`;
- merged results/statistics:
  `outputs/amortized_semantic_kv/hotpot_e290_last18_confirm_pooled_20260817/`;
- standard-200 exploratory layer runs:
  `outputs/amortized_semantic_kv/hotpot_writer_posthoc_{last6,last12,last18,full36}_int4_n200_20260817/`.
