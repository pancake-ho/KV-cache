# KVPacket / CacheBlend / KV capsule same-task comparison

Date: 2026-08-17

This report places the fixed Llama-3.1-8B 4-slot capsule, native KVPacket, and
LMCache CacheBlend on the same 290 question-disjoint HotpotQA-E cases.  It is a
comparison of distinct reuse contracts, not a claim that all methods receive
the same information or use the same serving stack.

## Frozen protocols and audit trail

- Capsule, generated Tail-KV, plaintext, and full-recompute rows come from the
  common E290 matrix.
- KVPacket uses its public `HFBackend`, `PacketSession`, `PacketTrainer`, and
  `evaluate_sample` APIs.  One 8-token header and 8-token trailer are trained
  for five epochs on the same 128 MuSiQue source cases as the capsule; no E290
  tuning is performed.
- CacheBlend uses LMCache `0.5.4rc5.dev3`, vLLM `0.23.0`, layerwise
  `LMCacheConnectorV1`, 256-token chunks, check layer 1, and the official
  separator-token insertion.  Documents are populated in reverse order under
  a fixed non-target query, then reused in target order with the real query
  freshly computed.
- An initial CacheBlend pass incorrectly populated the target query itself and
  is invalid.  It is retained under `hotpot_e290_blend_shard*` but excluded from
  every number below.  Corrected rows are named
  `hotpot_e290_blend_querymiss_shard*`; every row satisfies
  `0 < cached_tokens < prompt_tokens`.

The preregistered protocols are in
`docs/kvpacket_same_task_preregister_20260817.md` and
`docs/cacheblend_same_task_preregister_20260817.md`.

## Common quality matrix

| arm | EM | F1 | reuse contract |
|---|---:|---:|---|
| 4-slot capsule, last16 INT4 | 15.9% | .304 | jointly computed source summarized to four late-layer slots |
| generated Tail-KV | 21.4% | .310 | KV of Agent A's generated short answer |
| generated-answer plaintext | 33.4% | .468 | Agent A's decoded short answer |
| full dossier recompute | 46.2% | .597 | complete target-order prompt |
| KVPacket no-recompute ablation | 29.3% | .391 | independent document caches, no learned wrapper correction |
| **KVPacket** | **32.8%** | **.444** | independently cached documents with learned 8+8-token wrapper |
| KVPacket native full recompute | 45.9% | .597 | its own full target-order path |
| CacheBlend cold full prefill | 44.5% | .582 | marker-identical target prompt, no cache reuse |
| CacheBlend, 15% recompute | 1.03% | .025 | reverse-order document cache, local importance recomputation |

KVPacket improves over its no-recompute ablation by `+.0533` F1, paired 95%
CI `[+.0114,+.0964]`; its `+3.45` EM-point change is not significant
(`p=.164`).  It is substantially stronger than the capsule: `+.1402` F1,
CI `[+.0901,+.1902]`, and `+16.9` EM points (`p=5.38e-11`).  It remains below
its native full recompute by `-.1528` F1, CI `[-.2056,-.1002]`.

The CacheBlend 15% arm is below its paired cold path by `-.5567` F1, CI
`[-.6093,-.5038]`, and `-43.4` EM points.  This result is specific to long,
dependent documents cached under a different order and query; it is not a
claim that CacheBlend fails on its intended independent-chunk workloads.

## CacheBlend integration recovery gate

Because the 15% result was unexpectedly poor, it was quarantined until a
full-recompute recovery test could rule out a broken integration.  A strict
16-case output-identity gate first failed: check layer 1 reproduced 14/16
normalized answers and check layer 0 reproduced 15/16.  Plain-vLLM and
ratio-1 reruns each reproduced their own outputs on 16/16 cases, showing a
stable numerical-path difference rather than sampling noise.

A second gate was fixed before running ratio 1 on all E290 cases: paired EM
and F1 delta 95% CIs had to lie inside `[-.05,+.05]`, with at least 80%
normalized-answer agreement.  It passed:

| ratio-1 versus cold | result |
|---|---:|
| normalized answer agreement | 87.2% |
| EM delta (95% CI) | -.00345 `[-.0276,+.0207]` |
| F1 delta (95% CI) | +.00136 `[-.0208,+.0249]` |

The corrected integration therefore recovers aggregate cold quality when all
positions are recomputed, while exact greedy outputs are not guaranteed.

## CacheBlend recomputation curve

The ratio `.50` point was run only after the recovery gate and is explicitly a
post-result mechanism diagnostic.

| recompute ratio | EM | F1 | TTFT | outputs with a trigram repeated at least 3 times |
|---:|---:|---:|---:|---:|
| cold full prefill | 44.5% | .582 | 278.9 ms | -- |
| 15% | 1.03% | .025 | **172.0 ms** | 66.9% |
| 50% | 4.83% | .081 | 255.3 ms | 45.5% |
| 100% | 44.1% | .583 | 368.2 ms | 0% |

This is a sharp recovery frontier rather than a smooth quality--compute trade:
recomputing half of the positions still produces repetitive, corrupted
decoding on many cases.  The effect is strongest once prompts exceed 4K:

| prompt length | cases | cold F1 | 15% F1 | 50% F1 | 100% F1 |
|---|---:|---:|---:|---:|---:|
| <=4,096 | 31 | .585 | .133 | .354 | .559 |
| 4,097--8,192 | 90 | .588 | .012 | .038 | .613 |
| 8,193--12,288 | 83 | .607 | .005 | .014 | .594 |
| >12,288 | 86 | .550 | .020 | .092 | .551 |

The recovery result supports the following mechanism interpretation: for
dependent reordered documents, retaining context-conditioned KV at most
positions while repairing only a local subset creates a globally inconsistent
state.  The error compounds through later layers and enters a repetitive
decoding attractor.  This interpretation is an inference from the ratio and
length interventions; direct hidden-state divergence measurements remain a
future mechanistic test.

## Payload and systems interpretation

| packet | mean theoretical payload |
|---|---:|
| 4-slot last16 INT4 capsule | **67,584 B** |
| generated Tail-KV INT4 | 175,369 B |
| raw Hotpot document KV BF16 | 1.228 GB |
| KVPacket document + wrapper KV BF16 | 1.247 GB |
| CacheBlend retrieved KV BF16 | 1.238 GB |

KVPacket's 16 wrapper tokens per document make its packet 1.57% larger than
the raw document KV and about 18,454 times the capsule payload.  It reduces
online computation, not transport volume.  CacheBlend retrieves about 18,315
capsule-equivalents of BF16 KV.  The capsule therefore occupies a distinct
low-payload point, but it does not dominate KVPacket in quality.

KVPacket reports 147.2 ms TTFT versus 2,258.4 ms for its native full-recompute
path.  CacheBlend 15% reports 172.0 ms TTFT versus 278.9 ms cold, but its answer
quality is unusable here.  These timings are not cross-engine comparisons:
KVPacket uses its native HF backend, whereas CacheBlend uses vLLM and a Torch
fallback because the local LMCache CUDA extension has a Torch ABI mismatch.
CacheBlend timing is therefore integration-only.  Population cost is also
amortizable and is reported separately in the raw analysis.

## Defensible conclusion

The direct baselines do not support a blanket claim that the capsule “beats”
existing reuse methods.  They establish a three-way frontier:

- KVPacket preserves far more information and obtains higher quality, but
  transports approximately the full document KV;
- CacheBlend is efficient when a small correction suffices, but the official
  15%/50% local repair is not robust to this long dependent-reordering task;
- the learned capsule is the only tested point with a fixed four-slot payload
  and statistically causal task information, but its absolute quality remains
  below plaintext and KVPacket.

This sharpens the paper problem from generic KV reuse to **learning a globally
consistent task-sufficient state under an extreme transport budget**.

## Artifacts

- native KVPacket rows:
  `../KVPacket/outputs/crosskv_same_task/llama31_musique128_hotpot_e290_8x8_20260817/`;
- CacheBlend rows:
  `../LMCache/outputs/cacheblend_same_task/`;
- corrected common competitor matrix:
  `outputs/amortized_semantic_kv/llama31_hotpot_e290_competitor_matrix_querymiss_20260817/`;
- CacheBlend recovery analysis:
  `outputs/amortized_semantic_kv/cacheblend_ratio100_recovery_e290_20260817/`;
- CacheBlend ratio curve:
  `outputs/amortized_semantic_kv/cacheblend_ratio_curve_e290_20260817/`.
