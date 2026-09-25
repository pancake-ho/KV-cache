# Frozen-base boundary rank-16 follow-up results (2026-08-18)

> **Production-codec addendum (2026-08-18).**  The experiment below used the
> frozen fake-INT4 evaluator, whose scales remain FP32.  A subsequent real
> packet audit transmits FP16 scales and raises the unrestricted MuSiQue base
> from `.7219` to `.7583` F1.  Against that authoritative real-codec base,
> direct rank 16 is `.7740`, delta `+.0156`, ordinary CI
> `[-.0156,+.0521]`, cluster CI `[-.0153,+.0521]`; it is **not** a significant
> improvement.  The earlier gate result remains a valid statement about the
> preregistered fake-quant condition, but must not be used as the production
> wire claim.  See `docs/llama31_composable_delta_sidecar_results_20260818.md`.

## Decision

The single permitted rank-16 follow-up establishes a capacity frontier, not a
monotonic capacity improvement.  It is the first frozen-base model to obtain a
significant in-domain improvement over the unrestricted base while remaining
cross-task non-inferior to that base.  However, it does not significantly beat
rank 4, is slightly worse on 2Wiki, misses the aligned-model safety gate, and
again narrowly misses the 2Wiki source-causality gate.  Two of five core gates
pass, so the preregistered definition of a complete compact optimizer fails.

No further rank or learning-rate attempt is justified.  Rank 4 and rank 16 are
two operating points: rank 4 favors cross-task question F1, while rank 16 gives
the strongest statistically supported in-domain increment.

## Fixed intervention and audit

The only change from rank 4 was the block-14 boundary residual rank, increasing
trainable sender parameters from 32,768 to 131,072.  Four slots, source-read-15,
first16 INT4 payload (67,584 B), frozen base, data, optimizer, learning rate,
weight decay, weights, seed, and 3,200-step endpoint were unchanged.

- Final mean total/stage/bridge losses: `.8083/.0273/.3905`, versus
  `.9123/.0203/.4460` for rank 4.
- Training time: 1,121.7 s.
- Base embeddings and layerwise writer remain bitwise identical to the
  unrestricted checkpoint.
- Boundary parameter L2 norm: 6.8120.
- The topology still guarantees that K/V layers 0--14 are unchanged and only
  transmitted layer 15 plus downstream deep layers receive the residual.

## Absolute results

### MuSiQue alignment-confirm96

| model / packet | EM | F1 |
|---|---:|---:|
| rank-16 shift first16 | .146 | .255 |
| unrestricted base first16 | .677 | .722 |
| frozen-base rank-4 first16 | .698 | .763 |
| **frozen-base rank-16 first16** | **.719** | **.774** |
| full-parameter aligned first16 | .781 | .831 |
| rank-16 shift all | .125 | .215 |
| frozen-base rank-16 all | .729 | .780 |
| unrestricted base all | .823 | .843 |

Rank 16 improves bridge F1 from `.556` to `.622`, but final-answer all-layer
F1 falls from `.799` to `.780`.  The lower training loss therefore does not
translate monotonically into the evaluated continuation behavior.

### Opened 2WikiMQA-200, question-conditioned

| model / packet | EM | F1 |
|---|---:|---:|
| full-parameter aligned first16 | .150 | .241 |
| rank-16 shift first16 | .220 | .265 |
| **rank-16 first16** | **.270** | **.316** |
| unrestricted base first16 | .270 | .318 |
| frozen-base rank-4 first16 | .280 | .338 |
| rank-16 shift all | .145 | .219 |
| rank-16 all | .185 | .298 |
| unrestricted base all | .265 | .392 |

## Frozen gates

All intervals use 20,000 paired replicates; MuSiQue also reports the fixed
bridge-cluster bootstrap.

1. **In-domain increment: pass.** Rank 16 minus unrestricted first16 is
   `+.0521`, ordinary CI `[+.0052,+.1042]`, cluster CI
   `[+.0057,+.1050]`.
2. **In-domain aligned safety: fail.** Rank 16 minus full-parameter aligned is
   `-.0573`, just below the `-.05` point margin; ordinary/cluster CIs are
   `[-.1302,+.0156]` and `[-.1386,+.0194]`, whose lower bounds fail safety.
3. **Capacity hypothesis versus rank 4: fail.** The increment is only
   `+.0115`, CI `[-.0354,+.0583]`, below the required `+.02` and not
   significant.
4. **Cross-task retention and repair: pass.** Rank 16 minus unrestricted
   first16 is `-.0020`, CI `[-.0341,+.0290]`, satisfying both non-inferiority
   margins.  Versus full-parameter aligned it is `+.0754`, CI
   `[+.0129,+.1365]`.
5. **Causal state use on both sets: fail strictly.** MuSiQue correct minus
   shift is `+.5188`, ordinary CI `[+.4229,+.6115]`, cluster CI
   `[+.4220,+.6143]`.  2Wiki is `+.0513`, but CI
   `[-.0013,+.1047]` narrowly crosses zero.

The separate all-layer safety gate fails on both tasks.  MuSiQue rank 16 minus
base is `-.0625`, ordinary/cluster lower bounds `-.1094/-.1082`; 2Wiki is
`-.0938`, CI `[-.1529,-.0344]`.  Increasing a single compact boundary residual
cannot recreate the generic deep state removed by source-read-15.

## Implication and next mechanism

The important conjunction is real: rank 16 improves MuSiQue significantly
while remaining non-inferior to the transferable first16 base on 2Wiki, and it
does so by modifying only one of sixteen wire layers.  But merging the task
residual directly into that layer creates a capacity-dependent tradeoff.

The next design should stop treating this as one monolithic packet.  Store and
transport an immutable 16-layer base capsule plus an optional, separately
quantized layer-15 task delta.  The base remains the exact generic fallback;
the task agent composes the delta only when its route is selected.  A layer-15
four-slot INT4 sidecar is 4,224 B, only 6.25% of the 67,584-B base packet, and
can be sent alone when the document base is already cached.  This turns the
observed rank frontier into an explicit composable state interface rather than
choosing one compromise globally.

## Artifacts

- Protocol: `docs/llama31_frozen_base_boundary_rank16_preregister_20260818.md`
- Model/run:
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_r16_read15_cont3200_alignment96_wd001_20260818/`
- MuSiQue shift/analyses:
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_r16_shift1_alignment96_wd001_20260818/`,
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_r16_analysis_20260818/`
- 2Wiki first16/all/analyses:
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_r16_2wikimqa_first16_wd001_20260818/`,
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_r16_2wikimqa_all_wd001_20260818/`,
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_r16_2wikimqa_analysis_20260818/`.
