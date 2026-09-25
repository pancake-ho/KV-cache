# Frozen-base boundary-residual results (2026-08-18)

> **Production-codec addendum (2026-08-18).**  These tables use fake INT4
> with FP32 scales.  The real packet codec transmits FP16 scales and changes
> greedy outputs: the unrestricted MuSiQue base is `.7583` F1, direct rank 4
> is `.7625`, and independently encoded rank-4 base+delta is `.7833`.  Use
> `docs/llama31_composable_delta_sidecar_results_20260818.md` for authoritative
> wire results and do not mix the two scale representations in one gate.

## Decision

The rank-4 boundary residual validates the proposed semantic decomposition but
does not pass the complete preregistered gate set.  It preserves the
unrestricted capsule's cross-task first16 behavior and significantly repairs
the transfer loss of full-parameter source-read-15 training.  Its in-domain
point improvement is exactly the required size, but the confidence interval
crosses zero, and it remains significantly below the full-parameter aligned
model.  Two of the five core gates pass.

The useful result is mechanistic: freezing the transferable packet base
prevents the catastrophic cross-task forgetting seen with CE and KL training,
while one emission-aligned residual layer can add task information at exactly
the same four-slot/INT4 wire cost.  The failure is now localized to residual
capacity and deep-state reconstruction rather than base preservation.

## Intervention and audit

- Llama-3.1-8B-Instruct, four slots, first16 INT4 payload 67,584 B.
- The unrestricted slot embeddings and complete rank-4 layerwise writer were
  frozen bit-for-bit.
- Source reading stopped after blocks 0--14.
- A zero-initialized rank-4 residual after block 14 was the only trainable
  module: 32,768 parameters, no teacher KL, 3,200 fixed steps.
- Final mean losses: total `.9123`, stage CE `.0203`, bridge CE `.4460`.
- Full training time: 1,126.3 s; resident source-cache storage: 188.49 GiB.
- The corrected run used the preregistered weight decay 0.01.  The earlier
  misconfigured launch is labeled `ABORTED.md` and produced no checkpoint.

Real-model tensor audits establish the intended structure:

- before training, zero residual plus source-read-15 gives a first16 packet
  bitwise identical to the unrestricted base (maximum absolute delta 0);
- after training, all frozen base parameters remain bitwise identical;
- K/V layers 0--14 remain bitwise identical for a real Llama prefix;
- layer 15 changes (maximum absolute delta 2.09375), and layers 16--31 change
  downstream (maximum 7.109375); the trained boundary-parameter L2 norm is
  3.3247.

Thus the compact packet keeps 15 of 16 base layers exactly and uses only its
last transmitted layer as an additive task channel.

## Absolute results

### MuSiQue alignment-confirm96

| model / packet | EM | F1 |
|---|---:|---:|
| shifted residual first16 | .167 | .250 |
| unrestricted base first16 | .677 | .722 |
| **frozen-base residual first16** | **.698** | **.763** |
| full-parameter aligned first16 | .781 | .831 |
| shifted residual all | .115 | .204 |
| frozen-base residual all | .750 | .799 |
| unrestricted base all | .823 | .843 |
| full-parameter aligned all | .844 | .872 |

### Opened 2WikiMQA-200, question-conditioned

| model / packet | EM | F1 |
|---|---:|---:|
| full-parameter aligned first16 | .150 | .241 |
| pre-answer-KL aligned first16 | .220 | .271 |
| unrestricted base first16 | .270 | .318 |
| **frozen-base residual first16** | **.280** | **.338** |
| residual-shift first16 | .250 | .295 |
| full-parameter aligned all | .090 | .236 |
| pre-answer-KL aligned all | .220 | .290 |
| frozen-base residual all | .260 | .325 |
| unrestricted base all | .265 | .392 |
| residual-shift all | .210 | .244 |

## Frozen gates

All intervals use 20,000 paired bootstrap replicates.

1. **In-domain task increment: fail on significance.** Residual minus base
   first16 is `+.0406`, meeting the point threshold, but ordinary CI
   `[-.0260,+.1094]` and bridge-cluster CI `[-.0236,+.1111]` cross zero.
2. **In-domain safety: fail.** Residual minus full-parameter aligned first16 is
   `-.0688`, ordinary CI `[-.1375,-.0052]`; the point loss exceeds `.05` and
   the lower bounds exceed neither safety requirement.
3. **Cross-task base retention: pass.** Residual minus unrestricted first16 is
   `+.0198`, CI `[-.0204,+.0604]`; this is above the `-.03` point margin and
   its lower bound is safely above `-.08`.
4. **Cross-task repair over task-only bottleneck: pass.** Residual minus
   full-parameter aligned first16 is `+.0971`, CI `[+.0372,+.1564]`.
5. **Causal state use on both sets: fail strictly.** MuSiQue correct minus
   shift is `+.5125`, ordinary CI `[+.4135,+.6083]` and cluster CI
   `[+.4075,+.6169]`.  On 2Wiki first16 it is `+.0425`, CI
   `[-.0113,+.0957]`, so the required positive interval is missed.

The separate full-packet safety diagnostic passes on MuSiQue: residual minus
base all is `-.0438`, ordinary/cluster lower bounds `-.0969/-.0977`, both just
above `-.10`.  It fails on 2Wiki: `-.0665`, CI `[-.1278,-.0059]`.  Deep generic
state-readout F1 is `.0326` versus `.1644` for unrestricted, confirming that a
single boundary residual preserves the compact base packet but cannot rebuild
all source-dependent deep layers after the source-read mask.

## Interpretation

The previous KL student changed the entire summary generator and traded
in-domain accuracy against transfer.  This experiment instead changes the
geometry of optimization: a frozen transferable base plus a small task
residual.  It raises 2Wiki first16 from `.241` to `.338` while also raising
MuSiQue over the unrestricted base from `.722` to `.763`, at unchanged wire
bytes.  This is strong evidence for a modular KV-state design even though the
rank-4 instance is not the final optimizer.

The next development intervention should keep the same frozen 15-layer base
and same wire packet, but increase only the rank of the single layer-15
residual.  The motivation is fixed by this run: task CE remains much higher
than full-parameter aligned training, whereas cross-task preservation already
passes.  A single rank-16 follow-up tests residual capacity without moving the
boundary or modifying additional base K/V layers.

## Artifacts

- Protocol: `docs/llama31_frozen_base_boundary_residual_preregister_20260818.md`
- Valid model/run:
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_r4_read15_cont3200_alignment96_wd001_20260818/`
- MuSiQue shift:
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_shift1_alignment96_wd001_20260818/`
- MuSiQue paired analyses:
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_analysis_20260818/`
- 2Wiki first16/all and paired analyses:
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_2wikimqa_first16_wd001_20260818/`,
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_2wikimqa_all_wd001_20260818/`,
  `outputs/amortized_semantic_kv/llama31_frozen_base_boundary_2wikimqa_analysis_20260818/`.
