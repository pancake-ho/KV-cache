# K8/V4 canonical external 2Wiki diagnostic results (2026-08-18)

## Decision

The frozen K8/V4 candidate **fails both 2Wiki behavioral safety gates** on rows
16--199.  It retains strong source causality and exact receiver-independent
packet hashes, but its question-conditioned output is worse than both the
target-frame and K4/V4 canonical references.  K8/V4 therefore does not repair
the task-specific canonical failure and must not replace K4/V4 as the default.

This was an external diagnostic on already opened data, executed only after
the candidate passed its MuSiQue-200 confirmation.  Rows 0--15 used for the
earlier screen are excluded from every interval below.

## 2Wiki rows 16--199 (`n=184`)

| contrast | candidate F1 | reference F1 | delta (95% CI) |
|---|---:|---:|---:|
| K8/V4 canonical vs K4/V4 target | .32680 | .34187 | -.01507 `[-.04608,+.01330]` |
| K8/V4 canonical vs K4/V4 canonical | .32680 | .33659 | -.00979 `[-.04330,+.02204]` |
| K8/V4 correct vs shifted K8/V4 | .32680 | .25042 | +.07638 `[+.02464,+.12970]` |

Against target, the point loss is below the required `-.01` and the lower
bound is below `-.03`; gate 2 fails.  Against canonical K4/V4, the point is
below the required `-.005` and the lower bound is also below `-.03`; gate 3
fails.  Correct-source causality passes with 28 wins, 139 ties, and 17 losses.

State-readout K8/V4 is `.02079` F1.  Versus target it is `-.00064`, CI
`[-.01314,+.01040]`; correct versus shift is `+.01958`, CI
`[+.00759,+.03411]`.  This confirms source-specific state without rescuing the
question-conditioned safety failure.

## Mechanistic consequence

K8/V4 reduces layer-15 delta-vs-task NRMS from about `.20557` to `.03002`, yet
does not improve decoder behavior.  The earlier K4 canonical miss cannot be
explained as a simple shortage of Key quantization precision.  Autoregressive
outputs are discontinuous under small cache perturbations, and tensor distance
is a poor selection proxy for this packet.

The combined evidence is task-dependent:

- MuSiQue-200: K8/V4 safety and causal gates pass;
- 2Wiki-184: safety gates fail while causality passes;
- K8/V4 adds 48.43% cold bytes in both cases.

The correct paper conclusion is not “use more Key bits,” but “canonical
transport is exact at the packet level and preserves causal state, while
behavioral equivalence remains distribution-sensitive.”

## Artifacts

- Protocol:
  `docs/llama31_canonical_k8v4_2wiki_external_preregister_20260818.md`
- Candidate:
  `outputs/amortized_semantic_kv/llama31_r16_canonical_k8v4_sidecar_2wikimqa200_20260818/`
- Paired analyses:
  `outputs/amortized_semantic_kv/llama31_r16_canonical_k8v4_analysis_20260818/2wiki_holdout_*.json`
