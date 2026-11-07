# Canonical-RoPE packet results (2026-08-18)

## Decision

Receiver-independent packet identity is achieved exactly, and MuSiQue safety
and causality pass.  However, the two preregistered 2Wiki question-conditioned
safety lower bounds are `-.03323` and `-.03333`, just below the required
`-.03`.  The canonical method therefore **fails strictly** and cannot yet
support the general multi-agent fan-out claim.

The failure is narrow and mechanistically useful.  Point losses are only
`-.0070` and `-.0049`, but ordinary Cartesian INT4 quantization does not commute
with RoPE rotation.  The next intervention is a fixed-byte, polar Key codec
whose phase is transported in a canonical frame; changing the gate or simply
adding evaluation cases is not permitted.

## Exact target invariance

Keys are moved from sender positions to canonical positions 0--3, encoded,
then moved from 0--3 to each receiver position after decoding.  Values are
unchanged by repositioning.  The packet contains no receiver position.

- MuSiQue: for all 96 documents, the stage and bridge base packet bytes and
  delta packet bytes are respectively identical.  Runtime assertions pass on
  every case.
- 2Wiki: for all 200 documents, state-readout and question-conditioned use
  different receiver prompts, yet their per-document packet SHA-256 digests
  are identical.  The same audit passes for the shifted-source packets.
- Framed bytes remain 67,642 B base, 4,252 B delta, and 71,894 B cold total.
- Repository suite after the canonical path: `180 passed, 2 warnings`.

This proves a document packet can be cached independently of receiver prefix
length.  It does not by itself prove that its quantized behavior is safe.

## MuSiQue alignment-confirm96

| packet | target-frame F1 | canonical F1 | delta (95% CI) | cluster CI |
|---|---:|---:|---:|---:|
| direct rank 16 | .7740 | .7792 | +.0052 `[-.0156,+.0313]` | `[-.0147,+.0315]` |
| base + delta | .7948 | .8063 | +.0115 `[-.0104,+.0396]` | `[-.0109,+.0404]` |

Both point deltas exceed `-.01` and both ordinary/cluster lower bounds exceed
`-.03`; direct and composable MuSiQue safety gates pass.

Canonical sidecar correct minus shifted-source F1 is `+.53438`, ordinary CI
`[+.43542,+.63125]`, cluster CI `[+.43163,+.63229]`.  Causality passes.

## Opened 2WikiMQA-200

### Direct rank-16 packet

- state-readout: target `.02571`, canonical `.02651`, delta `+.00080`, CI
  `[-.01189,+.01302]`;
- question-conditioned: target `.31998`, canonical `.31297`, delta `-.00702`,
  CI `[-.03323,+.01889]`.

The question point passes, but the lower bound is `.00323` below the safety
margin.  Gate 2 fails.

### Base plus independent delta

- state-readout: target `.02471`, canonical `.02055`, delta `-.00416`, CI
  `[-.01059,+.00018]`;
- question-conditioned: target `.32919`, canonical `.32433`, delta `-.00486`,
  CI `[-.03333,+.02314]`.

Again the point passes but the lower bound is `.00333` below the margin.  Gate
3 fails.

Canonical correct minus shifted sidecar remains positive: question F1
`+.07322`, CI `[+.02183,+.12752]`; state-readout `+.01855`, CI
`[+.00400,+.03674]`.  Causality passes.

The unrestricted base is especially stable: target `.32354` versus canonical
`.32343`, delta `-.00012`, CI `[-.01785,+.01721]`.  Most of the uncertainty is
therefore in the task boundary packet rather than a universal loss of the
canonical frame.

## Gate table

| gate | result |
|---|---|
| receiver-target byte invariance | pass exactly |
| MuSiQue direct safety | pass |
| MuSiQue sidecar safety | pass |
| 2Wiki direct safety | **fail strictly** |
| 2Wiki sidecar safety | **fail strictly** |
| correct-source causality on both sets | pass |

Overall decision: fail, with exact systems semantics but insufficient 2Wiki
behavioral confidence.

## Artifacts

- Protocol: `docs/llama31_canonical_rope_packet_preregister_20260818.md`
- MuSiQue canonical direct/sidecar/shift:
  `outputs/amortized_semantic_kv/llama31_r16_canonical_first16_realint4_alignment96_20260818/`,
  `outputs/amortized_semantic_kv/llama31_r16_composable_delta_canonical_first16_realint4_alignment96_20260818/`,
  `outputs/amortized_semantic_kv/llama31_r16_composable_delta_canonical_shift1_first16_realint4_alignment96_20260818/`
- 2Wiki canonical direct/sidecar/base:
  `outputs/amortized_semantic_kv/llama31_r16_canonical_2wikimqa_first16_realint4_20260818/`,
  `outputs/amortized_semantic_kv/llama31_r16_composable_delta_canonical_2wikimqa_first16_realint4_20260818/`,
  `outputs/amortized_semantic_kv/llama31_base_canonical_2wikimqa_first16_realint4_20260818/`
- Analyses:
  `outputs/amortized_semantic_kv/llama31_r16_canonical_rope_analysis_20260818/`
