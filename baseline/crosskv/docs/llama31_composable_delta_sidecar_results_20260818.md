# Composable KV delta sidecar results (2026-08-18)

## Decision

Both preregistered primary gates pass.  A frozen rank-16 task capsule can be
represented as an independently encoded first16 base packet plus a separately
quantized layer-15 delta without behavioral loss on either MuSiQue or 2Wiki.
The same base bytes are the exact fallback, and the framed delta is 4,252 B.

The result supports an incremental-transfer claim after a document-base cache
hit.  It does **not** reduce one-shot cold bytes: base plus delta is 71,894 B,
6.29% larger than a monolithic 67,642-B packet.

## Mechanism and packet audit

For the same document and receiver position, the sender materializes the frozen
unrestricted base `C_base` and frozen boundary checkpoint `C_task`.  Runtime
topology checks prove that their transmitted layers 0--14 are bitwise equal.
The wire representation is

\[
  Q_4(C_{base}^{0:15}),\qquad
  Q_4(C_{task}^{15}-C_{base}^{15}).
\]

The difference is computed in FP32 from BF16 tensors.  Both components use the
real signed-INT4 codec with transmitted FP16 per-head/per-token scales, layer
indices, headers, and CRC32 checksums.  The receiver decodes both and adds only
layer 15.

| component | tensor/scales only | actual framed bytes |
|---|---:|---:|
| first16 base | 67,584 | 67,642 |
| layer-15 delta | 4,224 | 4,252 |
| cold base + delta | 71,808 | 71,894 |
| base-hit incremental transfer | 4,224 | 4,252 |

The framed sidecar is 6.286% of the framed base.  Packet tests prove that the
base bytes inside composition equal independently packing the same base, that
zero delta reconstructs the decoded base bitwise, layers 0--14 remain the exact
decoded base, and inactive layers 16--31 are zero.  The complete repository
suite passes: `180 passed, 2 warnings`.

## Primary rank-16 results

### MuSiQue alignment-confirm96, first16 real INT4

| representation | EM | F1 |
|---|---:|---:|
| direct rank-16 packet | .719 | .774 |
| **base + independent delta** | **.740** | **.795** |
| sidecar with source shifted by one | .177 | .272 |

Sidecar minus direct is `+.02083` F1, ordinary 95% CI
`[.00000,+.05208]`, and fixed bridge-cluster CI `[.00000,+.05208]`.
There are 2 wins, 94 ties, and 0 losses.  This passes the `-.01` point and
`-.03` lower-bound fidelity margins.

Correct sidecar minus shifted-source sidecar is `+.52292`, ordinary CI
`[+.42500,+.61875]`, cluster CI `[+.43043,+.61084]`.  The independently
quantized state therefore retains strong document causality.

### Opened 2WikiMQA-200, first16 real INT4

| protocol / representation | EM | F1 |
|---|---:|---:|
| state readout, direct | .005 | .0257 |
| state readout, sidecar | .005 | .0247 |
| question-conditioned, direct | .270 | .3200 |
| **question-conditioned, sidecar** | **.270** | **.3292** |
| question-conditioned, shifted sidecar | .205 | .2537 |

Question-conditioned sidecar minus direct is `+.00921`, paired CI
`[-.01207,+.03217]`; state-readout is `-.00100`, CI
`[-.01269,+.01019]`.  Both lower bounds exceed `-.03`, so the cross-task
fidelity gate passes.

Correct minus shifted sidecar is `+.07548` question F1, CI
`[+.02290,+.12934]`; state-readout is `+.02360`, CI
`[+.00875,+.04184]`.  Both causal contrasts are positive.

The layer-15 normalized RMS difference from a directly encoded task packet is
about `.2155` on MuSiQue and `.2172` on 2Wiki.  This is not tensor-negligible,
yet it changes few decoded answers and is behaviorally non-inferior.  Tensor
distance alone would have rejected a valid packetization.

## Rank-4 secondary operating point

Rank 4 was an audit, not a replacement for the primary.  It shows the effect is
not unique to rank 16.

| dataset | direct F1 | sidecar F1 | paired delta (95% CI) |
|---|---:|---:|---:|
| MuSiQue | .7625 | .7833 | +.0208 `[-.0052,+.0573]` |
| 2Wiki question | .3374 | .3431 | +.0057 `[-.0195,+.0318]` |

Both secondary lower bounds also exceed the `-.03` fidelity margin.

## Production-codec correction to earlier evidence

Earlier boundary reports used exact fake quantization with FP32 scales.  The
production packet transmits FP16 scales.  That small scale-rounding change
raises the unrestricted base on MuSiQue from `.7219` to `.7583` F1.  Therefore
the previous statement that rank 16 *significantly* improves the base does not
survive the production-codec reference:

- direct rank 16 minus real-codec base: `+.01563`, CI
  `[-.01563,+.05208]`, cluster CI `[-.01531,+.05208]`;
- sidecar rank 16 minus real-codec base: `+.03646`, CI
  `[-.00521,+.08333]`;
- on 2Wiki, sidecar rank 16 minus real-codec base is `+.00564`, CI
  `[-.02744,+.03865]`.

Thus the defensible result is preservation/non-inferiority and composability,
not a significant task gain over the real-codec base.  All new tables must use
the real codec as the authoritative wire result.

## Limit and next test

This experiment quantizes after Keys have been RoPE-repositioned to a receiver
position.  It proves exact fallback for a fixed route, but agents with different
system-prompt lengths would have different Key bytes.  The preregistered
canonical-RoPE follow-up encodes at positions 0--3 and applies receiver RoPE
after decode; it is required before claiming receiver-independent base reuse.

## Artifacts

- Protocol: `docs/llama31_composable_delta_sidecar_preregister_20260818.md`
- Core implementation: `src/xmodel_kv/composable_delta.py`
- Tests: `tests/test_composable_delta.py`, `tests/test_capsule_codec.py`
- Rank-16 MuSiQue direct/sidecar/shift:
  `outputs/amortized_semantic_kv/llama31_r16_first16_realint4_alignment96_20260818/`,
  `outputs/amortized_semantic_kv/llama31_r16_composable_delta_first16_realint4_alignment96_20260818/`,
  `outputs/amortized_semantic_kv/llama31_r16_composable_delta_shift1_first16_realint4_alignment96_20260818/`
- Rank-16 2Wiki direct/sidecar:
  `outputs/amortized_semantic_kv/llama31_r16_2wikimqa_first16_realint4_20260818/`,
  `outputs/amortized_semantic_kv/llama31_r16_composable_delta_2wikimqa_first16_realint4_20260818/`
- Analyses:
  `outputs/amortized_semantic_kv/llama31_r16_composable_delta_analysis_20260818/`
