# Equal-byte RoPE-polar Key codec results (2026-08-18)

## Decision

The fixed 3-bit-radius/5-bit-phase Key codec **fails the preregistered
confirmation**.  It preserves packet size and receiver-independent byte
identity exactly, and the transmitted state remains strongly document-causal
on MuSiQue.  However, on the untouched MuSiQue rows 16--95 it loses `.0125`
sidecar F1 relative to Cartesian canonical, with ordinary and cluster lower
bounds below the allowed `-.03` margin.  It also fails the 2Wiki repair and
causality confidence gates.

The disclosed rows 0--15 are excluded from every number below.  No alternate
bit split, wider Key representation, or changed threshold was selected after
seeing these results.

## Wire and invariance audit

Each rotate-half Key pair is encoded by a 3-bit unsigned radius and a 5-bit
phase, while Value retains signed Cartesian INT4.  The packet framing, layer
topology, FP16 scale count, CRC, and number of payload bytes are unchanged.

- first16 base packet: 67,642 framed bytes;
- layer-15 delta packet: 4,252 framed bytes;
- base-hit incremental transfer: 4,252 bytes;
- cold base plus delta: 71,894 bytes;
- MuSiQue: stage and bridge base/delta packets are byte-identical for every
  document;
- 2Wiki: state-readout and question-conditioned packet hashes are identical
  for every document, including the shifted-source control.

Gate 1 passes exactly.  The current reference implementation performs polar
trigonometry on CPU while constructing packets and is not a latency
optimization; behavioral safety is the controlling failure.  The complete
repository suite passes: `183 passed, 2 warnings`.

## MuSiQue confirmation, rows 16--95 (`n=80`)

| contrast | candidate F1 | reference F1 | delta (95% CI) | bridge-cluster CI |
|---|---:|---:|---:|---:|
| polar direct vs Cartesian canonical | .7538 | .7725 | -.0188 `[-.0500,.0000]` | `[-.0506,.0000]` |
| polar sidecar vs Cartesian canonical | .7800 | .7925 | -.0125 `[-.0313,.0000]` | `[-.0329,.0000]` |
| polar sidecar vs shifted polar sidecar | .7800 | .2638 | +.5163 `[+.4088,+.6213]` | `[+.3987,+.6288]` |

The sidecar comparison has 0 F1 wins, 78 ties, and 2 losses.  Its point loss
exceeds `.01`, its ordinary lower bound is below `-.03`, and its cluster lower
bound is also below `-.03`.  Gate 3 fails in all three ways.  The correct-source
causality contrast remains large and passes the MuSiQue part of gate 4.

## 2WikiMQA confirmation, rows 16--199 (`n=184`)

All entries below are question-conditioned paired comparisons.

| contrast | candidate F1 | reference F1 | delta (95% CI) |
|---|---:|---:|---:|
| polar sidecar vs Cartesian target-frame | .3270 | .3419 | -.0149 `[-.0537,+.0217]` |
| polar sidecar vs Cartesian canonical | .3270 | .3366 | -.0096 `[-.0496,+.0289]` |
| polar direct vs Cartesian canonical | .3364 | .3242 | +.0121 `[-.0231,+.0476]` |
| polar sidecar vs shifted polar sidecar | .3270 | .2686 | +.0584 `[-.0029,+.1195]` |

The target-frame repair loses `.0149` F1 and has a lower bound well below
`-.03`, so gate 2 fails.  The sidecar point comparison with Cartesian canonical
barely satisfies `>=-.01`, but its lower bound fails the margin, so the 2Wiki
part of gate 3 also fails.  Correct-source F1 is higher than shifted-source F1,
but the paired lower bound is slightly negative; the 2Wiki part of gate 4
therefore fails.

State-readout remains weak and is diagnostic rather than a rescue result.  For
polar sidecar versus Cartesian canonical its F1 delta is `+.00089`, CI
`[-.01007,+.00958]`; polar correct versus shifted is `+.01658`, CI
`[+.00548,+.03026]`.

## Gate table

| preregistered gate | result |
|---|---|
| equal bytes and target-invariant hashes | pass exactly |
| 2Wiki repair versus Cartesian target frame | **fail** |
| no regression versus Cartesian canonical on MuSiQue | **fail** |
| no regression versus Cartesian canonical on 2Wiki | **fail** |
| MuSiQue source causality | pass |
| 2Wiki question-conditioned source causality | **fail** |

Overall decision: **fail**.  Polar coordinates provide the desired algebraic
RoPE semantics, but a single byte per Key pair does not preserve the boundary
state reliably enough.  This rules out the fixed equal-rate repair; it does not
invalidate the Cartesian target-frame short-KV or composable-delta results.

## Artifacts

- Protocol: `docs/llama31_rope_polar_key_preregister_20260818.md`
- Polar MuSiQue direct/sidecar/shift:
  `outputs/amortized_semantic_kv/llama31_r16_polar_canonical_first16_realint4_alignment96_20260818/`,
  `outputs/amortized_semantic_kv/llama31_r16_polar_composable_delta_canonical_first16_realint4_alignment96_20260818/`,
  `outputs/amortized_semantic_kv/llama31_r16_polar_composable_delta_canonical_shift1_first16_realint4_alignment96_20260818/`
- Polar 2Wiki direct/sidecar/base:
  `outputs/amortized_semantic_kv/llama31_r16_polar_canonical_2wikimqa_first16_realint4_20260818/`,
  `outputs/amortized_semantic_kv/llama31_r16_polar_composable_delta_canonical_2wikimqa_first16_realint4_20260818/`,
  `outputs/amortized_semantic_kv/llama31_base_polar_canonical_2wikimqa_first16_realint4_20260818/`
- Confirmatory paired analyses:
  `outputs/amortized_semantic_kv/llama31_r16_rope_polar_analysis_20260818/`
