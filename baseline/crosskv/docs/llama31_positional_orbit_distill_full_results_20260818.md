# Positional-orbit distillation: frozen full-run result

Date: 2026-08-18.  This document reports the single 1,000-step continuation
and four-arm external evaluation frozen in
`llama31_positional_orbit_distill_full_preregister_20260818.md`.  No threshold,
checkpoint, rate, seed, or evaluation row was selected after observing these
results.

## Intervention and integrity

The candidate starts from the pre-pilot rank-16 boundary checkpoint.  Only the
131,072 parameters of the block-14 boundary adapter are trainable.  The
transferable base, soft slots, writer, and Llama weights remain frozen.  The
orbit objective makes the decoder behavior of a canonical-frame K4/V4
base-plus-delta composition imitate the corresponding target-frame
composition.

The training audit passes:

- exactly 1,000 ordered updates completed and every logged value is finite;
- mean orbit KL falls from `.240187` over steps 1--20 to `.080331` over steps
  981--1000, a 66.55% reduction;
- exactly 131,072 trainable parameters change;
- frozen capsule and independent base hashes are unchanged.

Every one of the four external arms contains exactly 150 rows under both
receiver protocols.  Real packets are 67,642-B immutable base plus 4,252-B
layer-15 delta (71,894 B cold, 4,252 B on a base hit).  All canonical base and
delta hashes are exactly receiver-protocol independent.  The wire audit
passes every row.

## New external MultifieldQA result

The primary protocol is question-conditioned transfer.  Values below are mean
answer F1 over the frozen 150 cases.

| arm | target frame | canonical frame |
|---|---:|---:|
| frozen parent | `.15462` | `.15077` |
| orbit candidate | `.15428` | `.16263` |

The preregistered paired comparisons are:

| gate or diagnostic | mean F1 delta | paired bootstrap 95% CI | result |
|---|---:|---:|---|
| target retention, `O_t-P_t` | `-.00034` | `[-.02677,+.02729]` | pass |
| canonical safety, `O_c-O_t` | `+.00836` | `[-.00568,+.02320]` | pass |
| correct source minus shifted source for `O_c` | `+.02581` | `[-.00705,+.05963]` | **fail** |
| repair DiD, `(O_c-O_t)-(P_c-P_t)` | `+.01220` | `[-.01299,+.03910]` | **fail** |
| `O_c-P_c` characterization | `+.01186` | `[-.01188,+.03698]` | inconclusive |

Optimization/integrity, target retention, canonical safety, and wire
invariants pass.  Source causality does not.  Therefore the preregistered
overall external-safety decision is **false**.  The positive DiD point estimate
also has an interval spanning zero, so the stronger claim that orbit
distillation generally repairs canonical behavior is **not supported**.

Exact-match accuracy is at or near zero for all capsule arms on this dataset,
so it supplies no positive secondary evidence.

## Secondary state-readout protocol

Absolute F1 is near the floor: parent target/canonical score
`.01762/.02231`, and orbit target/canonical score `.01112/.01226`.
`O_t-P_t` is `-.00650` with CI `[-.01654,+.00241]`; `O_c-O_t` is
`+.00114` with CI `[-.00324,+.00562]`; correct minus shifted is `+.00619`
with CI `[-.00146,+.01434]`; and DiD is `-.00355` with CI
`[-.01295,+.00534]`.  This protocol independently fails causality and repair.

## Post-hoc diagnosis (not decision evidence)

These comparisons were computed only after the frozen gates and cannot rescue
them:

- orbit-canonical question-conditioned capsule versus no state is `-.02176`
  F1, CI `[-.05807,+.01420]`;
- capsule versus generated-answer Tail-KV is `-.09853`, CI
  `[-.14439,-.05293]`;
- sender answer F1 itself averages `.55218`, so source-answer generation is
  not simply at the floor;
- among the 36 cases where the sender answer has F1 exactly 1, capsule exceeds
  no state by `.03889`, but correct-source minus shifted-source is still only
  `.02453` as an uncorrected exploratory estimate.

The most likely reading is that the optimization successfully makes the
quantized representation position-orbit consistent, but the four-slot
MuSiQue-trained representation is not a sufficiently strong transferable
semantic channel for MultifieldQA.  Canonical RoPE/INT4 error is therefore not
the dominant remaining bottleneck at this operating point.

## Opened in-domain characterization

After the external decision was frozen, the predeclared MuSiQue alignment rows
16--95 were evaluated to localize the failure.  These rows are not external
confirmation evidence and cannot rescue the failed MultifieldQA gate.

For the final Stage-B answer, parent target/canonical F1 is `.77875/.79250`
and orbit target/canonical is `.76625/.77375`.  Target retention is `-.01250`,
CI `[-.06250,+.03750]`; canonical safety is `+.00750`, CI
`[-.03000,+.04000]`; and repair DiD is `-.00625`, CI
`[-.05000,+.03125]`.  There is no final-answer orbit-repair effect.

For direct readout of the Stage-A intermediate answer, the parent
target/canonical scores are `.21345/.18220`, while orbit target/canonical
reaches `.34881/.36214`.  Orbit-canonical versus parent-canonical is
`+.17994`, CI `[+.10363,+.26119]`, but the repair DiD is only `+.04458`, CI
`[-.00708,+.09917]`.  Thus the extra continuation significantly improves
in-domain semantic readout, primarily through its CE objectives rather than a
canonical-specific effect.

A separate shift-1 characterization proves that the in-domain candidate is
strongly source-causal: orbit-canonical correct minus shifted is `+.51000` F1,
CI `[+.40625,+.61250]`, for the final answer and `+.34881`, CI
`[+.26304,+.43839]`, for bridge readout.  The contrast with MultifieldQA is
therefore a distribution-transfer failure, not evidence that the packet is an
unconditional task prior.

The machine-readable characterization is
`llama31_r16_orbitdistill_full1000_20260818/musique_align80_characterization.json`.

## Claim boundary and next test

This run supports exact receiver-independent packet identity and fixed-rate
target/canonical non-inferiority on this dataset.  It does **not** support
external source-causal transfer or a general positional-orbit repair claim.
The opened MuSiQue result establishes strong in-distribution source causality
and improved intermediate-state readout, but not a positional repair.  It does
not overturn the external failure.  The next representation experiment must
target task-invariant semantic content rather than continue tuning positional
or quantization geometry.

Reproducible machine-readable analyses are in:

- `llama31_r16_orbitdistill_full1000_20260818/multifieldqa150_question_conditioned_analysis.json`;
- `llama31_r16_orbitdistill_full1000_20260818/multifieldqa150_state_readout_analysis.json`.

The repository suite after adding the frozen four-arm analyzers is
`194 passed, 2 warnings`.
