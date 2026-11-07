# Llama-3.1 continued-writer 2WikiMQA follow-up results

## Decision

The new all-layer writer's cross-task gain is real but depth-skewed.  On the
already-open 2WikiMQA-200 set, all-layer question F1 improves by `.136` with a
paired CI above zero, while first16 improves by `.051` but narrowly misses its
strict F1 gate because the CI lower bound is `-.0026`.  First16 EM and its
correct-versus-shift causal contrast do improve significantly.

This supports full-depth data scaling and again shows that more training can
distribute useful state into deeper layers.  It does not provide sealed
confirmation of the compact gain.

## Absolute results

Question-conditioned protocol:

| checkpoint / packet | EM | F1 | payload |
|---|---:|---:|---:|
| parent first16 | .200 | .267 | 67,584 B |
| **continued first16** | **.270** | **.318** | 67,584 B |
| parent all | .130 | .256 | 135,168 B |
| **continued all** | **.265** | **.392** | 135,168 B |
| continued first16 shift-1 | .215 | .267 | 67,584 B |
| no-summary | .045 | .141 | 0 B |
| generated Tail-KV | .115 | .207--.210 | about 185 KB |

Generic state readout is `.0158` for continued first16 and `.1644` for
continued all, versus parent `.0215/.1251`.  Thus the added data improves deep
lexical/readout state while leaving shallow generic readout absent.

## Frozen gates

1. **Compact scaling transfer fails strictly:** continued minus repeated
   parent first16 question F1 is `+.0515`, CI `[-.0026,+.1064]`, below the
   required positive lower bound.  EM improves `+.070`, CI `[+.010,+.130]`,
   McNemar `p=.0336`.
2. **Full-state safety passes strongly:** continued minus parent all F1 is
   `+.1356`, CI `[+.0777,+.1944]`; EM improves `+.135`, CI
   `[+.075,+.200]`, McNemar `p=4.19e-5`.
3. **Causality retention passes:** continued first16 correct minus shift F1 is
   `+.0515`, CI `[+.0009,+.1031]`.  EM delta is `+.055`, whose interval touches
   zero.

Continued all versus continued first16 has F1 delta `+.0734`, CI
`[+.0147,+.1327]`, despite nearly equal EM (`.265/.270`).  Continued all also
improves parent state-readout F1 by `+.0393`, CI `[+.0097,+.0695]`; first16
state-readout delta is `-.0057`, CI crossing zero.

## Interpretation

The disjoint MuSiQue train256 produces clear external-distribution benefit,
so the new checkpoint is not merely memorizing MuSiQue IDs.  However, the
benefit is not uniformly concentrated in the compact early packet.  Joint
full-depth training strengthens a depth-indexed state channel, and deeper K/V
captures a growing share of transferable lexical/readout information.

The paper can claim a stronger 135 KB cross-task point and a suggestive 68 KB
improvement, but cannot claim confirmed compact scaling from this opened set.
The still-unread `2wikimqa_e` dataset is the appropriate sealed test if used
under a new protocol frozen before access.

Artifacts:

- preregistration:
  `docs/llama31_n768_2wikimqa_followup_preregister_20260818.md`;
- first16/all outputs:
  `outputs/amortized_semantic_kv/llama31_n768_2wikimqa_first16_int4_20260818/`,
  `outputs/amortized_semantic_kv/llama31_n768_2wikimqa_all_int4_20260818/`;
- paired analysis:
  `outputs/amortized_semantic_kv/llama31_n768_2wikimqa_followup_analysis_20260818/`.
