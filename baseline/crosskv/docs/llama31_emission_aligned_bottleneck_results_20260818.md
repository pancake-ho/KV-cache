# Llama-3.1 emission-aligned bottleneck confirmation results

## Decision

All four preregistered gates pass on a newly constructed 96-case confirmation
set that excludes every one of the 1,536 historical MuSiQue IDs.  Aligning the
continuous hidden-state bottleneck with the final emitted layer of the runtime
first16 packet improves compact F1 by about `.10--.11` over both the unaligned
source-read-16 writer and the compute-matched unrestricted writer, while
preserving full-state task quality and strong source causality.

The result supports a new mechanism claim:

> A sender information bottleneck only concentrates reusable KV state when its
> boundary hidden state is projected into a K/V layer that is actually present
> in the transmitted packet.  Full-depth gradients alone are insufficient;
> hidden-state and cache-layer boundaries must be emission-aligned.

## Frozen data

`alignment_confirm96_seed2087_exclude_used1536.jsonl` contains 96 unique cases
sampled from the final 103 eligible rows after excluding 1,536 unique prior
IDs.  It has zero overlap with all earlier train/development/confirmation sets.

- ordered-ID SHA256:
  `a466ebac4e54428b3de4f19e31d4801a4bc65ebd8222763ccff913b54785f876`;
- file SHA256:
  `faf31d719b27935cac90fbf2b817d2d2dffad07d26b1ee67df21a5586b4cdb3f`;
- 64 unique bridge-answer clusters.

## Method

All methods initialize the same four-slot/rank-4 writer parent and use the same
increment256, 3,200 updates, all-layer stage/bridge objective, and INT4 STE.

- **Unrestricted control:** every sender block may read the source.
- **Unaligned read-16:** blocks 0--15 read source; their final `h_16` first
  appears in packet layer 16, outside layers 0--15.
- **Emission-aligned read-15:** blocks 0--14 read source; `h_15` is projected
  into `K_15,V_15`, the last layer included by first16.  Blocks 15--31 attend
  only the four capsule positions.  No hidden state or gradient is detached.

## Absolute confirmation results

| writer / packet | stage EM | stage F1 | bridge F1 | payload |
|---|---:|---:|---:|---:|
| no-summary | .1563 | .2583 | -- | 0 B |
| unrestricted first16 | .6771 | .7219 | .1867 | 67,584 B |
| unaligned read-16 first16 | .6458 | .7302 | .1938 | 67,584 B |
| **aligned read-15 first16** | **.7813** | **.8313** | **.2868** | **67,584 B** |
| aligned read-15 shift first16 | .1563 | .2240 | .0169 | 67,584 B |
| unrestricted all | .8229 | .8427 | **.9090** | 135,168 B |
| unaligned read-16 all | .7396 | .7833 | .6882 | 135,168 B |
| **aligned read-15 all** | **.8438** | **.8719** | .5431 | 135,168 B |
| aligned read-15 shift all | .1979 | .2667 | .0132 | 135,168 B |

The aligned candidate's training loss is `.5724` (stage `.1173`, bridge
`.2275`), lower than unaligned read-16 `.7214` but higher than unrestricted
training.  Its generic lexical bridge readout remains below the unrestricted
all-layer writer even though downstream stage quality is higher.  The method
therefore concentrates task-sufficient state, not a complete textual summary.

## Frozen gates

Paired 20,000-replicate bootstrap, seed 2027:

1. **Alignment benefit passes.** Aligned minus unaligned first16 F1 is
   `+.1010`; ordinary CI `[+.0198,+.1833]`, bridge-cluster CI
   `[+.0081,+.1989]`.  EM delta is `+.1354`.
2. **Compact superiority passes.** Aligned minus unrestricted first16 F1 is
   `+.1094`; ordinary CI `[+.0313,+.1927]`, cluster CI
   `[+.0288,+.1975]`.  EM delta is `+.1042`, CI `[+.0104,+.2083]`.
3. **Full-state safety passes.** Aligned minus unrestricted all F1 is `+.0292`;
   ordinary CI `[-.0458,+.1073]`, cluster CI `[-.0534,+.1216]`.  The point
   estimate is above `-.03` and both lower bounds exceed the frozen `-.08`
   margin.
4. **Source causality passes.** Correct minus shift first16 F1 is `+.6073`;
   ordinary CI `[+.5146,+.6990]`, cluster CI `[+.5011,+.7116]`.  EM delta is
   `+.6250`, McNemar `p=2.73e-17`.

As a secondary check, aligned minus unaligned all-layer F1 is `+.0885`,
ordinary CI `[+.0115,+.1667]`, cluster CI `[+.0021,+.1853]`.

## Interpretation and limits

The read-16 negative and read-15 positive form a controlled adjacent-boundary
intervention.  Parameters, data, compute, losses, and wire format are fixed;
the only change determines whether the last source-conditioned hidden state is
serialized inside or immediately outside the compact packet.  This is stronger
mechanistic evidence than a broad layer-count sweep.

The result is currently same-model and in-domain.  It does not yet establish
cross-dataset superiority, cross-model transport, plaintext dominance, or an
end-to-end serving speedup.  The next required test is zero-target-training
evaluation on an already-open cross-task set; the exhausted sealed
2WikiMQA-E300 must not be reused for selection.

## Artifacts

- Protocol:
  `docs/llama31_emission_aligned_bottleneck_preregister_20260818.md`.
- Frozen dataset:
  `datasets/musique_handoff/alignment_confirm96_seed2087_exclude_used1536.jsonl`.
- Aligned candidate:
  `outputs/amortized_semantic_kv/llama31_increment256_source_read15_cont3200_alignment_confirm96_20260818/`.
- Shift control:
  `outputs/amortized_semantic_kv/llama31_increment256_source_read15_shift1_alignment_confirm96_20260818/`.
- Fixed unrestricted and unaligned references:
  `outputs/amortized_semantic_kv/llama31_increment256_allonly_alignment_confirm96_20260818/`,
  `outputs/amortized_semantic_kv/llama31_increment256_source_read16_alignment_confirm96_20260818/`.
- Paired analyses:
  `outputs/amortized_semantic_kv/llama31_emission_aligned_analysis_20260818/`.
