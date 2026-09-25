# Emission-aligned bottleneck 2WikiMQA cross-task results

## Decision

The emission-aligned source-read-15 writer does not transfer its in-domain
compact advantage to the already-open 2WikiMQA-200 task.  It remains better
than no-summary and has a positive point-estimate source contrast, but it is
significantly worse than the unrestricted writer at both first16 and all-layer
budgets.  All four preregistered cross-task gates fail.

The bottleneck therefore concentrates MuSiQue-sufficient state rather than a
task-invariant semantic summary.  Boundary alignment solves where state is
emitted; it does not by itself solve what semantics the bottleneck retains.

## Absolute question-conditioned results

| writer / packet | EM | F1 | payload |
|---|---:|---:|---:|
| no-summary | .045 | .141 | 0 B |
| generated-answer Tail-KV | .115--.125 | .212--.213 | about 185 KB |
| aligned read-15 shift first16 | .130 | .194 | 67,584 B |
| **aligned read-15 first16** | **.150** | **.241** | 67,584 B |
| unaligned read-16 first16 | .220 | .265 | 67,584 B |
| unrestricted first16 | **.270** | **.318** | 67,584 B |
| aligned read-15 shift all | .095 | .181 | 135,168 B |
| aligned read-15 all | .090 | .236 | 135,168 B |
| unaligned read-16 all | .180 | .246 | 135,168 B |
| **unrestricted all** | **.265** | **.392** | 135,168 B |

Aligned first16 generic state-readout F1 is `.0229`, close to unrestricted
`.0158`.  Aligned all-layer state-readout is only `.0506`, versus `.1644` for
unrestricted, showing that the bottleneck lost much of the transferable deep
readout state.

## Frozen gates

Paired 20,000-replicate bootstrap over 200 IDs:

1. **Alignment transfer fails.** Aligned minus unaligned first16 F1 is
   `-.0241`, CI `[-.0812,+.0325]`; point estimate is opposite the required
   `+.02`.
2. **Compact superiority transfer fails significantly.** Aligned minus
   unrestricted first16 is `-.0774`, CI `[-.1361,-.0192]`; EM delta `-.120`,
   CI `[-.185,-.055]`.
3. **Full-state safety fails significantly.** Aligned minus unrestricted all
   is `-.1553`, CI `[-.2118,-.0971]`, far below the safety margin.
4. **Source causality fails strictly.** Correct minus shift aligned first16 is
   `+.0468`, but CI `[-.0050,+.0999]` crosses zero.

Aligned minus unaligned all-layer F1 is `-.0096`, CI
`[-.0589,+.0396]`; both bottleneck writers lose strongly to the unrestricted
full packet.  The aligned writer is still useful versus no-summary in absolute
terms, but this does not satisfy an optimization-transfer claim.

## Mechanistic implication

The adjacent-boundary MuSiQue result remains valid: projecting `h_15` into
transmitted `K_15,V_15` is necessary for that compact task channel.  The 2Wiki
failure shows a second, orthogonal requirement: the four-slot bottleneck must
retain task-invariant behavior rather than only information rewarded by the
MuSiQue stage and bridge losses.

The next candidate should not change depth again.  Use the unrestricted
pre-answer writer—which has the strongest 2Wiki transfer—as a frozen teacher
for the aligned student.  Unlike the previously failed post-answer gold-Tail
teacher, this teacher contains no future answer tokens, uses the same four-slot
native-KV interface, and can distill general receiver behavior while the
aligned topology forces that behavior through the transmitted boundary.

## Artifacts

- Protocol: `docs/llama31_emission_aligned_2wikimqa_preregister_20260818.md`.
- Aligned first16/all:
  `outputs/amortized_semantic_kv/llama31_source_read15_2wikimqa_first16_int4_20260818/`,
  `outputs/amortized_semantic_kv/llama31_source_read15_2wikimqa_all_int4_20260818/`.
- Unaligned first16/all:
  `outputs/amortized_semantic_kv/llama31_source_read16_2wikimqa_first16_int4_20260818/`,
  `outputs/amortized_semantic_kv/llama31_source_read16_2wikimqa_all_int4_20260818/`.
- Paired analyses:
  `outputs/amortized_semantic_kv/llama31_emission_aligned_2wikimqa_analysis_20260818/`.
