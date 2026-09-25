# Pre-answer teacher distillation development results

## Decision

Frozen unrestricted pre-answer behavior partially restores cross-task quality
and source causality, but it trades away too much of the emission-aligned
in-domain gain and remains below the unrestricted writer.  Only one of five
preregistered gates passes.  The fixed `.1` KL candidate is rejected and no
weight scan is performed.

This result distinguishes two teacher mechanisms.  Unlike post-answer
gold-Tail distillation, the pre-answer teacher is directionally helpful on
2Wiki.  The remaining failure is catastrophic/task tradeoff from updating the
entire student representation and from supervising only MuSiQue answer
positions, not future-state mismatch.

## Method and training

- Student: source-read-15, initialized from the common N512 parent.
- Frozen teacher: unrestricted N768 continued writer, four pre-answer slots.
- Same increment256, 3,200 updates, all-layer INT4 receiver, stage/bridge CE.
- Additional KL on teacher-forced stage/bridge logits, temperature 2, weight
  `.1` selected only from a two-update loss-scale smoke.

Final training means are task stage CE `.2116`, bridge CE `.4832`, and
pre-answer KL `4.7589`; weighted total loss is `1.6539`.  KL fell from about
11.3 at step 100 to 4.34 at step 2400, but large heavy-tail spikes around steps
1700 and 2500 raised the final average.  The frozen final step was retained;
no best-checkpoint selection was used.

## MuSiQue alignment-confirm96

| writer / packet | EM | F1 | bridge F1 |
|---|---:|---:|---:|
| unrestricted first16 | .677 | .722 | .187 |
| no-teacher aligned first16 | **.781** | **.831** | .287 |
| teacher-student first16 | .708 | .771 | **.340** |
| unrestricted all | .823 | .843 | **.909** |
| no-teacher aligned all | **.844** | **.872** | .543 |
| teacher-student all | .792 | .842 | .646 |

Teacher-student minus no-teacher aligned first16 is `-.0604`, ordinary CI
`[-.1396,+.0146]`, bridge-cluster CI `[-.1400,+.0158]`.  The point estimate is
below the frozen `-.03` requirement and both lower bounds violate `-.08`.
In-domain compact safety fails.

Teacher-student first16 remains `+.0490` above unrestricted in point estimate,
but its ordinary/cluster intervals cross zero.  Teacher-student all is almost
identical to unrestricted (`-.00035` F1), showing the teacher pulls the aligned
state back toward the transferable control.

## 2WikiMQA-200

Question-conditioned results:

| writer / packet | EM | F1 |
|---|---:|---:|
| no-teacher aligned first16 | .150 | .241 |
| **teacher-student first16** | **.220** | **.271** |
| unrestricted first16 | .270 | .318 |
| teacher-student shift first16 | .165 | .202 |
| no-teacher aligned all | .090 | .236 |
| **teacher-student all** | **.220** | **.290** |
| unrestricted all | .265 | .392 |
| teacher-student shift all | .110 | .151 |

Paired gates:

1. **In-domain compact safety fails:** `-.0604`, intervals above.
2. **Cross-task recovery fails:** teacher-student minus no-teacher aligned
   first16 is `+.0297`, CI `[-.0224,+.0808]`, below the required `+.04` and not
   significant.  EM improves `+.070`, CI `[+.015,+.125]`.
3. **Compact safety versus teacher fails:** versus unrestricted first16 F1 is
   `-.0476`, CI `[-.1140,+.0180]`; point estimate is below `-.03` and lower
   bound below `-.08`.
4. **Full-state safety fails:** versus unrestricted all F1 is `-.1017`, CI
   `[-.1644,-.0397]`.
5. **Source causality passes:** correct minus shift first16 F1 is `+.0682`, CI
   `[+.0146,+.1225]`; EM delta `+.055`, whose CI lower bound is `.005`.

Secondary evidence supports partial semantic recovery: all-layer F1 improves
`+.0536` over no-teacher aligned (CI `[-.0036,+.1104]`), and first16 generic
state-readout exceeds unrestricted by `+.0208`, CI `[+.0005,+.0432]`.  This is
not enough to satisfy the primary task gates.

## Next mechanism

Do not scan KL weight.  Both no-teacher fine-tuning and output-KL distillation
update every slot/writer parameter, allowing the transferable base state to be
forgotten.  A more structural decomposition is:

1. initialize from and freeze the unrestricted teacher's embeddings and
   layerwise writer as a **transferable base**;
2. apply the proven source-read-15 topology;
3. add a zero-initialized trainable boundary residual after block 14, so it is
   projected directly into transmitted `K_15,V_15`;
4. train only that small residual (and, if preregistered, deep residual blocks)
   on task loss.

Layers 0--14 then remain exactly teacher-equivalent, while the last first16
layer can add task-sufficient state without rewriting the general base.  This
targets the observed failure more directly than another soft loss tradeoff.

## Artifacts

- Protocol: `docs/llama31_preanswer_teacher_distillation_preregister_20260818.md`.
- Student:
  `outputs/amortized_semantic_kv/llama31_increment256_source_read15_preteacher_w01_cont3200_alignment96_20260818/`.
- MuSiQue paired analyses:
  `outputs/amortized_semantic_kv/llama31_preanswer_teacher_analysis_20260818/`.
- 2Wiki first16/all:
  `outputs/amortized_semantic_kv/llama31_preteacher_w01_2wikimqa_first16_int4_20260818/`,
  `outputs/amortized_semantic_kv/llama31_preteacher_w01_2wikimqa_all_int4_20260818/`.
- 2Wiki paired analyses:
  `outputs/amortized_semantic_kv/llama31_preanswer_teacher_2wikimqa_analysis_20260818/`.
