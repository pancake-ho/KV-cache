# Pre-answer teacher distillation development protocol

Status: frozen after implementation, focused tests, and a two-update loss-scale
smoke; before the full candidate run.

## Motivation

Emission-aligned source-read-15 improves MuSiQue first16 F1 by `.109` over the
unrestricted compute control, but loses `.077` on zero-shot 2Wiki and loses
`.155` at all layers.  The bottleneck learns task-specific state.

Use the unrestricted continued writer as a frozen pre-answer teacher.  Teacher
and student both emit four native KV slots before answer decoding.  This differs
from the previously failed post-answer gold-Tail teacher: it contains no future
answer-token state and shares the student's interface and causal timing.

## Fixed method

- Student initialization: the common N512 parent, SHA256
  `b5848635fac01561b8f694aeca79d5d02d340a5c15eeb1e3967d21c90cb0b620`.
- Student topology: emission-aligned `source_read_layers=15`.
- Frozen teacher: unrestricted continued writer, SHA256
  `c488ccefca03cb3dbdfa89740aa2962cfe23979f0da28f9d1abc9594c1debefc`.
- Teacher topology: unrestricted source reads; four slots and rank-4 writer.
- Data and compute: the same increment256, 3,200 updates, seed 2027, LR
  `5e-4`, weight decay `.01`, all-layer-only receiver path, stage/bridge weights
  `1/2`, and INT4 STE.
- Student task loss remains `stage CE + 2 * bridge CE`.
- Distillation is
  `KL(teacher receiver || student receiver)` on teacher-forced stage and bridge
  answer positions, also weighted `1/2`, temperature 2.
- Fixed pre-answer KL weight: `.1`.

A two-update real-Llama smoke with weight 1 measured pre-answer KL around
`10--20` and task loss around `2--4`.  Weight `.1` is fixed from this scale so
the initial teacher contribution is roughly half the task loss.  No validation
metric or alternative weight was inspected.  No `.03/.3/1` scan is permitted.

## Development evaluations

1. Train once and evaluate first16/all INT4 on the already-open
   alignment-confirm96 set.
2. Without adaptation, evaluate first16/all on the already-open 2WikiMQA-200
   protocol with no-summary, shift-1, and generated Tail controls.

Fixed references are the no-teacher aligned read-15 student and unrestricted
continued writer already evaluated on both sets.  These are development gates,
not sealed claims.

## Gates

Paired 20,000-replicate bootstrap, seed 2027:

1. **In-domain compact safety:** teacher-student minus no-teacher aligned
   first16 MuSiQue F1 is at least `-.03`, with ordinary and bridge-cluster CI
   lower bounds above `-.08`.
2. **Cross-task recovery:** teacher-student minus no-teacher aligned first16
   2Wiki F1 is at least `+.04`, with positive CI lower bound.
3. **Cross-task compact safety versus teacher:** teacher-student minus
   unrestricted first16 2Wiki F1 is at least `-.03`, with CI lower bound above
   `-.08`.
4. **Cross-task full-state safety:** teacher-student minus unrestricted all
   2Wiki F1 has CI lower bound above `-.08`.
5. **Source causality:** correct minus shift teacher-student first16 on 2Wiki
   has a positive CI lower bound.

All gates are required before opening a third task as a sealed confirmation.
If in-domain safety passes but recovery fails, the teacher probes are too
narrow to preserve task-invariant semantics.  If recovery passes but compact
safety fails, distillation helps but still does not match the transferable
unrestricted state.  No second KL weight or boundary is allowed on these
development sets.

## Implementation evidence

- Existing focused capsule/distillation suite: `20 passed`.
- Real Llama two-update train/evaluate/checkpoint smoke completed.
- Teacher checkpoint is frozen (`requires_grad=False`) and is not serialized
  into the student packet.
- Training logs separately report post-answer `teacher_kl` and pre-answer
  `preanswer_teacher_kl`, preventing the two mechanisms from being conflated.
