# Multi-domain decision-supervised CE8 writer

Status: frozen on 2026-08-18 before observing parent or candidate output on
the new standard-LongBench development split.

## Motivation and scope

Native K/V coordinates, a fixed-query attention operator, and a full-prefix
on-policy plaintext-effect trajectory all rank state8 above equal-capacity
CE8, although CE8 is significantly better on final answers.  Native/final
gradients are not systematically opposed, and a simple source-contrastive
hinge has already failed in the progressive 4+4 pilot.  Therefore this
candidate changes neither representation nor loss.  It tests the remaining
direct hypothesis: decision-critical state directions require task- and
context-diverse downstream supervision.

This is one fixed data intervention, not a loss/weight/model scan.

## Leakage-audited data

Standard LongBench HotpotQA-200 and 2WikiMQA-200 are compared by normalized
question against every HotpotQA-E and 2WikiMQA-E question.  Ten Hotpot and six
2Wiki overlaps are removed.  Qasper and NarrativeQA remain untouched.

Each retained context/question/answer is converted into the same two-receiver
contract as MuSiQue:

- Agent A reads the complete multi-passage context and must retain its short
  answer before decoding it.
- The bridge receiver must verbalize that answer.
- In groups with eight unique answer strings, the final receiver uses the
  retained answer as a key in an eight-way candidate table.  Final answers are
  deterministically rotated within the group, so directly repeating the
  bridge answer cannot solve the stage task.

The synthetic split contains 312 train and 56 development cases.  Training
also replays the 256 MuSiQue cases used by the initialized parent, for 568
cases total.  No train/dev IDs or questions overlap, and no synthetic question
occurs in either E evaluation set.

- combined train SHA-256:
  `e6c7c4d60776b2222677dc0eec850704801e22cd3ed02ceae6b2f823c14b5bce`;
- standard-LongBench dev SHA-256:
  `2f4eec68a0a0c3700b4676c9f683b5443e94d6fdcfc3e38f1828d1f75668b9cb`;
- source prompt lengths over train+dev: minimum 1,175, mean 8,440.86,
  maximum 16,425 model tokens; all fit the model context.

## Frozen parent and training

- Parent: the existing all32 CE8 checkpoint
  `llama31_native_tail_scaleup_ce8_800_train128_eval64_20260818/capsule.pt`.
- Same Llama-3.1-8B-Instruct, eight slots, rank-4 layerwise writer, all 32
  transmitted layers, and straight-through K4/V4 training.
- Direct losses only: stage CE weight 1 and bridge CE weight 2.  No native
  state, attention-operator, trajectory, teacher-KL, source-contrast, orbit,
  or PCGrad term.
- 1,600 updates; AdamW LR `1e-4`, weight decay 0, clip 1; seed 2101; final
  checkpoint only.
- Source caches are prepared once and held on CPU.  No checkpoint selection or
  early stopping is allowed.
- Development generation cap 32; paper-facing external evaluations retain
  their existing 24-token protocol.

The parent and candidate are both evaluated with the real canonical all32
K4/V4 codec.  The packet remains the existing 270,426-byte eight-slot format.

## Frozen development gates

Use 20,000 paired bootstrap resamples.  All conditions are required before an
E-set evaluation is authorized:

1. On the 56 new synthetic development cases, candidate-minus-parent bridge
   F1 is at least `+.05` and its 95% CI lower bound is above zero.
2. Synthetic final F1 is non-inferior: point delta at least `-.02` and CI
   lower bound above `-.08`.
3. Candidate correct-source minus shift-1 F1 has a positive CI lower bound for
   both bridge and final routes.
4. On MuSiQue confirm rows 96--127, candidate-minus-parent final F1 is at
   least `-.03` with CI lower bound above `-.10`; bridge point delta is at
   least `-.05`.

If all four gates pass, run exactly the already established HotpotQA-E290 and
sealed 2WikiMQA-E protocols, comparing candidate, parent, shifted candidate,
no state, generated Tail-KV, and plaintext where already available.  Qasper
and NarrativeQA stay sealed until that external result is reported.

If any gate fails, reject this multi-domain continuation without changing the
mixture, learning rate, steps, candidate construction, or thresholds.  It
cannot establish cross-model transfer or end-to-end serving benefit.
