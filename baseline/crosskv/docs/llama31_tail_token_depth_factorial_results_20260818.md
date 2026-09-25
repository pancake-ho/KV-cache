# Generated Tail-KV token-by-depth factorial results

Date: 2026-08-18.  This report resolves the frozen protocol in
`llama31_tail_token_depth_factorial_preregister_20260818.md` without changing
its margins after observing the all-layer runs.

## Setup

- Model: Llama-3.1-8B-Instruct.
- Dataset: all 150 paired MultifieldQA cases used by the preceding external
  evaluation.
- State: KV of the sender's generated answer prefix, capped at 4, 8, or 16
  tokens; cap 24 is the fixed high-budget reference.
- Transport: canonical RoPE frame and the real framed signed K4/V4 codec.
- Factorial variable: first 16 layers versus all 32 layers.
- Primary protocol: question-conditioned receiver; 20,000 paired bootstrap
  replicates, seed 2027.

This is a post-answer Tail-KV experiment.  It is not evidence that a learned
pre-answer writer can directly emit the same state.

## Primary results

| token cap | mean realized positions | first16 F1 | all32 F1 | all32-first16 F1 (95% CI) | all32 bytes |
|---:|---:|---:|---:|---:|---:|
| 4 | 3.733 | .15606 | .20587 | +.04981 `[+.01889,+.08222]` | 126,246.8 |
| 8 | 6.460 | .18679 | .23159 | +.04480 `[+.01437,+.07829]` | 218,386.3 |
| 16 | 10.287 | .18878 | .25395 | +.06517 `[+.02873,+.10399]` | 347,697.0 |

Every all-layer depth effect has a positive lower confidence bound.  Thus
short Tail-KV positions are depth-indexed state: sending only early-layer KV
discards useful receiver state even when the retained token positions are
unchanged.

Within all32, cap8-minus-cap4 is `+.02572` F1 with CI
`[-.00329,+.05470]`; cap16-minus-cap8 is `+.02236` with CI
`[-.00321,+.04954]`; cap16-minus-cap4 is `+.04808` with CI
`[+.01436,+.08222]`.  Capacity still matters globally, although neither
adjacent step is individually resolved on 150 cases.  The all32 long-versus-
short interaction CI crosses zero, so the strong first16 length interaction
does not replicate at all32.

## Frozen cap-8 frontier decision

The cap-24 all-layer reference realizes 12.887 positions, scores `.26116` F1,
and uses 435,556.2 mean framed bytes.

| preregistered gate | observation | decision |
|---|---|---|
| cap8 vs cap24: mean >= -.03 and CI lower > -.06 | `-.02957`, CI `[-.05763,-.00150]` | pass |
| cap8 vs no state: mean and CI lower > 0 | `+.04720`, CI `[+.00270,+.09269]` | pass |
| cap8 bytes <= 60% of cap24 | 218,386.3 / 435,556.2 = 50.14% | pass |

The primary paper-frontier decision is therefore **authorized** under the
frozen margins.  The wording must remain precise: cap8 is measurably worse
than cap24 relative to a zero-difference test, but its loss stays within the
predeclared quality margins while removing 49.86% of the wire bytes.

## Secondary state-readout protocol

All32-minus-first16 remains positive at every cap: `+.09384`, `+.13186`, and
`+.16924`, all with positive CI lower bounds.  Cap8 is strongly better than no
state (`+.13560`, CI `[+.10285,+.16867]`) but is `-.06905` below cap24, CI
`[-.10597,-.03383]`; it therefore fails the secondary cap24 non-inferiority
gate.  This protocol is diagnostic and was not the primary selection rule.

## Interpretation

The strongest currently verified low-budget operating point is bounded,
all-layer, generated-answer Tail-KV: cap8 sends 6.46 positions on average and
passes the frozen primary cost-quality gates.  It establishes that useful
short native KV state exists and that both token capacity and layer depth are
independent rate axes.

It does not solve semantic compression before answer generation.  The learned
four-slot capsule remains much smaller (67,642 B first16, or 135,226 B for an
all-layer monolithic packet) but scores `.16263` F1 at its tested external
first16 point, and the causally nested learned 4+4 refinement failed.  The
research gap is therefore no longer “does any short KV state exist?” but “can
a pre-answer, source-causal writer reproduce the native bounded-Tail state
without first decoding answer tokens?”

Machine-readable primary and secondary analyses are respectively:

- `outputs/amortized_semantic_kv/llama31_r16_orbitdistill_full1000_20260818/tail_token_depth_question_conditioned_analysis.json`
- `outputs/amortized_semantic_kv/llama31_r16_orbitdistill_full1000_20260818/tail_token_depth_state_readout_analysis.json`

The repository test suite now passes: `208 passed, 2 warnings`.
