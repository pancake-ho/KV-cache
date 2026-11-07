# Generated Tail-KV capacity diagnostic

Status: frozen on 2026-08-18 after the positional-orbit external result and
answer-length diagnosis, before any output from the three runs below was
inspected.

This is a post-hoc mechanism diagnostic, not new external confirmation.  It
tests whether the MultifieldQA failure is consistent with an insufficient
number of native state positions.

## Frozen runs

Use the same 150 MultifieldQA cases, Llama-3.1-8B, receiver protocols, and
canonical real-INT4 path as the orbit evaluation.  Keep only the first 16
layers of generated answer Tail-KV.  Run sender generation caps of 4, 8, and
16 tokens; model EOS may make the actual tail shorter.  The learned four-slot
capsule is present only because the existing evaluator emits all controls; the
primary arm is `generated_tail`.

The primary endpoint is question-conditioned answer F1.  State-readout is
secondary.  Use paired 20,000-replicate bootstrap intervals with seed 2027.
Report actual tail positions and framed bytes rather than treating the cap as
the realized size.

## Frozen diagnostic decisions

1. **Global capacity support:** cap-16 minus cap-4 generated-tail F1 has a
   positive mean and positive 95% lower bound.
2. **Long-answer support:** on the already diagnosed 9+-word gold-answer
   stratum, cap-16 minus cap-4 has a positive mean and positive lower bound.
3. **Length interaction:** the cap-16-minus-cap-4 improvement on 9+-word
   answers exceeds that on 1--4-word answers, with a positive bootstrap lower
   bound for the difference.

Cap-8 comparisons and the 5--8-word stratum are descriptive.  If gate 1 fails,
do not justify a learned adaptive-slot method from this experiment.  If gate 1
passes but gates 2--3 fail, more tail positions help globally but the existing
length-based explanation is not supported.  This generated-answer oracle
mixes state capacity with answer truncation and cannot itself establish that a
pre-answer capsule with more slots will learn the same frontier.
