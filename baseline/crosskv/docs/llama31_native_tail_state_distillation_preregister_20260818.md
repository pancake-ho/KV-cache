# Equal-rate native Tail-KV state-distillation pilot

Status: frozen on 2026-08-18 after implementation/unit tests and one numerical
smoke step, before any multi-case arm is trained or evaluated.  The smoke used
one training example only to establish finite scale: unweighted canonical
K4/V4 relative state MSE was 1.5385 and gradient norm was 0.6306.  Its quality
output is not used to select an arm, threshold, case, or weight.

## Question

Can a four-slot pre-answer capsule learn the native state geometry of an
equal-length answer Tail-KV more effectively than CE-only continuation or the
existing behavioral-logit KL objective?

This is deliberately an equal-budget mechanism gate before an eight-slot
experiment.  Changing slot count now would confound representation capacity
with the proposed direct-state objective.

## Frozen arms

All arms initialize from
`llama31_increment256_allonly_cont3200_confirm128_20260818/capsule.pt` and train
the same 1,032,192 capsule parameters.  They use Llama-3.1-8B-Instruct,
MuSiQue increment-train rows 0--31, 400 updates, seed 2093, learning rate
`1e-4`, stage/bridge CE weights 1/2, all 32 packet layers, production-rate
K4/V4 STE, and no source-read bottleneck.

1. `CE`: no teacher objective.
2. `behavior-KL`: gold-answer Tail-KV receiver-logit KL, weight 0.1 and
   temperature 2.
3. `state`: canonical K4/V4 K/V relative MSE to the first at-most-four gold
   answer positions, weight 0.1.

Direct state matching covers layers 1--31.  Layer 0 is a frozen structural
negative control: its K/V is projected from the fixed soft-slot embedding
before the slot has read the source and therefore cannot reproduce a
sample-dependent answer-token embedding.

## Evaluation

- Development set: disjoint but previously opened MuSiQue confirm128 rows
  0--31.
- Each arm reports held-out canonical K4/V4 state relative MSE.
- Each checkpoint is then evaluated with the real framed INT4 codec, canonical
  positions, all 32 layers, once with the correct source and once with circular
  source shift 1.
- Primary quality endpoint: final-answer F1.  Bridge F1 is diagnostic.
- Paired bootstrap: 20,000 replicates, seed 2094.

## Frozen feasibility gates

The direct-state objective authorizes a larger cap8 distillation experiment
only if all conditions hold:

1. mean native-state loss over training steps 381--400 is at most 70% of the
   mean over steps 1--20;
2. held-out state MSE is at least 10% below CE and the paired 95% CI upper bound
   for `state - CE` is below zero;
3. correct-source final F1 `state - CE` is non-negative and its paired CI lower
   bound is greater than `-.05`;
4. the state arm's correct-source minus shifted-source final F1 is positive
   with a paired CI lower bound above zero.

The behavior-KL arm is a fixed comparator, not a selection fallback.  Failure
of any gate rejects direct per-position K/V regression at this topology and
rate; no weight scan is permitted on these 32 evaluation cases.
