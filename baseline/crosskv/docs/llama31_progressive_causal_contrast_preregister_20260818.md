# Progressive refinement with counterfactual source contrast

Status: frozen on 2026-08-18 after the CE-only 200-step pilot failed and a
2-step implementation smoke passed, before this follow-up is run.

## Motivation fixed by the failed pilot

The causally nested representation and wire invariants pass, and CE-only loss
falls 57.66%, but on opened eval rows 32--47 full-8 minus core-4 is only
`+.01392` F1 (CI `[-.04131,+.06784]`) and correct full-8 minus shift-1 is
`-.06075` (CI `[-.18791,+.03932]`).  Full-8 is also below no-state.  Thus more
steps or more rows are not authorized: correct-source CE does not force the
refinement channel to discriminate its source from question/format priors.

This follow-up changes that mechanism only.  It is not a threshold rescue.

## Fixed intervention

For every scheduled correct training case `i`, use case `(i+1) mod 32` as a
counterfactual source while keeping the target question and answer fixed.  Let
`CE+` and `CE-` be the teacher-forced answer losses under correct and shifted
full-8 states.  Optimize

\[
L = CE^+ + \max(0, 0.5 + CE^+ - CE^-).
\]

The hinge has fixed weight 1 and margin `.5` nats per answer token.  Both
states use canonical production-equivalent first16 K4/V4.  Only the same
147,456 refinement parameters receive gradients; the four-slot core remains
frozen.

Everything else is identical to the failed pilot: fresh initialization from
the same core (not from the failed refinement), MultifieldQA rows 0--31 train
and 32--47 final evaluation, 200 updates, LR `1e-4`, weight decay `.01`, clip
1, seed 2027, final checkpoint only, and shift-1 evaluation.

## Frozen gates

Use the same analyzer and gates as the CE-only pilot:

1. 200 finite steps, at least 20% first20-to-last20 total-loss reduction,
   exact 147,456 trainable changes, unchanged frozen-core hash;
2. full-8 minus core-4 mean F1 at least `+.02`, CI lower `>-.05`;
3. correct full-8 minus shifted full-8 positive mean and positive CI lower;
4. exact 67,642-B core and refinement packets and changed shift source IDs.

All gates must pass to authorize a larger opened-development run.  There is no
margin/weight/negative-selection scan.  Failure rejects this minimal
counterfactual refinement objective.
