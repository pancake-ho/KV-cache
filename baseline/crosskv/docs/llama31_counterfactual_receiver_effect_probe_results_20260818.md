# Counterfactual full-receiver effect probe results

## Decision

The frozen explanation gate fails in the opposite direction.  State8 is
significantly *closer* than CE8 to the plaintext handoff's marginal effect on
the complete on-policy receiver trajectory, yet remains significantly worse
on final F1.  No receiver-effect distillation candidate is authorized.

## Frozen result

All rows use the existing MuSiQue confirm cases 32--95, canonical all-layer
K4/V4, 20,000 paired bootstrap resamples, and seed 2099.

| arm | effect total | direction loss | log-norm loss | real-codec final F1 |
|---|---:|---:|---:|---:|
| state4 | .83342 | .80765 | .25764 | .88281 |
| CE8 | .75902 | .74677 | .12240 | **.92188** |
| state8 | **.72595** | **.71445** | **.11495** | .87240 |

State8-minus-CE8 paired contrasts are:

- total effect loss: `-.03307`, CI `[-.03803,-.02806]`; state8 is lower on
  62/64 cases;
- direction loss: `-.03232`, CI `[-.03730,-.02737]`;
- log-norm loss: `-.00745`, CI `[-.01165,-.00336]`;
- final F1: `-.04948`, CI `[-.10417,-.00781]`.

The state8-minus-CE8 effect-loss delta versus F1 delta Pearson correlation is
`-.128`.  F1 differs on only four cases, whereas the effect ordering is almost
universal.  State4 is farther than CE8 in total effect on all 64 cases
(`+.07440`, CI `[+.06491,+.08517]`), so the metric partly tracks gross
capacity but cannot distinguish the two equal-capacity solutions that matter.

## Interpretation

This result rules out a progressively broader sequence of representation
proxies for this decision:

1. per-position native K/V coordinates rank state8 above CE8;
2. a fixed-query K/V attention operator ranks state8 above CE8;
3. the marginal hidden trajectory under the complete prefix, on-policy
   receiver queries, and an explicit plaintext handoff teacher also ranks
   state8 above CE8.

The failure is therefore not merely omitted prefix competition or frozen
queries.  Native-state training appears to make the packet broadly more
token-/text-like while removing a small set of readout-critical directions
that decide four cases.  Averaging semantic similarity over layers and hidden
dimensions rewards the wrong invariances.  Another coordinate/operator/hidden
loss or weight scan is not justified.

The positive method must optimize downstream decisions directly while
changing data diversity or the state-writing structure.  A simple
correct-versus-shift NLL hinge has already failed in the frozen progressive
4+4 pilot, so it is not a new authorized fallback either.

## Artifact

`outputs/amortized_semantic_kv/llama31_native_tail_scaleup_state8_800_train128_eval64_20260818/receiver_effect_probe64.json`
