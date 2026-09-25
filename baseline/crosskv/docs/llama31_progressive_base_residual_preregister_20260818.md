# Llama-3.1 progressive base/residual preregistration

Status: frozen before constructing either new dataset and before any full
training or evaluation run.

## Question

Two shared-writer multi-budget variants failed on `budgetdev128`.  Stochastic
masking with a common loss degraded first16/all F1 by `.100/.124` relative to
an all-only control.  Removing lexical supervision from first16 updates still
degraded them by `.138/.113`.  Both retained strong correct-source versus
shifted-source causality.

Test the narrower mechanism hypothesis that receiver budgets interfere because
deep losses backpropagate through the early emitted state.  Train a progressive
packet in which layers 0--15 are an early functional base and layers 16--31 are
an optional deep residual.  Deep updates may consume the numerical early state
but cannot update its sender graph or emitted K/V; compact updates cannot update
the deep adapter blocks.

This protocol is confirmatory for the exact progressive representation chosen
after the exhausted `budgetdev128` experiments.  No loss-weight, split-point,
step-count, checkpoint, rank, slot, or mixture scan is allowed on the new
confirmation set.

## New data, frozen before construction

Use the official MuSiQue-answerable train source and the existing handoff
builder, with `case_split=train` and train-split decoys, matching every prior
N512/test/confirm/development case.  The previously used union consists of
1,152 unique IDs:

- N512 writer training set (512);
- old test256 (256);
- fresh first16 confirm256 (256);
- exhausted budgetdev128 (128).

Construct, in this order:

1. exactly 256 unique incremental training cases with seed 2081, excluding all
   1,152 IDs above;
2. exactly 128 unique confirmation cases with seed 2083, excluding all 1,152
   prior IDs and the new 256 training IDs.

Require zero train/confirmation overlap and zero overlap with every prior set.
Record ordered-ID SHA256, file SHA256, eligible counts, document lengths and
bridge-answer cluster counts before loading a model.  The untouched
`2wikimqa_e` file remains sealed.

## Fixed parent and candidates

Parent checkpoint:

`outputs/amortized_semantic_kv/llama31_n512_allonly_int4_budgetdev128_fixed1600_20260817/capsule.pt`

Its SHA256 is
`b5848635fac01561b8f694aeca79d5d02d340a5c15eeb1e3967d21c90cb0b620`.

Both candidates use frozen BF16 Llama-3.1-8B, four slots, rank-4 scale-1
sender adapter, the exact parent initialization, AdamW learning rate `5e-4`,
weight decay `.01`, gradient clip 1, INT4 STE, seed 2027, 3,200 optimizer
updates, and detached resident source caches on CPU.  Only the final checkpoint
is evaluated.

1. **Compute-matched all-only continuation.** Every update exposes all 32
   layers with stage weight 1 and bridge weight 2.
2. **Progressive base/residual.** Draw uniformly and independently per update
   from `first_16` and `all`.  First16 uses stage weight 1 and bridge weight 0.
   All uses stage weight 1 and bridge weight 4.  On all updates, detach emitted
   K/V for layers 0--15 and detach sender hidden state before the layer-16
   residual block.  Record realized draw counts.

The two runs therefore have equal optimizer-update and receiver-forward
budgets.  The progressive branches each receive about half the updates; this
is part of the frozen method, not corrected after inspection.

## Evaluation and gates

Evaluate the parent, compute-matched control, and progressive candidate on all
128 confirmation cases with correct-source `first_16 + INT4` and `all + INT4`.
Run shift-1 source assignment for the progressive candidate.  Stage F1 is
primary.  Report EM, bridge readout, no-summary, payload, training time, and
paired ordinary plus bridge-answer-cluster bootstrap intervals with 20,000
replicates.

The progressive method passes only if all conditions hold:

1. first16 F1 versus the compute-matched all-only continuation improves by at
   least `.03`, and the ordinary paired 95% CI lower bound is positive;
2. all-layer F1 is non-inferior to that control at margin `-.05`, so the paired
   95% CI lower bound is greater than `-.05`;
3. correct-source first16 beats progressive shift-1 with a positive paired F1
   CI lower bound.

Cluster bootstrap must agree in sign with gates 1 and 3 and must remain above
the `-.05` margin for gate 2.  Parent comparisons and oracle-union accuracy are
secondary.  If any quality gate fails, report the exact representation as a
negative result and do not try another split point or variant on these 128
cases.  If all pass, the claim is limited to same-model progressive packet
training and still requires cross-task and serving confirmation.

## Implementation audit frozen before data

- `SoftTailCapsule.materialize(..., gradient_boundary_layers=16)` detaches the
  input of the first deep writer block while preserving its parameter
  gradients.
- `detach_cache_layer_prefix(..., 16)` cuts direct receiver gradients through
  early emitted K/V without changing any values.
- A four-layer unit audit confirms zero early/embedding gradient and nonzero
  deep-block gradient under a deep loss.
- A real Llama two-update smoke confirms `all -> boundary 16` and
  `first_16 -> no boundary`, compatible checkpoint saving, loading, and
  evaluation.
- Focused tests before data construction: `17 passed`.

## Preconstruction source-path audit

The first builder invocation used the official answerable dev file because its
legacy required argument is named `--dev`.  The builder rejected the request
before opening an output file: only 197 dev-source cases are eligible, so 256
could not be sampled.  No dataset was created and no model was loaded.

Read-only inspection then confirmed that all historical dataset summaries use
`case_split=train`, that their IDs occur in the official answerable train
file, and that this source reproduces the historical 1,639 eligible rows.  The
construction input is therefore corrected to
`musique_ans_v1.0_train.jsonl` for both the legacy `--dev` case-source argument
and the `--train` decoy-pool argument.  This correction changes no candidate,
seed, exclusion, sample count, metric, or gate and is recorded before any new
data exists.

## Frozen construction audit

Construction completed before loading a model:

- incremental train256: 256 unique cases selected from 487 eligible rows after
  excluding exactly 1,152 prior IDs; 116 unique bridge answers; mean dossier
  length 24,264.73 characters; ordered-ID SHA256
  `62d5ee7106eeafd4179e172336d26a5c1dc0468cdf039e464dcbb2fdc299e882`;
  file SHA256
  `c0c5ec89ffe569fef6b94e387c1b4ceb42e271787cc0686a0cc249d6dc4ffa1b`;
- confirmation128: 128 unique cases selected from 231 eligible rows after
  excluding exactly 1,408 prior plus incremental-training IDs; 73 unique
  bridge answers; mean dossier length 24,283.66 characters; ordered-ID SHA256
  `5f611bc088e017013e9b19cea4f7fab9c2ee1bdca8b52eb17a68c9d7eb851663`;
  file SHA256
  `1900657b9479fcb080145e8c489b5fd478234d69d1d62f5a5797a73b5e44abef`.

The two new sets have zero ID overlap, and their union has zero overlap with
all 1,152 prior IDs.  Frozen paths:

- `datasets/musique_handoff/train256_increment_seed2081_exclude_used1152.jsonl`;
- `datasets/musique_handoff/confirm128_seed2083_exclude_used1408.jsonl`.
