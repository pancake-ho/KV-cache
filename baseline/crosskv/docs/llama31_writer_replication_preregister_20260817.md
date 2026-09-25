# Llama-3.1-8B state-writer replication protocol

Status: frozen before Llama writer training.

## Purpose

Test whether the pre-answer state-emission mechanism is specific to Qwen3-8B or
repeats in a second model family.  This is a model-family replication, not a
cross-model KV adapter: sender and receiver are both the same Llama model.

## No-search training configuration

Every optimization choice is copied from the final Qwen checkpoint without a
Llama sweep:

- model: `meta-llama/Llama-3.1-8B-Instruct`;
- MuSiQue train data: `train128_seed2028.jsonl`, all 128 cases;
- 4 learned slots;
- rank-4, scale-1 sender-only writer;
- 1,600 steps, AdamW lr `5e-4`, weight decay `.01`;
- stage weight 1, bridge weight 2, no teacher KL;
- resident detached source caches offloaded to CPU;
- BF16 base computation;
- fixed validation: MuSiQue dev indices 64--95;
- save the final step only; no early stopping or checkpoint selection.

Validation is diagnostic.  No hyperparameter may be changed and still belong
to this replication.  If the final writer produces no correct-source signal,
the run is reported as a negative model-family boundary.

## Pre-specified layer mapping

Qwen has 36 layers and the confirmed Hotpot packet uses the last 18.  Llama has
32 layers, so the architecture-normalized operating point is fixed in advance
as **last 16 layers + INT4**.  No first/last curve will be used to select this
number.

If the MuSiQue validation shows a nontrivial writer, the final checkpoint will
be evaluated without Llama-specific tuning on the already fixed Hotpot
protocol.  Since the Hotpot data have been inspected for the Qwen study, that
result is a model replication on a shared benchmark, not a new untouched data
confirmation.  Correct-source versus shifted-source remains the causal arm.

## Frozen execution instantiation

The fixed MuSiQue dev32 diagnostic passed before any Hotpot execution: the
all-layer BF16 capsule obtained downstream EM/F1 `0.40625/0.44375`, versus
`0.21875/0.265625` without a capsule, and intermediate-state F1 `0.65833`.
This only authorizes the pre-specified external test; it is not the external
claim itself.

The model-family replication therefore uses the same 290-question HotpotQA-E
set that is question-disjoint from the standard Hotpot200 analysis.  It is
split mechanically into `(offset,count) = (0,97), (97,97), (194,96)` on GPUs
0, 1, and 2.  Within every shard, shifted source is the next case modulo that
shard.  Both `state_readout` and `question_conditioned` protocols are run.
Capsule transport is fixed to `last_16` layers at INT4; generated-answer
Tail-KV uses all 32 layers at INT4.  All 290 rows will be pooled, with no
checkpoint, layer, example, or seed selection.
