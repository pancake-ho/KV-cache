# Llama-3.1 packet-aware state writer protocol

Status: frozen after the N=128/N=512 mechanism factorial and before training
the packet-aware arm or constructing its confirmation set.

## Motivation and hypothesis

Increasing writer training diversity from 128 to 512 cases improved the
four-slot state when every layer was transmitted, but made post-hoc layer
compression substantially worse.  On the fixed 256-case diagnostic, N=512
versus N=128 changed F1 by `+.0989` in all-layer BF16 and `+.0788` in
all-layer INT4, yet by `-.2019` in last-16 INT4.  Within N=512, replacing
all-layer INT4 with last-16 INT4 lost `.4800` F1; replacing BF16 with INT4 at
all layers lost only `.0303`.

The frozen hypothesis is therefore that the optimization objective, rather
than four-slot capacity, is the immediate bottleneck: unconstrained training
distributes task state across layers that will later be discarded.  A
packet-aware writer should learn to place useful state directly into the
layers and precision available on the wire.

## Single intervention

Copy the completed N=512 Llama training run exactly:

- frozen `meta-llama/Llama-3.1-8B-Instruct` in BF16;
- the identical 512-case strict superset training dataset;
- four shared state-emission slots and a rank-4, scale-1 sender-only writer;
- 1,600 AdamW updates, learning rate `5e-4`, weight decay `.01`;
- stage weight 1, bridge weight 2, no teacher KL;
- seed 2027 and resident detached source caches on CPU.

Change only the cache seen by the receiver losses.  After RoPE relocation,
zero capsule K/V outside the last 16 of 32 layers and fake-quantize the active
K/V to signed INT4 with the existing per-layer/K-or-V/head/token FP16 scale.
Use a straight-through estimator for rounding, so the forward pass exactly
matches fake INT4 while gradients pass through active vectors.  Unsent layers
receive no cache gradient.  Do not change slots, rank, loss weights, learning
rate, update count, or select an intermediate checkpoint.

The base Transformer and native K/V projections remain frozen.  The wire
format and receiver architecture do not change; this is state placement, not
a learned dense decoder.

## Development gate

Use the already inspected MuSiQue dev indices 64--95 only as a development
gate.  Evaluate the unconstrained N=512 checkpoint and the final packet-aware
checkpoint in all four cells of `all,last_16` by `BF16,INT4`; report every
cell.  Continue to a new confirmation set only if packet-aware last-16 INT4
improves paired F1 over unconstrained N=512 last-16 INT4 by at least `.10` and
does not fall below no-summary F1.  This gate is not confirmatory evidence.

## Fresh confirmation set

Only after the development gate passes, construct 256 new MuSiQue-train cases
with seed 2053 and the existing eight-candidate, minimum-24k-character builder.
Explicitly exclude the union of:

- `train512_superset_seed2041_exclude_test256.jsonl` (all writer training IDs);
- `test256_traincases_seed2031_exclude_train128.jsonl` (the inspected test IDs).

Assert 256 unique IDs and zero overlap with both inputs before evaluation.
Freeze the resulting JSONL and summary; do not resample based on outcomes.

On that set run:

1. unconstrained N=512, correct source, all four layer-by-precision cells;
2. packet-aware N=512, correct source, all four cells;
3. packet-aware N=512, circular shift-1 source, last-16 INT4 only.

Primary endpoint is paired F1 of packet-aware versus unconstrained at
last-16 INT4.  Required causal controls are packet-aware correct source versus
its shift-1 source and versus the row-matched no-summary arm.  The method gate
passes only if all three bootstrap 95% CIs have positive lower bounds.  Report
paired exact McNemar statistics, bridge readout, all-layer tradeoffs, and all
four unshifted cells regardless of outcome.  The fixed packet is 4 slots x 16
layers x INT4; no cell may be selected as a replacement operating point.

No Hotpot or other already inspected result may substitute for this fresh-set
gate.  A passing result authorizes a separately labeled cross-dataset
replication; a failure rejects this packet-aware configuration without a
layer/rank/slot sweep on the confirmation set.
