# Llama-3.1 writer data scaling and packet-localization results

## Outcome

The fixed N=512 diversity experiment produced a useful negative result for
post-hoc packet compression and a positive result for full-state learning.
More training cases did not degrade the four-slot state.  They improved it
when all layers were available, while making that state much less recoverable
from the previously fixed last-16-layer packet.

Accordingly, the preregistered Hotpot continuation gate failed and no new
Hotpot run was launched.  The result motivates end-to-end packet-aware state
placement rather than another slot/rank/layer sweep.

## Frozen training and first gate

The 512-case dataset is a strict superset of the original 128 cases, adds 384
seed-2041 cases, and has zero ID overlap with the frozen 256-case test.  Both
writers use four slots, rank 4, 1,600 updates, learning rate `5e-4`, and the
same stage/bridge loss.  Thus N=512 receives one quarter as many epochs at the
same update budget.

On the pre-existing 32-case MuSiQue dev gate with all-layer BF16, N=512
improved stage F1 from `.4438` to `.5469` and bridge F1 from `.6583` to
`.7333`, passing the preregistered development threshold.

## Fixed last-16 INT4 test

On 256 ID-disjoint MuSiQue-train cases, at the operating point frozen before
N=512 training:

| writer / source | EM | F1 | bridge F1 | payload |
|---|---:|---:|---:|---:|
| no-summary | .121 | .230 | -- | 0 |
| N=128, shift-1 | .152 | .242 | .000 | 67,584 B |
| **N=128, correct** | **.465** | **.551** | .151 | 67,584 B |
| N=512, shift-1 | .141 | .226 | .001 | 67,584 B |
| **N=512, correct** | **.266** | **.349** | .009 | 67,584 B |

N=512 versus N=128 had F1 delta `-.2019`, bootstrap 95% CI
`[-.2661,-.1372]`, and EM delta `-.1992`, CI `[-.2656,-.1289]`.  The Hotpot
gate therefore failed.  N=512 correct source still beat its shift-1 control by
`+.1227` F1, CI `[+.0842,+.1635]`, and no-summary by `+.1188`, CI
`[+.0805,+.1602]`; it retained causal source information rather than collapsing
to a task prior.

## Frozen post-result 2 x 2 diagnosis

Before any additional evaluation, the mechanism diagnostic was frozen to
report both writers under all combinations of all/last-16 layers and
BF16/INT4.  Results on the same 256 paired cases are:

| writer | all BF16 | all INT4 | last16 BF16 | last16 INT4 |
|---|---:|---:|---:|---:|
| N=128 F1 | .760 | .750 | .545 | .551 |
| N=512 F1 | **.859** | **.829** | .351 | .349 |
| N=512 minus N=128 | **+.099** | **+.079** | -.194 | -.202 |

The N=512-minus-N=128 paired F1 intervals are:

- all BF16: `[+.0495,+.1498]`;
- all INT4: `[+.0257,+.1338]`;
- last16 BF16: `[-.2571,-.1319]`;
- last16 INT4: `[-.2661,-.1372]`.

Within N=512, all-layer BF16 to all-layer INT4 costs only `.0303` F1,
CI `[-.0501,-.0127]`.  In contrast, all-layer INT4 to last-16 INT4 costs
`.4800`, CI `[-.5449,-.4140]`.  N=128 also loses `.1993` from the same layer
operation, but N=512 loses far more (`.4800`).

This rules out two tempting explanations:

1. fixed-update data scaling did not simply harm held-out generalization;
2. INT4 is not the primary reason the fixed packet failed.

The supported interpretation is that unconstrained training distributes
useful state across the receiver's depth, and increased data makes that
distributed solution stronger.  Post-hoc layer deletion is therefore not a
stable compression rule: representation quality and packet compressibility
can move in opposite directions.

## Real INT4 packet check

The N=128 last-16 arm was also passed through the implemented signed-nibble
codec with FP16 per-token/per-head scales, layer indices, framing, and CRC.
Its actual wire size is 67,642 B.  Real-codec F1 was `.5544` versus `.5505`
for fake INT4; paired delta `+.0039`, CI `[-.0078,+.0195]`, with 253/256 F1
ties.  The preregistered codec-equivalence gate passes.

On H200, the unoptimized Python reference implementation measured mean
GPU-to-bytes pack `4.65 ms` and bytes-to-GPU unpack `2.71 ms` over 200
iterations.  These are correctness/reference costs, not a fused-kernel claim.

## Method decision

The next frozen intervention is packet-aware writing: expose receiver losses
to the exact last-16 mask and straight-through INT4 during training, without
changing slots, rank, data, optimizer, or update count.  If it succeeds on a
new non-overlapping confirmation set, the central method claim becomes:

> A compact KV state should be written under its deployment budget, not
> obtained by pruning an unconstrained state after training.

Full protocol: `docs/llama31_packet_aware_writer_preregister_20260817.md`.
Machine-readable factorial:
`outputs/amortized_semantic_kv/llama31_writer_scaling_layer_quant_factorial_20260817/analysis.json`.
