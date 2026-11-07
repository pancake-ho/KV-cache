# Llama-3.1 first-half packet confirmation

## Decision

All three preregistered confirmatory gates pass on a newly frozen 256-case set
with zero ID overlap against both the 512 writer-training cases and the prior
256-case diagnostic.  The selected first16 INT4 packet is therefore supported
as a 67.6 KB operating point for this Llama writer and task distribution.

It is not lossless: the 135 KB all-layer packet remains substantially more
accurate.  The result establishes a quality/payload frontier and replicates
depth localization, not dominance of one packet.

## Dataset audit

- 256 unique cases, seed 2053;
- zero overlap with 512 training IDs;
- zero overlap with 256 prior test IDs;
- 99 unique bridge-answer clusters;
- mean Agent-A context: 6,043 tokens;
- file SHA256:
  `797cf5083fec9f5ae140e69b6376f900e417983645c6ba5d9b50cdac393fd4be`;
- ordered-ID SHA256:
  `2bcaaa6d1237425e17a0b18421e48c4111f307174bce1ceeeccde4fef2776204`.

## Results

All capsule arms use the same frozen N=512 Llama checkpoint and INT4.  The
only correct-source intervention is transmitted depth.

| arm | EM | F1 | bridge EM | bridge F1 | payload |
|---|---:|---:|---:|---:|---:|
| no-summary | .148 | .255 | -- | -- | 0 |
| shifted first16 | .152 | .252 | .000 | .007 | 67,584 B |
| correct last16 | .270 | .362 | .000 | .008 | 67,584 B |
| **correct first16** | **.570** | **.677** | .098 | .170 | **67,584 B** |
| **correct all layers** | **.812** | **.880** | .270 | .895 | 135,168 B |

The actual framed signed-nibble packet is 67,642 B for either 16-layer mask;
the table uses the existing tensor-plus-scale accounting for comparability.

## Frozen gates

First16 versus no-summary:

- F1 delta `+.4216`, ordinary bootstrap CI `[+.3546,+.4880]`;
- bridge-cluster bootstrap CI `[+.3380,+.5004]`;
- EM delta `+.4219`, 123 candidate-only versus 15 reference-only,
  McNemar `p=2.84e-22`.

First16 versus shift-1:

- F1 delta `+.4246`, ordinary CI `[+.3641,+.4845]`;
- cluster CI `[+.3494,+.5008]`;
- EM delta `+.4180`, 116 candidate-only versus 9 reference-only,
  `p=7.78e-25`.

First16 versus last16:

- F1 delta `+.3145`, ordinary CI `[+.2422,+.3856]`;
- cluster CI `[+.2178,+.4081]`;
- EM delta `+.3008`, 106 candidate-only versus 29 reference-only,
  `p=1.66e-11`.

All lower bounds are positive, so net utility, source causality, and depth
localization all replicate without further layer selection.

## Frontier and mechanism

First16 halves the all-layer payload but loses `.2032` F1, CI
`[-.2526,-.1551]`; it is a lower-bandwidth Pareto point rather than a
near-lossless compression.  Conversely, first16 and last16 have identical
bytes but differ by `.3145` F1.  Packet quality therefore depends on *where in
depth* state is made available, not just the number of K/V vectors.

The first16 packet has relatively weak generic bridge readout (`.170` F1) but
strong downstream question answering (`.677`).  The receiver can acquire
task-useful state in early layers and propagate it through its own later
computation without retaining capsule K/V there.  Last16 exposes state only
after much of the query computation has already occurred and is far weaker.

Together with the N=512 all/last factorial, this supports a more precise
model of short-summary KV:

> Four slots are not a single four-vector summary.  They form a depth-indexed
> state channel.  Early K/V controls when the receiver can ingest the state;
> later K/V refines or lexicalizes it.  Layer routing is therefore part of the
> codec topology.

This also explains why always-masked last-half training was a weak remedy: it
optimized a structurally late channel rather than choosing the functional
injection depth.

## Limits and next step

The confirmation cases are training-ID-disjoint but share the MuSiQue-train
source distribution and relation pool.  They do not establish cross-dataset
generalization.  The first16 packet is now frozen for the untouched LongBench
2WikiMQA-200 test specified in
`docs/llama31_2wikimqa_first16_preregister_20260817.md`.

Machine-readable comparisons:
`outputs/amortized_semantic_kv/llama31_writer_n512_confirm256_first16_analysis_20260817/`.
