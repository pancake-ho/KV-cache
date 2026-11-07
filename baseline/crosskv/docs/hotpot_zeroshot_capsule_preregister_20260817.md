# Frozen cross-dataset test: MuSiQue-trained KV capsule on HotpotQA

Status before execution: **protocol frozen; no result from the 200-case run has
been inspected**.  The two-case implementation smoke is excluded from every
reported estimate.

## Question

Does the fixed MuSiQue-trained state-emission capsule transfer task-relevant
state from a long HotpotQA RAG context without any HotpotQA gradient update?
This is a cross-dataset test of state writing, not a claim of a previously
untouched project benchmark: the same LongBench files were used by earlier raw
history-KV experiments, but never to train or select this capsule checkpoint.

## Frozen artifacts

- dataset: all 200 rows of `/data/datasets/longbench/hotpotqa.jsonl`, in file
  order;
- model: `Qwen/Qwen3-8B`, BF16/SDPA;
- checkpoint:
  `outputs/amortized_semantic_kv/musique_softcapsule_writer_r4_lr5e4_bridge2_s4_n128_val64_95_steps1600_20260817/capsule.pt`;
- capsule path: 4 soft slots, rank-4 sender-only writer, first 30 layers, INT4;
- generated-answer Tail-KV: greedy answer, all 36 layers, INT4;
- generation limits: 24 tokens for Agent A and Agent B;
- causal control: circular shift by one **inside each fixed execution shard**.

The 200 examples are split only for three-GPU execution as offsets/counts
`0/67`, `67/67`, and `134/66`.  Every example receives exactly one different
source in the shifted arm.  No example, prompt, layer, quantization, or decoding
choice may be removed or changed after inspecting outcomes.

## Frozen protocols and arms

For both protocols, Agent A receives the complete LongBench dossier and
question under the same short-answer source prompt used during MuSiQue capsule
training.  The capsule is emitted before any answer token.  Agent A is also
greedily decoded to provide the source-solvability estimate and post-answer
Tail-KV upper bound.

1. `state_readout`: Agent B receives the exact generic receiver system/query
   used by the MuSiQue bridge loss.  It does not receive the HotpotQA question.
   This isolates whether the capsule itself writes the computed answer state.
2. `question_conditioned`: Agent B receives the original question but not the
   long dossier.  This tests the intended RAG-summary use case.

Each protocol has four arms:

- `no_summary`;
- correct-source 4-slot capsule;
- circularly shifted 4-slot capsule;
- correct-source generated-answer Tail-KV.

The full-context Agent-A answer is reported as a source-computation upper
bound.  Tail-KV is not forced to the capsule's first-30 layer mask: doing so
would change its previously established operating point and unfairly weaken
the baseline.

## Endpoints and decision gates

All metrics use the existing HotpotQA normalized EM and token F1.

Primary causal gate:

- in `state_readout`, correct capsule must exceed shifted capsule, with paired
  F1 bootstrap 95% CI excluding zero and two-sided exact McNemar on EM;
- this gate tests source-conditioned OOD state emission independently of
  question-only parametric knowledge.

Primary RAG-utility gate:

- in `question_conditioned`, correct capsule must improve over `no_summary` and
  shifted capsule in mean F1, with paired bootstrap intervals;
- source answer and generated Tail-KV define achievable quality under the same
  full-context source computation.

Secondary reporting includes EM, wins/losses/ties, source-token length,
capsule-emission latency, source answer decode latency, receiver generation
latency, fixed capsule payload (126,720 bytes), and measured mean Tail-KV
payload.  Latencies are synchronized single-request measurements and will not
be presented as a serving-throughput benchmark.

If either capsule gate fails, the result is a method boundary: the current
writer is MuSiQue-distribution-specific.  It must not be hidden by retraining on
these 200 cases.  A subsequent Hotpot-trained model requires a new disjoint
split and must be labeled domain adaptation rather than zero-shot transfer.
