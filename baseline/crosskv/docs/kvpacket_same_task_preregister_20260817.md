# KVPacket same-task baseline protocol

Status: frozen before same-task KVPacket RECORD/training/evaluation.

## Purpose

Measure KVPacket on the same Llama-3.1-8B MuSiQue-to-Hotpot setting as the
4-slot state capsule. KVPacket and the capsule solve different transport
problems: KVPacket preserves independently cached full documents, whereas the
capsule summarizes a jointly computed source state. The comparison will
therefore report information assumptions, quality, payload, TTFT, and online
FLOPs rather than claiming they are interchangeable algorithms.

## Fixed method configuration

- unmodified KVPacket public `HFBackend`, `PacketSession`, `PacketTrainer`, and
  `evaluate_sample` paths;
- model: `meta-llama/Llama-3.1-8B-Instruct`, BF16, eager attention;
- one shared document wrapper with 8 learned header and 8 learned trailer
  tokens, float32 parameters initialized from model embeddings;
- teacher-KL objective, token-mean reduction, AdamW learning rate `1e-3` and
  zero weight decay;
- 5 epochs, maximum batch 64, target 4,096 teacher tokens per step,
  forward batch size 1, gradient checkpointing;
- deterministic seed 42; final checkpoint only; no layer, wrapper-length,
  learning-rate, epoch, or checkpoint selection.

The configuration is copied from KVPacket's maintained HotpotQA recipe except
that its Llama-3.2-3B model is replaced with the paper's shared
Llama-3.1-8B model and training data are fixed to the same 128 MuSiQue source
dossiers used by the capsule. RECORD uses Agent A's exact-answer task and at
most 24 generated teacher tokens.

## Fixed evaluation

- dataset: all 290 question-disjoint HotpotQA-E rows already frozen for the
  model-family replication;
- target prompt: the same Agent B system, dossier, question, and exact-answer
  instruction used by the full-recompute common baseline;
- context split mechanically at each `Passage N:` boundary, preserving the
  concatenated source text exactly;
- methods: KVPacket's native `full_recompute`, `no_recompute`, and `kv_packet`;
- greedy decode, at most 24 tokens;
- three execution shards: `(0,97)`, `(97,97)`, `(194,96)` on GPUs 0, 1, 2;
- primary quality metric: the same per-example LongBench-style token F1 and
  exact match as the capsule matrix;
- systems metrics: native KVPacket TTFT and online FLOPs, plus theoretical BF16
  payload for document bodies and 16 wrapper tokens per document.

No result on these 290 questions may be used to tune the wrapper. An OOD
failure is reported as a failure; a same-task Hotpot training sweep, if later
run, must be labeled separately.

## Integration scope

The only added code is a dataset/protocol driver at
`KVPacket/tasks/multi_doc/local_handoff.py`. It does not import or execute the
capsule implementation and uses KVPacket's native preprocessing, re-rotation,
session, and training code.
