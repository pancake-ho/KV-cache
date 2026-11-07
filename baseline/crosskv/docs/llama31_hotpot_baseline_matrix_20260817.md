# Llama-3.1-8B Hotpot E290 common-baseline matrix

Date: 2026-08-17

This matrix puts the fixed 4-slot capsule on the same rows, model, prompts,
hardware, answer normalization, and paired statistics as plaintext handoff and
full recomputation. It is a systems/quality coordinate system for subsequent
CacheBlend and KVPacket runs, not a claim that these arms have identical
information assumptions.

## Quality

| arm | EM | F1 | information available to Agent B |
|---|---:|---:|---|
| no summary | 10.7 | .195 | question only |
| shifted capsule | 8.62 | .154 | wrong source's 4-slot state |
| correct capsule | **15.9** | **.304** | 4-slot pre-answer state |
| generated Tail-KV | 21.4 | .310 | KV of Agent A's generated answer |
| generated-answer plaintext | 33.4 | .468 | Agent A's answer string |
| full target recompute | 46.2 | .597 | complete dossier and question |
| source full-KV continuation upper bound | 49.0 | .617 | complete Agent A prompt state; no B policy change |
| gold-answer plaintext oracle | 57.2 | .652 | dataset gold string |

Generated-answer plaintext exceeds capsule by `+.164` F1, paired 95% CI
`[+.104,+.224]`. Full recomputation exceeds capsule by `+.294`, CI
`[+.240,+.347]`. These are real quality gaps and must remain visible in the
paper.

The source full-KV row is an upper-bound continuation of Agent A's own prompt,
not a policy-compatible handoff into a different Agent B system prompt. It
nevertheless gives the correct scale for a method that transports the entire
joint source state.

## Transport

| arm | mean transmitted state |
|---|---:|
| generated-answer plaintext | 5.12 model tokens / 18.6 UTF-8 bytes |
| 4-slot capsule, last16 INT4 | 67,584 B |
| generated Tail-KV, all32 INT4 | 175,369 B |
| full source KV, theoretical all32 INT4 | 320.1 MB |
| full source KV, BF16 | 1.242 GB |

The capsule is about 4,736 times smaller than theoretical INT4 full KV and
18,372 times smaller than BF16 full KV. It is, however, about 3,629 times
larger than the short plaintext answer in actual string bytes. Therefore this
benchmark supports compression relative to document/full state, not a network
size win over a decoded short answer.

## Synchronized latency

The source dossier prefill is shared and excluded from the first three paths.

| path after source prefill | mean latency |
|---|---:|
| capsule emission + B prepare/generate | **253.3 ms** |
| A answer decode + B plaintext prefill/generate | 359.6 ms |
| A answer decode + Tail-KV prepare/B generate | 400.3 ms |
| B full-dossier recompute/generate | 830.4 ms |

The capsule saves 106.3 ms (`29.6%`) relative to generated-answer plaintext
in this synchronized single-request HF path because it replaces Agent A's
221.3 ms answer decoding with a 68.4 ms state-emission operation. This is a
latency win paired with lower answer quality and far more network bytes than
plaintext. It is not yet an end-to-end serving result.

## Artifacts

- evaluator: `src/xmodel_kv/cli/evaluate_hotpot_plaintext_baselines.py`;
- matrix analyzer: `src/xmodel_kv/cli/analyze_hotpot_baseline_matrix.py`;
- three baseline shards:
  `outputs/amortized_semantic_kv/llama31_hotpot_e290_plaintext_baselines_shard{0,1,2}_20260817/`;
- joined rows/statistics:
  `outputs/amortized_semantic_kv/llama31_hotpot_e290_baseline_matrix_20260817/`.
