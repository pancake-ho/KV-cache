# Short semantic-KV oracle probe (2026-08-17)

## Question

Can an already-prefilled Agent-A state be replaced by a very short, receiver-native
KV state that lets Agent B behave as if it had read a plaintext handoff summary?

This experiment is an **existence upper bound**, not an amortized compressor. The
Qwen3-8B weights are frozen and a fresh set of K/V tensors is optimized for each
conversation. No soft-token or summary-text prefill is used on the student path.

## Setup

- Model: `Qwen/Qwen3-8B` (36 layers, 8 KV heads, head dimension 128).
- Data: the first four CoQA validation conversations, eight facts/questions per
  conversation.
- Agent A reads the full story (299--442 story tokens). Its contextual KV segment is
  mean-pooled in de-RoPE content space and rerotated behind Agent B's system prefix.
- The plaintext teacher receives a gold structured briefing containing the same eight
  question-answer facts (103--129 briefing tokens; 127--153 cache tokens including
  chat framing).
- Only the compact per-instance K/V tensors are optimized. Loss is gold-answer CE plus
  KL distillation from the plaintext teacher, with 200 single-question updates.
- Budgets are measured in sequence slots. One Qwen3-8B BF16 KV slot is 144 KiB across
  all layers; four slots are 576 KiB.

The implementation is in:

- `src/xmodel_kv/semantic_kv_summary.py`
- `src/xmodel_kv/cli/probe_semantic_kv_summary.py`
- `tests/test_semantic_kv_summary.py`

## A. Task-conditioned existence test

All eight downstream questions participate in oracle optimization; generation is then
evaluated on those same eight distinct questions.

| KV slots | Mean slot compression vs teacher briefing | Initial F1 | Optimized F1 | Teacher KL/token |
|---:|---:|---:|---:|---:|
| 4 | 33.63x | 0.169 | **0.890** | **0.041** |
| 8 | 16.81x | 0.174 | 0.647 | 0.123 |
| 16 | 8.41x | 0.169 | 0.803 | 0.046 |
| 32 | 4.20x | 0.179 | 0.834 | 0.040 |

Common baselines across these rows:

- Plaintext briefing teacher F1: **0.953**.
- No-summary Agent-B F1: **0.132**.

Thus four direct KV slots recover 93.3% of the plaintext teacher's F1 while replacing
about 134.5 teacher-briefing cache slots on average. This is strong evidence that a
very short **task-conditioned handoff state** exists.

The non-monotonic slot curve is an optimization warning, not evidence that four slots
have greater intrinsic capacity. Direct K/V optimization has 73,728 scalars per slot;
larger budgets used the same learning rate and step count and were more sensitive to
local optimization failures.

## B. Held-out-question test

Only odd turns 1/3/5/7 participate in optimization. Even turns 2/4/6/8 are never used
for gradients and test whether the compact state retained reusable story semantics.

| KV slots | Fit-question F1 | Held-out F1 | No-summary held-out F1 | Teacher held-out F1 |
|---:|---:|---:|---:|---:|
| 4 | **0.899** | **0.250** | 0.127 | 0.938 |
| 16 | 0.847 | 0.101 | 0.127 | 0.938 |
| 32 | 0.474 | 0.084 | 0.127 | 0.938 |

Four slots retain a small amount of transferable information (0.250 vs. 0.127), but
the gap to the plaintext briefing is large. Larger budgets overfit or optimize
unstably under this first recipe.

## C. Query-agnostic self-study test

The stricter test never uses a CoQA question, answer, or answer-conditioned teacher
logit for optimization. For each story, a full-context teacher answers four fixed,
generic prompts:

1. factual synopsis;
2. entities, attributes, and relationships;
3. chronological event outline;
4. concrete downstream notes such as colors, counts, ownership, and locations.

The compact KV is optimized only to reproduce the tokens and distributions of those
four responses. All eight CoQA questions per story are then evaluated as unseen
queries. The frozen model, source-pool initialization, and direct receiver-native KV
representation are unchanged. This scan uses 16 validation conversations (128 unseen
questions), 300 updates, learning rate 0.01, and token-normalized CE plus KL.

Two baselines distinguish representation failure from missing teacher information:

- No-summary Agent B: **0.144 F1**.
- The same four self-study responses concatenated as plaintext: **0.470 F1**.
- The answer-conditioned gold briefing: **0.963 F1**, included only as a leaked-label
  ceiling and not as a fair self-study baseline.

The four generic responses produce an average 420.1-token plaintext handoff prefix.
The source cache segment averages 373.4 slots. Results below use one equally weighted
F1 per conversation. The confidence intervals are exploratory paired-t intervals for
the per-conversation improvement over no summary.

| KV slots | Pooled init F1 | Optimized unseen F1 | Delta vs no summary (95% CI) | Wins | Compression vs source / self-study text | BF16 KV size |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.154 | 0.208 | +0.065 [-0.029, 0.158] | 11/16 | 93.3x / 105.0x | 0.5625 MiB |
| 8 | 0.157 | **0.278** | **+0.135 [0.038, 0.231]** | **12/16** | 46.7x / 52.5x | 1.125 MiB |
| 16 | 0.157 | **0.291** | **+0.147 [0.013, 0.282]** | 11/16 | 23.3x / 26.3x | 2.25 MiB |

Eight and 16 slots therefore recover about 41% and 45% of the incremental F1 supplied
by the 420-token generic plaintext handoff, respectively:

```text
(KV F1 - no-summary F1) / (self-study-text F1 - no-summary F1)
```

This is the first positive evidence in this experiment that a short, query-agnostic
semantic KV state exists: none of the 128 evaluation questions or answers influenced
the optimized state, yet eight slots roughly double no-summary F1. The effect is not
just inherited from mean pooling; the unoptimized eight-slot initialization scores
0.157.

It is also not a solved compressor. Four of 16 eight-slot instances and five of 16
16-slot instances score below their no-summary baselines. Increasing the budget from
eight to 16 gives only a small average improvement, while 32 slots scored 0.345 on the
initial four-document scan and did not restore monotonic scaling. Low self-study
training loss can coexist with poor unseen QA, so the present bottleneck is the
semantic coverage and optimization geometry of the construction objective, not only
raw KV capacity.

There is additional headroom in the teacher material itself. The four generated
responses average 93.6 tokens each, and 56 of 64 hit the 96-token generation cap.
Their plaintext F1 of 0.470 remains far below the gold-briefing ceiling. Better
self-study prompts or a coverage objective may improve both the text and KV paths.

## Conclusion

The probe now supports two claims at different strengths:

> A very short, direct KV state exists when the receiver task/query distribution is
> known during state construction.

This remains the strong result from the answer-conditioned oracle (four slots, 0.890
F1). More importantly, the self-study experiment supports a weaker but genuinely
query-agnostic result:

> An eight-to-16-slot direct KV state can carry reusable story semantics to unseen
> downstream questions without any downstream question or answer in its construction
> loss.

The effect is statistically and practically visible on this 16-document probe, but it
is incomplete and unstable: 8/16 slots reach 0.278/0.291 F1 versus 0.144 with no
summary and 0.470 with the same information in plaintext. Thus the existence question
has a qualified **yes**; stable construction and high-fidelity replacement remain
open.

The next necessary step is no longer another slot-count sweep. It is an amortized
compressor trained across documents to map an existing source KV into eight or 16
receiver-native slots, using label-free self-study/coverage distillation and evaluated
on unseen documents as well as unseen questions. That separates a reusable method
from the current expensive per-instance existence oracle.

## Reproduction

Task-conditioned runs:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m xmodel_kv.cli.probe_semantic_kv_summary \
  --dataset datasets/coqa/data/validation-00000-of-00001.parquet \
  --model Qwen/Qwen3-8B \
  --output-dir outputs/semantic_kv_summary/coqa_oracle_source_idx01_q8_s200_20260817 \
  --device cuda:0 --conversation-indices 0,1 --budgets 4,8,16,32 \
  --max-questions 8 --initialization source_pool --steps 200
```

Held-out runs add:

```text
--train-turns 1,3,5,7
```

Raw results are under `outputs/semantic_kv_summary/` in the four
`coqa_oracle_{source,heldout}_idx{01,23}_q8_s200_20260817` directories.

Query-agnostic self-study runs use:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m xmodel_kv.cli.probe_semantic_kv_summary \
  --dataset datasets/coqa/data/validation-00000-of-00001.parquet \
  --model Qwen/Qwen3-8B \
  --output-dir outputs/semantic_kv_summary/coqa_selfstudy_b8_s300_idx0123_textbase_20260817 \
  --device cuda:0 --conversation-indices 0,1,2,3 --budgets 8 \
  --max-questions 8 --training-mode self_study --initialization source_pool \
  --self-study-max-new-tokens 96 --steps 300 --learning-rate 0.01 \
  --max-new-tokens 24
```

The aligned 16-document results are split across the
`coqa_selfstudy_b{4,8,16}_s300_idx{0123,0409,1015}*20260817` directories.
