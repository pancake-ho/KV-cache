# Receiver-Lens4 frozen feasibility result

Status: completed on 2026-08-18 under the frozen protocol in
`llama31_receiver_lens4_feasibility_preregister_20260818.md`.

## Decision

The fixed four-embedding receiver lens is **rejected**.  It improves the
56-case development point estimate and is strongly source-causal, but the
development final-answer confidence interval crosses zero and MuSiQue bridge
readout degrades significantly.  The retained-base gate therefore fails.
Consolidate-and-evict and external E-set evaluation are not authorized.

## Training integrity

- Receiver: Llama-3.1-8B-Instruct in BF16.
- Frozen semantic state: all32 CE8, 8 slots, canonical real K4/V4 packet.
- Packet payload: 270,426 bytes.
- Trainable state: exactly 4 x 4,096 = 16,384 lens-embedding parameters.
- Data/updates: 568 cases, 1,600 AdamW steps, LR `3e-3`.
- Objective: final/stage CE + 2 x bridge CE.
- Packet preparation/training time: 289.88 s / 472.60 s.
- Frozen base parameter hash before and after training:
  `5acd4c105c2cba8c427a641f0e8a502b5d486b54882b10255160891058272f44`.

The equality of the before/after hash proves that downstream adaptation did
not modify the semantic writer.  The four learned slots add no wire bytes and
524,288 local BF16 KV bytes before allocator overhead.

## Frozen evaluation

All values are token F1.  Confidence intervals use 20,000 paired bootstrap
resamples with seed 2112.

| split / arm | final | bridge |
|---|---:|---:|
| development correct / base | 0.107143 | 0.230825 |
| development correct / hard4 | 0.107143 | 0.229337 |
| development correct / lens4 | **0.196429** | **0.258092** |
| development shift-1 / lens4 | 0.071429 | 0.003247 |
| MuSiQue correct / base | 0.921875 | 0.890625 |
| MuSiQue correct / hard4 | 0.890625 | **0.927083** |
| MuSiQue correct / lens4 | 0.921875 | 0.812256 |

Key paired comparisons:

| comparison | metric | delta | 95% CI | wins/ties/losses |
|---|---|---:|---:|---:|
| dev lens4 - base | final | +0.089286 | [-0.035714, +0.214286] | 10/41/5 |
| dev lens4 - base | bridge | +0.027267 | [-0.012599, +0.077031] | 7/45/4 |
| dev lens4 - hard4 | final | +0.089286 | [-0.035714, +0.214286] | 10/41/5 |
| dev lens4 - hard4 | bridge | +0.028755 | [-0.007674, +0.076588] | 7/45/4 |
| dev correct - shift | final | +0.125000 | [+0.035714, +0.232143] | 8/47/1 |
| dev correct - shift | bridge | +0.254845 | [+0.162716, +0.352836] | 22/34/0 |
| MuSiQue lens4 - base | final | 0.000000 | [0, 0] | 0/32/0 |
| MuSiQue lens4 - base | bridge | -0.078369 | [-0.145930, -0.020833] | 0/26/6 |

## Gate audit

| frozen gate | result | reason |
|---|---|---|
| development final improvement | **fail** | point delta exceeds +.04, but CI lower bound is negative |
| development bridge safety | pass | point and lower-bound safety margins hold |
| correct-source causality | pass | final and bridge CI lower bounds are positive |
| matched hard4 control | pass | both point deltas are non-negative and safety bounds hold |
| MuSiQue retention | **fail** | bridge delta is -.07837 and its CI is fully negative |

## Mechanism conclusion

The learned suffix is not merely a delimiter: it beats the matched hard-token
control on development and loses most of its signal under source shift.  Thus
four contextualized local slots can access the immutable packet.  However, a
single global set of four embeddings changes the readout policy in a way that
trades new-domain final decisions against the original MuSiQue bridge channel.
Semantic immutability prevents sender forgetting, but does not by itself make
receiver adaptation task-safe.

This falsifies the fixed-embedding version, not the broader factorization into
immutable semantic memory and receiver-local decision state.  Slot-count,
learning-rate, initialization, and checkpoint scans are forbidden by the
frozen protocol.  A receiver-local conditional writer is justified only if a
separately frozen diagnostic finds that per-example lens gradients require
different task-conditioned directions rather than one shared update.

## Authoritative artifacts

- trained lens: `outputs/amortized_semantic_kv/llama31_receiver_lens4_train568_step1600_20260818/lens.pt`
- training configuration/log: same directory, `config.json` and
  `training_log.jsonl`
- development correct-source rows:
  `outputs/amortized_semantic_kv/llama31_receiver_lens4_dev56_correct_20260818/results.jsonl`
- development shift-1 rows:
  `outputs/amortized_semantic_kv/llama31_receiver_lens4_dev56_shift1_20260818/results.jsonl`
- MuSiQue safety rows:
  `outputs/amortized_semantic_kv/llama31_receiver_lens4_musique32_correct_20260818/results.jsonl`
- frozen bootstrap decision:
  `outputs/amortized_semantic_kv/llama31_receiver_lens4_train568_step1600_20260818/frozen_development_analysis.json`
