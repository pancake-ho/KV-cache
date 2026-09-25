# Llama-3.1 continued-writer sealed 2WikiMQA-E results

## Decision

The sealed test confirms that a four-slot, first16, real-INT4 packet carries
sample-specific task state to a previously untouched dataset, but it does not
confirm that the additional MuSiQue training data significantly improves that
compact packet over its parent checkpoint.

The three preregistered gates resolve as follows:

1. **Useful compact state passes.** New first16 minus no-summary question F1 is
   `+.1462`, paired bootstrap 95% CI `[+.1014,+.1917]`.
2. **Source causality passes.** Correct new first16 minus circularly shifted
   first16 question F1 is `+.0474`, CI `[+.0041,+.0911]`.
3. **Compact data scaling fails.** New first16 minus parent first16 question F1
   is only `+.0194`, below the frozen `+.02` point-estimate threshold, and its
   CI is `[-.0200,+.0591]`.

Consequently the file is exhausted for selection.  The defensible claim is
cross-task existence and causality of compact summary KV, not confirmed
cross-task scaling of its early-layer prefix.

## Frozen data and execution

- Dataset: `/data/datasets/longbench/2wikimqa_e.jsonl`.
- SHA256:
  `525b5b182089a4012cc7429c33f4208358778615173c4a09349429fc80c89641`.
- 300 rows, 300 unique IDs, and 300 unique normalized questions.
- Model: frozen BF16 Llama-3.1-8B-Instruct.
- Packet: four positions, actual signed-nibble INT4 serialization.
- Decoding: greedy, 24 source-answer tokens and 24 receiver tokens maximum.
- Protocols: generic `state_readout` and `question_conditioned`.
- Statistics: paired 20,000-replicate bootstrap, seed 2027.

All three frozen runs completed 300 cases and produced 600 result records each
(one for each receiver protocol).  No checkpoint, prompt, depth, precision, or
decode setting was selected using this dataset.

## Absolute results

Question-conditioned protocol:

| checkpoint / handoff | EM | F1 | payload |
|---|---:|---:|---:|
| no-summary | .0467 | .1545 | 0 B |
| generated-answer Tail-KV | .1267--.1300 | .2142--.2168 | about 209 KB |
| parent first16 capsule | .2200 | .2813 | 67,584 B |
| new first16 shifted capsule | .2033 | .2533 | 67,584 B |
| **new first16 capsule** | **.2467** | **.3007** | **67,584 B** |
| **new all-layer capsule** | **.2567** | **.3798** | **135,168 B** |
| new all-layer shifted capsule | .1067 | .1366 | 135,168 B |

The source agent itself reaches `.3667` EM and about `.467` F1.  This is not an
oracle-source evaluation: receiver quality includes upstream source-answer
errors.

Generic state readout:

| checkpoint / packet | EM | F1 | payload |
|---|---:|---:|---:|
| parent first16 | .0133 | .0218 | 67,584 B |
| new first16 | .0100 | .0186 | 67,584 B |
| new all-layer shifted | .0000 | .0034 | 135,168 B |
| **new all-layer** | **.0633** | **.1636** | **135,168 B** |

Thus first16 is useful when the receiver question transforms the state, but it
does not expose a generally decodable lexical answer.  The deeper half restores
that readout channel.

## Paired secondary analyses

| comparison | F1 delta | paired 95% CI | interpretation |
|---|---:|---:|---|
| all vs first16, question-conditioned | +.0791 | `[+.0295,+.1302]` | deeper state improves task quality at 2x bytes |
| all vs first16, state-readout | +.1450 | `[+.1117,+.1799]` | deep layers carry lexical/readout state |
| all correct vs shifted, question-conditioned | +.2431 | `[+.1917,+.2939]` | full packet is strongly sample-bound |
| all correct vs shifted, state-readout | +.1602 | `[+.1267,+.1953]` | readout is not a generic prompt prior |
| all vs generated Tail, question-conditioned | +.1656 | `[+.1098,+.2202]` | capsule wins on this sealed task |
| all vs generated Tail, state-readout | +.0665 | `[+.0297,+.1045]` | capsule also improves generic readout |
| first16 vs generated Tail, question-conditioned | +.0858 | `[+.0361,+.1359]` | compact packet wins despite lower payload |

The all-versus-first16 EM delta is only `+.010` and its CI crosses zero; the
F1 gain arises mainly from better partial and lexical answer recovery.  This is
consistent with the depth-indexed-state hypothesis and should not be described
as an across-the-board exact-accuracy gain.

## Interpretation

This sealed result separates three questions that were previously entangled:

- **Does short summary KV exist?** Yes.  Four source-conditioned KV positions
  over only the first 16 layers outperform both no state and a wrong-source
  state on a completely untouched dataset.
- **Is it merely a hard-token tail or answer prior?** No.  It beats the
  generated-answer Tail-KV baseline here, and circular source shifting removes
  the advantage.
- **Did the extra 256 MuSiQue cases make the compact cross-task packet better?**
  Not at the preregistered confidence level.  The improvement is small and
  uncertain, whereas the all-layer benefit is strong.

Together with the already-open 2WikiMQA-200 follow-up, the most likely mechanism
is that full-depth end-to-end continuation improves transferable state but
places much of the incremental information in deep K/V.  Runtime truncation to
first16 remains a valid compact operating point, but ordinary data scaling does
not reliably force new information into that compact prefix.

## Artifacts

- Preregistration:
  `docs/llama31_n768_2wikimqa_e_sealed_preregister_20260818.md`.
- New first16:
  `outputs/amortized_semantic_kv/llama31_n768_2wikimqa_e_first16_int4_sealed_20260818/`.
- Parent first16:
  `outputs/amortized_semantic_kv/llama31_parent_2wikimqa_e_first16_int4_sealed_20260818/`.
- New all-layer:
  `outputs/amortized_semantic_kv/llama31_n768_2wikimqa_e_all_int4_sealed_20260818/`.
- Paired analyses:
  `outputs/amortized_semantic_kv/llama31_n768_2wikimqa_e_sealed_analysis_20260818/`.
