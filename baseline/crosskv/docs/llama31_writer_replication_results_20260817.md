# Llama-3.1-8B fixed-writer model-family replication

Date: 2026-08-17

This is the result of the no-search protocol frozen in
`docs/llama31_writer_replication_preregister_20260817.md`. Sender and receiver
are both Llama-3.1-8B-Instruct. The writer uses the Qwen training
hyperparameters unchanged, and the transported capsule is the pre-specified
last 16 layers at INT4. No Llama layer curve or checkpoint selection was
performed.

## Training and fixed validation gate

- MuSiQue train: 128 cases, 4 slots, rank-4 writer, 1,600 steps;
- training time: 511.9 s on one H200;
- final mean stage CE / bridge CE: `.3141 / .0871`;
- fixed MuSiQue dev32 downstream EM/F1: `40.6% / .444`;
- no-summary downstream EM/F1: `21.9% / .266`;
- direct intermediate-state F1: `.658`.

This passed the pre-specified nontrivial-writer gate and authorized the
external evaluation. It was not used to tune the Hotpot operating point.

## HotpotQA-E 290-case replication

The benchmark is the same question-disjoint E290 set used for the Qwen
confirmation. It is therefore a shared-benchmark model-family replication,
not a second untouched-data confirmation. Agent A obtains `49.0%` EM and
`.617` F1 from an average of 9,472 source tokens.

| protocol | no summary | shifted capsule | correct capsule | generated Tail-KV |
|---|---:|---:|---:|---:|
| state readout EM / F1 | 0.0 / .007 | 0.0 / .013 | 0.34 / .016 | 1.38 / .148 |
| question-conditioned EM / F1 | 10.7 / .195 | 8.62 / .154 | **15.9 / .304** | 21.4 / .310 |

For question-conditioned use, correct capsule versus no summary is `+5.17`
EM points (McNemar `p=.00408`) and `+.1087` F1, with paired bootstrap 95% CI
`[+.0739,+.1452]`. Correct versus shifted capsule is `+7.24` EM points
(`p=1.04e-4`) and `+.1496` F1, CI `[+.1134,+.1871]`. The correct-source
causal effect therefore repeats in a second model family without a Llama
hyperparameter or layer search.

Generated Tail-KV has higher EM than the capsule (`+5.52` points,
`p=.0195`), but its mean F1 advantage is only `+.0058`, with CI
`[-.0394,+.0509]`. This does not establish equivalence, but it means the
290-case test does not resolve an F1 difference between the two at the fixed
operating point. On the 142 cases where Agent A is exactly correct, capsule
and Tail-KV F1 are `.413` and `.438`; the paired difference CI again crosses
zero (`-.053` to `+.103`), while Tail-KV retains a substantial EM advantage.

## Important representation boundary

The Llama capsule fails generic state readout: correct and shifted capsules
are essentially tied (`.016` versus `.013` F1). Yet the same packet strongly
improves question-conditioned answering and passes the correct-versus-shift
causal test. Thus the fixed late-layer state is not generally a textual
memory that any receiver prompt can verbalize. It is better described as a
query-usable continuation state whose value is exposed by the downstream
question. The Qwen and Llama results together show that direct
verbalizability and task utility are distinct axes.

## Payload and synchronized latency

- capsule: 4 slots, last 16/32 layers, INT4, `67,584 B`;
- generated Tail-KV: mean 5.19 tokens, all 32 layers, INT4, `175,369 B`;
- payload reduction: `61.5%`;
- capsule emission: 68.4 ms versus answer decoding: 221.3 ms;
- post-prefill handoff: 253.3 ms capsule versus 400.3 ms Tail-KV;
- post-prefill reduction: 147.0 ms, or `36.7%`.

These are synchronized single-request HF timings. They do not include real
serialization/network transfer or a serving-engine kernel. Unlike the Qwen
E290 result, where capsule quality is clearly lower, the Llama F1 comparison
is statistically unresolved; EM remains lower.

## Artifacts

- checkpoint and fixed dev32 results:
  `outputs/amortized_semantic_kv/llama31_8b_writer_r4_s4_n128_fixedqwenhyper_steps1600_20260817/`;
- three execution shards:
  `outputs/amortized_semantic_kv/llama31_hotpot_e290_last16_replication_shard{0,1,2}_20260817/`;
- pooled rows and paired analysis:
  `outputs/amortized_semantic_kv/llama31_hotpot_e290_last16_replication_pooled_20260817/`.

The common plaintext/full-recompute/full-KV comparison is reported separately
in `docs/llama31_hotpot_baseline_matrix_20260817.md`.
