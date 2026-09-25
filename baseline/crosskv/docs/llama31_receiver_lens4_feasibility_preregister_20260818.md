# Receiver-Lens four-slot feasibility gate

Status: frozen on 2026-08-18 before training or evaluating the lens.

## Question

Can a very small receiver-local suffix learn task-conditioned access to an
immutable CE8 semantic packet, improving new-domain decisions without modifying
or retransmitting the source state?

This is the first behavioral test of `receiver_lens_kv_design_20260818.md`.  It
does not test memory eviction, a receiver-side writer, external E sets, or
cross-model transfer.

## Frozen model and state

- Receiver: Llama-3.1-8B-Instruct, BF16.
- Semantic state: the existing all32 CE8 parent checkpoint
  `llama31_native_tail_scaleup_ce8_800_train128_eval64_20260818`.
- Every source packet is serialized once in the receiver-independent canonical
  frame with the real framed signed K4/V4 codec and decoded for training.  The
  packet is 270,426 bytes and remains bitwise frozen.
- Receiver order is `system -> CE8 packet -> user/history -> local suffix ->
  assistant readout -> answer`.  Thus the suffix sees both the immutable memory
  and the complete downstream user task before it is materialized.

## Frozen arms

1. **base:** CE8 with no local suffix.
2. **hard4:** append the four fixed tokenizer tokens for
   ` internal memory answer state`.
3. **lens4:** four trainable embeddings initialized exactly from those same
   hard-token embeddings.  No layerwise writer, prompt compiler, adapter, or
   auxiliary loss is allowed.

Hard4 and lens4 are therefore exactly equal before the first update.  Hard4 is
the matched local-compute control; lens4 adds no per-request network bytes but
adds four local native KV positions.

## Frozen training

- Data: the existing 568-case mixture of 312 standard-LongBench-derived
  HotpotQA/2WikiMQA handoffs and 256 MuSiQue replay cases.
- Direct losses only: final/stage CE weight 1 and bridge CE weight 2.
- AdamW, 1,600 updates, LR `3e-3`, weight decay 0, gradient clip 1.
- Seed 2111, final checkpoint only, no early stopping or checkpoint selection.
- Only the 16,384 lens-embedding parameters are trainable.  Hashes of the base
  capsule before and after training must be identical.

## Frozen evaluation

Use the production canonical K4/V4 packet and generation cap 32 on:

- the 56-case standard-LongBench development set, correct source and shift-1;
- MuSiQue confirm rows 96--127 as a 32-case safety set.

Report base, hard4, and lens4 on correct-source data.  Report at least lens4
under shift-1.  Use 20,000 paired bootstrap resamples with seed 2112.

## Frozen gates

All conditions are required before any consolidate-and-evict test is
authorized:

1. On development, lens4-minus-base final F1 is at least `+.04` and its 95% CI
   lower bound is above zero.
2. Development bridge is safe: lens4-minus-base point delta is at least `-.03`
   and CI lower bound is above `-.10`.
3. Lens4 correct-minus-shift has a positive CI lower bound for both final and
   bridge F1.
4. Lens4 is not merely a hard delimiter: versus hard4, final and bridge point
   deltas are non-negative and both CI lower bounds are above `-.08`.
5. On MuSiQue, lens4-minus-base final delta is at least `-.02` with CI lower
   above `-.08`; bridge delta is at least `-.05` with CI lower above `-.10`.

Failure rejects the fixed four-embedding lens.  It does not authorize changing
slot count, LR, steps, initialization phrase, or adding a writer.  A
receiver-local writer would require an independently frozen capacity
diagnostic.  No external E-set evaluation is allowed from this experiment.

## Cost boundary

The semantic network payload stays at 270,426 bytes.  Four local BF16 all-layer
KV positions occupy 524,288 receiver bytes before allocator overhead, but are
not transmitted.  Training and evaluation must report the extra materialization
time separately from source-packet preparation; end-to-end serving benefit is
not established by this gate.
