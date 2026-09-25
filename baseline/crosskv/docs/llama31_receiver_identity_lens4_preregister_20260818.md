# Receiver-identity Lens4 bank development gate

Status: frozen on 2026-08-18 before training or evaluating the identity lens.

## Question

The shared Receiver-Lens4 is source-causal but task-unsafe.  Its frozen
gradient diagnostic shows strong receiver-domain structure and significant
joint-objective conflict, although the stricter final-versus-bridge conflict
gate does not authorize a learned conditional writer.  Multi-agent runtimes,
however, already know the identity of the downstream agent.  This experiment
asks whether an explicit receiver-ID-indexed bank of tiny static lenses can
adapt one immutable semantic packet without sharing the readout parameters
across heterogeneous agents.

The selector is request metadata, not a learned router.  An unregistered
receiver gets no lens and executes the exact base path.  This makes semantic
retention for other receiver identities structural rather than a replay loss.

## Frozen data

The method-development split was generated once with the audited
`prepare_longbench_handoff_mix` builder, candidate count 8, seed 2114:

- train: 448 cases from LongBench `hotpotqa_e` and `2wikimqa_e`, SHA-256
  `e92f7e3b525aa50bbd5963bf96035926744136b07ebfb642216e19f216813296`;
- development: 128 cases (64/source), SHA-256
  `8b231e48085b5b6fad3f5042e528a2201821026a68e27c6940608309456ab667`.

All 400 standard LongBench Hotpot/2Wiki questions were excluded before the
split.  New train and development IDs are mutually disjoint and each has zero
overlap with the prior 568-case lens train and 56-case lens development sets.
The E corpora were used by earlier system comparisons, so this remains opened
method-development evidence, not a fresh paper test set.

## Frozen state and arms

- Receiver: Llama-3.1-8B-Instruct, BF16.
- Immutable semantic packet: the same all32 CE8 real canonical K4/V4 packet,
  8 slots and 270,426 bytes.  Base checkpoint SHA-256:
  `7a0ff34d9fe738a621abfdeab58ff62781dd11bdd7def977af29323ff68cab9e`.
- Lens shape: exactly 4 x 4,096 FP32 trainable embeddings initialized from
  ` internal memory answer state` as before.
- `base`: no local suffix.
- `hard4`: the four matched tokenizer tokens.
- `shared4`: the completed prior multi-domain Receiver-Lens4, checkpoint
  SHA-256
  `ce1764b97e9c3137846f2c5f984ec6597737ebebcc543bc3eb24a01201755224`.
- `identity4`: the new lens trained only for receiver ID
  `longbench_e_paired_lookup_v1`.

All arms receive identical source packets and receiver prompts.  Lens state is
receiver-local: zero wire bytes and 524,288 local BF16 KV bytes per activated
request.  The bank stores 65,536 FP32 parameter bytes per registered receiver.

## Frozen training

- Data: all 448 E-train cases, no MuSiQue replay.
- Only 16,384 identity-lens embeddings are trainable; CE8 and Llama are frozen.
- Objective: final/stage CE + 2 x bridge CE.
- AdamW, 1,600 updates, LR `3e-3`, weight decay 0, clip 1.
- Seed 2115, final checkpoint only, no checkpoint selection.
- Every source is serialized through the real canonical K4/V4 packet before
  training.  CE8 before/after parameter hashes must match exactly.

## Frozen evaluation

On all 128 E-development cases, generation cap 32:

1. correct source for base, hard4, shared4, and identity4;
2. shift-1 source for identity4.

Use 20,000 paired bootstrap resamples with seed 2116.  All comparisons are
paired by case ID.  The nonselected-receiver route must be an exact code-path
bypass returning no lens; it is an implementation invariant, not a behavioral
margin estimated on MuSiQue.

## Frozen gates

All conditions are required:

1. identity4-minus-base final F1 is at least `+.06` and its 95% CI lower bound
   is above zero;
2. identity4-minus-shared4 final F1 is at least `+.03` and its CI lower bound is
   above zero;
3. identity4-minus-base bridge F1 is at least `-.03` and its CI lower bound is
   above `-.08`;
4. identity4 correct-minus-shift final and bridge F1 both have CI lower bounds
   above zero;
5. versus hard4, identity4 final and bridge point deltas are non-negative and
   both CI lower bounds are above `-.08`;
6. CE8 is bitwise unchanged, the registered identity selects only identity4,
   and an unregistered identity selects the exact no-lens base route.

Passing all gates supports only the development claim that explicit
receiver-identity factorization removes the unsafe shared-readout tradeoff.  It
would authorize a separately frozen confirmation on a genuinely untouched
task family.  It would not authorize a learned conditional writer, eviction,
cross-model transfer, or a slot/rank/hyperparameter sweep.  Failure rejects the
static identity-bank version.
