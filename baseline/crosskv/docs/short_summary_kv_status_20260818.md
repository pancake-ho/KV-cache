# Short-summary KV: current method and evidence (2026-08-18)

## What the term means

“Short-summary KV” is not a subsequence selected from the document cache and
not the KV of a generated text summary.  It is a small, learned set of native
K/V positions that is conditioned on the complete source computation and is
intended to preserve only the state needed by a downstream query.

Three nearby objects must be kept separate in the current experiments:

1. **Generated Tail-KV:** decode a short answer prefix at the sender and pass
   the native KV of those real tokens.  This is the strongest oracle/teacher
   state, but it is post-answer and pays sender decoding latency.
2. **Learned short-summary KV (capsule):** emit virtual source-conditioned
   slots before an answer is decoded, retain only their KV, and train them for
   downstream use.  This is the actual proposed semantic-summary mechanism.
3. **Truncated document KV:** keep or delete positions from the original
   document cache.  The present method does not do this, because arbitrary
   token deletion changes attention normalization and positional semantics.

The current Llama-3.1-8B operating point replaces a roughly 7k-token source KV
with four virtual positions.  It transmits those positions for the first 16 of
32 layers in signed INT4, including real FP16 scales and framing.  The result is
a 67,642-B packet rather than a token-visible summary.

This is “summary” in the functional sense: a receiver should answer from the
state.  It is not required to reconstruct the source text or even directly
decode the intermediate entity.

## How it is produced

1. Append four learned soft emission slots after the sender's source and
   question.  They are not vocabulary tokens shown to the receiver.
2. Run the frozen Llama stack.  For the first 15 decoder blocks, the four slots
   may attend to the complete source, so their hidden states become
   source-conditioned.
3. After that boundary, deeper blocks can attend only to the four slots.  This
   prevents later emission layers from repeatedly depending on the long source
   while retaining full-depth computation and gradients.
4. A frozen transferable base writer produces the common four-slot state.  A
   small trained rank-16 boundary adapter modifies the hidden state after layer
   14, so the task-specific effect first appears in K/V layer 15.  Base-model
   weights and base-writer parameters remain frozen.
5. Keep only K/V for the four slots and selected layers.  Keys are RoPE-moved
   from sender positions to receiver positions; Values require no positional
   transform.
6. Serialize with the production nibble codec: signed Cartesian INT4 K/V with
   a transmitted FP16 scale per layer/head/slot vector, explicit layer indices,
   header, and CRC32.

The receiver prefills only its own short system/query prefix, appends the
decoded four-position cache, and continues normal autoregressive decoding.  It
does not prefill the source or a text summary.

## Current composable representation

The strongest new systems result separates transferable and task-specific
state on the wire:

\[
  P_{base}=Q_4(C_{base}^{0:15}), \qquad
  P_{task}=Q_4(C_{task}^{15}-C_{base}^{15}).
\]

The receiver decodes `P_base` and adds `P_task` only at layer 15.  Runtime
checks prove layers 0--14 in task and base are bitwise equal before packing,
and are exactly the decoded base after composition.

| component | framed bytes | role |
|---|---:|---|
| first16 base | 67,642 | reusable document state |
| layer-15 task delta | 4,252 | agent/task specialization |
| cold base + delta | 71,894 | first transfer |
| base-hit transfer | 4,252 | later agent/task |

The delta is 6.286% of the base.  This is an incremental-reuse result, not a
cold-transfer compression result.

## What is verified

- On MuSiQue-96 with the real codec, base+delta reaches `.795` F1 versus `.774`
  for a directly packed rank-16 task cache; paired delta `+.0208`, CI
  `[.0000,+.0521]`.
- On 2WikiMQA-200 question-conditioned transfer, base+delta is `.3292` versus
  `.3200`; delta `+.0092`, CI `[-.0121,+.0322]`.
- Correct-source versus shifted-source lower confidence bounds are positive on
  both datasets, so the packet is causally source-dependent rather than merely
  providing extra positions or a task prior.
- The real serialized base is the exact fallback.  No merged checkpoint must
  be regenerated when a task delta is absent.

These results support short functional state and composable task deltas at one
same-model Llama operating point.  They do not yet prove cross-model handoff,
arbitrary task families, or serving-level speedup.

## RoPE status

There are currently two position modes:

- **Target-frame packet:** move Keys directly to the receiver positions, then
  quantize.  This has the strongest behavioral evidence, but different
  receiver-prefix lengths produce different Key bytes.
- **Canonical packet:** move Keys to positions 0--3, quantize once, then after
  decode move them to each receiver's positions.  Packet hashes are exactly
  identical across receiver prefixes, so the systems semantics are correct.

Canonical Cartesian INT4 passes MuSiQue safety but narrowly fails the frozen
2Wiki safety lower bounds (`-.03323` and `-.03333` versus a required `>-.03`).
A preregistered equal-byte polar Key codec also fails.  A separately frozen
K8/V4 canonical control passes on a new MuSiQue-200 set but fails the original
2Wiki holdout while adding 48.43% bytes.  It lowers tensor NRMS by about 85%
without reliably improving decoded answers.  Therefore the paper may claim
receiver-independent byte identity as a mechanism, but not yet broad
behavioral non-inferiority for that mode.

## Important corrections and non-claims

- An earlier significant rank-16 improvement used fake INT4 with untransmitted
  FP32 scales.  The production codec uses transmitted FP16 scales and gives a
  stronger base; rank-16 versus that base is positive but not significant.
- Four slots do not mean four token IDs and cannot be decoded as a four-token
  text summary.
- Cold base+delta is larger than a monolithic packet.
- No cross-model KV adapter is currently successful.
- No vLLM/SGLang connector or network/concurrency benchmark exists yet.

## Positional-orbit result and best next experiment

Positional-orbit robustness has now been trained at fixed K4/V4 rate.  In the
single frozen 1,000-step run, orbit KL falls 66.55%, all frozen hashes remain
unchanged, and exact receiver-independent packet invariants pass.  On the new
MultifieldQA-150 external set, target retention and canonical safety pass, but
correct-source minus shifted-source F1 is only `+.02581` with CI
`[-.00705,+.05963]`.  The repair difference-in-differences is `+.01220` with
CI `[-.01299,+.03910]`.  Thus neither overall external safety nor a general
orbit-repair mechanism is established.

A post-hoc capsule-versus-no-state comparison is also negative and
inconclusive, while generated-answer Tail-KV is significantly stronger.  This
locates the main quality bottleneck in transferable semantic encoding and
receiver use, not in simple Key precision or positional relocation.  The next
quality experiment should therefore improve the source-causal content channel
under the same four-slot/rate budget, rather than scan another RoPE codec.
Opened MuSiQue rows confirm the diagnosis: orbit-canonical correct versus
shifted source is `+.510` final-answer F1 (CI `[+.406,+.613]`) and `+.349`
bridge F1 (CI `[+.263,+.438]`).  The state is strongly causal in-domain but
does not preserve that channel on MultifieldQA.

A frozen generated-Tail capacity diagnostic on MultifieldQA then isolates a
second limitation.  With first16 canonical INT4 fixed, raising the generation
cap from 4 to 8 positions improves F1 by `+.0307`, CI `[+.0066,+.0551]`;
8 to 16 adds only `+.0020`, CI crossing zero.  On 9+-word answers cap16 versus
cap4 adds `+.0602`, and its gain over the short-answer stratum is `+.0562`;
both lower confidence bounds are positive.  This authorizes one causally
nested 4-core+4-refinement capsule prototype, not a fixed 16-slot model.
That prototype now has a resolved negative result.  Exact core reuse and split
wire composition pass, but CE-only full-8 minus core-4 is only `+.0139` and
correct minus shifted is `-.0607`.  A fixed counterfactual hinge makes
full-minus-core `-.0408` and still reverses the source contrast.  Both pilots
reduce training loss by more than 55%, so the failure is semantic
generalization rather than optimization or packet plumbing.  No larger learned
refinement run is authorized.

A completed token-cap by layer-depth factorial now identifies the strongest
native-state frontier.  On MultifieldQA-150, generated Tail-KV with cap8 and
all 32 layers scores `.23159` question-conditioned F1 using 218,386 mean
framed bytes.  Cap24 scores `.26116` at 435,556 bytes; cap8-minus-cap24 is
`-.02957`, CI `[-.05763,-.00150]`, which passes the frozen non-inferiority
margins, while cap8-minus-no-state is `+.04720`, CI
`[+.00270,+.09269]`.  All32 also significantly improves over first16 at every
tested cap.  This establishes a bounded post-answer Tail-KV operating point
and shows that token count and layer depth are separate rate axes.  It does
not turn Tail-KV into a pre-answer learned summary; reproducing this native
state without decoding the answer remains the core representation problem.

The first direct attack on that gap is positive.  At the same four-slot/all32
rate, matching canonical K4/V4 layers 1--31 to a gold-answer Tail-KV teacher
reduces held-out state error from 1.425 to .624 and improves real-codec
MuSiQue final F1 from .859 to .922.  The paired delta is `+.0625`, CI
`[0,+.1563]`, while correct-source minus shifted-source is `+.7063`, CI
`[+.5625,+.8438]`.  A behavior-logit KL comparator instead falls to .703 F1.
All preregistered feasibility gates pass, authorizing a cap8 scale-up.  Because
this is opened same-family development evidence, it changes the next
experiment rather than the external paper claim.

The authorized cap8 scale-up then rejects a naive extrapolation.  State8 cuts
held-out coordinate MSE to 35.55% of CE8, with every one of 64 cases closer,
yet real-codec final F1 is `.8724` versus `.9219` for CE8 (`-.0495`, CI
`[-.1042,-.0078]`) and does not exceed state4.  Source causality remains
strong.  Thus direct coordinate loss is a useful four-slot regularizer but is
not the correct scalable semantic objective.  The next representation target
is the query-conditioned attention operator `softmax(qK^T)V`, including its
normalization mass, which is invariant to slot correspondence and permits
teacher and student to use different token counts.

A frozen attention-operator probe then shows that this refinement is still
insufficient if evaluated tail-only on fixed no-summary queries.  State8
improves operator error over CE8 on all 64 cases (`-.8434`, CI fully negative)
while remaining significantly worse in final F1; per-case correlation is only
`.13`.  The native lexical Tail is therefore not automatically the optimal
downstream summary target, or its effect must be measured jointly with the
full receiver prefix and on-policy query trajectory.  No operator-training run
is authorized.

Two frozen follow-ups close the narrower explanations.  First, native-state
and native-operator gradients are not systematically opposed to final-task
gradients: their mean cosines are approximately zero, both confidence
intervals cross zero, and only 46.9% of cases are negative.  A PCGrad-like
repair is therefore not authorized.  Second, a counterfactual full-receiver
probe compares each packet's marginal on-policy hidden trajectory with the
effect of an explicit plaintext handoff after subtracting a same-length zero-KV
control.  State8 is closer than CE8 on 62/64 cases (`-.03307` total loss, CI
fully negative) while still losing `-.04948` F1.  Thus even broad plaintext
effect similarity rewards the wrong equal-capacity solution.  The next method
must change direct task/data coverage or the state-writing structure, not scan
another K/V, operator, or hidden-similarity loss.
Separately, the next systems experiment should materialize each canonical base
packet once, store it, and measure a real base-hit path that transfers and
decodes only the delta.

The repository suite now passes `235 passed, 2 warnings`, including the
multi-domain data, frozen-gate, checkpoint-drift, and receiver-lens cache
invariant tests.

A fixed multi-domain direct-supervision continuation then tests whether broader
tasks alone resolve the missing decision directions.  It does not.  On 56 new
Hotpot/2Wiki-derived development cases, candidate-minus-parent final F1 is
`+.04847`, CI `[-.07143,+.16071]`, while bridge F1 is `-.09633`, CI
`[-.18225,-.01792]`.  On the MuSiQue safety slice, final F1 changes by
`-.03286` and bridge F1 collapses by `-.59639`, CI
`[-.66063,-.52895]`, with losses on all 32 cases.  Three of four frozen gates
fail, so no E-set evaluation is authorized.  Direct task diversity and replay
therefore do not make a shared eight-slot writer task invariant.

An offline checkpoint-drift localization further shows that this failure is not
confined to the slot embeddings or one short layer region.  Embeddings account
for 3.45% of squared parameter drift; fixed-probe functional drift is distributed
16.52%/43.11%/40.37% across early/middle/deep writer blocks.  The largest eight
blocks contain only 43.47%, and the best contiguous eight contain 34.01%, both
below the frozen 60% localization threshold.  A simple module-freezing patch is
therefore not motivated.  The next structural hypothesis is an immutable
semantic path plus a separately parameterized decision residual; it requires a
new preregistered causal test rather than another whole-writer loss scan.

The frozen Receiver-Lens4 feasibility experiment is now complete.  A local
lens can be materialized after an immutable prompt-plus-memory cache while
detaching the complete prefix gradient; the all32 CE8 base parameter hash is
identical before and after 1,600 lens-only updates.  On 56 development cases,
lens4 raises final F1 from `.1071` to `.1964`; correct-minus-shift is `+.1250`
with CI `[+.0357,+.2321]`, so the slots genuinely read the source packet rather
than acting only as a delimiter.  The lens-minus-base improvement CI still
crosses zero, however, and on 32 MuSiQue safety cases bridge F1 falls from
`.8906` to `.8123`, delta `-.07837`, CI `[-.14593,-.02083]`.  The fixed global
four-embedding lens therefore fails two frozen gates.  Eviction and external
evaluation are not authorized.  The next admissible step is a frozen gradient
diagnostic asking whether different receiver tasks require incompatible lens
directions; it is not a slot-count or hyperparameter sweep.

That frozen gradient diagnostic is also complete.  The primary aggregate
development-final versus MuSiQue-bridge cosine is `-.1722`, but its bootstrap
CI `[-.2626,+.0887]` crosses zero, so it does not authorize a conditional
receiver writer.  The secondary joint-objective cosine is significantly
negative (`-.3662`, CI `[-.5634,-.0519]`) and the own-domain gradient-centroid
margin is strongly positive (`+.4277`, CI `[+.2910,+.5617]`).  The 88-case
normalized joint-gradient matrix needs rank 53 for 90% energy.  Receiver-task
heterogeneity is therefore real, but the exact predeclared explanation of the
behavioral tradeoff is unresolved.  The next experiment must use an
independently frozen receiver-identity factorization on new development data,
not a post-hoc conditional-writer escalation.
