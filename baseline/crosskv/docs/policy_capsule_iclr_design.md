# Late-Bound Policy Capsule: pre-experiment and ICLR method plan

## 1. Research question

For a homogeneous multi-agent handoff, let

\[
C_A(H)=\operatorname{KV}(P_A\oplus H)
\]

be the immutable history cache produced under source policy `A`.  A receiving
agent `B` normally recomputes `H` under its own private prefix `P_B`.  Exact
Tail-KV instead keeps `C_A(H)` and appends the complete receiver policy as a
new causal suffix.  The proposed method asks whether that suffix can be
compiled into a fixed number `m` of continuous slots:

\[
Z_B=F_\phi(P_B)\in\mathbb{R}^{m\times d},\qquad
\widehat y\sim M(C_A(H)\oplus Z_B).
\]

The base LLM is frozen.  The capsule slots are processed *after* the shared
history, so their contextual K/V states depend on `H` without recomputing the
history.  At serving time the shared history blocks remain immutable and each
reader receives separate capsule blocks.

This is a stronger question than the existing `SoftPolicyCapsule` experiment.
That experiment uses a row-specific plaintext policy refresh (mean 66.82
tokens on the 200-row IID set) followed by one shared 16-token soft suffix.  It
raises compact-tail reuse accuracy from 90.0% to 95.5%; the complete plaintext
receiver tail averages 107.26 tokens and reaches 97.0%.  It does not yet encode
`P_B` into the soft slots.

## 2. Pre-registered pre-experiment

### 2.1 Student and teacher

- **Semantic teacher:** target-native front layout, `KV(P_B + H)`.  Its greedy
  action and top-32 teacher distribution are already stored by the screening
  pass.
- **Student cache:** a neutral receiver prefix containing the target tool schema,
  followed by source-imprinted history K/V.  The target policy itself appears
  only as compiler input and never as plaintext after the history.  This retains
  the hard case isolated by the policy-imprinting experiments without leaking
  `P_B` through a separate native prefix.
- **Compiler input:** the full natural-language target system prompt.  The
  current main arm obtains frozen base-model hidden states for this prompt and
  feeds them to a small trainable pooler.  This semantic encoding is compiled
  once per receiver policy and amortized across histories.  A token-embedding
  input is retained as the language-untrained ablation.  No row-specific target
  policy text is appended after `H`.
- **Compiler output:** `m` input-space virtual embeddings, processed after the
  stale history by the frozen LLM.  The first sweep uses `m in {8,16,32}`.

### 2.2 Baselines

1. target-native full re-prefill;
2. direct stale-history reuse without a tail;
3. complete plaintext target-policy tail (exact Tail-KV control);
4. 67-token compact plaintext target refresh;
5. compact refresh plus the existing shared 16-token soft suffix;
6. shared learned soft suffix with no prompt conditioner;
7. prompt-conditioned capsule (proposed pre-experiment).

The prompt compiler must also be evaluated with the source prompt and with its
fixed initialization capsule.  If swapping target/source compiler inputs does
not change behavior, the model has collapsed to a shared steering suffix and
cannot support a prompt-to-capsule claim.

### 2.2.1 Prompt compilation path

For a fixed receiver policy, compilation is an offline operation:

\[
S_B=\operatorname{stopgrad}(M_{enc}(P_B)),\qquad
Z_B=F_\phi(S_B).
\]

Only `Z_B` is attached to each request.  Online TTFT must therefore report the
`m`-slot frozen-LLM prefill separately from the one-time compilation cost.  For
dynamic one-shot policies, both amortized and non-amortized latency must be
reported.

### 2.3 Data and leakage controls

The existing V2 manifest contains long 3.5K--14.5K-token histories and BFCL
actions.  Native source and target behavior were screened before training.

- train: up to 1,200 eligible `train` rows;
- validation: 150 eligible `validation` rows;
- IID test: 200 eligible rows;
- OOD tests: `pair_ood`, `policy_ood`, `tool_ood`, `prompt_ood`, and
  `length_ood`;
- all model selection uses validation only;
- no test rows, role pairs, held-out policy dimensions, held-out tools, or
  held-out prompt templates may enter training.

The manifest audit already verifies zero pair overlap for `pair_ood`, zero
target-policy overlap for `policy_ood`, zero tool overlap for `tool_ood`, and
zero template overlap for `prompt_ood`.

### 2.4 Losses

For the target-native teacher action `y_t` and source action `y_s`:

\[
\mathcal L = \mathcal L_{CE}(y_t)
 + \lambda_{KD}\mathcal L_{topk\text{-}KD}
 + \lambda_c\operatorname{softplus}(\mu-
   [\ell(y_t)-\ell(y_s)])
 + \lambda_r\lVert Z_B-Z_0\rVert_2^2.
\]

Tokens around the first source/target action divergence receive additional
weight.  The top-k distillation term is diagnostic rather than a full-vocabulary
KL and must not be reported as exact KL.

### 2.5 Go / No-Go criteria

Proceed to the full method only if a 16- or 32-slot compiler satisfies all of:

1. IID compatible tool-call accuracy is at least 94% and no more than 3
   percentage points below the complete plaintext Tail-KV control;
2. source-action accuracy is at most 2%;
3. target-prompt capsules outperform source-prompt-swapped capsules by at
   least 10 percentage points, proving non-collapsed conditioning;
4. prompt-OOD and policy-OOD retain at least 90% of IID accuracy;
5. capsule compiler plus `m`-slot prefill is measurably cheaper than the
   complete plaintext tail in one-token TTFT tests.

Failure of criterion 3 is a method failure even if IID accuracy is high.  It
means the learned object is a universal readout override, not a compiled
receiver policy.

## 2.6 Running pre-experiment log

The following are diagnostic results, not final paper numbers:

| arm | training scale | IID/validation target EM | conditioning result |
|---|---:|---:|---|
| existing compact plaintext tail + shared k=16 suffix | 1,200 rows, 1 epoch | 95.5% on IID-200 | B policy still supplied as 66.82 plaintext tokens |
| input-embedding compiler, k=8 | 20 rows, 1 epoch | 10% on IID-10 | target/source prompt identical |
| input-embedding compiler, k=16 or 32 | 20 rows, 1 epoch | 0% on IID-10 | target/source prompt identical; some format collapse |
| input-embedding compiler, k=16 | one row repeated 100 times | 100% on the training row | without counterfactual loss, target and source prompts both produce target |
| input-embedding compiler + counterfactual source loss | one row repeated 100 times | 100% on the training row | target prompt gives target; source prompt gives source |
| input-embedding compiler + counterfactual loss, k=16/32 | 100 rows, 2 epochs | 3.3% / 6.7% on validation-30 | no target/source prompt gap |
| frozen-last-hidden compiler + counterfactual loss, k=16/32 | 20 rows, 2 epochs | 0% on validation-10 | insufficient data; no prompt gap |
| frozen-last-hidden compiler, k=16, lr=1e-4 | 1,200 rows, 1 epoch | 6% on validation-50 | source-prompt swap is 8%; negative 2pp gap |
| frozen-last-hidden compiler, k=16, lr=3e-4 | 1,200 rows, 1 epoch | 2% on validation-50 | source-prompt swap is 2%; zero gap |
| frozen-last-hidden compiler, k=16, lr=3e-4, source weight 5 | 1,200 rows, 1 epoch | 0% on validation-50 | both prompts retain the source action on 98% |
| structured `policy=value` oracle compiler, k=16 | 200 rows, 1 epoch | 6.7% on validation-30 | target/source descriptors identical; zero gap |
| binding-only hard tail, neutral prefix | no training | 93.5% on IID-200 | 12.26-token mean tail (9--16); 3% source leakage |
| minimal explicit hard tail, neutral prefix | no training | 97.0% on IID-200 | 28.08-token mean tail (25--31); 1% source leakage |

These diagnostics establish three facts.  First, the causal slot injection and
gradient path can exactly fit a receiver policy.  Second, a counterfactual
source-prompt arm is necessary to detect and prevent universal-steering
collapse.  Third, a language-untrained prompt encoder is not sample efficient;
the formal pilot therefore uses frozen semantic prompt states and the full
1,200-row training split.

The completed formal pilot adds a stronger negative result.  Frozen semantic
prompt states and 1,200 examples do not rescue the current compiler; increasing
counterfactual weight simply preserves the source policy.  Even an oracle
4--11-token structured descriptor fails, localizing much of the problem to the
constant-initialized capsule generator and its optimization rather than natural
language extraction alone.

More importantly, the synthetic benchmark is solved by hard tokens.  A
canonical binding-only system tail averages 12.26 tokens and reaches 93.5%
IID reuse accuracy.  A 28.08-token explicit rule reaches 97.0%, equal to
the complete plaintext Tail-KV result, with 1% source leakage.  Therefore a
16-slot learned capsule is currently Pareto dominated on this workload.  This
is a benchmark/method no-go, not evidence that late binding is impossible.
Future capsule training must use a prompt-copy or reconstruction initialization
and must be evaluated on policies whose semantics cannot be reduced to one
extractable key/value pair.

#### 2.6.1 Final hard-tail control

All rows use a neutral receiver prefix, target tool schema, source-imprinted
history K/V, and a canonical explicit receiver rule after the history.

| split | rows | target EM | source leakage | valid JSON |
|---|---:|---:|---:|---:|
| IID | 200 | 97.0% | 1.0% | 99.5% |
| held-out source/target pair | 200 | 99.5% | 0.0% | 100% |
| held-out policy family | 200 | 96.0% | 2.0% | 100% |
| held-out tools | 200 | 97.5% | 1.0% | 100% |
| held-out prompt template | 199 | 99.5% | 0.0% | 100% |
| extra-long history | 100 | 97.0% | 2.0% | 100% |

The IID native-neutral control is 97.5%, so stale-history reuse costs 0.5pp on
this arm.  Its paired table is 193 both correct, 2 native-only, 1 reuse-only,
and 4 both wrong.  The rule tail is 25--31 tokens (mean 28.08, median 28).

The 28-token hard tail also reaches 99.5% on held-out source/target pairs, 96.0%
on held-out policy families, 97.5% on held-out tools (200 rows each), 99.5% on
all 199 eligible held-out prompt templates, and 97.0% on 100 extra-long
histories.  Source leakage is 0--2% across these splits.  These results show
that the baseline is executing a late-bound rule rather than only memorizing
training pairs.  They also make hard-token prompt compilation the primary
baseline for any successor method.

### 2.7 Formal pilot decision

**Decision: NO-GO for the current prompt-to-16-slot compiler and for the
single-key/value benchmark as evidence of capsule value.**

The current method fails criteria 1--3: its best formal validation target EM is
6%, source leakage is 12% in that run, and its prompt-swap gap is negative.  The
hard-tail control passes the behavioral criteria while requiring only about 28
tokens; the binding-only control is about 12 tokens and already reaches 93.5%.
Consequently, more epochs or a larger compiler on this benchmark would be an
unprincipled hyperparameter search, not the next scientific experiment.

The next gate is a new compositional-policy benchmark and a reconstruction-
initialized compiler.  Continue only if all of the following hold:

1. policies contain conjunctions, exceptions, priorities, permissions, and
   private clauses that cannot be losslessly reduced to one `key=value` pair;
2. a matched-budget canonical hard summary is included at every slot budget;
3. the capsule starts from prompt-token reconstruction/copying and learns a
   residual, rather than generating every slot from a generic constant suffix;
4. full-sequence counterfactual distillation supervises both target and source
   policies, not only one first-divergence logit;
5. success is reproduced on held-out policy programs and at least two model
   families before any serving integration claim is made.

## 3. ICLR-level method after the pre-experiment

The prompt compiler is deliberately the smallest falsifiable method.  Three
objects that are easy to conflate must be separated:

1. **static KV template:** `F(P_B)` emits layer-wise K/V tensors and concatenates
   them directly.  This is cheapest, but its template cannot read `H` and is
   therefore context independent;
2. **causal embedding template (the current pilot):** `F(P_B)` emits `m` input
   embeddings.  Processing only these `m` positions lets them attend to the
   immutable source history, so their realized layer-wise K/V states are
   context dependent even though their seed is compiled once per receiver;
3. **explicit history-conditioned KV template:** `F(P_B,Q(C_A(H)))` directly
   emits or adjusts layer-wise K/V slots using a cheap cache probe.

A static KV template cannot in general reconstruct the native receiver state
because

\[
\operatorname{KV}(P_B\mid H_1)\ne
\operatorname{KV}(P_B\mid H_2).
\]

The causal embedding template partly avoids this impossibility: the frozen
Transformer itself acts as a history-conditioned materializer for the `m`
slots.  It pays `m` suffix-token forwards rather than replaying every history
token.  If this pilot is insufficient, the paper method should add a cheap,
explicit history probe:

\[
Z_{B,H}=F_\phi(E(P_B), Q(C_A(H))).
\]

`Q` must read only a small set of cached layers/tokens or pooled cache
statistics.  It must not replay the complete history.  A low-rank,
layer-gated read lens over the causally identified policy-imprinting layers can
be added only when the capsule risk score is high.  The runtime policy is:

1. direct exact reuse when the leakage risk is low;
2. prompt-conditioned capsule for ordinary handoffs;
3. capsule plus context-conditioned read lens for high-risk handoffs;
4. selective replay/full re-prefill fallback when confidence is insufficient.

The publishable claim is therefore not that a suffix soft prompt exists.  It
is that receiver policies can be late-bound to an immutable shared causal
memory through a compact, generalizable, risk-aware interface, with measured
behavioral fidelity and end-to-end serving gains.

### 3.1 Proposed full method: LateBindKV

For source cache `C_A(H)` and private receiver policy `P_B`, the full method has
four modules:

1. `E(P_B)` is a frozen semantic encoder and a small trainable policy pooler;
2. `Q(C_A(H))` selects a constant-size set of anchor states and cache statistics
   from causally identified policy-sensitive layers;
3. `G(E(P_B),Q(C_A(H)))` emits `m` slot seeds, low-rank per-layer residuals, and
   a confidence score;
4. the frozen LLM materializes the slots as suffix positions and either decodes
   or invokes selective replay when confidence is low.

The default fast path omits explicit residuals and uses only the causal
embedding slots.  Residuals and replay are conditional costs, not paid on every
handoff.  Each downstream reader gets copy-on-write capsule blocks; the shared
history cache is never mutated.

The essential training unit is not one `(history, target policy)` example but a
counterfactual tuple

\[
(C_A(H),P_A,P_B,y_A,y_B).
\]

The same history is decoded with both `P_A` and `P_B`.  A target-distillation
loss teaches fidelity, a source-condition loss teaches prompt dependence, and
a swap-margin loss requires the two capsules to disagree precisely where the
native agents disagree.  This construction turns “does the suffix work?” into
an identifiability test for the policy compiler.

The negative pilot changes the compiler initialization.  Let `T(P_B)` be the
base model's token embeddings for either the full policy or a verified policy
IR.  A length adaptor first constructs an information-preserving seed

\[
Z_B^0=R_\psi(T(P_B)),\qquad
\widehat T(P_B)=D_\omega(Z_B^0),
\]

with token/semantic reconstruction pretraining.  The online capsule is a
residual around this seed,

\[
Z_{B,H}=Z_B^0+G_\phi(Z_B^0,Q(C_A(H))).
\]

This removes the observed constant-initialization bottleneck and makes exact
policy identity available before behavioral distillation.  `R` must be compared
against head/tail truncation, mean pooling, learned prompt compression, and
matched-length hard summaries.  A structured policy IR is an oracle/production
variant, not a substitute for full-prompt generalization.

### 3.2 Complexity and systems claim

Let `n` be the shared-history length, `p` the receiver prompt length, and `m`
the capsule length.  Ignoring already cached source work, native B execution
re-prefills `n+p` positions.  The causal capsule processes only `m` new
positions whose queries attend to `n` cached positions.  Its attention work is
`O(L m(n+m)d)` and its MLP work is `O(L m d^2)`, versus replay work containing
`O(L n^2 d)` attention and `O(L n d^2)` MLP terms.  A direct-KV fast path can
remove even the `m`-position materialization cost, but must pass the fidelity
gate.

This asymptotic argument is only motivation.  The paper claim requires measured
TTFT/QPS/HBM on a serving engine, including cache lookup, slot materialization,
copy-on-write metadata, compiler amortization, and fallbacks.  FLOP estimates
alone are not a systems result.

### 3.3 Falsifiable hypotheses

- **H1, late-binding fidelity:** a capsule reaches a pre-registered
  non-inferiority margin against native B without plaintext `P_B` after `H`;
- **H2, policy identifiability:** swapping `P_B` for `P_A` changes behavior in
  the corresponding native direction, not merely the output format;
- **H3, compositional generalization:** the compiler transfers to held-out
  policy values, prompt paraphrases, source/target pairs, and tools;
- **H4, context necessity:** history-conditioned materialization beats static
  direct K/V specifically on histories with large source-policy imprint;
- **H5, useful speed-quality frontier:** at matched behavioral risk, LateBindKV
  improves P50/P95 TTFT or throughput over full replay and over the strongest
  hard-token refresh baseline.

Failure of H2 rejects the policy-compiler story.  Failure of H3 reduces the
method to a finite role lookup table.  Failure of H5 leaves a representation
result but not a compelling multi-agent serving result.

### 3.4 Novelty boundary

- Relative to **KVCOMM**, the shared historical cache is assumed available;
  the new problem is binding a previously unseen receiver policy while keeping
  that cache immutable.
- Relative to **KV-cache steering**, the intervention is prompt-conditioned,
  counterfactually tested, and optimized for exact agent policy rather than a
  single global behavior direction.
- Relative to **prefix/soft-prompt compression**, the object being compressed
  is a private receiver policy that is attached *after a cache produced under a
  conflicting source policy*, not merely a long prompt in its native context.
- Relative to **Tail-KV**, the complete plaintext B tail is replaced by a
  constant-size learned interface and a calibrated fallback policy.

Without held-out-policy generalization and real serving gains, this boundary is
not strong enough for an ICLR main-conference claim.

## 4. Required final evaluation

- Models: at least Qwen3-8B, Qwen3-14B, and one Llama-family checkpoint.
- Workloads: controlled policy conflicts, BFCL/tool authorization, long RAG,
  3/4-agent chains, fan-out/fan-in, and multiple predecessors.
- Fidelity: task score, compatible and strict tool-call EM, target-native
  agreement, source-action leakage, teacher-forced sequence KL/NLL, and
  prompt-injection/permission tests.
- Systems: prefill tokens/FLOPs, compiler time, one-token TTFT, P50/P95
  latency, throughput, HBM footprint, capsule storage, and concurrency.
- Ablations: slot count, compiler depth, input text versus structured policy,
  prompt-only versus history-conditioned, layer gates, risk fallback, and
  static/shared capsule.
- Statistics: paired bootstrap confidence intervals, McNemar tests for binary
  outcomes, non-inferiority intervals against native execution, and results
  stratified by policy family, prompt template, history length, and direction.
