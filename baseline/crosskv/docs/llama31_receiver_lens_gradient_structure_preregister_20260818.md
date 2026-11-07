# Receiver-Lens task-gradient structure diagnostic

Status: frozen on 2026-08-18 before computing any diagnostic gradient.

## Question

Receiver-Lens4 raises new-domain development final F1 while significantly
reducing the original MuSiQue bridge readout.  This diagnostic asks whether
that behavioral tradeoff is locally visible as incompatible, domain-structured
directions in the trained lens embedding space.  It does not update a model,
change the slot count, or evaluate a new method.

If one global lens is being asked to move in conflicting directions that are
predictable from the downstream task, then a query-conditioned receiver-local
writer is structurally motivated.  If the gradients are aligned or the
directions have no domain structure, adding such a writer is not authorized by
this result.

## Frozen artifacts and cases

- Receiver: local Llama-3.1-8B-Instruct, BF16.
- Frozen CE8 base checkpoint SHA-256:
  `7a0ff34d9fe738a621abfdeab58ff62781dd11bdd7def977af29323ff68cab9e`.
- Trained Receiver-Lens4 checkpoint SHA-256:
  `ce1764b97e9c3137846f2c5f984ec6597737ebebcc543bc3eb24a01201755224`.
- Development: all 56 rows of `longbench_standard_dev.jsonl`, SHA-256
  `2f4eec68a0a0c3700b4676c9f683b5443e94d6fdcfc3e38f1828d1f75668b9cb`.
- Safety: rows 96--127 of
  `confirm128_seed2083_exclude_used1408.jsonl`, SHA-256
  `1900657b9479fcb080145e8c489b5fd478234d69d1d62f5a5797a73b5e44abef`.
- Both diagnostic sets have zero case-ID overlap with all 568 lens training
  cases.  They are already-opened development/safety data, not external E sets.
- Bootstrap: 20,000 resamples, seed 2113.

Every document is re-encoded with the frozen CE8 writer and passed through the
real canonical K4/V4 codec.  The packet is 270,426 bytes.  The final trained
lens is used without modification.

## Frozen gradients

For each case, differentiate the following teacher-forced losses only with
respect to the same 16,384 Receiver-Lens4 embedding parameters:

- `final`: dependent-agent final-answer CE;
- `bridge`: intermediate/source-answer CE;
- `joint = final + 2 * bridge`, matching lens training.

Record each loss, gradient norm, and per-case final/bridge cosine.  Save the
complete FP32 gradient matrices as an audit artifact.  The Llama model, CE8
writer, and lens are not updated; before/after hashes of CE8 and lens parameters
must be identical.

The primary conflict comparison is the cosine between the raw aggregate
development-final gradient and raw aggregate MuSiQue-bridge gradient.  Its
confidence interval resamples cases independently within the two domains and
recomputes both aggregate vectors.  Development-joint versus MuSiQue-joint and
within-domain final versus bridge are predeclared secondary comparisons.

To test whether disagreement is task-conditioned rather than unstructured
case noise, L2-normalize each per-case joint gradient.  For every case, compute
the cosine to its own domain's leave-one-out centroid minus the cosine to the
other domain's centroid.  Bootstrap the mean of these 88 fixed per-case
margins.

For interpretation only, report the eigenvalue spectrum of the 88 normalized
joint gradients: stable rank, entropy effective rank, and ranks needed for
50%, 90%, and 95% energy.  No spectral threshold is an authorization gate.

## Frozen decision gate

Exactly one small query-conditioned receiver-writer feasibility experiment is
authorized only if both conditions hold:

1. the primary development-final versus MuSiQue-bridge aggregate-gradient
   cosine has a bootstrap 95% CI upper bound below zero; and
2. the own-domain-minus-other-domain centroid margin has a bootstrap 95% CI
   lower bound above zero.

Condition 1 connects the diagnostic directly to the observed behavioral
tradeoff.  Condition 2 requires that the direction be predictable from the
receiver task, which is what a conditional writer could exploit.  Point
estimates, effective rank, a negative fraction, or either condition alone do
not authorize the writer.

If the gate passes, it authorizes only a separately preregistered retained-base
feasibility test.  It does not authorize eviction, external evaluation, rank or
slot sweeps, modification of CE8, or a paper claim that the writer works.  If
the gate fails, the conditional-writer hypothesis is rejected at this point.
