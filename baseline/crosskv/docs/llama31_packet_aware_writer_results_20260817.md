# Llama-3.1 packet-aware writer development result

## Decision

The fixed packet-aware writer did not pass its development gate.  No fresh
confirmation set was constructed and no Hotpot run was authorized.

The implementation and optimization both worked: the writer was trained for
the fixed 1,600 updates with receiver losses seeing only last-16 INT4 capsule
K/V, and training loss fell to `.9448` (stage CE `.4624`, bridge CE `.2412`).
The failure is held-out utility, not a crashed or non-differentiable training
path.

## Dev32 results

| writer | all BF16 F1 | all INT4 F1 | last16 BF16 F1 | last16 INT4 F1 |
|---|---:|---:|---:|---:|
| unconstrained N=512 | .547 | .528 | .266 | .266 |
| packet-aware N=512 | .000 | .000 | **.315** | **.315** |

At the target last16-INT4 cell, packet-aware minus unconstrained F1 is
`+.0490`, bootstrap 95% CI `[-.1354,+.2292]`; EM changes by `+.0313`, CI
`[-.1563,+.2188]`.  It has 7 F1 wins, 21 ties, and 4 losses.  The frozen gate
required at least `+.10` F1, so it fails.  The same candidate-versus-no-summary
contrast is also only `+.0490`, because the unconstrained last-half arm equals
no-summary on this dev slice.

The target arm does carry more lexical state than the baseline: bridge F1 is
`.276` versus zero.  But 32-case downstream evidence is weak and cannot justify
opening a new set.

## What the all-layer collapse means

During packet-aware training, early-layer capsule K/V is masked before every
receiver loss.  Those unsent vectors still exist inside the checkpoint, but
receive no direct constraint as transmitted cache.  Restoring them at
evaluation reduces stage F1 from `.315` to zero.  This shows that an unobserved
layer is not safely equivalent to an omitted layer: arbitrary K/V can alter
the softmax and actively corrupt later computation.

This does not invalidate the deployed last-half packet, where those vectors
remain logically zero.  It does reject a naive multi-budget interpretation of
the checkpoint and suggests that a future budget-conditioned writer needs one
of:

- stochastic/all-plus-budget training so every emitted layer remains valid;
- an explicit null-cache regularizer for inactive layers;
- separate packet heads per supported depth budget.

## Next diagnostic

Before another training intervention, run the frozen first-16-only diagnostic
on both unconstrained writers.  If first16 alone preserves the strong N=512
state, simple depth routing is preferable to forcing all information into the
last half.  If both halves fail, the evidence favors cross-depth synergy and
the 135 KB all-layer INT4 packet over another post-hoc mask.

Protocol: `docs/llama31_half_depth_localization_diagnostic_20260817.md`.
Paired artifacts:
`outputs/amortized_semantic_kv/llama31_packetaware_writer_dev32_20260817/`.
