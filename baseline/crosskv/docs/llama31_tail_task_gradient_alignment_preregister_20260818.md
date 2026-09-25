# Tail-KV native-objective/task-gradient alignment diagnostic

Status: frozen on 2026-08-18 before observing any diagnostic output.

## Question

The cap8 scale-up established a sharp contradiction: direct native Tail-state
distillation makes the transmitted K/V much closer to a privileged gold-answer
Tail teacher, but makes final answer quality worse than equal-capacity CE
training.  The fixed-query attention-operator probe showed the same reversal.
This diagnostic asks whether the native targets are locally opposed to the
actual receiver task, rather than merely being imperfect distance metrics.

For every held-out case, all losses are differentiated with respect to the
same 1,048,576 trainable CE8 capsule parameters.  We measure the global cosine
between native-objective gradients and receiver-task gradients.  No parameter
is updated in this diagnostic.

## Frozen checkpoint and cases

- Model: local `Llama-3.1-8B-Instruct`, BF16.
- Capsule: CE8 from
  `llama31_native_tail_scaleup_ce8_800_train128_eval64_20260818/capsule.pt`.
  SHA-256:
  `7a0ff34d9fe738a621abfdeab58ff62781dd11bdd7def977af29323ff68cab9e`.
- Dataset: `confirm128_seed2083_exclude_used1408.jsonl`, rows 96--127
  (32 cases).
  SHA-256:
  `1900657b9479fcb080145e8c489b5fd478234d69d1d62f5a5797a73b5e44abef`.
- These cases do not overlap the checkpoint's 128 training IDs and were not
  used to decide the cap8 native-state result.  They are diagnostic cases, not
  a new paper test set.
- Bootstrap: 20,000 paired case resamples, seed 2098.

## Frozen representation and losses

- The student materializes eight slots after the full sender document.
- Its all-32-layer cache is moved to canonical position zero, fake-quantized
  and decoded with production-equivalent K4/V4, then moved after the receiver
  system prefix.
- Teacher: the first eight tokens of the gold bridge answer, materialized after
  the same sender document and passed through the same canonical K4/V4 path.
- Native losses use layers 1--31.  Layer 0 is excluded because a fixed soft
  embedding cannot independently match an arbitrary token's layer-0 K/V.
- `coordinate`: prefix-relative K/V MSE against the teacher.
- `operator`: teacher-normalized conditional-value error plus 0.1 times
  attention log-mass error, using fixed teacher-forced final receiver queries.
- `final`: teacher-forced CE of the dependent receiver's final answer.
- `bridge`: teacher-forced CE of the intermediate receiver's bridge answer.

The five predeclared alignments are `coordinate-final`,
`coordinate-bridge`, `operator-final`, `operator-bridge`, and
`final-bridge`.  Per-case global gradient dot products, norms, and cosines are
recorded; inference is on the mean cosine and negative fraction.

## Frozen decision gates

For a native objective `N`, define its bridge-minus-final alignment gap as
`cos(N, bridge) - cos(N, final)`.

Strong objective conflict requires both:

1. `N-final` mean-cosine bootstrap 95% CI has upper bound below zero, **or** at
   least 65% of cases have negative cosine; and
2. the bridge-minus-final gap has bootstrap 95% CI lower bound above zero.

The coordinate loss is primary; the operator loss is a separately declared
secondary diagnostic.  Exactly one task-tangent native-distillation candidate
is authorized only if either native objective passes both conflict conditions.
The projected auxiliary gradient must preserve the joint protected task
gradient (`final + 2 * bridge`) and may remove only its opposing component.
The final/bridge alignment is reported to determine whether the protected task
itself contains substantial conflict; it is not an authorization gate.

If neither native objective passes, no PCGrad/task-tangent training run is
authorized.  The next intervention must instead expose the auxiliary target to
the complete receiver prefix and on-policy receiver queries, because the local
gradient-conflict account would lack support.

This diagnostic cannot establish generalization, improve the frozen quality
result by itself, or convert these 32 cases into a confirmatory benchmark.
