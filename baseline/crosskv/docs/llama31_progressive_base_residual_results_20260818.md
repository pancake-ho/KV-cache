# Llama-3.1 progressive base/residual results

## Decision

The gradient-isolated progressive representation fails both preregistered
quality gates while retaining strong source causality.  In contrast, the
compute-matched all-layer continuation control produces a new best compact
packet: additional full-depth training improves first16 F1 by `.098` over its
parent with an ordinary and cluster-bootstrap CI above zero.

The result reverses the proposed mechanism.  Useful early summary state is not
improved by isolating it from deep losses.  It emerges from end-to-end
full-receiver credit assignment and can then be transmitted as a post-training
depth prefix.

## Frozen data audit

The incremental train256 excludes 1,152 prior IDs and has ordered-ID/file
SHA256 values
`62d5ee7106eeafd4179e172336d26a5c1dc0468cdf039e464dcbb2fdc299e882` /
`c0c5ec89ffe569fef6b94e387c1b4ceb42e271787cc0686a0cc249d6dc4ffa1b`.
The confirm128 additionally excludes those 256 IDs and has hashes
`5f611bc088e017013e9b19cea4f7fab9c2ee1bdca8b52eb17a68c9d7eb851663` /
`1900657b9479fcb080145e8c489b5fd478234d69d1d62f5a5797a73b5e44abef`.
It contains 128 unique cases and 73 bridge-answer clusters.  All overlap
audits are zero.

## Fixed training

Both new writers start from parent checkpoint SHA256
`b5848635fac01561b8f694aeca79d5d02d340a5c15eeb1e3967d21c90cb0b620`
and use 3,200 updates on the new train256.

- all-only continuation: all layers, stage/bridge weights 1/2;
- progressive: uniform first16/all, realized counts 1,608/1,592; first16
  stage/bridge 1/0, all stage/bridge 1/4; layer-16 sender and receiver gradient
  boundary on all updates.

Training took 1,298.4 s for all-only and 1,043.9 s for progressive.  Final
aggregate stage/bridge CE was `.0744/.0321` for all-only and `.2480/.0887` for
progressive.  The progressive deep loss became less stable after about step
2,300; the frozen final checkpoint was retained and no early checkpoint was
selected.

## Absolute confirm128 results

| writer / source | first16 EM | first16 F1 | all EM | all F1 |
|---|---:|---:|---:|---:|
| parent all-only N512 | .633 | .734 | .773 | .819 |
| **continued all-only +256** | **.773** | **.832** | **.836** | **.874** |
| progressive base/residual | .578 | .679 | .648 | .733 |
| progressive shift-1 | .156 | .258 | .203 | .301 |
| no-summary | .094 | .198 | .094 | .198 |

Payload is 67,584 B for first16 and 135,168 B for all.  For all-only, bridge
EM/F1 is `.117/.203` at first16 and `.812/.937` at all.  For progressive it is
`.000/.000` and `.633/.771`, confirming that the intended functional/lexical
separation occurred but did not improve task utility.

## Preregistered gates

Progressive versus compute-matched all-only:

1. **first16 improvement fails:** F1 delta `-.1531`, ordinary 95% CI
   `[-.2344,-.0742]`, bridge-cluster CI `[-.2392,-.0713]`; EM delta `-.1953`,
   McNemar `p=7.03e-5`;
2. **all non-inferiority fails:** F1 delta `-.1409`, ordinary CI
   `[-.2122,-.0724]`, cluster CI `[-.2127,-.0736]`; both lower bounds are far
   below the `-.05` margin;
3. **source causality passes:** correct versus shift first16 F1 delta `+.4209`,
   ordinary CI `[+.3323,+.5084]`, cluster CI `[+.3252,+.5186]`; EM delta
   `+.4219`, McNemar `p=9.00e-13`.

The packet remains a sample-specific causal state, but the progressive
optimization is worse.

## Secondary parent comparison

The all-only continuation improves parent first16 F1 by `+.0984`, ordinary CI
`[+.0359,+.1641]`, cluster CI `[+.0378,+.1621]`; EM improves by `+.1406`,
McNemar `p=.00053`.  Its all-layer F1 improves by `+.0549`, with ordinary CI
`[-.0133,+.1227]` and cluster CI `[-.0141,+.1259]`.

Progressive does not merely lose to the stronger continuation control.  It is
also lower than the parent by `.0547` first16 F1 (CI crosses zero) and `.0859`
all F1 (ordinary CI `[-.1622,-.0109]`, cluster CI
`[-.1688,-.0064]`).

## Mechanism conclusion

The early and deep cache are functionally different but not independently
optimizable modules.  Deep receiver losses provide useful credit to the
sender embeddings and early adapter blocks.  Cutting that path makes the
deep residual less stable and also prevents the compact base from benefiting
from full-task supervision.  Therefore:

- do not use stochastic multi-budget masking, budget-conditioned lexical loss,
  or a hard stop-gradient boundary as the main method;
- train the writer end to end with the full-depth task objective and choose a
  depth-prefix packet at serving time;
- treat more diverse writer data as the current proven optimization axis:
  N512 plus 256 new cases raises compact first16 performance without a
  compact-specific loss.

The next paper experiment should freeze the new all-only checkpoint on a
cross-task benchmark and measure the first16/all quality--payload frontier.  A
new learned router is not justified until a task demonstrates that static
runtime depth selection is insufficient.

Artifacts:

- preregistration:
  `docs/llama31_progressive_base_residual_preregister_20260818.md`;
- all-only continuation:
  `outputs/amortized_semantic_kv/llama31_increment256_allonly_cont3200_confirm128_20260818/`;
- progressive candidate:
  `outputs/amortized_semantic_kv/llama31_increment256_progressive_base_residual_cont3200_confirm128_20260818/`;
- progressive shift:
  `outputs/amortized_semantic_kv/llama31_increment256_progressive_base_residual_shift1_confirm128_20260818/`;
- parent evaluation:
  `outputs/amortized_semantic_kv/llama31_parent_allonly_confirm128_20260818/`;
- paired analyses:
  `outputs/amortized_semantic_kv/llama31_progressive_base_residual_confirm128_analysis_20260818/`.
