# Receiver-Lens task-gradient structure result

Status: completed on 2026-08-18 under the frozen protocol in
`llama31_receiver_lens_gradient_structure_preregister_20260818.md`.

## Decision

The conditional receiver-writer gate **fails**.  Gradient directions are
strongly domain-structured and the two domains' joint training objectives are
significantly opposed, but the predeclared primary comparison connecting the
observed development-final/MuSiQue-bridge behavioral tradeoff has a confidence
interval that crosses zero.  No conditional writer experiment is authorized
by this diagnostic.

## Integrity

- Development cases: 56; MuSiQue safety cases: 32.
- Overlap with 568 Receiver-Lens4 training IDs: zero.
- Every source used the real 270,426-byte canonical K4/V4 CE8 packet.
- Gradients were taken only with respect to the 16,384 trained lens embedding
  parameters; no optimizer was constructed.
- CE8 parameter hash before/after:
  `5acd4c105c2cba8c427a641f0e8a502b5d486b54882b10255160891058272f44`.
- Lens parameter hash before/after:
  `3ac97332ff05ae545a0be2d3dff60a5350bfde264d8bef2d49ced5e9ba968219`.
- Packet preparation/probe time: 46.86 s / 48.85 s.
- Bootstrap: 20,000 resamples, seed 2113.

## Aggregate-gradient comparisons

| comparison | aggregate cosine | 95% CI | role |
|---|---:|---:|---|
| development final vs MuSiQue bridge | -0.172237 | [-0.262603, +0.088737] | **primary gate: fail** |
| development joint vs MuSiQue joint | -0.366220 | [-0.563419, -0.051925] | secondary, significant conflict |
| development final vs bridge | +0.186878 | [-0.080641, +0.241370] | secondary |
| MuSiQue final vs bridge | -0.069923 | [-0.146217, +0.130952] | secondary |

`joint` is exactly `final + 2 * bridge`, matching Receiver-Lens4 training.
The primary point estimate is negative, but its uncertainty is asymmetric and
includes positive alignment; it cannot be promoted using the secondary result.

## Domain-conditioned structure

After normalizing every per-case joint gradient, the own-domain leave-one-out
centroid cosine minus other-domain centroid cosine is `+0.427728`, bootstrap
CI `[+0.291035,+0.561694]`; 71.59% of cases have positive margins.  The domain
means are `+.15250` on development and `+.90937` on MuSiQue.  Mean pairwise
cosines are:

- development within: `+.00729`;
- MuSiQue within: `+.31466`;
- cross-domain: `-.05747`.

Thus the receiver-task identity predicts gradient direction.  This passes the
second frozen condition, but conditions cannot be substituted for one another.

## Spectrum

For the 88 x 16,384 normalized joint-gradient matrix:

- stable rank: `4.1814`;
- entropy effective rank: `38.6108`;
- 50% / 90% / 95% energy ranks: `10 / 53 / 66`;
- the first component contains 23.92% of normalized energy.

The gradients are therefore neither one shared direction nor a four-direction
object.  This is descriptive evidence that fixed global embeddings average
heterogeneous requests, but spectral complexity was deliberately not a gate.

## Mechanism conclusion

Semantic immutability successfully prevents writer forgetting, and a trained
lens can causally read CE8, but the remaining problem is not established as a
single clean final-versus-bridge gradient collision.  The opened data support
a broader domain-dependent joint-objective conflict.  A query-conditioned
writer might fit that structure, but running it after the primary gate failed
would be post-hoc escalation.

The admissible design lesson is narrower: do not train one global readout state
across heterogeneous receiver identities and expect replay alone to make it
safe.  A future experiment must be independently motivated and frozen, for
example an explicitly receiver-identity-indexed lens bank evaluated on a new
development split.  It cannot reuse this failed diagnostic as confirmation.

## Artifacts

- summary and all per-case losses/norms:
  `outputs/amortized_semantic_kv/llama31_receiver_lens4_gradient_structure_dev56_musique32_20260818/analysis.json`
- complete FP32 final/bridge/joint gradient matrices:
  `outputs/amortized_semantic_kv/llama31_receiver_lens4_gradient_structure_dev56_musique32_20260818/analysis.gradients.pt`
- cached real-codec source packets:
  `outputs/amortized_semantic_kv/llama31_receiver_lens4_gradient_structure_dev56_musique32_20260818/analysis.packets.pt`
