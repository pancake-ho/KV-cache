# Multi-domain writer checkpoint-drift localization

Status: frozen on 2026-08-18 before computing any parent--candidate drift
statistics.

## Scope

This is an offline analysis of two already-trained checkpoints.  It starts no
training, performs no task generation, and does not inspect any sealed E set.
Its purpose is to determine whether the severe bridge forgetting observed after
multi-domain continuation is concentrated in a separable capsule module or is
distributed throughout the writer.

- Parent: `llama31_native_tail_scaleup_ce8_800_train128_eval64_20260818`.
- Candidate:
  `llama31_ce8_multidomain_decision_cont1600_train568_dev56_20260818`.

The checkpoints have the same eight-slot embeddings and the same 31 rank-4
layerwise residual blocks.  Slot identities and block indices are compared
directly because the candidate was initialized exactly from the parent.

## Frozen measurements

For the embeddings and every `down`/`up` tensor, report parent norm, candidate
norm, delta norm, relative delta, cosine, and share of total squared parameter
drift.  Parameter drift is secondary because a low-rank factorization admits
rescaling and basis changes that need not change its function.

The primary writer measurement therefore evaluates every residual block on the
same 512 deterministic Gaussian hidden vectors (seed 2103).  Inputs are RMS
normalized exactly as in the writer.  For each block report:

- RMS parent residual, candidate residual, and residual difference;
- difference RMS divided by parent residual RMS;
- cosine between flattened parent and candidate residuals;
- the block's share of summed functional-drift energy across all 31 blocks.

Also report functional-drift shares for early blocks 0--9, middle blocks 10--20,
and deep blocks 21--30; the eight individually largest blocks; and the most
concentrated contiguous window of eight blocks.

## Frozen interpretation

The result is called **embedding-dominant** only if embeddings contain at least
50% of total squared parameter drift.  It is called **layer-localized** only if
either the eight largest blocks or one contiguous eight-block window contains
at least 60% of functional-drift energy.  Otherwise it is **distributed**.

- Embedding-dominant drift motivates a future frozen-embedding intervention.
- Layer-localized drift motivates a future frozen-region intervention at the
  identified layers.
- Distributed drift rejects simple module freezing as a sufficient explanation
  and strengthens the case for an explicit immutable semantic base plus a
  separately parameterized decision residual.

These labels are structural diagnostics, not performance gates.  Random-probe
function drift does not establish that a changed block caused either the final
gain or bridge loss.  Any hybrid-checkpoint or retraining test must be separately
preregistered and run only after experimentation resumes.
