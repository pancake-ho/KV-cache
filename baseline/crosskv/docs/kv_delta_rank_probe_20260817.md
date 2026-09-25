# Qwen3-8B KV-delta rank probe (2026-08-17)

## Verdict

The measured KV delta is **not very low rank** under the strict rank-8/rank-16
criterion. A single shared low-dimensional basis across arbitrary contexts is a
clear no-go. An independently generated rank-32 matrix update is much closer,
especially for keys, but still leaves a non-negligible value-cache residual.

The defensible next design is therefore not a fixed universal delta basis. It is
a dynamic, layer-wise and K/V-asymmetric update, followed by a behavioral test at
ranks 16, 32, 48, and 64.

## Protocol

- Model: `Qwen/Qwen3-8B` (36 layers, 8 KV heads, head dimension 128).
- Data: MuSiQue development paragraphs.
- Two independent seeds selected 12 non-overlapping target documents in total.
- Each target was fixed to the same 128 tokens under 64 preceding contexts.
- Each context contained six fixed-width 32-token chunks.
- `permutation`: the same six chunks, with only their order changed.
- `composition`: both chunk membership and order changed.
- 48 contexts fit the basis; 16 unseen contexts measured reconstruction.
- Keys were transformed by the exact inverse of the model's RoPE at their source
  positions before comparison. Values were used directly.
- All aggregate explained-variance numbers below are weighted by delta Frobenius
  energy, so near-zero layer-0 cells cannot make the result look artificially good.

Two complementary ranks were measured:

1. **Shared context-basis rank.** For every layer/head/K-or-V cell, each context's
   `[128, 128]` cache block was flattened. PCA was fit on the 48 training contexts,
   and held-out deltas were oracle-projected onto that fixed basis. This is an
   optimistic upper bound for a mapper that uses one shared basis.
2. **Individual matrix-delta rank.** For a pair of contexts, the
   `Delta K` or `Delta V` matrix itself was decomposed by an oracle SVD. Each
   context pair may use different factors. This is an optimistic upper bound for
   a dynamic mapper that emits per-request low-rank factors.

The decision rule was fixed before the formal runs:

- strong low rank: rank-8 >= 90% and rank-16 >= 95% held-out energy;
- moderate: rank-16 >= 90% and rank-32 >= 95%;
- high-rank/reassess: rank-32 < 90% for a shared basis.

## Results across both seeds

### One shared context basis

| Context change | rank-8 | rank-16 | rank-32 | Decision |
|---|---:|---:|---:|---|
| Reorder only | 53.6% | 64.2% | 71.0% | High rank / no-go |
| Composition + order | 25.4% | 31.9% | 36.9% | High rank / no-go |

This result is decisive: even rank 32 fails to reconstruct most unseen
composition deltas. The poor generalization is not caused by RoPE or a single
outlier layer.

### A separate low-rank matrix for every context pair

| Context change | rank-8 | rank-16 | rank-32 | Interpretation |
|---|---:|---:|---:|---|
| Reorder only | 76.5% | 87.0% | 94.4% | Borderline around rank 32-48 |
| Composition + order | 69.0% | 81.5% | 91.8% | Needs more than rank 32 |

The median per-cell rank needed for 95% energy was 40-43.5 for reorder and
42-45 for composition across the two seeds. The cell-level p90 was about 58-60,
so a uniform rank-16 adapter is not supported by the data.

### Keys and values behave differently

| Context change | Cache | rank-8 | rank-16 | rank-32 |
|---|---|---:|---:|---:|
| Reorder only | K | 81.5% | 90.6% | 96.6% |
| Reorder only | V | 74.6% | 85.6% | 93.6% |
| Composition + order | K | 80.0% | 89.3% | 96.1% |
| Composition + order | V | 63.1% | 77.3% | 89.6% |

Keys are consistently more compressible. Values dominate the failure, especially
when the preceding document set changes. This argues for a heterogeneous method:
rank-32 K correction may be viable, while V needs a higher/adaptive rank, sparse
token refresh, or selective recomputation.

## Layer check

Layer 0 has essentially zero context delta, as expected: its projected K/V for a
token is formed before any attention to earlier tokens. From layer 1 onward the
high-rank trend persists. Middle-layer values are often the hardest cells; in the
composition regime, representative value-cache rank-32 matrix reconstruction was
only about 86-93%, with median per-cell r95 frequently around 50-57. The result is
therefore not an artifact of averaging a few bad final layers.

## What this establishes—and what it does not

This is a necessary-condition test using raw KV Frobenius energy. The oracle PCA
and SVD results are deliberately favorable: no learned system-prompt encoder can
beat them at the same rank on this reconstruction objective. Thus a fixed shared
rank-8/16 basis should be rejected.

However, raw KV residual energy is not the final task metric. Attention may ignore
some high-rank components. The next experiment should replace native cache blocks
with their oracle rank-8/16/32/48/64 reconstructions and measure attention-output
error, next-token KL, and MuSiQue answer accuracy. A positive result there would
support a dynamic K-low-rank plus V-refresh design, not the original universal
low-rank basis.

## Artifacts

- Seed 2027: `outputs/kv_delta_rank/qwen3_8b_musique_d6_c64_t128_matrix_gram_20260817/summary.json`
- Seed 2028: `outputs/kv_delta_rank/qwen3_8b_musique_d6_c64_t128_seed2028_20260817/summary.json`
- Per-cell results are in `rank_cells.jsonl` under the same directories.
- Probe CLI: `python -m xmodel_kv.cli.probe_kv_delta_rank --help`
