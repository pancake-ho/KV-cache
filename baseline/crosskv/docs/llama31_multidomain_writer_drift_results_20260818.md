# Multi-domain writer checkpoint-drift localization results

Date: 2026-08-18.  No training or task generation was run.  The analysis uses
the parent and rejected multi-domain candidate named in the frozen protocol
`llama31_multidomain_writer_drift_preregister_20260818.md`.

## Decision

The checkpoint change is **distributed** under the frozen thresholds.  Simple
embedding freezing or freezing one contiguous eight-block region is not
supported as a sufficient next intervention.

## Results

- The eight slot embeddings contain only `3.45%` of total squared parameter
  drift; the 31 layerwise writer blocks contain `96.55%`.
- Embeddings themselves move by `15.11%` of their parent L2 norm and retain
  cosine `.98869`, so their change is real but not dominant.
- Fixed-probe functional-drift energy is distributed `16.52% / 43.11% /
  40.37%` over early blocks 0--9, middle blocks 10--20, and deep blocks 21--30.
- The eight individually largest blocks are 11, 27, 18, 23, 30, 20, 29, and
  24.  Together they contain only `43.47%` of functional drift, below the
  frozen 60% localization threshold.
- The most concentrated contiguous eight-block window is 23--30 and contains
  only `34.01%`, also below threshold.
- Block 11 is the largest individual contributor at `10.36%`; no single late
  region explains the update.

## Interpretation boundary

The writer changed predominantly in middle and deep blocks, but the largest
changes are interleaved across depth.  This agrees with the earlier factorial
evidence that useful capsule state is depth indexed and argues against treating
the writer as one semantic embedding followed by an expendable late readout.

The random-probe analysis is structural rather than causal: it cannot identify
which exact changes cause the synthetic final gain or MuSiQue bridge loss.  It
does rule out the narrow explanation that one obvious checkpoint module absorbed
almost all of the update.  The evidence therefore favors an explicit immutable
semantic path plus a separately parameterized decision residual over another
whole-writer continuation with soft replay.

## Artifact

`outputs/amortized_semantic_kv/llama31_ce8_multidomain_decision_cont1600_train568_dev56_20260818/checkpoint_drift_localization.json`
