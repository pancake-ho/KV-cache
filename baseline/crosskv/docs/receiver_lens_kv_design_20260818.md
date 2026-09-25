# Receiver-Lens KV: structural successor after distributed writer forgetting

Status: the fixed four-embedding retained-base feasibility test is complete
and rejected.  See `llama31_receiver_lens4_feasibility_results_20260818.md`.

## Mechanism prototype status

The cache plumbing is implemented in `src/xmodel_kv/receiver_lens.py` and is
covered by CPU tests.  The tests establish that a plain lens matches a native
suffix at the same positions, the transported prefix remains bitwise unchanged,
gradients stop exactly at the semantic-cache boundary, receiver-local writer
hooks are removed after materialization, and lens Keys preserve their unrotated
content when relocated after memory eviction.  The behavioral feasibility run
also preserves the frozen CE8 parameter hash exactly.

Lens4 improves the development final-F1 point estimate by `+.0893` over both
base and hard4 and has strongly positive correct-minus-shift intervals.  It is
therefore a real source-conditioned readout.  But the improvement CI crosses
zero and MuSiQue bridge F1 degrades by `-.07837`, CI
`[-.14593,-.02083]`.  The retained-base gate fails, so consolidate-and-evict is
not authorized for this checkpoint.

## Evidence that motivates the change

The current all32 CE8 packet already carries strong reusable bridge semantics
on MuSiQue, but whole-writer multi-domain continuation improves neither final
F1 significantly nor semantic transfer safely.  Its bridge channel collapses,
and checkpoint drift is distributed across the layerwise writer.  Freezing only
the slot embeddings, one late boundary, or one contiguous writer region is
therefore not an evidence-based repair.

The next architecture should make semantic immutability exact rather than a
soft regularizer.  It should also avoid repeating two prior negative methods:

- a sender-side layer-15 boundary residual, which already gave only a small,
  unresolved gain under the production codec;
- a prompt-to-soft-slot compiler initialized from a generic constant suffix,
  which was Pareto dominated by 12--28 hard policy tokens on the earlier
  single-policy benchmark.

## Proposed factorization

Let `M_A(D)` be a frozen, source-conditioned semantic KV packet written by
Agent A.  Agent B keeps its system prompt, task description, question, and
candidate table as ordinary receiver-prefix text `P_B`; this proposal does not
try to compress that prompt.

After relocating and attaching `M_A(D)` to `P_B`, Agent B materializes a small
set of local lens slots:

\[
R_B = G_{\phi_B}(P_B, M_A(D)).
\]

`G` executes through the frozen receiver transformer, so each lens position can
attend jointly to the real downstream prompt and immutable semantic memory.
Only global lens embeddings and, if necessary, receiver-local low-rank writer
blocks are trained.  `M_A` remains bitwise fixed and is never updated by the
new task loss.

The initial safe path decodes with `P_B + M_A + R_B`.  A stronger systems path
uses `R_B` as a task-conditioned consolidation tail and evicts `M_A` before
long decoding, after relocating the lens cache to a contiguous RoPE frame.  The
latter is a separate gate: successful first-token readout does not imply that
the base memory can be removed for an entire continuation.

## Why this is distinct

The transmitted object is task-independent semantic state.  Task adaptation is
receiver-local, amortized across requests, and conditioned by the actual
downstream prompt.  A new agent can therefore apply a different lens to the
same packet without rewriting or retransmitting the source memory.  Unlike the
failed policy compiler, the receiver prompt remains explicit hard text; the
learned slots only read compact memory in that prompt's context.

This factorization directly targets the observed failure mode:

| component | changes across source documents | changes across receiver tasks | trainable in adaptation |
|---|---:|---:|---:|
| frozen semantic packet `M_A` | yes | no | no |
| receiver prompt `P_B` | yes | yes | no |
| local lens state `R_B` | yes | yes | yes |

## Minimum future falsification sequence

When experimentation resumes, use the rejected multi-domain split only as
development data and preserve the existing sealed sets.

1. **Readout feasibility:** freeze CE8 bitwise and train exactly four local lens
   slots on the existing 568-case mixture.  Compare against frozen CE8 with no
   lens and against four matched hard delimiter/readout tokens.
2. **Causality:** correct semantic packet must beat shift-1 and zero packets for
   both bridge and final routes.  Prompt-shuffle must change final behavior in
   the corresponding task direction.
3. **Retention:** MuSiQue bridge and final performance must remain within the
   existing safety margins because the base is exact; a gain obtained by the
   lens suppressing the base is not acceptable.
4. **Consolidate-and-evict:** only after the retained-base arm passes, compare
   full-continuation decoding with and without `M_A` after lens materialization.
5. **Rate accounting:** local adapter parameters are amortized model state, lens
   positions add compute but no network payload, and the transmitted packet
   remains 270,426 bytes.  Report both prefill and decode-cache costs.

Failure at step 1 has now rejected the fixed-lens version.  It must not trigger
a slot, rank, or learning-rate sweep.  A receiver-local writer is authorized
only by a separately frozen diagnostic showing that fixed embeddings are
under-capacity or require incompatible task-conditioned directions.

## Remaining scope limits

This design does not by itself solve cross-model KV transfer: `M_A` is still in
the sender model's native K/V basis.  It also does not yet beat plaintext on
wire size or prove serving gains.  Its purpose is narrower and testable: keep a
compact semantic state immutable while allowing different downstream agents to
derive task-specific continuation state without corrupting that memory.
