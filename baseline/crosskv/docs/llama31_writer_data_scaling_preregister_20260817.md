# Llama-3.1 writer data-scaling protocol

Status: frozen before construction of the 512-case training superset.

## Question

Does the quality gap between the four-slot capsule and KVPacket primarily come
from training a source-state writer on only 128 examples?  Test diversity at a
fixed optimization and representation budget, without searching slots, rank,
layers, learning rate, or checkpoints.

## Training-set construction

- Start with the exact 128 MuSiQue training cases used by the current Llama
  writer.
- Sample 384 additional eligible cases from the MuSiQue train split with seed
  2041 and append them, producing a 512-case strict superset.
- Explicitly exclude every ID in
  `test256_traincases_seed2031_exclude_train128.jsonl` from the 384 new cases.
- Retain the original 128 cases verbatim, including their decoy documents.
- Assert 512 unique IDs, all original 128 present, and zero overlap with the
  frozen 256-case test set before training.

## Fixed model and optimizer

Copy the existing Llama replication exactly:

- `meta-llama/Llama-3.1-8B-Instruct`, frozen BF16 base model;
- four learned state-emission slots;
- rank-4, scale-1 sender-only writer;
- 1,600 AdamW updates, learning rate `5e-4`, weight decay `.01`;
- stage weight 1, bridge weight 2, no teacher KL;
- resident detached source caches on CPU;
- final checkpoint only, no early stopping.

Keeping 1,600 updates fixed means this is explicitly a diversity-at-fixed-
compute experiment, not an equal-epoch scaling law.

## Gates and no-search policy

1. Evaluate the final checkpoint on the same MuSiQue dev indices 64--95 using
   all layers and BF16.  The old N=128 writer has stage F1 `.44375`.  Continue
   only if N=512 improves it by at least `.03` without reducing direct bridge
   F1 by more than `.05`.
2. Before the N=512 result is known, evaluate the frozen N=128 checkpoint on
   all 256 ID-disjoint test cases at the already fixed Llama operating point:
   last 16 layers, INT4.  Run both correct-source and circular shift-1 source
   assignments for each writer.  If gate 1 passes, evaluate N=512 identically
   and use paired EM/F1 statistics; correct-source quality is primary and shift
   is the causal control.
3. Authorize another Hotpot E290 diagnostic only if N=512 versus N=128 has a
   positive paired F1 95% CI on test256.  Hotpot has already been inspected and
   cannot serve as a new confirmation set.

No N=512 result may be used to change slots, writer rank, layer pattern,
quantization, or training steps under this protocol.  Failure is evidence that
data diversity alone does not close the quality gap.

## Orthogonal real-codec verification

While N=512 trains, replace fake INT4 on the already frozen N=128/test256 arm
with an actual signed-nibble packet using FP16 per-token/per-head scales, layer
indices, framing, and CRC.  This is an implementation check, not a model
selection.  The real-codec gate passes if its paired F1 delta versus fake INT4
has a 95% CI containing zero and absolute point delta below `.02`.  Report exact
wire bytes and measured pack/unpack costs regardless of outcome.

## Frozen post-result mechanism diagnostic

The fixed `last_16 + INT4` N=512 arm failed the primary scaling gate: on the
256-case test it reached `.3487` F1 versus `.5505` for N=128, with paired delta
`-.2019` and bootstrap 95% CI `[-.2661,-.1372]`.  Consequently the protocol
does **not** authorize another Hotpot run, and the following diagnostic cannot
be used to replace the failed operating point or select a better headline
number.

Before running any additional N=512 evaluation, freeze one full-factorial
mechanism diagnostic on the same 256 cases.  Evaluate both the N=128 and N=512
final checkpoints with correct source assignment under every cell of:

- transmitted layers: `all` and `last_16`;
- representation: BF16 and INT4 fake quantization.

Run all four cells for both writers and report them without selection.  This
factorial distinguishes three explanations:

1. N=512 loses even with all-layer BF16: fixed-update data scaling harmed
   held-out generalization rather than merely changing the packet location;
2. N=512 recovers with all layers but not `last_16`: useful state moved outside
   the late-layer packet;
3. N=512 recovers in BF16 but not INT4 at matched layers: the larger-data
   writer became quantization-sensitive.

Primary diagnostic contrasts are N=512 versus N=128 within each of the four
cells.  Secondary within-writer contrasts are `last_16` versus `all` at fixed
precision and INT4 versus BF16 at fixed layer pattern.  Use paired exact and
F1 bootstrap intervals.  Do not change slots, rank, checkpoint, generation
settings, or dataset after inspecting a cell.  Any new method motivated by
this result must be trained and confirmed on a fresh split.
