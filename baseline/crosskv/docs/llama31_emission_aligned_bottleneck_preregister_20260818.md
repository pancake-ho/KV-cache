# Llama-3.1 emission-aligned bottleneck confirmation protocol

Status: frozen before constructing or inspecting the new confirmation set and
before training the source-read-15 candidate.

## Single permitted correction

The source-read-16 development run preserves full-depth quality and strong
source causality but fails compact concentration.  Its bottleneck output
`h_16` is first projected into `K_16,V_16`, the 17th cache layer and therefore
outside a layers-0--15 first16 packet.

The only candidate in this protocol sets `source_read_layers=15`:

- sender blocks 0--14 may attend the complete source cache;
- their final output `h_15` is projected into `K_15,V_15`, the last transmitted
  first16 cache layer;
- sender blocks 15--31 cannot attend source K/V and transform only the causal
  four-slot state;
- hidden states and gradients remain continuous through all 32 blocks.

No receiver objective, loss weight, slot count, rank, quantizer, prompt, or
runtime layer pattern changes.  This is an emission-alignment correction, not
a boundary sweep.  Source-read-14/16/17 or another variant will not be tried on
the new set.

## New frozen confirmation construction

Construct 96 cases from the official MuSiQue answerable train source using
seed 2087 and the established strict two-hop/eight-candidate builder.  Exclude
the union of all 1,536 previously used IDs through these frozen case files:

1. `train512_superset_seed2041_exclude_test256.jsonl` (512);
2. `test256_traincases_seed2031_exclude_train128.jsonl` (256);
3. `confirm256_seed2053_exclude_train512_test256.jsonl` (256);
4. `budgetdev128_seed2061_exclude_used1024.jsonl` (128);
5. `train256_increment_seed2081_exclude_used1152.jsonl` (256);
6. `confirm128_seed2083_exclude_used1408.jsonl` (128).

Require exactly 1,536 unique excluded IDs, at least 96 eligible remaining
rows, 96 unique selected IDs, zero overlap, and record the ordered-ID/file
SHA256 values before loading a model.  Frozen output:
`datasets/musique_handoff/alignment_confirm96_seed2087_exclude_used1536.jsonl`.

This set is confirmation-only.  The candidate continues to train on the
already-open increment256; no selected confirmation text or metric may affect
training.

## Compute-matched training

Initialize from parent SHA256
`b5848635fac01561b8f694aeca79d5d02d340a5c15eeb1e3967d21c90cb0b620`.
Use the same increment256, Llama-3.1-8B, four slots, rank-4 writer, 3,200
updates, seed 2027, LR `5e-4`, weight decay `.01`, all-layer-only stage/bridge
weights `1/2`, and INT4 STE as both prior candidates and the ordinary control.
Evaluate first16/all INT4 on all 96 new cases.

Fixed references:

- ordinary unrestricted control:
  `llama31_increment256_allonly_cont3200_confirm128_20260818/capsule.pt`;
- unaligned source-read-16 candidate:
  `llama31_increment256_source_read16_cont3200_confirm128_20260818/capsule.pt`.

Evaluate both references on the new set without adaptation.  Also run exactly
one circular shift-1 source control for the aligned candidate.

## Confirmation gates

Stage F1 is primary.  Use paired 20,000-replicate ordinary and bridge-answer
cluster bootstrap intervals, seed 2027.  All gates are required:

1. **Alignment benefit:** aligned first16 minus unaligned read-16 first16 is at
   least `+.03`, with ordinary and cluster CI lower bounds above zero.
2. **Compact superiority:** aligned first16 minus unrestricted compute control
   is at least `+.02`, with ordinary and cluster CI lower bounds above zero.
3. **Full-state safety:** aligned all minus unrestricted control is at least
   `-.03`, with ordinary and cluster CI lower bounds above `-.08`.
4. **Source causality:** aligned correct minus shift-1 first16 is at least
   `+.10`, with positive ordinary and cluster CI lower bounds.

If alignment benefit passes but compact superiority fails, the off-by-one
mechanism is supported but the method is not a better compact optimizer.  If
all safety fails, the boundary state is useful to first16 but too restrictive
for the full packet.  No gate can be rescued by the old confirm128 or a
different packet depth.

## Frozen construction audit

Construction completed before loading a model:

- 1,639 total eligible source rows;
- exactly 1,536 unique excluded historical IDs;
- 103 eligible rows remained after exclusion;
- 96 selected rows and 96 unique IDs, with zero overlap against the excluded
  union;
- mean dossier length 24,268.81 characters, minimum 24,005;
- ordered-ID SHA256
  `a466ebac4e54428b3de4f19e31d4801a4bc65ebd8222763ccff913b54785f876`;
- file SHA256
  `faf31d719b27935cac90fbf2b817d2d2dffad07d26b1ee67df21a5586b4cdb3f`.

No case text, model output, or metric was inspected during this audit.
