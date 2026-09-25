# Llama-3.1 first-half packet confirmatory protocol

Status: frozen before construction of the confirmation set.

## Selected operating point and post-hoc disclosure

The `first_16 + INT4` packet was selected after inspecting the original
256-case layer-localization diagnostic.  On that set, the N=512 writer reached
`.6434` F1 with first16, `.3487` with last16, and `.8286` with all layers.
First16-minus-last16 F1 was `+.2947`, 95% CI `[+.2242,+.3661]`; first16 still
lost `.1853` to all layers.  This evidence is exploratory and cannot be used
as confirmation.

The fixed method for this protocol is therefore the already trained,
unconstrained N=512 Llama writer with exactly four slots, rank 4, and first 16
of 32 layers transmitted at INT4.  No new training or layer search is allowed.

## New frozen dataset

Construct exactly 256 MuSiQue-train handoff cases with seed 2053, eight answer
candidates, and minimum 24,000 Agent-A document characters.  Explicitly
exclude every ID in both:

- `train512_superset_seed2041_exclude_test256.jsonl`;
- `test256_traincases_seed2031_exclude_train128.jsonl`.

Use MuSiQue train for both target cases and decoy pools.  Assert 256 unique
IDs, zero overlap with all 512 training IDs, and zero overlap with all 256
previous diagnostic IDs.  Freeze the JSONL and its builder summary before any
model execution; do not resample based on results.

This is an ID-disjoint same-source-distribution confirmation.  It is not a
new original dataset split and will not be described as cross-dataset proof.

## Arms

Run the final unconstrained N=512 checkpoint with correct source assignment
under all three fixed INT4 layer patterns:

1. `all` (135,168 B fake-quant payload accounting);
2. `first_16` (67,584 B accounting; selected packet);
3. `last_16` (67,584 B depth-localization control).

Run one additional `first_16 + INT4` arm with circular shift-1 source
assignment.  Every arm uses the same questions, decoding settings, system
prefixes, and checkpoint.  The row-matched no-summary output is also retained.

## Gates and reporting

The selected packet passes only if paired bootstrap F1 95% confidence
intervals have positive lower bounds for all three contrasts:

- correct first16 versus no-summary (net utility);
- correct first16 versus shifted first16 (source causality);
- correct first16 versus correct last16 (depth localization replication).

Report exact McNemar tests, F1, bridge readout, wins/ties/losses, all-layer
quality gap, and payloads regardless of outcome.  No result may change the
layer count, bit width, slots, writer, or checkpoint on this set.

If all gates pass, the packet may advance to a separately preregistered new-
dataset replication.  If any gate fails, first16 remains exploratory and this
confirmation set is closed to further layer selection.
