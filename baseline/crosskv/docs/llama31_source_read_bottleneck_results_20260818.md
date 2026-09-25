# Llama-3.1 source-read-16 bottleneck development results

## Decision

The source-read bottleneck preserves full-depth credit assignment and learns to
recover most of the full packet after deep source access is removed.  However,
it does not improve the runtime first16 packet over the compute-matched ordinary
all-only continuation.  The compact-concentration gate fails; full-state safety
and source-causality pass.

This is an informative alignment failure rather than a recurrence of the hard
stop-gradient failure.  The four-slot hidden state after source-reading layer 15
is first projected into emitted K/V at layer 16 (the 17th packet layer), just
outside the runtime first16 packet.  The structural bottleneck and serialized
packet boundary are off by one Transformer state transition.

## Absolute confirm128 results

All entries use four slots and INT4.

| checkpoint / packet | stage EM | stage F1 | bridge F1 | payload |
|---|---:|---:|---:|---:|
| parent first16 | .6328 | .7336 | .2320 | 67,584 B |
| ordinary continued first16 control | **.7734** | **.8320** | .2027 | 67,584 B |
| source-read-16 first16 | .7188 | .7826 | .2594 | 67,584 B |
| source-read-16 shift-1 first16 | .1641 | .2917 | .0260 | 67,584 B |
| parent all | .7734 | .8188 | .8266 | 135,168 B |
| parent + untrained read-16 intervention all | .6484 | .7414 | .2826 | 135,168 B |
| ordinary continued all control | **.8359** | **.8737** | .9370 | 135,168 B |
| source-read-16 all | .8281 | .8633 | .7281 | 135,168 B |
| source-read-16 shift-1 all | .2109 | .2974 | .0367 | 135,168 B |

Training with the bottleneck has mean stage CE `.1688`, bridge CE `.2763`, and
combined loss `.7214`, versus `.0744/.0321` for the unrestricted compute
control.  The lexical objective remains much harder when all deep information
must cross four hidden positions at layer 16.

## Frozen gates

1. **Compact concentration fails.** Candidate minus control first16 stage F1 is
   `-.0495`, ordinary 95% CI `[-.1094,+.0091]`, bridge-cluster CI
   `[-.1174,+.0133]`.  The frozen requirement was at least `+.03` with positive
   lower bounds.
2. **Full-state safety passes.** Candidate minus control all stage F1 is
   `-.0104`, ordinary CI `[-.0625,+.0417]`, cluster CI
   `[-.0605,+.0362]`.  The point estimate is above `-.03` and both lower bounds
   are above the frozen `-.08` safety margin.
3. **Source causality passes strongly.** Correct minus shift first16 stage F1 is
   `+.4909`, ordinary CI `[+.4049,+.5768]`, cluster CI
   `[+.3949,+.5901]`.  EM delta is `+.5547`, McNemar
   `p=1.01e-18`.

The method is rejected because all three gates were required.

## Zero-training intervention

Applying `source_read_layers=16` to the unchanged parent proves the
intervention's locality:

- all 128 first16 stage outputs and metrics match the unrestricted parent, with
  paired F1 CI exactly `[0,0]`;
- all-layer stage F1 immediately loses `-.0773`, CI
  `[-.1289,-.0292]`;
- bridge F1 falls from `.8266` to `.2826`.

After 3,200 updates, the candidate all-layer stage F1 recovers from `.7414` to
`.8633`, and bridge F1 from `.2826` to `.7281`.  Therefore the continuous
hidden bottleneck is trainable and does transmit sample-specific information;
it simply does not place that information into the exact emitted layers sent by
first16.

## Mechanistic correction

For decoder layer index `l`, emitted `K_l,V_l` are projections of the hidden
state entering that layer, `h_l`.  Attention to the source in block `l` produces
`h_{l+1}`.  With `source_read_layers=16`, the final source-reading block is 15,
so its bottleneck output `h_16` first appears in `K_16,V_16`, outside packet
layers 0--15.

The unique representation-level follow-up is therefore an
**emission-aligned source-read-15 bottleneck**: blocks 0--14 read the source;
their output `h_15` is projected into `K_15,V_15`, the last transmitted layer of
first16; blocks 15--31 may only transform capsule state.  This is a derived
state/KV alignment correction, not a scan over arbitrary boundaries, and must
be tested on a new frozen set rather than this exhausted development set.

## Artifacts

- Protocol: `docs/llama31_source_read_bottleneck_preregister_20260818.md`.
- Candidate:
  `outputs/amortized_semantic_kv/llama31_increment256_source_read16_cont3200_confirm128_20260818/`.
- Shift control:
  `outputs/amortized_semantic_kv/llama31_increment256_source_read16_shift1_confirm128_20260818/`.
- Zero-training intervention:
  `outputs/amortized_semantic_kv/llama31_parent_source_read16_intervention_confirm128_20260818/`.
- Paired analyses:
  `outputs/amortized_semantic_kv/llama31_source_read16_analysis_20260818/`.
