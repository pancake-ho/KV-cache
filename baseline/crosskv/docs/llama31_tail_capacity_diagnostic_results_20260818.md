# Generated Tail-KV capacity diagnostic result

This reports the frozen post-hoc protocol in
`llama31_tail_capacity_diagnostic_preregister_20260818.md`.  It is a mechanism
diagnostic on the already opened MultifieldQA-150 set, not external
confirmation of a learned method.

## Primary question-conditioned result

All runs use canonical real INT4 and first16 layers.  Model EOS makes realized
tail lengths smaller than the generation caps.

| generation cap | mean realized KV positions | mean framed bytes | mean F1 |
|---:|---:|---:|---:|
| 4 | 3.73 | 63,136 | `.15606` |
| 8 | 6.46 | 109,206 | `.18679` |
| 16 | 10.29 | 173,862 | `.18878` |

Paired differences:

- cap 8 minus cap 4: `+.03073`, CI `[+.00665,+.05507]`;
- cap 16 minus cap 8: `+.00199`, CI `[-.02119,+.02576]`;
- cap 16 minus cap 4: `+.03272`, CI `[+.00556,+.06082]`.

All three frozen capacity decisions pass:

- global cap16-minus-cap4 is significantly positive;
- on the 80 cases with 9+-word gold answers, the delta is `+.06022`, CI
  `[+.01521,+.10808]`;
- long-answer minus 1--4-word-answer improvement is `+.05624`, CI
  `[+.01041,+.10494]`.

On 1--4-word answers the cap16-minus-cap4 delta is only `+.00397`, CI
`[-.00160,+.01351]`.  Thus the gain is concentrated where the learned
four-slot capsule previously lost its source-causal contrast.  The useful
frontier bends near cap 8; doubling from 8 to 16 adds 59.2% mean packet bytes
without a measurable aggregate improvement.

## Secondary state-readout result

State readout remains at the floor: cap 4/8/16 F1 is
`.01797/.02237/.02110`.  Cap16-minus-cap4 is `+.00312`, CI
`[-.00465,+.01121]`; none of its three capacity gates pass.  The capacity
effect is therefore specific to a receiver that is given the actual question,
not generic lexical recovery from the packet alone.

## Interpretation and method decision

This oracle mixes answer generation/truncation with KV-position capacity, so
it does not prove that an eight-slot pre-answer capsule will learn the same
gain.  It does establish a sufficiently strong mechanism gate to build one
controlled candidate.

The candidate should not replace four slots with a fixed eight.  It should be
a causally nested `4 core + 4 refinement` packet:

1. freeze the existing four core slots and all parameters that write them;
2. append four refinement slots, whose causal positions may attend to the
   source and core but cannot alter earlier core K/V;
3. prove that the core packet is bitwise identical whether emitted alone or as
   the prefix of the eight-slot state;
4. train only refinement parameters and a complexity gate;
5. send 4 slots for simple cases and append the refinement packet for complex
   cases.

This differs from the failed shared-slot layer-budget training: the state
representation itself is prefix-nested, and the low-budget operating point is
frozen by causal construction.  Qasper and NarrativeQA remain untouched and
are candidates for later confirmation; neither may be inspected or used until
the development protocol is frozen.

Machine-readable analyses:

- `llama31_r16_orbitdistill_full1000_20260818/tail_capacity_question_conditioned_analysis.json`;
- `llama31_r16_orbitdistill_full1000_20260818/tail_capacity_state_readout_analysis.json`.
