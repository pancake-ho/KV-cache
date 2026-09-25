# Llama-3.1 multi-budget writer results

## Decision

Both frozen stochastic multi-budget writers fail the preregistered quality
gates.  The first uses the same stage-plus-lexical objective at both budgets;
the one permitted mechanism follow-up removes lexical supervision from
first16 updates and increases it on all-layer updates.  Both remain strongly
source-causal, but both show negative transfer relative to the all-only
control.  INT4-aware training and an unconditioned lexical loss are therefore
not sufficient explanations.

## Frozen development audit

The 128-case set excludes exactly 1,024 IDs used by train512, old test256, and
fresh confirm256.  It contains 128 unique cases and 80 unique bridge-answer
clusters.

- ordered ID SHA256:
  `c5387a996a2aa277a2742361b0d63c2e4d90b5bdccbfc97fd3c6483a92e18d45`;
- file SHA256:
  `1d0c6e99abd163ce9e1223ecbc43fddf1a9f392e15ea14f8b1221d441c68785d`.

The realized multi-budget schedule was first16 801 updates and all-layer 799
updates.  Both candidates used the final 1,600-step checkpoint.

## Absolute results

| writer | first16 EM | first16 F1 | all EM | all F1 |
|---|---:|---:|---:|---:|
| historical all/BF16-trained N512 | .570 | .684 | **.797** | **.866** |
| INT4-aware all-only control | **.586** | **.691** | .789 | .841 |
| first16/all multi-budget | .469 | .591 | .586 | .717 |
| multi-budget shift-1 | .125 | .257 | .070 | .218 |
| functional/lexical multi-budget | .445 | .553 | .648 | .728 |
| functional/lexical shift-1 | .188 | .301 | .180 | .283 |
| no-summary | .172 | .295 | .172 | .295 |

The INT4-aware all-only control is indistinguishable from the historical
writer: first16 F1 delta `+.007`, CI `[-.074,+.088]`; all-layer delta `-.024`,
CI `[-.089,+.040]`.  Quantization-aware STE therefore does not explain the
multi-budget collapse.

## Preregistered gates

Multi-budget versus INT4 all-only:

- first16 F1 delta `-.100`, ordinary bootstrap CI `[-.187,-.016]`, bridge-cluster
  CI `[-.198,+.003]`; the required `+.03` improvement fails;
- all-layer F1 delta `-.124`, ordinary CI `[-.203,-.046]`, cluster CI
  `[-.207,-.040]`; the `-.05` non-inferiority gate fails because both lower
  bounds are far below the margin.

The causal gate passes: correct first16 versus shift-1 has F1 delta `+.334`,
ordinary CI `[+.247,+.420]` and cluster CI `[+.249,+.420]`; exact accuracy is
`.469` versus `.125`, McNemar `p=1.02e-9`.

## Mechanism diagnosis

At step 1 both trainings are identical.  By step 1,600, all-only mean
stage/bridge CE is `.534/.181`, while multi-budget is `.659/1.114`.  Prior depth
ablations show why: first16 carries task-usable functional state but is a poor
generic lexical readout path.  Requiring first16 updates to minimize bridge CE
forces one representation to satisfy an incompatible readout objective and
also degrades its all-layer state.

This first negative result rules out naive stochastic layer masking as the
method.  The preregistration permitted exactly one mechanism follow-up:
condition the objective on packet budget, so first16 updates optimize only
downstream task utility while all-layer updates retain a four-times-weighted
lexical bridge objective.  The following section reports that test.  No
further mixture-ratio or layer-pattern scan is permitted on this development
set.

## Frozen functional/lexical follow-up

The follow-up retained the same model, 512 training cases, four slots,
rank-four writer, 1,600 steps, uniform stochastic first16/all sampling and
INT4 STE.  Only the loss routing changed:

- first16 updates: stage weight 1, bridge weight 0;
- all-layer updates: stage weight 1, bridge weight 4.

The realized schedule was first16 801 and all-layer 799 updates.  Its aggregate
training means were stage CE `.825` and bridge CE `.186`; training took 609.4 s.

All three frozen gates are:

- **first16 improvement fails:** versus the all-only control, F1 delta
  `-.1375`, ordinary CI `[-.2273,-.0492]`, bridge-cluster CI
  `[-.2379,-.0394]`; exact accuracy delta is `-.1406`, McNemar `p=.0153`;
- **all-layer non-inferiority fails:** F1 delta `-.1135`, ordinary CI
  `[-.1992,-.0307]`, bridge-cluster CI `[-.2101,-.0177]`; the lower bounds are
  below the frozen `-.05` margin;
- **source-causality passes:** correct versus shift-1 first16 F1 delta
  `+.2523`, ordinary CI `[+.1633,+.3422]`, cluster CI `[+.1609,+.3455]`;
  exact accuracy is `.445` versus `.188`, McNemar `p=1.07e-6`.

The loss routing does not even improve first16 over the naive multi-budget
writer: delta `-.0375`, CI `[-.1250,+.0508]`.  It leaves all-layer essentially
unchanged relative to naive multi-budget (`+.0107`, CI
`[-.0740,+.0950]`).  The bridge behavior nevertheless shows that the routing
was implemented: first16 bridge EM/F1 is `.000/.0076`, while all-layer is
`.680/.855`.

## Revised mechanism boundary

The shared four-slot writer is not merely suffering from an inappropriate
lexical loss at the shallow budget.  Gradients from two receivers with
different depth topologies still compete in the same emitted K/V state even
when their lexical objectives are separated.  A fixed packet cannot be
assumed to be a prefix of a larger packet that is simultaneously optimal.

The next method must change the representation or routing, not tune another
loss weight on this set.  Viable hypotheses include a separately parameterized
early functional base plus optional deep residual, a budget token/router that
changes the emitted state itself, or teacher-to-student cross-layer
reconstruction.  Any such variant needs a new development pool and a new
sealed confirmation set; this budgetdev128 is exhausted for method selection.

Artifacts:

- preregistration: `docs/llama31_multibudget_writer_preregister_20260817.md`;
- historical evaluation:
  `outputs/amortized_semantic_kv/llama31_n512_historical_budgetdev128_first16_all_int4_20260817/`;
- all-only control:
  `outputs/amortized_semantic_kv/llama31_n512_allonly_int4_budgetdev128_fixed1600_20260817/`;
- naive multi-budget:
  `outputs/amortized_semantic_kv/llama31_n512_multibudget_first16_all_int4_budgetdev128_fixed1600_20260817/`;
- shift control:
  `outputs/amortized_semantic_kv/llama31_n512_multibudget_budgetdev128_shift1_first16_all_int4_20260818/`;
- paired analysis:
  `outputs/amortized_semantic_kv/llama31_multibudget_budgetdev128_analysis_20260818/`;
- functional/lexical writer:
  `outputs/amortized_semantic_kv/llama31_n512_functionallexical_first16_all_int4_budgetdev128_fixed1600_20260818/`;
- functional/lexical shift:
  `outputs/amortized_semantic_kv/llama31_n512_functionallexical_budgetdev128_shift1_first16_all_int4_20260818/`;
- functional/lexical paired analyses:
  `outputs/amortized_semantic_kv/llama31_functionallexical_budgetdev128_analysis_20260818/`.
