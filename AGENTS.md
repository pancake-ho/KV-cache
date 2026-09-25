# AGENTS.md

## Repository Purpose

This repository is for undergraduate research on KV-cache optimization for LLM inference.

Current and future experiments may include:
- KV-cache reuse
- KV-cache transfer across LoRA adapters or model variants
- cache compression / eviction / offloading
- cross-model KV mapping
- multi-LoRA KV sharing
- serving and inference efficiency experiments

Experimental branches may contain different research assumptions.
Do not assume that settings from one branch apply to another branch.

Examples of existing experimental branches include:
- `exp/lora-test`
- `exp/sparkv-test`

Always review the code and configuration in the PR's actual branch.

---

## General Review Philosophy

Prioritize correctness, reproducibility, and experimental validity over style.

Do not report stylistic preferences unless they can cause:
- incorrect experimental results,
- runtime failures,
- misleading metrics,
- excessive memory use,
- silent CPU fallback,
- incompatibility with model/framework versions,
- or irreproducible experiments.

Avoid speculative comments.

A review finding should identify:
1. the concrete code location,
2. the failure mode,
3. why it matters,
4. and, when possible, a minimal way to verify or fix it.

Do not recommend large refactors unless they are required to fix an actual problem.

---

## Research Assumptions

Do NOT assume any of the following unless they are explicitly defined by the current branch, config, or experiment:
- target LLM,
- model family,
- model size,
- tokenizer,
- attention architecture,
- number of KV heads,
- head dimension,
- precision,
- quantization,
- inference framework,
- GPU type,
- compression method,
- eviction policy,
- offloading policy,
- serving engine,
- dataset,
- or evaluation baseline.

If a PR relies on one of these assumptions without making it explicit, flag it.

Do not transfer assumptions from unrelated experiments or older branches.

---

## KV Cache Correctness

Pay particular attention to KV tensor semantics.

Verify:
- tensor shape,
- batch dimension,
- sequence/token dimension,
- layer dimension,
- KV-head dimension,
- head dimension,
- dtype,
- device,
- contiguity assumptions,
- and framework-specific cache layout.

Do not assume a universal KV layout.
Verify the layout used by the actual model/framework.

Flag:
- accidental K/V transposition,
- incorrect reshape/view,
- head-count mismatch,
- head-dimension mismatch,
- broadcasting that changes semantics,
- sequence offset errors,
- cache truncation at the wrong dimension,
- stale cache reuse,
- failure to free obsolete caches,
- and copies that unintentionally duplicate large cache tensors.

Check whether `.view()` is safe for non-contiguous tensors.
Prefer explicit validation when layout assumptions matter.

---

## Prefill and Decode

Keep prefill and decode measurements conceptually separate.

Flag code that:
- reports prefill-only improvement as end-to-end latency improvement,
- mixes prefill and decode timing,
- rebuilds a cache during a supposedly cache-reuse path,
- performs hidden recomputation that invalidates claimed reuse,
- or evaluates different generated lengths across baselines.

When comparing methods, verify that the following are controlled:
- input prompt,
- prompt length,
- generated length,
- batch size,
- model,
- dtype,
- hardware,
- software versions,
- decoding parameters,
- and random seed where relevant.

---

## Cross-Model KV Transfer

For cross-model KV mapping experiments, explicitly verify the complete mapping pipeline.

Check:
- source and target model identities,
- tokenizer compatibility,
- token alignment,
- source and target layer counts,
- KV head counts,
- per-head dimensions,
- source-layer selection,
- target-layer mapping,
- K and V mapping separately,
- dtype/device consistency,
- calibration data separation,
- and whether mapping artifacts correspond to the correct model pair.

For implementations inspired by:

`Cross-Model KV Cache Transfer in LLM Families:
A Closed-Form Linear Mapping for Prefill Reuse`

pay special attention to:
- per-head mapping,
- top-k source-layer selection,
- ridge-regression fitting,
- calibration-set construction,
- position-independent mapping,
- RoPE removal from source keys before fitting/mapping where required,
- correct application of target-side positional encoding afterward,
- and whether the mapped cache is actually decode-ready.

Do not accept reconstruction quality alone as proof of successful transfer.
Downstream continuation quality must also be evaluated.

Flag cases where:
- R² / cosine similarity is reported without downstream quality,
- source and target caches are accidentally generated from different token sequences,
- calibration examples leak into evaluation,
- positional information is mishandled,
- or mapper latency is compared against an unfair re-prefill baseline.

---

## CacheBridge-Style Experiments

When reviewing CacheBridge-inspired changes, verify that optimizations do not silently change the mathematical mapping.

Pay attention to:
- source-head / target-head correspondence,
- selected source-layer support,
- attention-aware weighting,
- sufficient-statistics construction,
- mapper storage,
- mapper application cost,
- and numerical equivalence between optimized and reference implementations.

Kernel or fused implementations should be checked against a small reference implementation before performance claims are trusted.

---

## Multi-LoRA / LRAgent Experiments

For multi-LoRA KV sharing, distinguish clearly between:
- base-model contribution,
- LoRA-specific contribution,
- full KV cache,
- low-rank cache,
- and reconstructed adapter contribution.

Verify that cache sharing does not accidentally remove adapter-specific information.

When implementing LRAgent-style methods, review:
- BaseShared logic,
- BaseLRShared logic,
- LoRA down-projection and up-projection dimensions,
- rank dimension,
- adapter identity,
- cache ownership,
- reconstruction timing,
- and whether multiple agents incorrectly share mutable state.

Do not assume Shared-A LoRA unless the experiment explicitly configures it.

---

## Memory Accounting

Theoretical KV size and actual GPU memory usage are different metrics.

Do not treat:
- tensor byte count,
- PyTorch allocated memory,
- PyTorch reserved memory,
- and GPU process memory

as interchangeable.

For theoretical cache memory, validate calculations using the actual:
- number of layers,
- number of KV heads,
- head dimension,
- sequence length,
- batch size,
- K/V factor,
- and dtype bytes.

For runtime GPU measurements, prefer recording both:
- peak allocated memory,
- peak reserved memory.

Ensure peak-memory statistics are reset before each independent measurement.

Flag results that compare memory numbers measured with inconsistent methods.

---

## Performance Measurement

GPU latency measurements must use synchronization where required.

For CUDA timing, verify appropriate use of:
- warm-up iterations,
- `torch.cuda.synchronize()` or CUDA events,
- repeated measurements,
- and exclusion of one-time initialization unless intentionally measured.

Do not trust a single timing sample.

Separate when possible:
- mapper construction time,
- cache transfer time,
- prefill latency,
- decode latency,
- TTFT,
- end-to-end latency,
- and throughput.

If host-to-device or device-to-device transfer is part of the method, make sure it is included or explicitly excluded.

---

## Accuracy and Quality

Any method that modifies, compresses, transfers, or approximates KV cache must report output quality together with efficiency.

Depending on the experiment, useful checks include:
- downstream task accuracy,
- perplexity / NLL,
- exact-match or task score,
- generation consistency,
- and attention/output similarity.

Do not conclude that a method preserves model quality from KV cosine similarity alone.

---

## Numerical Safety

Check for:
- NaN,
- Inf,
- unstable inversions,
- ill-conditioned regression,
- dtype overflow/underflow,
- accidental FP32/FP16/BF16 conversions,
- and mismatched devices.

Closed-form regression implementations should explicitly handle regularization and matrix conditioning.

---

## GPU and Device Handling

Research experiments are expected to use the intended GPU when GPU execution is required.

Flag silent fallback to CPU.

If CUDA is required and unavailable, prefer an explicit failure over silently continuing with a CPU experiment whose timing is not comparable.

Check all tensors participating in the same operation for consistent device and dtype.

---

## Reproducibility

Experimental code should make important settings explicit.

Prefer logging:
- branch / commit,
- dirty state when practical,
- model identifier,
- tokenizer,
- dtype,
- framework versions,
- CUDA / PyTorch version,
- GPU,
- dataset,
- seed,
- prompt length,
- generated length,
- batch size,
- cache method,
- and method-specific parameters.

Flag hard-coded paths, hidden defaults, and undeclared environment assumptions that prevent reproduction.

---

## Tests

For correctness changes, prefer small deterministic tests before expensive experiments.

At minimum, where applicable, verify:
- import / syntax,
- tensor shapes,
- finite values,
- CPU/GPU device behavior,
- deterministic seed behavior,
- cache save/load,
- mapping artifact save/load,
- edge sequence lengths,
- and baseline-equivalent behavior when the new method is disabled.

Do not claim that a test passed unless the test was actually executed.

---

## Review Output

Focus review comments on actionable bugs or experimental-validity risks.

Use severity roughly as:
- Critical: invalidates results, corrupts cache semantics, or makes experiments unusable.
- High: likely runtime error, incorrect model output, unfair comparison, or severe memory/timing error.
- Medium: reproducibility issue, missing validation, or edge-case correctness problem.
- Low: minor maintainability issue with a realistic future failure mode.

Avoid purely cosmetic review comments.
