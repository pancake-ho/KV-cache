# Proposed KV-cache research

Our original approach belongs here and must never be implemented inside baseline/.

No proposed model, mapper, KV injection mechanism, training result or accuracy/speed
improvement is implemented or claimed yet.

When implementing approved research, split modules by real responsibility:
- data: calibration/held-out, tokenizer alignment, prompt identity
- models: source/target model loading, dtype, attention/cache layouts
- mapping: mapper architectures, fitting and artifacts
- transfer: RoPE-aware K/V transformation and target cache injection
- train: fitting, checkpoint/resume
- eval: generation accuracy, NLL/PPL and native target baselines
- benchmark: prefill/decode/TTFT/E2E, throughput and memory accounting
- common: configs, seeds, experiment provenance

Add only modules that correspond to implemented functionality. Server-specific
Slurm, Docker, paths, CUDA device selection and job logs belong under run/.
