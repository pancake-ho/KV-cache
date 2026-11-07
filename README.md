# KV-cache 졸업연구 — Research repository

Track B: same-base Multi-LoRA representation analysis, same-family different-scale
Cross-Model KV analysis, and future original method implementation.

## Ownership

| Folder | Responsibility |
|---|---|
| baseline/crosskv | External CrossKV implementation, including 261007 Figure 2 experiments |
| baseline/lora_exp | Existing Multi-LoRA Q/K/V characterization |
| baseline/sparkv | SparKV reproduction |
| baseline/GP_semcache | Existing Git submodule; pinned and untouched |
| proposed | Future original methods, independent of all baseline implementations |
| run/seraph | Seraph Slurm submission and runtime |
| run/lab | Lab Docker submission and runtime |
| run/common | Shared launch checks, runtime manifest and result export |
| docs | Design, execution contracts and experiment inventory |
| results | Generated experiment artifacts (not committed) |

The 261007 Figure 2 uses single-source matched-head OLS and reports R2.
It is NOT a production Ridge mapper, KV transfer validation, or prefill speedup.

## Read-only checks

~~~bash
python -m unittest discover -s run/tests -v
python run/common/catalog.py --check
find run -name '*.sh' -exec bash -n {} \;
~~~

## Portable CrossKV experiment launchers

~~~bash
# Seraph (moana-master): single Slurm job, 1 allocated GPU
bash run/seraph/baseline/crosskv_figure2.sh --mode smoke \
  --dataset /data/datasets/YOUR_FILE.parquet --dry-run

# Lab-server host (already running Docker container)
bash run/lab/baseline/crosskv_figure2.sh --mode smoke \
  --container uav --container-repo /workspace/kv_cache \
  --dataset /workspace/datasets/YOUR_FILE.parquet --gpu 0 --dry-run
~~~

Both hosts execute the same unmodified Python Figure 2 workflow and pinned
model/config identities. Actual dataset access, GPU compatibility, installed
dependencies and scratch space must be verified before removing --dry-run.

Read run/README.md and docs/REFACTOR_PHASE2.md for safety and limitations.
