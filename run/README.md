# Run launchers: Seraph and lab server (Figure 2 only)

Scope: existing `baseline/crosskv/src/xmodel_kv/figure2` reproduction. These launchers do NOT implement Ridge/Neural mapping and do not modify baseline algorithms or saved runs.

## Seraph (moana-master)

```bash
cd /data/surt321/repos/lab/kv_cache
# Confirm permissions, partition and pending workloads first:
git status --short --branch
sinfo -p batch_eebme_ugrad
squeue -u "$USER"

bash run/seraph/figure2.sh \
  --dry-run --dataset /data/datasets/YOUR_FINEWEB_EDU_SINGLE_FILE.parquet \
  --mode smoke --tag fig2_refactor_smoke1
# Remove --dry-run when the job and available resources have been checked.
```

- One Slurm job, `gpu:1`, CPU <=16, RAM 29G, default partition `batch_eebme_ugrad`.
- All three phases run on *one* compute node and share local scratch under `/local_datasets/$USER/kv_figure2`.
- A single input file is copied to scratch once. Intermediate `.npy` stores remain on local scratch; only metrics/plots/manifests are exported to persistent `results/crosskv/figure2/<tag>/`.
- No Slurm job cancellation, scratch deletion or environment installation is performed. Scratch may be purged by the cluster after job completion; intermediate artifacts are **not** a durable checkpoint.
- This job may download model weights to local scratch if not otherwise configured. For a persistent read-only model cache, specify an approved path using `KV_FIG2_HF_HOME` after checking cluster policy.
- `--mode full` can need tens of GiB for extraction. `preflight.py` checks free space (default 20 GiB additional headroom), and the existing extractor also checks per-role capacity.
- Inspect `results/.../status.txt`, the Slurm `.out/.err`, and `results/.../results/summary.json` for completion.

## Lab server host (Docker)

```bash
# From the lab server HOST, after the repo is mounted into the running container.
nvidia-smi -i 0,1
# Check Docker availability and container GPU mappings before first use.
docker ps
bash run/lab/figure2.sh --container uav --container-repo /workspace/kv_cache \
  --dataset /workspace/datasets/YOUR_FINEWEB_EDU_SINGLE_FILE.parquet \
  --mode smoke --gpu 0 --tag fig2_lab_smoke1 --dry-run \
  --scratch-root /workspace/kv_scratch \
  --python /PATH/TO/EXISTING/PYTHON
# Remove --dry-run to run inside Docker after checking mounts and dependencies.
# Add --detach for long runs so SSH logout does not interrupt execution.
# Detached runs write docker.log and status.txt inside the result directory.
```

The Docker container, GPU, input file, Python interpreter, and writable scratch path **must already exist**. `--gpu` refers to the container-visible GPU index, which may differ from the physical host ID. It cannot reveal GPUs not already exposed by Docker. Inspect `docker exec ... nvidia-smi` to confirm mapping; do not assume the UAV Docker runtime has the KV-cache dependencies. No CPU fallback. Use `--detach` for long running jobs (`docker exec -d`); the launcher writes `docker.log` and `status.txt` in the selected persistent result directory.

## Execution contract

1. Figure 2 config, algorithm, Qwen3 model revisions, and existing test suite remain under `baseline/crosskv/`.
2. `run/common/` validates the environment and runs the same existing `prepare -> extract(source) -> extract(target) -> probe -> plot` Python modules.
3. Large per-layer activation arrays are never copied to persistent results by default. Result manifests include source/target metadata, stage output and branch/dirty status.
4. A fresh run tag is required. The launcher fails if the results directory exists and never overwrites historical runs.
5. Run `bash -n` checks locally; full model/GPU/scheduler verification requires actual target servers.

## Experiment-oriented interface

Current uniform entry points: run/seraph/baseline/crosskv_figure2.sh and run/lab/baseline/crosskv_figure2.sh. The Seraph legacy LoRA QKV adapter is run/seraph/baseline/lora_qkv.sh (--dry-run default; --submit explicit). SparKV and GP_semcache remain unported due to experiment branch guards and submodule/runtime uncertainty. See docs/REFACTOR_PHASE2.md.
