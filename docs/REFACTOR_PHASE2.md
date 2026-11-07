# 2차 정리 — modular ownership and server portability

## Scope and preservation
Branch refactor/baseline-proposed-run. Changes are additive except root README,
proposed README, run README, lab launcher GPU authorization and .gitignore.
All original baseline modules, submodule pins, configs, checkpoints, logs, and
historical experiment results remain untouched. No algorithm or paper claims
are changed.

## Experiment coverage
- CrossKV Figure 2 (Qwen3-1.7B -> 4B): supported launcher on both Seraph and
  lab Docker. Workflow: prepare -> source KV -> target KV -> OLS/R2 -> plot.
  The same baseline config/model revision is used. Real server testing pending.
- LoRA QKV: Seraph-only wrapper forwards to existing legacy script. The legacy
  YAML expands /data/$USER paths; lab support cannot be claimed without
  verifying dataset/adapter copies and rewriting external path configuration.
- SparKV: existing scripts enforce branch exp/sparkv-test and might remove
  success-path scratch. Their branch guard is retained; no wrapper invokes
  them on the refactor branch.
- GP_semcache: Git submodule reference, not moved or updated; integration unverified.
- Proposed: package namespace reserved; no algorithm implemented or tested.

## Module boundaries
run/ is only invocation, platform adaptation and provenance.
baseline/ remains independent; do not import proposed method into baselines.
proposed/ implements only our method and may import documented third-party APIs
without editing baseline code. Store model/mapper version, seed, source/target
cache shapes, split identity and platform info in each research artifact.

## Seraph constraints
Compute exclusively on allocated nodes, one GPU max, 16 CPUs max, 29G RAM.
Use batch_eebme_ugrad. NAS is persistent storage but large tensor scratch
must reside in compute-node /local_datasets. Single Figure 2 job keeps all
three heavy phases on the same node; intermediate tensors are NOT exported
and therefore are NOT durable checkpoints. Never silently fall back to CPU.

## Lab Docker constraints
Host GPUs 0/1 are the only authorized physical GPUs. The selected logical GPU
must resolve to one of their NVML UUIDs. Docker container must already be
running, with a bind-mounted repo/input and compatible Python packages.
The lab's UAV container is not assumed to contain the KV environment.
Detached jobs must be monitored via docker.log and status.txt.

## Verification
Run python -m unittest discover -s run/tests -v, python run/common/catalog.py
--check, and bash -n over run/**/*.sh. These are CPU-only structural checks.
GPU/scheduler/domain-output parity is not verified by modifying the GitHub
repository and requires host-side smoke evaluation.

## Recovery
Inspect git log and git show for the phase1 and phase2 commits. After preserving
worktree status and newer changes, git revert selected commits creates a
non-destructive rollback commit. Avoid reset --hard or force-push.

The pre-existing LoRA Slurm script itself is hardcoded to
/data/surt321/repos/lab/kv_cache; the wrapper now enforces that exact legacy path.

## Step 1 runtime safety
Both Figure 2 launchers pre-check scratch capacity BEFORE copying the input dataset,
including source-file size and the existing KV array estimate. A second check
is performed after staging. CPU-only launcher tests verify dry-run and guardrails;
these do not replace a live Slurm/Docker smoke test.
