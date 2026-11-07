# KV-cache repository restructuring — Phase 1 (proposal)

Branch basis: `refactor/baseline-proposed-run`, starting at `f69a1e2b0bcd6bfab04a6ec6dfae49c5f2af9387`.

## Boundaries

- `baseline/`: externally derived algorithms and archived reproductions (preserved unchanged).
- `proposed/`: user-developed method only (README placeholder in phase 1).
- `run/seraph/`: Slurm launcher; `run/lab/`: lab-Docker launcher; `run/common/`: shared read-only experiment runner, environment checks and lightweight artifact export.
- `results/`: ignored by existing `.gitignore`; old outputs are never moved, deleted, or overwritten.

## Figure 2 call graph

```text
run/{seraph,lab}/figure2.sh
  -> allocated/inside-container figure2_job.sh or figure2_inner.sh
  -> run/common/runtime.sh
  -> baseline/crosskv/jobs/figure2/check_env.py
  -> existing baseline/crosskv/tests/test_figure2*.py
  -> xmodel_kv.figure2.prepare
  -> xmodel_kv.figure2.extract --role source
  -> xmodel_kv.figure2.extract --role target
  -> xmodel_kv.figure2.probe --device cuda:0
  -> xmodel_kv.figure2.plot
```

These modules are invoked unmodified. In particular, the old Figure 2 uses single-source matched-head **OLS**, not the top-k Ridge production mapper.

## Compatibility and limitations

- The historical `baseline/crosskv/jobs/figure2/submit.sh` remains intact; new launchers coexist until the user confirms migration.
- No model-specific code is modified, and other baselines (`lora_exp`, `sparkv`, `GP_semcache` submodule) are untouched.
- Scratch is node-local and intentionally left for inspection; actual retention depends on host policy. A new job does not automatically resume incomplete extraction from another node.
- The hard dependency `transformers==4.57.3` is inherited; unsupported environments fail before model loading.
- Use unique run tags; simultaneous runs with the same tag are prohibited.
- GPU test/real Slurm or Docker runtime has **not** been run by these off-server static tests.
