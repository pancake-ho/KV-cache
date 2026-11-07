# Proposed method (reserved)

This directory is reserved for the **new research method**. No neural KV mapper, KV transfer or multi-user scheduling implementation has been validated here yet.

The current 2026-10-07 baseline is under `baseline/crosskv/`. Until a proposed method and its interfaces are approved, keep this directory separate from paper reproduction code. Future code will be split by responsibility (data / model / mapping / transfer / train / eval), not by the execution server.

Use `run/seraph/` and `run/lab/` for launchers only; keep experimental results and checkpoints out of Git.
