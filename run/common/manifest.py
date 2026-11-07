"""Capture platform, repository, model-agnostic environment metadata."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def cmd(root: Path, *argv: str) -> str | None:
    p = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    root = Path(os.environ["KV_MANIFEST_REPO"])
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": os.environ["KV_MANIFEST_PLATFORM"],
        "mode": os.environ["KV_MANIFEST_MODE"],
        "git_commit": cmd(root, "git", "rev-parse", "HEAD"),
        "git_branch": cmd(root, "git", "branch", "--show-current"),
        "git_status_porcelain": cmd(root, "git", "status", "--porcelain"),
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "hostname": platform.node(),
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "container": os.environ.get("KV_DOCKER_CONTAINER"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2) + "\n")
    print("Provenance:", args.output)


if __name__ == "__main__":
    main()
