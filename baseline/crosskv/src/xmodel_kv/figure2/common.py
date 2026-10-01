from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

KINDS = ("k_rope", "k_stripped", "v")
BASE_COMMIT = "dfa17b142e6f9e39a23becf77248e1c120a261d8"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def config(path, smoke=False):
    result = read_json(path)
    if smoke:
        result.update(train_sequences=8, heldout_sequences=4, sequence_length=128)
    for field in ("train_sequences", "heldout_sequences", "sequence_length", "stride", "flush_sequences"):
        if not isinstance(result[field], int) or result[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if result["schema_version"] != 1:
        raise ValueError("unsupported config schema")
    if result["storage_dtype"] != "float16":
        raise ValueError("this extraction schema uses float16 storage")
    if result["covariance_dtype"] not in ("float32", "float64"):
        raise ValueError("covariance_dtype must be float32 or float64")
    if not 0 < result["covariance_rcond"] < 1:
        raise ValueError("covariance_rcond must be in (0,1)")
    if not 0 < result["document_train_fraction"] < 1:
        raise ValueError("document_train_fraction must be in (0,1)")
    result["smoke"] = bool(smoke)
    return result


def provenance():
    def git(*args):
        p = subprocess.run(["git", *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    versions = {}
    for name in ("torch", "transformers", "numpy", "matplotlib", "pyarrow", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(), "hostname": platform.node(),
        "base_commit": BASE_COMMIT, "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"), "git_status": git("status", "--short"),
        "versions": versions, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


@contextlib.contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process is using {path}") from exc
        yield
