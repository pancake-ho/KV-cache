"""Validate the local D5 checkout and delegate its OPT experiment commands."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .semcache_smoke import git_state


D5_COMMIT = "5f325c542196e737a94dedc3f059d4f087d1b251"
CHECKOUT_ENV = "KV_TOP_SEMCACHE_ROOT"
SCRIPTS = {
    "lightperf": "159_r5_d5_lightperf.py",
    "sweep": "163_r5_d5_sweep_v5.py",
    "l6b": "165_r5_d5_l6b.py",
    "recovery": "166_r5_d5_l6_recovery.py",
}


def d5_git_state(root: Path) -> dict:
    state = git_state(root)
    if state["commit"] != D5_COMMIT or state["status"]:
        raise ValueError(f"Expected a clean SemCache checkout at {D5_COMMIT}: {root}")
    missing = [name for name in SCRIPTS.values() if not (root / "scripts" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"D5 entry points missing at {root}: {', '.join(missing)}")
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semcache-root", default=os.environ.get(CHECKOUT_ENV), type=Path)
    parser.add_argument("experiment", choices=SCRIPTS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.semcache_root is None:
        parser.error(f"provide --semcache-root or set {CHECKOUT_ENV}")
    root = args.semcache_root.expanduser().resolve()
    try:
        d5_git_state(root)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return subprocess.call([sys.executable, str(root / "scripts" / SCRIPTS[args.experiment]),
                            *args.arguments], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
