from __future__ import annotations

import subprocess
from pathlib import Path

EXPECTED_BRANCH = "exp/sparkv-test"
EXPECTED_HEAD = "aedc6c81bf352fa6b766afdc304adaa2fefcf9b9"

EXPECTED_BLOBS = {
    ".gitignore": "7f43f8b4473d8316dc8fbbe574a4cbc7d4da54ab",
    "baseline/sparkv/codec.py": "c89b261329ed44d61b85cedca7db0e71f7e4a335",
    "baseline/sparkv/executor.py": "d6b12371efb7ccec3724db7a3c90debb5748f1e0",
    "baseline/sparkv/experiment.py": "1be8aac793aa06d43ba8b673448e68ab13a49b7f",
    "baseline/sparkv/run/run_sparkv.sh": "be281e5374a593a127f6d6c80d0aefefc2b4a225",
}


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


def replace_once(
    text: str,
    old: str,
    new: str,
    label: str,
) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"[ERROR] {label}: expected exactly one match, found {count}. "
            "Refusing a broad/stale rewrite."
        )
    return text.replace(old, new, 1)


repo = Path(git("rev-parse", "--show-toplevel"))
branch = git("branch", "--show-current")
head = git("rev-parse", "HEAD")

print(f"branch={branch}")
print(f"head={head}")

if branch != EXPECTED_BRANCH:
    raise SystemExit(
        f"[ERROR] expected branch {EXPECTED_BRANCH}, got {branch}"
    )
if head != EXPECTED_HEAD:
    raise SystemExit(
        "[ERROR] This patch was reviewed against "
        f"{EXPECTED_HEAD}, but local HEAD is {head}. "
        "Do not apply a stale patch."
    )

for rel, expected in EXPECTED_BLOBS.items():
    path = repo / rel
    actual = git("hash-object", str(path))
    print(f"{rel}: {actual}")
    if actual != expected:
        raise SystemExit(
            "[ERROR] Target differs from the reviewed remote blob:\n"
            f"  file={rel}\n"
            f"  expected={expected}\n"
            f"  actual={actual}\n"
            "Refusing to overwrite local/newer work."
        )

# .gitignore
path = repo / ".gitignore"
path.write_text(
    "results/\n"
    "logs/\n"
    "baseline/sparkv/logs/\n"
    "**/__pycache__/\n"
    "*.py[cod]\n"
    ".pytest_cache/\n",
    encoding="utf-8",
)

# codec.py
path = repo / "baseline/sparkv/codec.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
'''    inverse = {
        code.to01(): symbol
        for symbol, code in codes.items()
    }

    # bitarray.decode(codebook) is implemented in C and returns an iterator.
    decoded = list(
        bits.decode(codes)
    )

    if len(decoded) != count:
        raise ValueError(
            "Huffman decoded symbol count mismatch: "
            f"expected={count}, got={len(decoded)}"
        )

    return np.asarray(
        decoded,
        dtype=np.uint8,
    )
''',
'''    # bitarray.decode(codebook) is implemented in C and returns an
    # iterator. Consume it directly into the final uint8 array instead of
    # materializing Python int objects first.
    iterator = bits.decode(
        codes
    )
    decoded = np.fromiter(
        iterator,
        dtype=np.uint8,
        count=count,
    )

    if decoded.size != count:
        raise ValueError(
            "Huffman decoded symbol count mismatch: "
            f"expected={count}, got={decoded.size}"
        )

    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(
            "Huffman decoded more symbols than declared: "
            f"count={count}"
        )

    return decoded
''',
    "codec Huffman fast path",
)
path.write_text(text, encoding="utf-8")

# executor.py
path = repo / "baseline/sparkv/executor.py"
text = path.read_text(encoding="utf-8")

text = replace_once(
    text,
'''    compute_wall_ms: float = 0.0
    support_attention_ms: float = 0.0
    physical_dependency_wait_ms: float = 0.0

    runtime_migrations: int = 0
''',
'''    compute_wall_ms: float = 0.0
    compute_effective_ms: float = 0.0
    support_attention_ms: float = 0.0
    physical_dependency_wait_ms: float = 0.0

    paper_not_ready_events: int = 0
    physical_not_ready_events: int = 0

    runtime_migrations: int = 0
''',
    "executor stats fields",
)

text = replace_once(
    text,
'''            "compute_wall_ms":
                self.compute_wall_ms,
            "support_attention_ms":
                self.support_attention_ms,
            "physical_dependency_wait_ms":
                self.physical_dependency_wait_ms,
            "runtime_migrations":
''',
'''            "compute_wall_ms":
                self.compute_wall_ms,
            "compute_effective_ms":
                self.compute_effective_ms,
            "support_attention_ms":
                self.support_attention_ms,
            "physical_dependency_wait_ms":
                self.physical_dependency_wait_ms,
            "paper_not_ready_events":
                self.paper_not_ready_events,
            "physical_not_ready_events":
                self.physical_not_ready_events,
            "runtime_migrations":
''',
    "executor stats to_dict",
)

text = replace_once(
    text,
'''    seed: int,
    controller_config: RuntimeControllerConfig,
) -> tuple[
''',
'''    seed: int,
    controller_config: RuntimeControllerConfig,
    cloud_source: CloudMemorySource | None = None,
) -> tuple[
''',
    "execute_sparkv signature",
)

text = replace_once(
    text,
'''    # Intentionally outside request timing.
    cloud = (
        CloudMemorySource(
            sample_dir,
            meta,
        )
    )
''',
'''    # For measured runs, callers should preload local cloud-artifact bytes
    # before starting the request timer. The fallback preserves the old debug
    # API when TTFT is not being measured.
    cloud = cloud_source
    if cloud is None:
        cloud = (
            CloudMemorySource(
                sample_dir,
                meta,
            )
        )
''',
    "executor cloud source",
)

text = replace_once(
    text,
'''        stage_compute_begin = [
            None
        ]
        stage_compute_end = [
            None
        ]
''',
'''        stage_compute_begin = [
            None
        ]
        stage_compute_end = [
            None
        ]
        stage_dependency_wait_ms = [
            0.0
        ]
''',
    "stage dependency accumulator",
)

text = replace_once(
    text,
'''                        if not ready:
                            next_remaining.append(
                                item
                            )
                            continue

                        if not (
                            engine.compute_unit(
                                c
                            )
                        ):
                            next_remaining.append(
                                item
                            )
                            continue
''',
'''                        if not ready:
                            stats.paper_not_ready_events += 1
                            next_remaining.append(
                                item
                            )
                            continue

                        if not (
                            engine.compute_unit(
                                c
                            )
                        ):
                            stats.physical_not_ready_events += 1
                            next_remaining.append(
                                item
                            )
                            continue
''',
    "readiness diagnostics",
)

text = replace_once(
    text,
'''                    if progress:
                        idle_begin = None
                        continue

                    # Let the concurrent streaming path deliver physical
                    # dependencies.  If it has finished and no further
                    # progress is possible, carry the compute operations to
                    # the next stage.
                    if stream_finished.is_set():
                        break

                    if idle_begin is None:
                        idle_begin = (
                            time.perf_counter()
                        )

                    time.sleep(
                        0.001
                    )

                if idle_begin is not None:
                    stats.physical_dependency_wait_ms += (
                        time.perf_counter()
                        - idle_begin
                    ) * 1000.0
''',
'''                    if progress:
                        if idle_begin is not None:
                            waited_ms = (
                                time.perf_counter()
                                - idle_begin
                            ) * 1000.0
                            stats.physical_dependency_wait_ms += (
                                waited_ms
                            )
                            stage_dependency_wait_ms[0] += (
                                waited_ms
                            )
                            idle_begin = None
                        continue

                    # Let the concurrent streaming path deliver physical
                    # dependencies. If it has finished and no further
                    # progress is possible, carry the compute operations to
                    # the next stage.
                    if stream_finished.is_set():
                        if idle_begin is not None:
                            waited_ms = (
                                time.perf_counter()
                                - idle_begin
                            ) * 1000.0
                            stats.physical_dependency_wait_ms += (
                                waited_ms
                            )
                            stage_dependency_wait_ms[0] += (
                                waited_ms
                            )
                            idle_begin = None
                        break

                    if idle_begin is None:
                        idle_begin = (
                            time.perf_counter()
                        )

                    time.sleep(
                        0.001
                    )

                if idle_begin is not None:
                    waited_ms = (
                        time.perf_counter()
                        - idle_begin
                    ) * 1000.0
                    stats.physical_dependency_wait_ms += (
                        waited_ms
                    )
                    stage_dependency_wait_ms[0] += (
                        waited_ms
                    )
                    idle_begin = None
''',
    "dependency wait accounting",
)

text = replace_once(
    text,
'''        stats.compute_wall_ms += (
            actual_compute_ms
        )

        controller.observe_stage(
            predicted_compute_ms=
                predicted_compute_ms,
            actual_compute_ms=
                actual_compute_ms,
            predicted_stream_ms=
                predicted_stream_ms,
            actual_stream_ms=
                actual_stream_ms,
        )
''',
'''        stats.compute_wall_ms += (
            actual_compute_ms
        )

        effective_compute_ms = max(
            0.0,
            actual_compute_ms
            - stage_dependency_wait_ms[0],
        )
        stats.compute_effective_ms += (
            effective_compute_ms
        )

        controller.observe_stage(
            predicted_compute_ms=
                predicted_compute_ms,
            actual_compute_ms=
                effective_compute_ms,
            predicted_stream_ms=
                predicted_stream_ms,
            actual_stream_ms=
                actual_stream_ms,
        )
''',
    "effective compute observation",
)

text = replace_once(
    text,
'''                "actual_compute_ms":
                    actual_compute_ms,
                "predicted_stream_ms":
''',
'''                "actual_compute_ms":
                    effective_compute_ms,
                "actual_compute_worker_ms":
                    actual_compute_ms,
                "dependency_wait_ms":
                    stage_dependency_wait_ms[0],
                "predicted_stream_ms":
''',
    "stage record compute fields",
)

path.write_text(text, encoding="utf-8")

# experiment.py
path = repo / "baseline/sparkv/experiment.py"
text = path.read_text(encoding="utf-8")

text = replace_once(
    text,
'''from baseline.sparkv.executor import (
    execute_sparkv,
)
''',
'''from baseline.sparkv.executor import (
    CloudMemorySource,
    execute_sparkv,
)
''',
    "experiment imports",
)

text = replace_once(
    text,
'''            if not (
                schedule_path
                .is_file()
            ):
                raise FileNotFoundError(
                    "missing schedule: "
                    f"{schedule_path}"
                )

            for repeat in range(
''',
'''            if not (
                schedule_path
                .is_file()
            ):
                raise FileNotFoundError(
                    "missing schedule: "
                    f"{schedule_path}"
                )

            meta = json.loads(
                (
                    sample_dir
                    / "meta.json"
                ).read_text(
                    encoding="utf-8"
                )
            )
            preload_begin = (
                time.perf_counter()
            )
            cloud_source = (
                CloudMemorySource(
                    sample_dir,
                    meta,
                )
            )
            cloud_preload_ms = (
                time.perf_counter()
                - preload_begin
            ) * 1000.0

            for repeat in range(
''',
    "experiment cloud preload",
)

text = replace_once(
    text,
'''                        controller_config=(
                            RuntimeControllerConfig(
                                window=(
                                    args.runtime_window
                                ),
                                imbalance_margin=(
                                    args.imbalance_margin
                                ),
                                max_migrations_per_stage=(
                                    args.max_migrations_per_stage
                                ),
                            )
                        ),
                    )
''',
'''                        controller_config=(
                            RuntimeControllerConfig(
                                window=(
                                    args.runtime_window
                                ),
                                imbalance_margin=(
                                    args.imbalance_margin
                                ),
                                max_migrations_per_stage=(
                                    args.max_migrations_per_stage
                                ),
                            )
                        ),
                        cloud_source=(
                            cloud_source
                        ),
                    )
''',
    "experiment cloud_source argument",
)

text = replace_once(
    text,
'''                    "schedule_path":
                        str(
                            schedule_path
                        ),
                }
''',
'''                    "schedule_path":
                        str(
                            schedule_path
                        ),
                    "cloud_preload_ms":
                        float(
                            cloud_preload_ms
                        ),
                    "measurement_scope":
                        (
                            "preloaded cloud bytes; "
                            "wire + Huffman decode + H2D "
                            "+ context rebuild + first token"
                        ),
                }
''',
    "experiment result metadata",
)

path.write_text(text, encoding="utf-8")

# run_sparkv.sh
path = repo / "baseline/sparkv/run/run_sparkv.sh"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
'''# Paper states 1024-token chunks but does not disclose Delta t.
readonly DELTA_MS="${DELTA_MS:-5.0}"
''',
'''# The paper defines Delta t but does not disclose its numerical value.
# `auto` is a reproduction/smoke fallback implemented by scheduler.py.
# It must not be reported as an author-provided setting.
readonly DELTA_MS="${DELTA_MS:-auto}"
''',
    "run_sparkv Delta default",
)
path.write_text(text, encoding="utf-8")

print("[OK] Remaining SparKV evaluation fixes applied.")
