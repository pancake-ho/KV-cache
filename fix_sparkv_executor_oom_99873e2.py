from __future__ import annotations

import subprocess
from pathlib import Path

EXPECTED_BRANCH = "exp/sparkv-test"
EXPECTED_HEAD = "99873e2feee89d779350d1ed50c732f45479f1d3"
EXPECTED_EXECUTOR_BLOB = "31324ac02d03c32f520b85eb2e8e975b971a4d85"
EXPECTED_TEST_BLOB = "59d72fb24739afd1791fc37528ca528ab18433a0"


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"[ERROR] {label}: expected exactly one match, found {count}. "
            "Refusing to modify a stale/local-newer file."
        )
    return text.replace(old, new, 1)


repo = Path(git("rev-parse", "--show-toplevel"))
branch = git("branch", "--show-current")
head = git("rev-parse", "HEAD")

print(f"branch={branch}")
print(f"head={head}")

if branch != EXPECTED_BRANCH:
    raise SystemExit(
        f"[ERROR] expected branch {EXPECTED_BRANCH}; got {branch}"
    )
if head != EXPECTED_HEAD:
    raise SystemExit(
        "[ERROR] patch reviewed against "
        f"{EXPECTED_HEAD}, but local HEAD is {head}"
    )

executor_path = repo / "baseline/sparkv/executor.py"
test_path = repo / "baseline/sparkv/tests/test_sparkv_executor.py"

executor_blob = git("hash-object", str(executor_path))
test_blob = git("hash-object", str(test_path))

if executor_blob != EXPECTED_EXECUTOR_BLOB:
    raise SystemExit(
        "[ERROR] executor.py differs from reviewed HEAD: "
        f"{executor_blob}"
    )
if test_blob != EXPECTED_TEST_BLOB:
    raise SystemExit(
        "[ERROR] test_sparkv_executor.py differs from reviewed HEAD: "
        f"{test_blob}"
    )

text = executor_path.read_text(encoding="utf-8")

# Inference mode is thread-local.  The executor's local-compute path runs in a
# Python worker thread, so model.eval() in the parent thread is not enough to
# disable autograd graph construction.
for signature in [
    '''    def _projection_context(
''',
    '''    def _compute_attention_part(
''',
    '''    def compute_unit(
''',
    '''    def try_finalize_layer(
''',
]:
    text = replace_once(
        text,
        signature,
        '''    @torch.inference_mode()
''' + signature,
        "inference-mode decorator",
    )

# Add explicit release of tensors that are dead immediately after one
# (token-chunk, layer) hidden state has been finalized.
anchor = '''    def try_finalize_layer(
        self,
        t: int,
        layer_idx: int,
    ) -> bool:
'''
release_method = '''    def _release_finalized_state(
        self,
        t: int,
        layer_idx: int,
    ) -> None:
        \"\"\"Release GPU intermediates dead after one layer finalizes.

        The final KV tensors remain owned by UnitStore and the next-layer
        hidden state remains in hidden_inputs[(t, layer_idx + 1)].  Q/K/V
        projections, the consumed input hidden state, per-head attention
        outputs, and local-head bookkeeping for the finalized layer are no
        longer needed and otherwise accumulate across the 8x36 geometry.
        \"\"\"
        key = (
            t,
            layer_idx,
        )

        self.projection_cache.pop(
            key,
            None,
        )
        self.hidden_inputs.pop(
            key,
            None,
        )
        self.local_heads.pop(
            key,
            None,
        )

        for head in range(
            self.H
        ):
            self.attention_parts.pop(
                Chunk(
                    t,
                    layer_idx,
                    head,
                ),
                None,
            )

''' + anchor
text = replace_once(
    text,
    anchor,
    release_method,
    "release finalized state method",
)

old_cleanup = '''            # Projection/attention state at this layer is no longer required
            # once the hidden state for the next layer has been materialized.
            # Keep attention_parts only until request completion for simpler
            # debugging; clear the large shared projections.
            self.projection_cache.pop(
                key,
                None,
            )

            return True
'''

new_cleanup = '''            # The next-layer hidden state and final KV ownership are now
            # materialized.  Retaining old hidden/projection/attention tensors
            # grows GPU memory with every finalized (t, layer), and is not
            # required for subsequent SparKV execution.
            self._release_finalized_state(
                t,
                layer_idx,
            )

            return True
'''
text = replace_once(
    text,
    old_cleanup,
    new_cleanup,
    "finalized-state cleanup",
)

executor_path.write_text(
    text,
    encoding="utf-8",
)

test_text = test_path.read_text(encoding="utf-8")

test_text = replace_once(
    test_text,
'''from baseline.sparkv.executor import (
    CloudMemorySource,
    UnitStore,
''',
'''from baseline.sparkv.executor import (
    CloudMemorySource,
    HybridQwen3Engine,
    UnitStore,
''',
    "test import HybridQwen3Engine",
)

test_text += r'''


def test_finalized_layer_releases_dead_intermediates():
    engine = (
        HybridQwen3Engine
        .__new__(
            HybridQwen3Engine
        )
    )
    engine.H = 2

    target = (
        0,
        0,
    )
    keep = (
        0,
        1,
    )

    engine.hidden_inputs = {
        target:
            torch.zeros(1),
        keep:
            torch.ones(1),
    }
    engine.projection_cache = {
        target: (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        ),
        keep: (
            torch.ones(1),
            torch.ones(1),
            torch.ones(1),
        ),
    }
    engine.local_heads = {
        target: {0, 1},
        keep: {0},
    }
    engine.attention_parts = {
        Chunk(0, 0, 0):
            torch.zeros(1),
        Chunk(0, 0, 1):
            torch.zeros(1),
        Chunk(0, 1, 0):
            torch.ones(1),
    }

    engine._release_finalized_state(
        0,
        0,
    )

    assert target not in (
        engine.hidden_inputs
    )
    assert target not in (
        engine.projection_cache
    )
    assert target not in (
        engine.local_heads
    )
    assert (
        Chunk(0, 0, 0)
        not in engine.attention_parts
    )
    assert (
        Chunk(0, 0, 1)
        not in engine.attention_parts
    )

    # State for the next layer must remain intact.
    assert keep in (
        engine.hidden_inputs
    )
    assert keep in (
        engine.projection_cache
    )
    assert keep in (
        engine.local_heads
    )
    assert (
        Chunk(0, 1, 0)
        in engine.attention_parts
    )
'''

test_path.write_text(
    test_text,
    encoding="utf-8",
)

print("[OK] Patched executor inference mode + finalized-state cleanup.")
print("[NEXT] py_compile, pytest, git diff --check, then review/commit.")
