"""CPU-only launcher contract tests; never submit jobs or contact Docker."""
from __future__ import annotations
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
SERAPH = ROOT / "run/seraph/baseline/crosskv_figure2.sh"
LAB = ROOT / "run/lab/baseline/crosskv_figure2.sh"


def run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )


class RunnerContractsTest(unittest.TestCase):
    def test_help_has_no_server_dependency(self):
        for script in (SERAPH, LAB):
            with self.subTest(script=script.name):
                result = run(script, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Usage:", result.stdout)

    def test_lab_dry_run_does_not_require_docker(self):
        result = run(LAB, "--mode", "smoke", "--dataset", "/missing/file.parquet",
                     "--gpu", "0", "--tag", "test_2026", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY RUN:", result.stdout)
        self.assertIn("test_2026", result.stdout)

    def test_lab_rejects_unapproved_gpu_index(self):
        result = run(LAB, "--mode", "smoke", "--dataset", "/missing/file.parquet",
                     "--gpu", "2", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU", result.stderr)

    def test_lab_rejects_relative_scratch_and_results(self):
        for flag, value in (("--scratch-root", "tmp/work"), ("--result-dir", "relative")):
            with self.subTest(flag=flag):
                result = run(LAB, "--mode", "smoke", "--dataset", "/missing/file.parquet",
                             flag, value, "--dry-run")
                self.assertNotEqual(result.returncode, 0)

    def test_seraph_dry_run_requires_shared_dataset(self):
        result = run(SERAPH, "--mode", "smoke", "--dataset", "/missing/file.parquet", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dataset", result.stderr)

    def test_reject_invalid_run_tag_on_lab(self):
        result = run(LAB, "--mode", "smoke", "--dataset", "/missing/file.parquet",
                     "--tag", "../escape", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tag", result.stderr)


if __name__ == "__main__":
    unittest.main()
