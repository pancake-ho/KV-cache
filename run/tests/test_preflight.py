"""CPU-only, stdlib tests for scratch sizing. Run: python -m unittest discover -s run/tests -v"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "common" / "preflight.py"
spec = importlib.util.spec_from_file_location("kv_preflight", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(preflight)


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.config = dict(train_sequences=500, heldout_sequences=64,
                           sequence_length=1024, stride=4,
                           expected_source_layers=28, expected_target_layers=36,
                           expected_kv_heads=8, expected_head_dim=128)

    def test_expected_storage_formula(self):
        expected = 564 * 256 * 64 * 8 * 128 * 3 * 2
        self.assertEqual(preflight.estimate_bytes(self.config, smoke=False), expected)

    def test_smoke_smaller_than_full(self):
        self.assertLess(preflight.estimate_bytes(self.config, smoke=True),
                        preflight.estimate_bytes(self.config, smoke=False))

    def test_missing_dataset_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                preflight.check_space(self.config, dataset=Path(tmp) / "absent.parquet",
                                      scratch=Path(tmp), smoke=True)


if __name__ == "__main__":
    unittest.main()
