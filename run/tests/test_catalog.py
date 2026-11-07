"""CPU-only research-boundary tests."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("kv_catalog", ROOT / "run/common/catalog.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class InventoryTests(unittest.TestCase):
    def test_tracks_separate(self):
        self.assertEqual(len(MODULE.EXPERIMENTS), 5)
        self.assertEqual(MODULE.EXPERIMENTS["proposed"]["scope"], "proposed")

    def test_paths_resolve(self):
        for row in MODULE.inventory(ROOT):
            if row["name"] != "semcache":
                self.assertTrue(row["source_exists"], row["name"])
            if row["seraph"]:
                self.assertTrue(row["seraph_exists"], row["name"])
            if row["lab"]:
                self.assertTrue(row["lab_exists"], row["name"])

    def test_legacy_status_explicit(self):
        for name in ("sparkv", "semcache", "proposed"):
            self.assertIsNone(MODULE.EXPERIMENTS[name]["seraph"])
            self.assertIsNone(MODULE.EXPERIMENTS[name]["lab"])
        self.assertIsNone(MODULE.EXPERIMENTS["lora-qkv"]["lab"])


if __name__ == "__main__":
    unittest.main()
