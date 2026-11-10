"""Editable settings for the Moana SemCache serving smoke."""

from getpass import getuser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MLP_DIR = REPO_ROOT / "results/neural_mapper_qwen3_17b_4b_k12_fp32_v1"
RIDGE_DIR = None  # Set to the paired run's ridge directory to include Ridge.

LORA_ROOT = Path("/data") / getuser() / "models/lora_exp/phase3c_cross_domain_seed2026"
LORA_ADAPTERS = {
    # Uncomment both entries to include the LoRA trace.
    # "medqa": LORA_ROOT / "medqa/seed2026/adapter_final",
    # "fpb": LORA_ROOT / "fpb/seed2026/adapter_final",
}

PREFIX_TOKENS = 512
SUFFIX_TOKENS = 24
NEW_TOKENS = 8
ALLOW_DOWNLOAD = False
COMPARE_KV_BITPACK = False  # Run raw and 5-bit cache modes with the same mapper.

OUTPUT_ROOT = REPO_ROOT / "results/semcache"
