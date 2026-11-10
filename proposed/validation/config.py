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

# Use the same Slurm entry point for the expanded controlled trace.
MULTI_USER = False
MODEL_06B = "Qwen/Qwen3-0.6B"
MODEL_17B = "Qwen/Qwen3-1.7B"
MODEL_4B = "Qwen/Qwen3-4B"
MULTI_USER_EXTRA_MODELS = {
    MODEL_06B: "c1899de289a04d12100db370d81485cdf75e47ca",
}

# Training provenance for existing and planned PEFT checkpoints.
# Entries are (HF dataset, configuration, train split, held-out split).
ADAPTER_DATASETS = {
    "medqa": ("awinml/medqa", "questions", "train", "test"),
    "medmcqa": ("openlifescienceai/medmcqa", None, "train", "validation"),
    "fpb": ("ChanceFocus/en-fpb", None, "train", "test"),
    "fiqasa": ("ChanceFocus/flare-fiqasa", None, "train", "test"),
    "gsm8k": ("openai/gsm8k", "main", "train", "test"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "train", "validation"),
}

# adapter_id: (base model, dataset key, local adapter directory)
MULTI_USER_ADAPTERS = {
    "medqa": (MODEL_4B, "medqa", LORA_ROOT / "medqa/seed2026/adapter_final"),
    "medmcqa": (MODEL_4B, "medmcqa", LORA_ROOT / "medmcqa/seed2026/adapter_final"),
    "fpb": (MODEL_4B, "fpb", LORA_ROOT / "fpb/seed2026/adapter_final"),
    "fiqasa": (MODEL_4B, "fiqasa", LORA_ROOT / "fiqasa/seed2026/adapter_final"),
    # Enable after training on the pinned 1.7B base:
    # "gsm8k": (MODEL_17B, "gsm8k", Path("/path/to/gsm8k/adapter_final")),
    # "hotpotqa": (MODEL_17B, "hotpotqa", Path("/path/to/hotpotqa/adapter_final")),
}

# Optional additional fitted Ridge artifacts, keyed by directed model pair.
MULTI_USER_EXTRA_RIDGE = {
    # (MODEL_06B, MODEL_17B): Path("/path/to/fitted_06b_to_17b_ridge"),
}

PREFIX_TOKENS = 512
SUFFIX_TOKENS = 24
NEW_TOKENS = 8
ALLOW_DOWNLOAD = False
COMPARE_KV_BITPACK = False  # Run raw and 5-bit cache modes with the same mapper.

OUTPUT_ROOT = REPO_ROOT / "results/semcache"
