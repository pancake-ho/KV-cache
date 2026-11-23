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
MULTI_USER = True

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

# 1.7B Adapters (현재는 내 폴더 환경에 맞춰져있기 때문에, 사용자별로 바꿔야함)
MULTI_USER_ADAPTERS["medqa_17b"] = (
    MODEL_17B,
    "medqa",
    Path("/data/surt321/models/lora_exp/phase3c_qwen3_17b_medqa_seed2026")
    / "job_217810/medqa/seed2026/adapter_final",
)

# Optional additional fitted Ridge artifacts, keyed by directed model pair.
MULTI_USER_EXTRA_RIDGE = {
    # (MODEL_06B, MODEL_17B): Path("/path/to/fitted_06b_to_17b_ridge"),
}
MULTI_USER_CROSS_MODEL_PROBES = (("medqa_17b", "medqa"),)

PREFIX_TOKENS = 512
SUFFIX_TOKENS = 24
NEW_TOKENS = 8
ALLOW_DOWNLOAD = False
COMPARE_KV_BITPACK = True  # Run raw and 5-bit cache modes with the same mapper.

OUTPUT_ROOT = REPO_ROOT / "results/semcache"

# Base-model LongBench E2E. An input report is data from the completed mapper evaluation.
E2E_DATA_ROOT = Path("/data") / getuser()
E2E_RESULTS = E2E_DATA_ROOT / "kv_mapper_optim_runs"
E2E_OUTPUT_ROOT = OUTPUT_ROOT
E2E_PAIR = "0.6b-8b"
E2E_QA_JOB = "223131"
E2E_QA_REPORT = None  # Specify a path instead of resolving E2E_QA_JOB.
E2E_PROFILE = "pilot"
E2E_SEMCACHE_ROOT = Path("/data/surt321/repos/lab/GP_semcache-r5-d38edf8")
E2E_TINYBERT_DIR = None
E2E_CACHEBRIDGE = {
    "1.7b-4b": E2E_RESULTS / "cachebridge/ridge-matched-512-attn-repair-fit/output/artifact",
}
E2E_REPAIR_FRACTION = 0.30
E2E_SEED = 2026
E2E_MIN_FREE_GIB = {"Qwen/Qwen3-4B": 15, "Qwen/Qwen3-8B": 22}

E2E_MIXED = {"id": "mixed", "same_share": 0.25, "cross_share": 0.50,
             "cache_mib": 1024, "codec": "raw"}
E2E_PACKED = {**E2E_MIXED, "id": "bitpack5", "codec": "bitpack5"}
E2E_PROFILES = {
    "pilot": {"documents": 8, "repeats": 1, "new_tokens": 8, "window": 1,
              "scenarios": [E2E_MIXED, E2E_PACKED]},
    "sweep": {"documents": 40, "repeats": 2, "new_tokens": 64, "window": 2,
              "scenarios": [
                  E2E_MIXED, {**E2E_MIXED, "id": "no_cross", "cross_share": 0},
                  {**E2E_MIXED, "id": "high_cross", "cross_share": 0.75},
                  {**E2E_MIXED, "id": "small_cache", "cache_mib": 512}, E2E_PACKED,
              ]},
    "legacy": {"documents": 40, "repeats": 2, "new_tokens": 64, "window": 2,
               "scenarios": [E2E_MIXED]},
}
