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
E2E_OUTPUT_ROOT = REPO_ROOT / "results/e2e"
E2E_PAIR = "1.7b-4b"
E2E_QA_JOB = "222976"
E2E_QA_REPORT = None  # Specify a path instead of resolving E2E_QA_JOB.
E2E_PROFILE = "semcache_policy_pilot"
E2E_SEMCACHE_ROOT = Path("/data/surt321/repos/lab/GP_semcache-5f325c5")
E2E_TINYBERT_DIR = Path("/data/surt321/models/semcache/tinybert-34707a3")
E2E_HF_HUB_CACHE = Path("/data/surt321/hf_cache/hub")
E2E_CACHEBRIDGE = {
    "1.7b-4b": E2E_RESULTS / "cachebridge/ridge-matched-512-attn-repair-fit/output/artifact",
}
E2E_REPAIR_FRACTION = 0.50
E2E_SEED = 2026
E2E_MIN_FREE_GIB = {"Qwen/Qwen3-4B": 15, "Qwen/Qwen3-8B": 22}
E2E_LORA_SOURCE_ADAPTER = "medqa_17b"
E2E_LORA_TARGET_ADAPTERS = ("medqa", "medmcqa", "fpb")
E2E_LORA_FIT_JOB = "223719"  # Set to the finished LoRA calibration job ID for evaluation.

E2E_MIXED = {"id": "mixed", "same_share": 0.25, "cross_share": 0.50,
             "cache_mib": 1024, "codec": "raw"}
E2E_PACKED = {**E2E_MIXED, "id": "bitpack5", "codec": "bitpack5"}
E2E_LORA_BASE = {
    "documents": 40, "repeats": 2, "new_tokens": 64, "window": 1,
    "scenarios": [{**E2E_MIXED, "same_share": 0, "cross_share": 1}],
}
E2E_LORA_SOURCE_ADAPTER = "medqa_17b"
E2E_LORA_TARGET_ADAPTERS = ("medqa", "medmcqa", "fpb")

# QASPER test archive from the dataset authors. The run uses local files only.
E2E_REUSE_DATASET = E2E_RESULTS / "datasets/qasper-test-and-evaluator-v0.3.tgz"
E2E_REUSE_DATASET_SHA256 = "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72"
E2E_REUSE_BASE = {
    "documents": 12, "repeats": 2, "new_tokens": 64,
    "questions_per_document": 4, "prefix_tokens": (2048, 3400),
}
E2E_REUSE_CASE = {**E2E_MIXED, "reuse_count": E2E_REUSE_BASE["questions_per_document"]}
E2E_REUSE_AMORTIZATION = {
    **E2E_REUSE_BASE, "reuse_experiment": "amortization",
    "scenarios": [{**E2E_REUSE_CASE, "id": f"reuse_{count}", "reuse_count": count}
                  for count in (1, 2, 4)],
}

E2E_PROFILES = {
    "cachebridge_v2_fit": {"fit_only": True},
    "reuse_amortization": E2E_REUSE_AMORTIZATION,
    "reuse_pilot": {
        **E2E_REUSE_AMORTIZATION, "documents": 4, "repeats": 1, "new_tokens": 8,
    },
    "reuse_pressure": {
        **E2E_REUSE_BASE, "reuse_experiment": "pressure", "pressure_window": 4,
        "scenarios": [{**E2E_REUSE_CASE, "id": f"reuse_cache_{mib}", "cache_mib": mib}
                      for mib in (512, 1024, 2048)],
    },
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
E2E_PROFILES["raw_sweep"] = {
    **E2E_PROFILES["sweep"],
    "scenarios": [
        case for case in E2E_PROFILES["sweep"]["scenarios"]
        if case["codec"] == "raw" and case["cache_mib"] == 1024
    ],
}
E2E_CACHEBRIDGE_V2_INITIALIZERS = {
    "1.7b-4b": E2E_RESULTS / "cachebridge/ridge-matched-512-head-local-fit/output/artifact",
}
E2E_CACHEBRIDGE_V2_MAX_HISTORY = 2048
E2E_CACHEBRIDGE_V2_FIT_JOB = None  # Set to the completed fitting job ID for comparison.

E2E_PROFILES["cachebridge_v2_compare"] = {
    **E2E_PROFILES["raw_sweep"], "compare_cachebridge_v2": True,
}
E2E_PROFILES.update({
    "lora_transfer_fit": {**E2E_LORA_BASE, "lora_stage": "fit"},
    "lora_transfer_pilot": {**E2E_LORA_BASE, "lora_stage": "evaluate",
                            "documents": 8, "repeats": 1, "new_tokens": 8},
    "lora_transfer_eval": {**E2E_LORA_BASE, "lora_stage": "evaluate"},
    "lora_cachebridge_v2_compare": {
        **E2E_LORA_BASE, "lora_stage": "evaluate",
        "lora_cachebridge_v2_history": E2E_CACHEBRIDGE_V2_MAX_HISTORY,
    },
})

# Paired LRU vs SemCache scoring under the same native-prefix cache pressure.
E2E_SEMCACHE_POLICY_CASES = [
    {**E2E_REUSE_CASE, "id": "lru", "cache_mib": 512, "cache_policy": "lru"},
    {**E2E_REUSE_CASE, "id": "semcache", "cache_mib": 512,
     "cache_policy": "semcache_frequency_age_size"},
]
E2E_PROFILES["semcache_policy_pressure"] = {
    **E2E_PROFILES["reuse_pressure"], "scenarios": E2E_SEMCACHE_POLICY_CASES,
}
E2E_PROFILES["semcache_policy_pilot"] = {
    **E2E_PROFILES["semcache_policy_pressure"], "documents": 4, "repeats": 1, "new_tokens": 8,
}
