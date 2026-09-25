#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import transformers
import yaml
import peft

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lora_exp.common.manifest import save_json
from lora_exp.common.seed import set_seed
from lora_exp.model.factory import (
    load_base_model,
    load_tokenizer,
)
from lora_exp.model.lora_factory import attach_lora
from lora_exp.model.validate import (
    get_parameter_statistics,
    get_trainable_parameter_names,
    summarize_linear_module_suffixes,
    validate_architecture,
    validate_finite_tensor,
    validate_lora_only_trainable,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
    )

    parser.add_argument(
        "--lora-config",
        default="configs/lora.yaml",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--output-dir",
        default="runs/env_check",
    )

    return parser.parse_args()


def read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return result.stdout.strip()

        return (
            f"returncode={result.returncode}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )

    except Exception as exc:
        return f"<failed: {exc!r}>"


def find_targeted_module_names(model) -> list[str]:
    names = getattr(model, "targeted_module_names", None)

    if names is None:
        # Fallback for PEFT versions where this property is absent.
        names = sorted(
            {
                name.rsplit(".", 1)[0]
                for name, _ in model.named_parameters()
                if ".lora_" in name
            }
        )

    return list(names)


def inspect_cache(past_key_values):
    """
    Support both modern Transformers Cache objects and
    legacy tuple-of-tuples representations.
    """
    layers = []

    # Modern Cache API
    if hasattr(past_key_values, "layers"):
        for idx, layer in enumerate(past_key_values.layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)

            if key is None:
                key = getattr(layer, "key_cache", None)

            if value is None:
                value = getattr(layer, "value_cache", None)

            if key is None or value is None:
                layers.append(
                    {
                        "layer": idx,
                        "error": (
                            "Could not locate key/value tensors "
                            "in modern Cache layer."
                        ),
                    }
                )
                continue

            layers.append(
                {
                    "layer": idx,
                    "key_shape": list(key.shape),
                    "value_shape": list(value.shape),
                    "key_dtype": str(key.dtype),
                    "value_dtype": str(value.dtype),
                    "key_finite": bool(torch.isfinite(key).all().item()),
                    "value_finite": bool(
                        torch.isfinite(value).all().item()
                    ),
                }
            )

        return layers

    # Some Cache implementations expose key_cache/value_cache
    if (
        hasattr(past_key_values, "key_cache")
        and hasattr(past_key_values, "value_cache")
    ):
        for idx, (key, value) in enumerate(
            zip(
                past_key_values.key_cache,
                past_key_values.value_cache,
            )
        ):
            layers.append(
                {
                    "layer": idx,
                    "key_shape": list(key.shape),
                    "value_shape": list(value.shape),
                    "key_dtype": str(key.dtype),
                    "value_dtype": str(value.dtype),
                    "key_finite": bool(torch.isfinite(key).all().item()),
                    "value_finite": bool(
                        torch.isfinite(value).all().item()
                    ),
                }
            )

        return layers

    # Legacy format
    if isinstance(past_key_values, (list, tuple)):
        for idx, item in enumerate(past_key_values):
            key, value = item[:2]

            layers.append(
                {
                    "layer": idx,
                    "key_shape": list(key.shape),
                    "value_shape": list(value.shape),
                    "key_dtype": str(key.dtype),
                    "value_dtype": str(value.dtype),
                    "key_finite": bool(torch.isfinite(key).all().item()),
                    "value_finite": bool(
                        torch.isfinite(value).all().item()
                    ),
                }
            )

        return layers

    raise TypeError(
        "Unsupported past_key_values type: "
        f"{type(past_key_values)}"
    )


def main():
    args = parse_args()

    os.chdir(PROJECT_ROOT)

    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_yaml = read_yaml(
        PROJECT_ROOT / args.model_config
    )
    lora_yaml = read_yaml(
        PROJECT_ROOT / args.lora_config
    )

    model_cfg = model_yaml["model"]
    expected_cfg = model_yaml["expected_architecture"]
    lora_cfg = lora_yaml["lora"]

    print("=" * 80)
    print("PHASE 0 — ENVIRONMENT / MODEL / LoRA VALIDATION")
    print("=" * 80)

    manifest = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "seed": args.seed,
        "project_root": str(PROJECT_ROOT),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "git_commit": command_output(
            ["git", "rev-parse", "HEAD"]
        ),
        "git_status": command_output(
            ["git", "status", "--short"]
        ),
        "model_config": model_cfg,
        "expected_architecture": expected_cfg,
        "lora_config": lora_cfg,
    }

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            "Do not continue with CPU fallback."
        )

    device = "cuda"

    gpu_info = {
        "name": torch.cuda.get_device_name(0),
        "capability": list(
            torch.cuda.get_device_capability(0)
        ),
        "total_memory_bytes": torch.cuda.get_device_properties(
            0
        ).total_memory,
    }

    manifest["gpu"] = gpu_info

    print("\n[1/8] Runtime")
    print(f"  Python            : {sys.version.split()[0]}")
    print(f"  Torch ver-        : {torch.__version__}")
    print(f"  Transformers ver- : {transformers.__version__}")
    print(f"  PEFT ver-         : {peft.__version__}")
    print(f"  CUDA              : {torch.version.cuda}")
    print(f"  GPU               : {gpu_info['name']}")

    print("\n[2/8] Loading tokenizer...")
    tokenizer = load_tokenizer(model_cfg)

    print("\n[3/8] Loading Qwen3-4B base in BF16...")
    base_model = load_base_model(
        model_cfg,
        device=device,
        use_cache=True,
    )

    actual_arch = validate_architecture(
        base_model,
        expected_cfg,
    )

    manifest["actual_architecture"] = actual_arch

    print("\n[4/8] Architecture")
    for key, value in actual_arch.items():
        print(f"  {key:24s}: {value}")

    linear_suffixes = summarize_linear_module_suffixes(
        base_model
    )

    manifest["linear_module_suffix_counts"] = (
        linear_suffixes
    )

    print("\n[5/8] Linear module suffix counts")
    for name, count in linear_suffixes.items():
        print(f"  {name:20s}: {count}")

    # Base is frozen before attaching LoRA.
    for param in base_model.parameters():
        param.requires_grad = False

    print("\n[6/8] Attaching plain LoRA...")
    model = attach_lora(
        base_model,
        lora_cfg,
    )

    validate_lora_only_trainable(model)

    param_stats = get_parameter_statistics(model)
    trainable_names = get_trainable_parameter_names(model)
    targeted_modules = find_targeted_module_names(model)

    manifest["parameter_statistics"] = param_stats
    manifest["trainable_parameter_names"] = trainable_names
    manifest["targeted_module_names"] = targeted_modules

    model.print_trainable_parameters()

    print(
        f"  Targeted modules : {len(targeted_modules)}"
    )

    print("\n  Example targeted modules:")
    for name in targeted_modules[:30]:
        print(f"    - {name}")

    print("\n[7/8] Forward / KV-cache smoke test")

    # We deliberately use a plain text forward here rather than
    # generation. The current goal is to validate model/cache
    # mechanics, not generation behavior.
    text = (
        "Question: What organ pumps blood through "
        "the human body?\nAnswer:"
    )

    batch = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    model.eval()

    with torch.no_grad():
        output = model(
            **batch,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

    validate_finite_tensor(
        output.logits,
        name="logits",
    )

    if output.past_key_values is None:
        raise RuntimeError(
            "past_key_values is None despite use_cache=True."
        )

    cache_info = inspect_cache(
        output.past_key_values
    )

    manifest["smoke_test"] = {
        "text": text,
        "input_ids_shape": list(
            batch["input_ids"].shape
        ),
        "logits_shape": list(
            output.logits.shape
        ),
        "logits_finite": True,
        "cache_type": str(
            type(output.past_key_values)
        ),
        "cache_layers": cache_info,
    }

    if len(cache_info) != expected_cfg[
        "num_hidden_layers"
    ]:
        raise RuntimeError(
            "Unexpected KV cache layer count: "
            f"{len(cache_info)} != "
            f"{expected_cfg['num_hidden_layers']}"
        )

    # Validate representative layers.
    expected_kv_heads = expected_cfg[
        "num_key_value_heads"
    ]
    expected_head_dim = expected_cfg[
        "head_dim"
    ]

    for layer_info in cache_info:
        if "error" in layer_info:
            raise RuntimeError(
                f"Cache inspection failed: {layer_info}"
            )

        k_shape = layer_info["key_shape"]
        v_shape = layer_info["value_shape"]

        if len(k_shape) != 4 or len(v_shape) != 4:
            raise RuntimeError(
                "KV cache tensor is expected to be rank-4. "
                f"Got K={k_shape}, V={v_shape}"
            )

        if k_shape[1] != expected_kv_heads:
            raise RuntimeError(
                "Unexpected number of KV heads: "
                f"{k_shape[1]} != {expected_kv_heads}"
            )

        if k_shape[-1] != expected_head_dim:
            raise RuntimeError(
                "Unexpected KV head dimension: "
                f"{k_shape[-1]} != {expected_head_dim}"
            )

        if not layer_info["key_finite"]:
            raise RuntimeError(
                f"Non-finite key cache at layer "
                f"{layer_info['layer']}."
            )

        if not layer_info["value_finite"]:
            raise RuntimeError(
                f"Non-finite value cache at layer "
                f"{layer_info['layer']}."
            )

    print(
        f"  Input shape  : "
        f"{batch['input_ids'].shape}"
    )
    print(
        f"  Logits shape : "
        f"{output.logits.shape}"
    )
    print(
        f"  Cache layers : "
        f"{len(cache_info)}"
    )

    first = cache_info[0]
    last = cache_info[-1]

    print(
        f"  Layer 0 K/V  : "
        f"{first['key_shape']} / "
        f"{first['value_shape']}"
    )

    print(
        f"  Layer 35 K/V : "
        f"{last['key_shape']} / "
        f"{last['value_shape']}"
    )

    manifest_path = output_dir / "manifest.json"

    save_json(
        manifest,
        manifest_path,
    )

    print("\n[8/8] PASS")
    print(
        "  Qwen3-4B architecture       : PASS"
    )
    print(
        "  BF16 / no quantization      : PASS"
    )
    print(
        "  LoRA-only trainable params  : PASS"
    )
    print(
        "  Forward finite              : PASS"
    )
    print(
        "  KV-cache shape/finite       : PASS"
    )
    print(
        f"\n  Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()