from __future__ import annotations

from collections import Counter
from typing import Any

import torch


def validate_architecture(
    model,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """
    실제 model 구조와의 mismatch 출력
    
    현재는 Qwen3-4B만이 대상으로, 이에 대해서만 로그 출력
    """
    cfg = model.config

    actual = {
        "model_type": cfg.model_type,
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": getattr(
            cfg,
            "head_dim",
            cfg.hidden_size // cfg.num_attention_heads,
        ),
    }

    mismatches = {}

    for key, expected_value in expected.items():
        actual_value = actual.get(key)

        if actual_value != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "actual": actual_value,
            }

    if mismatches:
        raise RuntimeError(
            "Qwen architecture validation failed:\n"
            + "\n".join(
                f"  {key}: expected={value['expected']}, "
                f"actual={value['actual']}"
                for key, value in mismatches.items()
            )
        )

    return actual


def summarize_linear_module_suffixes(model) -> dict[str, int]:
    suffix_counter = Counter()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix_counter[name.split(".")[-1]] += 1

    return dict(sorted(suffix_counter.items()))


def get_parameter_statistics(model) -> dict[str, float | int]:
    """
    전체 파라미터 및 LoRA에 따른 훈련 가능 파라미터 값 출력
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    percentage = (
        100.0 * trainable / total
        if total > 0
        else 0.0
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": percentage,
    }


def get_trainable_parameter_names(model) -> list[str]:
    """
    훈련 가능 파라미터에 대한 name 값 출력
    """
    return [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    ]


def validate_lora_only_trainable(model) -> None:
    """
    lora에 따른 훈련 가능 파라미터 검증
    """
    bad = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # PEFT standard LoRA trainables contain lora_ in their names.
        # modules_to_save is intentionally not used in this experiment.
        if "lora_" not in name:
            bad.append(name)

    if bad:
        preview = "\n".join(f"  - {name}" for name in bad[:30])

        raise RuntimeError(
            "Non-LoRA parameters are unexpectedly trainable:\n"
            f"{preview}"
        )


def validate_finite_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    """
    infinite tensor 값이 나타나지 않도록 오류를 검증
    """
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{name} contains NaN or Inf."
        )