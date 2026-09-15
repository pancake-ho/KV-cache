from __future__ import annotations

from typing import Any

from peft import LoraConfig, TaskType, get_peft_model


TASK_TYPE_MAP = {
    "CAUSAL_LM": TaskType.CAUSAL_LM,
}


def build_lora_config(lora_cfg: dict[str, Any]) -> LoraConfig:
    """
    lora 파라미터 값에 맞춰 config 구조 구축
    """
    task_type_name = lora_cfg["task_type"]

    if task_type_name not in TASK_TYPE_MAP:
        raise ValueError(
            f"Unsupported task_type={task_type_name!r}. "
            f"Supported={list(TASK_TYPE_MAP.keys())}"
        )

    modules_to_save = lora_cfg.get("modules_to_save")

    return LoraConfig(
        task_type=TASK_TYPE_MAP[task_type_name],
        inference_mode=lora_cfg.get("inference_mode", False),
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        modules_to_save=modules_to_save,
    )


def attach_lora(base_model, lora_cfg: dict[str, Any]):
    """
    기본 base model에 lora 결합
    """
    config = build_lora_config(lora_cfg)
    model = get_peft_model(base_model, config)

    return model