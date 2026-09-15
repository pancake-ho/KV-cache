from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """
    dtype 입력 오류 시 에러 출력
    """
    try:
        return DTYPE_MAP[dtype_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dtype={dtype_name!r}. "
            f"Supported={list(DTYPE_MAP.keys())}"
        ) from exc


def load_tokenizer(model_cfg: dict[str, Any]):
    """
    model에 따른 tokenizer load
    """
    model_id = model_cfg["model_id"]
    revision = model_cfg.get("revision", "main")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model(
    model_cfg: dict[str, Any],
    *,
    device: str,
    use_cache: bool,
):
    """
    양자화 과정을 거치지 않고 base model을 load

    중요 사항:
      - no BitsAndBytesConfig
      - no load_in_4bit
      - no load_in_8bit
    """
    model_id = model_cfg["model_id"]
    revision = model_cfg.get("revision", "main")
    torch_dtype = resolve_dtype(model_cfg["dtype"])

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        low_cpu_mem_usage=model_cfg.get("low_cpu_mem_usage", True),
    )

    model.config.use_cache = use_cache
    model.to(device)

    return model