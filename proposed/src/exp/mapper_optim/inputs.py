"""Aligned Qwen3 inputs and calibration/evaluation split checks."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    revision: str | None
    layers: int
    heads: int
    width: int
    context: int

    @classmethod
    def inspect(cls, name: str, revision: str | None) -> ModelIdentity:
        cfg = AutoConfig.from_pretrained(name, revision=revision)
        if (cfg.model_type != "qwen3" or set(cfg.layer_types) != {"full_attention"}
                or cfg.rope_scaling):
            raise ValueError("Requires dense Qwen3, full causal attention, and default RoPE")
        return cls(name, cfg._commit_hash or revision, cfg.num_hidden_layers,
                   cfg.num_key_value_heads, cfg.head_dim, cfg.max_position_embeddings)

    def metadata(self) -> dict:
        return asdict(self)

    def load(self, device: torch.device, dtype: torch.dtype):
        return AutoModelForCausalLM.from_pretrained(
            self.name, revision=self.revision, dtype=dtype, attn_implementation="sdpa",
        ).to(device).eval()


def token_hash(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()


def read_examples(source: ModelIdentity, target: ModelIdentity, path: Path, *,
                  length: int, limit: int) -> list[dict]:
    if not 2 <= length <= min(source.context, target.context) or limit < 1:
        raise ValueError("Invalid token length, model context, or example limit")
    src = AutoTokenizer.from_pretrained(source.name, revision=source.revision)
    tgt = AutoTokenizer.from_pretrained(target.name, revision=target.revision)
    if src.get_vocab() != tgt.get_vocab() or src.special_tokens_map != tgt.special_tokens_map:
        raise ValueError("Source and target tokenizers differ in vocabulary or special tokens")
    examples, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Expected a nonempty text at {path}:{line_number}")
            enc = dict(add_special_tokens=False, truncation=True, max_length=length,
                       return_attention_mask=False, verbose=False)
            ids = src(text, **enc)["input_ids"]
            if ids != tgt(text, **enc)["input_ids"]:
                raise ValueError(f"Tokenizer mismatch at {path}:{line_number}")
            if len(ids) != length:
                continue
            digest = token_hash(ids)
            if digest in seen:
                continue
            seen.add(digest)
            examples.append({"ids": ids, "sha256": digest,
                             "text_sha256": hashlib.sha256(text.strip().encode()).hexdigest(),
                             "line": line_number})
            if len(examples) == limit:
                break
    if len(examples) < limit:
        raise ValueError(f"{path}: found {len(examples)}/{limit} distinct rows with >= {length} tokens")
    return examples


def check_split(examples: list[dict], calibration: list[dict], calibration_length: int) -> None:
    fit_tokens = {row["sha256"] for row in calibration}
    fit_texts = {row["text_sha256"] for row in calibration}
    for row in examples:
        if (row["text_sha256"] in fit_texts
                or token_hash(row["ids"][:calibration_length]) in fit_tokens):
            raise ValueError(f"Evaluation row {row['line']} overlaps calibration")
        # Catch identical input prefixes when evaluation uses a shorter context.
        count = min(len(row["ids"]), calibration_length)
        if token_hash(row["ids"][:count]) in {
            token_hash(item["ids"][:count]) for item in calibration
        }:
            raise ValueError(f"Evaluation row {row['line']} shares a calibration token prefix")
