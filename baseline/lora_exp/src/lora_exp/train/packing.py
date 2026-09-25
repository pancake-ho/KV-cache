from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from datasets import Dataset


def load_parquet_dataset(
    path: str | Path,
) -> Dataset:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset parquet does not exist: {path}"
        )

    return Dataset.from_parquet(
        str(path)
    )


def estimate_available_full_blocks(
    dataset: Dataset,
    *,
    token_count_field: str,
    sequence_length: int,
    append_eos: bool,
) -> int:
    """
    Estimate the number of full fixed-length blocks available.

    Phase 1 already measured Qwen3 token counts for paper_text.
    We reuse those values to choose a common token budget before
    doing the more expensive exact tokenization/packing.
    """
    if token_count_field not in dataset.column_names:
        raise KeyError(
            f"{token_count_field!r} not found in dataset. "
            f"Available columns={dataset.column_names}"
        )

    eos_extra = 1 if append_eos else 0

    total_tokens = 0

    for count in dataset[token_count_field]:
        count = int(count)

        if count <= 0:
            raise RuntimeError(
                f"Invalid token count found: {count}"
            )

        total_tokens += (
            count + eos_extra
        )

    return total_tokens // sequence_length


def build_packed_blocks(
    dataset: Dataset,
    *,
    tokenizer,
    text_field: str,
    sequence_length: int,
    num_blocks: int,
    seed: int,
    append_eos: bool,
    tokenizer_batch_size: int = 128,
) -> Dataset:
    """
    Deterministically shuffle examples and pack them into exactly
    `num_blocks` fixed-length causal-LM token blocks.

    Each source example is separated with EOS when requested.

    The final incomplete block is intentionally discarded.
    """
    if num_blocks <= 0:
        raise ValueError(
            f"num_blocks must be positive, got {num_blocks}"
        )

    if text_field not in dataset.column_names:
        raise KeyError(
            f"{text_field!r} is missing. "
            f"Available columns={dataset.column_names}"
        )

    eos_token_id = tokenizer.eos_token_id

    if append_eos and eos_token_id is None:
        raise RuntimeError(
            "append_eos=True but tokenizer.eos_token_id is None"
        )

    indices = list(
        range(len(dataset))
    )

    rng = random.Random(seed)
    rng.shuffle(indices)

    blocks: list[list[int]] = []
    buffer: list[int] = []

    target_tokens = (
        num_blocks * sequence_length
    )

    produced_tokens = 0

    for start in range(
        0,
        len(indices),
        tokenizer_batch_size,
    ):
        batch_indices = indices[
            start:start + tokenizer_batch_size
        ]

        texts = [
            str(dataset[int(idx)][text_field])
            for idx in batch_indices
        ]

        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )

        for token_ids in encoded["input_ids"]:
            buffer.extend(
                int(token_id)
                for token_id in token_ids
            )

            if append_eos:
                buffer.append(
                    int(eos_token_id)
                )

            while (
                len(buffer) >= sequence_length
                and len(blocks) < num_blocks
            ):
                block = buffer[
                    :sequence_length
                ]

                buffer = buffer[
                    sequence_length:
                ]

                if len(block) != sequence_length:
                    raise RuntimeError(
                        "Internal packing error: "
                        f"{len(block)} != {sequence_length}"
                    )

                blocks.append(block)
                produced_tokens += sequence_length

            if len(blocks) >= num_blocks:
                break

        if len(blocks) >= num_blocks:
            break

    if len(blocks) != num_blocks:
        raise RuntimeError(
            "Could not produce requested number of blocks. "
            f"requested={num_blocks}, "
            f"produced={len(blocks)}, "
            f"dataset_rows={len(dataset)}"
        )

    if produced_tokens != target_tokens:
        raise RuntimeError(
            "Packed token-budget mismatch: "
            f"expected={target_tokens}, "
            f"actual={produced_tokens}"
        )

    return Dataset.from_dict(
        {
            "input_ids": blocks,
        }
    )


class FixedBlockCausalCollator:
    """
    Collator for already-packed fixed-length sequences.

    Labels equal input_ids, so this performs standard causal-LM
    full-sequence SFT, matching the paper's text-field SFT style.
    """

    def __call__(
        self,
        features: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import torch

        input_ids = torch.tensor(
            [
                feature["input_ids"]
                for feature in features
            ],
            dtype=torch.long,
        )

        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.long,
        )

        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }