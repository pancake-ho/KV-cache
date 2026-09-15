#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_ROOT = (
    PROJECT_ROOT / "src"
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

from lora_exp.model.factory import (
    load_base_model,
    load_tokenizer,
)

from lora_exp.train.packing import (
    build_packed_blocks,
    estimate_available_full_blocks,
    load_parquet_dataset,
    FixedBlockCausalCollator,
)


def read_yaml(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def read_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    obj: Any,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def safe_extract(
    archive: Path,
    destination: Path,
):
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    root = destination.resolve()

    with tarfile.open(
        archive,
        "r:gz",
    ) as tar:
        for member in tar.getmembers():
            candidate = (
                destination
                / member.name
            ).resolve()

            if (
                candidate != root
                and root
                not in candidate.parents
            ):
                raise RuntimeError(
                    f"Unsafe archive path: "
                    f"{member.name}"
                )

        tar.extractall(
            destination
        )


def evaluate_blocks(
    model,
    dataset,
) -> float:
    collator = (
        FixedBlockCausalCollator()
    )

    losses: list[float] = []

    model.eval()

    with torch.no_grad():
        for idx in range(
            len(dataset)
        ):
            batch = collator(
                [dataset[idx]]
            )

            batch = {
                key: value.cuda(
                    non_blocking=True
                )
                for key, value
                in batch.items()
            }

            output = model(
                **batch,
                use_cache=False,
                return_dict=True,
            )

            loss = float(
                output.loss.detach().cpu()
            )

            if not math.isfinite(loss):
                raise RuntimeError(
                    "Non-finite evaluation loss"
                )

            losses.append(loss)

    return (
        sum(losses)
        / len(losses)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
    )

    parser.add_argument(
        "--train-config",
        default="configs/train.yaml",
    )

    parser.add_argument(
        "--work-dir",
        required=True,
    )

    args = parser.parse_args()

    model_cfg = read_yaml(
        PROJECT_ROOT
        / args.model_config
    )["model"]

    cfg = read_yaml(
        PROJECT_ROOT
        / args.train_config
    )["phase2"]

    work_dir = Path(
        os.path.expandvars(
            args.work_dir
        )
    )

    if work_dir.exists():
        raise RuntimeError(
            f"work_dir exists: {work_dir}"
        )

    work_dir.mkdir(
        parents=True
    )

    receipt = read_json(
        PROJECT_ROOT
        / cfg["phase1_receipt"]
    )

    archive = Path(
        receipt["archive"]
    )

    extract_root = (
        work_dir / "phase1"
    )

    safe_extract(
        archive,
        extract_root,
    )

    dataset_root = (
        extract_root
        / cfg[
            "dataset_root_name"
        ]
    )

    tokenizer = load_tokenizer(
        model_cfg
    )

    datasets = list(
        cfg["datasets"]
    )

    seq_len = int(
        cfg["sequence_length"]
    )

    text_field = (
        cfg["text_field"]
    )

    token_count_field = (
        cfg["token_count_field"]
    )

    append_eos = bool(
        cfg[
            "append_eos_between_samples"
        ]
    )

    eval_target = int(
        cfg[
            "validation_blocks"
        ]
    )

    seed = int(
        cfg["seed"]
    )

    eval_sets = {}

    print(
        "[BUILD FIXED VALIDATION SETS]"
    )

    for idx, name in enumerate(
        datasets
    ):
        val_ds = load_parquet_dataset(
            dataset_root
            / name
            / "validation.parquet"
        )

        available = (
            estimate_available_full_blocks(
                val_ds,
                token_count_field=(
                    token_count_field
                ),
                sequence_length=(
                    seq_len
                ),
                append_eos=(
                    append_eos
                ),
            )
        )

        blocks = min(
            eval_target,
            available,
        )

        eval_sets[name] = (
            build_packed_blocks(
                val_ds,
                tokenizer=tokenizer,
                text_field=(
                    text_field
                ),
                sequence_length=(
                    seq_len
                ),
                num_blocks=blocks,
                seed=(
                    seed
                    + idx * 1000
                    + 1
                ),
                append_eos=(
                    append_eos
                ),
            )
        )

        print(
            f"  {name}: "
            f"{blocks} blocks"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    base = load_base_model(
        model_cfg,
        device="cuda",
        use_cache=False,
    )

    output_root = Path(
        os.path.expandvars(
            cfg["output_root"]
        )
    )

    adapter_paths = {
        name: (
            output_root
            / name
            / f"seed{seed}"
            / "adapter_final"
        )
        for name in datasets
    }

    for name, path in (
        adapter_paths.items()
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{name} adapter missing: "
                f"{path}"
            )

    first_name = datasets[0]

    model = PeftModel.from_pretrained(
        base,
        adapter_paths[
            first_name
        ],
        adapter_name=first_name,
        is_trainable=False,
    )

    for name in datasets[1:]:
        model.load_adapter(
            adapter_paths[name],
            adapter_name=name,
            is_trainable=False,
        )

    model.eval()

    rows = []

    # --------------------------------------------------------
    # Base
    # --------------------------------------------------------

    print(
        "\n[EVAL] BASE"
    )

    with model.disable_adapter():
        for target_dataset in datasets:
            loss = evaluate_blocks(
                model,
                eval_sets[
                    target_dataset
                ],
            )

            rows.append(
                {
                    "model": "base",
                    "dataset": (
                        target_dataset
                    ),
                    "eval_loss": (
                        loss
                    ),
                }
            )

            print(
                f"  base -> "
                f"{target_dataset}: "
                f"{loss:.6f}"
            )

    # --------------------------------------------------------
    # Adapters
    # --------------------------------------------------------

    for adapter_name in datasets:
        model.set_adapter(
            adapter_name
        )

        print(
            f"\n[EVAL] "
            f"{adapter_name}"
        )

        for target_dataset in datasets:
            loss = evaluate_blocks(
                model,
                eval_sets[
                    target_dataset
                ],
            )

            rows.append(
                {
                    "model": (
                        adapter_name
                    ),
                    "dataset": (
                        target_dataset
                    ),
                    "eval_loss": (
                        loss
                    ),
                }
            )

            print(
                f"  {adapter_name} -> "
                f"{target_dataset}: "
                f"{loss:.6f}"
            )

    base_losses = {
        row["dataset"]:
        row["eval_loss"]
        for row in rows
        if row["model"] == "base"
    }

    for row in rows:
        row[
            "improvement_vs_base"
        ] = (
            base_losses[
                row["dataset"]
            ]
            - row["eval_loss"]
        )

    result_dir = (
        output_root
        / "cross_eval"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        result_dir
        / "cross_eval.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "dataset",
                "eval_loss",
                "improvement_vs_base",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    save_json(
        {
            "status": "PASS",
            "seed": seed,
            "rows": rows,
        },
        result_dir
        / "cross_eval.json",
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CROSS-EVALUATION COMPLETE"
    )

    print(
        f"CSV: {csv_path}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()