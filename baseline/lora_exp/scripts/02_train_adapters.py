#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import torch
import yaml

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
    load_tokenizer,
)

from lora_exp.train.packing import (
    build_packed_blocks,
    estimate_available_full_blocks,
    load_parquet_dataset,
)

from lora_exp.train.runner import (
    train_one_adapter,
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
        "--train-config",
        default="configs/train.yaml",
    )

    parser.add_argument(
        "--work-dir",
        required=True,
    )

    return parser.parse_args()


def read_yaml(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def read_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_output(
    args: list[str],
) -> str:
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return (
            f"<git failed: "
            f"{result.stderr.strip()}>"
        )

    return result.stdout.strip()


def safe_extract(
    archive_path: Path,
    destination: Path,
) -> None:
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_resolved = (
        destination.resolve()
    )

    with tarfile.open(
        archive_path,
        "r:gz",
    ) as tar:
        for member in tar.getmembers():
            candidate = (
                destination
                / member.name
            ).resolve()

            if (
                destination_resolved
                not in candidate.parents
                and candidate
                != destination_resolved
            ):
                raise RuntimeError(
                    "Unsafe path in archive: "
                    f"{member.name}"
                )

        tar.extractall(
            destination
        )


def main():
    args = parse_args()

    os.chdir(
        PROJECT_ROOT
    )

    model_yaml = read_yaml(
        PROJECT_ROOT
        / args.model_config
    )

    lora_yaml = read_yaml(
        PROJECT_ROOT
        / args.lora_config
    )

    train_yaml = read_yaml(
        PROJECT_ROOT
        / args.train_config
    )

    model_cfg = model_yaml["model"]
    lora_cfg = lora_yaml["lora"]
    cfg = train_yaml["phase2"]

    work_dir = Path(
        os.path.expandvars(
            args.work_dir
        )
    ).resolve()

    if work_dir.exists():
        raise RuntimeError(
            f"work_dir already exists: {work_dir}"
        )

    work_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    receipt_path = (
        PROJECT_ROOT
        / cfg["phase1_receipt"]
    )

    if not receipt_path.exists():
        raise FileNotFoundError(
            "Phase-1 receipt not found: "
            f"{receipt_path}"
        )

    receipt = read_json(
        receipt_path
    )

    if receipt.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Phase-1 receipt status is not PASS."
        )

    archive_path = Path(
        receipt["archive"]
    )

    if not archive_path.exists():
        raise FileNotFoundError(
            "Phase-1 archive is missing: "
            f"{archive_path}"
        )

    print(
        "[VERIFY] Phase-1 archive SHA256..."
    )

    actual_sha = sha256_file(
        archive_path
    )

    expected_sha = receipt[
        "archive_sha256"
    ]

    if actual_sha != expected_sha:
        raise RuntimeError(
            "Phase-1 archive SHA256 mismatch.\n"
            f"expected={expected_sha}\n"
            f"actual={actual_sha}"
        )

    print(
        f"[PASS] archive sha256={actual_sha}"
    )

    extract_root = (
        work_dir / "phase1"
    )

    print(
        "[EXTRACT] Phase-1 archive..."
    )

    safe_extract(
        archive_path,
        extract_root,
    )

    dataset_root = (
        extract_root
        / cfg[
            "dataset_root_name"
        ]
    )

    if not dataset_root.exists():
        raise RuntimeError(
            "Extracted dataset root missing: "
            f"{dataset_root}"
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

    append_eos = bool(
        cfg[
            "append_eos_between_samples"
        ]
    )

    token_count_field = cfg[
        "token_count_field"
    ]

    text_field = cfg[
        "text_field"
    ]

    train_sources = {}
    validation_sources = {}
    available_blocks = {}

    print(
        "\n[BUDGET AUDIT]"
    )

    for dataset_name in datasets:
        train_path = (
            dataset_root
            / dataset_name
            / "train.parquet"
        )

        val_path = (
            dataset_root
            / dataset_name
            / "validation.parquet"
        )

        train_ds = (
            load_parquet_dataset(
                train_path
            )
        )

        val_ds = (
            load_parquet_dataset(
                val_path
            )
        )

        train_sources[
            dataset_name
        ] = train_ds

        validation_sources[
            dataset_name
        ] = val_ds

        blocks = (
            estimate_available_full_blocks(
                train_ds,
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

        available_blocks[
            dataset_name
        ] = blocks

        print(
            f"  {dataset_name:12s}: "
            f"rows={len(train_ds):7d}, "
            f"available_full_blocks="
            f"{blocks}"
        )

    common_blocks = min(
        available_blocks.values()
    )

    max_blocks = cfg.get(
        "max_common_train_blocks"
    )

    if max_blocks is not None:
        common_blocks = min(
            common_blocks,
            int(max_blocks),
        )

    if common_blocks <= 0:
        raise RuntimeError(
            "Common train block budget is zero."
        )

    print(
        "\n[COMMON BUDGET]"
    )

    print(
        f"  blocks/adapter = {common_blocks}"
    )

    print(
        f"  tokens/epoch   = "
        f"{common_blocks * seq_len}"
    )

    print(
        f"  epochs         = "
        f"{cfg['num_train_epochs']}"
    )

    print(
        f"  total tokens/adapter = "
        f"{int(common_blocks * seq_len * float(cfg['num_train_epochs']))}"
    )

    output_root = Path(
        os.path.expandvars(
            cfg["output_root"]
        )
    ).resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan = {
        "status": "READY",
        "git_branch": git_output(
            [
                "git",
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ]
        ),
        "git_commit": git_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ]
        ),
        "git_status": git_output(
            [
                "git",
                "status",
                "--short",
            ]
        ),
        "phase1_receipt": str(
            receipt_path
        ),
        "phase1_archive": str(
            archive_path
        ),
        "phase1_archive_sha256": (
            actual_sha
        ),
        "model": model_cfg,
        "lora": lora_cfg,
        "datasets": datasets,
        "available_train_blocks": (
            available_blocks
        ),
        "common_train_blocks": int(
            common_blocks
        ),
        "sequence_length": seq_len,
        "tokens_per_epoch": int(
            common_blocks * seq_len
        ),
        "training": {
            key: value
            for key, value
            in cfg.items()
            if key not in {
                "phase1_receipt",
                "dataset_root_name",
                "datasets",
                "text_field",
                "token_count_field",
                "output_root",
            }
        },
    }

    save_json(
        plan,
        output_root
        / "phase2_plan.json",
    )

    seed = int(
        cfg["seed"]
    )

    eval_blocks = int(
        cfg[
            "validation_blocks"
        ]
    )

    summaries = {}

    for idx, dataset_name in enumerate(
        datasets
    ):
        dataset_seed = (
            seed
            + idx * 1000
        )

        print(
            "\n"
            + "#" * 80
        )

        print(
            f"PACKING {dataset_name}"
        )

        print(
            "#" * 80
        )

        packed_train = (
            build_packed_blocks(
                train_sources[
                    dataset_name
                ],
                tokenizer=tokenizer,
                text_field=(
                    text_field
                ),
                sequence_length=(
                    seq_len
                ),
                num_blocks=(
                    common_blocks
                ),
                seed=dataset_seed,
                append_eos=(
                    append_eos
                ),
            )
        )

        available_eval = (
            estimate_available_full_blocks(
                validation_sources[
                    dataset_name
                ],
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

        actual_eval_blocks = min(
            eval_blocks,
            available_eval,
        )

        if actual_eval_blocks <= 0:
            raise RuntimeError(
                f"{dataset_name}: "
                "no full validation blocks"
            )

        packed_eval = (
            build_packed_blocks(
                validation_sources[
                    dataset_name
                ],
                tokenizer=tokenizer,
                text_field=(
                    text_field
                ),
                sequence_length=(
                    seq_len
                ),
                num_blocks=(
                    actual_eval_blocks
                ),
                seed=dataset_seed + 1,
                append_eos=(
                    append_eos
                ),
            )
        )

        adapter_output = (
            output_root
            / dataset_name
            / f"seed{seed}"
        )

        summary = train_one_adapter(
            dataset_name=(
                dataset_name
            ),
            train_dataset=(
                packed_train
            ),
            eval_dataset=(
                packed_eval
            ),
            model_cfg=model_cfg,
            lora_cfg=lora_cfg,
            train_cfg=cfg,
            output_dir=(
                adapter_output
            ),
            seed=seed,
        )

        summaries[
            dataset_name
        ] = summary

        save_json(
            {
                "status": "RUNNING",
                "summaries": summaries,
            },
            output_root
            / "phase2_summary.json",
        )

        del packed_train
        del packed_eval

        torch.cuda.empty_cache()

    save_json(
        {
            "status": "PASS",
            "summaries": summaries,
        },
        output_root
        / "phase2_summary.json",
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PHASE 2 TRAINING PASS"
    )

    print(
        "=" * 80
    )

    for name, summary in (
        summaries.items()
    ):
        print(
            f"{name:12s} | "
            f"eval_loss="
            f"{summary['eval_metrics']['eval_loss']:.6f}"
        )


if __name__ == "__main__":
    main()