#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import datasets
import huggingface_hub
import pyarrow
import transformers
import yaml
from huggingface_hub import HfApi
from transformers import AutoTokenizer


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )


from lora_exp.data.pipeline import (
    add_token_counts,
    build_dataset_stats,
    convert_shards_to_records,
    filter_records,
    save_prepared_dataset,
    source_manifest,
)

from lora_exp.data.registry import (
    load_flashcards,
    load_medmcqa,
    load_pubmedqa,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/data.yaml",
    )

    parser.add_argument(
        "--work-dir",
        required=True,
    )

    parser.add_argument(
        "--archive-dir",
        required=True,
    )

    parser.add_argument(
        "--project-manifest-dir",
        default="data/manifests",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def command_output(
    command: list[str],
) -> str:
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


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_yaml(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def resolve_model_revision(
    model_id: str,
) -> str:
    info = HfApi().model_info(
        model_id
    )

    sha = getattr(
        info,
        "sha",
        None,
    )

    if not sha:
        raise RuntimeError(
            "Could not resolve tokenizer/model "
            f"revision for {model_id}"
        )

    return str(sha)


def write_json(
    data: dict,
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
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def main():
    args = parse_args()

    os.chdir(PROJECT_ROOT)

    cfg = read_yaml(
        PROJECT_ROOT / args.config
    )

    phase_cfg = cfg["phase1"]
    dataset_cfg = cfg["datasets"]

    work_dir = Path(
        os.path.expandvars(
            args.work_dir
        )
    ).resolve()

    archive_dir = Path(
        os.path.expandvars(
            args.archive_dir
        )
    ).resolve()

    project_manifest_dir = (
        PROJECT_ROOT
        / args.project_manifest_dir
    )

    if work_dir.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Work directory already exists: "
                f"{work_dir}\n"
                "Use a new job-specific directory "
                "or pass --overwrite."
            )

        shutil.rmtree(work_dir)

    work_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    project_manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hf_cache = Path(
        os.environ.get(
            "HF_DATASETS_CACHE",
            work_dir / "hf_cache",
        )
    )

    hf_cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "PHASE 1 — DATASET ACQUISITION / "
        "NORMALIZATION / AUDIT",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    tokenizer_model_id = (
        phase_cfg[
            "tokenizer_model_id"
        ]
    )

    tokenizer_revision = (
        resolve_model_revision(
            tokenizer_model_id
        )
    )

    print(
        "\n[1/7] Loading Qwen3 tokenizer",
        flush=True,
    )
    print(
        f"  model_id : "
        f"{tokenizer_model_id}",
        flush=True,
    )
    print(
        f"  revision : "
        f"{tokenizer_revision}",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            tokenizer_model_id,
            revision=tokenizer_revision,
            use_fast=True,
        )
    )

    # We measure true sequence length ourselves,
    # so prevent misleading tokenizer max-length warnings.
    tokenizer.model_max_length = int(
        1e9
    )

    print(
        "\n[2/7] Loading source datasets",
        flush=True,
    )

    all_shards = {
        "medmcqa": load_medmcqa(
            dataset_cfg["medmcqa"],
            cache_dir=hf_cache,
        ),
        "pubmedqa": load_pubmedqa(
            dataset_cfg["pubmedqa"],
            cache_dir=hf_cache,
        ),
        "flashcards": load_flashcards(
            dataset_cfg["flashcards"],
            cache_dir=hf_cache,
        ),
    }

    for name, shards in (
        all_shards.items()
    ):
        print(
            f"\n  {name}:",
            flush=True,
        )

        for shard in shards:
            print(
                "    "
                f"{shard.repo_id} @ "
                f"{shard.revision[:12]} | "
                f"{shard.config_name} | "
                f"{shard.split_name} | "
                f"rows={len(shard.dataset)}",
                flush=True,
            )

    prepared_root = (
        work_dir
        / phase_cfg["output_name"]
    )

    prepared_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "phase": 1,
        "timestamp": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "status": "running",
        "project_root": str(
            PROJECT_ROOT
        ),
        "work_dir": str(
            work_dir
        ),
        "prepared_root": str(
            prepared_root
        ),
        "hostname": (
            platform.node()
        ),
        "slurm_job_id": (
            os.environ.get(
                "SLURM_JOB_ID"
            )
        ),
        "slurm_partition": (
            os.environ.get(
                "SLURM_JOB_PARTITION"
            )
        ),
        "git_branch": command_output(
            [
                "git",
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ]
        ),
        "git_commit": command_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ]
        ),
        "git_status": command_output(
            [
                "git",
                "status",
                "--short",
            ]
        ),
        "python": sys.version,
        "transformers": (
            transformers.__version__
        ),
        "datasets": (
            datasets.__version__
        ),
        "huggingface_hub": (
            huggingface_hub.__version__
        ),
        "pyarrow": (
            pyarrow.__version__
        ),
        "tokenizer": {
            "model_id": (
                tokenizer_model_id
            ),
            "revision": (
                tokenizer_revision
            ),
        },
        "phase1_config": (
            phase_cfg
        ),
        "dataset_config": (
            dataset_cfg
        ),
        "sources": {},
        "prepared": {},
        "notes": [
            (
                "max_input_length=1024 is "
                "inherited from the selected "
                "TMLR setup."
            ),
            (
                "Token-length filtering uses "
                "Qwen3-4B tokenizer instead of "
                "the paper's Llama tokenizer "
                "because Qwen3-4B is the "
                "current experimental backbone."
            ),
            (
                "paper_text preserves each "
                "dataset-specific SFT format."
            ),
            (
                "normalized_text is stored for "
                "future format-controlled "
                "ablation; it is not the "
                "primary Phase-2 training input."
            ),
            (
                "Exact duplicate samples are "
                "audited but not silently "
                "removed."
            ),
        ],
    }

    max_input_length = int(
        phase_cfg[
            "max_input_length"
        ]
    )

    batch_size = int(
        phase_cfg[
            "tokenize_batch_size"
        ]
    )

    val_fraction = float(
        phase_cfg[
            "validation_fraction"
        ]
    )

    seed = int(
        phase_cfg["seed"]
    )

    preview_rows = int(
        phase_cfg[
            "preview_rows"
        ]
    )

    print(
        "\n[3/7] Converting to canonical records",
        flush=True,
    )

    for logical_dataset in (
        "medmcqa",
        "pubmedqa",
        "flashcards",
    ):
        print(
            f"\n--- {logical_dataset} ---",
            flush=True,
        )

        shards = all_shards[
            logical_dataset
        ]

        current_cfg = (
            dataset_cfg[
                logical_dataset
            ]
        )

        manifest["sources"][
            logical_dataset
        ] = source_manifest(
            shards
        )

        records = (
            convert_shards_to_records(
                logical_dataset=(
                    logical_dataset
                ),
                shards=shards,
                cfg=current_cfg,
            )
        )

        print(
            f"  canonical records: "
            f"{len(records)}",
            flush=True,
        )

        print(
            "  tokenizing paper_text "
            "and normalized_text...",
            flush=True,
        )

        add_token_counts(
            records,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

        filtered, filter_stats = (
            filter_records(
                records,
                max_input_length=(
                    max_input_length
                ),
            )
        )

        print(
            f"  retained after filter: "
            f"{len(filtered)} / "
            f"{len(records)}",
            flush=True,
        )

        # stats is completed after deterministic
        # train/validation split.
        stats = build_dataset_stats(
            logical_dataset=(
                logical_dataset
            ),
            records_before_filter=(
                records
            ),
            records_after_filter=(
                filtered
            ),
            split_sizes={},
            filter_stats=(
                filter_stats
            ),
        )

        prepared = (
            save_prepared_dataset(
                logical_dataset=(
                    logical_dataset
                ),
                records=filtered,
                output_root=(
                    prepared_root
                ),
                validation_fraction=(
                    val_fraction
                ),
                seed=seed,
                preview_rows=(
                    preview_rows
                ),
                stats=stats,
            )
        )

        manifest["prepared"][
            logical_dataset
        ] = prepared

        print(
            f"  train rows      : "
            f"{prepared['train_rows']}",
            flush=True,
        )

        print(
            f"  validation rows : "
            f"{prepared['validation_rows']}",
            flush=True,
        )

        # Release source datasets before moving
        # to the next large dataset.
        del records
        del filtered

    print(
        "\n[4/7] Writing Phase-1 manifest",
        flush=True,
    )

    manifest["status"] = "prepared"

    manifest_path = (
        prepared_root
        / "phase1_manifest.json"
    )

    write_json(
        manifest,
        manifest_path,
    )

    print(
        "\n[5/7] Creating persistent archive",
        flush=True,
    )

    job_id = (
        os.environ.get(
            "SLURM_JOB_ID",
            "manual",
        )
    )

    archive_name = (
        f"phase1_"
        f"{phase_cfg['output_name']}_"
        f"job{job_id}.tar.gz"
    )

    archive_path = (
        archive_dir
        / archive_name
    )

    with tarfile.open(
        archive_path,
        "w:gz",
    ) as tar:
        tar.add(
            prepared_root,
            arcname=(
                phase_cfg[
                    "output_name"
                ]
            ),
        )

    archive_sha256 = (
        sha256_file(
            archive_path
        )
    )

    print(
        f"  archive       : "
        f"{archive_path}",
        flush=True,
    )

    print(
        f"  archive sha256: "
        f"{archive_sha256}",
        flush=True,
    )

    print(
        "\n[6/7] Writing receipt",
        flush=True,
    )

    receipt = {
        "phase": 1,
        "timestamp": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "status": "PASS",
        "job_id": job_id,
        "git_branch": (
            manifest[
                "git_branch"
            ]
        ),
        "git_commit": (
            manifest[
                "git_commit"
            ]
        ),
        "git_status": (
            manifest[
                "git_status"
            ]
        ),
        "archive": str(
            archive_path
        ),
        "archive_size_bytes": (
            archive_path.stat().st_size
        ),
        "archive_sha256": (
            archive_sha256
        ),
        "tokenizer": (
            manifest["tokenizer"]
        ),
        "prepared": (
            manifest["prepared"]
        ),
    }

    receipt_path = (
        project_manifest_dir
        / f"phase1_job{job_id}_receipt.json"
    )

    manifest_copy_path = (
        project_manifest_dir
        / f"phase1_job{job_id}_manifest.json"
    )

    write_json(
        receipt,
        receipt_path,
    )

    shutil.copy2(
        manifest_path,
        manifest_copy_path,
    )

    print(
        "\n[7/7] PASS",
        flush=True,
    )

    print(
        "  MedMCQA / PubMedQA / "
        "Medical Meadow prepared.",
        flush=True,
    )

    print(
        "  No LoRA training was "
        "performed in Phase 1.",
        flush=True,
    )

    print(
        f"  receipt: "
        f"{receipt_path}",
        flush=True,
    )

    print(
        f"  manifest: "
        f"{manifest_copy_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()