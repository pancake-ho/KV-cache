#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path

import yaml
from transformers import AutoTokenizer

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

from lora_exp.probe.builder import (
    build_dataset_probes,
)
from lora_exp.train.packing import (
    load_parquet_dataset,
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
                    "Unsafe archive path: "
                    f"{member.name}"
                )

        tar.extractall(
            destination
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/characterize.yaml"
        ),
    )

    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
    )

    parser.add_argument(
        "--work-dir",
        required=True,
    )

    args = parser.parse_args()

    cfg = read_yaml(
        PROJECT_ROOT
        / args.config
    )["phase3"]

    model_cfg = read_yaml(
        PROJECT_ROOT
        / args.model_config
    )["model"]

    work_dir = Path(
        os.path.expandvars(
            args.work_dir
        )
    ).resolve()

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

    if receipt.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Phase-1 receipt is not PASS"
        )

    archive = Path(
        receipt["archive"]
    )

    if not archive.exists():
        raise FileNotFoundError(
            archive
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
            "phase1_dataset_root"
        ]
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_cfg["model_id"],
            revision=model_cfg[
                "revision"
            ],
            trust_remote_code=(
                model_cfg.get(
                    "trust_remote_code",
                    False,
                )
            ),
        )
    )

    all_probes = []

    datasets = [
        "medmcqa",
        "pubmedqa",
        "flashcards",
    ]

    for idx, name in enumerate(
        datasets
    ):
        val_path = (
            dataset_root
            / name
            / "validation.parquet"
        )

        val_dataset = (
            load_parquet_dataset(
                val_path
            )
        )

        probes = (
            build_dataset_probes(
                val_dataset,
                logical_dataset=name,
                tokenizer=tokenizer,
                samples_per_dataset=int(
                    cfg[
                        "samples_per_dataset"
                    ]
                ),
                max_prompt_tokens=int(
                    cfg[
                        "max_prompt_tokens"
                    ]
                ),
                seed=(
                    int(
                        cfg["probe_seed"]
                    )
                    + idx * 1000
                ),
            )
        )

        all_probes.extend(
            probes
        )

        print(
            f"{name:12s}: "
            f"{len(probes)} probes"
        )

    probe_path = (
        PROJECT_ROOT
        / cfg["probe_file"]
    )

    probe_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with probe_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for probe in all_probes:
            f.write(
                json.dumps(
                    probe,
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "status": "PASS",
        "probe_seed": (
            cfg["probe_seed"]
        ),
        "samples_per_dataset": (
            cfg[
                "samples_per_dataset"
            ]
        ),
        "total_probes": len(
            all_probes
        ),
        "dataset_counts": dict(
            Counter(
                p["dataset"]
                for p in all_probes
            )
        ),
        "variant_counts": dict(
            Counter(
                p["variant"]
                for p in all_probes
            )
        ),
        "answer_excluded": True,
        "probe_file": str(
            probe_path
        ),
    }

    manifest_path = (
        PROJECT_ROOT
        / cfg["probe_manifest"]
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n[PASS] Controlled probe set built."
    )
    print(
        f"probe_file={probe_path}"
    )
    print(
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()