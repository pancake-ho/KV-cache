#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel

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

from lora_exp.analysis.similarity import (
    compare_feature_sets,
)

from lora_exp.instrument.qkv_capture import (
    PrefillQKVCapture,
)

from lora_exp.model.factory import (
    load_base_model,
    load_tokenizer,
)


CSV_FIELDS = [
    "probe_id",
    "base_record_id",
    "dataset",
    "domain",
    "source_format",
    "subject",
    "topic",
    "variant",
    "context_mode",
    "token_count",

    "pair_a",
    "pair_b",

    "tensor_type",
    "layer",

    "num_vectors",
    "seq_len",
    "num_heads",

    "cosine_mean",
    "cosine_median",
    "cosine_std",
    "cosine_p05",
    "cosine_p95",
    "fraction_cos_ge_095",
    "fraction_cos_ge_099",
    "global_cosine",
    "symmetric_rel_l2",

    "delta_global_cosine",
    "delta_norm_ratio_a",
    "delta_norm_ratio_b",
    "delta_distance_vs_base",
]


def read_yaml(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def git_output(args):
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return (
            f"<failed: "
            f"{result.stderr.strip()}>"
        )

    return result.stdout.strip()


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

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. "
            "CPU fallback forbidden."
        )

    cfg = read_yaml(
        PROJECT_ROOT
        / args.config
    )["phase3"]

    model_yaml = read_yaml(
        PROJECT_ROOT
        / args.model_config
    )

    model_cfg = model_yaml["model"]

    arch = model_yaml[
        "expected_architecture"
    ]

    probe_path = (
        PROJECT_ROOT
        / cfg["probe_file"]
    )

    probes = load_jsonl(
        probe_path
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

    detail_path = (
        output_root
        / "qkv_detail.csv"
    )

    completed_path = (
        output_root
        / "completed_probes.txt"
    )

    manifest_path = (
        output_root
        / "phase3_manifest.json"
    )

    completed = set()

    if (
        cfg.get(
            "resume",
            True,
        )
        and completed_path.exists()
    ):
        completed = {
            line.strip()
            for line in completed_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }

    tokenizer = load_tokenizer(
        model_cfg
    )

    base_model = load_base_model(
        model_cfg,
        device="cuda",
        use_cache=True,
    )

    adapters = list(
        cfg["adapters"]
    )

    adapter_seed = int(
        cfg["adapter_seed"]
    )

    phase2_root = Path(
        os.path.expandvars(
            cfg[
                "phase2_output_root"
            ]
        )
    ).resolve()

    adapter_paths = {
        name: (
            phase2_root
            / name
            / f"seed{adapter_seed}"
            / "adapter_final"
        )
        for name in adapters
    }

    for name, path in (
        adapter_paths.items()
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    first_adapter = adapters[0]

    model = PeftModel.from_pretrained(
        base_model,
        adapter_paths[
            first_adapter
        ],
        adapter_name=first_adapter,
        is_trainable=False,
    )

    for name in adapters[1:]:
        model.load_adapter(
            adapter_paths[name],
            adapter_name=name,
            is_trainable=False,
        )

    model.eval()
    model.config.use_cache = True

    capture = PrefillQKVCapture(
        model,
        num_layers=int(
            arch[
                "num_hidden_layers"
            ]
        ),
        num_q_heads=int(
            arch[
                "num_attention_heads"
            ]
        ),
        num_kv_heads=int(
            arch[
                "num_key_value_heads"
            ]
        ),
        head_dim=int(
            arch["head_dim"]
        ),
    )

    print(
        f"[Q CAPTURE MODE] "
        f"{capture.q_mode}"
    )

    model_variants = [
        "base"
    ] + adapters

    pair_list = list(
        itertools.combinations(
            model_variants,
            2,
        )
    )

    manifest = {
        "status": "RUNNING",
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
        "model": model_cfg,
        "architecture": arch,
        "adapter_seed": (
            adapter_seed
        ),
        "adapters": adapters,
        "adapter_paths": {
            k: str(v)
            for k, v
            in adapter_paths.items()
        },
        "probe_file": str(
            probe_path
        ),
        "total_probes": len(
            probes
        ),
        "q_capture_mode": (
            capture.q_mode
        ),
        "raw_tensor_saved": False,
    }

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

    write_header = (
        not detail_path.exists()
        or detail_path.stat().st_size == 0
    )

    detail_file = detail_path.open(
        "a",
        newline="",
        encoding="utf-8",
    )

    writer = csv.DictWriter(
        detail_file,
        fieldnames=CSV_FIELDS,
    )

    if write_header:
        writer.writeheader()

    try:
        for probe_idx, probe in enumerate(
            probes,
            start=1,
        ):
            probe_id = probe[
                "probe_id"
            ]

            if probe_id in completed:
                print(
                    f"[SKIP] {probe_id}"
                )
                continue

            batch = tokenizer(
                probe["prompt"],
                return_tensors="pt",
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )

            token_count = int(
                batch[
                    "input_ids"
                ].shape[1]
            )

            if (
                token_count
                > int(
                    cfg[
                        "max_prompt_tokens"
                    ]
                )
            ):
                raise RuntimeError(
                    f"{probe_id}: "
                    f"{token_count} tokens"
                )

            input_ids = (
                batch["input_ids"]
                .to("cuda")
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ]
                .to("cuda")
            )

            features = {}

            # ------------------------------
            # Base
            # ------------------------------
            with model.disable_adapter():
                features["base"] = (
                    capture.capture(
                        input_ids=input_ids,
                        attention_mask=(
                            attention_mask
                        ),
                    )
                )

            # ------------------------------
            # LoRA variants
            # ------------------------------
            for adapter_name in adapters:
                model.set_adapter(
                    adapter_name
                )

                features[
                    adapter_name
                ] = capture.capture(
                    input_ids=input_ids,
                    attention_mask=(
                        attention_mask
                    ),
                )

            # ------------------------------
            # Pairwise comparison
            # ------------------------------
            for pair_a, pair_b in (
                pair_list
            ):
                use_delta = (
                    pair_a != "base"
                    and pair_b != "base"
                )

                rows = (
                    compare_feature_sets(
                        features[
                            pair_a
                        ],
                        features[
                            pair_b
                        ],
                        base_features=(
                            features["base"]
                            if use_delta
                            else None
                        ),
                    )
                )

                for metric_row in rows:
                    out = {
                        "probe_id": (
                            probe_id
                        ),
                        "base_record_id": (
                            probe[
                                "base_record_id"
                            ]
                        ),
                        "dataset": (
                            probe["dataset"]
                        ),
                        "domain": (
                            probe["domain"]
                        ),
                        "source_format": (
                            probe[
                                "source_format"
                            ]
                        ),
                        "subject": (
                            probe["subject"]
                        ),
                        "topic": (
                            probe["topic"]
                        ),
                        "variant": (
                            probe["variant"]
                        ),
                        "context_mode": (
                            probe[
                                "context_mode"
                            ]
                        ),
                        "token_count": (
                            token_count
                        ),
                        "pair_a": pair_a,
                        "pair_b": pair_b,
                    }

                    out.update(
                        metric_row
                    )

                    writer.writerow(out)

            detail_file.flush()

            with completed_path.open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    probe_id + "\n"
                )

            completed.add(
                probe_id
            )

            del features
            del batch
            del input_ids
            del attention_mask

            if probe_idx % 8 == 0:
                torch.cuda.empty_cache()

            print(
                f"[{probe_idx:4d}/"
                f"{len(probes):4d}] "
                f"PASS {probe_id}"
            )

    finally:
        capture.close()
        detail_file.close()

    manifest[
        "status"
    ] = "PASS"

    manifest[
        "completed_probes"
    ] = len(completed)

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
        "\n===================================="
    )
    print(
        "PHASE 3 QKV CHARACTERIZATION PASS"
    )
    print(
        f"detail={detail_path}"
    )
    print(
        "===================================="
    )


if __name__ == "__main__":
    main()