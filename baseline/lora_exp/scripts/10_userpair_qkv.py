#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lora_exp.instrument.qkv_capture import PrefillQKVCapture
from lora_exp.model.factory import load_base_model, load_tokenizer
from lora_exp.probe.builder import build_native_prompt, build_normalized_prompt
from lora_exp.train.packing import load_parquet_dataset


TENSOR_TYPES = ["q_attn", "k_cache", "v_cache"]
POOLING_MODES = ["mean_pool", "last_token"]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            candidate = (destination / member.name).resolve()
            if candidate != root and root not in candidate.parents:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
        tar.extractall(destination)


def git_output(args: list[str]) -> str:
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return f"<failed: {result.stderr.strip()}>"
    return result.stdout.strip()


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def subject_indices(dataset, subject: str) -> list[int]:
    return [
        idx
        for idx, value in enumerate(dataset["subject"])
        if clean(value) == subject
    ]


def state_hash(
    record_id: str,
    variant: str,
    adapter_id: str,
) -> str:
    text = f"{record_id}|{variant}|{adapter_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def pooled_arrays(
    features: dict[str, list[torch.Tensor]],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}

    for tensor_type, layers in features.items():
        mean_layers = []
        last_layers = []

        for layer in layers:
            x = layer.float()

            # x: [T, H, D]
            mean_layers.append(
                x.mean(dim=0).to(dtype=torch.float16).numpy()
            )
            last_layers.append(
                x[-1].to(dtype=torch.float16).numpy()
            )

        output[f"{tensor_type}__mean_pool"] = np.stack(
            mean_layers,
            axis=0,
        )
        output[f"{tensor_type}__last_token"] = np.stack(
            last_layers,
            axis=0,
        )

    return output


def mean_head_cosine(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    # a,b: [H,D]
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    numerator = np.sum(a * b, axis=-1)
    denominator = (
        np.linalg.norm(a, axis=-1)
        * np.linalg.norm(b, axis=-1)
    )

    cosine = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(cosine))


def bootstrap_mean(
    values: np.ndarray,
    *,
    replicates: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)

    indices = rng.integers(
        0,
        len(values),
        size=(replicates, len(values)),
    )
    boot = values[indices].mean(axis=1)

    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(
        boot,
        [alpha, 1.0 - alpha],
    )

    return float(values.mean()), float(low), float(high)


def sign_flip_p(
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))

    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(replicates, len(values)),
    )
    null = np.abs((signs * values).mean(axis=1))

    return float(
        (np.count_nonzero(null >= observed) + 1)
        / (replicates + 1)
    )


def independent_permutation_p(
    a: np.ndarray,
    b: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    observed = abs(float(a.mean() - b.mean()))
    combined = np.concatenate([a, b])
    n_a = len(a)

    exceed = 0

    for _ in range(replicates):
        perm = rng.permutation(combined)
        diff = abs(
            float(
                perm[:n_a].mean()
                - perm[n_a:].mean()
            )
        )
        if diff >= observed:
            exceed += 1

    return float((exceed + 1) / (replicates + 1))


def benjamini_hochberg(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def build_probe_records(
    val_ds,
    topics: list[dict[str, Any]],
    tokenizer,
    *,
    records_per_topic: int,
    max_prompt_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for topic_idx, topic in enumerate(topics):
        subject = topic["subject"]
        indices = subject_indices(val_ds, subject)

        rng = random.Random(seed + topic_idx * 1000)
        rng.shuffle(indices)

        selected = 0

        for idx in indices:
            row = val_ds[int(idx)]

            native = build_native_prompt(row)
            normalized = build_normalized_prompt(
                row,
                include_context=True,
            )

            native_ids = tokenizer(
                native,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )["input_ids"]

            normalized_ids = tokenizer(
                normalized,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )["input_ids"]

            if (
                len(native_ids) <= 0
                or len(normalized_ids) <= 0
                or len(native_ids) > max_prompt_tokens
                or len(normalized_ids) > max_prompt_tokens
            ):
                continue

            record_id = clean(row["record_id"])
            if not record_id:
                continue

            records.append(
                {
                    "record_id": record_id,
                    "topic_id": topic["topic_id"],
                    "subject": subject,
                    "native_prompt": native,
                    "normalized_prompt": normalized,
                    "native_tokens": len(native_ids),
                    "normalized_tokens": len(normalized_ids),
                }
            )

            selected += 1
            if selected >= records_per_topic:
                break

        if selected != records_per_topic:
            raise RuntimeError(
                f"{topic['topic_id']} / {subject}: "
                f"needed {records_per_topic}, got {selected}"
            )

    return records


def build_record_pairs(
    records: list[dict[str, Any]],
    topics: list[dict[str, Any]],
    *,
    same_count: int,
    diff_count: int,
) -> list[dict[str, Any]]:
    if len(topics) != 4:
        raise RuntimeError(
            "Final factorial pairing currently requires exactly 4 topics."
        )

    by_topic = {
        topic["topic_id"]: [
            row
            for row in records
            if row["topic_id"] == topic["topic_id"]
        ]
        for topic in topics
    }

    if same_count % 2 != 0:
        raise RuntimeError("same_topic_records_per_topic must be even")

    if diff_count % 3 != 0:
        raise RuntimeError(
            "different_topic_records_per_topic must be divisible by 3"
        )

    pairs: list[dict[str, Any]] = []
    pair_counter = 0

    # --------------------------------------------------------
    # same-topic:
    # first same_count records/topic, adjacent disjoint pairing.
    # --------------------------------------------------------
    for topic in topics:
        topic_id = topic["topic_id"]
        pool = by_topic[topic_id][:same_count]

        for i in range(0, len(pool), 2):
            left = pool[i]
            right = pool[i + 1]

            pairs.append(
                {
                    "pair_id": f"pair{pair_counter:04d}",
                    "topic_relation": "same",
                    "left_record_id": left["record_id"],
                    "right_record_id": right["record_id"],
                    "left_topic_id": left["topic_id"],
                    "right_topic_id": right["topic_id"],
                }
            )
            pair_counter += 1

    # --------------------------------------------------------
    # different-topic:
    # use disjoint second pool.
    # For 4 topics A,B,C,D and 3 groups:
    #   g0: A-B, C-D
    #   g1: A-C, B-D
    #   g2: A-D, B-C
    # Thus all 6 topic pairs are represented equally.
    # --------------------------------------------------------
    ids = [topic["topic_id"] for topic in topics]
    group_size = diff_count // 3

    diff_pools = {}
    for topic_id in ids:
        start = same_count
        stop = same_count + diff_count
        pool = by_topic[topic_id][start:stop]

        diff_pools[topic_id] = [
            pool[g * group_size : (g + 1) * group_size]
            for g in range(3)
        ]

    pair_specs = [
        (ids[0], ids[1], 0),
        (ids[2], ids[3], 0),
        (ids[0], ids[2], 1),
        (ids[1], ids[3], 1),
        (ids[0], ids[3], 2),
        (ids[1], ids[2], 2),
    ]

    for left_topic, right_topic, group_idx in pair_specs:
        left_group = diff_pools[left_topic][group_idx]
        right_group = diff_pools[right_topic][group_idx]

        for left, right in zip(left_group, right_group):
            pairs.append(
                {
                    "pair_id": f"pair{pair_counter:04d}",
                    "topic_relation": "different",
                    "left_record_id": left["record_id"],
                    "right_record_id": right["record_id"],
                    "left_topic_id": left["topic_id"],
                    "right_topic_id": right["topic_id"],
                }
            )
            pair_counter += 1

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/topic_userpair.yaml",
    )
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")

    cfg = read_yaml(PROJECT_ROOT / args.config)["phase3b"]
    model_yaml = read_yaml(
        PROJECT_ROOT / cfg["model_config"]
    )
    model_cfg = model_yaml["model"]
    arch = model_yaml["expected_architecture"]

    output_root = Path(
        os.path.expandvars(cfg["userpair_output_root"])
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    topic_root = Path(
        os.path.expandvars(cfg["topic_adapter_output_root"])
    ).resolve()

    topic_plan = read_json(topic_root / "topic_plan.json")
    topic_eval = read_json(topic_root / "topic_cross_eval.json")

    if topic_eval.get("status") != "PASS":
        raise RuntimeError("topic_cross_eval.json is not PASS")

    topics = topic_plan["topics"]
    if len(topics) != int(cfg["num_topics"]):
        raise RuntimeError("topic_plan topic count mismatch")

    work_dir = Path(os.path.expandvars(args.work_dir)).resolve()
    if work_dir.exists():
        raise RuntimeError(f"work_dir exists: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=False)

    receipt = read_json(PROJECT_ROOT / cfg["phase1_receipt"])
    archive = Path(receipt["archive"])
    extract_root = work_dir / "phase1"
    safe_extract(archive, extract_root)

    dataset_root = extract_root / cfg["phase1_dataset_root"]
    val_ds = load_parquet_dataset(
        dataset_root / cfg["source_dataset"] / "validation.parquet"
    )

    tokenizer = load_tokenizer(model_cfg)

    records = build_probe_records(
        val_ds,
        topics,
        tokenizer,
        records_per_topic=int(cfg["records_per_topic"]),
        max_prompt_tokens=int(cfg["max_prompt_tokens"]),
        seed=int(cfg["seed"]),
    )

    probe_file = PROJECT_ROOT / cfg["userpair_probe_file"]
    probe_file.parent.mkdir(parents=True, exist_ok=True)

    with probe_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    same_count = int(cfg["same_topic_records_per_topic"])
    diff_count = int(cfg["different_topic_records_per_topic"])

    if same_count + diff_count != int(cfg["records_per_topic"]):
        raise RuntimeError(
            "same_topic_records_per_topic + "
            "different_topic_records_per_topic must equal records_per_topic"
        )

    pairs = build_record_pairs(
        records,
        topics,
        same_count=same_count,
        diff_count=diff_count,
    )

    save_json(
        {
            "status": "PASS",
            "topics": topics,
            "records": records,
            "pairs": pairs,
        },
        output_root / "userpair_plan.json",
    )

    # ========================================================
    # Load all topic adapters into one base model.
    # ========================================================

    base = load_base_model(
        model_cfg,
        device="cuda",
        use_cache=True,
    )

    def adapter_path(topic: dict[str, Any]) -> Path:
        return Path(topic["adapter_output"]) / "adapter_final"

    first = topics[0]

    model = PeftModel.from_pretrained(
        base,
        adapter_path(first),
        adapter_name=first["topic_id"],
        is_trainable=False,
    )

    for topic in topics[1:]:
        model.load_adapter(
            adapter_path(topic),
            adapter_name=topic["topic_id"],
            is_trainable=False,
        )

    model.eval()
    model.config.use_cache = True

    capture = PrefillQKVCapture(
        model,
        num_layers=int(arch["num_hidden_layers"]),
        num_q_heads=int(arch["num_attention_heads"]),
        num_kv_heads=int(arch["num_key_value_heads"]),
        head_dim=int(arch["head_dim"]),
    )

    print(f"[Q CAPTURE MODE] {capture.q_mode}")

    feature_dir = output_root / "feature_cache"
    feature_dir.mkdir(parents=True, exist_ok=True)

    record_by_id = {
        row["record_id"]: row
        for row in records
    }

    state_paths: dict[tuple[str, str, str], Path] = {}

    # ========================================================
    # Capture compact prompt-level features.
    # ========================================================

    total_states = (
        len(records)
        * 2
        * len(topics)
    )
    completed = 0

    try:
        for record in records:
            for variant in ["native", "normalized"]:
                prompt = record[f"{variant}_prompt"]

                batch = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                )

                token_count = int(batch["input_ids"].shape[1])
                if token_count > int(cfg["max_prompt_tokens"]):
                    raise RuntimeError(
                        f"{record['record_id']} {variant}: "
                        f"{token_count} > max_prompt_tokens"
                    )

                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")

                for topic in topics:
                    adapter_id = topic["topic_id"]
                    key = (
                        record["record_id"],
                        variant,
                        adapter_id,
                    )

                    file_name = (
                        state_hash(
                            record["record_id"],
                            variant,
                            adapter_id,
                        )
                        + ".npz"
                    )
                    path = feature_dir / file_name
                    state_paths[key] = path

                    if (
                        bool(cfg.get("resume_features", True))
                        and path.exists()
                    ):
                        completed += 1
                        continue

                    model.set_adapter(adapter_id)

                    features = capture.capture(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    arrays = pooled_arrays(features)

                    np.savez_compressed(
                        path,
                        **arrays,
                    )

                    completed += 1
                    if completed % 16 == 0:
                        print(
                            f"[FEATURES] {completed}/{total_states}"
                        )
                        torch.cuda.empty_cache()

                del batch
                del input_ids
                del attention_mask

    finally:
        capture.close()

    if completed != total_states:
        raise RuntimeError(
            f"Feature-state count mismatch: "
            f"{completed} != {total_states}"
        )

    # ========================================================
    # Load feature cache into RAM.
    # ========================================================

    state_cache: dict[
        tuple[str, str, str],
        dict[str, np.ndarray],
    ] = {}

    for key, path in state_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path) as data:
            state_cache[key] = {
                name: data[name]
                for name in data.files
            }

    adapter_ids = [
        topic["topic_id"]
        for topic in topics
    ]

    same_adapter_assignments = [
        (adapter, adapter)
        for adapter in adapter_ids
    ]

    different_adapter_assignments = [
        (a, b)
        for a in adapter_ids
        for b in adapter_ids
        if a != b
    ]

    same_form_assignments = [
        ("native", "native"),
        ("normalized", "normalized"),
    ]

    different_form_assignments = [
        ("native", "normalized"),
        ("normalized", "native"),
    ]

    detail_rows: list[dict[str, Any]] = []

    # ========================================================
    # Factorial pairwise similarity:
    #   topic_relation   same / different
    #   adapter_relation same / different
    #   form_relation    same / different
    #
    # Each row is one independent semantic record-pair
    # observation after averaging nuisance adapter/form identities.
    # ========================================================

    for pair_idx, pair in enumerate(pairs, start=1):
        left_id = pair["left_record_id"]
        right_id = pair["right_record_id"]

        for adapter_relation, adapter_assignments in [
            ("same", same_adapter_assignments),
            ("different", different_adapter_assignments),
        ]:
            for form_relation, form_assignments in [
                ("same", same_form_assignments),
                ("different", different_form_assignments),
            ]:
                for tensor_type in TENSOR_TYPES:
                    for pooling in POOLING_MODES:
                        array_key = f"{tensor_type}__{pooling}"

                        for layer in range(
                            int(arch["num_hidden_layers"])
                        ):
                            values = []

                            for (
                                left_adapter,
                                right_adapter,
                            ) in adapter_assignments:
                                for (
                                    left_form,
                                    right_form,
                                ) in form_assignments:
                                    left = state_cache[
                                        (
                                            left_id,
                                            left_form,
                                            left_adapter,
                                        )
                                    ][array_key][layer]

                                    right = state_cache[
                                        (
                                            right_id,
                                            right_form,
                                            right_adapter,
                                        )
                                    ][array_key][layer]

                                    values.append(
                                        mean_head_cosine(
                                            left,
                                            right,
                                        )
                                    )

                            detail_rows.append(
                                {
                                    "pair_id": pair["pair_id"],
                                    "topic_relation": pair["topic_relation"],
                                    "adapter_relation": adapter_relation,
                                    "form_relation": form_relation,
                                    "tensor_type": tensor_type,
                                    "pooling": pooling,
                                    "layer": layer,
                                    "cosine_mean": float(
                                        np.mean(values)
                                    ),
                                    "left_topic_id": pair["left_topic_id"],
                                    "right_topic_id": pair["right_topic_id"],
                                }
                            )

        if pair_idx % 8 == 0:
            print(
                f"[PAIR ANALYSIS] {pair_idx}/{len(pairs)}"
            )

    detail = pd.DataFrame(detail_rows)
    detail_path = output_root / "userpair_detail.csv"
    detail.to_csv(detail_path, index=False)

    # ========================================================
    # Overall pair-level cells:
    # first average 36 layers within each semantic pair.
    # ========================================================

    overall_pair = (
        detail.groupby(
            [
                "pair_id",
                "topic_relation",
                "adapter_relation",
                "form_relation",
                "tensor_type",
                "pooling",
            ],
            observed=True,
        )["cosine_mean"]
        .mean()
        .reset_index()
    )

    boot_reps = int(cfg["bootstrap_replicates"])
    perm_reps = int(cfg["permutation_replicates"])
    ci_level = float(cfg["ci_level"])
    stats_seed = int(cfg["stats_seed"])

    cell_rows: list[dict[str, Any]] = []
    counter = 0

    for keys, group in overall_pair.groupby(
        [
            "topic_relation",
            "adapter_relation",
            "form_relation",
            "tensor_type",
            "pooling",
        ],
        observed=True,
    ):
        (
            topic_relation,
            adapter_relation,
            form_relation,
            tensor_type,
            pooling,
        ) = keys

        values = group["cosine_mean"].to_numpy(dtype=float)
        mean, low, high = bootstrap_mean(
            values,
            replicates=boot_reps,
            ci_level=ci_level,
            rng=np.random.default_rng(stats_seed + counter),
        )
        counter += 1

        cell_rows.append(
            {
                "topic_relation": topic_relation,
                "adapter_relation": adapter_relation,
                "form_relation": form_relation,
                "tensor_type": tensor_type,
                "pooling": pooling,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_pairs": len(values),
            }
        )

    cells = pd.DataFrame(cell_rows)
    cells.to_csv(
        output_root / "userpair_cells.csv",
        index=False,
    )

    # ========================================================
    # Main effects
    # Positive means:
    #   topic  : same-topic > different-topic
    #   adapter: same-LoRA  > different-LoRA
    #   form   : same-form  > different-form
    # ========================================================

    effect_rows: list[dict[str, Any]] = []
    effect_counter = 10000

    for tensor_type in TENSOR_TYPES:
        for pooling in POOLING_MODES:
            work = overall_pair[
                (overall_pair["tensor_type"] == tensor_type)
                & (overall_pair["pooling"] == pooling)
            ].copy()

            # ------------------------------------------------
            # Adapter effect: paired within semantic pair,
            # after averaging form relation.
            # ------------------------------------------------
            adapter_pair = (
                work.groupby(
                    [
                        "pair_id",
                        "topic_relation",
                        "adapter_relation",
                    ],
                    observed=True,
                )["cosine_mean"]
                .mean()
                .reset_index()
                .pivot(
                    index=["pair_id", "topic_relation"],
                    columns="adapter_relation",
                    values="cosine_mean",
                )
                .dropna()
                .reset_index()
            )

            adapter_diff = (
                adapter_pair["same"]
                - adapter_pair["different"]
            ).to_numpy(dtype=float)

            mean, low, high = bootstrap_mean(
                adapter_diff,
                replicates=boot_reps,
                ci_level=ci_level,
                rng=np.random.default_rng(
                    stats_seed + effect_counter
                ),
            )
            p_value = sign_flip_p(
                adapter_diff,
                replicates=perm_reps,
                rng=np.random.default_rng(
                    stats_seed + effect_counter + 1
                ),
            )

            effect_rows.append(
                {
                    "effect": "same_vs_different_lora",
                    "tensor_type": tensor_type,
                    "pooling": pooling,
                    "mean_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p_value,
                    "n_pairs": len(adapter_diff),
                }
            )
            effect_counter += 2

            # ------------------------------------------------
            # Form effect: paired within semantic pair,
            # after averaging adapter relation.
            # ------------------------------------------------
            form_pair = (
                work.groupby(
                    [
                        "pair_id",
                        "topic_relation",
                        "form_relation",
                    ],
                    observed=True,
                )["cosine_mean"]
                .mean()
                .reset_index()
                .pivot(
                    index=["pair_id", "topic_relation"],
                    columns="form_relation",
                    values="cosine_mean",
                )
                .dropna()
                .reset_index()
            )

            form_diff = (
                form_pair["same"]
                - form_pair["different"]
            ).to_numpy(dtype=float)

            mean, low, high = bootstrap_mean(
                form_diff,
                replicates=boot_reps,
                ci_level=ci_level,
                rng=np.random.default_rng(
                    stats_seed + effect_counter
                ),
            )
            p_value = sign_flip_p(
                form_diff,
                replicates=perm_reps,
                rng=np.random.default_rng(
                    stats_seed + effect_counter + 1
                ),
            )

            effect_rows.append(
                {
                    "effect": "same_vs_different_form",
                    "tensor_type": tensor_type,
                    "pooling": pooling,
                    "mean_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p_value,
                    "n_pairs": len(form_diff),
                }
            )
            effect_counter += 2

            # ------------------------------------------------
            # Topic effect:
            # same-topic and different-topic use disjoint
            # semantic record pools by construction.
            # ------------------------------------------------
            topic_pair = (
                work.groupby(
                    [
                        "pair_id",
                        "topic_relation",
                    ],
                    observed=True,
                )["cosine_mean"]
                .mean()
                .reset_index()
            )

            same_values = topic_pair[
                topic_pair["topic_relation"] == "same"
            ]["cosine_mean"].to_numpy(dtype=float)

            diff_values = topic_pair[
                topic_pair["topic_relation"] == "different"
            ]["cosine_mean"].to_numpy(dtype=float)

            observed_diff = float(
                same_values.mean()
                - diff_values.mean()
            )

            # Independent bootstrap for difference of means.
            rng = np.random.default_rng(
                stats_seed + effect_counter
            )
            same_idx = rng.integers(
                0,
                len(same_values),
                size=(boot_reps, len(same_values)),
            )
            diff_idx = rng.integers(
                0,
                len(diff_values),
                size=(boot_reps, len(diff_values)),
            )

            boot_diff = (
                same_values[same_idx].mean(axis=1)
                - diff_values[diff_idx].mean(axis=1)
            )

            alpha = (1.0 - ci_level) / 2.0
            low, high = np.quantile(
                boot_diff,
                [alpha, 1.0 - alpha],
            )

            p_value = independent_permutation_p(
                same_values,
                diff_values,
                replicates=perm_reps,
                rng=np.random.default_rng(
                    stats_seed + effect_counter + 1
                ),
            )

            effect_rows.append(
                {
                    "effect": "same_vs_different_topic",
                    "tensor_type": tensor_type,
                    "pooling": pooling,
                    "mean_difference": observed_diff,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "p_value": p_value,
                    "n_pairs": int(
                        len(same_values)
                        + len(diff_values)
                    ),
                }
            )
            effect_counter += 2

    effects = pd.DataFrame(effect_rows)
    effects["q_value_bh"] = benjamini_hochberg(
        effects["p_value"].tolist()
    )

    effects.to_csv(
        output_root / "userpair_main_effects.csv",
        index=False,
    )

    # ========================================================
    # Layer-wise topic curves, averaging adapter/form relation.
    # ========================================================

    layer_topic = (
        detail.groupby(
            [
                "topic_relation",
                "tensor_type",
                "pooling",
                "layer",
            ],
            observed=True,
        )["cosine_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    layer_topic["sem"] = (
        layer_topic["std"]
        / np.sqrt(layer_topic["count"])
    )
    layer_topic["ci95_low"] = (
        layer_topic["mean"]
        - 1.96 * layer_topic["sem"]
    )
    layer_topic["ci95_high"] = (
        layer_topic["mean"]
        + 1.96 * layer_topic["sem"]
    )

    layer_topic.to_csv(
        output_root / "userpair_layer_topic.csv",
        index=False,
    )

    # ========================================================
    # Final one-page figure.
    # ========================================================

    figure_path = output_root / "phase3b_userpair_overview.png"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
    )

    # A/B: 8 factorial cells for mean_pool / last_token.
    relation_order = list(
        itertools.product(
            ["same", "different"],
            ["same", "different"],
            ["same", "different"],
        )
    )

    def relation_label(item):
        topic_rel, adapter_rel, form_rel = item
        return (
            f"T:{topic_rel[0].upper()} "
            f"L:{adapter_rel[0].upper()} "
            f"F:{form_rel[0].upper()}"
        )

    x = np.arange(len(relation_order))
    width = 0.25

    for ax, pooling, title in [
        (
            axes[0, 0],
            "mean_pool",
            "A. Prompt-level mean-pooled Q/K/V similarity",
        ),
        (
            axes[0, 1],
            "last_token",
            "B. Last-token Q/K/V similarity",
        ),
    ]:
        for tensor_idx, tensor_type in enumerate(TENSOR_TYPES):
            means = []
            lows = []
            highs = []

            for relation in relation_order:
                topic_rel, adapter_rel, form_rel = relation

                row = cells[
                    (cells["topic_relation"] == topic_rel)
                    & (cells["adapter_relation"] == adapter_rel)
                    & (cells["form_relation"] == form_rel)
                    & (cells["tensor_type"] == tensor_type)
                    & (cells["pooling"] == pooling)
                ].iloc[0]

                means.append(float(row["mean"]))
                lows.append(float(row["mean"] - row["ci_low"]))
                highs.append(float(row["ci_high"] - row["mean"]))

            ax.bar(
                x + (tensor_idx - 1) * width,
                means,
                width=width,
                yerr=np.vstack([lows, highs]),
                capsize=2,
                label=tensor_type,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [relation_label(r) for r in relation_order],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel("Cosine similarity")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

    # C: main effects, mean_pool.
    ax = axes[1, 0]

    effect_order = [
        "same_vs_different_topic",
        "same_vs_different_lora",
        "same_vs_different_form",
    ]
    effect_labels = [
        "Same topic - Different topic",
        "Same LoRA - Different LoRA",
        "Same form - Different form",
    ]

    x_eff = np.arange(len(effect_order))

    for tensor_idx, tensor_type in enumerate(TENSOR_TYPES):
        means = []
        lows = []
        highs = []

        for effect in effect_order:
            row = effects[
                (effects["effect"] == effect)
                & (effects["tensor_type"] == tensor_type)
                & (effects["pooling"] == "mean_pool")
            ].iloc[0]

            means.append(
                1000.0 * float(row["mean_difference"])
            )
            lows.append(
                1000.0
                * float(
                    row["mean_difference"]
                    - row["ci_low"]
                )
            )
            highs.append(
                1000.0
                * float(
                    row["ci_high"]
                    - row["mean_difference"]
                )
            )

        ax.bar(
            x_eff + (tensor_idx - 1) * width,
            means,
            width=width,
            yerr=np.vstack([lows, highs]),
            capsize=3,
            label=tensor_type,
        )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x_eff)
    ax.set_xticklabels(effect_labels, rotation=12, ha="right")
    ax.set_ylabel("Δ cosine similarity × 10³")
    ax.set_title("C. Factor main effects (mean-pooled)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    # D: layer-wise same-topic vs different-topic mean_pool.
    ax = axes[1, 1]

    for tensor_type in TENSOR_TYPES:
        for topic_relation, linestyle in [
            ("same", "-"),
            ("different", "--"),
        ]:
            group = layer_topic[
                (layer_topic["tensor_type"] == tensor_type)
                & (layer_topic["pooling"] == "mean_pool")
                & (layer_topic["topic_relation"] == topic_relation)
            ].sort_values("layer")

            line = ax.plot(
                group["layer"],
                group["mean"],
                linestyle=linestyle,
                label=f"{tensor_type}/{topic_relation}",
            )[0]

            ax.fill_between(
                group["layer"].to_numpy(),
                group["ci95_low"].to_numpy(),
                group["ci95_high"].to_numpy(),
                alpha=0.10,
            )

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("D. Same-topic vs different-topic by layer")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Phase 3B — Cross-user Q/K/V commonality: "
        "Topic × LoRA × Prompt Form",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ========================================================
    # Human-readable summary.
    # ========================================================

    summary_lines = [
        "PHASE 3B USER-PAIR FACTORIAL RESULTS",
        "=" * 80,
        (
            "Interpretation scope: prompt-level pooled/last-token "
            "representation commonality across DIFFERENT user queries."
        ),
        (
            "This is NOT direct token-position-aligned KV-cache reuse "
            "or end-to-end quality preservation."
        ),
        "",
        "[MAIN EFFECTS]",
    ]

    for _, row in effects.sort_values(
        ["pooling", "effect", "tensor_type"]
    ).iterrows():
        summary_lines.append(
            f"{row['pooling']:10s} | "
            f"{row['effect']:28s} | "
            f"{row['tensor_type']:8s} | "
            f"diff={row['mean_difference']:+.6f} "
            f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
            f"p={row['p_value']:.6g} | "
            f"q={row['q_value_bh']:.6g}"
        )

    key_results_path = output_root / "phase3b_key_results.txt"
    key_results_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "status": "PASS",
        "git_branch": git_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status": git_output(["git", "status", "--short"]),
        "q_capture_mode": capture.q_mode,
        "num_topics": len(topics),
        "topics": topics,
        "num_probe_records": len(records),
        "num_record_pairs": len(pairs),
        "num_feature_states": total_states,
        "interpretation_scope": (
            "cross-query prompt-level Q/K/V representation similarity "
            "after token pooling; not direct cache reuse"
        ),
        "outputs": {
            "probe_file": str(probe_file),
            "detail": str(detail_path),
            "cells": str(output_root / "userpair_cells.csv"),
            "main_effects": str(
                output_root / "userpair_main_effects.csv"
            ),
            "layer_topic": str(
                output_root / "userpair_layer_topic.csv"
            ),
            "figure": str(figure_path),
            "key_results": str(key_results_path),
        },
    }

    save_json(
        manifest,
        output_root / "phase3b_manifest.json",
    )

    print("\n" + "=" * 80)
    print("PHASE 3B USER-PAIR CHARACTERIZATION PASS")
    print(f"figure={figure_path}")
    print(f"effects={output_root / 'userpair_main_effects.csv'}")
    print(f"summary={key_results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
