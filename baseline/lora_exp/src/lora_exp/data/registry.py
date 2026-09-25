from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


@dataclass
class LoadedShard:
    logical_dataset: str
    repo_id: str
    revision: str
    config_name: str
    split_name: str
    dataset: Dataset


def resolve_dataset_revision(repo_id: str) -> str:
    """
    Resolve HF 'main' to an immutable commit SHA.
    The resolved SHA is stored in the Phase-1 manifest.
    """
    info = HfApi().dataset_info(repo_id)
    sha = getattr(info, "sha", None)

    if not sha:
        raise RuntimeError(
            f"Could not resolve dataset revision for {repo_id}"
        )

    return str(sha)


def _load(
    *,
    repo_id: str,
    revision: str,
    config_name: str | None,
    split: str,
    cache_dir: str | Path,
    trust_remote_code: bool = False,
) -> Dataset:
    """
    Load one Hugging Face dataset split at an immutable revision.

    Remote dataset code is disabled by default.
    It must be explicitly enabled from the experiment configuration.
    """

    # Do not accidentally enable arbitrary remote dataset code.
    trusted_remote_code_repos = {
        "bigbio/pubmed_qa",
    }

    if trust_remote_code and repo_id not in trusted_remote_code_repos:
        raise RuntimeError(
            "trust_remote_code=True was requested for an "
            f"unapproved dataset repository: {repo_id}"
        )

    kwargs: dict[str, Any] = {
        "path": repo_id,
        "split": split,
        "revision": revision,
        "cache_dir": str(cache_dir),
        "trust_remote_code": trust_remote_code,
    }

    if config_name is not None:
        kwargs["name"] = config_name

    return load_dataset(**kwargs)


def load_medmcqa(
    cfg: dict[str, Any],
    *,
    cache_dir: str | Path,
) -> list[LoadedShard]:
    repo_id = cfg["repo_id"]
    revision = resolve_dataset_revision(repo_id)

    ds = _load(
        repo_id=repo_id,
        revision=revision,
        config_name=cfg.get("config_name"),
        split=cfg["source_split"],
        cache_dir=cache_dir,
    )

    limit = cfg.get("sample_limit")

    if limit is not None:
        if len(ds) < int(limit):
            raise RuntimeError(
                f"MedMCQA source has only {len(ds)} rows, "
                f"requested {limit}"
            )

        ds = ds.select(range(int(limit)))

    expected = int(cfg["expected_selected_rows"])

    if len(ds) != expected:
        raise RuntimeError(
            "MedMCQA source row mismatch: "
            f"expected={expected}, actual={len(ds)}"
        )

    return [
        LoadedShard(
            logical_dataset="medmcqa",
            repo_id=repo_id,
            revision=revision,
            config_name=cfg.get("config_name") or "default",
            split_name=cfg["source_split"],
            dataset=ds,
        )
    ]


def load_pubmedqa(
    cfg: dict[str, Any],
    *,
    cache_dir: str | Path,
) -> list[LoadedShard]:
    """
    Paper-faithful composition:

    PQA-L fold0:
      train + validation -> expected 500 rows

    PQA-A:
      first 50,000 rows of train
    """
    repo_id = cfg["repo_id"]
    revision = resolve_dataset_revision(repo_id)

    shards: list[LoadedShard] = []

    labeled_total = 0

    for split_name in cfg["labeled_splits"]:
        ds = _load(
            repo_id=repo_id,
            revision=revision,
            config_name=cfg["labeled_config"],
            split=split_name,
            cache_dir=cache_dir,
            trust_remote_code=bool(
                cfg.get("trust_remote_code", False)
            ),
        )

        labeled_total += len(ds)

        shards.append(
            LoadedShard(
                logical_dataset="pubmedqa",
                repo_id=repo_id,
                revision=revision,
                config_name=cfg["labeled_config"],
                split_name=split_name,
                dataset=ds,
            )
        )

    expected_labeled = int(
        cfg["expected_labeled_rows"]
    )

    if labeled_total != expected_labeled:
        raise RuntimeError(
            "PubMedQA labeled fold0 row mismatch: "
            f"expected={expected_labeled}, "
            f"actual={labeled_total}. "
            "Do not silently substitute another PQA-L split."
        )

    artificial_limit = int(
        cfg["artificial_limit"]
    )

    # Use split slicing to match the original paper code's
    # train[:50000] behavior.
    artificial_split = (
        f"{cfg['artificial_split']}"
        f"[:{artificial_limit}]"
    )

    artificial_ds = _load(
        repo_id=repo_id,
        revision=revision,
        config_name=cfg["artificial_config"],
        split=artificial_split,
        cache_dir=cache_dir,
        trust_remote_code=bool(
            cfg.get("trust_remote_code", False)
        ),
    )

    expected_artificial = int(
        cfg["expected_artificial_rows"]
    )

    if len(artificial_ds) != expected_artificial:
        raise RuntimeError(
            "PubMedQA artificial row mismatch: "
            f"expected={expected_artificial}, "
            f"actual={len(artificial_ds)}"
        )

    shards.append(
        LoadedShard(
            logical_dataset="pubmedqa",
            repo_id=repo_id,
            revision=revision,
            config_name=cfg["artificial_config"],
            split_name=artificial_split,
            dataset=artificial_ds,
        )
    )

    return shards


def load_flashcards(
    cfg: dict[str, Any],
    *,
    cache_dir: str | Path,
) -> list[LoadedShard]:
    repo_id = cfg["repo_id"]
    revision = resolve_dataset_revision(repo_id)

    limit = int(cfg["take_first"])

    split = f"{cfg['source_split']}[:{limit}]"

    ds = _load(
        repo_id=repo_id,
        revision=revision,
        config_name=cfg.get("config_name"),
        split=split,
        cache_dir=cache_dir,
    )

    expected = int(cfg["expected_selected_rows"])

    if len(ds) != expected:
        raise RuntimeError(
            "Medical Meadow row mismatch: "
            f"expected={expected}, actual={len(ds)}"
        )

    return [
        LoadedShard(
            logical_dataset="flashcards",
            repo_id=repo_id,
            revision=revision,
            config_name=cfg.get("config_name") or "default",
            split_name=split,
            dataset=ds,
        )
    ]