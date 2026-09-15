from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from datasets import Dataset


ANSWER_PREFIX = "The answer is:"


def clean(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def ensure_terminal(question: str) -> str:
    question = clean(question)

    if not question:
        return question

    if question.endswith(("?", ".", ":", "-")):
        return question

    return question + "?"


def normalize_choices(
    choices: Any,
) -> list[str]:
    if choices is None:
        return []

    return [
        clean(choice)
        for choice in list(choices)
    ]


def build_native_prompt(
    row: dict[str, Any],
) -> str:
    dataset = clean(row["dataset"])
    question = ensure_terminal(
        row["question"]
    )

    context = clean(
        row.get("context")
    )

    choices = normalize_choices(
        row.get("choices")
    )

    if dataset == "medmcqa":
        if len(choices) != 4:
            raise ValueError(
                "MedMCQA probe must have 4 choices"
            )

        return (
            f"MCQ: {question}\n"
            "Options:\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n"
            f"{ANSWER_PREFIX}"
        )

    if dataset == "pubmedqa":
        return (
            f"Closed Question: {context}\n"
            f"{question}\n"
            f"{ANSWER_PREFIX}"
        )

    if dataset == "flashcards":
        return (
            f"Open Question: {question}\n"
            f"{ANSWER_PREFIX}"
        )

    raise ValueError(
        f"Unsupported dataset={dataset}"
    )


def build_normalized_prompt(
    row: dict[str, Any],
    *,
    include_context: bool = True,
) -> str:
    question = clean(
        row["question"]
    )

    context = clean(
        row.get("context")
    )

    choices = normalize_choices(
        row.get("choices")
    )

    lines = [
        "Question:",
        question,
    ]

    if (
        include_context
        and context
    ):
        lines.extend(
            [
                "",
                "Context:",
                context,
            ]
        )

    if choices:
        lines.extend(
            [
                "",
                "Choices:",
            ]
        )

        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for idx, choice in enumerate(
            choices
        ):
            lines.append(
                f"{labels[idx]}. {choice}"
            )

    lines.extend(
        [
            "",
            "Answer:",
        ]
    )

    return "\n".join(lines)


def _stratified_order_medmcqa(
    dataset: Dataset,
    *,
    seed: int,
) -> list[int]:
    """
    Round-robin over MedMCQA subjects so that the probe set does
    not accidentally consist mostly of one medical subject.
    """
    rng = random.Random(seed)

    groups: dict[
        str,
        list[int],
    ] = defaultdict(list)

    subjects = dataset["subject"]

    for idx, subject in enumerate(
        subjects
    ):
        key = clean(subject)

        if not key:
            key = "__unknown__"

        groups[key].append(idx)

    keys = list(groups.keys())
    rng.shuffle(keys)

    for key in keys:
        rng.shuffle(groups[key])

    ordered: list[int] = []

    active = True
    cursor = 0

    while active:
        active = False

        for key in keys:
            if cursor < len(
                groups[key]
            ):
                ordered.append(
                    groups[key][cursor]
                )

                active = True

        cursor += 1

    return ordered


def _random_order(
    dataset: Dataset,
    *,
    seed: int,
) -> list[int]:
    indices = list(
        range(len(dataset))
    )

    rng = random.Random(seed)
    rng.shuffle(indices)

    return indices


def build_dataset_probes(
    dataset: Dataset,
    *,
    logical_dataset: str,
    tokenizer,
    samples_per_dataset: int,
    max_prompt_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    if logical_dataset == "medmcqa":
        ordered_indices = (
            _stratified_order_medmcqa(
                dataset,
                seed=seed,
            )
        )
    else:
        ordered_indices = _random_order(
            dataset,
            seed=seed,
        )

    probes: list[
        dict[str, Any]
    ] = []

    selected_base_records = 0

    for idx in ordered_indices:
        if (
            selected_base_records
            >= samples_per_dataset
        ):
            break

        row = dataset[int(idx)]

        native = build_native_prompt(
            row
        )

        normalized = (
            build_normalized_prompt(
                row,
                include_context=True,
            )
        )

        variants = [
            (
                "native",
                "original",
                native,
            ),
            (
                "normalized",
                "original",
                normalized,
            ),
        ]

        # PubMedQA provides a controlled context ablation:
        # exact same question and normalized form,
        # but the supporting context is removed.
        if logical_dataset == "pubmedqa":
            no_context = (
                build_normalized_prompt(
                    row,
                    include_context=False,
                )
            )

            variants.append(
                (
                    "normalized_no_context",
                    "removed",
                    no_context,
                )
            )

        encoded_variants = []

        valid = True

        for (
            variant,
            context_mode,
            prompt,
        ) in variants:
            ids = tokenizer(
                prompt,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )["input_ids"]

            token_count = len(ids)

            if (
                token_count <= 0
                or token_count
                > max_prompt_tokens
            ):
                valid = False
                break

            encoded_variants.append(
                (
                    variant,
                    context_mode,
                    prompt,
                    token_count,
                )
            )

        if not valid:
            continue

        base_record_id = clean(
            row["record_id"]
        )

        if not base_record_id:
            raise RuntimeError(
                "Probe source has no record_id"
            )

        for (
            variant,
            context_mode,
            prompt,
            token_count,
        ) in encoded_variants:
            probe_id = (
                f"{base_record_id}"
                f"::{variant}"
            )

            probes.append(
                {
                    "probe_id": probe_id,
                    "base_record_id": (
                        base_record_id
                    ),
                    "dataset": (
                        logical_dataset
                    ),
                    "domain": clean(
                        row.get("domain")
                    ),
                    "source_format": clean(
                        row.get(
                            "source_format"
                        )
                    ),
                    "subject": clean(
                        row.get("subject")
                    ),
                    "topic": clean(
                        row.get("topic")
                    ),
                    "variant": variant,
                    "context_mode": (
                        context_mode
                    ),
                    "prompt": prompt,
                    "token_count": (
                        token_count
                    ),
                    "source_split": (
                        "validation"
                    ),
                    "answer_excluded": True,
                }
            )

        selected_base_records += 1

    if (
        selected_base_records
        != samples_per_dataset
    ):
        raise RuntimeError(
            f"{logical_dataset}: requested "
            f"{samples_per_dataset} base samples "
            f"but only selected "
            f"{selected_base_records}"
        )

    return probes