from __future__ import annotations

import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path

import numpy as np

from xmodel_kv.hotpotqa import (
    answer_em,
    answer_f1,
)


def normalize(text: str) -> str:
    text = text.lower()

    text = "".join(
        ch
        for ch in text
        if ch not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def gold_containment(
    prediction: str,
    golds: list[str],
) -> float:

    pred = normalize(
        prediction
    )

    for gold in golds:

        g = normalize(
            str(gold)
        )

        if (
            g
            and g in pred
        ):
            return 1.0

    return 0.0


def load_rows(
    path: Path,
) -> list[dict]:

    result = []

    with path.open(
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            result.append(
                json.loads(line)
            )

    return result


def rescore_row(
    row: dict,
) -> dict:

    golds = [
        str(x)
        for x in row[
            "gold_answers"
        ]
    ]

    for key in (
        "source_native",
        "target_native",
        "crosskv_transfer",
    ):

        prediction = row[key][
            "clean"
        ]

        row[key][
            "rescored_em"
        ] = answer_em(
            prediction,
            golds,
        )

        row[key][
            "rescored_f1"
        ] = answer_f1(
            prediction,
            golds,
        )

        row[key][
            "gold_containment"
        ] = gold_containment(
            prediction,
            golds,
        )

    return row


def aggregate(
    rows: list[dict],
) -> dict:

    if not rows:
        return {
            "n": 0,
        }

    result = {
        "n":
            len(rows),

        "mean_context_tokens":
            float(
                np.mean(
                    [
                        x[
                            "context_tokens"
                        ]
                        for x in rows
                    ]
                )
            ),
    }

    for key in (
        "source_native",
        "target_native",
        "crosskv_transfer",
    ):

        result[
            f"{key}_em"
        ] = float(
            np.mean(
                [
                    x[key][
                        "rescored_em"
                    ]
                    for x in rows
                ]
            )
        )

        result[
            f"{key}_f1"
        ] = float(
            np.mean(
                [
                    x[key][
                        "rescored_f1"
                    ]
                    for x in rows
                ]
            )
        )

        result[
            f"{key}_answer_hit"
        ] = float(
            np.mean(
                [
                    x[key][
                        "gold_containment"
                    ]
                    for x in rows
                ]
            )
        )

    # ==================================================
    # Answer-hit based retention
    #
    # This is diagnostic, NOT the official metric.
    # ==================================================

    target_hit_rows = [
        x
        for x in rows
        if (
            x["target_native"][
                "gold_containment"
            ]
            == 1.0
        )
    ]

    target_hits = len(
        target_hit_rows
    )

    retained_hits = sum(
        1
        for x in target_hit_rows
        if (
            x[
                "crosskv_transfer"
            ][
                "gold_containment"
            ]
            == 1.0
        )
    )

    result[
        "target_answer_hits"
    ] = target_hits

    result[
        "crosskv_retained_target_hits"
    ] = retained_hits

    result[
        "answer_hit_retention"
    ] = (
        retained_hits
        / target_hits
        if target_hits
        else None
    )

    # ==================================================
    # Escalation:
    #
    # source does not contain gold,
    # target contains gold.
    # Can transferred KV recover it?
    # ==================================================

    escalation = [
        x
        for x in rows
        if (
            x["source_native"][
                "gold_containment"
            ]
            == 0.0
            and
            x["target_native"][
                "gold_containment"
            ]
            == 1.0
        )
    ]

    escalation_recovered = sum(
        1
        for x in escalation
        if (
            x[
                "crosskv_transfer"
            ][
                "gold_containment"
            ]
            == 1.0
        )
    )

    result[
        "source_miss_target_hit"
    ] = len(
        escalation
    )

    result[
        "source_miss_target_hit_crosskv_hit"
    ] = escalation_recovered

    result[
        "escalation_recovery_rate"
    ] = (
        escalation_recovered
        / len(escalation)
        if escalation
        else None
    )

    # ==================================================
    # Mapper damage:
    #
    # both source + target have gold answer,
    # but CrossKV loses it.
    # ==================================================

    result[
        "source_hit_target_hit_crosskv_miss"
    ] = sum(
        1
        for x in rows
        if (
            x["source_native"][
                "gold_containment"
            ]
            == 1.0
            and
            x["target_native"][
                "gold_containment"
            ]
            == 1.0
            and
            x["crosskv_transfer"][
                "gold_containment"
            ]
            == 0.0
        )
    )

    # ==================================================
    # Continuous F1 retention
    #
    # Only target-F1-positive cases.
    # ==================================================

    positive_target_f1 = [
        x
        for x in rows
        if (
            x["target_native"][
                "rescored_f1"
            ]
            > 0
        )
    ]

    ratios = []

    for x in positive_target_f1:

        target_f1 = (
            x["target_native"][
                "rescored_f1"
            ]
        )

        transfer_f1 = (
            x["crosskv_transfer"][
                "rescored_f1"
            ]
        )

        ratios.append(
            transfer_f1
            / target_f1
        )

    result[
        "target_f1_positive_cases"
    ] = len(
        positive_target_f1
    )

    result[
        "crosskv_over_target_f1_ratio_mean"
    ] = (
        float(
            np.mean(ratios)
        )
        if ratios
        else None
    )

    result[
        "crosskv_over_target_f1_ratio_median"
    ] = (
        float(
            np.median(ratios)
        )
        if ratios
        else None
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--summary",
        required=True,
    )

    args = parser.parse_args()

    rows = load_rows(
        Path(
            args.input
        )
    )

    rows = [
        rescore_row(
            row
        )
        for row in rows
    ]

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    grouped = defaultdict(
        list
    )

    for row in rows:

        grouped[
            row["context_bin"]
        ].append(row)

    summary = {
        "documents":
            len(rows),

        "metric_note": (
            "EM and token F1 are QA metrics. "
            "gold_containment/answer_hit is a "
            "diagnostic metric indicating whether "
            "a normalized gold string occurs inside "
            "the model's cleaned first-line answer."
        ),

        "overall":
            aggregate(
                rows
            ),

        "by_context_bin": {
            key:
                aggregate(value)

            for key, value
            in sorted(
                grouped.items()
            )
        },
    }

    summary_path = Path(
        args.summary
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
