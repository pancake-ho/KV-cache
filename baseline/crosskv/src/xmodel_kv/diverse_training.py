from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_screen_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in paths:
        with path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in rows:
                    raise ValueError(f"duplicate screening row: {row['id']}")
                rows[row["id"]] = row
    return rows


def select_eligible_rows(
    manifest_rows: list[dict[str, Any]],
    screen_rows: dict[str, dict[str, Any]],
    *,
    limits: dict[str, int],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected = {split: [] for split in limits}
    available = {split: 0 for split in limits}
    for row in manifest_rows:
        split = row["split"]
        if split not in limits:
            continue
        screen = screen_rows.get(row["id"])
        if not screen or not screen.get("eligible", False):
            continue
        available[split] += 1
        if len(selected[split]) < limits[split]:
            combined = dict(row)
            combined["screen"] = screen
            selected[split].append(combined)
    shortages = {
        split: {"requested": limits[split], "available": available[split]}
        for split in limits
        if len(selected[split]) < limits[split]
    }
    if shortages:
        raise ValueError(f"eligible split shortages: {shortages}")
    audit = {
        "selected": {split: len(rows) for split, rows in selected.items()},
        "available": available,
        "direct_reuse": {
            split: {
                "target_accuracy": sum(
                    row["screen"]["direct_reuse"]["target"]["compatible_em"]
                    for row in rows
                )
                / len(rows),
                "source_accuracy": sum(
                    row["screen"]["direct_reuse"]["source"]["compatible_em"]
                    for row in rows
                )
                / len(rows),
            }
            for split, rows in selected.items()
        },
    }
    return selected, audit
