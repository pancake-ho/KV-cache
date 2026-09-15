from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """
    실행 결과용 JSON
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )