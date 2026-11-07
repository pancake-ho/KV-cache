from __future__ import annotations

import glob
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    files: list[Path]
    if path.is_dir():
        files = sorted(path.rglob("*.parquet")) + sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.json"))
    else:
        files = [Path(p) for p in sorted(glob.glob(str(path)))]
    if not files:
        raise FileNotFoundError(f"no Parquet/JSON/JSONL files found under {path}")

    for file in files:
        if file.suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(file)
            for batch in parquet.iter_batches(batch_size=256):
                yield from batch.to_pylist()
        elif file.suffix == ".jsonl":
            with file.open() as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        elif file.suffix == ".json":
            with file.open() as handle:
                value = json.load(handle)
            if isinstance(value, list):
                yield from value
            elif isinstance(value, dict) and isinstance(value.get("data"), list):
                yield from value["data"]
            else:
                yield value


def record_to_text(record: dict[str, Any], text_field: str = "text") -> str | None:
    value = _nested_get(record, text_field)
    if isinstance(value, str) and value.strip():
        return value
    for fallback in ("content", "context", "document", "prompt"):
        value = record.get(fallback)
        if isinstance(value, str) and value.strip():
            return value
    conversation = record.get("conversation") or record.get("messages")
    if isinstance(conversation, list):
        parts = []
        for message in conversation:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(f"{message.get('role', 'unknown')}: {message['content']}")
            elif isinstance(message, str):
                parts.append(message)
        if parts:
            return "\n\n".join(parts)
    return None


def packed_token_sequences(
    records: Iterable[dict[str, Any]],
    tokenizer,
    *,
    count: int,
    sequence_length: int,
    text_field: str = "text",
    add_eos_between_documents: bool = True,
) -> np.ndarray:
    """Deterministically pack documents into fixed-length token sequences."""
    eos = tokenizer.eos_token_id
    buffer: list[int] = []
    output: list[list[int]] = []
    for record in records:
        text = record_to_text(record, text_field=text_field)
        if text is None:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if add_eos_between_documents and eos is not None:
            ids.append(eos)
        buffer.extend(ids)
        while len(buffer) >= sequence_length and len(output) < count:
            output.append(buffer[:sequence_length])
            del buffer[:sequence_length]
        if len(output) == count:
            break
    if len(output) != count:
        raise ValueError(f"corpus only produced {len(output)}/{count} sequences of length {sequence_length}")
    return np.asarray(output, dtype=np.int32)


def _nested_get(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
