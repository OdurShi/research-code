from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from .taxonomy import Taxonomy
from .types import StreamItem
from .utils import read_json, read_jsonl


def load_taxonomy(path: str | Path) -> Taxonomy:
    return Taxonomy.from_dict(read_json(path))


def load_stream(path: str | Path) -> Iterator[StreamItem]:
    rows = read_jsonl(path)
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("id", index))
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate stream id: {sample_id}")
        seen_ids.add(sample_id)
        if "text" not in row:
            raise ValueError(f"Stream row {index} has no text field")
        text = str(row["text"])
        timestamp = None if row.get("timestamp") is None else str(row["timestamp"])
        metadata = {key: value for key, value in row.items() if key not in {"id", "text", "timestamp"}}
        yield StreamItem(
            id=sample_id,
            text=text,
            index=index,
            timestamp=timestamp,
            metadata=metadata,
        )
