from __future__ import annotations

import hashlib
import json
import math
import os
import random
import unicodedata
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT_ID = "__root__"


def normalize_name(value: str) -> str:
    """Canonical normalization used for duplicate-name checks."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def canonical_json_bytes(value: Any) -> bytes:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def stable_text_hash(text: str) -> str:
    return sha256_hex(unicodedata.normalize("NFC", text))


def token_sequence_hash(token_ids: Sequence[int]) -> str:
    # A JSON encoding avoids architecture-dependent integer byte widths.
    return sha256_hex([int(x) for x in token_ids])


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    arr = np.asarray(values, dtype=np.float64)
    m = float(np.max(arr))
    if math.isinf(m):
        return m
    return m + float(np.log(np.exp(arr - m).sum(dtype=np.float64)))


def logaddexp_many(values: Sequence[float]) -> float:
    return logsumexp(values)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False)
    atomic_write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    os.replace(tmp, path)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]
