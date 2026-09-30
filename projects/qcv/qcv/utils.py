from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s\?\!\.]+$")


def normalize_text(text: str) -> str:
    """Normalize question text for deterministic matching and deduplication."""
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = _WHITESPACE.sub(" ", text.strip().lower())
    return _TRAILING_PUNCT.sub("", text)


def stable_hash(parts: Iterable[str], length: int = 16) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def alphabet_fingerprint(alphabet: Sequence[str]) -> str:
    return stable_hash(alphabet, length=20)


def canonical_decimal(value: Any, precision: int | None = None) -> str:
    """Return a canonical finite decimal string without scientific notation."""
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Not a finite decimal value: {value!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"Not a finite decimal value: {value!r}")
    if precision is not None:
        quantum = Decimal(1).scaleb(-precision)
        dec = dec.quantize(quantum, rounding=ROUND_HALF_UP)
    normalized = format(dec, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"-0", ""}:
        normalized = "0"
    return normalized


def decimal_value(value: str) -> Decimal:
    try:
        out = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Answer {value!r} is not numeric") from exc
    if not out.is_finite():
        raise ValueError(f"Answer {value!r} is not finite")
    return out


def json_dump(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def json_load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on {path}:{line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    tmp.replace(path)


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError(f"Temperature must be positive and finite, got {temperature}")
    values = np.asarray(values, dtype=np.float64) / temperature
    values = values - np.max(values)
    exp = np.exp(values)
    denom = float(np.sum(exp))
    if denom <= 0 or not math.isfinite(denom):
        raise FloatingPointError("Softmax normalization failed")
    return exp / denom


def logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - maximum)
    return np.squeeze(maximum, axis=axis) + np.log(np.sum(shifted, axis=axis))


def first_argmin(values: Sequence[float], atol: float = 0.0) -> int:
    if not values:
        raise ValueError("Cannot select from an empty sequence")
    best_index = 0
    best_value = float(values[0])
    for index, value in enumerate(values[1:], start=1):
        scalar = float(value)
        if scalar < best_value - atol:
            best_index = index
            best_value = scalar
    return best_index


def first_argmax(values: Sequence[float], atol: float = 0.0) -> int:
    if not values:
        raise ValueError("Cannot select from an empty sequence")
    best_index = 0
    best_value = float(values[0])
    for index, value in enumerate(values[1:], start=1):
        scalar = float(value)
        if scalar > best_value + atol:
            best_index = index
            best_value = scalar
    return best_index


def batched(sequence: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    for start in range(0, len(sequence), batch_size):
        yield sequence[start : start + batch_size]
