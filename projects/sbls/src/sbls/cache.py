from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import SequenceScore


class SQLiteCache:
    """Thread-safe disk cache for token likelihoods and embeddings."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.path), timeout=60.0, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA temp_store=MEMORY")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS likelihoods (
                    cache_key TEXT PRIMARY KEY,
                    log_likelihood REAL NOT NULL,
                    scored_tokens INTEGER NOT NULL,
                    terminated_by TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def get_score(self, cache_key: str) -> SequenceScore | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT log_likelihood, scored_tokens, terminated_by FROM likelihoods WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return SequenceScore(
            log_likelihood=float(row[0]),
            scored_tokens=int(row[1]),
            terminated_by=str(row[2]),
        )

    def put_score(self, cache_key: str, score: SequenceScore, metadata: dict) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO likelihoods
                (cache_key, log_likelihood, scored_tokens, terminated_by, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    float(score.log_likelihood),
                    int(score.scored_tokens),
                    score.terminated_by,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                ),
            )

    def get_embedding(self, cache_key: str) -> np.ndarray | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT dimension, vector FROM embeddings WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        dimension = int(row[0])
        array = np.frombuffer(row[1], dtype=np.float32).copy()
        if array.size != dimension:
            raise RuntimeError(f"Corrupt embedding cache entry {cache_key}")
        return array

    def put_embedding(self, cache_key: str, vector: np.ndarray, metadata: dict) -> None:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (cache_key, dimension, vector, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    cache_key,
                    int(array.size),
                    sqlite3.Binary(array.tobytes(order="C")),
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                ),
            )

    def get_many_embeddings(self, keys: Iterable[str]) -> dict[str, np.ndarray]:
        key_list = list(keys)
        if not key_list:
            return {}
        result: dict[str, np.ndarray] = {}
        # SQLite has a finite variable limit; query in conservative batches.
        for start in range(0, len(key_list), 500):
            batch = key_list[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            with self._lock:
                rows = self._connection.execute(
                    f"SELECT cache_key, dimension, vector FROM embeddings WHERE cache_key IN ({placeholders})",
                    batch,
                ).fetchall()
            for cache_key, dimension, blob in rows:
                array = np.frombuffer(blob, dtype=np.float32).copy()
                if array.size != int(dimension):
                    raise RuntimeError(f"Corrupt embedding cache entry {cache_key}")
                result[str(cache_key)] = array
        return result

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
