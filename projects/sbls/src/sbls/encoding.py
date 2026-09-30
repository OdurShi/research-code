from __future__ import annotations

import gc
from typing import Iterable, Protocol

import numpy as np

from .cache import SQLiteCache
from .config import EncoderConfig
from .utils import sha256_hex, stable_text_hash


class TextEncoder(Protocol):
    def encode(self, texts: Iterable[str]) -> np.ndarray: ...

    def unload(self) -> None: ...


class BGETextEncoder:
    """Frozen unit-normalized BAAI/bge-m3 encoder with disk caching."""

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self._model = None
        self._cache = SQLiteCache(config.cache_path) if config.use_cache else None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for BGETextEncoder. Install the project dependencies."
            ) from exc
        self._model = SentenceTransformer(
            self.config.model_id,
            revision=self.config.revision,
            device=self.config.device,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        )

    def _cache_key(self, text: str) -> str:
        return sha256_hex(
            {
                "kind": "embedding",
                "model": self.config.model_id,
                "revision": self.config.revision,
                "normalized": True,
                "text_hash": stable_text_hash(text),
            }
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)
        keys = [self._cache_key(text) for text in text_list]
        cached = self._cache.get_many_embeddings(keys) if self._cache is not None else {}
        missing_indices = [index for index, key in enumerate(keys) if key not in cached]
        new_vectors: dict[int, np.ndarray] = {}
        if missing_indices:
            self._ensure_model()
            missing_texts = [text_list[index] for index in missing_indices]
            vectors = self._model.encode(
                missing_texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(missing_indices):
                raise RuntimeError("Encoder returned an unexpected shape")
            for local, index in enumerate(missing_indices):
                vector = vectors[local]
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(norm) or norm <= 0.0:
                    raise RuntimeError(f"Encoder produced an invalid vector for input {index}")
                vector = (vector / norm).astype(np.float32, copy=False)
                new_vectors[index] = vector
                if self._cache is not None:
                    self._cache.put_embedding(
                        keys[index],
                        vector,
                        {
                            "model_id": self.config.model_id,
                            "revision": self.config.revision,
                            "text_hash": stable_text_hash(text_list[index]),
                        },
                    )
        ordered: list[np.ndarray] = []
        dimension: int | None = None
        for index, key in enumerate(keys):
            vector = cached.get(key, new_vectors.get(index))
            if vector is None:
                raise RuntimeError("Internal embedding-cache assembly failure")
            if dimension is None:
                dimension = vector.size
            elif vector.size != dimension:
                raise RuntimeError("Embedding dimensions differ within one batch")
            norm = float(np.linalg.norm(vector))
            if not np.isclose(norm, 1.0, rtol=1e-4, atol=1e-5):
                vector = vector / norm
            ordered.append(np.asarray(vector, dtype=np.float32))
        return np.stack(ordered, axis=0)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                return

    def close(self) -> None:
        self.unload()
        if self._cache is not None:
            self._cache.close()
            self._cache = None
