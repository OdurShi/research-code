from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from .codebook import Codebook
from .types import CovarianceModel, QuestionSpec


def whitening_for_mode(model: CovarianceModel, mode: str) -> np.ndarray:
    if mode == "full":
        return np.asarray(model.whitening, dtype=np.float64)
    if mode == "diagonal":
        diagonal = np.diag(np.asarray(model.covariance, dtype=np.float64))
        if np.any(diagonal <= 0.0):
            raise ValueError("Diagonal covariance contains a non-positive entry")
        return np.diag(1.0 / np.sqrt(diagonal))
    if mode == "identity":
        return np.eye(model.dimension, dtype=np.float64)
    raise ValueError(f"Unknown metric mode {mode!r}")


def _minimum_cross_distance(left: np.ndarray, right: np.ndarray, chunk_size: int = 2048) -> float:
    best = math.inf
    for start in range(0, left.shape[0], chunk_size):
        chunk = left[start : start + chunk_size]
        differences = chunk[:, None, :] - right[None, :, :]
        squared = np.sum(differences * differences, axis=2)
        local = float(np.min(squared))
        if local < best:
            best = local
            if best <= 0.0:
                return 0.0
    return math.sqrt(best)


_SEPARATION_CACHE: dict[tuple[str, str], float] = {}


def target_class_separation(codebook: Codebook, whitening: np.ndarray) -> float:
    if codebook.codewords.shape[0] == 0:
        raise ValueError("Cannot compute target separation for an empty codebook")
    whitening = np.asarray(whitening, dtype=np.float64)
    metric_hash = hashlib.sha256(whitening.tobytes()).hexdigest()[:24]
    cache_key = (codebook.structure_key, metric_hash)
    cached = _SEPARATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    answers = [answer for answer in codebook.target_alphabet if answer in codebook.target_classes]
    if len(answers) < 2:
        _SEPARATION_CACHE[cache_key] = 0.0
        return 0.0
    transformed = codebook.codewords @ whitening.T
    best = math.inf
    for left_index, left_answer in enumerate(answers):
        left = transformed[codebook.target_classes[left_answer]]
        for right_answer in answers[left_index + 1 :]:
            right = transformed[codebook.target_classes[right_answer]]
            distance = _minimum_cross_distance(left, right)
            if distance < best:
                best = distance
                if best <= 0.0:
                    _SEPARATION_CACHE[cache_key] = 0.0
                    return 0.0
    result = float(best)
    _SEPARATION_CACHE[cache_key] = result
    return result


def stack_observation(
    questions: Sequence[QuestionSpec], probabilities: Mapping[str, np.ndarray]
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for question in questions:
        try:
            block = np.asarray(probabilities[question.question_id], dtype=np.float64)
        except KeyError as exc:
            raise KeyError(f"Missing probability block for {question.question_id}") from exc
        if block.shape != (len(question.role.alphabet),):
            raise ValueError(f"Probability block shape mismatch for {question.question_id}")
        if np.any(block < 0.0) or not np.isclose(float(np.sum(block)), 1.0, atol=1e-8):
            raise ValueError(f"Invalid probability distribution for {question.question_id}")
        blocks.append(block)
    return np.concatenate(blocks, axis=0)
