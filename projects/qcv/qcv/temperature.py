from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .utils import logsumexp, softmax


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll: float
    converged: bool
    iterations: int


def _objective(log_temperature: np.ndarray, scores: np.ndarray, gold: np.ndarray) -> tuple[float, np.ndarray]:
    scalar = float(log_temperature[0])
    temperature = float(np.exp(scalar))
    scaled = scores / temperature
    nll_vector = -scaled[np.arange(scores.shape[0]), gold] + logsumexp(scaled, axis=1)
    probabilities = np.exp(scaled - logsumexp(scaled, axis=1)[:, None])
    expected = np.sum(probabilities * scores, axis=1)
    gradient = np.mean((scores[np.arange(scores.shape[0]), gold] - expected) / temperature)
    return float(np.mean(nll_vector)), np.asarray([gradient], dtype=np.float64)


def fit_temperature(
    score_rows: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
) -> TemperatureFit:
    scores = np.asarray(score_rows, dtype=np.float64)
    gold = np.asarray(gold_indices, dtype=np.int64)
    if scores.ndim != 2:
        raise ValueError("score_rows must form a two-dimensional array")
    if scores.shape[0] != gold.shape[0]:
        raise ValueError("The number of score rows and gold labels differs")
    if scores.shape[0] == 0:
        raise ValueError("Cannot fit a temperature without calibration examples")
    if np.any(gold < 0) or np.any(gold >= scores.shape[1]):
        raise ValueError("A gold index is outside the score alphabet")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Temperature calibration received non-finite scores")

    result = minimize(
        fun=lambda x: _objective(x, scores, gold),
        x0=np.asarray([0.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-12.0, 12.0)],
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000, "maxls": 100},
    )
    temperature = float(np.exp(float(result.x[0])))
    nll, _ = _objective(np.asarray([np.log(temperature)]), scores, gold)
    return TemperatureFit(
        temperature=temperature,
        nll=float(nll),
        converged=bool(result.success),
        iterations=int(result.nit),
    )


def calibrated_probabilities(scores: Sequence[float], temperature: float) -> np.ndarray:
    return softmax(np.asarray(scores, dtype=np.float64), temperature=temperature)
