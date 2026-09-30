from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .codebook import Codebook
from .utils import first_argmin


@dataclass(frozen=True)
class DecodeResult:
    target_answer: str
    target_energies: dict[str, float]
    target_margin: float
    projection_gap: float
    best_codeword_index: int


def _energies(codewords: np.ndarray, observation: np.ndarray, whitening: np.ndarray) -> np.ndarray:
    residuals = observation[None, :] - codewords
    whitened = residuals @ np.asarray(whitening, dtype=np.float64).T
    return np.sum(whitened * whitened, axis=1)


def feasible_projection_decode(
    codebook: Codebook,
    observation: np.ndarray,
    whitening: np.ndarray,
) -> DecodeResult:
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (codebook.dimension,):
        raise ValueError("Observation dimension does not match codebook")
    if codebook.codewords.shape[0] == 0:
        raise ValueError("Cannot decode an empty codebook")

    feasible_energy = _energies(codebook.codewords, observation, whitening)
    target_energies: dict[str, float] = {}
    target_best_indices: dict[str, int] = {}
    ordered_answers = [
        answer for answer in codebook.target_alphabet if answer in codebook.target_classes
    ]
    for answer in ordered_answers:
        indices = codebook.target_classes[answer]
        local = feasible_energy[indices]
        local_index = first_argmin(local.tolist())
        global_index = int(indices[local_index])
        target_energies[answer] = float(feasible_energy[global_index])
        target_best_indices[answer] = global_index

    winner_position = first_argmin([target_energies[a] for a in ordered_answers])
    winner = ordered_answers[winner_position]
    winner_energy = target_energies[winner]
    competitors = [target_energies[a] for a in ordered_answers if a != winner]
    margin = float(min(competitors) - winner_energy) if competitors else float("inf")

    unconstrained_energy = _energies(codebook.unconstrained_codewords, observation, whitening)
    projection_gap = float(np.min(feasible_energy) - np.min(unconstrained_energy))
    if projection_gap < 0.0 and projection_gap > -1e-10:
        projection_gap = 0.0

    return DecodeResult(
        target_answer=winner,
        target_energies=target_energies,
        target_margin=margin,
        projection_gap=projection_gap,
        best_codeword_index=target_best_indices[winner],
    )
