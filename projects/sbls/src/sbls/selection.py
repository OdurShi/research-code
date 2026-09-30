from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .types import FrozenCandidate, SelectionStep
from .utils import logsumexp


class SelectionError(RuntimeError):
    """Raised when the sequential evidence process receives an invalid update."""


def cycle_error_level(alpha: float, cycle_index: int) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if cycle_index < 1:
        raise ValueError("cycle_index is one-based and must be positive")
    weight = 1.0 / (cycle_index * (cycle_index + 1.0))
    # expm1/log1p preserve accuracy for late cycles.
    return -math.expm1(weight * math.log1p(-alpha))


def cycle_boundary(alpha: float, cycle_index: int) -> tuple[float, float]:
    eta = cycle_error_level(alpha, cycle_index)
    boundary = 0.5 + 1.0 / (2.0 * eta)
    return eta, boundary


@dataclass
class SequentialBayesianSelector:
    candidates: list[FrozenCandidate]
    alpha: float
    cycle_index: int
    horizon: int

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("At least one non-null candidate is required")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if len({candidate.identifier for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate identifiers must be unique")
        self.eta, self.boundary = cycle_boundary(self.alpha, self.cycle_index)
        self.log_boundary = math.log(self.boundary)
        self.log_likelihood_ratios = {candidate.identifier: 0.0 for candidate in self.candidates}
        self.n = 0
        self.closed = False

    def _log_e_value(self) -> float:
        m = len(self.candidates)
        log_average_lr = logsumexp(list(self.log_likelihood_ratios.values())) - math.log(m)
        return logsumexp([math.log(0.5), math.log(0.5) + log_average_lr])

    def update(
        self,
        *,
        null_log_likelihood: float,
        candidate_log_likelihoods: dict[str, float],
    ) -> SelectionStep:
        if self.closed:
            raise SelectionError("Cannot update a closed selector")
        expected = set(self.log_likelihood_ratios)
        if set(candidate_log_likelihoods) != expected:
            missing = expected - set(candidate_log_likelihoods)
            extra = set(candidate_log_likelihoods) - expected
            raise SelectionError(f"Candidate likelihood keys differ; missing={missing}, extra={extra}")
        if not math.isfinite(null_log_likelihood):
            raise SelectionError("Null likelihood is non-finite")
        for candidate_id, value in candidate_log_likelihoods.items():
            if not math.isfinite(value):
                raise SelectionError(f"Candidate likelihood is non-finite for {candidate_id}")
            self.log_likelihood_ratios[candidate_id] += value - null_log_likelihood
        self.n += 1
        log_e = self._log_e_value()
        crossed = log_e >= self.log_boundary
        selected = None
        if crossed:
            selected = min(
                self.log_likelihood_ratios,
                key=lambda candidate_id: (-self.log_likelihood_ratios[candidate_id], candidate_id),
            )
            self.closed = True
        elif self.n >= self.horizon:
            self.closed = True
        return SelectionStep(
            n=self.n,
            log_likelihood_ratios=dict(self.log_likelihood_ratios),
            log_e_value=log_e,
            boundary=self.boundary,
            crossed=crossed,
            selected_candidate_id=selected,
        )

    @property
    def exhausted(self) -> bool:
        return self.closed and self.n >= self.horizon


@dataclass
class FixedHorizonBayesSelector:
    """Decision-isolation baseline using the same future likelihood ratios."""

    candidates: list[FrozenCandidate]
    horizon: int
    log_bayes_threshold: float

    def __post_init__(self) -> None:
        if not self.candidates or self.horizon < 1:
            raise ValueError("Fixed-horizon selector requires candidates and a positive horizon")
        self.n = 0
        self.log_likelihood_ratios = {candidate.identifier: 0.0 for candidate in self.candidates}
        self.closed = False

    def update(self, null_log_likelihood: float, candidate_log_likelihoods: dict[str, float]) -> str | None:
        if self.closed:
            raise SelectionError("Selector is closed")
        for candidate_id in self.log_likelihood_ratios:
            self.log_likelihood_ratios[candidate_id] += candidate_log_likelihoods[candidate_id] - null_log_likelihood
        self.n += 1
        if self.n < self.horizon:
            return None
        self.closed = True
        winner = min(
            self.log_likelihood_ratios,
            key=lambda candidate_id: (-self.log_likelihood_ratios[candidate_id], candidate_id),
        )
        return winner if self.log_likelihood_ratios[winner] >= self.log_bayes_threshold else "stay"
