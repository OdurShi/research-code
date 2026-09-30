from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp


@dataclass
class GaussianBOCPD:
    """Bayesian online changepoint detection for a scalar mismatch stream.

    This is an exact conjugate Normal-Gamma implementation with a constant hazard.
    It supports the paper's BOCPD-triggered candidate-generation baseline.
    """

    hazard: float = 1.0 / 200.0
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0
    max_run_length: int = 2048

    def __post_init__(self) -> None:
        if not 0.0 < self.hazard < 1.0:
            raise ValueError("hazard must lie in (0, 1)")
        if min(self.kappa0, self.alpha0, self.beta0) <= 0.0:
            raise ValueError("Normal-Gamma hyperparameters must be positive")
        self.log_run_probs = np.array([0.0], dtype=np.float64)
        self.mu = np.array([self.mu0], dtype=np.float64)
        self.kappa = np.array([self.kappa0], dtype=np.float64)
        self.alpha = np.array([self.alpha0], dtype=np.float64)
        self.beta = np.array([self.beta0], dtype=np.float64)
        self.time = 0

    @staticmethod
    def _student_t_logpdf(x: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        degrees = 2.0 * alpha
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)
        z2 = (x - mu) ** 2 / scale2
        return (
            gammaln((degrees + 1.0) / 2.0)
            - gammaln(degrees / 2.0)
            - 0.5 * (np.log(degrees * math.pi) + np.log(scale2))
            - ((degrees + 1.0) / 2.0) * np.log1p(z2 / degrees)
        )

    def update(self, value: float) -> dict[str, float]:
        x = float(value)
        if not math.isfinite(x):
            raise ValueError("BOCPD observation must be finite")
        predictive = self._student_t_logpdf(x, self.mu, self.kappa, self.alpha, self.beta)
        log_growth = self.log_run_probs + predictive + math.log1p(-self.hazard)
        log_cp = logsumexp(self.log_run_probs + predictive + math.log(self.hazard))
        new_log_probs = np.concatenate([[log_cp], log_growth])
        if new_log_probs.size > self.max_run_length + 1:
            new_log_probs = new_log_probs[: self.max_run_length + 1]
        new_log_probs -= logsumexp(new_log_probs)

        new_mu = np.empty_like(new_log_probs)
        new_kappa = np.empty_like(new_log_probs)
        new_alpha = np.empty_like(new_log_probs)
        new_beta = np.empty_like(new_log_probs)
        new_mu[0] = self.mu0
        new_kappa[0] = self.kappa0
        new_alpha[0] = self.alpha0
        new_beta[0] = self.beta0
        old_count = new_log_probs.size - 1
        kappa_old = self.kappa[:old_count]
        mu_old = self.mu[:old_count]
        alpha_old = self.alpha[:old_count]
        beta_old = self.beta[:old_count]
        kappa_new = kappa_old + 1.0
        new_mu[1:] = (kappa_old * mu_old + x) / kappa_new
        new_kappa[1:] = kappa_new
        new_alpha[1:] = alpha_old + 0.5
        new_beta[1:] = beta_old + 0.5 * kappa_old * (x - mu_old) ** 2 / kappa_new

        self.log_run_probs = new_log_probs
        self.mu, self.kappa, self.alpha, self.beta = new_mu, new_kappa, new_alpha, new_beta
        self.time += 1
        cp_probability = float(math.exp(new_log_probs[0]))
        run_length = int(np.argmax(new_log_probs))
        return {
            "time": self.time,
            "changepoint_probability": cp_probability,
            "map_run_length": run_length,
        }


@dataclass(frozen=True)
class PointwiseRejector:
    score_threshold: float

    def decide(self, score: float, predicted_label_id: str) -> tuple[str, str | None]:
        return ("predict", predicted_label_id) if score >= self.score_threshold else ("defer", None)
