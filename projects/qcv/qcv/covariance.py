from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import CovarianceModel


def fit_residual_covariance(
    signature: str,
    role_keys: Sequence[str],
    residual_rows: Sequence[Sequence[float]],
    eigenvalue_tolerance: float = 1e-12,
) -> CovarianceModel:
    """Fit the identity-target Ledoit-Wolf covariance in Eqs. (3), (10), and (11)."""
    residuals = np.asarray(residual_rows, dtype=np.float64)
    if residuals.ndim != 2:
        raise ValueError("residual_rows must form a two-dimensional array")
    sample_count, dimension = residuals.shape
    if sample_count < 2:
        raise ValueError("At least two complete residual observations are required")
    if dimension < 1:
        raise ValueError("Residual dimension must be positive")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Residual covariance received non-finite values")

    centered = residuals - np.mean(residuals, axis=0, keepdims=True)
    covariance_empirical = (centered.T @ centered) / float(sample_count)
    mu = float(np.trace(covariance_empirical) / dimension)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("Residual variance is zero or non-finite")

    identity = np.eye(dimension, dtype=np.float64)
    delta = float(np.sum((covariance_empirical - mu * identity) ** 2) / dimension)
    fourth_moments = np.sum(centered * centered, axis=1) ** 2
    beta_numerator = float(np.mean(fourth_moments) - np.sum(covariance_empirical**2))
    beta = beta_numerator / float(dimension * sample_count)
    beta = max(beta, 0.0)
    if delta > 0.0:
        alpha_lw = float(np.clip(beta / delta, 0.0, 1.0))
    else:
        alpha_lw = 0.0
    alpha = max(alpha_lw, 1.0 / sample_count)
    alpha = min(alpha, 1.0)

    covariance = (1.0 - alpha) * covariance_empirical + alpha * mu * identity
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    minimum = float(np.min(eigenvalues))
    if minimum <= eigenvalue_tolerance:
        raise ValueError(
            f"Shrunk covariance is not positive definite: minimum eigenvalue={minimum:.3e}"
        )
    inverse_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    whitening = eigenvectors @ inverse_sqrt @ eigenvectors.T
    whitening = 0.5 * (whitening + whitening.T)

    return CovarianceModel(
        signature=signature,
        role_keys=tuple(str(item) for item in role_keys),
        dimension=dimension,
        sample_count=sample_count,
        alpha_lw=alpha_lw,
        alpha=alpha,
        mu=mu,
        covariance=covariance,
        whitening=whitening,
    )
