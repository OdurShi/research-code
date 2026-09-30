from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .encoding import TextEncoder
from .taxonomy import Taxonomy
from .types import RankedResidual, ResidualRecord, TaxonomyEvidence
from .utils import token_sequence_hash


class DiscoveryError(RuntimeError):
    """Raised when residual geometry cannot be computed deterministically."""


@dataclass(frozen=True)
class ResidualComputation:
    embedding: np.ndarray
    residual: np.ndarray
    normalized_residual: np.ndarray | None
    local_label_ids: tuple[str, ...]
    rank: int
    cutoff: float


def posterior_local_neighborhood(evidence: TaxonomyEvidence, taxonomy_size: int) -> tuple[str, ...]:
    threshold = 1.0 / taxonomy_size
    labels = tuple(
        sorted(label_id for label_id, probability in evidence.label_posteriors.items() if probability >= threshold)
    )
    if not labels:
        # Numerically, at least one posterior must be >=1/K. This fallback preserves the theorem's intended set.
        labels = (
            min(
                evidence.label_posteriors,
                key=lambda label_id: (-evidence.label_posteriors[label_id], label_id),
            ),
        )
    return labels


def _svd_rank(singular_values: np.ndarray, rows: int, columns: int) -> tuple[int, float]:
    if singular_values.size == 0:
        return 0, 0.0
    sigma1 = float(singular_values[0])
    cutoff = np.finfo(np.float64).eps * max(rows, columns) * sigma1
    rank = int(np.count_nonzero(singular_values > cutoff))
    return rank, cutoff


def orthogonal_residual(z: np.ndarray, description_embeddings: np.ndarray) -> tuple[np.ndarray, int, float]:
    vector = np.asarray(z, dtype=np.float64).reshape(-1)
    descriptions = np.asarray(description_embeddings, dtype=np.float64)
    if descriptions.ndim != 2 or descriptions.shape[1] != vector.size:
        raise DiscoveryError("Description embeddings and sample embedding have incompatible shapes")
    norms = np.linalg.norm(descriptions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise DiscoveryError("Description embeddings contain zero or non-finite vectors")
    descriptions = descriptions / norms[:, None]
    _, singular_values, vh = np.linalg.svd(descriptions, full_matrices=False)
    rank, cutoff = _svd_rank(singular_values, descriptions.shape[0], descriptions.shape[1])
    if rank == 0:
        projection = np.zeros_like(vector)
    else:
        retained = vh[:rank, :]
        projection = retained.T @ (retained @ vector)
    residual = vector - projection
    # Appendix A.1 discards residuals that are numerically zero under the same
    # machine-precision cutoff used for the description-space SVD rank.
    if float(np.linalg.norm(residual)) <= cutoff:
        residual = np.zeros_like(residual)
    return residual.astype(np.float32), rank, cutoff


def compute_residual(
    *,
    text: str,
    taxonomy: Taxonomy,
    evidence: TaxonomyEvidence,
    encoder: TextEncoder,
) -> ResidualComputation:
    local_ids = posterior_local_neighborhood(evidence, taxonomy.size)
    description_texts = [taxonomy.get(label_id).description for label_id in local_ids]
    matrix = encoder.encode([text, *description_texts])
    if matrix.shape[0] != len(description_texts) + 1:
        raise DiscoveryError("Encoder returned an unexpected number of vectors")
    z = matrix[0]
    norm = float(np.linalg.norm(z))
    if not np.isfinite(norm) or norm <= 0.0:
        raise DiscoveryError("Sample embedding is zero or non-finite")
    z = (z / norm).astype(np.float32)
    residual, rank, cutoff = orthogonal_residual(z, matrix[1:])
    residual_norm = float(np.linalg.norm(residual))
    normalized = None if residual_norm <= 0.0 else (residual / residual_norm).astype(np.float32)
    return ResidualComputation(
        embedding=z,
        residual=residual,
        normalized_residual=normalized,
        local_label_ids=local_ids,
        rank=rank,
        cutoff=cutoff,
    )


def canonical_leading_direction(matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    residuals = np.asarray(matrix, dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[0] < 1:
        raise DiscoveryError("Residual matrix must be non-empty and two-dimensional")
    if np.any(~np.isfinite(residuals)):
        raise DiscoveryError("Residual matrix contains non-finite values")
    norms = np.linalg.norm(residuals, axis=1)
    if np.any(norms <= 0.0):
        raise DiscoveryError("Residual matrix contains a zero row")
    residuals = residuals / norms[:, None]
    _, singular_values, vh = np.linalg.svd(residuals, full_matrices=False)
    if singular_values.size == 0:
        raise DiscoveryError("SVD returned no singular values")
    top_eigenvalue = float(singular_values[0] ** 2)
    tolerance = np.finfo(np.float64).eps * max(residuals.shape) * max(1.0, top_eigenvalue)
    multiplicity = int(np.count_nonzero(np.abs(singular_values**2 - top_eigenvalue) <= tolerance))
    top_basis = vh[:multiplicity, :]
    projector = top_basis.T @ top_basis
    direction = None
    for coordinate in range(projector.shape[0]):
        candidate = projector[:, coordinate]
        norm = float(np.linalg.norm(candidate))
        if norm > tolerance:
            direction = candidate / norm
            break
    if direction is None:
        raise DiscoveryError("Unable to construct a canonical leading direction")
    second = float(singular_values[multiplicity] ** 2) if multiplicity < singular_values.size else 0.0
    eigengap = 0.0 if top_eigenvalue <= 0 else (top_eigenvalue - second) / top_eigenvalue
    return direction.astype(np.float32), top_eigenvalue, float(eigengap)


def rank_residuals(records: Iterable[ResidualRecord]) -> list[RankedResidual]:
    record_list = list(records)
    if not record_list:
        return []
    matrix = np.stack([record.normalized_residual for record in record_list], axis=0)
    direction, _, _ = canonical_leading_direction(matrix)
    scores = [float(np.dot(record.normalized_residual, direction) ** 2) for record in record_list]
    order = sorted(
        range(len(record_list)),
        key=lambda index: (
            -scores[index],
            record_list[index].arrival_index,
            record_list[index].token_hash,
        ),
    )
    return [
        RankedResidual(record=record_list[index], spectral_score=scores[index], rank=rank + 1)
        for rank, index in enumerate(order)
    ]


def make_residual_record(
    *,
    sample_id: str,
    text: str,
    arrival_index: int,
    token_ids: list[int],
    computation: ResidualComputation,
) -> ResidualRecord | None:
    if computation.normalized_residual is None:
        return None
    return ResidualRecord(
        sample_id=sample_id,
        text=text,
        arrival_index=arrival_index,
        embedding=computation.embedding,
        residual=computation.residual,
        normalized_residual=computation.normalized_residual,
        local_label_ids=computation.local_label_ids,
        token_hash=token_sequence_hash(token_ids),
    )
