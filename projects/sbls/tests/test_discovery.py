from __future__ import annotations

import numpy as np

from sbls.discovery import canonical_leading_direction, orthogonal_residual, rank_residuals
from sbls.types import ResidualRecord


def test_orthogonal_residual_is_orthogonal() -> None:
    z = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    z /= np.linalg.norm(z)
    descriptions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    residual, rank, _ = orthogonal_residual(z, descriptions)
    assert rank == 2
    assert np.allclose(descriptions @ residual, 0.0, atol=1e-7)
    assert residual[2] > 0


def test_canonical_direction_repeated_eigenvalue() -> None:
    matrix = np.eye(3, dtype=np.float32)
    direction, top, gap = canonical_leading_direction(matrix)
    assert np.allclose(direction, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.isclose(top, 1.0)
    assert np.isclose(gap, 1.0)  # all top singular values are in the repeated eigenspace; next is zero


def test_ranking_ties_by_arrival_then_hash() -> None:
    records = [
        ResidualRecord("late", "x", 2, np.ones(2), np.ones(2), np.array([1.0, 0.0]), ("a",), "b"),
        ResidualRecord("early", "y", 1, np.ones(2), np.ones(2), np.array([1.0, 0.0]), ("a",), "a"),
    ]
    ranked = rank_residuals(records)
    assert [item.record.sample_id for item in ranked] == ["early", "late"]
