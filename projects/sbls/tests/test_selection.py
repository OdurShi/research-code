from __future__ import annotations

import math

from sbls.selection import SequentialBayesianSelector, cycle_boundary, cycle_error_level
from sbls.taxonomy import Taxonomy
from sbls.types import CandidateSpec, FrozenCandidate, LabelNode, NewLabelSpec


def candidate(identifier: str) -> FrozenCandidate:
    taxonomy = Taxonomy([LabelNode("a", "a", "a", prior=1.0)])
    spec = CandidateSpec("add", taxonomy.root_id, (NewLabelSpec("b", "b"),), None)
    return FrozenCandidate(identifier, spec, taxonomy.apply_candidate(identifier, spec), {"id": identifier})


def test_error_spending_product() -> None:
    alpha = 0.05
    product = 1.0
    for j in range(1, 10001):
        eta = cycle_error_level(alpha, j)
        product *= 1.0 - eta
    assert product >= 1.0 - alpha
    assert product < 1.0


def test_boundary_matches_formula() -> None:
    eta, boundary = cycle_boundary(0.05, 1)
    assert math.isclose(eta, 1.0 - math.sqrt(0.95), rel_tol=1e-12)
    assert math.isclose(boundary, 0.5 + 1.0 / (2.0 * eta), rel_tol=1e-12)


def test_selector_crosses_and_tie_breaks_by_id() -> None:
    selector = SequentialBayesianSelector([candidate("b"), candidate("a")], alpha=0.5, cycle_index=1, horizon=5)
    step = selector.update(null_log_likelihood=-10.0, candidate_log_likelihoods={"a": 0.0, "b": 0.0})
    assert step.crossed
    assert step.selected_candidate_id == "a"
