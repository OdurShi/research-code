from __future__ import annotations

import math

from sbls.taxonomy import Taxonomy
from sbls.types import CandidateSpec, LabelNode, NewLabelSpec
from sbls.utils import sha256_hex


def base_taxonomy() -> Taxonomy:
    return Taxonomy(
        [
            LabelNode("a", "payment", "payment questions", prior=0.6),
            LabelNode("b", "refund", "refund questions", prior=0.4),
        ]
    )


def test_add_prior_transform() -> None:
    taxonomy = base_taxonomy()
    spec = CandidateSpec(
        operation="add",
        parent_id=taxonomy.root_id,
        source_label_id=None,
        new_labels=(NewLabelSpec("chargeback", "unauthorized card dispute"),),
    )
    candidate_id = sha256_hex({"x": 1})
    edited = taxonomy.apply_candidate(candidate_id, spec)
    priors = sorted(node.prior for node in edited.labels)
    assert edited.version == 1
    assert math.isclose(sum(priors), 1.0)
    assert any(math.isclose(value, 1.0 / 3.0) for value in priors)
    assert math.isclose(edited.get("a").prior, 0.4)
    assert math.isclose(edited.get("b").prior, 0.4 * 2.0 / 3.0)


def test_split_prior_transform() -> None:
    taxonomy = base_taxonomy()
    spec = CandidateSpec(
        operation="split",
        parent_id=taxonomy.root_id,
        source_label_id="b",
        new_labels=(
            NewLabelSpec("merchant refund", "a merchant-issued refund"),
            NewLabelSpec("card dispute", "an unauthorized card transaction dispute"),
        ),
    )
    edited = taxonomy.apply_candidate(sha256_hex({"split": 1}), spec)
    assert not edited.contains("b")
    new_priors = [node.prior for node in edited.labels if node.id != "a"]
    assert len(new_priors) == 2
    assert all(math.isclose(value, 0.2) for value in new_priors)
    assert math.isclose(edited.get("a").prior, 0.6)
