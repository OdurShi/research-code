from __future__ import annotations

import numpy as np
import pytest

from sbls.candidates import CandidateValidationError, parse_proposal_json, validate_and_freeze_candidates
from sbls.taxonomy import Taxonomy
from sbls.types import LabelNode, RankedResidual, ResidualRecord


def ranked_batch() -> list[RankedResidual]:
    record = ResidualRecord(
        sample_id="x",
        text="unauthorized charge",
        arrival_index=1,
        embedding=np.array([1.0, 0.0]),
        residual=np.array([1.0, 0.0]),
        normalized_residual=np.array([1.0, 0.0]),
        local_label_ids=("refund",),
        token_hash="0",
    )
    return [RankedResidual(record, 1.0, 1)]


def taxonomy() -> Taxonomy:
    return Taxonomy([LabelNode("refund", "refund", "merchant refund", prior=1.0)])


def test_rejects_surrounding_text() -> None:
    with pytest.raises(CandidateValidationError):
        parse_proposal_json('Here is JSON: {"candidates":[]}')


def test_freezes_valid_add() -> None:
    raw = '{"candidates":[{"operation":"add","parent_id":"__root__","source_label_id":null,"new_labels":[{"name":"card dispute","description":"an unauthorized card purchase dispute"}]}]}'
    candidates, audit = validate_and_freeze_candidates(
        raw_output=raw,
        taxonomy=taxonomy(),
        ranked_batch=ranked_batch(),
        proposal_budget=4,
        token_counter=lambda text: len(text.split()),
        max_name_tokens=8,
        max_description_tokens=48,
    )
    assert len(candidates) == 1
    assert candidates[0].spec.operation == "add"
    assert audit[0]["status"] == "retained"


def test_rejects_duplicate_deployed_name() -> None:
    raw = '{"candidates":[{"operation":"add","parent_id":"__root__","source_label_id":null,"new_labels":[{"name":"Refund","description":"duplicate"}]}]}'
    candidates, audit = validate_and_freeze_candidates(
        raw_output=raw,
        taxonomy=taxonomy(),
        ranked_batch=ranked_batch(),
        proposal_budget=4,
        token_counter=lambda text: len(text.split()),
        max_name_tokens=8,
        max_description_tokens=48,
    )
    assert candidates == []
    assert audit[0]["status"] == "rejected"
