from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict
from typing import Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .taxonomy import Taxonomy, TaxonomyError
from .types import CandidateSpec, FrozenCandidate, NewLabelSpec, RankedResidual
from .utils import normalize_name, sha256_hex


class CandidateValidationError(ValueError):
    """Raised when proposer output violates the frozen local-edit grammar."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _NewLabelJSON(_StrictModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class _CandidateJSON(_StrictModel):
    operation: str
    parent_id: str = Field(min_length=1)
    source_label_id: str | None = None
    new_labels: list[_NewLabelJSON]

    @model_validator(mode="after")
    def validate_shape(self):
        if self.operation == "add":
            if self.source_label_id is not None:
                raise ValueError("add must not contain source_label_id")
            if len(self.new_labels) != 1:
                raise ValueError("add must contain exactly one new label")
        elif self.operation == "split":
            if self.source_label_id is None:
                raise ValueError("split requires source_label_id")
            if len(self.new_labels) != 2:
                raise ValueError("split must contain exactly two new labels")
        else:
            raise ValueError("operation must be add or split")
        return self


class _ProposalJSON(_StrictModel):
    candidates: list[_CandidateJSON]


def _clean_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def parse_proposal_json(raw: str) -> list[dict]:
    stripped = raw.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise CandidateValidationError("Proposer output must contain exactly one JSON object and no surrounding text")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise CandidateValidationError(f"Invalid proposer JSON: {exc}") from exc
    try:
        parsed = _ProposalJSON.model_validate(payload)
    except ValidationError as exc:
        raise CandidateValidationError(str(exc)) from exc
    return [candidate.model_dump() for candidate in parsed.candidates]


def proposal_local_label_ids(ranked: Iterable[RankedResidual]) -> tuple[str, ...]:
    labels: set[str] = set()
    for item in ranked:
        labels.update(item.record.local_label_ids)
    return tuple(sorted(labels))


def validate_and_freeze_candidates(
    *,
    raw_output: str,
    taxonomy: Taxonomy,
    ranked_batch: list[RankedResidual],
    proposal_budget: int,
    token_counter: Callable[[str], int],
    max_name_tokens: int,
    max_description_tokens: int,
) -> tuple[list[FrozenCandidate], list[dict]]:
    """Strictly parse, validate, canonicalize, deduplicate, and freeze proposer edits.

    The second return value is a complete rejection audit. Invalid entries consume
    proposer budget and are never regenerated.
    """
    if proposal_budget < 2:
        raise ValueError("proposal_budget includes stay and must be at least 2")
    candidate_objects = parse_proposal_json(raw_output)
    local_ids = set(proposal_local_label_ids(ranked_batch))
    existing_names = {normalize_name(node.name) for node in taxonomy.labels}
    retained: list[FrozenCandidate] = []
    seen_ids: set[str] = set()
    audit: list[dict] = []
    for position, obj in enumerate(candidate_objects):
        if len(retained) >= proposal_budget - 1:
            audit.append({"position": position, "status": "ignored_budget_exhausted"})
            continue
        try:
            operation = obj["operation"]
            parent_id = _clean_text(obj["parent_id"])
            source_label_id = obj.get("source_label_id")
            if source_label_id is not None:
                source_label_id = _clean_text(source_label_id)
            cleaned_labels: list[NewLabelSpec] = []
            normalized_new_names: set[str] = set()
            for new_label in obj["new_labels"]:
                name = _clean_text(new_label["name"])
                description = _clean_text(new_label["description"])
                if not name or not description:
                    raise CandidateValidationError("new label name and description must be non-empty")
                if token_counter(name) > max_name_tokens:
                    raise CandidateValidationError(
                        f"new label name exceeds {max_name_tokens} proposer tokens"
                    )
                if token_counter(description) > max_description_tokens:
                    raise CandidateValidationError(
                        f"new label description exceeds {max_description_tokens} proposer tokens"
                    )
                normalized = normalize_name(name)
                if normalized in existing_names or normalized in normalized_new_names:
                    raise CandidateValidationError("duplicate label name after Unicode normalization")
                normalized_new_names.add(normalized)
                cleaned_labels.append(NewLabelSpec(name=name, description=description))
            if operation == "add":
                if parent_id != taxonomy.root_id and parent_id not in local_ids:
                    raise CandidateValidationError("add parent lies outside the posterior-local neighborhood")
                spec = CandidateSpec(
                    operation="add",
                    parent_id=parent_id,
                    source_label_id=None,
                    new_labels=tuple(cleaned_labels),
                )
            elif operation == "split":
                if source_label_id is None or source_label_id not in local_ids:
                    raise CandidateValidationError("split source lies outside the posterior-local neighborhood")
                if not taxonomy.is_leaf(source_label_id):
                    raise CandidateValidationError("split source is not a deployed leaf")
                expected_parent = taxonomy.parent_id(source_label_id)
                if parent_id != expected_parent:
                    raise CandidateValidationError(
                        f"split parent must equal the source parent {expected_parent!r}"
                    )
                spec = CandidateSpec(
                    operation="split",
                    parent_id=parent_id,
                    source_label_id=source_label_id,
                    new_labels=tuple(cleaned_labels),
                )
            else:
                raise CandidateValidationError("unsupported operation")
            canonical_payload = {
                "operation": spec.operation,
                "parent_id": spec.parent_id,
                "source_label_id": spec.source_label_id,
                "new_labels": [
                    {"name": label.name, "description": label.description}
                    for label in spec.new_labels
                ],
            }
            identifier = sha256_hex(canonical_payload)
            if identifier in seen_ids:
                audit.append(
                    {
                        "position": position,
                        "status": "rejected",
                        "reason": "duplicate canonical candidate",
                        "candidate_id": identifier,
                    }
                )
                continue
            candidate_taxonomy = taxonomy.apply_candidate(identifier, spec)
            candidate_taxonomy.validate()
            frozen = FrozenCandidate(
                identifier=identifier,
                spec=spec,
                taxonomy=candidate_taxonomy,
                canonical_payload=canonical_payload,
            )
            retained.append(frozen)
            seen_ids.add(identifier)
            audit.append(
                {
                    "position": position,
                    "status": "retained",
                    "candidate_id": identifier,
                    "payload": canonical_payload,
                }
            )
        except (CandidateValidationError, TaxonomyError, KeyError, TypeError, ValueError) as exc:
            audit.append({"position": position, "status": "rejected", "reason": str(exc)})
    return retained, audit
