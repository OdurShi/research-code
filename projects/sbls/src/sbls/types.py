from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


Operation = Literal["add", "split"]
Decision = Literal["stay", "add", "split"]
PredictionAction = Literal["predict", "defer"]


@dataclass(frozen=True)
class LabelNode:
    id: str
    name: str
    description: str
    parent_id: str | None = None
    prior: float = 0.0


@dataclass(frozen=True)
class SequenceScore:
    log_likelihood: float
    scored_tokens: int
    terminated_by: Literal["eos", "trunc"]


@dataclass(frozen=True)
class TaxonomyEvidence:
    log_mixture: float
    label_posteriors: dict[str, float]
    label_log_likelihoods: dict[str, float]
    scored_tokens: int


@dataclass(frozen=True)
class DeferralDecision:
    action: PredictionAction
    predicted_label_id: str | None
    score: float
    taxonomy_evidence: TaxonomyEvidence
    background_log_likelihood: float


@dataclass(frozen=True)
class ResidualRecord:
    sample_id: str
    text: str
    arrival_index: int
    embedding: np.ndarray
    residual: np.ndarray
    normalized_residual: np.ndarray
    local_label_ids: tuple[str, ...]
    token_hash: str


@dataclass(frozen=True)
class RankedResidual:
    record: ResidualRecord
    spectral_score: float
    rank: int


@dataclass(frozen=True)
class NewLabelSpec:
    name: str
    description: str


@dataclass(frozen=True)
class CandidateSpec:
    operation: Operation
    parent_id: str
    new_labels: tuple[NewLabelSpec, ...]
    source_label_id: str | None = None


@dataclass(frozen=True)
class FrozenCandidate:
    identifier: str
    spec: CandidateSpec
    taxonomy: Any
    canonical_payload: dict[str, Any]


@dataclass
class SelectionStep:
    n: int
    log_likelihood_ratios: dict[str, float]
    log_e_value: float
    boundary: float
    crossed: bool
    selected_candidate_id: str | None


@dataclass
class CycleRecord:
    cycle_index: int
    proposal_time: int
    taxonomy_version_before: int
    candidate_ids: list[str]
    eta: float
    boundary: float
    horizon: int
    proposal_sample_ids: list[str]
    proposal_payloads: list[dict[str, Any]]
    evidence_trajectory: list[dict[str, Any]] = field(default_factory=list)
    decision: Decision | None = None
    selected_candidate_id: str | None = None
    stopping_time: int | None = None
    close_reason: str | None = None
    taxonomy_version_after: int | None = None


@dataclass(frozen=True)
class StreamItem:
    id: str
    text: str
    index: int
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    sample_id: str
    index: int
    timestamp: str | None
    predicted_label_id: str | None
    action: PredictionAction
    deferral_score: float | None
    taxonomy_version: int
    cycle_index: int | None
    phase: Literal["discovery", "validation"]
    metadata: dict[str, Any] = field(default_factory=dict)
