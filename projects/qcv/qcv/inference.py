from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .calibration import probabilities_for_record
from .codebook import build_codebook
from .decoding import feasible_projection_decode
from .geometry import stack_observation, target_class_separation, whitening_for_mode
from .selection import (
    SelectionResult,
    exact_select,
    fixed_select,
    greedy_select,
    random_select,
    reliability_select,
)
from .types import CalibrationArtifacts, Prediction, ScoreRecord


@dataclass(frozen=True)
class InferenceOptions:
    budget: int = 5
    selection: str = "qcv"
    metric: str = "full"
    strict_coverage: bool = True
    random_seed: int = 0
    exact_subset_limit: int = 10_000

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be at least one")
        if self.selection not in {"qcv", "exact", "fixed", "random", "reliability"}:
            raise ValueError(f"Unsupported selection method {self.selection!r}")
        if self.metric not in {"full", "diagonal", "identity"}:
            raise ValueError(f"Unsupported metric mode {self.metric!r}")


def _target_only_answer(record: ScoreRecord, probabilities: dict[str, np.ndarray]) -> str:
    target = record.compiled.target
    distribution = probabilities[target.question_id]
    return target.role.alphabet[int(np.argmax(distribution))]


def _all_required_subsets(record: ScoreRecord, budget: int) -> Iterable[tuple[object, ...]]:
    maximum = min(budget - 1, len(record.compiled.auxiliaries))
    yield (record.compiled.target,)
    for size in range(1, maximum + 1):
        for subset in itertools.combinations(record.compiled.auxiliaries, size):
            yield (record.compiled.target,) + subset


def coverage_reason(
    record: ScoreRecord,
    artifacts: CalibrationArtifacts,
    options: InferenceOptions,
) -> str | None:
    if len(record.compiled.auxiliaries) < options.budget - 1:
        return "insufficient_valid_auxiliaries"
    for question in record.compiled.all_questions:
        if question.role.key not in artifacts.temperatures:
            return f"missing_temperature:{question.role.key}"
    if not options.strict_coverage:
        return None
    for questions in _all_required_subsets(record, options.budget):
        try:
            codebook = build_codebook(record.compiled, questions)
        except ValueError:
            return "invalid_codebook"
        if codebook.codewords.shape[0] == 0 or len(codebook.target_classes) < 1:
            return f"empty_feasible_code:{codebook.signature}"
        model = artifacts.covariances.get(codebook.signature)
        if model is None:
            return f"unsupported_signature:{codebook.signature}"
        if model.dimension != codebook.dimension:
            return f"dimension_mismatch:{codebook.signature}"
        eigenvalues = np.linalg.eigvalsh(model.covariance)
        if float(np.min(eigenvalues)) <= 0.0:
            return f"non_positive_definite:{codebook.signature}"
    return None


def _select(
    record: ScoreRecord,
    artifacts: CalibrationArtifacts,
    options: InferenceOptions,
) -> SelectionResult:
    if options.selection == "qcv":
        return greedy_select(record.compiled, options.budget, artifacts, options.metric)
    if options.selection == "exact":
        return exact_select(
            record.compiled,
            options.budget,
            artifacts,
            options.metric,
            options.exact_subset_limit,
        )
    if options.selection == "fixed":
        return fixed_select(record.compiled, options.budget)
    if options.selection == "random":
        return random_select(record.compiled, options.budget, options.random_seed)
    if options.selection == "reliability":
        return reliability_select(record.compiled, options.budget, artifacts)
    raise AssertionError(options.selection)


def infer_record(
    record: ScoreRecord,
    artifacts: CalibrationArtifacts,
    options: InferenceOptions,
) -> Prediction:
    try:
        probabilities = probabilities_for_record(record, artifacts)
    except KeyError as exc:
        target = record.compiled.target
        raw = np.asarray(record.scores[target.question_id].mean_logprobs, dtype=np.float64)
        target_only = target.role.alphabet[int(np.argmax(raw))]
        return Prediction(
            instance_id=record.instance_id,
            covered=False,
            target_only_answer=target_only,
            final_answer=target_only,
            selected_question_ids=(),
            target_margin=None,
            projection_gap=None,
            target_class_separation=None,
            reason=f"temperature_error:{exc}",
        )

    target_only = _target_only_answer(record, probabilities)
    reason = coverage_reason(record, artifacts, options)
    if reason is not None:
        return Prediction(
            instance_id=record.instance_id,
            covered=False,
            target_only_answer=target_only,
            final_answer=target_only,
            selected_question_ids=(),
            target_margin=None,
            projection_gap=None,
            target_class_separation=None,
            reason=reason,
        )

    try:
        selection = _select(record, artifacts, options)
        questions = (record.compiled.target,) + selection.selected
        codebook = build_codebook(record.compiled, questions)
        covariance = artifacts.covariance_for(codebook.signature)
        whitening = whitening_for_mode(covariance, options.metric)
        observation = stack_observation(codebook.questions, probabilities)
        decoded = feasible_projection_decode(codebook, observation, whitening)
        separation = target_class_separation(codebook, whitening)
    except (KeyError, ValueError, FloatingPointError) as exc:
        return Prediction(
            instance_id=record.instance_id,
            covered=False,
            target_only_answer=target_only,
            final_answer=target_only,
            selected_question_ids=(),
            target_margin=None,
            projection_gap=None,
            target_class_separation=None,
            reason=f"inference_error:{type(exc).__name__}:{exc}",
        )

    return Prediction(
        instance_id=record.instance_id,
        covered=True,
        target_only_answer=target_only,
        final_answer=decoded.target_answer,
        selected_question_ids=tuple(question.question_id for question in selection.selected),
        target_margin=decoded.target_margin,
        projection_gap=decoded.projection_gap,
        target_class_separation=separation,
        energies=decoded.target_energies,
        reason="qcv",
    )
