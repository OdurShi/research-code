from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .codebook import build_codebook
from .covariance import fit_residual_covariance
from .geometry import stack_observation
from .signatures import role_keys
from .temperature import calibrated_probabilities, fit_temperature
from .types import CalibrationArtifacts, QuestionSpec, ScoreRecord


@dataclass(frozen=True)
class CalibrationSummary:
    records_seen: int
    role_models: int
    covariance_models: int
    unsupported_signatures: int
    skipped_incomplete_assignments: int


def probabilities_for_record(
    record: ScoreRecord,
    artifacts: CalibrationArtifacts,
) -> dict[str, np.ndarray]:
    probabilities: dict[str, np.ndarray] = {}
    for question in record.compiled.all_questions:
        temperature = artifacts.temperature_for(question.role)
        scores = record.scores[question.question_id].mean_logprobs
        probabilities[question.question_id] = calibrated_probabilities(scores, temperature)
    return probabilities


def _question_subsets(record: ScoreRecord, max_budget: int) -> Iterable[tuple[QuestionSpec, ...]]:
    maximum_auxiliaries = min(max_budget - 1, len(record.compiled.auxiliaries))
    yield (record.compiled.target,)
    for size in range(1, maximum_auxiliaries + 1):
        for subset in itertools.combinations(record.compiled.auxiliaries, size):
            yield (record.compiled.target,) + subset


def fit_calibration(
    records: Sequence[ScoreRecord],
    max_budget: int,
) -> tuple[CalibrationArtifacts, CalibrationSummary]:
    if max_budget < 1:
        raise ValueError("max_budget must be at least one")
    if not records:
        raise ValueError("No calibration records were provided")

    role_scores: dict[str, list[tuple[tuple[float, ...], int]]] = defaultdict(list)
    for record in records:
        for question in record.compiled.all_questions:
            if question.variable not in record.gold:
                continue
            answer = str(record.gold[question.variable])
            try:
                gold_index = question.role.answer_index(answer)
            except ValueError:
                continue
            role_scores[question.role.key].append(
                (record.scores[question.question_id].mean_logprobs, gold_index)
            )

    temperatures: dict[str, float] = {}
    role_accuracy: dict[str, float] = {}
    temperature_details: dict[str, dict[str, float | int | bool]] = {}
    for role_key, observations in sorted(role_scores.items()):
        score_rows = [item[0] for item in observations]
        gold_indices = [item[1] for item in observations]
        fit = fit_temperature(score_rows, gold_indices)
        temperatures[role_key] = fit.temperature
        correct = 0
        for scores, gold_index in observations:
            probabilities = calibrated_probabilities(scores, fit.temperature)
            if int(np.argmax(probabilities)) == gold_index:
                correct += 1
        role_accuracy[role_key] = correct / len(observations)
        temperature_details[role_key] = {
            "temperature": fit.temperature,
            "nll": fit.nll,
            "converged": fit.converged,
            "iterations": fit.iterations,
            "observations": len(observations),
        }

    provisional = CalibrationArtifacts(
        temperatures=temperatures,
        role_accuracy=role_accuracy,
        covariances={},
        metadata={},
    )
    residuals_by_signature: dict[str, list[np.ndarray]] = defaultdict(list)
    role_keys_by_signature: dict[str, tuple[str, ...]] = {}
    skipped_incomplete = 0

    for record in records:
        available_probabilities: dict[str, np.ndarray] = {}
        for question in record.compiled.all_questions:
            if question.role.key not in provisional.temperatures:
                continue
            available_probabilities[question.question_id] = calibrated_probabilities(
                record.scores[question.question_id].mean_logprobs,
                provisional.temperature_for(question.role),
            )
        for questions in _question_subsets(record, max_budget):
            try:
                codebook = build_codebook(record.compiled, questions)
            except (KeyError, ValueError):
                continue
            role_keys_by_signature.setdefault(codebook.signature, role_keys(codebook.questions))
            if codebook.codewords.shape[0] == 0:
                continue
            variables = [question.variable for question in questions]
            if any(variable not in record.gold for variable in variables):
                skipped_incomplete += 1
                continue
            if any(question.question_id not in available_probabilities for question in questions):
                skipped_incomplete += 1
                continue
            try:
                assignment = {variable: str(record.gold[variable]) for variable in variables}
                true_code = codebook.encode_assignment(assignment)
                if not any(np.array_equal(true_code, codeword) for codeword in codebook.codewords):
                    skipped_incomplete += 1
                    continue
                observation = stack_observation(codebook.questions, available_probabilities)
            except (KeyError, ValueError):
                skipped_incomplete += 1
                continue
            residuals_by_signature[codebook.signature].append(observation - true_code)

    covariances = {}
    unsupported = 0
    covariance_failures: dict[str, str] = {}
    for signature, signature_role_keys in sorted(role_keys_by_signature.items()):
        residual_rows = residuals_by_signature.get(signature, [])
        if not residual_rows:
            unsupported += 1
            covariance_failures[signature] = "no complete calibration residuals"
            continue
        try:
            covariances[signature] = fit_residual_covariance(
                signature=signature,
                role_keys=signature_role_keys,
                residual_rows=residual_rows,
            )
        except ValueError as exc:
            unsupported += 1
            covariance_failures[signature] = str(exc)

    metadata = {
        "method": "QCV",
        "implementation_version": "1.0.0",
        "max_budget": max_budget,
        "records_seen": len(records),
        "temperature_details": temperature_details,
        "covariance_failures": covariance_failures,
        "covariance_sample_counts": {
            signature: model.sample_count for signature, model in covariances.items()
        },
    }
    artifacts = CalibrationArtifacts(
        temperatures=temperatures,
        role_accuracy=role_accuracy,
        covariances=covariances,
        metadata=metadata,
    )
    summary = CalibrationSummary(
        records_seen=len(records),
        role_models=len(temperatures),
        covariance_models=len(covariances),
        unsupported_signatures=unsupported,
        skipped_incomplete_assignments=skipped_incomplete,
    )
    return artifacts, summary
