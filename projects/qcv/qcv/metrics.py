from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .calibration import probabilities_for_record
from .types import CalibrationArtifacts, Prediction, ScoreRecord
from .utils import normalize_text


def answers_equal(task: str, prediction: str, gold: str) -> bool:
    pred = normalize_text(str(prediction))
    truth = normalize_text(str(gold))
    if task.lower() == "chartqa":
        try:
            pred_number = Decimal(pred.replace(",", "").replace("%", ""))
            gold_number = Decimal(truth.replace(",", "").replace("%", ""))
        except InvalidOperation:
            return pred == truth
        if gold_number == 0:
            return pred_number == gold_number
        return abs(pred_number - gold_number) <= abs(gold_number) * Decimal("0.05")
    return pred == truth


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    overall_accuracy: float
    coverage: float
    covered_accuracy: float | None
    uncovered_accuracy: float | None
    target_only_accuracy: float
    correction_rate: float | None
    corruption_rate: float | None
    mean_error_correlation: float | None
    effective_measurements: float | None
    correct_measurement_survival: float | None
    margin_error_auroc: float | None
    gap_error_auroc: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "total": self.total,
            "overall_accuracy": self.overall_accuracy,
            "coverage": self.coverage,
            "covered_accuracy": self.covered_accuracy,
            "uncovered_accuracy": self.uncovered_accuracy,
            "target_only_accuracy": self.target_only_accuracy,
            "correction_rate": self.correction_rate,
            "corruption_rate": self.corruption_rate,
            "mean_error_correlation": self.mean_error_correlation,
            "effective_measurements": self.effective_measurements,
            "correct_measurement_survival": self.correct_measurement_survival,
            "margin_error_auroc": self.margin_error_auroc,
            "gap_error_auroc": self.gap_error_auroc,
        }


def _safe_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(labels) < 2 or len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _pairwise_error_correlation(matrix: np.ndarray) -> float | None:
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return None
    correlations: list[float] = []
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            x = matrix[:, left]
            y = matrix[:, right]
            if np.var(x) == 0.0 or np.var(y) == 0.0:
                continue
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return _safe_mean(correlations)


def evaluate_predictions(
    records: Sequence[ScoreRecord],
    predictions: Sequence[Prediction],
    artifacts: CalibrationArtifacts | None = None,
) -> EvaluationSummary:
    by_id = {prediction.instance_id: prediction for prediction in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Prediction IDs are not unique")
    usable_records = [record for record in records if record.target_answer is not None]
    if not usable_records:
        raise ValueError("No target answers are available for evaluation")

    final_correct: list[bool] = []
    target_correct: list[bool] = []
    covered_flags: list[bool] = []
    margin_labels: list[int] = []
    margin_scores: list[float] = []
    gap_labels: list[int] = []
    gap_scores: list[float] = []
    measurement_errors: list[list[int]] = []
    survival_rows: list[float] = []

    for record in usable_records:
        if record.instance_id not in by_id:
            raise KeyError(f"Missing prediction for {record.instance_id}")
        prediction = by_id[record.instance_id]
        gold = str(record.target_answer)
        target_ok = answers_equal(record.task, prediction.target_only_answer, gold)
        final_ok = answers_equal(record.task, prediction.final_answer, gold)
        target_correct.append(target_ok)
        final_correct.append(final_ok)
        covered_flags.append(prediction.covered)

        if prediction.covered and prediction.target_margin is not None:
            margin_labels.append(0 if final_ok else 1)
            margin_scores.append(-float(prediction.target_margin))
        if prediction.covered and prediction.projection_gap is not None:
            gap_labels.append(0 if final_ok else 1)
            gap_scores.append(float(prediction.projection_gap))

        if artifacts is not None and prediction.covered:
            try:
                probabilities = probabilities_for_record(record, artifacts)
            except KeyError:
                continue
            question_ids = (record.compiled.target.question_id,) + prediction.selected_question_ids
            row: list[int] = []
            complete = True
            for question_id in question_ids:
                question = record.compiled.question_by_id(question_id)
                if question.variable not in record.gold:
                    complete = False
                    break
                pred_answer = question.role.alphabet[int(np.argmax(probabilities[question_id]))]
                row.append(0 if answers_equal(record.task, pred_answer, record.gold[question.variable]) else 1)
            if complete:
                measurement_errors.append(row)
                if row[0] == 1 and len(row) > 1:
                    survival_rows.append(float(np.mean([1 - error for error in row[1:]])))

    final_array = np.asarray(final_correct, dtype=bool)
    target_array = np.asarray(target_correct, dtype=bool)
    covered_array = np.asarray(covered_flags, dtype=bool)
    corrected_denom = np.sum(~target_array)
    corrupted_denom = np.sum(target_array)
    correction_rate = (
        float(np.sum(final_array & ~target_array) / corrected_denom)
        if corrected_denom > 0
        else None
    )
    corruption_rate = (
        float(np.sum(~final_array & target_array) / corrupted_denom)
        if corrupted_denom > 0
        else None
    )

    covered_accuracy = (
        float(np.mean(final_array[covered_array])) if np.any(covered_array) else None
    )
    uncovered_accuracy = (
        float(np.mean(final_array[~covered_array])) if np.any(~covered_array) else None
    )

    rho = None
    beff = None
    if measurement_errors:
        widths = {len(row) for row in measurement_errors}
        if len(widths) == 1:
            matrix = np.asarray(measurement_errors, dtype=np.float64)
            rho = _pairwise_error_correlation(matrix)
            if rho is not None:
                budget = matrix.shape[1]
                denominator = 1.0 + (budget - 1) * rho
                beff = float(budget) if denominator <= 0 else min(float(budget), budget / denominator)

    return EvaluationSummary(
        total=len(usable_records),
        overall_accuracy=float(np.mean(final_array)),
        coverage=float(np.mean(covered_array)),
        covered_accuracy=covered_accuracy,
        uncovered_accuracy=uncovered_accuracy,
        target_only_accuracy=float(np.mean(target_array)),
        correction_rate=correction_rate,
        corruption_rate=corruption_rate,
        mean_error_correlation=rho,
        effective_measurements=beff,
        correct_measurement_survival=_safe_mean(survival_rows),
        margin_error_auroc=_safe_auroc(margin_labels, margin_scores),
        gap_error_auroc=_safe_auroc(gap_labels, gap_scores),
    )


def paired_bootstrap_accuracy_difference(
    first_correct: Sequence[bool],
    second_correct: Sequence[bool],
    replicates: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    first = np.asarray(first_correct, dtype=np.float64)
    second = np.asarray(second_correct, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Paired correctness arrays must be one-dimensional and equal length")
    if first.size == 0:
        raise ValueError("Cannot bootstrap an empty comparison")
    generator = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = generator.integers(0, first.size, size=first.size)
        differences[index] = np.mean(first[sample] - second[sample])
    estimate = float(np.mean(first - second))
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return estimate, float(lower), float(upper)


def mcnemar_exact(first_correct: Sequence[bool], second_correct: Sequence[bool]) -> float:
    first = np.asarray(first_correct, dtype=bool)
    second = np.asarray(second_correct, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Paired correctness arrays differ in shape")
    first_only = int(np.sum(first & ~second))
    second_only = int(np.sum(~first & second))
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(first_only, second_only), discordant, 0.5, alternative="two-sided").pvalue)


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(p_value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted
