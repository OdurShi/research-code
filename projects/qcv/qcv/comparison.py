from __future__ import annotations

from pathlib import Path
from typing import Any

from .metrics import answers_equal, mcnemar_exact, paired_bootstrap_accuracy_difference
from .pipeline import load_predictions, load_score_records
from .utils import json_dump


def compare_prediction_files(
    score_path: str | Path,
    first_prediction_path: str | Path,
    second_prediction_path: str | Path,
    output_path: str | Path,
    bootstrap_replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    records = load_score_records(score_path)
    first = {item.instance_id: item for item in load_predictions(first_prediction_path)}
    second = {item.instance_id: item for item in load_predictions(second_prediction_path)}
    first_correct: list[bool] = []
    second_correct: list[bool] = []
    for record in records:
        if record.target_answer is None:
            continue
        if record.instance_id not in first or record.instance_id not in second:
            raise KeyError(f"Missing paired prediction for {record.instance_id}")
        first_correct.append(
            answers_equal(record.task, first[record.instance_id].final_answer, record.target_answer)
        )
        second_correct.append(
            answers_equal(record.task, second[record.instance_id].final_answer, record.target_answer)
        )
    estimate, lower, upper = paired_bootstrap_accuracy_difference(
        first_correct,
        second_correct,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    result = {
        "paired_examples": len(first_correct),
        "accuracy_difference": estimate,
        "bootstrap_95_ci": [lower, upper],
        "mcnemar_exact_p": mcnemar_exact(first_correct, second_correct),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }
    json_dump(output_path, result)
    return result
