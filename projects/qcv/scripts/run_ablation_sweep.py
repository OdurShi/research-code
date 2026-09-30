from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from qcv.calibration import probabilities_for_record
from qcv.inference import InferenceOptions
from qcv.pipeline import evaluate_jsonl, infer_jsonl, load_score_records
from qcv.types import CalibrationArtifacts, Prediction
from qcv.utils import write_jsonl


def write_target_only(scores: Path, artifacts_dir: Path, output: Path) -> None:
    artifacts = CalibrationArtifacts.load(artifacts_dir)
    predictions: list[Prediction] = []
    for record in load_score_records(scores):
        probabilities = probabilities_for_record(record, artifacts)
        target = record.compiled.target
        answer = target.role.alphabet[int(np.argmax(probabilities[target.question_id]))]
        predictions.append(
            Prediction(
                instance_id=record.instance_id,
                covered=True,
                target_only_answer=answer,
                final_answer=answer,
                selected_question_ids=(),
                target_margin=None,
                projection_gap=None,
                target_class_separation=None,
                reason="target_only",
            )
        )
    write_jsonl(output, [prediction.to_dict() for prediction in predictions])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete deterministic QCV ablation and budget sweep")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[3, 5, 7, 9])
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = Path(args.scores)
    artifacts = Path(args.artifacts)
    rows: list[dict] = []

    target_predictions = output_dir / "target-only.predictions.jsonl"
    write_target_only(scores, artifacts, target_predictions)
    target_metrics = evaluate_jsonl(
        scores,
        target_predictions,
        output_dir / "target-only.metrics.json",
        artifact_directory=artifacts,
    )
    rows.append({"method": "target-only", "budget": 1, **target_metrics})

    variants = {
        "fixed-probes": ("fixed", "full"),
        "random-qcv": ("random", "full"),
        "reliability-qcv": ("reliability", "full"),
        "identity-qcv": ("qcv", "identity"),
        "diagonal-qcv": ("qcv", "diagonal"),
        "qcv": ("qcv", "full"),
        "exact-qcv": ("exact", "full"),
    }
    for budget in args.budgets:
        for name, (selection, metric) in variants.items():
            stem = f"{name}.B{budget}"
            prediction_path = output_dir / f"{stem}.predictions.jsonl"
            metric_path = output_dir / f"{stem}.metrics.json"
            infer_jsonl(
                scores,
                artifacts,
                prediction_path,
                InferenceOptions(
                    budget=budget,
                    selection=selection,
                    metric=metric,
                    random_seed=args.random_seed,
                ),
            )
            metrics = evaluate_jsonl(
                scores,
                prediction_path,
                metric_path,
                artifact_directory=artifacts,
            )
            rows.append({"method": name, "budget": budget, **metrics})

    columns = sorted({key for row in rows for key in row})
    with (output_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
