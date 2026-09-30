from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .calibration import fit_calibration
from .compilers import CompilerRegistry
from .inference import InferenceOptions, infer_record
from .metrics import evaluate_predictions
from .models import HuggingFaceVLMScorer, ModelLoadOptions
from .types import Prediction, QuestionScores, ScoreRecord
from .utils import json_dump, read_jsonl, write_jsonl


def compile_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    grammar_directory: str | Path,
    keep_uncovered: bool = True,
) -> dict[str, int]:
    registry = CompilerRegistry()
    registry.register_directory(grammar_directory)
    output: list[dict[str, Any]] = []
    compiled_count = 0
    rejected_count = 0
    for row in read_jsonl(input_path):
        instance_id = str(row.get("instance_id", row.get("id", "")))
        task = str(row.get("task", ""))
        question = str(row.get("question", ""))
        if not instance_id or not task or not question:
            raise ValueError("Each raw row requires instance_id/id, task, and question")
        result = registry.compiler_for(task).compile(instance_id, question)
        enriched = dict(row)
        enriched["instance_id"] = instance_id
        enriched["compile_status"] = result.reason
        enriched["matching_rules"] = list(result.matching_rules)
        if result.compiled is not None:
            enriched["compiled"] = result.compiled.to_dict()
            output.append(enriched)
            compiled_count += 1
        else:
            rejected_count += 1
            if keep_uncovered:
                output.append(enriched)
    write_jsonl(output_path, output)
    return {"compiled": compiled_count, "rejected": rejected_count, "written": len(output)}


def score_jsonl_hf(
    input_path: str | Path,
    output_path: str | Path,
    model_options: ModelLoadOptions,
) -> dict[str, int]:
    scorer = HuggingFaceVLMScorer(model_options)
    output: list[dict[str, Any]] = []
    skipped = 0
    for row in read_jsonl(input_path):
        if "compiled" not in row:
            skipped += 1
            continue
        from .types import CompiledInstance

        compiled = CompiledInstance.from_dict(row["compiled"])
        media = dict(row.get("media", {}))
        if not media:
            raise ValueError(f"Record {compiled.instance_id} lacks media information")
        scores: dict[str, QuestionScores] = {}
        latencies: dict[str, float] = {}
        for question in compiled.all_questions:
            start = time.perf_counter()
            values = scorer.score_question(question, media)
            latencies[question.question_id] = time.perf_counter() - start
            scores[question.question_id] = QuestionScores(question.question_id, values)
        record = ScoreRecord(
            instance_id=compiled.instance_id,
            task=compiled.task,
            compiled=compiled,
            scores=scores,
            gold={str(k): str(v) for k, v in row.get("gold", {}).items()},
            target_answer=(None if row.get("answer") is None else str(row["answer"])),
            media=media,
            metadata={
                **dict(row.get("metadata", {})),
                "model": model_options.model_name_or_path,
                "dtype": model_options.dtype,
                "question_latency_seconds": latencies,
            },
        )
        output.append(record.to_dict())
    write_jsonl(output_path, output)
    return {"scored": len(output), "skipped": skipped}


def load_score_records(path: str | Path) -> list[ScoreRecord]:
    return [ScoreRecord.from_dict(row) for row in read_jsonl(path)]


def load_predictions(path: str | Path) -> list[Prediction]:
    return [Prediction.from_dict(row) for row in read_jsonl(path)]


def calibrate_jsonl(
    input_path: str | Path,
    output_directory: str | Path,
    max_budget: int,
) -> dict[str, Any]:
    records = load_score_records(input_path)
    artifacts, summary = fit_calibration(records, max_budget=max_budget)
    artifacts.save(output_directory)
    result = {
        "records_seen": summary.records_seen,
        "role_models": summary.role_models,
        "covariance_models": summary.covariance_models,
        "unsupported_signatures": summary.unsupported_signatures,
        "skipped_incomplete_assignments": summary.skipped_incomplete_assignments,
    }
    json_dump(Path(output_directory) / "calibration_summary.json", result)
    return result


def infer_jsonl(
    input_path: str | Path,
    artifact_directory: str | Path,
    output_path: str | Path,
    options: InferenceOptions,
) -> dict[str, int]:
    from .types import CalibrationArtifacts

    artifacts = CalibrationArtifacts.load(artifact_directory)
    records = load_score_records(input_path)
    predictions = [infer_record(record, artifacts, options) for record in records]
    write_jsonl(output_path, [prediction.to_dict() for prediction in predictions])
    covered = sum(prediction.covered for prediction in predictions)
    return {"predictions": len(predictions), "covered": covered, "uncovered": len(predictions) - covered}


def evaluate_jsonl(
    score_path: str | Path,
    prediction_path: str | Path,
    output_path: str | Path,
    artifact_directory: str | Path | None = None,
) -> dict[str, Any]:
    from .types import CalibrationArtifacts

    records = load_score_records(score_path)
    predictions = load_predictions(prediction_path)
    artifacts = None if artifact_directory is None else CalibrationArtifacts.load(artifact_directory)
    summary = evaluate_predictions(records, predictions, artifacts=artifacts)
    result = summary.to_dict()
    json_dump(output_path, result)
    return result
