#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sbls.metrics import evaluate_edit_events, evaluate_prequential_predictions
from sbls.utils import read_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and aggregate SBLS run directories")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for metadata_path in sorted(args.runs_root.rglob("run_metadata.json")):
        run_dir = metadata_path.parent
        relative = run_dir.relative_to(args.runs_root)
        if len(relative.parts) < 3:
            continue
        dataset, event, backbone = relative.parts[-3:]
        predictions = read_jsonl(run_dir / "predictions.jsonl")
        cycles = read_jsonl(run_dir / "cycles.jsonl")
        manifest_path = args.prepared_root / dataset / event / "manifest.json"
        metrics = {}
        if any(row.get("metadata", {}).get("oracle_label") is not None for row in predictions):
            metrics.update(evaluate_prequential_predictions(predictions))
        if manifest_path.exists():
            metrics.update(evaluate_edit_events(cycles, read_json(manifest_path)))
        rows.append(
            {
                "dataset": dataset,
                "event": event,
                "backbone": backbone,
                "prequential_macro_f1": metrics.get("prequential_macro_f1"),
                "edit_f1": metrics.get("edit_f1"),
                "false_edit_probability": metrics.get("false_edit_probability"),
                "recall_at_h": metrics.get("recall_at_h"),
                "mean_detection_delay": metrics.get("mean_detection_delay"),
            }
        )
    if not rows:
        raise RuntimeError("No completed runs were found")
    frame = pd.DataFrame(rows).sort_values(["dataset", "event", "backbone"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = (
        frame.groupby("backbone", dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("backbone")
    )
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"), index=False)
    print(json.dumps({"runs": len(frame), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
