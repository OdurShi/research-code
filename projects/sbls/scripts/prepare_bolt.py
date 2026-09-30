#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


TEXT_FIELDS = ("text", "sentence", "utterance", "content", "question", "title", "document")
LABEL_FIELDS = ("label", "category", "intent", "class", "target", "topic")
ID_FIELDS = ("id", "uid", "index", "example_id")


def read_any(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        rows = []
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            for key in ("data", "examples", "train", "test", "validation", "items"):
                candidate = value.get(key)
                if isinstance(candidate, list) and all(isinstance(row, dict) for row in candidate):
                    return candidate
            # Some datasets use {label: [texts]}.
            if all(isinstance(items, list) for items in value.values()):
                rows = []
                for label, items in value.items():
                    for index, item in enumerate(items):
                        if isinstance(item, str):
                            rows.append({"text": item, "label": label, "id": f"{label}:{index}"})
                        elif isinstance(item, dict):
                            rows.append({**item, "label": item.get("label", label)})
                if rows:
                    return rows
        raise ValueError(f"Unsupported JSON structure in {path}")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported file type: {path}")


def detect_field(rows: list[dict[str, Any]], candidates: Iterable[str], explicit: str | None) -> str:
    if explicit is not None:
        if not any(explicit in row for row in rows):
            raise ValueError(f"Explicit field {explicit!r} was not found")
        return explicit
    counts = {field: sum(field in row and row[field] not in (None, "") for row in rows) for field in candidates}
    field, count = max(counts.items(), key=lambda item: item[1])
    if count == 0:
        raise ValueError(f"Could not detect a field among {tuple(candidates)}")
    return field


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a BOLT dataset directory to canonical SBLS JSONL")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-field")
    parser.add_argument("--label-field")
    parser.add_argument("--id-field")
    parser.add_argument("--description-file", type=Path)
    args = parser.parse_args()

    roots = [args.dataset_dir / "origin_data", args.dataset_dir]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files = sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() in {".json", ".jsonl", ".csv", ".tsv"}
            )
            if files:
                break
    if not files:
        raise FileNotFoundError(f"No supported data files found under {args.dataset_dir}")

    canonical: list[dict[str, Any]] = []
    for source in files:
        try:
            rows = read_any(source)
        except ValueError:
            continue
        if not rows:
            continue
        text_field = detect_field(rows, TEXT_FIELDS, args.text_field)
        label_field = detect_field(rows, LABEL_FIELDS, args.label_field)
        id_field = args.id_field
        if id_field is None:
            available = [field for field in ID_FIELDS if any(field in row for row in rows)]
            id_field = available[0] if available else None
        split_name = source.stem
        for index, row in enumerate(rows):
            text = row.get(text_field)
            label = row.get(label_field)
            if text is None or label is None:
                continue
            sample_id = row.get(id_field) if id_field is not None else None
            if sample_id is None:
                sample_id = f"{source.relative_to(args.dataset_dir)}:{index}"
            canonical.append(
                {
                    "id": str(sample_id),
                    "text": str(text),
                    "label": str(label),
                    "partition": str(row.get("partition", row.get("split", split_name))),
                }
            )
    if not canonical:
        raise RuntimeError("No canonical examples could be extracted")
    # Deterministically disambiguate duplicate ids without dropping data.
    seen: dict[str, int] = {}
    for row in canonical:
        base = row["id"]
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        if occurrence:
            row["id"] = f"{base}#{occurrence}"

    labels = sorted({row["label"] for row in canonical})
    if args.description_file:
        descriptions = json.loads(args.description_file.read_text(encoding="utf-8"))
        if not isinstance(descriptions, dict):
            raise ValueError("Description file must contain a JSON object")
        missing = [label for label in labels if label not in descriptions]
        if missing:
            raise ValueError(f"Descriptions are missing {len(missing)} labels: {missing[:10]}")
        descriptions = {
            label: (
                descriptions[label]["description"]
                if isinstance(descriptions[label], dict)
                else descriptions[label]
            )
            for label in labels
        }
    else:
        descriptions = {
            label: "texts in the category " + label.replace("_", " ").replace("-", " ")
            for label in labels
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in canonical:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "descriptions.json").write_text(
        json.dumps(descriptions, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(
            {
                "source_files": [str(path) for path in files],
                "examples": len(canonical),
                "labels": len(labels),
                "label_counts": {
                    label: sum(row["label"] == label for row in canonical) for label in labels
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
