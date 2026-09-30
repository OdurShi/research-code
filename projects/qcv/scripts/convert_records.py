from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from qcv.utils import read_jsonl, write_jsonl


def get_path(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if isinstance(current, list):
            current = current[int(component)]
        else:
            current = current[component]
    return current


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from read_jsonl(path)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        yield from payload
        return
    if isinstance(payload, dict):
        for key in ("data", "questions", "annotations", "records"):
            if key in payload and isinstance(payload[key], list):
                yield from payload[key]
                return
    raise ValueError("Input JSON must be a list or contain data/questions/annotations/records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Map a JSON/JSONL dataset into the exact QCV raw-record schema")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--id-field", required=True)
    parser.add_argument("--question-field", required=True)
    parser.add_argument("--answer-field", required=True)
    parser.add_argument("--media-field", required=True)
    parser.add_argument("--media-root", default="")
    parser.add_argument("--media-type", choices=["image", "video"], default="image")
    parser.add_argument("--gold-sidecar", help="JSON object mapping instance IDs to variable-answer maps")
    args = parser.parse_args()

    sidecar: dict[str, dict[str, Any]] = {}
    if args.gold_sidecar:
        raw_sidecar = json.loads(Path(args.gold_sidecar).read_text(encoding="utf-8"))
        sidecar = {str(key): dict(value) for key, value in raw_sidecar.items()}

    media_root = Path(args.media_root) if args.media_root else None
    output: list[dict[str, Any]] = []
    for record in load_records(Path(args.input)):
        instance_id = str(get_path(record, args.id_field))
        media_value = str(get_path(record, args.media_field))
        media_path = str(media_root / media_value) if media_root is not None else media_value
        output.append(
            {
                "instance_id": instance_id,
                "task": args.task,
                "question": str(get_path(record, args.question_field)),
                "answer": str(get_path(record, args.answer_field)),
                "media": {"type": args.media_type, "path": media_path},
                "gold": {str(k): str(v) for k, v in sidecar.get(instance_id, {}).items()},
                "metadata": {"source_record": record},
            }
        )
    write_jsonl(args.output, output)
    print(json.dumps({"written": len(output)}, indent=2))


if __name__ == "__main__":
    main()
