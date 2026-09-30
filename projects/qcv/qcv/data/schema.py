from __future__ import annotations

from typing import Any, Mapping


def validate_raw_record(value: Mapping[str, Any]) -> dict[str, Any]:
    instance_id = str(value.get("instance_id", value.get("id", ""))).strip()
    task = str(value.get("task", "")).strip()
    question = str(value.get("question", "")).strip()
    if not instance_id:
        raise ValueError("Raw record requires a non-empty instance_id or id")
    if not task:
        raise ValueError(f"Record {instance_id} requires a task")
    if not question:
        raise ValueError(f"Record {instance_id} requires a question")
    media = value.get("media", {})
    if media and not isinstance(media, Mapping):
        raise ValueError(f"Record {instance_id} media must be an object")
    gold = value.get("gold", {})
    if gold and not isinstance(gold, Mapping):
        raise ValueError(f"Record {instance_id} gold must be an object")
    output = dict(value)
    output["instance_id"] = instance_id
    output["task"] = task
    output["question"] = question
    output["media"] = dict(media)
    output["gold"] = {str(k): str(v) for k, v in dict(gold).items()}
    if value.get("answer") is not None:
        output["answer"] = str(value["answer"])
    return output
