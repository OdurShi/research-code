from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from .taxonomy import Taxonomy
from .types import LabelNode
from .utils import normalize_name, read_json, read_jsonl, sha256_hex


@dataclass(frozen=True)
class LabeledExample:
    id: str
    text: str
    label: str
    partition: str | None = None
    timestamp: str | None = None


def load_labeled_examples(
    path: str | Path,
    *,
    text_field: str = "text",
    label_field: str = "label",
    id_field: str = "id",
    partition_field: str | None = "partition",
    timestamp_field: str | None = "timestamp",
) -> list[LabeledExample]:
    rows = read_jsonl(path)
    examples: list[LabeledExample] = []
    for index, row in enumerate(rows):
        if text_field not in row or label_field not in row:
            raise ValueError(f"Row {index + 1} lacks {text_field!r} or {label_field!r}")
        examples.append(
            LabeledExample(
                id=str(row.get(id_field, index)),
                text=str(row[text_field]),
                label=str(row[label_field]),
                partition=(None if partition_field is None or row.get(partition_field) is None else str(row[partition_field])),
                timestamp=(None if timestamp_field is None or row.get(timestamp_field) is None else str(row[timestamp_field])),
            )
        )
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("Input example ids must be unique")
    return examples


def load_label_descriptions(path: str | Path | None, labels: Iterable[str]) -> dict[str, str]:
    label_list = sorted(set(labels))
    if path is None:
        return {label: f"texts belonging to the category {label}" for label in label_list}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Label-description file must be a JSON object")
    result = {}
    for label in label_list:
        if label not in payload:
            raise ValueError(f"Missing description for label {label!r}")
        value = payload[label]
        if isinstance(value, dict):
            value = value.get("description")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid description for label {label!r}")
        result[label] = value.strip()
    return result


def _label_id(label: str) -> str:
    return f"label_{sha256_hex(normalize_name(label))[:16]}"


def _sample_without_replacement_or_cycle(
    rng: np.random.Generator,
    examples: list[LabeledExample],
    count: int,
) -> list[LabeledExample]:
    if not examples:
        raise ValueError("Cannot sample from an empty class")
    if count <= len(examples):
        indices = rng.choice(len(examples), size=count, replace=False)
    else:
        indices = rng.choice(len(examples), size=count, replace=True)
    return [examples[int(index)] for index in indices]


def _interleave_by_prevalence(
    rng: np.random.Generator,
    class_examples: dict[str, list[LabeledExample]],
    length: int,
    probabilities: dict[str, float],
) -> list[LabeledExample]:
    labels = sorted(probabilities)
    probs = np.asarray([probabilities[label] for label in labels], dtype=np.float64)
    probs = probs / probs.sum()
    pools = {label: list(rng.permutation(class_examples[label])) for label in labels}
    cursors = {label: 0 for label in labels}
    result: list[LabeledExample] = []
    for label_index in rng.choice(len(labels), size=length, replace=True, p=probs):
        label = labels[int(label_index)]
        pool = pools[label]
        cursor = cursors[label]
        if cursor >= len(pool):
            pool = list(rng.permutation(class_examples[label]))
            pools[label] = pool
            cursor = 0
        result.append(pool[cursor])
        cursors[label] = cursor + 1
    return result


def _taxonomy_for_labels(labels: list[str], descriptions: dict[str, str], *, version: int = 0) -> Taxonomy:
    return Taxonomy(
        [
            LabelNode(
                id=_label_id(label),
                name=label,
                description=descriptions[label],
                parent_id=None,
                prior=1.0 / len(labels),
            )
            for label in sorted(labels)
        ],
        version=version,
    )


def _stream_row(
    example: LabeledExample,
    *,
    oracle_label: str | None,
    event_id: str,
    oracle_action: str,
    event_onset: int,
    source_partition: str | None = None,
) -> dict[str, Any]:
    return {
        "id": example.id,
        "text": example.text,
        "timestamp": example.timestamp,
        "oracle_label": oracle_label,
        "event_id": event_id,
        "oracle_action": oracle_action,
        "event_onset": event_onset,
        "source_partition": source_partition if source_partition is not None else example.partition,
    }


def _finalize_stream_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign unique deterministic stream ids while retaining source-example ids.

    Controlled streams sample with replacement when a class has fewer source
    observations than requested.  The online loader requires one unique id per
    arrival, so every occurrence receives an index-derived identifier.
    """
    finalized: list[dict[str, Any]] = []
    for stream_index, row in enumerate(rows, start=1):
        source_id = str(row["id"])
        occurrence = dict(row)
        occurrence["source_id"] = source_id
        occurrence["id"] = f"stream-{stream_index:09d}-{sha256_hex(source_id)[:10]}"
        finalized.append(occurrence)
    return finalized


def generate_add_stream(
    examples: list[LabeledExample],
    descriptions: dict[str, str],
    *,
    seed: int,
    initial_kcr: float = 0.75,
    warmup_per_class: int = 10,
    post_length: int = 1000,
    new_class_prevalence: float = 0.10,
    horizon: int = 256,
) -> tuple[Taxonomy, list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < initial_kcr < 1.0 or not 0.0 < new_class_prevalence < 1.0:
        raise ValueError("initial_kcr and new_class_prevalence must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[LabeledExample]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)
    labels = sorted(by_label)
    if len(labels) < 2:
        raise ValueError("Add streams require at least two classes")
    shuffled = list(rng.permutation(labels))
    known_count = min(len(labels) - 1, max(1, int(round(initial_kcr * len(labels)))))
    known = sorted(shuffled[:known_count])
    novel = sorted(shuffled[known_count:])
    new_label = novel[0]
    taxonomy = _taxonomy_for_labels(known, descriptions)
    warmup: list[LabeledExample] = []
    for label in known:
        warmup.extend(_sample_without_replacement_or_cycle(rng, by_label[label], warmup_per_class))
    warmup = list(rng.permutation(warmup))
    event_onset = len(warmup) + 1
    old_prob = (1.0 - new_class_prevalence) / len(known)
    probabilities = {label: old_prob for label in known}
    probabilities[new_label] = new_class_prevalence
    post = _interleave_by_prevalence(
        rng,
        {label: by_label[label] for label in [*known, new_label]},
        post_length,
        probabilities,
    )
    event_id = "add-1"
    rows = [
        _stream_row(
            example,
            oracle_label=example.label,
            event_id="warmup",
            oracle_action="stay",
            event_onset=1,
        )
        for example in warmup
    ]
    rows.extend(
        _stream_row(
            example,
            oracle_label=example.label,
            event_id=event_id,
            oracle_action="add",
            event_onset=event_onset,
        )
        for example in post
    )
    manifest = {
        "stream_type": "add",
        "seed": seed,
        "initial_kcr": initial_kcr,
        "default_horizon": horizon,
        "events": [
            {
                "event_id": event_id,
                "onset_index": event_onset,
                "end_index": event_onset + post_length - 1,
                "horizon": horizon,
                "action": "add",
                "parent_id": taxonomy.root_id,
                "acceptable_new_name_sets": [[new_label]],
                "oracle_new_labels": [new_label],
            }
        ],
    }
    rows = _finalize_stream_rows(rows)
    return taxonomy, rows, manifest


def generate_split_stream(
    examples: list[LabeledExample],
    descriptions: dict[str, str],
    *,
    child_labels: tuple[str, str],
    coarse_name: str,
    coarse_description: str,
    seed: int,
    warmup_per_class: int = 10,
    post_length: int = 1000,
    child_ratio: tuple[int, int] = (1, 1),
    horizon: int = 256,
) -> tuple[Taxonomy, list[dict[str, Any]], dict[str, Any]]:
    left, right = child_labels
    if left == right:
        raise ValueError("Split child labels must differ")
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[LabeledExample]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)
    if left not in by_label or right not in by_label:
        raise ValueError("Both split child labels must be present in the dataset")
    other_labels = sorted(set(by_label) - {left, right})
    coarse_id = f"coarse_{sha256_hex(normalize_name(coarse_name))[:16]}"
    labels = [
        LabelNode(
            id=_label_id(label),
            name=label,
            description=descriptions[label],
            parent_id=None,
            prior=0.0,
        )
        for label in other_labels
    ]
    labels.append(
        LabelNode(
            id=coarse_id,
            name=coarse_name,
            description=coarse_description,
            parent_id=None,
            prior=0.0,
        )
    )
    taxonomy = Taxonomy(labels)
    warmup: list[tuple[LabeledExample, str]] = []
    for label in other_labels:
        for example in _sample_without_replacement_or_cycle(rng, by_label[label], warmup_per_class):
            warmup.append((example, label))
    for child in (left, right):
        for example in _sample_without_replacement_or_cycle(rng, by_label[child], warmup_per_class):
            warmup.append((example, coarse_name))
    warmup = list(rng.permutation(warmup))
    event_onset = len(warmup) + 1
    ratio_total = child_ratio[0] + child_ratio[1]
    structural_prevalence = min(0.20, 2.0 / max(2, len(other_labels) + 1))
    probabilities = {
        label: (1.0 - structural_prevalence) / max(1, len(other_labels))
        for label in other_labels
    }
    if not other_labels:
        probabilities = {}
        structural_prevalence = 1.0
    probabilities[left] = structural_prevalence * child_ratio[0] / ratio_total
    probabilities[right] = structural_prevalence * child_ratio[1] / ratio_total
    post = _interleave_by_prevalence(
        rng,
        {label: by_label[label] for label in [*other_labels, left, right]},
        post_length,
        probabilities,
    )
    event_id = "split-1"
    rows = [
        _stream_row(
            example,
            oracle_label=oracle_label,
            event_id="warmup",
            oracle_action="stay",
            event_onset=1,
        )
        for example, oracle_label in warmup
    ]
    rows.extend(
        _stream_row(
            example,
            oracle_label=example.label,
            event_id=event_id,
            oracle_action="split",
            event_onset=event_onset,
        )
        for example in post
    )
    manifest = {
        "stream_type": "split",
        "seed": seed,
        "default_horizon": horizon,
        "events": [
            {
                "event_id": event_id,
                "onset_index": event_onset,
                "end_index": event_onset + post_length - 1,
                "horizon": horizon,
                "action": "split",
                "source_label_id": coarse_id,
                "parent_id": taxonomy.root_id,
                "acceptable_new_name_sets": [[left, right], [right, left]],
                "oracle_new_labels": [left, right],
            }
        ],
    }
    rows = _finalize_stream_rows(rows)
    return taxonomy, rows, manifest


def generate_prior_drift_stream(
    examples: list[LabeledExample],
    descriptions: dict[str, str],
    *,
    seed: int,
    warmup_length: int = 500,
    post_length: int = 1000,
    head_mass: float = 0.80,
    horizon: int = 256,
) -> tuple[Taxonomy, list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[LabeledExample]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)
    labels = sorted(by_label)
    taxonomy = _taxonomy_for_labels(labels, descriptions)
    uniform = {label: 1.0 / len(labels) for label in labels}
    warmup = _interleave_by_prevalence(rng, by_label, warmup_length, uniform)
    event_onset = len(warmup) + 1
    head_count = max(1, len(labels) // 5)
    head = set(labels[:head_count])
    tail = set(labels) - head
    probabilities = {}
    for label in labels:
        if label in head:
            probabilities[label] = head_mass / len(head)
        else:
            probabilities[label] = (1.0 - head_mass) / max(1, len(tail))
    post = _interleave_by_prevalence(rng, by_label, post_length, probabilities)
    event_id = "prior-drift-1"
    rows = [
        _stream_row(example, oracle_label=example.label, event_id="warmup", oracle_action="stay", event_onset=1)
        for example in warmup
    ]
    rows.extend(
        _stream_row(example, oracle_label=example.label, event_id=event_id, oracle_action="stay", event_onset=event_onset)
        for example in post
    )
    manifest = {
        "stream_type": "prior_drift",
        "seed": seed,
        "default_horizon": horizon,
        "events": [
            {
                "event_id": event_id,
                "onset_index": event_onset,
                "end_index": event_onset + post_length - 1,
                "horizon": horizon,
                "action": "stay",
            }
        ],
    }
    rows = _finalize_stream_rows(rows)
    return taxonomy, rows, manifest


def generate_covariate_drift_stream(
    examples: list[LabeledExample],
    descriptions: dict[str, str],
    *,
    seed: int,
    warmup_length: int = 500,
    post_length: int = 1000,
    horizon: int = 256,
) -> tuple[Taxonomy, list[dict[str, Any]], dict[str, Any]]:
    partitions = sorted({example.partition for example in examples if example.partition is not None})
    if len(partitions) < 2:
        # Deterministic source partition when the dataset has no native split field.
        converted = []
        for example in examples:
            part = "a" if int(sha256_hex(example.id)[:8], 16) % 2 == 0 else "b"
            converted.append(LabeledExample(example.id, example.text, example.label, part, example.timestamp))
        examples = converted
        partitions = ["a", "b"]
    pre_part, post_part = partitions[0], partitions[1]
    rng = np.random.default_rng(seed)
    labels = sorted({example.label for example in examples})
    taxonomy = _taxonomy_for_labels(labels, descriptions)
    pre_by_label: dict[str, list[LabeledExample]] = defaultdict(list)
    post_by_label: dict[str, list[LabeledExample]] = defaultdict(list)
    for example in examples:
        if example.partition == pre_part:
            pre_by_label[example.label].append(example)
        if example.partition == post_part:
            post_by_label[example.label].append(example)
    shared = sorted(label for label in labels if pre_by_label[label] and post_by_label[label])
    if not shared:
        raise ValueError("Covariate drift partitions have no shared labels")
    probabilities = {label: 1.0 / len(shared) for label in shared}
    warmup = _interleave_by_prevalence(rng, pre_by_label, warmup_length, probabilities)
    event_onset = len(warmup) + 1
    post = _interleave_by_prevalence(rng, post_by_label, post_length, probabilities)
    event_id = "covariate-drift-1"
    rows = [
        _stream_row(example, oracle_label=example.label, event_id="warmup", oracle_action="stay", event_onset=1, source_partition=pre_part)
        for example in warmup
    ]
    rows.extend(
        _stream_row(example, oracle_label=example.label, event_id=event_id, oracle_action="stay", event_onset=event_onset, source_partition=post_part)
        for example in post
    )
    manifest = {
        "stream_type": "covariate_drift",
        "seed": seed,
        "default_horizon": horizon,
        "events": [
            {
                "event_id": event_id,
                "onset_index": event_onset,
                "end_index": event_onset + post_length - 1,
                "horizon": horizon,
                "action": "stay",
                "pre_partition": pre_part,
                "post_partition": post_part,
            }
        ],
    }
    rows = _finalize_stream_rows(rows)
    return taxonomy, rows, manifest


def generate_anomaly_stream(
    examples: list[LabeledExample],
    anomaly_examples: list[LabeledExample],
    descriptions: dict[str, str],
    *,
    seed: int,
    warmup_length: int = 500,
    post_length: int = 1000,
    anomaly_count: int = 20,
    horizon: int = 256,
) -> tuple[Taxonomy, list[dict[str, Any]], dict[str, Any]]:
    if anomaly_count < 1 or anomaly_count >= post_length:
        raise ValueError("anomaly_count must lie in [1, post_length)")
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[LabeledExample]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)
    labels = sorted(by_label)
    taxonomy = _taxonomy_for_labels(labels, descriptions)
    probabilities = {label: 1.0 / len(labels) for label in labels}
    warmup = _interleave_by_prevalence(rng, by_label, warmup_length, probabilities)
    post = _interleave_by_prevalence(rng, by_label, post_length - anomaly_count, probabilities)
    anomalies = _sample_without_replacement_or_cycle(rng, anomaly_examples, anomaly_count)
    # Evenly separated anomalies avoid an artificial recurring cluster.
    slots = np.linspace(0, post_length - 1, anomaly_count, dtype=int)
    merged: list[tuple[LabeledExample, bool] | None] = [None] * post_length
    for slot, anomaly in zip(slots, anomalies):
        merged[int(slot)] = (anomaly, True)
    normal_iter = iter(post)
    for index in range(post_length):
        if merged[index] is None:
            merged[index] = (next(normal_iter), False)
    event_onset = len(warmup) + 1
    event_id = "anomaly-1"
    rows = [
        _stream_row(example, oracle_label=example.label, event_id="warmup", oracle_action="stay", event_onset=1)
        for example in warmup
    ]
    for entry in merged:
        assert entry is not None
        example, is_anomaly = entry
        rows.append(
            _stream_row(
                example,
                oracle_label=None if is_anomaly else example.label,
                event_id=event_id,
                oracle_action="stay",
                event_onset=event_onset,
            )
        )
        if is_anomaly:
            rows[-1]["oracle_sample_action"] = "defer"
            rows[-1]["is_anomaly"] = True
    rows = _finalize_stream_rows(rows)
    manifest = {
        "stream_type": "anomaly",
        "seed": seed,
        "default_horizon": horizon,
        "events": [
            {
                "event_id": event_id,
                "onset_index": event_onset,
                "end_index": event_onset + post_length - 1,
                "horizon": horizon,
                "action": "stay",
                "anomaly_ids": sorted(row["id"] for row in rows if row.get("is_anomaly")),
            }
        ],
    }
    return taxonomy, rows, manifest
