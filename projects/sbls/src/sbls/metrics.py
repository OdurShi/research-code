from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .utils import normalize_name


DEFER_LABEL = "__defer__"


@dataclass(frozen=True)
class F1Summary:
    per_class: dict[str, float]
    macro_f1: float
    counts: dict[str, dict[str, int]]


def hungarian_label_mapping(
    oracle_labels: Iterable[str],
    predicted_labels: Iterable[str | None],
    *,
    ignore_predicted: set[str] | None = None,
) -> dict[str, str]:
    truth = [str(x) for x in oracle_labels]
    pred = [DEFER_LABEL if x is None else str(x) for x in predicted_labels]
    if len(truth) != len(pred):
        raise ValueError("oracle_labels and predicted_labels must have equal length")
    ignored = ignore_predicted or {DEFER_LABEL}
    true_values = sorted(set(truth))
    pred_values = sorted(set(pred) - ignored)
    if not true_values or not pred_values:
        return {}
    true_index = {label: index for index, label in enumerate(true_values)}
    pred_index = {label: index for index, label in enumerate(pred_values)}
    contingency = np.zeros((len(pred_values), len(true_values)), dtype=np.int64)
    for true_label, pred_label in zip(truth, pred):
        if pred_label in pred_index:
            contingency[pred_index[pred_label], true_index[true_label]] += 1
    rows, columns = linear_sum_assignment(-contingency)
    return {pred_values[row]: true_values[column] for row, column in zip(rows, columns)}


def fixed_label_f1(
    oracle_labels: Iterable[str],
    predicted_labels: Iterable[str],
    labels: Iterable[str],
) -> F1Summary:
    truth = list(oracle_labels)
    pred = list(predicted_labels)
    classes = list(labels)
    if len(truth) != len(pred):
        raise ValueError("oracle and predicted arrays must have equal length")
    per_class: dict[str, float] = {}
    counts: dict[str, dict[str, int]] = {}
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(truth, pred))
        fp = sum(t != label and p == label for t, p in zip(truth, pred))
        fn = sum(t == label and p != label for t, p in zip(truth, pred))
        denominator = 2 * tp + fp + fn
        f1 = 0.0 if denominator == 0 else (2.0 * tp) / denominator
        per_class[label] = f1
        counts[label] = {"tp": int(tp), "fp": int(fp), "fn": int(fn)}
    macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return F1Summary(per_class=per_class, macro_f1=macro, counts=counts)


def evaluate_prequential_predictions(
    predictions: list[dict[str, Any]],
    *,
    mapping_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    usable = [row for row in predictions if row.get("metadata", {}).get("oracle_label") is not None]
    if not usable:
        raise ValueError("Predictions contain no metadata.oracle_label values")
    mapping_source = mapping_records or [
        row for row in usable if bool(row.get("metadata", {}).get("audit", False))
    ]
    mapping_scope = "disjoint_audit"
    if not mapping_source:
        mapping_source = usable
        mapping_scope = "evaluation_stream_fallback"
    mapping = hungarian_label_mapping(
        [str(row["metadata"]["oracle_label"]) for row in mapping_source],
        [row.get("predicted_label_id") for row in mapping_source],
    )
    truth = [str(row["metadata"]["oracle_label"]) for row in usable]
    mapped = [
        DEFER_LABEL
        if row.get("predicted_label_id") is None
        else mapping.get(str(row["predicted_label_id"]), f"__unmapped__:{row['predicted_label_id']}")
        for row in usable
    ]
    classes = sorted(set(truth))
    summary = fixed_label_f1(truth, mapped, classes)
    accuracy = float(np.mean([t == p for t, p in zip(truth, mapped)]))
    return {
        "prequential_macro_f1": 100.0 * summary.macro_f1,
        "prequential_accuracy": 100.0 * accuracy,
        "per_class_f1": {key: 100.0 * value for key, value in summary.per_class.items()},
        "label_mapping": mapping,
        "mapping_scope": mapping_scope,
        "evaluated_examples": len(usable),
        "deferred_examples": sum(value == DEFER_LABEL for value in mapped),
    }


def _selected_payload(cycle: dict[str, Any]) -> dict[str, Any] | None:
    selected = cycle.get("selected_candidate_id")
    if selected is None:
        return None
    ids = cycle.get("candidate_ids", [])
    payloads = cycle.get("proposal_payloads", [])
    for candidate_id, payload in zip(ids, payloads):
        if candidate_id == selected:
            return payload
    return None


def _edit_identity_matches(event: dict[str, Any], cycle: dict[str, Any]) -> bool:
    if cycle.get("decision") != event.get("action"):
        return False
    if event.get("action") == "stay":
        return cycle.get("decision") == "stay"
    payload = _selected_payload(cycle)
    if payload is None:
        return False
    expected_source = event.get("source_label_id")
    if expected_source is not None and payload.get("source_label_id") != expected_source:
        return False
    expected_parent = event.get("parent_id")
    if expected_parent is not None and payload.get("parent_id") != expected_parent:
        return False
    acceptable_sets = event.get("acceptable_new_name_sets")
    if acceptable_sets:
        actual = {normalize_name(item["name"]) for item in payload.get("new_labels", [])}
        expected = [
            {normalize_name(name) for name in name_set}
            for name_set in acceptable_sets
        ]
        if actual not in expected:
            return False
    return True


def evaluate_edit_events(cycles: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    events = manifest.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Manifest must contain a non-empty events array")
    ordered_cycles = sorted(cycles, key=lambda row: (int(row["proposal_time"]), int(row["cycle_index"])))
    true_actions: list[str] = []
    predicted_actions: list[str] = []
    identity_correct: list[bool] = []
    structural_total = 0
    structural_detected = 0
    delays: list[int] = []
    false_edit_intervals = 0
    no_edit_intervals = 0
    event_rows: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event["event_id"])
        action = str(event["action"])
        if action not in {"stay", "add", "split"}:
            raise ValueError(f"Unsupported oracle action in event {event_id}: {action}")
        onset = int(event["onset_index"])
        horizon = int(event.get("horizon", manifest.get("default_horizon", 256)))
        end = int(event.get("end_index", onset + horizon))
        interval_cycles = [
            cycle
            for cycle in ordered_cycles
            if onset <= int(cycle["proposal_time"]) + int(cycle.get("stopping_time") or 0) <= end
        ]
        detection_end = min(end, onset + horizon)
        candidate_cycles = [
            cycle
            for cycle in interval_cycles
            if int(cycle["proposal_time"]) + int(cycle.get("stopping_time") or 0) <= detection_end
        ]
        committed = [cycle for cycle in interval_cycles if cycle.get("decision") in {"add", "split"}]
        if action == "stay":
            no_edit_intervals += 1
            if committed:
                chosen = committed[0]
                predicted = str(chosen["decision"])
                correct = False
                false_edit_intervals += 1
            else:
                chosen = None
                predicted = "stay"
                correct = True
        else:
            structural_total += 1
            chosen = None
            correct = False
            predicted = "stay"
            for cycle in candidate_cycles:
                if cycle.get("decision") in {"add", "split"}:
                    chosen = cycle
                    predicted = str(cycle["decision"])
                    correct = _edit_identity_matches(event, cycle)
                    break
            if correct and chosen is not None:
                structural_detected += 1
                decision_time = int(chosen["proposal_time"]) + int(chosen.get("stopping_time") or 0)
                delays.append(decision_time - onset)
        true_actions.append(action)
        predicted_actions.append(predicted)
        identity_correct.append(correct)
        event_rows.append(
            {
                "event_id": event_id,
                "oracle_action": action,
                "predicted_action": predicted,
                "identity_correct": correct,
                "matched_cycle_index": None if chosen is None else chosen.get("cycle_index"),
            }
        )
    # Identity-aware confusion: an incorrect decision contributes an FN for the true class
    # and an FP for the emitted class, including stay. This matches the Never-edit behavior.
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in ("stay", "add", "split")}
    for truth, pred, correct in zip(true_actions, predicted_actions, identity_correct):
        if correct:
            counts[truth]["tp"] += 1
        else:
            counts[truth]["fn"] += 1
            counts[pred]["fp"] += 1
    per_class = {}
    for label, values in counts.items():
        denominator = 2 * values["tp"] + values["fp"] + values["fn"]
        per_class[label] = 0.0 if denominator == 0 else 2 * values["tp"] / denominator
    edit_f1 = float(np.mean(list(per_class.values())))
    return {
        "edit_f1": 100.0 * edit_f1,
        "per_action_f1": {key: 100.0 * value for key, value in per_class.items()},
        "confusion_counts": counts,
        "recall_at_h": 0.0 if structural_total == 0 else 100.0 * structural_detected / structural_total,
        "mean_detection_delay": None if not delays else float(np.mean(delays)),
        "median_detection_delay": None if not delays else float(np.median(delays)),
        "false_edit_probability": 0.0 if no_edit_intervals == 0 else 100.0 * false_edit_intervals / no_edit_intervals,
        "events": event_rows,
    }


def old_class_consistency(before: Iterable[str | None], after: Iterable[str | None]) -> float:
    before_list = list(before)
    after_list = list(after)
    if len(before_list) != len(after_list):
        raise ValueError("before and after predictions must have equal length")
    if not before_list:
        return 0.0
    return 100.0 * float(np.mean([left == right for left, right in zip(before_list, after_list)]))


def normalized_leaf_edit_distance(predicted_leaves: Iterable[str], oracle_leaves: Iterable[str]) -> float:
    pred = {normalize_name(value) for value in predicted_leaves}
    oracle = {normalize_name(value) for value in oracle_leaves}
    return 100.0 * len(pred.symmetric_difference(oracle)) / max(1, len(oracle))
