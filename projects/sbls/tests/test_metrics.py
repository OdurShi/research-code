from __future__ import annotations

from sbls.metrics import evaluate_edit_events, fixed_label_f1


def test_never_edit_edit_f1_matches_paper_example() -> None:
    truth = ["stay"] * 604 + ["add"] * 198 + ["split"] * 198
    pred = ["stay"] * 1000
    summary = fixed_label_f1(truth, pred, ["stay", "add", "split"])
    assert round(100 * summary.macro_f1, 1) == 25.1


def test_identity_aware_event_metric() -> None:
    cycles = [
        {
            "cycle_index": 1,
            "proposal_time": 10,
            "stopping_time": 3,
            "decision": "add",
            "selected_candidate_id": "x",
            "candidate_ids": ["x"],
            "proposal_payloads": [
                {
                    "operation": "add",
                    "parent_id": "__root__",
                    "source_label_id": None,
                    "new_labels": [{"name": "novel", "description": "d"}],
                }
            ],
        }
    ]
    manifest = {
        "default_horizon": 256,
        "events": [
            {
                "event_id": "e",
                "onset_index": 10,
                "end_index": 100,
                "action": "add",
                "acceptable_new_name_sets": [["novel"]],
            }
        ],
    }
    result = evaluate_edit_events(cycles, manifest)
    assert result["recall_at_h"] == 100.0
    assert result["events"][0]["identity_correct"] is True


def test_structural_recall_is_limited_to_horizon_but_stay_fep_uses_full_interval():
    late_add_cycle = {
        "proposal_time": 15,
        "stopping_time": 10,
        "cycle_index": 1,
        "decision": "add",
        "selected_candidate_id": "c1",
        "candidate_ids": ["c1"],
        "proposal_payloads": [
            {
                "operation": "add",
                "parent_id": "__root__",
                "source_label_id": None,
                "new_labels": [{"name": "novel", "description": "novel class"}],
            }
        ],
    }
    structural_manifest = {
        "default_horizon": 5,
        "events": [
            {
                "event_id": "e1",
                "onset_index": 10,
                "end_index": 40,
                "horizon": 5,
                "action": "add",
                "parent_id": "__root__",
                "acceptable_new_name_sets": [["novel"]],
            }
        ],
    }
    result = evaluate_edit_events([late_add_cycle], structural_manifest)
    assert result["recall_at_h"] == 0.0
    assert result["edit_f1"] == 0.0

    stay_manifest = {
        "default_horizon": 5,
        "events": [
            {
                "event_id": "s1",
                "onset_index": 10,
                "end_index": 40,
                "horizon": 5,
                "action": "stay",
            }
        ],
    }
    stay_result = evaluate_edit_events([late_add_cycle], stay_manifest)
    assert stay_result["false_edit_probability"] == 100.0
