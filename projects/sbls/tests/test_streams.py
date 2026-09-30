from __future__ import annotations

from sbls.io import load_stream
from sbls.streams import LabeledExample, generate_add_stream, generate_anomaly_stream
from sbls.utils import write_jsonl


def test_generated_stream_ids_are_unique_even_when_source_examples_repeat(tmp_path):
    examples = [
        LabeledExample(id="a0", text="alpha", label="a"),
        LabeledExample(id="b0", text="beta", label="b"),
    ]
    descriptions = {"a": "alpha texts", "b": "beta texts"}
    _, rows, _ = generate_add_stream(
        examples,
        descriptions,
        seed=3,
        initial_kcr=0.5,
        warmup_per_class=4,
        post_length=20,
        new_class_prevalence=0.25,
    )
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert all("source_id" in row for row in rows)
    path = tmp_path / "stream.jsonl"
    write_jsonl(path, rows)
    assert len(list(load_stream(path))) == len(rows)


def test_anomaly_marking_is_not_confused_by_colliding_source_ids():
    normal = [
        LabeledExample(id="0", text="normal zero", label="known"),
        LabeledExample(id="1", text="normal one", label="known"),
    ]
    anomalies = [LabeledExample(id="0", text="off taxonomy", label="other")]
    descriptions = {"known": "known texts"}
    _, rows, manifest = generate_anomaly_stream(
        normal,
        anomalies,
        descriptions,
        seed=7,
        warmup_length=4,
        post_length=10,
        anomaly_count=2,
    )
    marked = [row for row in rows if row.get("is_anomaly")]
    assert len(marked) == 2
    assert all(row["text"] == "off taxonomy" for row in marked)
    assert all(row["oracle_label"] is None for row in marked)
    assert set(manifest["events"][0]["anomaly_ids"]) == {row["id"] for row in marked}
    assert any(row["source_id"] == "0" and not row.get("is_anomaly") for row in rows)
