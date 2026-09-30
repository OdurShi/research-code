# Input and output formats

## Taxonomy JSON

Required top-level fields:

- `labels`: non-empty array.

Optional top-level fields:

- `version`: non-negative integer, default `0`.
- `root_id`: root sentinel, default `__root__`.

Each label contains:

- `id`: unique stable identifier;
- `name`: non-empty human-readable label name;
- `description`: non-empty natural-language description;
- `parent_id`: another label id or `null` for a root child;
- `prior`: positive number. If all priors are omitted, uniform priors are assigned.

Names must be unique after Unicode NFKC normalization, case folding, and whitespace collapse. The graph must be acyclic.

## Stream JSONL

Required per line:

- `text`: input text.

Optional method fields:

- `id`: unique sample identifier; the one-based line number is used when omitted;
- `timestamp`: retained in logs.

All remaining fields are copied into `metadata` and are invisible to the method. Recommended evaluation fields are:

- `oracle_label`;
- `oracle_sample_action`: `predict` or `defer`;
- `event_id`;
- `oracle_action`: `stay`, `add`, or `split`;
- `event_onset`;
- `audit`: boolean marking a disjoint label-mapping audit set;
- `source_partition`.

## Event manifest

```json
{
  "default_horizon": 256,
  "events": [
    {
      "event_id": "add-1",
      "onset_index": 501,
      "end_index": 1500,
      "horizon": 256,
      "action": "add",
      "parent_id": "__root__",
      "acceptable_new_name_sets": [["cash withdrawal fee"]]
    }
  ]
}
```

For split events, include `source_label_id`, `parent_id`, and one or more acceptable unordered child-name sets. For stay intervals, only the interval and action are required.

## Proposer JSON

The proposer must return exactly one object with one field, `candidates`.

Add:

```json
{
  "operation": "add",
  "parent_id": "__root__",
  "source_label_id": null,
  "new_labels": [
    {"name": "card dispute", "description": "disputes involving an unauthorized card purchase"}
  ]
}
```

Binary split:

```json
{
  "operation": "split",
  "parent_id": "__root__",
  "source_label_id": "refund",
  "new_labels": [
    {"name": "merchant refund", "description": "a refund initiated by a merchant"},
    {"name": "card dispute", "description": "a disputed or unauthorized card purchase"}
  ]
}
```

Unknown fields, invalid identifiers, non-leaf split sources, duplicate names, and edits outside the posterior-local neighborhood are rejected.
