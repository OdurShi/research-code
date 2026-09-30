# SBLS: Sequential Bayesian Label-Space Selection

This package implements sequential label-space selection. It includes deployed-taxonomy inference, parameter-free deferral, posterior-local orthogonal residualization, Rayleigh–Ritz spectral ranking, deterministic `add` and binary-`split` proposal validation, future-only Bayesian evidence accumulation, continual error spending, early stopping, taxonomy updates, disk caching, structured logging, controlled-stream construction, and evaluation.

## Implementation map

| Component | Implementation |
|---|---|
| Taxonomy mixture and posterior, Eq. (1) | `src/sbls/scoring.py` |
| Parameter-free deferral, Eq. (2) | `classify_or_defer` |
| Posterior-local projector and residual, Eq. (3) | `src/sbls/discovery.py` |
| Canonical leading direction and spectral ranking, Eq. (4), App. A.1 | `canonical_leading_direction`, `rank_residuals` |
| Frozen `stay`/`add`/binary-`split` alternatives, Eqs. (5)–(6) | `src/sbls/candidates.py`, `taxonomy.py` |
| Future-only likelihood ratios and mixture e-process, Eq. (7) | `SequentialBayesianSelector` |
| Error spending and floor-aware boundary, Eq. (8) | `cycle_error_level`, `cycle_boundary` |
| Deterministic winner selection, Eq. (9) | `SequentialBayesianSelector.update` |
| End-to-end Algorithm 1 | `src/sbls/engine.py` |
| BGE-M3 residual encoder | `src/sbls/encoding.py` |
| Qwen proposer with greedy decoding and strict JSON parser | `src/sbls/proposer.py`, `candidates.py` |
| FP32 likelihood/e-process accumulation and disk cache | `scoring.py`, `cache.py` |
| Controlled add/split/drift/anomaly streams | `src/sbls/streams.py` |
| Prequential F1, identity-aware Edit-F1, FEP, Recall@H, delay | `src/sbls/metrics.py` |
| Gaussian BOCPD baseline core | `src/sbls/baselines.py` |

## Installation

Python 3.10+ and an NVIDIA GPU are recommended. Large-model runs require suitable GPU memory; keep only one large model resident at a time.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[nf4,plots]"
```

For systems where `bitsandbytes` is unavailable, set `proposer.quantization: none` in the YAML configuration.

Model access may require accepting the corresponding Hugging Face licenses and authenticating with `huggingface-cli login`.

For fixed-version experiments, replace each `revision: main` and `tokenizer_revision: main` value with an immutable model revision. The runner records both the requested and runtime-resolved revisions.

## Input files

### Initial taxonomy

```json
{
  "version": 0,
  "root_id": "__root__",
  "labels": [
    {
      "id": "payment",
      "name": "payment",
      "description": "questions about making or completing a payment",
      "parent_id": null,
      "prior": 0.3333333333333333
    }
  ]
}
```

Priors may be omitted for every label; the loader then assigns a uniform prior. Explicit priors must all be positive and are normalized once.

### Unlabeled stream

Only `id` and `text` are consumed by SBLS. Other fields are retained as evaluator metadata.

```json
{"id":"1","text":"The refund was successful","timestamp":"2026-01-01T00:00:00Z","oracle_label":"refund"}
{"id":"2","text":"Dispute this unauthorized card purchase","timestamp":"2026-01-01T00:01:00Z","oracle_label":"card dispute"}
```

See `docs/input_formats.md` for the complete schemas.

## Run SBLS

```bash
sbls run \
  --config configs/default.yaml \
  --taxonomy /path/to/initial_taxonomy.json \
  --stream /path/to/stream.jsonl \
  --output runs/banking77_add_seed13
```

The run directory contains:

- `predictions.jsonl`: per-input deployed prediction or deferral;
- `proposal_attempts.jsonl`: ranked proposal batches, raw proposer outputs, and validation audit;
- `cycles.jsonl`: candidate identifiers, error budget, complete log-likelihood-ratio/e-process trajectory, decision, and stopping time;
- `final_taxonomy.json`;
- `config.resolved.yaml`;
- `run_metadata.json`.

## Generate controlled streams

Add event:

```bash
sbls generate-controlled \
  --event add \
  --data data/banking77.jsonl \
  --descriptions data/banking77_descriptions.json \
  --output prepared/banking77_add \
  --seed 13
```

Binary split:

```bash
sbls generate-controlled \
  --event split \
  --data data/banking77.jsonl \
  --descriptions data/banking77_descriptions.json \
  --split-left card_payment_fee_charged \
  --split-right cash_withdrawal_fee_charged \
  --coarse-name transaction fee \
  --coarse-description "fees charged for card payments or cash withdrawals" \
  --output prepared/banking77_split
```

The generator writes `initial_taxonomy.json`, `stream.jsonl`, and `manifest.json`.

## Evaluate

```bash
sbls evaluate \
  --run-dir runs/banking77_add_seed13 \
  --manifest prepared/banking77_add/manifest.json
```

Generated labels are aligned to oracle labels with a Hungarian assignment. If records marked `metadata.audit=true` are present, the mapping is estimated exclusively from that disjoint audit set. Otherwise, the evaluator estimates the mapping from the evaluation stream and records the mapping scope in the metrics output.

## Tests

```bash
PYTHONPATH=src pytest
```

The test suite covers prior-preserving edits, strict candidate validation, SVD rank handling, repeated-eigenvalue canonicalization, spectral tie-breaking, continual error spending, e-process crossing, identity-aware Edit-F1, and an end-to-end taxonomy update with deterministic test fixtures.

## Important numerical details

- Evidence scores only the input-text continuation and termination tokens; conditioning tokens are excluded.
- Short sequences terminate with EOS. Truncated sequences use `evidence.truncation_token_id` when supplied, otherwise the tokenization of `evidence.truncation_text`.
- The same scored continuation is used for every candidate taxonomy in a cycle.
- Mixture likelihoods use FP32 log-sum-exp; likelihood-ratio and e-process state are stored in log space.
- SVD rank uses `eps * max(rows, columns) * sigma_1`.
- Equal spectral scores are ordered by arrival time and then SHA-256 of the token sequence.
- Equal candidate likelihood ratios are resolved by the smallest canonical SHA-256 identifier.
- Proposal and validation observations are never reused within the same cycle.
- Invalid proposal JSON is not retried. Invalid numerical evidence closes the active cycle with `stay` and is logged.

## License

The source code is provided under the BSD-3-Clause license. Model weights, datasets, BOLT assets, and third-party components are governed by their respective licenses.
