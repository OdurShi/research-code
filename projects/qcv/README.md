# Question-Code Verification (QCV)

Utilities for question-code verification.
It implements typed question compilation, complete-answer-domain scoring,
temperature calibration, residual-covariance estimation, covariance-whitened
question selection, feasible-code projection, evaluation metrics, and paired
statistical comparisons.

## 1. Package structure

- `qcv/compilers/`: deterministic grammar loading and typed compilation.
- `qcv/configs/`: task grammars for CLEVR, GQA, ChartQA, and AGQA-Decomp.
- `qcv/models/`: multimodal model scoring and media loading.
- `qcv/calibration.py`: role temperatures and residual covariance models.
- `qcv/selection.py`: greedy, exact, fixed, random, and reliability selection.
- `qcv/decoding.py`: covariance-aware feasible-code projection.
- `qcv/metrics.py`: accuracy, coverage, correction, corruption, and geometry metrics.
- `qcv/comparison.py`: paired bootstrap and exact McNemar comparisons.
- `scripts/`: data conversion, alphabet fitting, grammar validation, and ablation sweeps.
- `tests/`: deterministic unit tests for the compiler, codebook, covariance, and decoder.

The archive contains source code and task grammars. Model weights, benchmark
media, annotations, and generated score files are supplied at execution time.

## 2. Installation

Core dependencies:

```bash
cd qcv
python -m pip install -e .
```

Multimodal model scoring dependencies:

```bash
python -m pip install -e ".[vlm]"
```

The implementation requires Python 3.10 or later.

## 3. Input records

Commands consume JSONL files with one record per example. A raw record has the
following structure:

```json
{
  "instance_id": "clevr-val-000001",
  "task": "clevr",
  "question": "How many red cubes are there?",
  "answer": "3",
  "media": {
    "type": "image",
    "path": "data/clevr/images/CLEVR_val_000001.png"
  },
  "gold": {
    "count_a": "3",
    "exists_a": "yes",
    "count_not_a": "4",
    "count_total": "7",
    "exists_not_a": "yes"
  },
  "metadata": {}
}
```

Required fields:

- `instance_id`: globally unique example identifier.
- `task`: grammar identifier (`clevr`, `gqa`, `chartqa`, or `agqa`).
- `question`: target question text.
- `media`: image, video, or explicit frame specification.

Evaluation records also include `answer`. Calibration records include `gold`,
whose keys are the typed variables produced by the compiler. Test-time
selection and decoding do not read gold annotations.

## 4. Data conversion

CLEVR JSON conversion:

```bash
python scripts/convert_clevr.py \
  --questions data/clevr/questions/train_questions.json \
  --scenes data/clevr/scenes/train_scenes.json \
  --images-dir data/clevr/images/train \
  --output data/clevr_train_qcv.jsonl \
  --drop-incomplete-gold
```

Generic JSON or JSONL conversion:

```bash
python scripts/convert_records.py \
  --input data/source.json \
  --output data/test_raw.jsonl \
  --task gqa \
  --id-field question_id \
  --question-field question \
  --answer-field answer \
  --media-field image \
  --media-root data/gqa/images
```

## 5. Experimental workflow

### 5.1 Compile target questions

```bash
qcv compile \
  --input data/calibration_raw.jsonl \
  --output runs/calibration_compiled.jsonl \
  --grammars qcv/configs

qcv compile \
  --input data/test_raw.jsonl \
  --output runs/test_compiled.jsonl \
  --grammars qcv/configs
```

Compilation normalizes text, requires a unique schema match, orders auxiliary
questions by template index and normalized text, and rejects invalid types,
empty alphabets, malformed variables, or non-executable constraints.

### 5.2 Score complete answer domains

```bash
qcv score \
  --backend hf \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --input runs/calibration_compiled.jsonl \
  --output runs/calibration_scores.jsonl

qcv score \
  --backend hf \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --input runs/test_compiled.jsonl \
  --output runs/test_scores.jsonl
```

Candidate answers are teacher-forced in a padded batch. Each candidate score
is the mean log-probability of its answer tokens. Special tokens, padding, and
answer-termination tokens are excluded from the length normalization. A
positive `--max-batch-candidates` value enables memory-bounded chunking.

### 5.3 Fit calibration artifacts

```bash
qcv calibrate \
  --input runs/calibration_scores.jsonl \
  --output-dir runs/calibration_artifacts \
  --max-budget 9
```

The command writes:

- `temperatures.json`: scalar temperatures indexed by role and alphabet.
- `role_accuracy.json`: calibration accuracy used by Reliability-QCV.
- `covariances.npz`: shrinkage covariance and symmetric inverse square root.
- `covariance_index.json`: signature dimensions, sample counts, and shrinkage parameters.
- `metadata.json`: calibration configuration and unsupported signatures.

Residual covariance is estimated only from complete, alphabet-compatible role
signatures. Incomplete signatures are not padded or merged.

### 5.4 Run QCV inference

```bash
qcv infer \
  --input runs/test_scores.jsonl \
  --artifacts runs/calibration_artifacts \
  --output runs/qcv_B5_predictions.jsonl \
  --budget 5 \
  --selection qcv \
  --metric full
```

Strict coverage is enabled by default. A record uses target-only fallback when
compilation fails, fewer than `B-1` valid auxiliaries are available, a required
temperature or covariance model is unavailable, or the feasible codebook is
empty or dimensionally inconsistent.

### 5.5 Evaluate predictions

```bash
qcv evaluate \
  --scores runs/test_scores.jsonl \
  --predictions runs/qcv_B5_predictions.jsonl \
  --artifacts runs/calibration_artifacts \
  --output runs/qcv_B5_metrics.json
```

ChartQA numeric answers use 5% relaxed accuracy. Other tasks use normalized
exact match unless a task-specific evaluator is supplied upstream.

### 5.6 Paired statistical comparison

```bash
qcv compare \
  --scores runs/test_scores.jsonl \
  --first runs/qcv_B5_predictions.jsonl \
  --second runs/diagonal_B5_predictions.jsonl \
  --bootstrap-replicates 10000 \
  --output runs/qcv_vs_diagonal.json
```

### 5.7 Budget and ablation sweep

```bash
python scripts/run_ablation_sweep.py \
  --scores runs/test_scores.jsonl \
  --artifacts runs/calibration_artifacts \
  --output-dir runs/sweep \
  --budgets 3 5 7 9
```

The sweep evaluates Target-only, Fixed-Probes, Random-QCV,
Reliability-QCV, Identity-QCV, Diagonal-QCV, QCV, and Exact-QCV and
writes predictions, metrics, and `sweep.csv`.

## 6. Numeric answer domains

Numeric domains are fixed before test inference. A finite alphabet can be
fitted from calibration annotations with:

```bash
python scripts/fit_alphabets.py \
  --input runs/chartqa_train_compiled.jsonl \
  --base-grammar qcv/configs/chartqa.yaml \
  --output-grammar runs/chartqa_frozen.yaml \
  --derive-chart-arithmetic
```

Calibration and test inference must use the same frozen grammar.

## 7. Determinism and numerical conventions

- The QCV path is deterministic for fixed inputs, score files, and calibration artifacts.
- Covariance estimation, eigendecomposition, whitening, distances, and energies use `float64`.
- Residual covariance uses `S = X^T X / N`.
- Shrinkage uses `alpha = max(alpha_LW, 1/N)`, clipped to `[0, 1]`.
- Target-class and codeword ties follow canonical answer-product order.
- Greedy-selection ties follow compiler order.
- Exact-QCV falls back to greedy selection above the configured subset limit.
- Unsupported signatures trigger target-only fallback rather than pseudoinverse or zero padding.

## 8. Validation

Run the static, unit, and grammar checks from the repository root:

```bash
python -m compileall -q qcv scripts tests
python -m pytest -q
python scripts/validate_grammars.py
```

The validated environment and command results are recorded in `VALIDATION.md`.
File checksums are recorded in `FILE_MANIFEST.sha256` and can be checked with:

```bash
shasum -a 256 -c FILE_MANIFEST.sha256
```

## 9. License

The source code is distributed under the terms in `LICENSE`.
