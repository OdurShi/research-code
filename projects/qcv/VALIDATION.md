# Validation report

Validation date: 2026-08-01

## Environment

- Python 3.12.13
- NumPy 2.3.5
- SciPy 1.17.0
- scikit-learn 1.8.0
- PyYAML 6.0.3
- Pillow 12.2.0
- pytest 9.0.2

## Commands

The following commands were executed from the repository root:

```bash
python -m pip install -e .
python -m compileall -q qcv scripts tests
python -m pytest -q
python scripts/validate_grammars.py
qcv --help
```

## Results

- Editable package installation: passed.
- Python bytecode compilation: passed.
- Unit tests: 5 passed.
- Grammar and codebook validation: 15 cases passed.
- Command-line entry point: loaded successfully with all six commands.

The grammar validation covers the configured CLEVR, GQA, ChartQA, and
AGQA-Decomp rule families and verifies that every validation question compiles
to a non-empty feasible codebook with the expected auxiliary-question count.
