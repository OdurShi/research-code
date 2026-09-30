from pathlib import Path

import numpy as np

from qcv.codebook import build_codebook
from qcv.compilers import GrammarCompiler, load_task_grammar


ROOT = Path(__file__).resolve().parents[1]


def test_count_codebook_enforces_arithmetic_and_existence() -> None:
    compiler = GrammarCompiler(load_task_grammar(ROOT / "qcv" / "configs" / "clevr.yaml"))
    compiled = compiler.compile("x", "How many red cubes are there?").compiled
    assert compiled is not None
    codebook = build_codebook(compiled, compiled.all_questions)
    assignment = {
        "count_a": "3",
        "exists_a": "yes",
        "count_not_a": "4",
        "count_total": "7",
        "exists_not_a": "yes",
    }
    vector = codebook.encode_assignment(assignment)
    assert any(np.array_equal(vector, row) for row in codebook.codewords)

    invalid = {
        "count_a": "3",
        "exists_a": "no",
        "count_not_a": "4",
        "count_total": "7",
        "exists_not_a": "yes",
    }
    invalid_vector = codebook.encode_assignment(invalid)
    assert not any(np.array_equal(invalid_vector, row) for row in codebook.codewords)
    assert "3" in codebook.target_classes
