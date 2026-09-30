from pathlib import Path

import numpy as np

from qcv.codebook import build_codebook
from qcv.compilers import GrammarCompiler, load_task_grammar
from qcv.decoding import feasible_projection_decode


ROOT = Path(__file__).resolve().parents[1]


def test_projection_recovers_true_target_for_small_residual() -> None:
    compiler = GrammarCompiler(load_task_grammar(ROOT / "qcv" / "configs" / "clevr.yaml"))
    compiled = compiler.compile("x", "How many red cubes are there?").compiled
    assert compiled is not None
    questions = (compiled.target,) + compiled.auxiliaries[:3]
    codebook = build_codebook(compiled, questions)
    assignment = {
        "count_a": "3",
        "exists_a": "yes",
        "count_not_a": "4",
        "count_total": "7",
    }
    true_code = codebook.encode_assignment(assignment)
    observation = true_code + np.linspace(-0.01, 0.01, codebook.dimension)
    result = feasible_projection_decode(codebook, observation, np.eye(codebook.dimension))
    assert result.target_answer == "3"
    assert result.target_margin > 0
