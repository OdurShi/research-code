from pathlib import Path

from qcv.compilers import GrammarCompiler, load_task_grammar


ROOT = Path(__file__).resolve().parents[1]


def test_count_compiler_is_deterministic() -> None:
    compiler = GrammarCompiler(load_task_grammar(ROOT / "qcv" / "configs" / "clevr.yaml"))
    first = compiler.compile("x", "How many red cubes are there?")
    second = compiler.compile("x", "How many   red cubes are there ?")
    assert first.compiled is not None
    assert second.compiled is not None
    assert first.compiled.rule_name == "count_complement"
    assert first.compiled.target.role.name == "count"
    assert [q.template_index for q in first.compiled.auxiliaries] == [1, 2, 3, 4]
    assert [q.variable for q in first.compiled.auxiliaries] == [
        "exists_a",
        "count_not_a",
        "count_total",
        "exists_not_a",
    ]
    assert [q.to_dict() for q in first.compiled.auxiliaries] == [q.to_dict() for q in second.compiled.auxiliaries]
    assert first.compiled.constraints == second.compiled.constraints


def test_unmatched_question_is_rejected() -> None:
    compiler = GrammarCompiler(load_task_grammar(ROOT / "qcv" / "configs" / "clevr.yaml"))
    result = compiler.compile("x", "What color is the cube?")
    assert result.compiled is None
    assert result.reason == "no_unique_schema_match"
