from __future__ import annotations

from pathlib import Path

from .grammar import GrammarCompiler, load_task_grammar


class CompilerRegistry:
    def __init__(self) -> None:
        self._compilers: dict[str, GrammarCompiler] = {}

    def register_grammar(self, path: str | Path) -> None:
        compiler = GrammarCompiler(load_task_grammar(path))
        task = compiler.grammar.task
        if task in self._compilers:
            raise ValueError(f"A compiler is already registered for task {task!r}")
        self._compilers[task] = compiler

    def register_directory(self, directory: str | Path) -> None:
        paths = sorted(Path(directory).glob("*.yaml")) + sorted(Path(directory).glob("*.yml"))
        if not paths:
            raise ValueError(f"No YAML grammars were found in {directory}")
        for path in paths:
            self.register_grammar(path)

    def compiler_for(self, task: str) -> GrammarCompiler:
        try:
            return self._compilers[task]
        except KeyError as exc:
            raise KeyError(f"No compiler is registered for task {task!r}") from exc

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted(self._compilers))
