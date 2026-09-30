from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..codebook import build_codebook
from ..types import CompiledInstance, ConstraintSpec, QuestionSpec, RoleSpec
from ..utils import canonical_decimal, normalize_text


@dataclass(frozen=True)
class TaskGrammar:
    task: str
    version: str
    roles: Mapping[str, RoleSpec]
    rules: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CompileResult:
    compiled: CompiledInstance | None
    reason: str
    matching_rules: tuple[str, ...] = ()


def _expand_alphabet(value: Mapping[str, Any]) -> tuple[str, ...]:
    if "values" in value:
        return tuple(str(item) for item in value["values"])
    if "integer_range" in value:
        spec = value["integer_range"]
        start = int(spec["start"])
        stop = int(spec["stop"])
        step = int(spec.get("step", 1))
        if step <= 0 or stop < start:
            raise ValueError(f"Invalid integer_range {spec}")
        return tuple(str(number) for number in range(start, stop + 1, step))
    if "decimal_range" in value:
        spec = value["decimal_range"]
        start = float(spec["start"])
        stop = float(spec["stop"])
        step = float(spec["step"])
        precision = int(spec["precision"])
        if step <= 0 or stop < start:
            raise ValueError(f"Invalid decimal_range {spec}")
        count = int(round((stop - start) / step))
        values = [canonical_decimal(start + index * step, precision) for index in range(count + 1)]
        if float(values[-1]) < stop - 10 ** (-(precision + 2)):
            values.append(canonical_decimal(stop, precision))
        return tuple(dict.fromkeys(values))
    raise ValueError("An alphabet requires values, integer_range, or decimal_range")


def load_task_grammar(path: str | Path) -> TaskGrammar:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Grammar {path} must contain a YAML mapping")
    roles: dict[str, RoleSpec] = {}
    for name, spec in raw["roles"].items():
        alphabet = _expand_alphabet(spec)
        roles[str(name)] = RoleSpec(
            name=str(name),
            alphabet=alphabet,
            answer_type=str(spec.get("answer_type", "categorical")),
            numeric_precision=(
                None if spec.get("numeric_precision") is None else int(spec["numeric_precision"])
            ),
        )
    rules = tuple(dict(rule) for rule in raw["rules"])
    return TaskGrammar(
        task=str(raw["task"]),
        version=str(raw.get("version", "1.0")),
        roles=roles,
        rules=rules,
    )


def _format_nested(value: Any, bindings: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return value.format_map(bindings)
    if isinstance(value, list):
        return [_format_nested(item, bindings) for item in value]
    if isinstance(value, tuple):
        return tuple(_format_nested(item, bindings) for item in value)
    if isinstance(value, dict):
        return {str(key): _format_nested(item, bindings) for key, item in value.items()}
    return value


class GrammarCompiler:
    """Deterministic text-only compiler with unique regex matching."""

    def __init__(self, grammar: TaskGrammar):
        self.grammar = grammar
        self._compiled_rules: list[tuple[Mapping[str, Any], re.Pattern[str]]] = []
        for rule in grammar.rules:
            pattern = re.compile(str(rule["pattern"]), flags=re.IGNORECASE)
            self._compiled_rules.append((rule, pattern))

    def compile(self, instance_id: str, question: str) -> CompileResult:
        normalized = normalize_text(question)
        matches: list[tuple[Mapping[str, Any], re.Match[str]]] = []
        for rule, pattern in self._compiled_rules:
            match = pattern.fullmatch(normalized)
            if match is not None:
                matches.append((rule, match))
        names = tuple(str(rule["name"]) for rule, _ in matches)
        if not matches:
            return CompileResult(None, "no_unique_schema_match", names)
        if len(matches) != 1:
            return CompileResult(None, "ambiguous_schema_match", names)

        rule, match = matches[0]
        bindings = {key: value.strip() for key, value in match.groupdict().items() if value is not None}
        bindings["question"] = question.strip()
        target_spec = dict(rule["target"])
        target_role_name = str(target_spec["role"])
        if target_role_name not in self.grammar.roles:
            return CompileResult(None, f"unknown_target_role:{target_role_name}", names)
        target = QuestionSpec(
            question_id="q0",
            text=question.strip(),
            variable=str(target_spec["variable"]),
            role=self.grammar.roles[target_role_name],
            template_index=0,
            is_target=True,
        )

        generated: list[QuestionSpec] = []
        for aux in rule.get("auxiliaries", []):
            role_name = str(aux["role"])
            if role_name not in self.grammar.roles:
                return CompileResult(None, f"unknown_auxiliary_role:{role_name}", names)
            try:
                text = str(aux["text"]).format_map(bindings).strip()
            except KeyError as exc:
                return CompileResult(None, f"missing_template_binding:{exc.args[0]}", names)
            index = int(aux["index"])
            generated.append(
                QuestionSpec(
                    question_id=f"q{index:04d}",
                    text=text,
                    variable=str(aux["variable"]),
                    role=self.grammar.roles[role_name],
                    template_index=index,
                    is_target=False,
                )
            )

        deduplicated: dict[str, QuestionSpec] = {}
        target_normalized = normalize_text(target.text)
        for candidate in sorted(generated, key=lambda item: (item.template_index, item.text)):
            canonical = normalize_text(candidate.text)
            if canonical == target_normalized:
                continue
            if canonical not in deduplicated:
                deduplicated[canonical] = candidate
        auxiliaries = tuple(
            sorted(deduplicated.values(), key=lambda item: (item.template_index, item.text))
        )
        variables = [target.variable] + [item.variable for item in auxiliaries]
        if len(set(variables)) != len(variables):
            return CompileResult(None, "duplicate_variables", names)

        constraints: list[ConstraintSpec] = []
        for item in rule.get("constraints", []):
            parameters = _format_nested(dict(item.get("parameters", {})), bindings)
            constraint = ConstraintSpec(
                kind=str(item["kind"]),
                variables=tuple(str(variable) for variable in item["variables"]),
                parameters=parameters,
            )
            unknown_variables = set(constraint.variables).difference(variables)
            if unknown_variables:
                return CompileResult(
                    None,
                    f"constraint_unknown_variables:{','.join(sorted(unknown_variables))}",
                    names,
                )
            constraints.append(constraint)

        try:
            compiled = CompiledInstance(
                instance_id=str(instance_id),
                task=self.grammar.task,
                target=target,
                auxiliaries=auxiliaries,
                constraints=tuple(constraints),
                rule_name=str(rule["name"]),
                bindings=bindings,
                compiler_version=self.grammar.version,
            )
            full_codebook = build_codebook(compiled, compiled.all_questions)
            if full_codebook.codewords.shape[0] == 0:
                return CompileResult(None, "empty_feasible_code", names)
        except (KeyError, ValueError) as exc:
            return CompileResult(None, f"validation_error:{type(exc).__name__}:{exc}", names)
        return CompileResult(compiled, "compiled", names)
