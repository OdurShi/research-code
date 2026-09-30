from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .constraints import active_constraints, evaluate_constraint
from .signatures import ordered_questions, signature_for_questions
from .types import CompiledInstance, ConstraintSpec, QuestionSpec
from .utils import stable_hash


@dataclass(frozen=True)
class Codebook:
    structure_key: str
    questions: tuple[QuestionSpec, ...]
    signature: str
    constraints: tuple[ConstraintSpec, ...]
    assignments: tuple[dict[str, str], ...]
    codewords: np.ndarray
    target_classes: dict[str, np.ndarray]
    unconstrained_assignments: tuple[dict[str, str], ...]
    unconstrained_codewords: np.ndarray
    block_slices: dict[str, slice]

    @property
    def dimension(self) -> int:
        return int(self.codewords.shape[1])

    @property
    def target_alphabet(self) -> tuple[str, ...]:
        return self.questions[0].role.alphabet

    def encode_assignment(self, assignment: Mapping[str, str]) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float64)
        for question in self.questions:
            if question.variable not in assignment:
                raise KeyError(f"Missing value for {question.variable!r}")
            answer = str(assignment[question.variable])
            index = question.role.answer_index(answer)
            block = self.block_slices[question.question_id]
            vector[block.start + index] = 1.0
        return vector

    def class_codewords(self, target_answer: str) -> np.ndarray:
        try:
            indices = self.target_classes[target_answer]
        except KeyError as exc:
            raise KeyError(f"Target answer {target_answer!r} has no feasible class") from exc
        return self.codewords[indices]


def _block_slices(questions: Sequence[QuestionSpec]) -> dict[str, slice]:
    output: dict[str, slice] = {}
    offset = 0
    for question in questions:
        width = len(question.role.alphabet)
        output[question.question_id] = slice(offset, offset + width)
        offset += width
    return output


def _encode_tuple(
    questions: Sequence[QuestionSpec],
    answers: Sequence[str],
    blocks: Mapping[str, slice],
) -> np.ndarray:
    dimension = sum(len(question.role.alphabet) for question in questions)
    vector = np.zeros(dimension, dtype=np.float64)
    for question, answer in zip(questions, answers, strict=True):
        index = question.role.answer_index(answer)
        block = blocks[question.question_id]
        vector[block.start + index] = 1.0
    return vector


_CODEBOOK_CACHE: dict[str, Codebook] = {}


def _structure_key(compiled: CompiledInstance, questions: Sequence[QuestionSpec]) -> str:
    ordered = ordered_questions(questions)
    selected_variables = {question.variable for question in ordered}
    constraints = active_constraints(compiled.constraints, selected_variables)
    parts: list[str] = [compiled.task, compiled.rule_name, compiled.compiler_version]
    for question in ordered:
        parts.extend([question.question_id, question.variable, question.role.key])
    for constraint in constraints:
        parts.append(constraint.kind)
        parts.extend(constraint.variables)
        parts.append(repr(sorted(dict(constraint.parameters).items())))
    return stable_hash(parts, length=40)


def build_codebook(
    compiled: CompiledInstance,
    questions: Sequence[QuestionSpec],
) -> Codebook:
    ordered = ordered_questions(questions)
    if ordered[0].question_id != compiled.target.question_id:
        raise ValueError("The selected target does not belong to the compiled instance")
    selected_ids = {question.question_id for question in ordered}
    valid_ids = {question.question_id for question in compiled.all_questions}
    if not selected_ids.issubset(valid_ids):
        raise ValueError("Selected questions do not all belong to the compiled instance")

    structure_key = _structure_key(compiled, ordered)
    cached = _CODEBOOK_CACHE.get(structure_key)
    if cached is not None:
        return cached

    variables = {question.variable for question in ordered}
    constraints = active_constraints(compiled.constraints, variables)
    blocks = _block_slices(ordered)
    answer_products = itertools.product(*(question.role.alphabet for question in ordered))

    feasible_assignments: list[dict[str, str]] = []
    feasible_vectors: list[np.ndarray] = []
    unconstrained_assignments: list[dict[str, str]] = []
    unconstrained_vectors: list[np.ndarray] = []
    seen_vectors: set[bytes] = set()

    for answers in answer_products:
        assignment = {
            question.variable: answer
            for question, answer in zip(ordered, answers, strict=True)
        }
        vector = _encode_tuple(ordered, answers, blocks)
        unconstrained_assignments.append(assignment)
        unconstrained_vectors.append(vector)
        if all(evaluate_constraint(constraint, assignment) for constraint in constraints):
            key = vector.tobytes()
            if key not in seen_vectors:
                seen_vectors.add(key)
                feasible_assignments.append(assignment)
                feasible_vectors.append(vector)

    dimension = sum(len(question.role.alphabet) for question in ordered)
    if not feasible_vectors:
        codewords = np.empty((0, dimension), dtype=np.float64)
    else:
        codewords = np.stack(feasible_vectors, axis=0)
    unconstrained_codewords = np.stack(unconstrained_vectors, axis=0)

    target_classes: dict[str, np.ndarray] = {}
    target_variable = ordered[0].variable
    for target_answer in ordered[0].role.alphabet:
        indices = [
            index
            for index, assignment in enumerate(feasible_assignments)
            if assignment[target_variable] == target_answer
        ]
        if indices:
            target_classes[target_answer] = np.asarray(indices, dtype=np.int64)

    codebook = Codebook(
        structure_key=structure_key,
        questions=ordered,
        signature=signature_for_questions(ordered),
        constraints=constraints,
        assignments=tuple(feasible_assignments),
        codewords=codewords,
        target_classes=target_classes,
        unconstrained_assignments=tuple(unconstrained_assignments),
        unconstrained_codewords=unconstrained_codewords,
        block_slices=blocks,
    )
    _CODEBOOK_CACHE[structure_key] = codebook
    return codebook
