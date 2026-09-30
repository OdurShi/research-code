from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Sequence

from .codebook import Codebook, build_codebook
from .geometry import target_class_separation, whitening_for_mode
from .signatures import signature_for_questions
from .types import CalibrationArtifacts, CompiledInstance, QuestionSpec
from .utils import stable_hash


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[QuestionSpec, ...]
    separation: float
    evaluated_subsets: int


def _supported_codebook(
    compiled: CompiledInstance,
    questions: Sequence[QuestionSpec],
    artifacts: CalibrationArtifacts,
) -> tuple[Codebook, object]:
    codebook = build_codebook(compiled, questions)
    if codebook.codewords.shape[0] == 0 or len(codebook.target_classes) < 2:
        raise KeyError("The selected set has no usable feasible target classes")
    covariance = artifacts.covariance_for(codebook.signature)
    if covariance.dimension != codebook.dimension:
        raise ValueError("Covariance dimension and codebook dimension differ")
    return codebook, covariance


def greedy_select(
    compiled: CompiledInstance,
    budget: int,
    artifacts: CalibrationArtifacts,
    metric_mode: str = "full",
) -> SelectionResult:
    if budget < 1:
        raise ValueError("budget must be at least one")
    required = budget - 1
    if len(compiled.auxiliaries) < required:
        raise KeyError("The compiled pool is smaller than the requested budget")

    selected: list[QuestionSpec] = []
    evaluated = 0
    final_separation = 0.0
    for _ in range(required):
        best_question: QuestionSpec | None = None
        best_score = -math.inf
        for candidate in compiled.auxiliaries:
            if candidate in selected:
                continue
            questions = (compiled.target,) + tuple(selected) + (candidate,)
            try:
                codebook, covariance = _supported_codebook(compiled, questions, artifacts)
            except (KeyError, ValueError):
                continue
            whitening = whitening_for_mode(covariance, metric_mode)
            score = target_class_separation(codebook, whitening)
            evaluated += 1
            if score > best_score:
                best_question = candidate
                best_score = score
        if best_question is None:
            raise KeyError("No supported candidate remains during greedy selection")
        selected.append(best_question)
        final_separation = best_score
    return SelectionResult(tuple(selected), float(final_separation), evaluated)


def exact_select(
    compiled: CompiledInstance,
    budget: int,
    artifacts: CalibrationArtifacts,
    metric_mode: str = "full",
    subset_limit: int = 10_000,
) -> SelectionResult:
    required = budget - 1
    if required < 0:
        raise ValueError("budget must be at least one")
    if len(compiled.auxiliaries) < required:
        raise KeyError("The compiled pool is smaller than the requested budget")
    count = math.comb(len(compiled.auxiliaries), required)
    if count > subset_limit:
        return greedy_select(compiled, budget, artifacts, metric_mode)
    best_subset: tuple[QuestionSpec, ...] | None = None
    best_score = -math.inf
    evaluated = 0
    for subset in itertools.combinations(compiled.auxiliaries, required):
        questions = (compiled.target,) + subset
        try:
            codebook, covariance = _supported_codebook(compiled, questions, artifacts)
        except (KeyError, ValueError):
            continue
        score = target_class_separation(codebook, whitening_for_mode(covariance, metric_mode))
        evaluated += 1
        if score > best_score:
            best_subset = subset
            best_score = score
    if best_subset is None:
        raise KeyError("No supported exact subset exists")
    return SelectionResult(best_subset, float(best_score), evaluated)


def fixed_select(compiled: CompiledInstance, budget: int) -> SelectionResult:
    required = budget - 1
    if len(compiled.auxiliaries) < required:
        raise KeyError("The compiled pool is smaller than the requested budget")
    return SelectionResult(compiled.auxiliaries[:required], float("nan"), 0)


def reliability_select(
    compiled: CompiledInstance,
    budget: int,
    artifacts: CalibrationArtifacts,
) -> SelectionResult:
    required = budget - 1
    if len(compiled.auxiliaries) < required:
        raise KeyError("The compiled pool is smaller than the requested budget")
    ordered = sorted(
        compiled.auxiliaries,
        key=lambda question: (
            -artifacts.role_reliability(question.role),
            question.template_index,
            question.text,
        ),
    )
    selected = tuple(sorted(ordered[:required], key=lambda q: (q.template_index, q.text)))
    return SelectionResult(selected, float("nan"), 0)


def random_select(
    compiled: CompiledInstance,
    budget: int,
    seed: int,
) -> SelectionResult:
    required = budget - 1
    if len(compiled.auxiliaries) < required:
        raise KeyError("The compiled pool is smaller than the requested budget")
    instance_seed = int(stable_hash([str(seed), compiled.instance_id], length=16), 16)
    generator = random.Random(instance_seed)
    selected = generator.sample(list(compiled.auxiliaries), required)
    selected.sort(key=lambda q: (q.template_index, q.text, q.question_id))
    return SelectionResult(tuple(selected), float("nan"), 0)
