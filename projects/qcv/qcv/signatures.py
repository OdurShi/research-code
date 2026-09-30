from __future__ import annotations

from typing import Sequence

from .types import QuestionSpec
from .utils import stable_hash


def ordered_questions(questions: Sequence[QuestionSpec]) -> tuple[QuestionSpec, ...]:
    targets = [question for question in questions if question.is_target]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one target question, found {len(targets)}")
    auxiliaries = sorted(
        (question for question in questions if not question.is_target),
        key=lambda item: (item.template_index, item.text, item.question_id),
    )
    return (targets[0],) + tuple(auxiliaries)


def role_keys(questions: Sequence[QuestionSpec]) -> tuple[str, ...]:
    return tuple(question.role.key for question in ordered_questions(questions))


def signature_for_questions(questions: Sequence[QuestionSpec]) -> str:
    ordered = ordered_questions(questions)
    keys = [question.role.key for question in ordered]
    return stable_hash(keys, length=32)
