from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .utils import alphabet_fingerprint, json_dump, json_load


@dataclass(frozen=True)
class RoleSpec:
    name: str
    alphabet: tuple[str, ...]
    answer_type: str = "categorical"
    numeric_precision: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Role name cannot be empty")
        if not self.alphabet:
            raise ValueError(f"Role {self.name!r} has an empty alphabet")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError(f"Role {self.name!r} has duplicate canonical answers")
        if self.answer_type not in {"categorical", "boolean", "numeric", "relation"}:
            raise ValueError(f"Unsupported answer_type {self.answer_type!r}")

    @property
    def fingerprint(self) -> str:
        return alphabet_fingerprint(self.alphabet)

    @property
    def key(self) -> str:
        return f"{self.name}:{self.fingerprint}"

    def answer_index(self, answer: str) -> int:
        try:
            return self.alphabet.index(answer)
        except ValueError as exc:
            raise ValueError(
                f"Answer {answer!r} is not in the alphabet for role {self.name!r}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alphabet": list(self.alphabet),
            "answer_type": self.answer_type,
            "numeric_precision": self.numeric_precision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoleSpec":
        return cls(
            name=str(value["name"]),
            alphabet=tuple(str(item) for item in value["alphabet"]),
            answer_type=str(value.get("answer_type", "categorical")),
            numeric_precision=(
                None
                if value.get("numeric_precision") is None
                else int(value["numeric_precision"])
            ),
        )


@dataclass(frozen=True)
class QuestionSpec:
    question_id: str
    text: str
    variable: str
    role: RoleSpec
    template_index: int
    is_target: bool = False

    def __post_init__(self) -> None:
        if not self.question_id:
            raise ValueError("question_id cannot be empty")
        if not self.variable:
            raise ValueError("Question variable cannot be empty")
        if self.template_index < 0:
            raise ValueError("template_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "variable": self.variable,
            "role": self.role.to_dict(),
            "template_index": self.template_index,
            "is_target": self.is_target,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuestionSpec":
        return cls(
            question_id=str(value["question_id"]),
            text=str(value["text"]),
            variable=str(value["variable"]),
            role=RoleSpec.from_dict(value["role"]),
            template_index=int(value["template_index"]),
            is_target=bool(value.get("is_target", False)),
        )


@dataclass(frozen=True)
class ConstraintSpec:
    kind: str
    variables: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("Constraint kind cannot be empty")
        if not self.variables:
            raise ValueError("Constraint variables cannot be empty")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError(f"Constraint {self.kind!r} repeats a variable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "variables": list(self.variables),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConstraintSpec":
        return cls(
            kind=str(value["kind"]),
            variables=tuple(str(item) for item in value["variables"]),
            parameters=dict(value.get("parameters", {})),
        )


@dataclass(frozen=True)
class CompiledInstance:
    instance_id: str
    task: str
    target: QuestionSpec
    auxiliaries: tuple[QuestionSpec, ...]
    constraints: tuple[ConstraintSpec, ...]
    rule_name: str
    bindings: Mapping[str, str] = field(default_factory=dict)
    compiler_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.target.is_target:
            raise ValueError("Compiled target must have is_target=True")
        question_ids = [self.target.question_id] + [q.question_id for q in self.auxiliaries]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("Compiled questions must have unique question IDs")
        variables = [self.target.variable] + [q.variable for q in self.auxiliaries]
        if len(set(variables)) != len(variables):
            raise ValueError("Compiled questions must have unique variables")
        ordered = sorted(
            self.auxiliaries,
            key=lambda item: (item.template_index, item.text, item.question_id),
        )
        if list(self.auxiliaries) != ordered:
            raise ValueError("Auxiliaries must be in deterministic compiler order")

    @property
    def all_questions(self) -> tuple[QuestionSpec, ...]:
        return (self.target,) + self.auxiliaries

    def question_by_id(self, question_id: str) -> QuestionSpec:
        for question in self.all_questions:
            if question.question_id == question_id:
                return question
        raise KeyError(question_id)

    def question_by_variable(self, variable: str) -> QuestionSpec:
        for question in self.all_questions:
            if question.variable == variable:
                return question
        raise KeyError(variable)

    def ordered_subset(self, auxiliary_ids: Sequence[str]) -> tuple[QuestionSpec, ...]:
        selected = {str(item) for item in auxiliary_ids}
        unknown = selected.difference({q.question_id for q in self.auxiliaries})
        if unknown:
            raise KeyError(f"Unknown auxiliary question IDs: {sorted(unknown)}")
        return (self.target,) + tuple(
            q for q in self.auxiliaries if q.question_id in selected
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "task": self.task,
            "target": self.target.to_dict(),
            "auxiliaries": [q.to_dict() for q in self.auxiliaries],
            "constraints": [c.to_dict() for c in self.constraints],
            "rule_name": self.rule_name,
            "bindings": dict(self.bindings),
            "compiler_version": self.compiler_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompiledInstance":
        return cls(
            instance_id=str(value["instance_id"]),
            task=str(value["task"]),
            target=QuestionSpec.from_dict(value["target"]),
            auxiliaries=tuple(
                QuestionSpec.from_dict(item) for item in value.get("auxiliaries", [])
            ),
            constraints=tuple(
                ConstraintSpec.from_dict(item) for item in value.get("constraints", [])
            ),
            rule_name=str(value["rule_name"]),
            bindings=dict(value.get("bindings", {})),
            compiler_version=str(value.get("compiler_version", "1.0")),
        )


@dataclass(frozen=True)
class QuestionScores:
    question_id: str
    mean_logprobs: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.question_id:
            raise ValueError("question_id cannot be empty")
        if not self.mean_logprobs:
            raise ValueError("mean_logprobs cannot be empty")
        values = np.asarray(self.mean_logprobs, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite score for {self.question_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "mean_logprobs": list(self.mean_logprobs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuestionScores":
        return cls(
            question_id=str(value["question_id"]),
            mean_logprobs=tuple(float(item) for item in value["mean_logprobs"]),
        )


@dataclass(frozen=True)
class ScoreRecord:
    instance_id: str
    task: str
    compiled: CompiledInstance
    scores: Mapping[str, QuestionScores]
    gold: Mapping[str, str] = field(default_factory=dict)
    target_answer: str | None = None
    media: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.instance_id != self.compiled.instance_id:
            raise ValueError("ScoreRecord and CompiledInstance IDs differ")
        expected = {q.question_id for q in self.compiled.all_questions}
        missing = expected.difference(self.scores)
        if missing:
            raise ValueError(f"Missing scores for questions: {sorted(missing)}")
        for question in self.compiled.all_questions:
            values = self.scores[question.question_id].mean_logprobs
            if len(values) != len(question.role.alphabet):
                raise ValueError(
                    f"Score length mismatch for {question.question_id}: "
                    f"{len(values)} != {len(question.role.alphabet)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "task": self.task,
            "compiled": self.compiled.to_dict(),
            "scores": {key: value.to_dict() for key, value in self.scores.items()},
            "gold": dict(self.gold),
            "target_answer": self.target_answer,
            "media": dict(self.media),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreRecord":
        scores = {
            str(key): QuestionScores.from_dict(item)
            for key, item in value["scores"].items()
        }
        return cls(
            instance_id=str(value["instance_id"]),
            task=str(value["task"]),
            compiled=CompiledInstance.from_dict(value["compiled"]),
            scores=scores,
            gold={str(k): str(v) for k, v in value.get("gold", {}).items()},
            target_answer=(
                None if value.get("target_answer") is None else str(value["target_answer"])
            ),
            media=dict(value.get("media", {})),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class CovarianceModel:
    signature: str
    role_keys: tuple[str, ...]
    dimension: int
    sample_count: int
    alpha_lw: float
    alpha: float
    mu: float
    covariance: np.ndarray
    whitening: np.ndarray

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        whitening = np.asarray(self.whitening, dtype=np.float64)
        if covariance.shape != (self.dimension, self.dimension):
            raise ValueError("Covariance shape does not match dimension")
        if whitening.shape != (self.dimension, self.dimension):
            raise ValueError("Whitening shape does not match dimension")
        if self.sample_count < 2:
            raise ValueError("A covariance model requires at least two samples")


@dataclass
class CalibrationArtifacts:
    temperatures: dict[str, float]
    role_accuracy: dict[str, float]
    covariances: dict[str, CovarianceModel]
    metadata: dict[str, Any] = field(default_factory=dict)

    def temperature_for(self, role: RoleSpec) -> float:
        try:
            return float(self.temperatures[role.key])
        except KeyError as exc:
            raise KeyError(f"Missing calibrated temperature for role {role.key}") from exc

    def role_reliability(self, role: RoleSpec) -> float:
        return float(self.role_accuracy.get(role.key, float("-inf")))

    def covariance_for(self, signature: str) -> CovarianceModel:
        try:
            return self.covariances[signature]
        except KeyError as exc:
            raise KeyError(f"Unsupported covariance signature {signature}") from exc

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        covariance_index: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}
        for index, (signature, model) in enumerate(sorted(self.covariances.items())):
            prefix = f"cov_{index:06d}"
            arrays[f"{prefix}_covariance"] = np.asarray(model.covariance, dtype=np.float64)
            arrays[f"{prefix}_whitening"] = np.asarray(model.whitening, dtype=np.float64)
            covariance_index[signature] = {
                "prefix": prefix,
                "role_keys": list(model.role_keys),
                "dimension": model.dimension,
                "sample_count": model.sample_count,
                "alpha_lw": model.alpha_lw,
                "alpha": model.alpha,
                "mu": model.mu,
            }
        np.savez_compressed(directory / "covariances.npz", **arrays)
        json_dump(directory / "temperatures.json", self.temperatures)
        json_dump(directory / "role_accuracy.json", self.role_accuracy)
        json_dump(directory / "covariance_index.json", covariance_index)
        json_dump(directory / "metadata.json", self.metadata)

    @classmethod
    def load(cls, directory: str | Path) -> "CalibrationArtifacts":
        directory = Path(directory)
        temperatures = {
            str(k): float(v) for k, v in json_load(directory / "temperatures.json").items()
        }
        role_accuracy = {
            str(k): float(v) for k, v in json_load(directory / "role_accuracy.json").items()
        }
        covariance_index = json_load(directory / "covariance_index.json")
        arrays = np.load(directory / "covariances.npz")
        covariances: dict[str, CovarianceModel] = {}
        for signature, info in covariance_index.items():
            prefix = str(info["prefix"])
            covariances[str(signature)] = CovarianceModel(
                signature=str(signature),
                role_keys=tuple(str(item) for item in info["role_keys"]),
                dimension=int(info["dimension"]),
                sample_count=int(info["sample_count"]),
                alpha_lw=float(info["alpha_lw"]),
                alpha=float(info["alpha"]),
                mu=float(info["mu"]),
                covariance=np.asarray(arrays[f"{prefix}_covariance"], dtype=np.float64),
                whitening=np.asarray(arrays[f"{prefix}_whitening"], dtype=np.float64),
            )
        return cls(
            temperatures=temperatures,
            role_accuracy=role_accuracy,
            covariances=covariances,
            metadata=dict(json_load(directory / "metadata.json")),
        )


@dataclass(frozen=True)
class Prediction:
    instance_id: str
    covered: bool
    target_only_answer: str
    final_answer: str
    selected_question_ids: tuple[str, ...]
    target_margin: float | None
    projection_gap: float | None
    target_class_separation: float | None
    energies: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "covered": self.covered,
            "target_only_answer": self.target_only_answer,
            "final_answer": self.final_answer,
            "selected_question_ids": list(self.selected_question_ids),
            "target_margin": self.target_margin,
            "projection_gap": self.projection_gap,
            "target_class_separation": self.target_class_separation,
            "energies": dict(self.energies),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Prediction":
        return cls(
            instance_id=str(value["instance_id"]),
            covered=bool(value["covered"]),
            target_only_answer=str(value["target_only_answer"]),
            final_answer=str(value["final_answer"]),
            selected_question_ids=tuple(str(item) for item in value.get("selected_question_ids", [])),
            target_margin=(
                None if value.get("target_margin") is None else float(value["target_margin"])
            ),
            projection_gap=(
                None if value.get("projection_gap") is None else float(value["projection_gap"])
            ),
            target_class_separation=(
                None
                if value.get("target_class_separation") is None
                else float(value["target_class_separation"])
            ),
            energies={str(k): float(v) for k, v in value.get("energies", {}).items()},
            reason=str(value.get("reason", "")),
        )
