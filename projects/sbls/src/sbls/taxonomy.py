from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Iterable

from .types import CandidateSpec, FrozenCandidate, LabelNode, NewLabelSpec
from .utils import ROOT_ID, normalize_name, sha256_hex


class TaxonomyError(ValueError):
    """Raised when a taxonomy or local edit violates structural invariants."""


class Taxonomy:
    def __init__(
        self,
        labels: Iterable[LabelNode],
        *,
        version: int = 0,
        root_id: str = ROOT_ID,
    ) -> None:
        self.version = int(version)
        self.root_id = root_id
        label_list = list(labels)
        if not label_list:
            raise TaxonomyError("A taxonomy must contain at least one label")
        self._labels: dict[str, LabelNode] = {}
        for node in label_list:
            if node.id == root_id:
                raise TaxonomyError(f"Label id {root_id!r} is reserved for the root")
            if node.id in self._labels:
                raise TaxonomyError(f"Duplicate label id: {node.id}")
            if not node.name.strip() or not node.description.strip():
                raise TaxonomyError(f"Label {node.id!r} has an empty name or description")
            self._labels[node.id] = node
        self._normalize_priors()
        self.validate()

    def _normalize_priors(self) -> None:
        priors = [node.prior for node in self._labels.values()]
        if all(float(p) == 0.0 for p in priors):
            mass = 1.0 / len(priors)
            self._labels = {
                label_id: replace(node, prior=mass) for label_id, node in self._labels.items()
            }
            return
        if any((not math.isfinite(float(p))) or float(p) <= 0.0 for p in priors):
            raise TaxonomyError("All explicit priors must be finite and strictly positive")
        total = float(sum(priors))
        self._labels = {
            label_id: replace(node, prior=float(node.prior) / total)
            for label_id, node in self._labels.items()
        }

    @property
    def labels(self) -> tuple[LabelNode, ...]:
        return tuple(self._labels[label_id] for label_id in sorted(self._labels))

    @property
    def label_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._labels))

    @property
    def size(self) -> int:
        return len(self._labels)

    def get(self, label_id: str) -> LabelNode:
        try:
            return self._labels[label_id]
        except KeyError as exc:
            raise TaxonomyError(f"Unknown label id: {label_id}") from exc

    def contains(self, label_id: str) -> bool:
        return label_id in self._labels

    def children(self, parent_id: str) -> tuple[LabelNode, ...]:
        parent = None if parent_id == self.root_id else parent_id
        return tuple(sorted((n for n in self._labels.values() if n.parent_id == parent), key=lambda n: n.id))

    def is_leaf(self, label_id: str) -> bool:
        self.get(label_id)
        return not any(node.parent_id == label_id for node in self._labels.values())

    def parent_id(self, label_id: str) -> str:
        parent = self.get(label_id).parent_id
        return self.root_id if parent is None else parent

    def ancestors(self, label_id: str) -> tuple[str, ...]:
        current = self.get(label_id)
        result: list[str] = []
        seen: set[str] = set()
        while current.parent_id is not None:
            if current.parent_id in seen:
                raise TaxonomyError("Cycle detected while traversing ancestors")
            seen.add(current.parent_id)
            result.append(current.parent_id)
            current = self.get(current.parent_id)
        return tuple(result)

    def validate(self) -> None:
        normalized: dict[str, str] = {}
        total = 0.0
        for node in self._labels.values():
            if node.parent_id == self.root_id:
                raise TaxonomyError(
                    f"Node {node.id!r} uses the root sentinel as parent_id; use null/None instead"
                )
            if node.parent_id is not None and node.parent_id not in self._labels:
                raise TaxonomyError(
                    f"Node {node.id!r} references missing parent {node.parent_id!r}"
                )
            key = normalize_name(node.name)
            if not key:
                raise TaxonomyError(f"Node {node.id!r} has an empty normalized name")
            if key in normalized:
                raise TaxonomyError(
                    f"Duplicate normalized names: {node.id!r} and {normalized[key]!r}"
                )
            normalized[key] = node.id
            if not math.isfinite(node.prior) or node.prior <= 0.0:
                raise TaxonomyError(f"Node {node.id!r} has an invalid prior")
            total += node.prior
            self.ancestors(node.id)
        if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
            raise TaxonomyError(f"Priors sum to {total}, not 1")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "root_id": self.root_id,
            "labels": [
                {
                    "id": node.id,
                    "name": node.name,
                    "description": node.description,
                    "parent_id": node.parent_id,
                    "prior": node.prior,
                }
                for node in self.labels
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Taxonomy":
        if not isinstance(payload, dict):
            raise TaxonomyError("Taxonomy payload must be an object")
        labels_raw = payload.get("labels")
        if not isinstance(labels_raw, list):
            raise TaxonomyError("Taxonomy payload must contain a labels array")
        labels = []
        for item in labels_raw:
            if not isinstance(item, dict):
                raise TaxonomyError("Each taxonomy label must be an object")
            labels.append(
                LabelNode(
                    id=str(item["id"]),
                    name=str(item["name"]),
                    description=str(item["description"]),
                    parent_id=(None if item.get("parent_id") in (None, payload.get("root_id", ROOT_ID)) else str(item["parent_id"])),
                    prior=float(item.get("prior", 0.0)),
                )
            )
        return cls(
            labels,
            version=int(payload.get("version", 0)),
            root_id=str(payload.get("root_id", ROOT_ID)),
        )

    def copy(self) -> "Taxonomy":
        return Taxonomy(copy.deepcopy(self.labels), version=self.version, root_id=self.root_id)

    def local_edges(self, label_ids: Iterable[str]) -> list[dict[str, str]]:
        ids = set(label_ids)
        edges: list[dict[str, str]] = []
        for label_id in sorted(ids):
            node = self.get(label_id)
            parent = self.root_id if node.parent_id is None else node.parent_id
            edges.append({"parent_id": parent, "child_id": node.id})
        return edges

    def apply_candidate(self, candidate_id: str, spec: CandidateSpec) -> "Taxonomy":
        if spec.operation == "add":
            return self._apply_add(candidate_id, spec)
        if spec.operation == "split":
            return self._apply_split(candidate_id, spec)
        raise TaxonomyError(f"Unsupported operation: {spec.operation}")

    def _new_label_id(self, candidate_id: str, index: int, name: str) -> str:
        digest = sha256_hex({"candidate_id": candidate_id, "index": index, "name": normalize_name(name)})
        return f"sbls_{digest[:16]}"

    def _resolve_parent(self, parent_id: str) -> str | None:
        if parent_id == self.root_id:
            return None
        if parent_id not in self._labels:
            raise TaxonomyError(f"Unknown parent id: {parent_id}")
        return parent_id

    def _apply_add(self, candidate_id: str, spec: CandidateSpec) -> "Taxonomy":
        if spec.source_label_id is not None:
            raise TaxonomyError("An add candidate must not include source_label_id")
        if len(spec.new_labels) != 1:
            raise TaxonomyError("An add candidate must contain exactly one new label")
        parent = self._resolve_parent(spec.parent_id)
        k = self.size
        old_scale = k / (k + 1.0)
        labels = [replace(node, prior=node.prior * old_scale) for node in self.labels]
        new = spec.new_labels[0]
        labels.append(
            LabelNode(
                id=self._new_label_id(candidate_id, 0, new.name),
                name=new.name,
                description=new.description,
                parent_id=parent,
                prior=1.0 / (k + 1.0),
            )
        )
        return Taxonomy(labels, version=self.version + 1, root_id=self.root_id)

    def _apply_split(self, candidate_id: str, spec: CandidateSpec) -> "Taxonomy":
        if spec.source_label_id is None:
            raise TaxonomyError("A split candidate requires source_label_id")
        if len(spec.new_labels) != 2:
            raise TaxonomyError("A split candidate must contain exactly two new labels")
        source = self.get(spec.source_label_id)
        if not self.is_leaf(source.id):
            raise TaxonomyError(f"Split source {source.id!r} is not a leaf")
        expected_parent = self.root_id if source.parent_id is None else source.parent_id
        if spec.parent_id != expected_parent:
            raise TaxonomyError(
                f"Split parent {spec.parent_id!r} does not match source parent {expected_parent!r}"
            )
        labels = [node for node in self.labels if node.id != source.id]
        for index, new in enumerate(spec.new_labels):
            labels.append(
                LabelNode(
                    id=self._new_label_id(candidate_id, index, new.name),
                    name=new.name,
                    description=new.description,
                    parent_id=source.parent_id,
                    prior=source.prior / 2.0,
                )
            )
        return Taxonomy(labels, version=self.version + 1, root_id=self.root_id)

    def descriptor_map(self) -> dict[str, tuple[str, str]]:
        return {node.id: (node.name, node.description) for node in self.labels}

    def priors(self) -> dict[str, float]:
        return {node.id: node.prior for node in self.labels}

    def __repr__(self) -> str:
        return f"Taxonomy(version={self.version}, size={self.size})"
