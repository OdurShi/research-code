from __future__ import annotations

from decimal import Decimal
from typing import Mapping

from .types import ConstraintSpec
from .utils import decimal_value


_TRUE_VALUES = {"yes", "true", "1"}
_FALSE_VALUES = {"no", "false", "0"}


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Boolean answer must be one of {_TRUE_VALUES | _FALSE_VALUES}, got {value!r}")


def _numeric_equal(left: Decimal, right: Decimal, tolerance: Decimal) -> bool:
    return abs(left - right) <= tolerance


def evaluate_constraint(constraint: ConstraintSpec, assignment: Mapping[str, str]) -> bool:
    """Evaluate one executable answer-space predicate."""
    if any(variable not in assignment for variable in constraint.variables):
        raise KeyError(
            f"Constraint {constraint.kind!r} requires variables {constraint.variables}, "
            f"but the assignment contains {sorted(assignment)}"
        )

    kind = constraint.kind
    p = dict(constraint.parameters)
    values = [assignment[name] for name in constraint.variables]
    tolerance = Decimal(str(p.get("tolerance", "0")))

    if kind == "sum_eq":
        if len(values) != 3:
            raise ValueError("sum_eq requires variables (left, right, total)")
        return _numeric_equal(
            decimal_value(values[0]) + decimal_value(values[1]),
            decimal_value(values[2]),
            tolerance,
        )

    if kind == "difference_eq":
        if len(values) != 3:
            raise ValueError("difference_eq requires variables (left, right, difference)")
        return _numeric_equal(
            decimal_value(values[0]) - decimal_value(values[1]),
            decimal_value(values[2]),
            tolerance,
        )

    if kind == "exists_iff_positive":
        if len(values) != 2:
            raise ValueError("exists_iff_positive requires variables (count, existence)")
        return _as_bool(values[1]) == (decimal_value(values[0]) > 0)

    if kind == "threshold_boolean":
        if len(values) != 2:
            raise ValueError("threshold_boolean requires variables (numeric, boolean)")
        numeric = decimal_value(values[0])
        threshold = Decimal(str(p["threshold"]))
        operator = str(p["operator"])
        comparisons = {
            "gt": numeric > threshold,
            "ge": numeric >= threshold,
            "lt": numeric < threshold,
            "le": numeric <= threshold,
            "eq": _numeric_equal(numeric, threshold, tolerance),
            "ne": not _numeric_equal(numeric, threshold, tolerance),
        }
        if operator not in comparisons:
            raise ValueError(f"Unsupported threshold operator {operator!r}")
        return _as_bool(values[1]) == comparisons[operator]

    if kind == "inverse_relation":
        if len(values) != 2:
            raise ValueError("inverse_relation requires variables (forward, reverse)")
        inverse_map = {str(k): str(v) for k, v in p["inverse_map"].items()}
        if values[0] not in inverse_map:
            return False
        return values[1] == inverse_map[values[0]]

    if kind == "relation_boolean":
        if len(values) != 2:
            raise ValueError("relation_boolean requires variables (relation, boolean)")
        expected_relation = str(p["relation"])
        return _as_bool(values[1]) == (values[0] == expected_relation)

    if kind == "comparison_from_values":
        if len(values) != 3:
            raise ValueError("comparison_from_values requires variables (left, right, comparison)")
        left = decimal_value(values[0])
        right = decimal_value(values[1])
        labels = {
            "greater": str(p.get("greater", "greater")),
            "less": str(p.get("less", "less")),
            "equal": str(p.get("equal", "equal")),
        }
        if _numeric_equal(left, right, tolerance):
            expected = labels["equal"]
        elif left > right:
            expected = labels["greater"]
        else:
            expected = labels["less"]
        return values[2] == expected

    if kind == "equal":
        if len(values) != 2:
            raise ValueError("equal requires two variables")
        return values[0] == values[1]

    if kind == "not_equal":
        if len(values) != 2:
            raise ValueError("not_equal requires two variables")
        return values[0] != values[1]

    if kind == "boolean_not":
        if len(values) != 2:
            raise ValueError("boolean_not requires two variables")
        return _as_bool(values[1]) == (not _as_bool(values[0]))

    if kind == "one_hot_relations":
        if len(values) < 2:
            raise ValueError("one_hot_relations requires one relation and at least one Boolean")
        labels = [str(item) for item in p["labels"]]
        if len(labels) != len(values) - 1:
            raise ValueError("one_hot_relations label count does not match Boolean variables")
        relation = values[0]
        booleans = [_as_bool(item) for item in values[1:]]
        expected = [relation == label for label in labels]
        return booleans == expected

    raise ValueError(f"Unsupported constraint kind {kind!r}")


def active_constraints(
    constraints: tuple[ConstraintSpec, ...], variables: set[str]
) -> tuple[ConstraintSpec, ...]:
    """Return predicates executable for the selected question set."""
    return tuple(c for c in constraints if set(c.variables).issubset(variables))
