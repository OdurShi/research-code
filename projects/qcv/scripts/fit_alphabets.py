from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from qcv.types import CompiledInstance
from qcv.utils import canonical_decimal, read_jsonl


def numeric_sort(values: set[str]) -> list[str]:
    try:
        return [item for _, item in sorted((Decimal(item), item) for item in values)]
    except Exception:
        return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze role alphabets from a disjoint training/calibration JSONL before test evaluation"
    )
    parser.add_argument("--input", required=True, help="Compiled JSONL with compiled and gold fields")
    parser.add_argument("--base-grammar", required=True)
    parser.add_argument("--output-grammar", required=True)
    parser.add_argument(
        "--derive-chart-arithmetic",
        action="store_true",
        help="Derive signed_difference and sum_value closures from first_value and second_value per record",
    )
    args = parser.parse_args()

    grammar = yaml.safe_load(Path(args.base_grammar).read_text(encoding="utf-8"))
    observed: dict[str, set[str]] = {str(name): set() for name in grammar["roles"]}

    for row in read_jsonl(args.input):
        if "compiled" not in row:
            continue
        compiled = CompiledInstance.from_dict(row["compiled"])
        gold = {str(k): str(v) for k, v in row.get("gold", {}).items()}
        role_values: dict[str, str] = {}
        for question in compiled.all_questions:
            if question.variable in gold:
                value = gold[question.variable]
                observed.setdefault(question.role.name, set()).add(value)
                role_values[question.role.name] = value
        if args.derive_chart_arithmetic and {
            "first_value",
            "second_value",
        }.issubset(role_values):
            left = Decimal(role_values["first_value"])
            right = Decimal(role_values["second_value"])
            precision = max(
                int(grammar["roles"]["first_value"].get("numeric_precision", 0)),
                int(grammar["roles"]["second_value"].get("numeric_precision", 0)),
            )
            observed.setdefault("signed_difference", set()).add(
                canonical_decimal(left - right, precision)
            )
            observed.setdefault("sum_value", set()).add(canonical_decimal(left + right, precision))

    for role_name, role_spec in grammar["roles"].items():
        values = observed.get(str(role_name), set())
        if not values:
            continue
        role_spec.pop("integer_range", None)
        role_spec.pop("decimal_range", None)
        role_spec["values"] = numeric_sort(values)

    output = Path(args.output_grammar)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(grammar, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
