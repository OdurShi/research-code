from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qcv.compilers import GrammarCompiler, load_task_grammar
from qcv.utils import write_jsonl


def execute_program(scene: dict[str, Any], program: list[dict[str, Any]]) -> list[Any]:
    objects = scene["objects"]
    relationships = scene.get("relationships", {})
    outputs: list[Any] = []
    for node in program:
        function = str(node["function"])
        inputs = [outputs[int(index)] for index in node.get("inputs", [])]
        values = list(node.get("value_inputs", []))
        if function == "scene":
            result: Any = set(range(len(objects)))
        elif function.startswith("filter_"):
            attribute = function.removeprefix("filter_")
            expected = values[0]
            result = {index for index in inputs[0] if objects[index][attribute] == expected}
        elif function == "unique":
            if len(inputs[0]) != 1:
                raise ValueError("unique received a non-singleton set")
            result = next(iter(inputs[0]))
        elif function == "relate":
            relation = str(values[0])
            anchor = int(inputs[0])
            relation_rows = relationships[relation]
            result = set(int(index) for index in relation_rows[anchor])
        elif function.startswith("same_"):
            attribute = function.removeprefix("same_")
            anchor = int(inputs[0])
            expected = objects[anchor][attribute]
            result = {
                index
                for index, obj in enumerate(objects)
                if index != anchor and obj[attribute] == expected
            }
        elif function == "union":
            result = set(inputs[0]).union(inputs[1])
        elif function == "intersect":
            result = set(inputs[0]).intersection(inputs[1])
        elif function == "count":
            result = len(inputs[0])
        elif function == "exist":
            result = len(inputs[0]) > 0
        elif function.startswith("query_"):
            attribute = function.removeprefix("query_")
            result = objects[int(inputs[0])][attribute]
        elif function.startswith("equal_"):
            result = inputs[0] == inputs[1]
        elif function == "less_than":
            result = int(inputs[0]) < int(inputs[1])
        elif function == "greater_than":
            result = int(inputs[0]) > int(inputs[1])
        else:
            raise ValueError(f"Unsupported CLEVR program function {function!r}")
        outputs.append(result)
    return outputs


def derive_gold(
    compiled: Any,
    question: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, str]:
    answer = str(question["answer"]).lower()
    rule = compiled.rule_name
    if rule == "count_complement":
        count_a = int(answer)
        total = len(scene["objects"])
        complement = total - count_a
        return {
            "count_a": str(count_a),
            "exists_a": "yes" if count_a > 0 else "no",
            "count_not_a": str(complement),
            "count_total": str(total),
            "exists_not_a": "yes" if complement > 0 else "no",
        }
    if rule == "existence_count":
        outputs = execute_program(scene, question["program"])
        final_node = question["program"][-1]
        if final_node["function"] != "exist" or not final_node.get("inputs"):
            raise ValueError("Existence question does not end in an executable exist node")
        object_set = outputs[int(final_node["inputs"][0])]
        count_a = len(object_set)
        total = len(scene["objects"])
        complement = total - count_a
        return {
            "exists_a": "yes" if bool(outputs[-1]) else "no",
            "count_a": str(count_a),
            "count_not_a": str(complement),
            "count_total": str(total),
            "exists_not_a": "yes" if complement > 0 else "no",
        }
    if rule == "spatial_inverse_categorical":
        inverse = {"left": "right", "right": "left", "front": "behind", "behind": "front"}
        if answer not in inverse:
            raise ValueError(f"Unsupported spatial relation answer {answer!r}")
        return {
            "relation_ab": answer,
            "relation_ba": inverse[answer],
            "is_left": "yes" if answer == "left" else "no",
            "is_right": "yes" if answer == "right" else "no",
            "is_front": "yes" if answer == "front" else "no",
            "is_behind": "yes" if answer == "behind" else "no",
        }
    raise ValueError(f"Gold derivation is not defined for compiler rule {rule!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official CLEVR JSON files to QCV JSONL")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--grammar",
        default=str(Path(__file__).resolve().parents[1] / "qcv" / "configs" / "clevr.yaml"),
    )
    parser.add_argument("--drop-incomplete-gold", action="store_true")
    args = parser.parse_args()

    question_payload = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    scene_payload = json.loads(Path(args.scenes).read_text(encoding="utf-8"))
    questions = question_payload["questions"]
    scenes = {scene["image_filename"]: scene for scene in scene_payload["scenes"]}
    compiler = GrammarCompiler(load_task_grammar(args.grammar))
    rows: list[dict[str, Any]] = []
    for question in questions:
        image_filename = str(question["image_filename"])
        instance_id = str(question.get("question_index", len(rows)))
        result = compiler.compile(instance_id, str(question["question"]))
        if result.compiled is None:
            continue
        gold: dict[str, str] = {}
        try:
            gold = derive_gold(result.compiled, question, scenes[image_filename])
        except (KeyError, ValueError, TypeError):
            if args.drop_incomplete_gold:
                continue
            gold = {result.compiled.target.variable: str(question["answer"]).lower()}
        rows.append(
            {
                "instance_id": instance_id,
                "task": "clevr",
                "question": str(question["question"]),
                "answer": str(question["answer"]).lower(),
                "media": {
                    "type": "image",
                    "path": str(Path(args.images_dir) / image_filename),
                },
                "gold": gold,
                "metadata": {
                    "image_filename": image_filename,
                    "question_family_index": question.get("question_family_index"),
                    "split": question_payload.get("split"),
                },
            }
        )
    write_jsonl(args.output, rows)
    print(json.dumps({"written": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
