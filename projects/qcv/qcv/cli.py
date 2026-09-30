from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .comparison import compare_prediction_files
from .inference import InferenceOptions
from .models import ModelLoadOptions
from .pipeline import (
    calibrate_jsonl,
    compile_jsonl,
    evaluate_jsonl,
    infer_jsonl,
    score_jsonl_hf,
)


def _print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qcv", description="Question-Code Verification pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="Compile target questions into typed QCV pools")
    compile_parser.add_argument("--input", required=True)
    compile_parser.add_argument("--output", required=True)
    compile_parser.add_argument("--grammars", default=str(Path(__file__).resolve().parent / "configs"))
    compile_parser.add_argument("--drop-uncovered", action="store_true")

    score_parser = subparsers.add_parser("score", help="Score complete answer alphabets")
    score_parser.add_argument("--input", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--backend", choices=["hf"], default="hf")
    score_parser.add_argument("--model")
    score_parser.add_argument("--device", default="cuda")
    score_parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    score_parser.add_argument("--attention-implementation")
    score_parser.add_argument(
        "--max-batch-candidates",
        type=int,
        default=0,
        help="0 scores the complete answer alphabet in one padded batch; a positive value enables chunking",
    )

    calibrate_parser = subparsers.add_parser("calibrate", help="Fit role temperatures and residual covariances")
    calibrate_parser.add_argument("--input", required=True)
    calibrate_parser.add_argument("--output-dir", required=True)
    calibrate_parser.add_argument("--max-budget", type=int, default=9)

    infer_parser = subparsers.add_parser("infer", help="Run QCV selection and feasible projection")
    infer_parser.add_argument("--input", required=True)
    infer_parser.add_argument("--artifacts", required=True)
    infer_parser.add_argument("--output", required=True)
    infer_parser.add_argument("--budget", type=int, default=5)
    infer_parser.add_argument(
        "--selection", choices=["qcv", "exact", "fixed", "random", "reliability"], default="qcv"
    )
    infer_parser.add_argument("--metric", choices=["full", "diagonal", "identity"], default="full")
    infer_parser.add_argument("--non-strict-coverage", action="store_true")
    infer_parser.add_argument("--random-seed", type=int, default=0)
    infer_parser.add_argument("--exact-subset-limit", type=int, default=10_000)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compute QCV accuracy and correction metrics")
    evaluate_parser.add_argument("--scores", required=True)
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--artifacts")

    compare_parser = subparsers.add_parser("compare", help="Paired bootstrap and exact McNemar comparison")
    compare_parser.add_argument("--scores", required=True)
    compare_parser.add_argument("--first", required=True)
    compare_parser.add_argument("--second", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    compare_parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compile":
        _print_result(
            compile_jsonl(
                args.input,
                args.output,
                args.grammars,
                keep_uncovered=not args.drop_uncovered,
            )
        )
        return
    if args.command == "score":
        if not args.model:
            parser.error("--model is required for scoring")
        options = ModelLoadOptions(
            model_name_or_path=args.model,
            device=args.device,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            max_batch_candidates=args.max_batch_candidates,
        )
        _print_result(score_jsonl_hf(args.input, args.output, options))
        return
    if args.command == "calibrate":
        _print_result(calibrate_jsonl(args.input, args.output_dir, args.max_budget))
        return
    if args.command == "infer":
        options = InferenceOptions(
            budget=args.budget,
            selection=args.selection,
            metric=args.metric,
            strict_coverage=not args.non_strict_coverage,
            random_seed=args.random_seed,
            exact_subset_limit=args.exact_subset_limit,
        )
        _print_result(infer_jsonl(args.input, args.artifacts, args.output, options))
        return
    if args.command == "evaluate":
        _print_result(
            evaluate_jsonl(
                args.scores,
                args.predictions,
                args.output,
                artifact_directory=args.artifacts,
            )
        )
        return
    if args.command == "compare":
        _print_result(
            compare_prediction_files(
                args.scores,
                args.first,
                args.second,
                args.output,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
