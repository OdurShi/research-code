#!/usr/bin/env python3
"""
Batch Qwen semantic assignments: loads QwenVisionTeacher ONCE and processes all run-dirs.
Saves ~45s model-load overhead per query.

Input manifest JSON (--manifest):
  [
    {
      "query_id":   "americano_q001",
      "run_dir":    "/abs/path/to/query_worldtube_run",
      "query_text": "The hand pouring coffee.",
      "output_path": "/abs/path/to/semantic_assignments_qwen.json"
    }, ...
  ]

Skips entries where output_path already exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "4DGaussians"
for _c in (PROJECT_ROOT, EXTERNAL_ROOT):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from d4resplat.semantics.qwen_assignment import (
    QwenVisionTeacher,
    _resolve_qwen_model,
    export_qwen_semantic_assignments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Qwen semantic assignments (single model load)")
    parser.add_argument("--manifest",                  required=True)
    parser.add_argument("--qwen-model",                default=None)
    parser.add_argument("--max-entities",              type=int,   default=12)
    parser.add_argument("--shortlist-k",               type=int,   default=24)
    parser.add_argument("--bootstrap-weight",          type=float, default=0.45)
    parser.add_argument("--dynamic-overlap-threshold", type=float, default=0.10)
    parser.add_argument("--summary-path",              default=None)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    pending = [item for item in items if not Path(item["output_path"]).exists()]
    print(f"[batch_qwen_sem] {len(items)} total, {len(pending)} pending, {len(items)-len(pending)} already done")
    if not pending:
        print("[batch_qwen_sem] all done, exiting")
        return

    print("[batch_qwen_sem] loading QwenVisionTeacher…")
    t0 = time.time()
    model_path = _resolve_qwen_model(args.qwen_model)
    teacher = QwenVisionTeacher(model_path)
    print(f"[batch_qwen_sem] model loaded in {time.time()-t0:.1f}s  ({model_path})")

    results = []
    for idx, item in enumerate(pending):
        query_id   = item["query_id"]
        run_dir    = item["run_dir"]
        query_text = item.get("query_text")

        print(f"\n[batch_qwen_sem] [{idx+1}/{len(pending)}] {query_id}")
        t_q = time.time()
        try:
            out_path = export_qwen_semantic_assignments(
                run_dir=run_dir,
                max_entities=args.max_entities,
                query=query_text,
                shortlist_k=args.shortlist_k,
                bootstrap_weight=args.bootstrap_weight,
                dynamic_overlap_min=args.dynamic_overlap_threshold,
                _teacher=teacher,
            )
            elapsed = time.time() - t_q
            payload = json.load(open(out_path))
            n_qwen = payload.get("num_qwen_assignments", "?")
            n_total = payload.get("num_assignments", "?")
            print(f"[batch_qwen_sem] {query_id} done in {elapsed:.1f}s  "
                  f"{n_qwen}/{n_total} entities assigned  → {out_path}")
            results.append({"query_id": query_id, "status": "ok",
                            "n_qwen": n_qwen, "n_total": n_total, "elapsed": elapsed})
        except Exception as exc:
            elapsed = time.time() - t_q
            print(f"[batch_qwen_sem] {query_id} ERROR: {exc}")
            results.append({"query_id": query_id, "status": "error", "error": str(exc), "elapsed": elapsed})

    print(f"\n[batch_qwen_sem] all {len(pending)} processed")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"[batch_qwen_sem] ok={ok}  errors={len(results)-ok}")

    if args.summary_path:
        Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[batch_qwen_sem] summary → {args.summary_path}")


if __name__ == "__main__":
    main()
