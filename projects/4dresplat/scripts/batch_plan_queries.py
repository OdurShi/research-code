#!/usr/bin/env python3
"""
Batch query planner: loads QwenQueryPlanner ONCE and plans all queries sequentially.
Saves ~45s model-load overhead per query.

Input manifest JSON (--manifest):
  [
    {
      "query_id":   "americano_q001",
      "query_text": "The hand pouring coffee.",
      "dataset_dir": "/abs/path/to/dataset",
      "output_path": "/abs/path/to/query_plan.json"
    }, ...
  ]

Skips entries where output_path already exists.
Writes a summary to --summary-path (optional).
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

from d4resplat.semantics.qwen_query_planner import (
    QwenQueryPlanner,
    _resolve_qwen_model,
    plan_query_entities,
    plan_query_entities_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch query planner (single Qwen load)")
    parser.add_argument("--manifest",      required=True, help="JSON list of {query_id,query_text,dataset_dir,output_path}")
    parser.add_argument("--qwen-model",    default=None)
    parser.add_argument("--coarse-stride", type=int, default=32)
    parser.add_argument("--fine-frames",   type=int, default=16)
    parser.add_argument("--batch-size",    type=int, default=32, help="queries per model.generate call")
    parser.add_argument("--summary-path",  default=None)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    pending = [item for item in items if not Path(item["output_path"]).exists()]
    print(f"[batch_plan] {len(items)} total, {len(pending)} pending, {len(items)-len(pending)} already done")
    if not pending:
        print("[batch_plan] all done, exiting")
        return

    print("[batch_plan] loading QwenQueryPlanner…")
    t0 = time.time()
    model_path = _resolve_qwen_model(args.qwen_model)
    teacher = QwenQueryPlanner(model_path)
    print(f"[batch_plan] model loaded in {time.time()-t0:.1f}s  ({model_path})")

    for item in pending:
        Path(item["output_path"]).parent.mkdir(parents=True, exist_ok=True)

    t_batch = time.time()
    plans = plan_query_entities_batch(pending, batch_size=args.batch_size, _teacher=teacher)
    total_elapsed = time.time() - t_batch

    results = []
    for item, plan in zip(pending, plans):
        query_id = item["query_id"]
        if plan:
            results.append({"query_id": query_id, "status": "ok", "subject": plan.get("subject"),
                            "time_window": plan.get("time_window")})
        else:
            results.append({"query_id": query_id, "status": "error", "error": "empty plan"})

    print(f"\n[batch_plan] {len(pending)} queries in {total_elapsed:.1f}s  "
          f"({total_elapsed/max(len(pending),1):.1f}s/query)")

    print(f"\n[batch_plan] all {len(pending)} queries processed")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"[batch_plan] ok={ok}  errors={len(results)-ok}")

    if args.summary_path:
        Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[batch_plan] summary → {args.summary_path}")


if __name__ == "__main__":
    main()
