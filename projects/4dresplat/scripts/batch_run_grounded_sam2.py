#!/usr/bin/env python3
"""
Batch SAM runner: loads GroundingDINO + SAM3.1 ONCE and processes all queries.

Input manifest JSON (--manifest):
  [
    {
      "query_id":        "americano_q001",
      "dataset_dir":     "/abs/path/to/dataset",
      "query_plan_path": "/abs/path/to/query_plan.json",
      "output_dir":      "/abs/path/to/grounded_sam2/"
    }, ...
  ]

Skips entries where output_dir/grounded_sam2_query_tracks.json already exists.
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

from d4resplat.semantics.sam3_backend import (
    load_grounded_sam2_models,
    run_grounded_sam2_query,
)

DEFAULT_SAM2_MODEL = str(PROJECT_ROOT / "sam3.1" / "sam3.1_multiplex.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SAM runner (single model load)")
    parser.add_argument("--manifest",              required=True)
    parser.add_argument("--grounding-model-id",    default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam2-model-id",          default=DEFAULT_SAM2_MODEL)
    parser.add_argument("--detector-frame-stride", type=int,   default=6)
    parser.add_argument("--max-detector-frames",   type=int,   default=48)
    parser.add_argument("--detection-top-k",       type=int,   default=5)
    parser.add_argument("--box-threshold",         type=float, default=0.25)
    parser.add_argument("--text-threshold",        type=float, default=0.20)
    parser.add_argument("--prompt-type",           default="point")
    parser.add_argument("--num-point-prompts",     type=int,   default=16)
    parser.add_argument("--track-window-radius",   type=int,   default=160)
    parser.add_argument("--frame-subsample-stride",type=int,   default=10)
    parser.add_argument("--num-anchor-seeds",      type=int,   default=3)
    parser.add_argument("--summary-path",          default=None)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    done_key = "grounded_sam2_query_tracks.json"
    pending = [item for item in items
               if not (Path(item["output_dir"]) / done_key).exists()]
    print(f"[sam_batch] {len(items)} total, {len(pending)} pending, "
          f"{len(items)-len(pending)} already done")
    if not pending:
        print("[sam_batch] all done, exiting")
        return

    print("[sam_batch] loading models…")
    t0 = time.time()
    models = load_grounded_sam2_models(
        grounding_model_id=args.grounding_model_id,
        sam2_model_id=args.sam2_model_id,
    )
    print(f"[sam_batch] models ready in {time.time()-t0:.1f}s")

    results = []
    for idx, item in enumerate(pending):
        query_id       = item["query_id"]
        dataset_dir    = item["dataset_dir"]
        query_plan_path = item["query_plan_path"]
        output_dir     = item["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n[sam_batch] [{idx+1}/{len(pending)}] {query_id}")
        t_q = time.time()
        try:
            run_grounded_sam2_query(
                dataset_dir=dataset_dir,
                query_plan_path=query_plan_path,
                output_dir=output_dir,
                grounding_model_id=args.grounding_model_id,
                sam2_model_id=args.sam2_model_id,
                detector_frame_stride=args.detector_frame_stride,
                max_detector_frames=args.max_detector_frames,
                detection_top_k=args.detection_top_k,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                prompt_type=args.prompt_type,
                num_point_prompts=args.num_point_prompts,
                track_window_radius=args.track_window_radius,
                frame_subsample_stride=args.frame_subsample_stride,
                num_anchor_seeds=args.num_anchor_seeds,
                _preloaded_models=models,
            )
            elapsed = time.time() - t_q
            print(f"[sam_batch] {query_id} done in {elapsed:.1f}s")
            results.append({"query_id": query_id, "status": "ok", "elapsed": elapsed})
        except Exception as exc:
            elapsed = time.time() - t_q
            print(f"[sam_batch] {query_id} ERROR ({elapsed:.1f}s): {exc}")
            results.append({"query_id": query_id, "status": "error",
                            "error": str(exc), "elapsed": elapsed})

    ok = sum(1 for r in results if r["status"] == "ok")
    total_elapsed = sum(r["elapsed"] for r in results)
    print(f"\n[sam_batch] {len(pending)} queries: ok={ok} errors={len(results)-ok} "
          f"total={total_elapsed:.1f}s ({total_elapsed/max(len(pending),1):.1f}s/query)")

    if args.summary_path:
        Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[sam_batch] summary → {args.summary_path}")


if __name__ == "__main__":
    main()
