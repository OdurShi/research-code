#!/usr/bin/env python3
"""
Build a flat benchmark JSON from scene-based benchmark files.

Merges benchmark/generated/queries_all.json + benchmark/scenes/*/answers.json
into a single flat list compatible with evaluate_ours_benchmark.py.

Usage:
  python scripts/build_benchmark_json.py \
    --benchmark-dir benchmark \
    --scenes americano espresso \
    --output reports/americano_benchmark.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-dir", default="benchmark",
                   help="Root of benchmark directory (default: benchmark/)")
    p.add_argument("--scenes", nargs="+", default=None,
                   help="Scenes to include (default: all with answers.json)")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--only-ready", action="store_true", default=False,
                   help="Only include answers with answer_status=ready (default: True)")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    bdir = repo_root / args.benchmark_dir

    queries_path = bdir / "generated" / "queries_all.json"
    queries_all: list[dict] = json.loads(queries_path.read_text())

    # Auto-detect scenes if not specified
    scenes_dir = bdir / "scenes"
    if args.scenes is None:
        available_scenes = [d.name for d in sorted(scenes_dir.iterdir()) if d.is_dir()]
    else:
        available_scenes = args.scenes

    # Build answer_id → answer map
    answer_map: dict[str, dict] = {}
    for scene in available_scenes:
        answers_path = scenes_dir / scene / "answers.json"
        if not answers_path.exists():
            continue
        for ans in json.loads(answers_path.read_text()):
            answer_map[ans["answer_id"]] = ans

    # Filter queries to requested scenes
    scene_set = set(available_scenes)

    result: list[dict] = []
    for q in queries_all:
        scene_key = q.get("scene_key", "")
        if scene_key not in scene_set:
            continue

        ans = answer_map.get(q.get("answer_id", ""))
        if ans is None:
            continue
        if args.only_ready and ans.get("answer_status") != "ready":
            continue

        item = {
            "query_id": q["query_id"],
            "question": q.get("query_text", ""),
            "scene_key": scene_key,
            "answer_id": ans["answer_id"],
            "target_entity_ids": ans.get("target_entity_ids", []),
            "target_labels": ans.get("target_labels", []),
            "ground_truth": {
                # Use eval_frame_ids (sparse GT annotation frames) as existence_frames.
                # eval_frame_ids = frames annotated for eval; target_frame_ids = all active frames.
                "existence_frames": ans.get("eval_frame_ids") or ans.get("target_frame_ids", []),
                "frames": ans.get("frames", []),
            },
        }
        result.append(item)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(result)} queries ({len(available_scenes)} scenes) → {out}")


if __name__ == "__main__":
    main()
