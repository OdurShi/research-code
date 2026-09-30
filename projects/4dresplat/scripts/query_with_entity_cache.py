"""
Per-query orchestrator with incremental entity-cache lookup.

Entry point: after Step 3a (Qwen query plan already produced).

Flow:
  [Cache lookup by subject phrase]
    HIT  → proposal_dir already exists → skip 3b+3c, go to 3d
    MISS → 3b (SAM3 text tracking)
         → 3c (Gaussian entity extraction, build_query_proposal_dir)
         → cache.add(subject, proposal_dir)
         → 3d (entity selection)
  → 3d: select_qwen_query_entities
  → 3e: render_query_video

Usage:
  python scripts/query_with_entity_cache.py \
    --run-dir       runs/d4resplat/hypernerf/americano \
    --dataset-dir   data/hypernerf/misc/americano \
    --query-plan    /tmp/out/query_plan.json \
    --query         "when does the person pour coffee?" \
    --output-dir    /tmp/out/q1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _c in (PROJECT_ROOT, PROJECT_ROOT / "external" / "4DGaussians"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from d4resplat.semantics.sam3_backend import run_sam3_text_query
from d4resplat.semantics.query_proposal_bridge import build_query_proposal_dir
from d4resplat.semantics import render_hypernerf_query_video


# ─── tiny helpers ─────────────────────────────────────────────────────────────

def _read_json(p: Path) -> dict[str, Any]:
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)

def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)

def _norm(s: str) -> str:
    return " ".join(re.sub(r"[_\-]", " ", s.lower()).split())

def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    return len(ta & tb) / len(ta | tb) if ta or tb else 0.0


# ─── entity cache ─────────────────────────────────────────────────────────────

class EntityCache:
    """
    Incremental phrase-indexed cache of Gaussian entity proposal dirs.

    Index file: <cache_dir>/entity_index.json
    {
      "entries": [
        {
          "subject":        "glass cup",
          "proposal_dir":   "/abs/path/to/glass_cup_abc123",
          "time_window":    {"start_frame": 30, "end_frame": 180},
          "source_dataset": "/abs/path/to/americano"
        }, ...
      ]
    }
    """
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = cache_dir / "entity_index.json"
        self._data: dict[str, Any] = _read_json(self._path) if self._path.exists() else {"entries": []}

    def _save(self) -> None:
        _write_json(self._path, self._data)

    def lookup(
        self,
        subject: str,
        dataset_dir: Path,
        time_window: dict[str, Any] | None = None,
        threshold: float = 0.60,
    ) -> dict[str, Any] | None:
        ds = str(dataset_dir.resolve())
        best: dict[str, Any] | None = None
        best_score = 0.0
        for entry in self._data.get("entries", []):
            if entry.get("source_dataset") != ds:
                continue
            score = _jaccard(subject, entry.get("subject", ""))
            if score < threshold:
                continue
            # Optional temporal overlap
            if time_window:
                qs = int(time_window.get("start_frame", 0))
                qe = int(time_window.get("end_frame", 10**9))
                cs = int(entry.get("time_window", {}).get("start_frame", 0))
                ce = int(entry.get("time_window", {}).get("end_frame", 10**9))
                if ce < qs or cs > qe:
                    continue
            if not Path(entry.get("proposal_dir", "")).exists():
                continue
            if score > best_score:
                best_score, best = score, entry
        if best:
            print(f"[cache] HIT  '{subject}'  score={best_score:.2f}  → {best['proposal_dir']}")
        else:
            print(f"[cache] MISS '{subject}'")
        return best

    def add(self, subject: str, proposal_dir: Path, dataset_dir: Path,
            time_window: dict[str, Any] | None = None) -> None:
        self._data.setdefault("entries", []).append({
            "subject":        subject,
            "proposal_dir":   str(proposal_dir.resolve()),
            "time_window":    time_window or {},
            "source_dataset": str(dataset_dir.resolve()),
        })
        self._save()
        print(f"[cache] STORED '{subject}'  → {proposal_dir}")


# ─── SAM time-window refinement ───────────────────────────────────────────────

def _refine_query_plan_time_window(
    query_plan_path: Path,
    tracks_path: Path,
    output_dir: Path,
) -> Path:
    """
    Read SAM3.1 tracks, extract active-frame range, and write a refined
    query_plan.json with time_window replaced by the SAM-derived window.

    Returns the path to the (possibly new) query_plan file.
    """
    tracks = _read_json(tracks_path)
    sam_tw = tracks.get("sam_time_window", {})
    sam_start = sam_tw.get("start_frame")
    sam_end   = sam_tw.get("end_frame")

    if sam_start is None or sam_end is None or sam_tw.get("active_frame_count", 0) == 0:
        print(f"[refine_tw] SAM returned no active frames — keeping Qwen seed window")
        return query_plan_path

    plan = _read_json(query_plan_path)
    old_tw = plan.get("time_window", {})
    plan["time_window"] = {
        "start_frame": int(sam_start),
        "end_frame":   int(sam_end),
        "confidence":  "sam_derived",
        "active_frame_count": int(sam_tw.get("active_frame_count", 0)),
        "qwen_seed_window": old_tw,
    }
    refined_path = output_dir / "query_plan_refined.json"
    _write_json(refined_path, plan)
    print(f"[refine_tw] time_window refined: "
          f"[{old_tw.get('start_frame')}, {old_tw.get('end_frame')}] (Qwen seed) → "
          f"[{sam_start}, {sam_end}] (SAM derived, {sam_tw.get('active_frame_count')} active frames)")
    return refined_path


# ─── step runners ──────────────────────────────────────────────────────────────

def _step_3b(
    dataset_dir: Path,
    query_plan_path: Path,
    tracks_dir: Path,
    sam3_checkpoint: str | None,
    frame_stride: int,
    n_ref_frames: int,
) -> Path:
    """SAM3.1 text tracking → sam3_query_tracks.json"""
    print("[3b] SAM3.1 text tracking …")
    out = run_sam3_text_query(
        dataset_dir=dataset_dir,
        query_plan_path=query_plan_path,
        output_dir=tracks_dir,
        sam3_checkpoint=sam3_checkpoint,
        frame_subsample_stride=frame_stride,
        n_ref_frames=n_ref_frames,
    )
    tracks_path = Path(out) / "sam3_query_tracks.json"
    print(f"[3b] done  → {tracks_path}")
    return tracks_path


def _step_3c(
    run_dir: Path,
    dataset_dir: Path,
    tracks_path: Path,
    proposal_dir: Path,
    max_track_frames: int,
    proposal_keep_ratio: float,
    min_gaussians: int,
    max_gaussians: int,
) -> Path:
    """2D mask → Gaussian entity proposals"""
    print("[3c] Building Gaussian entity proposals …")
    out = build_query_proposal_dir(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        tracks_path=tracks_path,
        output_dir=proposal_dir,
        max_track_frames=max_track_frames,
        proposal_keep_ratio=proposal_keep_ratio,
        min_gaussians=min_gaussians,
        max_gaussians=max_gaussians,
    )
    print(f"[3c] done  → {out}")
    return Path(out)


def _step_3d(
    assignments_path: Path,
    query: str,
    query_plan_path: Path,
    output_path: Path,
    qwen_model: str | None,
) -> Path:
    """Qwen entity selection"""
    import subprocess
    print("[3d] Qwen entity selection …")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "select_qwen_query_entities.py"),
        "--assignments-path", str(assignments_path),
        "--query", query,
        "--output-path", str(output_path),
        "--query-plan-path", str(query_plan_path),
    ]
    if qwen_model:
        cmd += ["--qwen-model", qwen_model]
    subprocess.run(cmd, check=True)
    print(f"[3d] done  → {output_path}")
    return output_path


def _step_3e(
    run_dir: Path,
    dataset_dir: Path,
    selection_path: Path,
    output_dir: Path,
    fps: int,
    stride: int,
) -> Path:
    """Render overlay video / masks"""
    print("[3e] Rendering query video …")
    out = render_hypernerf_query_video(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        selection_path=selection_path,
        output_dir=output_dir,
        fps=fps,
        stride=stride,
    )
    print(f"[3e] done  → {out}")
    return Path(out)


# ─── main orchestrator ────────────────────────────────────────────────────────

def run_query_pipeline(
    *,
    run_dir: Path,
    dataset_dir: Path,
    query: str,
    query_plan_path: Path,          # Step 3a output (already produced)
    output_dir: Path,
    entity_cache_dir: Path | None = None,
    sam3_checkpoint: str | None = None,
    qwen_model: str | None = None,
    cache_threshold: float = 0.60,
    frame_stride: int = 1,
    n_ref_frames: int = 3,
    max_track_frames: int = 16,
    proposal_keep_ratio: float = 0.03,
    min_gaussians: int = 256,
    max_gaussians: int = 4096,
    fps: int = 12,
    render_stride: int = 1,
    skip_render: bool = False,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read subject + time_window from 3a output
    plan = _read_json(query_plan_path)
    subject: str = str(plan.get("subject", "")).strip()
    if not subject:
        phrases = plan.get("query_subject_phrases") or []
        subject = str(phrases[0]).strip() if phrases else "object"
    time_window: dict[str, Any] = plan.get("time_window") or {}

    # ── Entity cache ──────────────────────────────────────────────────────────
    if entity_cache_dir is None:
        entity_cache_dir = run_dir / "entitybank" / "query_cache"
    cache = EntityCache(entity_cache_dir)

    cache_entry = None if force_rebuild else cache.lookup(
        subject=subject,
        dataset_dir=dataset_dir,
        time_window=time_window,
        threshold=cache_threshold,
    )

    if cache_entry is not None:
        # ── HIT: skip 3b + 3c ────────────────────────────────────────────────
        proposal_dir = Path(cache_entry["proposal_dir"])
        tracks_path: Path | None = None      # not needed; proposal already built
        cache_hit = True
        print(f"[orchestrator] Cache HIT — skipping 3b+3c")
    else:
        # ── MISS: run 3b then 3c, then cache ─────────────────────────────────
        cache_hit = False
        tracks_dir = output_dir / "sam3_tracks"
        tracks_path = _step_3b(
            dataset_dir=dataset_dir,
            query_plan_path=query_plan_path,
            tracks_dir=tracks_dir,
            sam3_checkpoint=sam3_checkpoint,
            frame_stride=frame_stride,
            n_ref_frames=n_ref_frames,
        )

        # ── Refine time_window from SAM active-frame range ────────────────────
        query_plan_path = _refine_query_plan_time_window(
            query_plan_path=query_plan_path,
            tracks_path=tracks_path,
            output_dir=output_dir,
        )
        time_window = _read_json(query_plan_path).get("time_window", {})

        safe = re.sub(r"[^\w]+", "_", subject.lower()).strip("_")[:40]
        proposal_dir = entity_cache_dir / f"{safe}_{uuid.uuid4().hex[:6]}"
        _step_3c(
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            tracks_path=tracks_path,
            proposal_dir=proposal_dir,
            max_track_frames=max_track_frames,
            proposal_keep_ratio=proposal_keep_ratio,
            min_gaussians=min_gaussians,
            max_gaussians=max_gaussians,
        )
        cache.add(subject, proposal_dir, dataset_dir, time_window)

    # ── 3d: entity selection ──────────────────────────────────────────────────
    assignments_path = proposal_dir / "assignments.json"
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Step 3c output missing: {assignments_path}\n"
            "Run with --force-rebuild to bypass cache."
        )
    selection_path = output_dir / "selected_entities.json"
    _step_3d(
        assignments_path=assignments_path,
        query=query,
        query_plan_path=query_plan_path,
        output_path=selection_path,
        qwen_model=qwen_model,
    )

    # ── 3e: render ────────────────────────────────────────────────────────────
    render_dir: Path | None = None
    if not skip_render:
        render_dir = output_dir / "render"
        _step_3e(
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            selection_path=selection_path,
            output_dir=render_dir,
            fps=fps,
            stride=render_stride,
        )

    result = {
        "query":          query,
        "subject":        subject,
        "cache_hit":      cache_hit,
        "proposal_dir":   str(proposal_dir),
        "tracks_path":    str(tracks_path) if tracks_path else None,
        "selection_path": str(selection_path),
        "render_dir":     str(render_dir) if render_dir else None,
    }
    _write_json(output_dir / "pipeline_result.json", result)
    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Per-query orchestrator (3b→3c→3d→3e) with entity cache")
    p.add_argument("--run-dir",          required=True)
    p.add_argument("--dataset-dir",      required=True)
    p.add_argument("--query",            required=True)
    p.add_argument("--query-plan",       required=True,  help="Step 3a output: query_plan.json")
    p.add_argument("--output-dir",       required=True)
    p.add_argument("--entity-cache-dir", default=None,   help="Default: <run-dir>/entitybank/query_cache")
    p.add_argument("--sam3-checkpoint",  default=None)
    p.add_argument("--qwen-model",       default=None)
    p.add_argument("--cache-threshold",  type=float, default=0.60)
    p.add_argument("--frame-stride",     type=int,   default=1,   help="SAM3 frame subsampling stride")
    p.add_argument("--n-ref-frames",     type=int,   default=3,   help="SAM3 reference frames")
    p.add_argument("--max-track-frames", type=int,   default=16)
    p.add_argument("--fps",              type=int,   default=12)
    p.add_argument("--render-stride",    type=int,   default=1)
    p.add_argument("--skip-render",      action="store_true")
    p.add_argument("--force-rebuild",    action="store_true", help="Bypass cache, always run 3b+3c")
    args = p.parse_args()

    result = run_query_pipeline(
        run_dir=Path(args.run_dir),
        dataset_dir=Path(args.dataset_dir),
        query=args.query,
        query_plan_path=Path(args.query_plan),
        output_dir=Path(args.output_dir),
        entity_cache_dir=Path(args.entity_cache_dir) if args.entity_cache_dir else None,
        sam3_checkpoint=args.sam3_checkpoint,
        qwen_model=args.qwen_model,
        cache_threshold=args.cache_threshold,
        frame_stride=args.frame_stride,
        n_ref_frames=args.n_ref_frames,
        max_track_frames=args.max_track_frames,
        fps=args.fps,
        render_stride=args.render_stride,
        skip_render=args.skip_render,
        force_rebuild=args.force_rebuild,
    )
    print("\n=== Pipeline Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
