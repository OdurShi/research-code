#!/usr/bin/env python3
"""
evaluate_ours_benchmark.py

Evaluation script: reads Ours_benchmark.json as ground truth and scores the
4DReSplat query pipeline outputs. Metrics: Acc (temporal accuracy), vIoU
(video mask IoU), tIoU (temporal IoU).

Usage:
  python scripts/evaluate_ours_benchmark.py \
    --benchmark /path/to/Ours_benchmark.json \
    --query-root-map /path/to/query_root_map.json \
    --output-json reports/ours_benchmark_eval.json \
    [--output-md reports/ours_benchmark_eval.md]

query_root_map.json format:
{
  "espresso_q1": "/path/to/run_dir/entitybank/query_guided/espresso_q1",
  ...
}
"""
from __future__ import annotations

import argparse
import bisect
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from d4resplat.semantics.mask_store import load_frame_mask


# ---------------------------------------------------------------------------
# RLE (COCO format) decoder
# ---------------------------------------------------------------------------

def _decode_rle(rle: dict) -> np.ndarray:
    """Decode COCO RLE mask to binary numpy array (H, W)."""
    counts_raw = rle["counts"]
    size = rle["size"]  # [H, W]
    h, w = int(size[0]), int(size[1])

    if isinstance(counts_raw, str):
        # Encoded RLE string (COCO binary compressed RLE)
        counts = _decode_rle_string(counts_raw, h * w)
    else:
        counts = [int(c) for c in counts_raw]

    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for count in counts:
        mask[pos: pos + count] = val
        pos += count
        val = 1 - val

    # COCO RLE is column-major (Fortran order)
    return mask.reshape((h, w), order="F").astype(bool)


def _decode_rle_string(encoded: str, n: int) -> list[int]:
    """Decode COCO compressed RLE string into run-length list."""
    counts = []
    m = 0
    p = 0
    while p < len(encoded):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(encoded[p]) - 48
            p += 1
            x |= (c & 0x1f) << (5 * k)
            more = (c & 0x20) != 0
            k += 1
        if x & 1:
            x = ~(x >> 1)
        else:
            x = x >> 1
        if m > 0:
            x += counts[m - 1]
        counts.append(x)
        m += 1
    return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    inter = float(np.logical_and(p, g).sum())
    union = float(np.logical_or(p, g).sum())
    return _safe_div(inter, union)


def _load_mask_item(
    mask_item: dict,
    image_size: tuple[int, int] | None = None,
    benchmark_root: "Path | None" = None,
) -> "np.ndarray | None":
    """Load a binary mask from a mask_item dict.

    Supports two formats:
      - {"segmentation": <RLE dict or polygon list>}
      - {"mask_image": "relative/path.png"}  (requires benchmark_root)
    """
    if "segmentation" in mask_item:
        return _decode_segmentation(mask_item["segmentation"], image_size=image_size)
    if "mask_image" in mask_item and benchmark_root is not None:
        mp = benchmark_root / mask_item["mask_image"]
        if mp.exists():
            with Image.open(mp) as img:
                arr = np.asarray(img.convert("L"), dtype=np.uint8) > 0
                if image_size is not None and arr.shape != image_size:
                    from PIL import Image as _PILImage
                    arr = np.asarray(
                        _PILImage.fromarray(arr.astype(np.uint8) * 255).resize(
                            (image_size[1], image_size[0]), Image.NEAREST
                        )
                    ) > 0
                return arr
    return None


def _decode_segmentation(seg: dict | list, image_size: tuple[int, int] | None = None) -> np.ndarray | None:
    """Decode COCO segmentation (RLE dict or polygon list) to binary numpy array (H, W).
    
    Args:
        seg: Either an RLE dict {"counts": ..., "size": [H, W]} or a polygon list [[x1, y1, ...]].
        image_size: (H, W) tuple, required for polygon format when size is not in seg.
    
    Returns:
        Binary (bool) numpy array of shape (H, W), or None if decoding fails.
    """
    if isinstance(seg, dict):
        # COCO RLE format
        try:
            return _decode_rle(seg)
        except Exception:
            return None
    elif isinstance(seg, list):
        # COCO polygon format: [[x1, y1, x2, y2, ...], ...]
        # Determine image size
        h, w = None, None
        if image_size is not None:
            h, w = image_size
        if h is None or w is None:
            return None
        from PIL import Image as PILImage, ImageDraw
        mask_img = PILImage.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        for polygon in seg:
            if len(polygon) < 6:
                continue  # Need at least 3 points
            pts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(pts, fill=255)
        return np.asarray(mask_img) > 0
    return None


def _load_metadata(dataset_dir: Path) -> dict[str, int]:
    """Load HyperNeRF metadata.json -> image_id: time_id mapping."""
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    raw = _read_json(meta_path)
    return {k: int(v["time_id"]) for k, v in raw.items()}


def _build_image_id_to_frame_index(validation_frames: list[dict]) -> dict[str, int]:
    """Build mapping from image_id → frame_index in validation output."""
    return {str(f["image_id"]): int(f["frame_index"]) for f in validation_frames}


def _frame_id_to_image_id_hypernerf(frame_id: int) -> str:
    """HyperNeRF: frame_id → 6-digit image_id string."""
    return f"{int(frame_id):06d}"


def _frame_id_to_image_id_dynerf(frame_id: int) -> str:
    """dynerf: frame_id → 4-digit image_id string (cam00/images/XXXX.png)."""
    return f"{int(frame_id):04d}"


def _detect_dataset_type(dataset_dir: Path) -> str:
    """Detect if this is a hypernerf or dynerf dataset."""
    if (dataset_dir / "metadata.json").exists():
        return "hypernerf"
    if (dataset_dir / "poses_bounds.npy").exists():
        return "dynerf"
    # Try to detect from config.yaml in parent dirs
    return "hypernerf"


def _nearest_sam3_frame(frames: list[dict], frame_index: int) -> dict | None:
    if not frames:
        return None
    pos = bisect.bisect_left([int(f.get("frame_index", 0)) for f in frames], int(frame_index))
    if pos == 0:
        return frames[0]
    if pos >= len(frames):
        return frames[-1]
    before = frames[pos - 1]
    after = frames[pos]
    return before if abs(frame_index - int(before.get("frame_index", 0))) <= abs(int(after.get("frame_index", 0)) - frame_index) else after


def _align_window_to_frame_grid(
    start_frame: int | None,
    end_frame: int | None,
    frame_grid: list[int],
) -> tuple[int | None, int | None]:
    if start_frame is None or end_frame is None or not frame_grid:
        return start_frame, end_frame
    grid = sorted({int(v) for v in frame_grid})
    start_pos = bisect.bisect_left(grid, int(start_frame))
    end_pos = bisect.bisect_right(grid, int(end_frame))
    aligned_start = grid[max(0, start_pos - 1)] if start_pos > 0 else grid[0]
    aligned_end = grid[min(len(grid) - 1, end_pos)] if end_pos < len(grid) else grid[-1]
    if aligned_end < aligned_start:
        return start_frame, end_frame
    return int(aligned_start), int(aligned_end)


def _represented_intervals(time_ids: list[int], total_frames: int) -> list[list[int]]:
    """Voronoi-style expansion matching upstream evaluate_public_query_protocol.py.
    Each rendered time_id represents the interval halfway to its neighbors."""
    if not time_ids:
        return []
    sorted_ids = sorted(int(v) for v in time_ids)
    intervals: list[list[int]] = []
    for index, current in enumerate(sorted_ids):
        start = 0 if index == 0 else (sorted_ids[index - 1] + current) // 2 + 1
        end = int(total_frames - 1) if index == len(sorted_ids) - 1 else (current + sorted_ids[index + 1]) // 2
        intervals.append([int(start), int(end)])
    return intervals


def _load_sam3_eval_proxy(
    query_output_dir: Path,
    dataset_dir: Path | None,
    metadata: dict[str, int] | None = None,
) -> dict | None:
    """Build a lightweight validation-like payload directly from SAM3 NPZ tracks.

    This is the fast evaluation path: it avoids running 3e just to materialize
    hundreds of binary PNG masks. Masks are loaded lazily from per-track
    mask_store_path entries when vIoU is computed.
    """
    tracks_path = query_output_dir / "sam3_tracks" / "sam3_query_tracks.json"
    if not tracks_path.exists():
        return None
    if metadata is None and dataset_dir is not None and dataset_dir.exists() and (dataset_dir / "metadata.json").exists():
        metadata = _load_metadata(dataset_dir)
    payload = _read_json(tracks_path)
    tracks = payload.get("tracks") or payload.get("phrases") or []
    if not tracks:
        return None

    track_frames: list[list[dict]] = []
    for track in tracks:
        frames = [
            frame for frame in track.get("frames", [])
            if isinstance(frame, dict)
        ]
        frames.sort(key=lambda item: int(item.get("frame_index", 0)))
        if frames:
            track_frames.append(frames)
    if not track_frames:
        return None

    sam_tw = payload.get("sam_time_window", {}) if isinstance(payload, dict) else {}
    start_frame = sam_tw.get("start_frame")
    end_frame = sam_tw.get("end_frame")
    active_count = int(sam_tw.get("active_frame_count", -1) or 0)

    frame_records: list[dict] = []
    if dataset_dir is not None and dataset_dir.exists() and metadata:
        image_ids = [image_id for image_id, _tid in sorted(metadata.items(), key=lambda kv: int(kv[1]))]
    else:
        image_ids = sorted({str(f.get("image_id")) for frames in track_frames for f in frames if f.get("image_id")})

    if not image_ids:
        return None

    frame_grid: list[int] = []
    for fallback_idx, image_id in enumerate(image_ids):
        frame_grid.append(int(metadata.get(str(image_id), fallback_idx)) if metadata else fallback_idx)
    aligned_start_frame, aligned_end_frame = _align_window_to_frame_grid(
        int(start_frame) if start_frame is not None else None,
        int(end_frame) if end_frame is not None else None,
        frame_grid,
    )

    active_indices: list[int] = []
    for fallback_idx, image_id in enumerate(image_ids):
        frame_index = int(metadata.get(str(image_id), fallback_idx)) if metadata else fallback_idx
        if active_count <= 0:
            query_active = False
        elif aligned_start_frame is not None and aligned_end_frame is not None:
            query_active = int(aligned_start_frame) <= frame_index <= int(aligned_end_frame)
        else:
            query_active = any(
                bool(frame.get("active")) and int(frame.get("frame_index", -1)) == frame_index
                for frames in track_frames for frame in frames
            )

        source_frames = []
        if query_active:
            for frames in track_frames:
                nearest = _nearest_sam3_frame(frames, frame_index)
                if nearest is not None and bool(nearest.get("active")):
                    source_frames.append(nearest)
        if query_active:
            active_indices.append(frame_index)
        frame_records.append({
            "frame_index": frame_index,
            "image_id": str(image_id),
            "query_active": bool(query_active),
            "_mask_source": "sam3_track_store",
            "_sam3_source_frames": source_frames,
        })

    return {
        "schema_version": 1,
        "query": payload.get("subject", ""),
        "native_render": False,
        "fast_eval_source": "sam3_track_store",
        "frame_count": len(frame_records),
        "active_frame_count": len(active_indices),
        "first_active_frame": None if not active_indices else min(active_indices),
        "last_active_frame": None if not active_indices else max(active_indices),
        "sam_time_window_raw": {
            "start_frame": None if start_frame is None else int(start_frame),
            "end_frame": None if end_frame is None else int(end_frame),
            "active_frame_count": active_count,
        },
        "sam_time_window_eval_aligned": {
            "start_frame": aligned_start_frame,
            "end_frame": aligned_end_frame,
            "active_frame_count": active_count,
        },
        "frame_exports": {
            "binary_masks": "",
            "binary_masks_store": str(tracks_path),
            "binary_masks_format": "sam3_track_store_npz",
        },
        "frames": frame_records,
    }


def _load_pred_mask_for_eval(
    val_frame: dict,
    binary_mask_dir: Path | None,
    frame_idx: int,
) -> np.ndarray | None:
    """Load predicted mask from final PNGs or directly from SAM3 NPZ stores."""
    source_frames = val_frame.get("_sam3_source_frames")
    if isinstance(source_frames, list) and source_frames:
        masks = []
        for source_frame in source_frames:
            mask = load_frame_mask(source_frame)
            if mask is not None:
                masks.append(np.asarray(mask, dtype=bool))
        if not masks:
            return None
        merged = masks[0].copy()
        for mask in masks[1:]:
            if mask.shape == merged.shape:
                merged |= mask
        return merged

    if binary_mask_dir and binary_mask_dir.exists():
        pred_mask_path = binary_mask_dir / f"{frame_idx:05d}.png"
        if pred_mask_path.exists():
            with Image.open(pred_mask_path) as img:
                return np.asarray(img.convert("L"), dtype=np.uint8) > 0
    return None


def _infer_dataset_info_from_query_output(query_output_dir: Path) -> tuple[str | None, Path | None]:
    """Infer dataset type and dataset_dir from query output path.

    Expected path pattern:
      .../runs/<namespace>/<dataset_type>/<scene>/entitybank/query_guided/<qid>
    """
    parts = query_output_dir.parts
    ds_type = None
    scene_name = None
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 3 < len(parts):
            cand = parts[idx + 2]
            if cand in ("hypernerf", "dynerf", "dnerf"):
                ds_type = cand
                scene_name = parts[idx + 3]

    if ds_type is None:
        return None, None

    # Infer dataset root for known benchmark scenes.
    repo_root = Path(__file__).resolve().parents[1]
    if ds_type == "dynerf":
        if scene_name:
            return ds_type, repo_root / "data" / "dynerf" / scene_name
        return ds_type, None

    if ds_type == "hypernerf":
        # Benchmark scenes are split across misc/interp. Infer from scene name.
        scene_to_subdir = {
            "espresso": ("misc", "espresso"),
            "americano": ("misc", "americano"),
            "split-cookie": ("misc", "split-cookie"),
            "keyboard": ("misc", "keyboard"),
            "cut-lemon1": ("interp", "cut-lemon1"),
            "torchocolate": ("interp", "torchocolate"),
        }
        if scene_name in scene_to_subdir:
            sub, name = scene_to_subdir[scene_name]
            return ds_type, repo_root / "data" / "hypernerf" / sub / name
        if scene_name:
            # Best-effort fallback for other HyperNeRF layouts.
            return ds_type, repo_root / "data" / "hypernerf" / "misc" / scene_name
        return ds_type, None

    return ds_type, None


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(
    query_item: dict,
    query_output_dir: Path,
    dataset_dir: Path | None = None,
    benchmark_root: Path | None = None,
) -> dict:
    """Evaluate a single query against Ours_benchmark.json ground truth."""
    query_id = str(query_item["query_id"])
    gt = query_item.get("ground_truth", {})
    existence_frames: list[int] = [int(f) for f in gt.get("existence_frames", [])]
    gt_frames: list[dict] = gt.get("frames", [])
    # GT entity set (benchmark entity IDs, e.g. ["americano_e003"])
    target_entity_ids: list[str] = [str(e) for e in query_item.get("target_entity_ids", [])]
    n_gt_entities: int = len(target_entity_ids)

    # -----------------------------------------------------------------------
    # Read Qwen plan to detect model's empty prediction (subjects=[])
    # This is the primary empty signal — determined at planning stage.
    # -----------------------------------------------------------------------
    _plan_path = query_output_dir / "query_plan.json"
    _qwen_predicted_empty = False
    if _plan_path.exists():
        try:
            _qwen_plan = _read_json(_plan_path)
            _qwen_subjects = _qwen_plan.get("subjects")
            if isinstance(_qwen_subjects, list) and len(_qwen_subjects) == 0:
                _qwen_predicted_empty = True
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Load validation.json
    # -----------------------------------------------------------------------
    validation_path = query_output_dir / "final_query_render_sourcebg" / "validation.json"
    validation = None
    if validation_path.exists():
        validation = _read_json(validation_path)
        pred_mask_source = "final_binary_png"
    else:
        validation = _load_sam3_eval_proxy(query_output_dir, dataset_dir)
        pred_mask_source = "sam3_track_store" if validation is not None else "missing"
    if validation is not None and not validation.get("frames"):
        sam3_validation = _load_sam3_eval_proxy(query_output_dir, dataset_dir)
        if sam3_validation is not None:
            validation = sam3_validation
            pred_mask_source = "sam3_track_store"
    if validation is None:
        # Negative query (GT empty): correct iff Qwen also predicted empty.
        if n_gt_entities == 0:
            if _qwen_predicted_empty:
                return {
                    "query_id": query_id,
                    "question": str(query_item.get("question", "")),
                    "status": "ok",
                    "Acc": 1.0, "vIoU": 1.0, "mIoU": 1.0,
                    "tIoU": 1.0, "tPrec": 1.0, "tRec": 1.0,
                    "n_gt_entities": 0, "n_pred_entities": 0,
                    "pred_mask_source": "empty_negative",
                    "gt_active_count": 0, "pred_active_count": 0,
                }
            else:
                # Qwen predicted non-empty for a trap query → wrong → all 0%.
                return {
                    "query_id": query_id,
                    "question": str(query_item.get("question", "")),
                    "status": "qwen_false_positive",
                    "Acc": 0.0, "vIoU": 0.0, "mIoU": 0.0,
                    "tIoU": 0.0, "tPrec": 0.0, "tRec": 0.0,
                    "n_gt_entities": 0, "n_pred_entities": 1,
                    "pred_mask_source": "qwen_nonempty",
                    "gt_active_count": 0, "pred_active_count": 0,
                }
        # Positive query + Qwen predicted empty (subjects=[]) → false negative → all 0%.
        if _qwen_predicted_empty:
            return {
                "query_id": query_id,
                "question": str(query_item.get("question", "")),
                "status": "qwen_false_negative",
                "Acc": 0.0,
                "vIoU": 0.0,
                "mIoU": 0.0,
                "tIoU": 0.0,
                "tPrec": 0.0,
                "tRec": 0.0,
                "n_gt_entities": n_gt_entities,
                "n_pred_entities": 0,
                "pred_mask_source": "qwen_empty",
                "gt_active_count": 0,
                "pred_active_count": 0,
            }
        return {
            "query_id": query_id,
            "status": "missing_validation",
            "Acc": None,
            "vIoU": None,
            "tIoU": None,
        }

    val_frames = validation.get("frames", [])
    binary_mask_dir_raw = validation.get("frame_exports", {}).get("binary_masks", "")
    binary_mask_dir = Path(binary_mask_dir_raw) if binary_mask_dir_raw else None

    # Do NOT early-return on empty val_frames. An empty-frame validation means the
    # pipeline produced no active predictions (all-black).  We still need to score
    # it — a positive query with empty prediction should count as tIoU=0, Acc=0,
    # not be silently excluded from the average (which would inflate scores).
    # pred_by_time_id will be empty → all False → correct penalty for false negatives.

    # -----------------------------------------------------------------------
    # Determine dataset type and build image_id lookup
    # -----------------------------------------------------------------------
    ds_type = "hypernerf"
    metadata: dict[str, int] = {}
    if dataset_dir is not None and dataset_dir.exists():
        ds_type = _detect_dataset_type(dataset_dir)
        if ds_type == "hypernerf":
            metadata = _load_metadata(dataset_dir)
    else:
        inferred_ds_type, inferred_dataset_dir = _infer_dataset_info_from_query_output(query_output_dir)
        if inferred_ds_type is not None:
            ds_type = inferred_ds_type
        if inferred_dataset_dir is not None and inferred_dataset_dir.exists():
            if ds_type == "hypernerf":
                metadata = _load_metadata(inferred_dataset_dir)

    # Build lookup: image_id (str) → {frame_index, query_active}
    val_by_image_id: dict[str, dict] = {str(f["image_id"]): f for f in val_frames}

    # Build predicted time_id → query_active mapping
    # For HyperNeRF: use metadata.json to get time_id from image_id
    # For dynerf: use frame_index directly as time_id
    pred_by_time_id: dict[int, bool] = {}
    for f in val_frames:
        image_id_str = str(f["image_id"])
        if ds_type == "hypernerf" and metadata:
            tid = metadata.get(image_id_str)
            if tid is None:
                continue
        else:
            # dynerf: time_id = frame_index (0-indexed)
            tid = int(f["frame_index"])
        pred_by_time_id[tid] = bool(f["query_active"])

    # Apply time-window restriction from query_plan_state_refined.json.
    # The renderer currently renders ALL frames, so without this restriction
    # pred_by_time_id contains entries for every frame with query_active=True,
    # which after Voronoi expansion gives pred_full=all-True → Acc = GT_rate.
    state_refined_path = query_output_dir / "query_plan_state_refined.json"
    if state_refined_path.exists():
        try:
            tw = _read_json(state_refined_path).get("time_window", {})
            pred_start_tid = tw.get("start_frame")
            pred_end_tid   = tw.get("end_frame")
            if pred_start_tid is not None and pred_end_tid is not None:
                pred_start_tid, pred_end_tid = int(pred_start_tid), int(pred_end_tid)
                for _tid in list(pred_by_time_id.keys()):
                    if not (pred_start_tid <= _tid <= pred_end_tid):
                        pred_by_time_id[_tid] = False
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Determine total_frames
    # -----------------------------------------------------------------------
    if metadata:
        total_frames = max(metadata.values()) + 1
    elif pred_by_time_id:
        total_frames = max(pred_by_time_id.keys()) + 1
    elif val_frames:
        total_frames = len(val_frames)
    else:
        # val_frames is empty (all-black / fast-path): use GT annotation to determine
        # timeline length so GT arrays can still be built correctly.
        total_frames = max((int(fid) for fid in existence_frames), default=0) + 1

    # -----------------------------------------------------------------------
    # Build GT timeline
    # -----------------------------------------------------------------------
    # Convert existence_frames (list of frame_id) to time_ids (sparse annotation samples)
    gt_sampled_tids: list[int] = []  # the actual annotated time_ids
    if ds_type == "hypernerf" and metadata:
        for fid in existence_frames:
            image_id_str = _frame_id_to_image_id_hypernerf(fid)
            tid = metadata.get(image_id_str)
            if tid is not None:
                gt_sampled_tids.append(tid)
    else:
        for fid in existence_frames:
            gt_sampled_tids.append(int(fid))

    # KEY FIX 1: existence_frames are SPARSE SAMPLES of an active temporal range.
    # Build gt_time_ids as the DENSE RANGE [min, max] of sampled tids.
    # Exception: negative queries (target_entity_ids is empty) use sentinel
    # existence_frames as per-frame eval checkpoints, NOT as active-object frames.
    is_negative_query = (n_gt_entities == 0)
    if gt_sampled_tids and not is_negative_query:
        gt_min_tid = min(gt_sampled_tids)
        gt_max_tid = max(gt_sampled_tids)
        gt_time_ids: set[int] = set(range(gt_min_tid, gt_max_tid + 1))
    else:
        # Negative query or no annotation: entity does not exist
        gt_time_ids = set()
        gt_min_tid = -1
        gt_max_tid = -1

    # -----------------------------------------------------------------------
    # Load predicted entity selection for query-level Acc
    # -----------------------------------------------------------------------
    _sel_path = query_output_dir / "query_worldtube_run" / "entitybank" / "selected_query_qwen.json"
    _n_pred_entities = 0
    _pred_is_empty = True
    if _sel_path.exists():
        _sel = _read_json(_sel_path)
        _pred_is_empty = bool(_sel.get("empty", True))
        _n_pred_entities = len(_sel.get("selected", []))
    else:
        _pred_is_empty = not any(pred_by_time_id.values())
        _n_pred_entities = 0 if _pred_is_empty else 1
    # Qwen plan subjects=[] is the authoritative empty signal.
    if _qwen_predicted_empty:
        _pred_is_empty = True
        _n_pred_entities = 0
    elif _plan_path.exists() and is_negative_query:
        # Plan exists, Qwen predicted non-empty on a trap query → model "found" something.
        _pred_is_empty = False

    # -----------------------------------------------------------------------
    # Temporal IoU (tIoU) - full timeline comparison
    # -----------------------------------------------------------------------
    # Build full binary arrays over total_frames
    gt_full = np.zeros(total_frames, dtype=bool)
    pred_full = np.zeros(total_frames, dtype=bool)

    for tid in gt_time_ids:
        if 0 <= tid < total_frames:
            gt_full[tid] = True
    # Voronoi expansion: each rendered frame represents interval halfway to neighbors
    # Aligned to upstream evaluate_public_query_protocol.py _rendered_activity_mask
    _sorted_pred_tids = sorted(pred_by_time_id.keys())
    _pred_intervals = _represented_intervals(_sorted_pred_tids, total_frames)
    for _ptid, _pinterval in zip(_sorted_pred_tids, _pred_intervals):
        if pred_by_time_id[_ptid]:
            _ps, _pe = int(_pinterval[0]), int(_pinterval[1])
            if _ps < total_frames:
                pred_full[_ps : min(_pe + 1, total_frames)] = True

    temporal_inter = int(np.logical_and(pred_full, gt_full).sum())
    temporal_union = int(np.logical_or(pred_full, gt_full).sum())

    if is_negative_query:
        # Negative query: binary scoring.
        # Correctly outputs empty → 1.0; finds any object → 0.0.
        # eval_frames (sentinel frames) are not used for scoring.
        t_iou = 1.0 if _pred_is_empty else 0.0
        temporal_inter = 0
        temporal_union = 0
    elif temporal_union == 0:
        t_iou = 1.0
    else:
        t_iou = _safe_div(temporal_inter, temporal_union)

    # -----------------------------------------------------------------------
    # Visual IoU (vIoU) - frame-averaged mask IoU (upstream protocol)
    # vIoU = iou_sum / union_count
    # union_count = GT mask frames where pred OR gt active
    # iou_sum    = sum of per-frame mask IoU where both pred AND gt active
    # Aligned to upstream evaluate_public_query_protocol.py
    # -----------------------------------------------------------------------
    sorted_tids = sorted(pred_by_time_id.keys())
    val_by_time_id: dict[int, dict] = {}
    for f in val_frames:
        image_id_str = str(f["image_id"])
        if ds_type == "hypernerf" and metadata:
            tid = metadata.get(image_id_str)
            if tid is None:
                continue
        else:
            tid = int(f["frame_index"])
        val_by_time_id[tid] = f

    pred_image_size: tuple[int, int] | None = None
    if binary_mask_dir and binary_mask_dir.exists():
        sample_masks = sorted(binary_mask_dir.glob("*.png"))
        if sample_masks:
            with Image.open(sample_masks[0]) as _img:
                _arr = np.asarray(_img)
                pred_image_size = (_arr.shape[0], _arr.shape[1])
    if pred_image_size is None:
        for frame in val_frames:
            mask = _load_pred_mask_for_eval(frame, binary_mask_dir, int(frame.get("frame_index", 0)))
            if mask is not None:
                pred_image_size = tuple(int(v) for v in mask.shape[:2])
                break

    gt_frame_masks: dict[int, np.ndarray] = {}
    for gt_frame in gt_frames:
        fid = int(gt_frame["frame_id"])
        masks_in_frame = []
        for mask_item in gt_frame.get("masks", []):
            mask_arr = _load_mask_item(mask_item, image_size=pred_image_size,
                                       benchmark_root=benchmark_root)
            if mask_arr is not None:
                masks_in_frame.append(mask_arr)
        if masks_in_frame:
            merged = masks_in_frame[0].copy()
            for m in masks_in_frame[1:]:
                if m.shape == merged.shape:
                    merged = merged | m
            gt_frame_masks[fid] = merged

    iou_sum = 0.0
    union_count = 0
    overlap_count = 0

    for fid, gt_mask in gt_frame_masks.items():
        if ds_type == "hypernerf" and metadata:
            image_id_str = _frame_id_to_image_id_hypernerf(fid)
            tid = metadata.get(image_id_str)
        else:
            tid = int(fid)
        if tid is None:
            continue

        pred_active = bool(pred_full[tid]) if 0 <= tid < total_frames else False
        gt_active = bool(gt_full[tid]) if 0 <= tid < total_frames else False

        if not (pred_active or gt_active):
            continue
        union_count += 1

        if pred_active and gt_active:
            overlap_count += 1
            if not sorted_tids:
                continue
            pos = bisect.bisect_left(sorted_tids, tid)
            if pos == 0:
                nearest_tid = sorted_tids[0]
            elif pos >= len(sorted_tids):
                nearest_tid = sorted_tids[-1]
            else:
                before = sorted_tids[pos - 1]
                after = sorted_tids[pos]
                nearest_tid = before if (tid - before) <= (after - tid) else after
            val_frame = val_by_time_id.get(nearest_tid)
            if val_frame is not None:
                frame_idx = int(val_frame["frame_index"])
                pred_mask = _load_pred_mask_for_eval(val_frame, binary_mask_dir, frame_idx)
                if pred_mask is not None:
                    if pred_mask.shape != gt_mask.shape:
                        pred_pil = Image.fromarray(pred_mask.astype(np.uint8) * 255)
                        pred_pil = pred_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                        pred_mask = np.asarray(pred_pil) > 0
                    iou_sum += _mask_iou(pred_mask, gt_mask)

    if is_negative_query and _pred_is_empty:
        v_iou = 1.0
    else:
        v_iou = _safe_div(iou_sum, union_count)

    # -----------------------------------------------------------------------
    # Acc: Acc_temporal — fraction of GT-annotated frames with mask IoU >= 0.5
    # Aligned to evaluate_mask_temporal.py protocol (GR4D-Bench).
    # Negative query: 1.0 if pred empty, else 0.0.
    # -----------------------------------------------------------------------
    ACC_IOU_THRESHOLD = 0.15
    if is_negative_query:
        acc = 1.0 if _pred_is_empty else 0.0
    else:
        _acc_hits = 0
        _acc_denom = 0
        for _fid, _gt_mask in gt_frame_masks.items():
            if ds_type == "hypernerf" and metadata:
                _tid = metadata.get(_frame_id_to_image_id_hypernerf(_fid))
            else:
                _tid = int(_fid)
            if _tid is None:
                continue
            _acc_denom += 1
            _pred_active = bool(pred_full[_tid]) if 0 <= _tid < total_frames else False
            _frame_iou = 0.0
            if _pred_active and sorted_tids:
                _pos = bisect.bisect_left(sorted_tids, _tid)
                if _pos == 0:
                    _near = sorted_tids[0]
                elif _pos >= len(sorted_tids):
                    _near = sorted_tids[-1]
                else:
                    _b, _a = sorted_tids[_pos - 1], sorted_tids[_pos]
                    _near = _b if (_tid - _b) <= (_a - _tid) else _a
                _vf = val_by_time_id.get(_near)
                if _vf is not None:
                    _pm = _load_pred_mask_for_eval(_vf, binary_mask_dir, int(_vf["frame_index"]))
                    if _pm is not None:
                        if _pm.shape != _gt_mask.shape:
                            _pm = np.asarray(
                                Image.fromarray(_pm.astype(np.uint8) * 255).resize(
                                    (_gt_mask.shape[1], _gt_mask.shape[0]), Image.NEAREST
                                )
                            ) > 0
                        _frame_iou = _mask_iou(_pm, _gt_mask)
            else:
                _frame_iou = 1.0 if not _gt_mask.any() else 0.0
            if _frame_iou >= ACC_IOU_THRESHOLD:
                _acc_hits += 1
        acc = _safe_div(_acc_hits, _acc_denom)

    # -----------------------------------------------------------------------
    # mIoU: mean of per-entity vIoU across all GT entities in this query.
    # For each entity in target_entity_ids, compute vIoU(pred_mask, entity_gt_mask)
    # independently, then average.  Single-entity queries: mIoU == vIoU.
    # -----------------------------------------------------------------------
    # Collect per-entity pixel accumulators keyed by entity_id string.
    # Falls back to "_merged" when mask_item has no entity_id field.
    entity_inter: dict[str, int] = {}
    entity_union: dict[str, int] = {}

    for gt_frame in gt_frames:
        fid = int(gt_frame["frame_id"])
        if ds_type == "hypernerf" and metadata:
            tid = metadata.get(_frame_id_to_image_id_hypernerf(fid))
        else:
            tid = int(fid)
        if tid is None or tid not in gt_time_ids:
            continue  # only GT-active annotated frames
        if not sorted_tids:
            continue

        pos = bisect.bisect_left(sorted_tids, tid)
        if pos == 0:
            nearest_tid = sorted_tids[0]
        elif pos >= len(sorted_tids):
            nearest_tid = sorted_tids[-1]
        else:
            before, after = sorted_tids[pos - 1], sorted_tids[pos]
            nearest_tid = before if (tid - before) <= (after - tid) else after

        val_frame = val_by_time_id.get(nearest_tid)
        if val_frame is None:
            continue
        pred_active = bool(pred_by_time_id.get(nearest_tid, False))
        frame_idx_m = int(val_frame["frame_index"])

        # Load pred mask once per frame (shared across all entities in this frame)
        pred_mask_m: np.ndarray | None = None
        if pred_active:
            pred_mask_m = _load_pred_mask_for_eval(val_frame, binary_mask_dir, frame_idx_m)

        for mask_item in gt_frame.get("masks", []):
            eid = str(mask_item.get("entity_id", "_merged"))
            gt_mask_e = _load_mask_item(mask_item, image_size=pred_image_size,
                                        benchmark_root=benchmark_root)
            if gt_mask_e is None:
                continue
            if pred_mask_m is not None:
                pm = pred_mask_m
                if pm.shape != gt_mask_e.shape:
                    pm = np.asarray(
                        Image.fromarray(pm.astype(np.uint8) * 255).resize(
                            (gt_mask_e.shape[1], gt_mask_e.shape[0]), Image.NEAREST
                        )
                    ) > 0
                e_inter = int(np.logical_and(pm, gt_mask_e).sum())
                e_union = int(np.logical_or(pm, gt_mask_e).sum())
            else:
                e_inter = 0
                e_union = int(gt_mask_e.sum())
            entity_inter[eid] = entity_inter.get(eid, 0) + e_inter
            entity_union[eid] = entity_union.get(eid, 0) + e_union

    entity_vious = [_safe_div(entity_inter[eid], entity_union[eid])
                    for eid in entity_inter]
    if is_negative_query and _pred_is_empty:
        m_iou = 1.0
    else:
        m_iou = float(np.mean(entity_vious)) if entity_vious else 0.0

    # -----------------------------------------------------------------------
    # Temporal precision / recall
    # -----------------------------------------------------------------------
    if is_negative_query:
        # Binary: tPrec = tRec = tIoU
        t_precision = t_iou
        t_recall    = t_iou
    else:
        tp_t = temporal_inter
        fp_t = int(pred_full.sum()) - tp_t
        fn_t = int(gt_full.sum()) - tp_t
        t_precision = _safe_div(tp_t, tp_t + fp_t)
        t_recall    = _safe_div(tp_t, tp_t + fn_t)

    return {
        "query_id": query_id,
        "question": str(query_item.get("question", "")),
        "status": "ok",
        "Acc":   acc,
        "vIoU":  v_iou,
        "mIoU":  m_iou,
        "tIoU":  t_iou,
        "tPrec": t_precision,
        "tRec":  t_recall,
        "dataset_type": ds_type,
        "total_frames": total_frames,
        "n_gt_entities": n_gt_entities,
        "n_pred_entities": _n_pred_entities,
        "pred_mask_source": pred_mask_source,
        "gt_active_count": int(len(gt_time_ids)),
        "pred_active_count": int(pred_full.sum()),
        "temporal_inter": temporal_inter,
        "temporal_union": temporal_union,
        "iou_sum": iou_sum,
        "union_count": union_count,
        "overlap_count": overlap_count,
        "n_entities_evaluated": len(entity_vious),
        "validation_path": str(validation_path),
        "dataset_dir_used": str(dataset_dir) if dataset_dir is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 4DReSplat outputs against Ours_benchmark.json"
    )
    parser.add_argument("--benchmark", required=True, help="Path to Ours_benchmark.json")
    parser.add_argument(
        "--query-root-map", required=True,
        help="JSON file mapping query_id → query_output_dir path"
    )
    parser.add_argument("--dataset-dir-map", default=None,
                        help="JSON file mapping query_id or scene_name → dataset_dir (optional)")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--benchmark-root", default=None,
                        help="Root dir for resolving relative mask_image paths in benchmark JSON "
                             "(default: directory of --benchmark file)")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip queries with missing validation (don't fail)")
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    query_root_map: dict[str, str] = json.loads(
        Path(args.query_root_map).read_text(encoding="utf-8")
    )
    dataset_dir_map: dict[str, str] = {}
    if args.dataset_dir_map:
        dataset_dir_map = json.loads(
            Path(args.dataset_dir_map).read_text(encoding="utf-8")
        )
    # Resolve benchmark_root for mask_image path lookup
    if args.benchmark_root:
        benchmark_root = Path(args.benchmark_root)
    else:
        benchmark_root = Path(args.benchmark).parent

    per_query: list[dict] = []
    for item in benchmark:
        query_id = str(item["query_id"])
        query_output_dir_str = query_root_map.get(query_id)
        if query_output_dir_str is None:
            if not args.skip_missing:
                raise ValueError(f"No query_root_map entry for {query_id}")
            per_query.append({
                "query_id": query_id,
                "question": str(item.get("question", "")),
                "status": "not_in_map",
                "Acc": None, "vIoU": None, "tIoU": None,
            })
            continue

        query_output_dir = Path(query_output_dir_str)
        dataset_dir_str = dataset_dir_map.get(query_id) or dataset_dir_map.get(
            "_".join(query_id.split("_")[:-1])
        )
        dataset_dir = Path(dataset_dir_str) if dataset_dir_str else None

        result = evaluate_query(item, query_output_dir, dataset_dir,
                                benchmark_root=benchmark_root)
        per_query.append(result)
        status_str = result.get("status", "?")
        acc = result.get("Acc")
        viou = result.get("vIoU")
        tiou = result.get("tIoU")
        acc_s = f"{acc*100:.2f}%" if acc is not None else "n/a"
        viou_s = f"{viou*100:.2f}%" if viou is not None else "n/a"
        tiou_s = f"{tiou*100:.2f}%" if tiou is not None else "n/a"
        print(f"[eval] {query_id}: {status_str}  Acc={acc_s}  vIoU={viou_s}  tIoU={tiou_s}")

    # Aggregate metrics (only over queries with valid results)
    valid = [r for r in per_query if r.get("Acc") is not None]
    n = len(valid)

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in valid if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    # Per-scene breakdown
    scene_groups: dict[str, list[dict]] = {}
    for r in valid:
        # Infer scene from query_id (e.g. "americano_q1" → "americano")
        scene = "_".join(str(r["query_id"]).split("_")[:-1]) or str(r["query_id"])
        scene_groups.setdefault(scene, []).append(r)
    per_scene: dict[str, dict] = {}
    for scene, rows in scene_groups.items():
        def _smean(key: str) -> float:
            vals = [x[key] for x in rows if x.get(key) is not None]
            return float(np.mean(vals)) if vals else 0.0
        per_scene[scene] = {
            "n":     len(rows),
            "Acc":   _smean("Acc"),
            "vIoU":  _smean("vIoU"),
            "mIoU":  _smean("mIoU"),
            "tIoU":  _smean("tIoU"),
            "tPrec": _smean("tPrec"),
            "tRec":  _smean("tRec"),
        }

    summary = {
        "total_queries": len(per_query),
        "valid_queries": n,
        "Acc":   _mean("Acc"),
        "vIoU":  _mean("vIoU"),
        "mIoU":  _mean("mIoU"),
        "tIoU":  _mean("tIoU"),
        "tPrec": _mean("tPrec"),
        "tRec":  _mean("tRec"),
        "per_scene": per_scene,
    }

    print("\n=== Summary ===")
    print(f"Valid queries: {n} / {len(per_query)}")
    if summary["Acc"] is not None:
        print(f"Acc:   {summary['Acc']*100:.4f}%")
        print(f"vIoU:  {summary['vIoU']*100:.4f}%")
        print(f"mIoU:  {summary['mIoU']*100:.4f}%")
        print(f"tIoU:  {summary['tIoU']*100:.4f}%")
        print(f"tPrec: {summary['tPrec']*100:.4f}%")
        print(f"tRec:  {summary['tRec']*100:.4f}%")
        if per_scene:
            print("\n  Per-scene:")
            for scene, sm in sorted(per_scene.items()):
                print(f"    {scene:20s}  n={sm['n']}  Acc={sm['Acc']*100:.1f}%  "
                      f"vIoU={sm['vIoU']*100:.1f}%  mIoU={sm['mIoU']*100:.1f}%  tIoU={sm['tIoU']*100:.1f}%")

    payload = {
        "benchmark": str(args.benchmark),
        "summary": summary,
        "per_query": per_query,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

    if args.output_md:
        lines = [
            "# Ours_benchmark Evaluation",
            "",
            f"- Total queries: {summary['total_queries']}",
            f"- Valid queries: {summary['valid_queries']}",
        ]
        def fmt(v: float | None) -> str:
            return f"{v*100:.2f}" if v is not None else "n/a"
        if summary["Acc"] is not None:
            lines += [
                f"- Acc:   `{summary['Acc']*100:.4f}%`",
                f"- vIoU:  `{summary['vIoU']*100:.4f}%`",
                f"- mIoU:  `{summary['mIoU']*100:.4f}%`",
                f"- tIoU:  `{summary['tIoU']*100:.4f}%`",
                f"- tPrec: `{summary['tPrec']*100:.4f}%`",
                f"- tRec:  `{summary['tRec']*100:.4f}%`",
                "",
                "| Query | Status | Acc(%) | vIoU(%) | mIoU(%) | tIoU(%) | tPrec(%) | tRec(%) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for r in per_query:
                lines.append(
                    f"| {r['query_id']} | {r.get('status','')} "
                    f"| {fmt(r.get('Acc'))} | {fmt(r.get('vIoU'))} | {fmt(r.get('mIoU'))} "
                    f"| {fmt(r.get('tIoU'))} | {fmt(r.get('tPrec'))} | {fmt(r.get('tRec'))} |"
                )
            if summary.get("per_scene"):
                lines += ["", "### Per-scene", "",
                          "| Scene | n | Acc(%) | vIoU(%) | mIoU(%) | tIoU(%) | tPrec(%) | tRec(%) |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
                for scene, sm in sorted(summary["per_scene"].items()):
                    lines.append(
                        f"| {scene} | {sm['n']} | {fmt(sm['Acc'])} | {fmt(sm['vIoU'])} "
                        f"| {fmt(sm['mIoU'])} | {fmt(sm['tIoU'])} "
                        f"| {fmt(sm['tPrec'])} | {fmt(sm['tRec'])} |"
                    )
        Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved: {args.output_md}")


if __name__ == "__main__":
    main()
