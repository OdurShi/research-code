from __future__ import annotations

import json
import os
import re
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GSAM2_ROOT = _PROJECT_ROOT / "external" / "Grounded-SAM-2"
_SAM3_CHECKPOINT_FALLBACK = _PROJECT_ROOT.parent / "hyperGS" / "sam3.1" / "sam3.1_multiplex.pt"
for _candidate in (_PROJECT_ROOT, _GSAM2_ROOT, _PROJECT_ROOT.parent / "hyperGS"):
    candidate_str = str(_candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from .source_images import resolve_dataset_image_entries
from .mask_store import save_mask_store_npz


def _import_sam2_predictors() -> tuple[Any, Any, Any]:
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore
        from utils.track_utils import sample_points_from_masks  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Grounded-SAM-2 dependencies are unavailable. Install the original backend, or switch to SAM3.1."
        ) from exc
    return SAM2ImagePredictor, SAM2VideoPredictor, sample_points_from_masks


def _resolve_sam3_checkpoint(explicit_value: str | None) -> Path:
    env_override = os.environ.get("D4RESPLAT_SAM3_CHECKPOINT")
    candidates = [
        Path(explicit_value) if explicit_value else None,
        Path(env_override) if env_override else None,
        _PROJECT_ROOT / "models" / "sam3.1" / "sam3.1_multiplex.pt",
        _SAM3_CHECKPOINT_FALLBACK,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return candidates[0] if candidates[0] is not None else _SAM3_CHECKPOINT_FALLBACK


def _should_use_sam3(model_id_or_path: str) -> bool:
    backend = os.environ.get("D4RESPLAT_SAM_BACKEND", "").strip().lower()
    if backend == "sam3":
        return True
    text = str(model_id_or_path).strip().lower()
    return text.endswith(".pt") or "sam3" in text


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _load_sam3_predictor(checkpoint_path: Path) -> Any:
    try:
        from sam3 import build_sam3_predictor  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependency 'sam3'. Install it with: pip install git+https://github.com/facebookresearch/sam3.git"
        ) from exc
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3.1 checkpoint not found: {checkpoint_path}")
    use_fa3 = _env_flag("D4RESPLAT_SAM3_USE_FA3", True)
    if use_fa3:
        try:
            import flash_attn_interface  # type: ignore  # noqa: F401
        except Exception:
            print("[sam3] flash_attn_interface unavailable; disabling use_fa3")
            use_fa3 = False
    return build_sam3_predictor(
        checkpoint_path=str(checkpoint_path),
        version="sam3.1",
        compile=_env_flag("D4RESPLAT_SAM3_COMPILE", False),
        use_fa3=use_fa3,
        async_loading_frames=_env_flag("D4RESPLAT_SAM3_ASYNC_LOADING", True),
    )


def _bbox_mask_from_xyxy(image_size: tuple[int, int], box_xyxy: list[float]) -> np.ndarray:
    width, height = image_size
    left, top, right, bottom = [int(round(float(value))) for value in box_xyxy]
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _materialize_jpeg_frame_dir(image_entries: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in image_entries:
        source_path = Path(entry["image_path"])
        target_path = output_dir / f"{int(entry['frame_index']):05d}.jpg"
        if target_path.exists() or target_path.is_symlink():
            continue
        os.symlink(source_path, target_path)
    return output_dir


def _materialize_local_jpeg_frame_dir(
    image_entries: list[dict[str, Any]],
    output_dir: Path,
    start_frame_index: int,
    end_frame_index: int,
) -> tuple[Path, list[dict[str, Any]]]:
    local_entries = [
        entry
        for entry in image_entries
        if int(start_frame_index) <= int(entry["frame_index"]) <= int(end_frame_index)
    ]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for local_index, entry in enumerate(local_entries):
        source_path = Path(entry["image_path"])
        target_path = output_dir / f"{int(local_index):05d}.jpg"
        os.symlink(source_path, target_path)
    return output_dir, local_entries


def _sample_search_entries(image_entries: list[dict[str, Any]], frame_stride: int, max_frames: int) -> list[dict[str, Any]]:
    sampled = image_entries[:: max(int(frame_stride), 1)]
    if max_frames > 0 and len(sampled) > int(max_frames):
        indices = np.linspace(0, len(sampled) - 1, num=int(max_frames), dtype=np.int32)
        sampled = [sampled[int(index)] for index in indices.tolist()]
    return sampled


def _normalize_query_text(text: str) -> str:
    normalized = " ".join(str(text).strip().lower().split())
    if not normalized:
        raise ValueError("detector phrase must be non-empty")
    if not normalized.endswith("."):
        normalized = f"{normalized}."
    return normalized


def _draw_box_preview(image: Image.Image, box: list[float], label: str, score: float) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    left, top, right, bottom = [float(v) for v in box]
    top = max(0.0, top)  # clamp top to canvas boundary
    color = (255, 96, 32)
    draw.rectangle((left, top, right, bottom), fill=color + (56,), outline=color + (255,), width=5)
    caption = f"{label} {score:.2f}"
    label_y0 = max(0.0, top - 26.0)
    label_y1 = max(label_y0 + 1.0, top)  # ensure y1 > y0
    draw.rectangle((left, label_y0, min(canvas.width - 1.0, left + 220.0), label_y1), fill=(18, 18, 18, 220))
    draw.text((left + 4.0, max(0.0, top - 20.0)), caption, fill=(240, 240, 240, 255))
    return canvas


def _mask_component_bboxes(mask: np.ndarray, min_area: int = 128, min_area_ratio: float = 0.12) -> list[list[int]]:
    try:
        from scipy import ndimage  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for GSAM2 mask component analysis.") from exc
    binary = np.asarray(mask > 0, dtype=np.uint8)
    if binary.max() <= 0:
        return []
    labels, num = ndimage.label(binary)
    min_keep_area = max(int(min_area), int(binary.sum() * float(min_area_ratio)))
    boxes: list[list[int]] = []
    for component_id in range(1, int(num) + 1):
        ys, xs = np.where(labels == component_id)
        if ys.size == 0 or xs.size == 0:
            continue
        if ys.size < min_keep_area:
            continue
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return boxes


def _draw_mask_preview(image: Image.Image, mask: np.ndarray, label: str, score: float) -> Image.Image:
    canvas = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    binary = np.asarray(mask > 0, dtype=bool)
    color = np.asarray([255, 96, 32], dtype=np.uint8)
    if binary.any():
        canvas[binary] = (0.55 * canvas[binary] + 0.45 * color).astype(np.uint8)
    preview = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(preview, "RGBA")
    for left, top, right, bottom in _mask_component_bboxes(mask, min_area=64, min_area_ratio=0.05):
        draw.rectangle((left, top, right, bottom), outline=(255, 96, 32, 255), width=4)
    bbox = _mask_bbox(mask)
    if bbox is not None:
        left, top, right, bottom = bbox
        draw.rectangle((left, top, right, bottom), outline=(255, 224, 96, 255), width=2)
        draw.rectangle((left, max(0, top - 26), min(preview.width - 1, left + 280), top), fill=(18, 18, 18, 220))
        draw.text((left + 4, max(0, top - 20)), f"{label} {score:.2f}", fill=(240, 240, 240, 255))
    return preview


def _stable_split_frames(frame_rows: list[dict[str, Any]], min_components: int = 2, min_stable: int = 2) -> list[int]:
    active_rows = [
        row
        for row in sorted(frame_rows, key=lambda item: int(item["frame_index"]))
        if int(row.get("component_count", 0)) >= int(min_components)
    ]
    if not active_rows:
        return []
    stable: list[int] = []
    run: list[int] = []
    prev = None
    for row in active_rows:
        frame_index = int(row["frame_index"])
        if prev is None or frame_index == prev + 1:
            run.append(frame_index)
        else:
            if len(run) >= int(min_stable):
                stable.append(run[0])
            run = [frame_index]
        prev = frame_index
    if len(run) >= int(min_stable):
        stable.append(run[0])
    return stable


def _run_phrase_detection(
    phrase: str,
    sampled_entries: list[dict[str, Any]],
    processor: Any,
    grounding_model: Any,
    device: str,
    box_threshold: float,
    text_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    query_text = _normalize_query_text(phrase)
    detections: list[dict[str, Any]] = []
    for entry in sampled_entries:
        image = Image.open(entry["image_path"]).convert("RGB")
        inputs = processor(images=image, text=query_text, return_tensors="pt")
        moved_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = grounding_model(**moved_inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            moved_inputs["input_ids"],
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[image.size[::-1]],
        )[0]
        for detection_index, (box_tensor, score_tensor, label) in enumerate(
            zip(results.get("boxes", []), results.get("scores", []), results.get("labels", []))
        ):
            if detection_index >= int(top_k):
                break
            box = [float(value) for value in box_tensor.detach().cpu().tolist()]
            detections.append(
                {
                    "phrase": phrase,
                    "query_text": query_text,
                    "frame_index": int(entry["frame_index"]),
                    "image_id": str(entry["image_id"]),
                    "image_path": str(entry["image_path"]),
                    "time_value": float(entry["time_value"]),
                    "label": str(label),
                    "score": float(score_tensor.detach().cpu().item()),
                    "bbox_xyxy": box,
                }
            )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def _bbox_center(box: list[float]) -> np.ndarray:
    left, top, right, bottom = [float(value) for value in box]
    return np.asarray([0.5 * (left + right), 0.5 * (top + bottom)], dtype=np.float32)


def _bbox_diag(box: list[float]) -> float:
    left, top, right, bottom = [float(value) for value in box]
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    return float(np.hypot(width, height))


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
    bx0, by0, bx1, by1 = [float(value) for value in box_b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(inter_x1 - inter_x0, 0.0)
    inter_h = max(inter_y1 - inter_y0, 0.0)
    inter_area = inter_w * inter_h
    area_a = max(ax1 - ax0, 1.0) * max(ay1 - ay0, 1.0)
    area_b = max(bx1 - bx0, 1.0) * max(by1 - by0, 1.0)
    union = max(area_a + area_b - inter_area, 1.0e-6)
    return float(inter_area / union)


def _bbox_proximity(box_a: list[float], box_b: list[float]) -> float:
    center_a = _bbox_center(box_a)
    center_b = _bbox_center(box_b)
    distance = float(np.linalg.norm(center_a - center_b))
    scale = max(0.5 * (_bbox_diag(box_a) + _bbox_diag(box_b)), 1.0)
    return float(np.exp(-0.5 * (distance / scale) ** 2))


def _combo_box_cohesion(combo: tuple[dict[str, Any], ...]) -> float:
    if len(combo) <= 1:
        return 0.0
    values: list[float] = []
    for first_index in range(len(combo)):
        for second_index in range(first_index + 1, len(combo)):
            box_a = combo[first_index]["bbox_xyxy"]
            box_b = combo[second_index]["bbox_xyxy"]
            iou = _bbox_iou(box_a, box_b)
            proximity = _bbox_proximity(box_a, box_b)
            values.append(0.45 * iou + 0.55 * proximity)
    return float(np.mean(values)) if values else 0.0


def _select_anchor_detections(
    detections_by_phrase: dict[str, list[dict[str, Any]]],
    must_track_phrases: list[str],
    total_frames: int,
    top_n: int = 3,
    successor_phrases: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    required = [phrase for phrase in must_track_phrases if detections_by_phrase.get(phrase)]
    successor_set = {str(item).strip() for item in successor_phrases or [] if str(item).strip()}
    if not required:
        required = [phrase for phrase, detections in detections_by_phrase.items() if detections]
    if len(required) <= 1:
        for phrase, detections in detections_by_phrase.items():
            if detections:
                selected[phrase] = detections[0]
        return selected

    frame_norm = max(int(total_frames) - 1, 1)
    best_combo: tuple[float, tuple[dict[str, Any], ...]] | None = None

    best_per_frame: dict[str, dict[int, dict[str, Any]]] = {}
    for phrase in required:
        frame_rows: dict[int, dict[str, Any]] = {}
        for detection in detections_by_phrase[phrase]:
            frame_index = int(detection["frame_index"])
            if frame_index not in frame_rows or float(detection["score"]) > float(frame_rows[frame_index]["score"]):
                frame_rows[frame_index] = detection
        best_per_frame[phrase] = frame_rows

    common_frames = sorted(set.intersection(*(set(rows.keys()) for rows in best_per_frame.values()))) if best_per_frame else []
    for frame_index in common_frames:
        combo = tuple(best_per_frame[phrase][frame_index] for phrase in required)
        mean_score = float(np.mean([float(item["score"]) for item in combo]))
        cohesion = _combo_box_cohesion(combo)
        combo_score = mean_score + 0.38 * cohesion
        if best_combo is None or combo_score > best_combo[0]:
            best_combo = (combo_score, combo)

    if best_combo is None:
        candidate_lists = [detections_by_phrase[phrase][: max(1, int(top_n))] for phrase in required]
        for combo in product(*candidate_lists):
            frames = [int(item["frame_index"]) for item in combo]
            scores = [float(item["score"]) for item in combo]
            frame_span = (max(frames) - min(frames)) / frame_norm
            cohesion = _combo_box_cohesion(combo)
            combo_score = float(np.mean(scores) - 0.35 * frame_span + 0.30 * cohesion)
            if best_combo is None or combo_score > best_combo[0]:
                best_combo = (combo_score, combo)

    if best_combo is not None:
        _, combo = best_combo
        for phrase, detection in zip(required, combo):
            selected[phrase] = detection

    if selected:
        anchor_center = float(np.mean([int(item["frame_index"]) for item in selected.values()]))
    else:
        anchor_center = 0.0
    for phrase, detections in detections_by_phrase.items():
        if not detections:
            continue
        if phrase in selected:
            continue
        candidate_rows = detections[: max(1, int(top_n) * 4)]
        if phrase in successor_set:
            later_rows = [item for item in candidate_rows if float(item["frame_index"]) >= anchor_center]
            if later_rows:
                candidate_rows = later_rows
        selected[phrase] = max(
            candidate_rows,
            key=lambda item: float(item["score"])
            - 0.18 * abs(float(item["frame_index"]) - anchor_center) / frame_norm
            + (0.08 if phrase in successor_set and float(item["frame_index"]) >= anchor_center else 0.0),
        )
    return selected


def _select_multi_anchor_detections(
    detections_by_phrase: dict[str, list[dict[str, Any]]],
    detector_phrases: list[str],
    query_plan: dict[str, Any],
    total_frames: int,
    max_anchors_per_phrase: int = 3,
    top_n_per_anchor: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    context_frames = sorted(
        int(frame.get("frame_index", 0))
        for frame in query_plan.get("context_frames", [])
        if isinstance(frame, dict)
    )
    if not context_frames:
        context_frames = [0, max(int(total_frames) // 2, 0), max(int(total_frames) - 1, 0)]
    frame_norm = max(int(total_frames) - 1, 1)
    anchor_map: dict[str, list[dict[str, Any]]] = {}
    for phrase in detector_phrases:
        detections = detections_by_phrase.get(phrase, [])
        if not detections:
            anchor_map[phrase] = []
            continue
        chosen: list[dict[str, Any]] = []
        used_frames: list[int] = []
        target_count = max(1, int(max_anchors_per_phrase))
        if len(context_frames) <= target_count:
            target_frames = list(context_frames)
        else:
            target_indices = np.linspace(0, len(context_frames) - 1, num=target_count, dtype=np.int32)
            target_frames = [context_frames[int(index)] for index in target_indices.tolist()]
        min_sep = max(8, int(total_frames // max(6, int(max_anchors_per_phrase) * 2)))
        best_per_frame: dict[int, dict[str, Any]] = {}
        for row in detections:
            frame_index = int(row["frame_index"])
            current = best_per_frame.get(frame_index)
            if current is None or float(row["score"]) > float(current["score"]):
                best_per_frame[frame_index] = row
        per_frame_rows = list(best_per_frame.values())
        for target_frame in target_frames:
            candidates = per_frame_rows or detections
            ranked = sorted(
                candidates,
                key=lambda item: (
                    abs(int(item["frame_index"]) - int(target_frame)) / frame_norm
                    - 0.55 * float(item["score"])
                ),
            )
            picked = None
            for row in ranked:
                frame_index = int(row["frame_index"])
                if any(abs(frame_index - used) < min_sep for used in used_frames):
                    continue
                picked = row
                break
            if picked is None:
                continue
            chosen.append(picked)
            used_frames.append(int(picked["frame_index"]))
            if len(chosen) >= int(max_anchors_per_phrase):
                break
        if not chosen:
            chosen = [detections[0]]
        anchor_map[phrase] = sorted(chosen, key=lambda item: int(item["frame_index"]))
    return anchor_map


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _extract_detector_phrases(query_plan: dict[str, Any]) -> list[str]:
    """Extract detector phrases, preferring the new `subjects` list from Qwen batch planner."""
    subjects = query_plan.get("subjects")
    if isinstance(subjects, list) and subjects:
        return [str(s).strip() for s in subjects if str(s).strip()]
    for key in ("detector_phrases", "query_subject_phrases"):
        phrases = [str(p).strip() for p in query_plan.get(key, []) if str(p).strip()]
        if phrases:
            return phrases
    subject = str(query_plan.get("subject", "")).strip()
    if subject:
        return [subject]
    return []


def load_grounded_sam2_models(
    grounding_model_id: str = "IDEA-Research/grounding-dino-base",
    sam2_model_id: str = "facebook/sam2-hiera-large",
) -> dict[str, Any]:
 
    import time as _time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = _time.time()
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
    grounding_model.eval()
    print(f"[sam_batch] GroundingDINO loaded in {_time.time()-t0:.1f}s")

    use_sam3 = _should_use_sam3(sam2_model_id)
    image_predictor = video_predictor = sample_points_fn = sam3_predictor = None
    resolved_sam_model = sam2_model_id
    t1 = _time.time()
    if use_sam3:
        resolved_checkpoint = _resolve_sam3_checkpoint(sam2_model_id)
        sam3_predictor = _load_sam3_predictor(resolved_checkpoint)
        resolved_sam_model = str(resolved_checkpoint)
    else:
        SAM2ImagePredictor, SAM2VideoPredictor, sample_points_fn = _import_sam2_predictors()
        image_predictor = SAM2ImagePredictor.from_pretrained(sam2_model_id)
        video_predictor = SAM2VideoPredictor.from_pretrained(sam2_model_id)
    print(f"[sam_batch] SAM model loaded in {_time.time()-t1:.1f}s  (use_sam3={use_sam3})")

    return {
        "processor": processor,
        "grounding_model": grounding_model,
        "device": device,
        "use_sam3": use_sam3,
        "sam3_predictor": sam3_predictor,
        "image_predictor": image_predictor,
        "video_predictor": video_predictor,
        "sample_points_from_masks": sample_points_fn,
        "resolved_sam_model": resolved_sam_model,
    }


def run_grounded_sam2_query(
    dataset_dir: str | Path,
    query_plan_path: str | Path,
    output_dir: str | Path,
    grounding_model_id: str = "IDEA-Research/grounding-dino-base",
    sam2_model_id: str = "facebook/sam2-hiera-large",
    detector_frame_stride: int = 12,
    max_detector_frames: int = 12,
    detection_top_k: int = 3,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    prompt_type: str = "point",
    num_point_prompts: int = 16,
    track_window_radius: int = 120,
    frame_subsample_stride: int = 10,
    num_anchor_seeds: int = 3,
    _preloaded_models: dict[str, Any] | None = None,
) -> Path:
    dataset_dir = Path(dataset_dir)
    query_plan_path = Path(query_plan_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phrase_dir = output_dir / "phrases"
    phrase_dir.mkdir(parents=True, exist_ok=True)

    query_plan = _read_json(query_plan_path)
    detector_phrases = _extract_detector_phrases(query_plan)
    if not detector_phrases:
        raise ValueError(f"No detector_phrases found in {query_plan_path}")
    must_track_phrases = [str(item).strip() for item in query_plan.get("must_track_phrases", []) if str(item).strip()]
    successor_phrases = [str(item).strip() for item in query_plan.get("query_successor_phrases", []) if str(item).strip()]

    phrase_entities = {
        str(e["phrase"]).strip(): e
        for e in query_plan.get("phrase_entities", [])
        if isinstance(e, dict) and str(e.get("phrase", "")).strip()
    }
    def _detector_hint(phrase: str) -> str:
        entity = phrase_entities.get(phrase)
        if entity:
            hint = str(entity.get("detector_hint") or "").strip()
            if hint:
                return hint
            rel = entity.get("relation")
            if isinstance(rel, dict) and rel.get("container"):
                return str(rel["container"]).strip()
        return phrase

    image_entries = resolve_dataset_image_entries(dataset_dir)
    image_entries = image_entries[:: max(int(frame_subsample_stride), 1)]
    if not image_entries:
        raise ValueError(f"No image entries available after frame subsampling under {dataset_dir}")
    sampled_entries = _sample_search_entries(
        image_entries,
        frame_stride=detector_frame_stride,
        max_frames=max_detector_frames,
    )

    if _preloaded_models is not None:
        models = _preloaded_models
    else:
        models = load_grounded_sam2_models(grounding_model_id, sam2_model_id)

    device = models["device"]
    use_sam3 = models["use_sam3"]
    processor = models["processor"]
    grounding_model = models["grounding_model"]
    sam3_predictor = models["sam3_predictor"]
    image_predictor = models["image_predictor"]
    video_predictor = models["video_predictor"]
    sample_points_from_masks = models["sample_points_from_masks"]
    resolved_sam_model = models["resolved_sam_model"]

    detections_by_phrase: dict[str, list[dict[str, Any]]] = {}
    for phrase in detector_phrases:
        hint = _detector_hint(phrase)
        raw_detections = _run_phrase_detection(
            phrase=hint,
            sampled_entries=sampled_entries,
            processor=processor,
            grounding_model=grounding_model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            top_k=detection_top_k,
        )
        # Remap detections back to original phrase so downstream logic stays consistent
        for det in raw_detections:
            det["label"] = phrase
        detections_by_phrase[phrase] = raw_detections
    selected_anchor_by_phrase = _select_anchor_detections(
        detections_by_phrase=detections_by_phrase,
        must_track_phrases=must_track_phrases,
        total_frames=len(image_entries),
        top_n=min(4, max(1, int(detection_top_k))),
        successor_phrases=successor_phrases,
    )
    multi_anchor_by_phrase = _select_multi_anchor_detections(
        detections_by_phrase=detections_by_phrase,
        detector_phrases=detector_phrases,
        query_plan=query_plan,
        total_frames=len(image_entries),
        max_anchors_per_phrase=max(1, int(num_anchor_seeds)),
        top_n_per_anchor=max(2, int(detection_top_k)),
    )
    max_frame_index = max(int(entry["frame_index"]) for entry in image_entries)

    phrase_payloads: list[dict[str, Any]] = []
    phrase_tracks: list[dict[str, Any]] = []
    for phrase_index, phrase in enumerate(detector_phrases):
        detections = detections_by_phrase.get(phrase, [])
        phrase_output_dir = phrase_dir / f"{phrase_index:02d}_{phrase.replace(' ', '_')}"
        phrase_output_dir.mkdir(parents=True, exist_ok=True)
        if not detections:
            phrase_payloads.append(
                {
                    "phrase": phrase,
                    "object_id": int(phrase_index + 1),
                    "detections": [],
                    "status": "no_detection",
                }
            )
            phrase_tracks.append(
                {
                    "phrase": phrase,
                    "object_id": int(phrase_index + 1),
                    "status": "no_detection",
                    "anchor_frame_index": None,
                    "frames": [
                        {
                            "frame_index": int(entry["frame_index"]),
                            "image_id": str(entry["image_id"]),
                            "time_value": float(entry["time_value"]),
                            "active": False,
                            "bbox_xyxy": None,
                        }
                        for entry in image_entries
                    ],
                }
            )
            continue

        selected_anchors = multi_anchor_by_phrase.get(phrase) or [selected_anchor_by_phrase.get(phrase, detections[0])]
        best = selected_anchor_by_phrase.get(phrase, selected_anchors[0])
        object_id = int(phrase_index + 1)
        effective_track_window_radius = max(
            int(track_window_radius),
            int(np.ceil(float(len(image_entries)) / max(len(selected_anchors), 1))),
        )
        anchors_dir = phrase_output_dir / "anchors"
        anchors_dir.mkdir(parents=True, exist_ok=True)
        combined_segments: dict[int, np.ndarray] = {}
        anchor_rows: list[dict[str, Any]] = []
        first_mask_bbox = None
        first_preview_path = None
        first_anchor_mask_preview_path = None
        for anchor_index, anchor_detection in enumerate(selected_anchors):
            anchor_frame_index = int(anchor_detection["frame_index"])
            anchor_image = Image.open(anchor_detection["image_path"]).convert("RGB")
            preview = _draw_box_preview(anchor_image, anchor_detection["bbox_xyxy"], phrase, anchor_detection["score"])
            preview_path = anchors_dir / f"anchor_{anchor_index:02d}_detection.png"
            preview.save(preview_path)

            if use_sam3:
                mask = _bbox_mask_from_xyxy(anchor_image.size, anchor_detection["bbox_xyxy"])
                masks = np.expand_dims(mask, axis=0)
            else:
                assert image_predictor is not None
                image_predictor.set_image(np.array(anchor_image.convert("RGB")))
                box = np.asarray([anchor_detection["bbox_xyxy"]], dtype=np.float32)
                masks, _, _ = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=False,
                )
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                mask = masks[0].astype(np.uint8)
            mask_bbox = _mask_bbox(mask)
            anchor_mask_preview_path = anchors_dir / f"anchor_{anchor_index:02d}_mask.png"
            if first_mask_bbox is None:
                first_mask_bbox = mask_bbox
                first_preview_path = preview_path

            track_start = max(0, anchor_frame_index - int(effective_track_window_radius))
            track_end = min(max_frame_index, anchor_frame_index + int(effective_track_window_radius))
            local_frame_dir, local_entries = _materialize_local_jpeg_frame_dir(
                image_entries=image_entries,
                output_dir=phrase_output_dir / f"video_window_jpg_{anchor_index:02d}",
                start_frame_index=track_start,
                end_frame_index=track_end,
            )
            local_anchor_candidates = [
                index
                for index, entry in enumerate(local_entries)
                if int(entry["frame_index"]) == int(anchor_frame_index)
            ]
            if not local_anchor_candidates:
                raise ValueError(f"Unable to find anchor frame {anchor_frame_index} inside the local window for phrase '{phrase}'.")
            local_anchor_index = int(local_anchor_candidates[0])
            local_video_segments: dict[int, np.ndarray] = {}
            if use_sam3:
                assert sam3_predictor is not None
                session_id = f"d4resplat_{uuid.uuid4().hex[:8]}"
                sam3_predictor.start_session(resource_path=str(local_frame_dir), session_id=session_id)
                try:
                    with torch.inference_mode():
                        add_resp = sam3_predictor.add_prompt(
                            session_id=session_id,
                            frame_idx=local_anchor_index,
                            text=_detector_hint(phrase),
                            obj_id=object_id,
                        )
                        # Check if add_prompt actually detected something before propagating.
                        _resp_out = (add_resp or {}).get("outputs") or {}
                        _resp_masks = _resp_out.get("out_binary_masks")
                        _detected = (
                            _resp_masks is not None
                            and hasattr(_resp_masks, "__len__")
                            and len(_resp_masks) > 0
                            and int(_resp_masks[0].sum()) >= 10
                        )
                        if not _detected:
                            print(
                                f"[sam3] add_prompt detected nothing for phrase={phrase!r}"
                                f" at anchor frame {local_anchor_index}; skipping propagation."
                            )
                        for _direction in (("forward", "backward") if _detected else ()):
                            _propagate_iter = sam3_predictor.propagate_in_video(
                                session_id=session_id,
                                propagation_direction=_direction,
                                output_prob_thresh=0.1,
                            )
                            for result in _propagate_iter:
                                out_frame_idx = int(result["frame_index"])
                                outputs = result["outputs"]
                                out_obj_ids = outputs["out_obj_ids"]
                                binary_masks = outputs["out_binary_masks"]
                                for out_obj_id, raw_mask in zip(out_obj_ids, binary_masks):
                                    # SAM3 multiplex uses 0-based internal IDs (obj_id param is silently dropped);
                                    # accept all returned masks and union them into a single segment per frame.
                                    mask_array = raw_mask.cpu().numpy() if hasattr(raw_mask, "cpu") else np.asarray(raw_mask)
                                    m = mask_array.astype(np.uint8)
                                    if out_frame_idx in local_video_segments:
                                        local_video_segments[out_frame_idx] = np.maximum(local_video_segments[out_frame_idx], m)
                                    else:
                                        local_video_segments[out_frame_idx] = m
                finally:
                    sam3_predictor.close_session(session_id=session_id)
            else:
                assert video_predictor is not None
                assert sample_points_from_masks is not None
                inference_state = video_predictor.init_state(
                    video_path=str(local_frame_dir),
                    offload_video_to_cpu=True,
                    async_loading_frames=True,
                )
                if prompt_type == "mask":
                    video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=local_anchor_index,
                        obj_id=object_id,
                        mask=mask,
                    )
                elif prompt_type == "box":
                    video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=local_anchor_index,
                        obj_id=object_id,
                        box=np.asarray(anchor_detection["bbox_xyxy"], dtype=np.float32),
                    )
                else:
                    points = sample_points_from_masks(masks=masks.astype(np.uint8), num_points=int(num_point_prompts))[0]
                    labels = np.ones((points.shape[0],), dtype=np.int32)
                    video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=local_anchor_index,
                        obj_id=object_id,
                        points=points,
                        labels=labels,
                    )

                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                    for mask_index, out_obj_id in enumerate(out_obj_ids):
                        if int(out_obj_id) != object_id:
                            continue
                        local_video_segments[int(out_frame_idx)] = (
                            out_mask_logits[mask_index] > 0.0
                        ).detach().cpu().numpy().astype(np.uint8)
            for local_frame_index, mask_array in local_video_segments.items():
                if local_frame_index < 0 or local_frame_index >= len(local_entries):
                    continue
                global_frame_index = int(local_entries[local_frame_index]["frame_index"])
                flat_mask = np.asarray(mask_array[0] if mask_array.ndim == 3 else mask_array, dtype=np.uint8)
                if global_frame_index in combined_segments:
                    combined_segments[global_frame_index] = np.maximum(combined_segments[global_frame_index], flat_mask)
                else:
                    combined_segments[global_frame_index] = flat_mask
            preview_mask = combined_segments.get(anchor_frame_index, mask)
            anchor_mask_preview = _draw_mask_preview(anchor_image, preview_mask, phrase, anchor_detection["score"])
            anchor_mask_preview.save(anchor_mask_preview_path)
            if first_anchor_mask_preview_path is None:
                first_anchor_mask_preview_path = anchor_mask_preview_path
            anchor_rows.append(
                {
                    "anchor_index": int(anchor_index),
                    "anchor_frame_index": int(anchor_frame_index),
                    "anchor_image_id": str(anchor_detection["image_id"]),
                    "anchor_time_value": float(anchor_detection["time_value"]),
                    "anchor_bbox_xyxy": anchor_detection["bbox_xyxy"],
                    "anchor_score": float(anchor_detection["score"]),
                    "anchor_label": str(anchor_detection["label"]),
                    "anchor_preview_path": str(preview_path),
                    "anchor_mask_preview_path": str(anchor_mask_preview_path),
                    "anchor_mask_bbox_xyxy": mask_bbox,
                "track_window": {
                        "start_frame_index": int(track_start),
                        "end_frame_index": int(track_end),
                    },
                }
            )

        # Fallback: if SAM3/SAM2 propagation produced no masks at all, seed combined_segments
        # from each anchor's detection bbox so the phrase retains at least its anchor frame.
        if not combined_segments and anchor_rows:
            for anchor_row in anchor_rows:
                aframe = int(anchor_row["anchor_frame_index"])
                anchor_entry = next((e for e in image_entries if int(e["frame_index"]) == aframe), None)
                if anchor_entry is None:
                    continue
                anchor_img = Image.open(anchor_entry["image_path"]).convert("RGB")
                combined_segments[aframe] = _bbox_mask_from_xyxy(anchor_img.size, anchor_row["anchor_bbox_xyxy"])

        bbox_by_global_frame: dict[int, list[int] | None] = {}
        track_mask_dir = phrase_output_dir / "track_masks"
        track_overlay_dir = phrase_output_dir / "track_overlays"
        track_mask_dir.mkdir(parents=True, exist_ok=True)
        track_overlay_dir.mkdir(parents=True, exist_ok=True)

        # Build per-frame metadata first (CPU-only, sequential)
        pending_writes: list[tuple[int, dict, np.ndarray, Path, Path]] = []
        frame_rows: list[dict[str, Any]] = []
        for global_frame_index in sorted(combined_segments.keys()):
            matching_entries = [e for e in image_entries if int(e["frame_index"]) == int(global_frame_index)]
            if not matching_entries:
                continue
            local_entry = matching_entries[0]
            flat_mask = combined_segments[global_frame_index]
            bbox = _mask_bbox(flat_mask)
            bbox_by_global_frame[global_frame_index] = bbox
            mask_path    = track_mask_dir    / f"{global_frame_index:05d}.png"
            overlay_path = track_overlay_dir / f"{global_frame_index:05d}.png"
            component_bboxes = _mask_component_bboxes(flat_mask, min_area=64, min_area_ratio=0.06)
            row = {
                "frame_index":    global_frame_index,
                "image_id":       str(local_entry["image_id"]),
                "time_value":     float(local_entry["time_value"]),
                "active":         True,
                "bbox_xyxy":      bbox,
                "mask_path":      str(mask_path),
                "overlay_path":   str(overlay_path),
                "component_count": int(len(component_bboxes)),
                "component_bboxes": component_bboxes,
                "mask_area_px":   int((flat_mask > 0).sum()),
            }
            frame_rows.append(row)
            pending_writes.append((global_frame_index, local_entry, flat_mask, mask_path, overlay_path))

        # Write mask + overlay PNGs in parallel
        def _write_frame(item: tuple) -> None:
            _, local_entry, flat_mask, mask_path, overlay_path = item
            Image.fromarray((flat_mask > 0).astype(np.uint8) * 255, mode="L").save(mask_path)
            src = Image.open(local_entry["image_path"]).convert("RGB")
            _draw_mask_preview(src, flat_mask, phrase, float(best["score"])).save(overlay_path)

        with ThreadPoolExecutor(max_workers=8) as _pool:
            list(_pool.map(_write_frame, pending_writes))

        frame_rows_by_index = {int(row["frame_index"]): row for row in frame_rows}
        split_frames = _stable_split_frames(frame_rows)

        phrase_payloads.append(
            {
                "phrase": phrase,
                "object_id": object_id,
                "status": "seeded",
                "anchor_frame_index": int(best["frame_index"]),
                "anchor_image_id": best["image_id"],
                "anchor_time_value": float(best["time_value"]),
                "anchor_bbox_xyxy": best["bbox_xyxy"],
                "anchor_score": float(best["score"]),
                "anchor_label": str(best["label"]),
                "anchor_mask_bbox_xyxy": first_mask_bbox,
                "track_window": anchor_rows[0]["track_window"] if anchor_rows else None,
                "anchor_preview_path": None if first_preview_path is None else str(first_preview_path),
                "anchor_mask_preview_path": None if first_anchor_mask_preview_path is None else str(first_anchor_mask_preview_path),
                "anchors": anchor_rows,
                "track_mask_dir": str(track_mask_dir),
                "track_overlay_dir": str(track_overlay_dir),
                "split_frames": split_frames,
                "detections": detections[: min(len(detections), 12)],
            }
        )
        track_frames = []
        for entry in image_entries:
            frame_index = int(entry["frame_index"])
            row = frame_rows_by_index.get(frame_index)
            if row is None:
                track_frames.append(
                    {
                        "frame_index": int(frame_index),
                        "image_id": str(entry["image_id"]),
                        "time_value": float(entry["time_value"]),
                        "active": False,
                        "bbox_xyxy": None,
                        "mask_path": None,
                        "overlay_path": None,
                        "component_count": 0,
                        "component_bboxes": [],
                        "mask_area_px": 0,
                    }
                )
                continue
            track_frames.append(row)
        phrase_tracks.append(
            {
                "phrase": phrase,
                "object_id": object_id,
                "status": "seeded",
                "anchor_frame_index": anchor_frame_index,
                "split_frames": split_frames,
                "frames": track_frames,
            }
        )

    payload = {
        "schema_version": 1,
        "dataset_dir": str(dataset_dir),
        "query_plan_path": str(query_plan_path),
        "grounding_model_id": grounding_model_id,
        "sam2_model_id": resolved_sam_model,
        "sam_backend": "sam3.1" if use_sam3 else "grounded-sam2",
        "prompt_type": prompt_type,
        "frame_subsample_stride": int(frame_subsample_stride),
        "num_tracking_frames": int(len(image_entries)),
        "detector_frame_stride": int(detector_frame_stride),
        "max_detector_frames": int(max_detector_frames),
        "track_window_radius": int(track_window_radius),
        "num_anchor_seeds": int(num_anchor_seeds),
        "phrases": phrase_payloads,
        "tracks": phrase_tracks,
    }
    _write_json(output_dir / "grounded_sam2_query_tracks.json", payload)
    return output_dir


class _FeatureCacheProxy(dict):
    """dict subclass whose .clear() is a no-op, preserving all cached data.

    SAM3's reset_state() calls feature_cache.clear() between queries. All
    entries are video-level caches that don't change between queries on the
    same session: string keys (multigpu_buffer, grounding_cache, text,
    tracking_bounds) and integer keys (per-frame SAM2 backbone features stored
    here by the tracker via cached_features=feature_cache). Clearing integer
    keys causes KeyError: 'images' because the tracker has no other fallback.
    """

    def clear(self) -> None:
        if "grounding_cache" in self:
            del self["grounding_cache"]


def _add_prompt_preserve_cache(
    predictor: Any,
    session_id: str,
    frame_idx: int,
    phrase: str,
    min_pixels: int = 4,
    min_score: float = 0.0,
    output_prob_thresh: float = 0.5,
) -> tuple[bool, int]:
    """Call add_prompt() while keeping the precomputed visual feature cache.

    Normally add_prompt() → reset_state() clears feature_cache entirely,
    forcing a ~40 s ViT re-encoding for all frames.  This function installs
    _FeatureCacheProxy as inference_state["feature_cache"] (if not already)
    so that reset_state()'s .clear() is a no-op.  The proxy stays installed
    permanently — propagate_in_video() writes backbone features into it, and
    tracker states (created with cached_features=proxy inside add_prompt())
    read from the same object.  Removing the proxy after add_prompt() would
    disconnect the two, causing _get_image_feature cache misses → KeyError.

    Returns (detected: bool, frame_idx: int).
    """
    session_entry = getattr(predictor, "_all_inference_states", {}).get(session_id)
    if session_entry is None:
        raise RuntimeError(f"[sam3] preserve_cache: session {session_id!r} not found")

    inference_state = session_entry["state"]

    if not isinstance(inference_state["feature_cache"], _FeatureCacheProxy):
        inference_state["feature_cache"] = _FeatureCacheProxy(inference_state["feature_cache"])

    try:
        with torch.inference_mode():
            resp = predictor.add_prompt(
                session_id=session_id,
                frame_idx=frame_idx,
                text=phrase,
                obj_id=1,
                output_prob_thresh=output_prob_thresh,
            )
    except Exception as _exc:
        import traceback as _tb
        print(f"[sam3] _add_prompt_preserve_cache ERROR frame={frame_idx} phrase={phrase!r}")
        _tb.print_exc()
        raise

    out = (resp or {}).get("outputs") or {}
    m = out.get("out_binary_masks")
    if m is None or not hasattr(m, "__len__") or len(m) == 0 or int(m[0].sum()) < min_pixels:
        return False, frame_idx
    if min_score > 0.0:
        probs = out.get("out_probs")
        if probs is not None and hasattr(probs, "__len__") and len(probs) > 0:
            if float(probs[0]) < min_score:
                return False, frame_idx
    return True, frame_idx


def _select_reference_frames(
    all_entries: list[dict[str, Any]],
    start_frame: int | None,
    end_frame:   int | None,
    n_refs: int = 3,
) -> list[int]:
    """Pick n_refs frame indices spread within [start_frame, end_frame] as SAM3 seeds."""
    valid = [int(e["frame_index"]) for e in all_entries]
    lo = start_frame if start_frame is not None else valid[0]
    hi = end_frame   if end_frame   is not None else valid[-1]
    lo = max(valid[0], lo)
    hi = min(valid[-1], hi)
    candidates = [fi for fi in valid if lo <= fi <= hi]
    if not candidates:
        candidates = valid
    if len(candidates) <= n_refs:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, num=n_refs, dtype=int)
    return [candidates[int(i)] for i in indices]


def _write_empty_sam3_tracks(output_dir: str | Path, query_plan_path: str | Path) -> None:
    """Write an empty sam3_query_tracks.json for negative/zero-target queries."""
    import json as _json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qp = _json.loads(Path(query_plan_path).read_text())
    out = {
        "query": qp.get("query", ""),
        "subjects": [],
        "tracks": [],
        "active_frame_indices": [],
        "negative_query": True,
    }
    (output_dir / "sam3_query_tracks.json").write_text(_json.dumps(out, indent=2))


def run_sam3_text_query(
    dataset_dir: str | Path,
    query_plan_path: str | Path,
    output_dir: str | Path,
    sam3_checkpoint: str | None = None,
    frame_subsample_stride: int = 1,
    _preloaded_predictor: Any | None = None,
    _preloaded_frame_dir: Path | None = None,
    _preloaded_resource_path: Any | None = None,
    _preloaded_session_id: str | None = None,
    _visual_cache_ready: bool = False,
    _offload_video_to_cpu: bool | None = None,
    start_time_id: int | None = None,
    end_time_id: int | None = None,
    ablation_raw_query: bool = False,
) -> Path:
    
    dataset_dir     = Path(dataset_dir)
    query_plan_path = Path(query_plan_path)
    output_dir      = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load plan ─────────────────────────────────────────────────────────────
    query_plan = _read_json(query_plan_path)
    subjects   = [str(s).strip() for s in (query_plan.get("subjects") or []) if str(s).strip()]
    query_text = str(query_plan.get("query", "")).strip()

    if ablation_raw_query:
        if not query_text:
            print(f"[sam3] ablation_raw_query: empty query, skipping SAM3 tracking")
            _write_empty_sam3_tracks(output_dir, query_plan_path)
            return
        phrases = [query_text]
        print(f"[sam3] ablation_raw_query: phrases={phrases}")
    else:
        # subjects=[] means Qwen identified this as a negative/zero-target query.
        # Output empty tracks immediately without calling SAM3.
        if not subjects:
            print(f"[sam3] subjects=[] → negative query, skipping SAM3 tracking")
            _write_empty_sam3_tracks(output_dir, query_plan_path)
            return
        phrases = subjects
        print(f"[sam3] subjects={subjects}")

    def _make_short_phrase(p: str) -> str:
        words = p.split()
        # Remove trailing position words
        _pos = {"center", "left", "right", "upper", "lower", "top", "bottom",
                "front", "back", "middle"}
        while words and words[-1].lower() in _pos:
            words = words[:-1]
        # Strip leading state pre-modifiers that precede the actual noun
        _lead = {"complete", "whole", "intact", "unbroken"}
        while words and words[0].lower() in _lead:
            words = words[1:]
        # Cut at state post-modifiers and spatial prepositions
        _cut = {"above", "below", "full", "more", "containing", "with", "filled",
                "holding", "broken", "empty", "cracked", "split", "half"}
        for i, w in enumerate(words):
            if w.lower() in _cut:
                words = words[:i]
                break
        return " ".join(words[:4]).strip()

    # Build extended phrase list: full phrase first, then short fallback (deduped),
    # then a further fallback stripping the first modifier word from the short phrase.
    # e.g. "clear glass cup empty center" → "clear glass cup" → "glass cup"
    _seen: set[str] = set()
    _extended: list[str] = []
    for _p in phrases:
        if _p and _p not in _seen:
            _extended.append(_p)
            _seen.add(_p)
        _short = _make_short_phrase(_p)
        if _short and _short != _p and _short not in _seen:
            _extended.append(_short)
            _seen.add(_short)
        _short_words = _short.split()
        if len(_short_words) > 2:
            _shorter = " ".join(_short_words[1:])
            if _shorter and _shorter not in _seen:
                _extended.append(_shorter)
                _seen.add(_shorter)
    phrases = _extended

    # ── load frames ───────────────────────────────────────────────────────────
    all_entries = resolve_dataset_image_entries(dataset_dir)
    _windowed = start_time_id is not None and end_time_id is not None
    # For non-windowed queries apply stride upfront; windowed queries defer
    # stride until AFTER the window filter so narrow windows keep enough frames.
    if not _windowed:
        if frame_subsample_stride > 1:
            all_entries = all_entries[::frame_subsample_stride]
        # Auto-subsample when >600 frames remain to avoid CUDA OOM in SAM3.1
        if len(all_entries) > 600 and frame_subsample_stride == 1:
            all_entries = all_entries[::2]
    if not all_entries:
        raise ValueError(f"No frames found under {dataset_dir}")

    if _windowed:
        meta_path = Path(dataset_dir) / "metadata.json"
        if meta_path.exists():
            import json as _json_mod
            _meta_raw = _json_mod.loads(meta_path.read_text(encoding="utf-8"))
            # HyperNeRF: metadata[image_id]["time_id"]
            _tid_map: dict[str, int] = {k: int(v["time_id"]) for k, v in _meta_raw.items() if "time_id" in v}
            windowed_entries = [
                e for e in all_entries
                if _tid_map.get(str(e["image_id"]), -1) >= start_time_id
                and _tid_map.get(str(e["image_id"]), -1) <= end_time_id
            ]
        else:
            # DyNeRF: time_id == frame_index
            windowed_entries = [
                e for e in all_entries
                if start_time_id <= int(e["frame_index"]) <= end_time_id
            ]
        if windowed_entries:
            # Apply a finer stride within the window (stride/4: 20→5, 10→2, 1→1)
            _win_stride = max(1, frame_subsample_stride // 4)
            all_entries = windowed_entries[::_win_stride]
            print(f"[sam3] time-window [{start_time_id},{end_time_id}]: "
                  f"{len(windowed_entries)} → {len(all_entries)} frames (win_stride={_win_stride})")
            # Don't reuse the full-dataset session; open a fresh one below.
            _preloaded_session_id = None
            _preloaded_resource_path = None
            _preloaded_frame_dir = None
            _visual_cache_ready = False
        else:
            print(f"[sam3] WARNING: time-window [{start_time_id},{end_time_id}] "
                  f"matched 0 frames — running on full video as fallback")

    if _offload_video_to_cpu is None:
        _offload_video_to_cpu = _env_flag("D4RESPLAT_SAM3_OFFLOAD_VIDEO_TO_CPU", True)

    # build JPEG frame dir (per-query when windowed, shared when not)
    if _preloaded_resource_path is not None:
        frame_resource = _preloaded_resource_path
        frame_dir = Path(_preloaded_frame_dir) if _preloaded_frame_dir is not None else None
    elif _preloaded_frame_dir is not None:
        frame_dir = Path(_preloaded_frame_dir)
        frame_resource = str(frame_dir)
    else:
        frame_dir = output_dir / "_sam3_frames"
        _materialize_jpeg_frame_dir(all_entries, frame_dir)
        frame_resource = str(frame_dir)

    fi_to_slot = {int(e["frame_index"]): i for i, e in enumerate(all_entries)}

    # middle frame of the time window — used as seed for every phrase session
    seed_candidates = [int(e["frame_index"]) for e in all_entries]
    seed_fi   = seed_candidates[len(seed_candidates) // 2]
    seed_slot = fi_to_slot[seed_fi]

    # ── load SAM3 predictor (once, reused across phrase sessions) ─────────────
    if _preloaded_predictor is not None:
        predictor = _preloaded_predictor
    else:
        checkpoint = _resolve_sam3_checkpoint(sam3_checkpoint)
        predictor  = _load_sam3_predictor(checkpoint)

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    writer_workers = max(1, int(os.environ.get("D4RESPLAT_MASK_WRITER_WORKERS", "4")))

    all_tracks: list[dict[str, Any]] = []
    all_active_indices: list[int] = []

    if len(seed_candidates) <= 40:
        # Small window — try every frame as a potential seed.
        seed_slots_to_try = list(dict.fromkeys(
            fi_to_slot[fi] for fi in seed_candidates
        ))
    else:
        _n_seeds = max(3, min(9, max(1, len(seed_candidates) // 50) + 1))
        _step    = max(1, len(seed_candidates) // _n_seeds)
        seed_slots_to_try = list(dict.fromkeys(
            fi_to_slot[seed_candidates[min(i * _step, len(seed_candidates) - 1)]]
            for i in range(_n_seeds)
        ))
        # Always include the middle frame as final fallback.
        _mid = fi_to_slot[seed_candidates[len(seed_candidates) // 2]]
        if _mid not in seed_slots_to_try:
            seed_slots_to_try.append(_mid)

    MIN_DETECT_PIXELS = 4   # reject only absolute noise (single-pixel); let tiny detections through
    # Adaptive threshold search: start at 0.45, adjust based on propagation result.
    #   active_frames == 0 or no detection  → lower threshold (missed object)
    #   active_frames > TOO_MANY_RATIO      → raise threshold (over-detection / noise)
    #   otherwise                           → success
    # Override list via SAM3_DETECT_SCORE_THRESHOLDS (must be sorted ascending).
    # Only goes downward: start at 0.45, lower if not detected, stop at 0.25.
    _SCORE_THRESHOLDS: list[float] = sorted(
        float(x.strip())
        for x in os.environ.get("SAM3_DETECT_SCORE_THRESHOLDS", "0.25,0.35,0.45").split(",")
        if x.strip()
    ) or [0.25, 0.35, 0.45]
    _START_SCORE = float(os.environ.get("SAM3_DETECT_SCORE_START", "0.45"))
    # Start at the threshold closest to _START_SCORE
    _thresh_start_idx = min(
        range(len(_SCORE_THRESHOLDS)),
        key=lambda i: abs(_SCORE_THRESHOLDS[i] - _START_SCORE),
    )

    def _check_detection(out: dict, min_score: float) -> bool:
        """Return True if add_prompt output passes both area and score thresholds."""
        m = out.get("out_binary_masks")
        if m is None or not hasattr(m, "__len__") or len(m) == 0:
            return False
        if int(m[0].sum()) < MIN_DETECT_PIXELS:
            return False
        if min_score > 0.0:
            probs = out.get("out_probs")
            if probs is not None and hasattr(probs, "__len__") and len(probs) > 0:
                if float(probs[0]) < min_score:
                    return False
        return True

    def _try_detect_first(session_id: str, phrase: str,
                          slots: list[int], min_score: float = 0.0) -> tuple[bool, int]:
        """First-phrase detection via standard add_prompt (builds visual cache)."""
        for slot in slots:
            resp = predictor.add_prompt(
                session_id=session_id,
                frame_idx=slot,
                text=phrase,
                obj_id=1,
                output_prob_thresh=max(0.05, min_score),
            )
            out = (resp or {}).get("outputs") or {}
            if _check_detection(out, min_score):
                return True, slot
        return False, slots[0]

    def _try_detect_soft(session_id: str, phrase: str,
                         slots: list[int], min_score: float = 0.0) -> tuple[bool, int]:
        """Subsequent-phrase detection via cache-preserving add_prompt (reuses visual buffer)."""
        for slot in slots:
            detected, _ = _add_prompt_preserve_cache(
                predictor, session_id, slot, phrase,
                min_pixels=MIN_DETECT_PIXELS, min_score=min_score,
                output_prob_thresh=max(0.05, min_score),
            )
            if detected:
                return True, slot
        return False, slots[0]

    MAX_RETRIES = 3
    _shared_session_id: str | None = _preloaded_session_id
    _owns_session = _preloaded_session_id is None  # only close if we opened it

    def _ensure_session() -> str:
        nonlocal _shared_session_id
        if _shared_session_id is None:
            resp = predictor.start_session(
                resource_path=frame_resource,
                offload_video_to_cpu=bool(_offload_video_to_cpu),
            )
            _shared_session_id = resp["session_id"]
        return _shared_session_id

    writer_futures: list[tuple[str, Any]] = []
    # If the caller says the visual cache is already warm, use soft reset from the start.
    _first_phrase = not _visual_cache_ready
    _skip_phrases: set[str] = set()  # short fallbacks whose parent already succeeded
    with ThreadPoolExecutor(max_workers=writer_workers) as writer_pool:
        for phrase in phrases:
            if phrase in _skip_phrases:
                print(f"[sam3] skipping redundant fallback phrase={phrase!r} (parent succeeded)")
                continue
            obj_id = 1
            slot_to_mask: dict[int, np.ndarray] = {}
            winning_fi = all_entries[seed_slots_to_try[0]]["frame_index"]
            detected_any = False

            thresh_idx: int = _thresh_start_idx

            while not detected_any and thresh_idx >= 0:
                score_thresh = _SCORE_THRESHOLDS[thresh_idx]
                remaining_slots = list(seed_slots_to_try)

                for attempt in range(MAX_RETRIES):
                    if not remaining_slots:
                        break

                    session_id = _ensure_session()

                    if _first_phrase:
                        detected, winning_slot = _try_detect_first(
                            session_id, phrase, remaining_slots, min_score=score_thresh)
                        _first_phrase = False
                    else:
                        detected, winning_slot = _try_detect_soft(
                            session_id, phrase, remaining_slots, min_score=score_thresh)
                    winning_fi = all_entries[winning_slot]["frame_index"]

                    if not detected:
                        print(f"[sam3] phrase={phrase!r} score_thr={score_thresh:.2f} attempt={attempt+1}: "
                              f"no detection → lowering threshold")
                        thresh_idx -= 1
                        break  # restart while-loop with lower threshold

                    slot_to_mask_attempt: dict[int, np.ndarray] = {}
                    _prop_thresh = max(0.05, score_thresh)
                    try:
                        for item in predictor.propagate_in_video(session_id=session_id,
                                                                  propagation_direction="forward",
                                                                  output_prob_thresh=_prop_thresh):
                            s = int(item["frame_index"])
                            outputs = item.get("outputs") or {}
                            masks = outputs.get("out_binary_masks")
                            if masks is None or len(masks) == 0:
                                continue
                            # SAM3 multiplex uses 0-based internal IDs regardless of obj_id param;
                            # union all returned masks into one segment per frame.
                            combined: np.ndarray | None = None
                            for raw_mask in masks:
                                if hasattr(raw_mask, "detach"):
                                    m = raw_mask.detach().cpu().numpy()
                                elif hasattr(raw_mask, "cpu"):
                                    m = raw_mask.cpu().numpy()
                                else:
                                    m = np.asarray(raw_mask)
                                m = np.asarray(m, dtype=np.uint8)
                                combined = m if combined is None else np.maximum(combined, m)
                            if combined is not None:
                                slot_to_mask_attempt[s] = combined
                    except Exception as _prop_exc:
                        print(f"[sam3] propagate_in_video SKIP phrase={phrase!r} attempt={attempt+1}: {_prop_exc}")
                        slot_to_mask_attempt = {}
                        # Session may be in an inconsistent state; force reopen for next phrase.
                        if _owns_session and _shared_session_id is not None:
                            try:
                                predictor.close_session(session_id=_shared_session_id)
                            except Exception:
                                pass
                        _shared_session_id = None
                        _first_phrase = True
                        break  # skip remaining attempts for this phrase
                    # Backward propagation from the same seed frame to cover frames
                    # before the anchor (e.g. object detected at frame 650 but present
                    # from frame 300 onward).
                    try:
                        for item in predictor.propagate_in_video(session_id=session_id,
                                                                  propagation_direction="backward",
                                                                  output_prob_thresh=_prop_thresh):
                            s = int(item["frame_index"])
                            outputs = item.get("outputs") or {}
                            masks = outputs.get("out_binary_masks")
                            if masks is None or len(masks) == 0:
                                continue
                            combined: np.ndarray | None = None
                            for raw_mask in masks:
                                if hasattr(raw_mask, "detach"):
                                    m = raw_mask.detach().cpu().numpy()
                                elif hasattr(raw_mask, "cpu"):
                                    m = raw_mask.cpu().numpy()
                                else:
                                    m = np.asarray(raw_mask)
                                m = np.asarray(m, dtype=np.uint8)
                                combined = m if combined is None else np.maximum(combined, m)
                            if combined is not None:
                                if s in slot_to_mask_attempt:
                                    slot_to_mask_attempt[s] = np.maximum(slot_to_mask_attempt[s], combined)
                                else:
                                    slot_to_mask_attempt[s] = combined
                    except Exception as _back_exc:
                        print(f"[sam3] backward propagation SKIP phrase={phrase!r}: {_back_exc}")

                    active_count = sum(1 for m in slot_to_mask_attempt.values() if m.max() > 0)
                    total_slots = len(all_entries)
                    print(
                        f"[sam3] phrase={phrase!r}  score_thr={score_thresh:.2f}  attempt={attempt+1}"
                        f"  seed_frame={winning_fi}  detected=YES  active_frames={active_count}/{total_slots}"
                    )

                    if active_count > 0:
                        slot_to_mask = slot_to_mask_attempt
                        detected_any = True
                        # Mark short and 2nd-level stripped fallbacks as redundant
                        _short = _make_short_phrase(phrase)
                        if _short and _short != phrase:
                            _skip_phrases.add(_short)
                            _sw = _short.split()
                            if len(_sw) > 2:
                                _skip_phrases.add(" ".join(_sw[1:]))
                        break  # success

                    # propagation gave 0 active frames: bad seed, try next one
                    if winning_slot in remaining_slots:
                        remaining_slots.remove(winning_slot)

                # Exhausted MAX_RETRIES attempts at this threshold without finding
                # any active frames. Lower the threshold to eventually terminate.
                # (remaining_slots may still have items but we've used our retry budget.)
                if not detected_any:
                    thresh_idx -= 1

            if not detected_any:
                print(f"[sam3] '{phrase}' — propagation failed, trying per-frame grounding fallback")
                pf_masks: dict[int, np.ndarray] = {}
                _pf_session_id: str | None = None
                _pf_owns_session: bool = False
                try:
                    # Reuse existing session if still open (visual cache already warm),
                    # otherwise open a fresh one.
                    if _shared_session_id is not None:
                        _pf_session_id = _shared_session_id
                        _pf_owns_session = False
                    else:
                        _pf_resp = predictor.start_session(
                            resource_path=frame_resource,
                            offload_video_to_cpu=bool(_offload_video_to_cpu),
                        )
                        _pf_session_id = _pf_resp["session_id"]
                        _pf_owns_session = True
                        _shared_session_id = _pf_session_id
                        _first_phrase = True

                    # Scan: seed slots + evenly-spaced coverage of full video
                    _pf_step = max(1, len(all_entries) // 20)
                    _pf_slots = sorted(set(
                        seed_slots_to_try
                        + list(range(0, len(all_entries), _pf_step))
                    ))
                    print(f"[sam3] per-frame scan: {len(_pf_slots)} frames for phrase={phrase!r}")

                    for _pf_slot in _pf_slots:
                        try:
                            with torch.inference_mode():
                                _pf_resp_add = predictor.add_prompt(
                                    session_id=_pf_session_id,
                                    frame_idx=_pf_slot,
                                    text=phrase,
                                    obj_id=1,
                                    output_prob_thresh=0.1,
                                )
                            _pf_out = (_pf_resp_add or {}).get("outputs") or {}
                            _pf_bin = _pf_out.get("out_binary_masks")
                            if _pf_bin is not None and len(_pf_bin) > 0:
                                _pf_arr = _pf_bin[0]
                                if hasattr(_pf_arr, "detach"):
                                    _pf_arr = _pf_arr.detach().cpu().numpy()
                                _pf_arr = np.asarray(_pf_arr, dtype=np.uint8)
                                if int(_pf_arr.sum()) >= MIN_DETECT_PIXELS:
                                    pf_masks[_pf_slot] = _pf_arr
                        except Exception as _pf_frame_exc:
                            print(f"[sam3] per-frame fallback slot={_pf_slot}: {_pf_frame_exc}")

                    if _pf_owns_session and _pf_session_id is not None:
                        try:
                            predictor.close_session(session_id=_pf_session_id)
                        except Exception:
                            pass
                        _shared_session_id = None
                        _first_phrase = True
                except Exception as _pf_outer_exc:
                    print(f"[sam3] per-frame fallback outer error for phrase={phrase!r}: {_pf_outer_exc}")

                if pf_masks:
                    slot_to_mask = pf_masks
                    detected_any = True
                    print(f"[sam3] per-frame fallback SUCCESS: {len(pf_masks)} frames for phrase={phrase!r}")
                else:
                    print(f"[sam3] WARNING: '{phrase}' — no detection across all thresholds and per-frame fallback.")

            phrase_masks_dir = masks_dir / phrase.replace(" ", "_").replace("/", "_")
            phrase_masks_dir.mkdir(exist_ok=True)
            mask_store_path = phrase_masks_dir / "track_masks.npz"
            active_frame_indices: list[int] = []
            active_masks: list[np.ndarray] = []
            track_frames: list[dict[str, Any]] = []

            for entry in all_entries:
                fi = int(entry["frame_index"])
                slot = fi_to_slot[fi]
                mask_arr = slot_to_mask.get(slot)
                active = mask_arr is not None and bool(mask_arr.max() > 0)
                bbox = _mask_bbox(mask_arr) if active else None
                mask_item_index = None
                if active and mask_arr is not None:
                    mask_item_index = len(active_frame_indices)
                    active_frame_indices.append(fi)
                    active_masks.append(np.asarray(mask_arr > 0, dtype=np.uint8))
                    all_active_indices.append(fi)
                track_frames.append(
                    {
                        "frame_index": fi,
                        "image_id": str(entry["image_id"]),
                        "time_value": float(entry["time_value"]),
                        "active": active,
                        "bbox_xyxy": bbox,
                        "mask_path": None,
                        "mask_store_path": str(mask_store_path) if active else None,
                        "mask_store_format": "npz" if active else None,
                        "mask_store_item_index": mask_item_index,
                    }
                )

            if active_masks:
                writer_futures.append(
                    (
                        phrase,
                        writer_pool.submit(
                            save_mask_store_npz,
                            mask_store_path,
                            active_frame_indices,
                            np.stack(active_masks, axis=0),
                        ),
                    )
                )

            phrase_active = [f["frame_index"] for f in track_frames if f["active"]]
            print(f"[sam3]   active_frames={len(phrase_active)}")
            all_tracks.append(
                {
                    "phrase": phrase,
                    "object_id": obj_id,
                    "status": "seeded",
                    "anchor_frame_index": winning_fi,
                    "frames": track_frames,
                }
            )

        for phrase, future in writer_futures:
            future.result()
            print(f"[sam3] wrote mask store for phrase={phrase!r}")

    # ── close shared session (only if we opened it; batch caller manages its own) ─
    if _owns_session and _shared_session_id is not None:
        try:
            predictor.close_session(session_id=_shared_session_id)
        except Exception:
            pass
        torch.cuda.empty_cache()

    # ── Derive time window from union of all active frames ────────────────────
    if all_active_indices:
        sam_time_window = {
            "start_frame":      int(min(all_active_indices)),
            "end_frame":        int(max(all_active_indices)),
            "confidence":       "sam_derived",
            "active_frame_count": len(set(all_active_indices)),
        }
    else:
        sam_time_window = {
            "start_frame":      None,
            "end_frame":        None,
            "confidence":       "sam_no_active_frames",
            "active_frame_count": 0,
        }
    print(f"[sam3] derived time_window: [{sam_time_window['start_frame']}, "
          f"{sam_time_window['end_frame']}]  active_frames={sam_time_window['active_frame_count']}")

    payload = {
        "schema_version":         2,
        "dataset_dir":            str(dataset_dir),
        "query_plan_path":        str(query_plan_path),
        "sam_backend":            "sam3.1_text",
        "subjects":               subjects,
        "sam_time_window":        sam_time_window,
        "seed_frame_index":       seed_fi,
        "frame_subsample_stride": int(frame_subsample_stride),
        "num_tracking_frames":    len(all_entries),
        "tracks":                 all_tracks,
    }
    out_path = output_dir / "sam3_query_tracks.json"
    _write_json(out_path, payload)
    return output_dir
