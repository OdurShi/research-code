import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from d4resplat.semantics.mask_store import frame_has_mask, load_frame_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "4DGaussians"
for candidate in (PROJECT_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from d4resplat.semantics.qwen_query_planner import QwenQueryPlanner, _resolve_qwen_model
from d4resplat.semantics.source_images import resolve_dataset_image_entries, resolve_dataset_time_values


PROMPT_TEMPLATE = """You are selecting candidate aliases from a query-specific 4DReSplat 4D entity library.
You will receive a natural-language query, a query plan, a small candidate set of reconstructed entities,
and candidate entity-pair interaction windows. Decide which candidate aliases are the main query subjects and,
if relevant, which candidate aliases are successor state objects. Do not choose numeric entity ids.

Total test frames: {total_frames}

Temporal field guide (all segment indices are test-frame indices, 0-based):
- support_segments_test: frames where the entity is reconstructed / visible
- moving_segments_test: frames where the entity is actively moving
- stationary_segments_test: frames where the entity is stationary
- query_relevant_segments_test: frames where this entity participates in a detected interaction

Query:
{query}

Query plan:
{query_plan_json}

Candidate entities:
{candidate_json}

Candidate interaction windows:
{pair_json}

Return exactly one JSON object with keys:
- query: original query
- subject_phrases: array of candidate aliases
- successor_phrases: array of candidate aliases
- notes: short string

Rules:
- candidate aliases must come from the candidate entities only.
- Prefer query_subject_phrases from the query plan.
- Ignore optional context objects unless the query directly asks for them.
- For action queries, the subject phrases should be the directly interacting objects.
- If a successor object is present, put it in successor_phrases rather than subject_phrases.
- Output valid JSON only.

CRITICAL temporal-selectivity rules (apply these STRICTLY):
1. FULL-FRAME COVERAGE MEANS BACKGROUND: If an entity's support_segments_test covers nearly all {total_frames} frames
   (coverage ratio > 0.85), it is almost certainly a static background object. Do NOT select it unless the query
   EXPLICITLY asks for static/always-present objects (e.g. "always stationary" / "never moves").
2. PREFER TEMPORALLY LOCALIZED ENTITIES: For queries about actions, interactions, or dynamic events,
   prefer entities whose support_segments_test or moving_segments_test covers only a SUBSET of frames.
3. STATIC QUERY EXCEPTION: If the query asks about objects that "always / never move / remain still",
   then full-frame coverage IS correct and should be selected.
4. INTERACTION GROUNDING: Strongly prefer entities that appear in query_relevant_segments_test or
   have high interaction_score in pair windows — these are the entities actually involved in the event.
5. AVOID SCENE BACKGROUND: Do not select entities whose proposal_phrase or global_desc suggests
   scene background (e.g. table surface, floor, counter, background wall) unless explicitly asked.
6. STRICT ATTRIBUTE MATCHING (MOST CRITICAL): If the query specifies a specific color, material,
   texture, or distinctive attribute (e.g. "blue glass", "red keyboard",
   "black cookie"), you MUST verify that the candidate entity's description ACTUALLY MATCHES
   that specific attribute. If NO candidate entity matches the specific attribute described in the query,
   return subject_phrases: [] (empty array). DO NOT substitute a "closest match" that differs in the
   key queried attribute. Example: if query asks for a "blue glass" but only a transparent/clear glass
   exists, return [].
7. NON-EXISTENCE DETECTION: If after careful examination NO candidate entity satisfies ALL the
   described properties in the query (object type + color/material/state + context), return
   subject_phrases: [] and explain in notes. The model will then correctly predict "entity absent"
   for all frames, which is the correct answer for negative queries.
8. EXCLUSION QUERIES: If the query says "all objects EXCEPT X" (or equivalent), select ALL
   relevant moving/interacting objects EXCEPT those explicitly excluded. May result in multiple entities.
9. USE QUERY PLAN NOTES: The query_plan_json may contain a "notes" field that the planner added after
   analyzing the scene. If the notes say the queried entity does NOT exist or has different attributes
   (e.g. "the glass is transparent, not blue", or "X does not exist in the scene"), you MUST return subject_phrases: [] because the query is asking
   for something that is not present. The query plan notes represent ground-truth scene analysis.
10. ENTITY VERIFICATION CHECKLIST: Before selecting an entity, verify ALL of the following:
    (a) Object category matches (e.g., "glass" vs "cup" vs "bottle")
    (b) Color matches if specified (e.g., "blue" must be actually blue)
    (c) Material/texture matches if specified (e.g., "stainless steel" vs "plastic")
    (d) Location/context matches if specified (e.g., "on tray" vs "on table")
    (e) State matches if specified (e.g., "solid" vs "melting", "complete" vs "broken")
    If ANY required attribute doesn't match, do NOT select the entity — return [].
"""


TEMPORAL_EXPANSION_FACTOR = 2.5


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _normalize_track_state_mode(value: Any) -> str | None:
    text = " ".join(str(value).strip().lower().replace("-", "_").split())
    if not text:
        return None
    normalized = text.replace(" ", "_")
    if normalized in {"none", "null", "unknown", "n/a", "na"}:
        return None
    return normalized


def _selection_track_state_mode(payload: dict[str, Any]) -> str | None:
    for key in ("track_state_mode", "state_mode", "query_state_mode"):
        normalized = _normalize_track_state_mode(payload.get(key))
        if normalized:
            return normalized

    notes = str(payload.get("notes", "")).strip()
    for marker in ("Track state mode=", "State mode="):
        if marker not in notes:
            continue
        tail = notes.split(marker, 1)[1]
        normalized = _normalize_track_state_mode(tail.split(";", 1)[0].strip())
        if normalized:
            return normalized

    contact_pair = payload.get("contact_pair") or {}
    source = _normalize_track_state_mode(contact_pair.get("source"))
    if not source:
        return None
    if source.startswith("single_subject_track_"):
        suffix = source.removeprefix("single_subject_track_")
        return suffix or "support"
    if "support" in source:
        return "support"
    return None


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _sample_time_values(run_dir: Path) -> np.ndarray:
    payload = np.load(run_dir / "entitybank" / "trajectory_samples.npz")
    return payload["time_values"].astype(np.float32)


def _test_time_values(run_dir: Path, dataset_dir: Path | None = None) -> np.ndarray:
    config = _read_simple_yaml(run_dir / "config.yaml")
    source_path = Path(config.get("source_path", ""))
    # Fallback: if config.yaml missing (scenes with only cfg_args), use provided dataset_dir
    # Note: Path("").exists() returns True (resolves to cwd), so check parts instead
    if not source_path.parts and dataset_dir is not None and dataset_dir.exists():
        source_path = dataset_dir
    if (source_path / "dataset.json").exists() and (source_path / "metadata.json").exists():
        _, test_times = resolve_dataset_time_values(source_path)
        dataset_payload = _read_json(source_path / "dataset.json")
        metadata_payload = _read_json(source_path / "metadata.json")
        all_ids = list(dataset_payload["ids"])
        if dataset_payload.get("val_ids"):
            val_ids = set(dataset_payload["val_ids"])
            test_ids = [image_id for image_id in all_ids if image_id in val_ids]
        else:
            i_train = [index for index in range(len(all_ids)) if index % 4 == 0]
            i_test = (np.asarray(i_train, dtype=np.int64) + 2)[:-1]
            test_ids = [all_ids[int(index)] for index in i_test]
        max_time = max(float(metadata_payload[image_id]["warp_id"]) for image_id in all_ids)
        return np.asarray(
            [float(metadata_payload[image_id]["warp_id"]) / max(max_time, 1.0) for image_id in test_ids],
            dtype=np.float32,
        )

    gt_candidates = sorted((run_dir / "test").glob("ours_*/gt"))
    if gt_candidates:
        gt_files = sorted(gt_candidates[-1].glob("*.png"))
        if len(gt_files) <= 1:
            return np.zeros((len(gt_files),), dtype=np.float32)
        return np.linspace(0.0, 1.0, num=len(gt_files), dtype=np.float32)

    entries = resolve_dataset_image_entries(source_path)
    return np.asarray([float(entry["time_value"]) for entry in entries], dtype=np.float32)


def _dataset_time_values(dataset_dir: Path) -> tuple[list[str], np.ndarray]:
    return resolve_dataset_time_values(dataset_dir)


def _mask_from_segments(segments: list[list[int]], frame_count: int) -> np.ndarray:
    mask = np.zeros((frame_count,), dtype=bool)
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            continue
        start = max(int(segment[0]), 0)
        end = min(int(segment[1]), frame_count - 1)
        if end < start:
            continue
        mask[start : end + 1] = True
    return mask


def _resample_mask(sample_mask: np.ndarray, sample_times: np.ndarray, test_times: np.ndarray) -> np.ndarray:
    sample_mask = np.asarray(sample_mask, dtype=np.float32)
    indices = np.abs(test_times[:, None] - sample_times[None, :]).argmin(axis=1)
    return sample_mask[indices] > 0.5


def _ranges_from_mask(mask: np.ndarray) -> list[list[int]]:
    indices = np.where(np.asarray(mask, dtype=bool))[0]
    if indices.size == 0:
        return []
    ranges: list[list[int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        ranges.append([start, prev])
        start = value
        prev = value
    ranges.append([start, prev])
    return ranges


def _map_segments_to_test(
    segments: list[list[int]],
    sample_times: np.ndarray,
    test_times: np.ndarray,
) -> list[list[int]]:
    if sample_times.size == 0 or test_times.size == 0:
        return []
    sample_mask = _mask_from_segments(segments, int(sample_times.shape[0]))
    if not sample_mask.any():
        return []
    return _ranges_from_mask(_resample_mask(sample_mask.astype(np.float32), sample_times, test_times))


def _support_segment_from_window(support_window: dict[str, Any]) -> list[list[int]]:
    if not support_window:
        return []
    start = int(support_window.get("frame_start", support_window.get("start", 0)))
    end = int(support_window.get("frame_end", support_window.get("end", start)))
    if end < start:
        start, end = end, start
    return [[start, end]]


def _merge_ranges(ranges: list[list[int]]) -> list[list[int]]:
    if not ranges:
        return []
    ordered = sorted(([int(start), int(end)] for start, end in ranges), key=lambda item: (item[0], item[1]))
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev = merged[-1]
        if start <= prev[1] + 1:
            prev[1] = max(prev[1], end)
            continue
        merged.append([start, end])
    return merged


def _normalize_phrase(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _phrase_tokens(value: str) -> set[str]:
    return {token for token in _normalize_phrase(value).split() if token}


def _candidate_phrase_score(
    target: str,
    candidate: dict[str, Any],
    phrase_entity: dict[str, Any] | None = None,
) -> float:
    target_norm = _normalize_phrase(target)
    target_tokens = _phrase_tokens(target)
    texts = [
        candidate.get("proposal_alias", ""),
        candidate.get("proposal_phrase", ""),
        candidate.get("static_text", ""),
        candidate.get("global_desc", ""),
        " ".join(str(tag) for tag in candidate.get("concept_tags", [])),
    ]
    # Augment target tokens with structured entity info from phrase_entities:
    # - For relational phrases (e.g. "coffee in glass"), also match against container object
    # - For attribute-rich phrases, add attributes as extra match tokens
    extra_target_texts: list[str] = []
    if phrase_entity:
        obj = str(phrase_entity.get("object") or "").strip()
        if obj:
            extra_target_texts.append(obj)
        rel = phrase_entity.get("relation")
        # Only add container when the object is IN the container (type="in").
        # For type="contains" (the object holds something else), the container is
        # not an alternative name for this object and should not broaden matching.
        if isinstance(rel, dict) and str(rel.get("type") or "").strip().lower() == "in":
            container = str(rel.get("container") or "").strip()
            if container:
                extra_target_texts.append(container)
        for attr in phrase_entity.get("attributes") or []:
            extra_target_texts.append(str(attr).strip())
        hint = str(phrase_entity.get("detector_hint") or "").strip()
        if hint:
            extra_target_texts.append(hint)

    all_target_texts = [target] + extra_target_texts
    all_target_tokens = set()
    for t in all_target_texts:
        all_target_tokens |= _phrase_tokens(t)

    exact_bonus = 0.0
    best_overlap = 0.0
    for text in texts:
        text_norm = _normalize_phrase(text)
        if not text_norm:
            continue
        if text_norm == target_norm:
            exact_bonus = max(exact_bonus, 1.0)
        elif text_norm.startswith(target_norm) or target_norm.startswith(text_norm):
            exact_bonus = max(exact_bonus, 0.8)
        elif target_norm and target_norm in text_norm:
            exact_bonus = max(exact_bonus, 0.6)
        else:
            # Check each extra target text (object / container / attributes)
            for extra in extra_target_texts:
                extra_norm = _normalize_phrase(extra)
                if not extra_norm:
                    continue
                if text_norm == extra_norm:
                    exact_bonus = max(exact_bonus, 0.85)
                elif extra_norm in text_norm or text_norm in extra_norm:
                    exact_bonus = max(exact_bonus, 0.55)
        text_tokens = _phrase_tokens(text_norm)
        if all_target_tokens and text_tokens:
            overlap = len(all_target_tokens & text_tokens) / max(len(all_target_tokens | text_tokens), 1)
            best_overlap = max(best_overlap, float(overlap))
    return float(exact_bonus + best_overlap)


def _range_len(segment: list[int]) -> int:
    return max(0, int(segment[1]) - int(segment[0]) + 1)


def _first_range_start(ranges: list[list[int]]) -> int:
    if not ranges:
        return 10**9
    return min(int(start) for start, _ in ranges)


def _select_phrase_ids(
    candidates: list[dict[str, Any]],
    phrases: list[str],
    allow_missing: bool = False,
    phrase_entities: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[int], dict[str, list[int]]]:
    normalized_targets = [_normalize_phrase(phrase) for phrase in phrases if str(phrase).strip()]
    selected_ids: list[int] = []
    selected_by_phrase: dict[str, list[int]] = {}
    if not normalized_targets:
        return selected_ids, selected_by_phrase
    for target in normalized_targets:
        entity_ctx = (phrase_entities or {}).get(target)
        def _score(item: dict[str, Any]) -> float:
            return _candidate_phrase_score(target, item, phrase_entity=entity_ctx)
        ranked = sorted(
            candidates,
            key=lambda item: (
                -_score(item),
                -float(item.get("quality", 0.0)),
                _first_range_start(item.get("query_relevant_segments_test", [])),
                int(item.get("id", -1)),
            ),
        )
        matches = [item for item in ranked if _score(item) >= 0.45]
        if not matches and ranked and _score(ranked[0]) >= 0.30:
            matches = [ranked[0]]
        matches.sort(
            key=lambda item: (
                -_score(item),
                -float(item.get("quality", 0.0)),
                _first_range_start(item.get("query_relevant_segments_test", [])),
                int(item.get("id", -1)),
            )
        )
        if not matches:
            if allow_missing:
                continue  # skip phrases not found in entitybank (e.g. static queries)
            if ranked:
                # SAM tracking likely failed (0 active frames → spatial-cluster fallback);
                # take best-scored entity rather than crashing.
                print(f"[select] WARNING: no strong match for '{target}', using best-available entity id={ranked[0]['id']} (score={_score(ranked[0]):.3f})")
                matches = [ranked[0]]
            else:
                continue
        selected = matches[0]
        entity_id = int(selected["id"])
        selected_ids.append(entity_id)
        selected_by_phrase[target] = [int(item["id"]) for item in matches]
    return selected_ids, selected_by_phrase


def _choose_subject_pair(
    pair_candidates: list[dict[str, Any]],
    subject_ids: list[int],
) -> dict[str, Any]:
    unique_subject_ids = sorted({int(value) for value in subject_ids})
    if len(unique_subject_ids) < 2:
        raise ValueError("Action-query composition requires at least two subject entities.")
    wanted = set(unique_subject_ids)
    rows = []
    for row in pair_candidates:
        entity_a = int(row["entity_a"])
        entity_b = int(row["entity_b"])
        if {entity_a, entity_b} != wanted:
            continue
        segments = row.get("contact_segments_test", [])
        if not segments:
            continue
        rows.append(row)
    if not rows:
        # No contact pairs found — fall back to any pair covering both entities,
        # using proximity_segments or query_relevant_segments as interaction window.
        for row in pair_candidates:
            entity_a = int(row["entity_a"])
            entity_b = int(row["entity_b"])
            if {entity_a, entity_b} != wanted:
                continue
            fallback_segs = (
                row.get("proximity_segments_test")
                or row.get("query_relevant_segments_test")
                or row.get("co_active_segments_test")
                or []
            )
            if fallback_segs:
                rows.append({**row, "contact_segments_test": fallback_segs})
        if not rows:
            # Last resort: synthesize a dummy pair with empty segments so downstream
            # code can still produce a result rather than crashing.
            print("[select] WARNING: no interaction window found; using dummy pair fallback.")
            rows = [{"entity_a": unique_subject_ids[0], "entity_b": unique_subject_ids[1],
                     "contact_segments_test": [], "contact_ratio": 0.0, "interaction_score": 0.0}]
    rows.sort(
        key=lambda item: (
            -float(item.get("contact_ratio", 0.0)),
            -float(item.get("interaction_score", 0.0)),
            _first_range_start(item.get("contact_segments_test", [])),
        )
    )
    return rows[0]


def _successor_stop_frame(
    candidates: list[dict[str, Any]],
    successor_ids: list[int],
    start_frame: int,
    contact_end: int,
) -> int:
    if not successor_ids:
        return int(contact_end)
    stop = int(contact_end)
    for entity_id in successor_ids:
        row = next((candidate for candidate in candidates if int(candidate["id"]) == int(entity_id)), None)
        if row is None:
            continue
        for segment in row.get("query_relevant_segments_test", []):
            segment_start = int(segment[0])
            if segment_start < start_frame:
                continue
            stop = max(stop, segment_start)
            return stop
        for segment in row.get("support_segments_test", []):
            segment_start = int(segment[0])
            if segment_start < start_frame:
                continue
            stop = max(stop, segment_start)
            return stop
    return stop


def _entity_phrase_map(run_dir: Path) -> dict[int, dict[str, Any]]:
    entities_path = run_dir / "entitybank" / "entities.json"
    payload = _read_json(entities_path)
    return {int(entity["id"]): entity for entity in payload.get("entities", [])}


def _resolve_query_tracks_payload(query_plan_path: Path | None) -> dict[str, Any] | None:
    if query_plan_path is None:
        return None
    base = query_plan_path.parent
    # Priority: sam3 text-only tracks (new pipeline) → grounded_sam2 tracks (old pipeline)
    candidates = [
        base / "sam3_tracks" / "sam3_query_tracks.json",
        base / "sam3_query_tracks.json",
        base / "grounded_sam2" / "grounded_sam2_query_tracks.json",
    ]
    for tracks_path in candidates:
        if tracks_path.exists():
            return _read_json(tracks_path)
    return None


def _query_track_items(tracks_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not tracks_payload:
        return []
    tracks = tracks_payload.get("tracks")
    if isinstance(tracks, list) and tracks:
        return tracks
    phrases = tracks_payload.get("phrases")
    if isinstance(phrases, list):
        return phrases
    return []


def _track_active_frame_count(track: dict[str, Any] | None) -> int:
    if not isinstance(track, dict):
        return 0
    return int(
        sum(
            1
            for frame in track.get("frames", [])
            if bool(frame.get("active")) and frame_has_mask(frame)
        )
    )


def _query_track_by_phrase(tracks_payload: dict[str, Any] | None, phrase: str) -> dict[str, Any] | None:
    items = _query_track_items(tracks_payload)
    if not items:
        return None
    phrase_tokens = _phrase_tokens(phrase)
    phrase_norm = _normalize_phrase(phrase)

    # First: exact phrase match, but prefer a usable track that actually has active masks.
    exact_matches = [track for track in items if _normalize_phrase(track.get("phrase", "")) == phrase_norm]
    if exact_matches:
        exact_active = [track for track in exact_matches if _track_active_frame_count(track) > 0]
        if exact_active:
            exact_active.sort(key=lambda track: (-_track_active_frame_count(track), str(track.get("phrase", ""))))
            return exact_active[0]

    # Second: token overlap among tracks that actually have active masks.
    # This handles cases where the refined subject phrase is more specific than the SAM phrase,
    # e.g. "clear glass cup filled with dark espresso liquid center" -> "clear glass cup".
    best_track = None
    best_score = 0
    best_active_count = -1
    for track in items:
        active_count = _track_active_frame_count(track)
        if active_count <= 0:
            continue
        track_tokens = _phrase_tokens(track.get("phrase", ""))
        overlap = len(phrase_tokens & track_tokens)
        if overlap > best_score or (overlap == best_score and active_count > best_active_count):
            best_score = overlap
            best_track = track
            best_active_count = active_count
    if best_score > 0:
        return best_track

    # Last resort: preserve previous exact-match behavior even if that exact track had no active masks.
    if exact_matches:
        return exact_matches[0]
    return None


def _load_binary_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _dilate_mask(mask: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    if kernel_size <= 1:
        return np.asarray(mask, dtype=bool)
    if kernel_size % 2 == 0:
        kernel_size += 1
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    dilated = image.filter(ImageFilter.MaxFilter(kernel_size))
    return np.asarray(dilated, dtype=np.uint8) > 0


def _track_contact_segments_test(
    tracks_payload: dict[str, Any] | None,
    phrase_a: str,
    phrase_b: str,
    test_times: np.ndarray,
) -> list[list[int]]:
    track_a = _query_track_by_phrase(tracks_payload, phrase_a)
    track_b = _query_track_by_phrase(tracks_payload, phrase_b)
    if track_a is None or track_b is None:
        return []

    frames_a = {
        int(frame["frame_index"]): frame
        for frame in track_a.get("frames", [])
        if bool(frame.get("active")) and frame_has_mask(frame)
    }
    frames_b = {
        int(frame["frame_index"]): frame
        for frame in track_b.get("frames", [])
        if bool(frame.get("active")) and frame_has_mask(frame)
    }
    common_indices = sorted(set(frames_a.keys()) & set(frames_b.keys()))
    if not common_indices:
        return []

    contact_test_indices: list[int] = []
    for frame_index in common_indices:
        frame_a = frames_a[frame_index]
        frame_b = frames_b[frame_index]
        mask_a = load_frame_mask(frame_a)
        mask_b = load_frame_mask(frame_b)
        if mask_a is None or mask_b is None:
            continue
        if mask_a.shape != mask_b.shape:
            continue
        dilated_a = _dilate_mask(mask_a, kernel_size=11)
        dilated_b = _dilate_mask(mask_b, kernel_size=11)
        touching = bool((dilated_a & mask_b).any() or (dilated_b & mask_a).any())
        if not touching:
            continue
        time_value = 0.5 * (float(frame_a.get("time_value", 0.0)) + float(frame_b.get("time_value", 0.0)))
        contact_test_indices.append(int(np.abs(test_times - time_value).argmin()))
    return _merge_ranges([[index, index] for index in sorted(set(contact_test_indices))])


def _track_split_start_frame(
    tracks_payload: dict[str, Any] | None,
    phrase: str,
    test_times: np.ndarray,
    start_frame: int | None = None,
) -> int | None:
    track = _query_track_by_phrase(tracks_payload, phrase)
    if track is None:
        return None
    split_times: list[float] = []
    frames = [frame for frame in track.get("frames", []) if bool(frame.get("active"))]
    streak = 0
    for frame in frames:
        component_count = int(frame.get("component_count", 0))
        if component_count >= 2:
            streak += 1
        else:
            streak = 0
        if streak >= 2:
            split_times.append(float(frame.get("time_value", 0.0)))
            break
    if not split_times:
        for frame in frames:
            if int(frame.get("component_count", 0)) >= 2:
                split_times.append(float(frame.get("time_value", 0.0)))
                break
    if not split_times:
        return None
    nearest = int(np.abs(test_times - np.asarray(split_times, dtype=np.float32)[0]).argmin())
    if start_frame is not None and nearest < int(start_frame):
        return None
    return nearest


def _query_plan_frame_to_test_index(
    query_plan_payload: dict[str, Any],
    test_times: np.ndarray,
    frame_index: int | None,
) -> int | None:
    if frame_index is None:
        return None
    dataset_dir_value = query_plan_payload.get("dataset_dir")
    if not dataset_dir_value:
        return None
    dataset_dir = Path(dataset_dir_value)
    try:
        all_ids, all_times = _dataset_time_values(dataset_dir)
    except Exception:
        return None
    if int(frame_index) < 0 or int(frame_index) >= len(all_ids):
        return None
    time_value = float(all_times[int(frame_index)])
    return int(np.abs(test_times - time_value).argmin())


def _query_plan_boundary_test_index(
    query_plan_payload: dict[str, Any],
    test_times: np.ndarray,
    boundary_kind: str,
    phrase: str | None = None,
) -> int | None:
    phrase_norm = _normalize_phrase(phrase or "")
    if boundary_kind == "start":
        hints = query_plan_payload.get("phase_transition_hints") or []
        for hint in hints:
            hint_phrase = _normalize_phrase(hint.get("phrase", ""))
            if phrase_norm and hint_phrase and hint_phrase != phrase_norm:
                continue
            index = _query_plan_frame_to_test_index(
                query_plan_payload,
                test_times,
                hint.get("first_post_change_frame_index"),
            )
            if index is not None:
                return index
    refined_window = query_plan_payload.get("refined_temporal_window") or {}
    frame_index = refined_window.get("start_frame_index") if boundary_kind == "start" else refined_window.get("end_frame_index")
    return _query_plan_frame_to_test_index(query_plan_payload, test_times, frame_index)


def _query_plan_window_test_range(
    query_plan_payload: dict[str, Any],
    test_times: np.ndarray,
) -> tuple[int | None, int | None]:
    refined_window = query_plan_payload.get("refined_temporal_window") or {}
    start_frame = refined_window.get("start_frame_index")
    end_frame = refined_window.get("end_frame_index")
    # bisect_state writes "time_window" not "refined_temporal_window"; fall back to it
    if start_frame is None and end_frame is None:
        bisect_window = query_plan_payload.get("time_window") or {}
        conf = str(bisect_window.get("confidence", ""))
        if conf and "no_detection" not in conf:
            start_frame = bisect_window.get("start_frame")
            end_frame = bisect_window.get("end_frame")
    start_index = _query_plan_frame_to_test_index(query_plan_payload, test_times, start_frame)
    end_index = _query_plan_frame_to_test_index(query_plan_payload, test_times, end_frame)
    return start_index, end_index


def _clip_segment_to_plan_window(
    segment: list[int],
    query_plan_payload: dict[str, Any],
    test_times: np.ndarray,
) -> list[int]:
    start = int(segment[0])
    end = int(segment[1])
    plan_start, plan_end = _query_plan_window_test_range(query_plan_payload, test_times)
    if plan_start is not None:
        start = max(start, int(plan_start))
    if plan_end is not None:
        end = min(end, int(plan_end))
    if end < start:
        if plan_start is not None and plan_end is not None and int(plan_end) >= int(plan_start):
            return [int(plan_start), int(plan_end)]
        return [int(segment[0]), int(segment[1])]
    return [start, end]


def _intersect_ranges(
    ranges_a: list[list[int]],
    ranges_b: list[list[int]],
    frame_count: int,
) -> list[list[int]]:
    if frame_count <= 0 or not ranges_a or not ranges_b:
        return []
    mask = _mask_from_segments(ranges_a, frame_count) & _mask_from_segments(ranges_b, frame_count)
    return _ranges_from_mask(mask)


def _track_active_mask_test(
    tracks_payload: dict[str, Any] | None,
    phrase: str,
    test_times: np.ndarray,
) -> np.ndarray:
    mask = np.zeros((int(test_times.shape[0]),), dtype=bool)
    track = _query_track_by_phrase(tracks_payload, phrase)
    if track is None:
        return mask
    active_times = np.asarray(
        [
            float(frame.get("time_value", 0.0))
            for frame in track.get("frames", [])
            if bool(frame.get("active")) and frame_has_mask(frame)
        ],
        dtype=np.float32,
    )
    if active_times.size == 0 or test_times.size == 0:
        return mask
    active_times = np.sort(active_times)
    tolerance = 0.012
    if active_times.size >= 2:
        diffs = np.diff(active_times)
        if diffs.size:
            tolerance = max(tolerance, float(np.median(diffs)) * 0.65)
    nearest = np.abs(test_times[:, None] - active_times[None, :]).argmin(axis=1)
    nearest_times = active_times[nearest]
    mask = np.abs(test_times - nearest_times) <= float(tolerance)
    mask &= test_times >= float(active_times.min()) - float(tolerance)
    mask &= test_times <= float(active_times.max()) + float(tolerance)
    return mask


def _track_active_segments_test(
    tracks_payload: dict[str, Any] | None,
    phrase: str,
    test_times: np.ndarray,
) -> list[list[int]]:
    return _ranges_from_mask(_track_active_mask_test(tracks_payload, phrase, test_times))


def _zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    std = float(array.std())
    if std <= 1.0e-6:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - float(array.mean())) / std).astype(np.float32)


def _smooth_series(values: np.ndarray, radius: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0 or radius <= 0:
        return array
    kernel = np.ones((int(radius) * 2 + 1,), dtype=np.float32)
    padded = np.pad(array, (int(radius), int(radius)), mode="edge")
    smoothed = np.convolve(padded, kernel / kernel.sum(), mode="valid")
    return smoothed.astype(np.float32)


def _is_exclusion_query(query_norm: str) -> bool:
    """Return True if the query is an exclusion query: 'all objects EXCEPT X'."""
    exclusion_patterns = [
        "except",
        "excluding",
        "other than",
        "besides",
    ]
    return any(pat in query_norm for pat in exclusion_patterns)


def _query_state_mode(query_norm: str) -> str | None:
    if "above the midpoint" in query_norm or "above midpoint" in query_norm or "midpoint of the cup" in query_norm:
        return "above_midpoint"
    if "full" in query_norm:
        return "full"
    if "empty" in query_norm:
        return "empty"
    if "light colored" in query_norm or "light-colored" in query_norm or "lighter" in query_norm:
        return "light"
    if "darker" in query_norm or " dark " in f" {query_norm} " or query_norm.endswith(" dark"):
        return "dark"
    if "opened" in query_norm or " open " in f" {query_norm} " or query_norm.endswith(" open"):
        return "opened"
    if "closed" in query_norm or " close " in f" {query_norm} " or query_norm.endswith(" close"):
        return "closed"
    # Static-throughout-video queries
    _static_query_patterns = [
        "always stationary",
        "always static",
        "always remain stationary",
        "stationary throughout",
        "never move",
        "physically stationary",
    ]
    if any(p in query_norm for p in _static_query_patterns):
        return "static"
    return None


def _track_state_series(
    query_plan_payload: dict[str, Any],
    tracks_payload: dict[str, Any] | None,
    phrase: str,
    test_times: np.ndarray,
) -> dict[str, np.ndarray] | None:
    dataset_dir_value = query_plan_payload.get("dataset_dir")
    if not dataset_dir_value:
        return None
    track = _query_track_by_phrase(tracks_payload, phrase)
    if track is None:
        return None
    dataset_dir = Path(dataset_dir_value)
    try:
        image_entries = resolve_dataset_image_entries(dataset_dir)
    except Exception:
        return None
    image_path_by_id = {str(entry["image_id"]): Path(entry["image_path"]) for entry in image_entries}
    frame_rows = [
        frame
        for frame in track.get("frames", [])
        if bool(frame.get("active")) and frame_has_mask(frame)
    ]
    if not frame_rows:
        return None

    track_times: list[float] = []
    area_values: list[float] = []
    aspect_values: list[float] = []
    height_values: list[float] = []
    width_values: list[float] = []
    weighted_dark_values: list[float] = []
    lower_dark_values: list[float] = []
    upper_dark_values: list[float] = []
    for frame in frame_rows:
        image_path = image_path_by_id.get(str(frame.get("image_id", "")))
        if image_path is None or not image_path.exists():
            continue
        mask = load_frame_mask(frame)
        if mask is None:
            continue
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        height = max(y1 - y0 + 1, 1)
        width = max(x1 - x0 + 1, 1)
        yy = np.arange(mask.shape[0], dtype=np.float32)[:, None]
        rel = (yy - float(y0)) / max(float(y1 - y0), 1.0)
        weighted = np.where(mask, rel, 0.0)
        lower_mask = mask & (yy > (float(y0 + y1) * 0.5))
        upper_mask = mask & (yy <= (float(y0 + y1) * 0.5))
        weighted_dark = float((weighted * (gray < 0.45)).sum() / max(weighted.sum(), 1.0e-6))
        lower_dark = float((gray[lower_mask] < 0.45).mean()) if lower_mask.any() else 0.0
        upper_dark = float((gray[upper_mask] < 0.45).mean()) if upper_mask.any() else 0.0
        track_times.append(float(frame.get("time_value", 0.0)))
        area_values.append(float(mask.mean()))
        aspect_values.append(float(width / max(height, 1)))
        height_values.append(float(height / max(mask.shape[0], 1)))
        width_values.append(float(width / max(mask.shape[1], 1)))
        weighted_dark_values.append(weighted_dark)
        lower_dark_values.append(lower_dark)
        upper_dark_values.append(upper_dark)

    if not track_times:
        return None
    order = np.argsort(np.asarray(track_times, dtype=np.float32))
    times = np.asarray(track_times, dtype=np.float32)[order]

    def interp(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)[order]
        return np.interp(test_times, times, array).astype(np.float32)

    area_series = interp(area_values)
    aspect_series = interp(aspect_values)
    height_series = interp(height_values)
    width_series = interp(width_values)
    weighted_dark_series = interp(weighted_dark_values)
    lower_dark_series = interp(lower_dark_values)
    upper_dark_series = interp(upper_dark_values)
    fill_score = _smooth_series(
        weighted_dark_series + 0.35 * lower_dark_series + 0.15 * upper_dark_series,
        radius=1,
    )
    dark_score = _smooth_series(
        0.70 * weighted_dark_series + 0.30 * lower_dark_series,
        radius=1,
    )
    open_score = _smooth_series(
        _zscore(area_series) + 0.70 * _zscore(height_series) - 0.80 * _zscore(aspect_series),
        radius=1,
    )
    return {
        "valid_mask": _track_active_mask_test(tracks_payload, phrase, test_times),
        "fill_score": fill_score.astype(np.float32),
        "dark_score": dark_score.astype(np.float32),
        "open_score": open_score.astype(np.float32),
        "area": area_series.astype(np.float32),
        "aspect": aspect_series.astype(np.float32),
        "height": height_series.astype(np.float32),
        "width": width_series.astype(np.float32),
    }


def _track_state_segments_test(
    query: str,
    query_plan_payload: dict[str, Any],
    tracks_payload: dict[str, Any] | None,
    phrase: str,
    test_times: np.ndarray,
) -> tuple[list[list[int]], dict[str, Any] | None]:
    query_norm = _normalize_phrase(query)
    state_mode = _query_state_mode(query_norm)
    frame_count = int(test_times.shape[0])
    visible_mask = _track_active_mask_test(tracks_payload, phrase, test_times)
    visible_segments = _ranges_from_mask(visible_mask)
    if state_mode is None:
        return [], {
            "state_mode": "support",
            "visible_segments_test": visible_segments,
        }
    plan_start, plan_end = _query_plan_window_test_range(query_plan_payload, test_times)
    series = _track_state_series(query_plan_payload, tracks_payload, phrase, test_times)
    if series is None:
        if plan_start is not None or plan_end is not None:
            s = 0 if state_mode == "empty" else (int(plan_start) if plan_start is not None else 0)
            e = int(plan_end) if plan_end is not None else frame_count - 1
            derived_mask = np.zeros((frame_count,), dtype=bool)
            derived_mask[s : e + 1] = True
            return _ranges_from_mask(derived_mask), {
                "state_mode": state_mode,
                "strategy": "bisect_window_direct",
                "fallback": "no_sam_evidence",
            }
        return [], None
    valid_mask = np.asarray(series["valid_mask"], dtype=bool)
    if not valid_mask.any():
        if plan_start is not None or plan_end is not None:
            s = 0 if state_mode == "empty" else (int(plan_start) if plan_start is not None else 0)
            e = int(plan_end) if plan_end is not None else frame_count - 1
            derived_mask = np.zeros((frame_count,), dtype=bool)
            derived_mask[s : e + 1] = True
            return _ranges_from_mask(derived_mask), {
                "state_mode": state_mode,
                "strategy": "bisect_window_direct",
                "fallback": "no_valid_mask",
            }
        return [], None

    metadata: dict[str, Any] = {
        "state_mode": state_mode,
        "visible_segments_test": visible_segments,
    }
    score = None

    def _normalized_progress(score_array: np.ndarray) -> tuple[np.ndarray, float, float]:
        valid_scores = np.asarray(score_array[valid_mask], dtype=np.float32)
        if valid_scores.size == 0:
            return np.zeros_like(score_array, dtype=np.float32), 0.0, 0.0
        low = float(np.quantile(valid_scores, 0.05))
        high = float(np.quantile(valid_scores, 0.95))
        scale = max(high - low, 1.0e-6)
        progress = np.clip((np.asarray(score_array, dtype=np.float32) - low) / scale, 0.0, 1.0)
        return progress.astype(np.float32), low, high

    def _first_progress_onset(
        progress_array: np.ndarray,
        *,
        threshold: float,
        min_index: int,
        sustain_window: int = 1,
        sustain_count: int = 1,
    ) -> int | None:
        active = valid_mask & (np.arange(frame_count) >= int(max(min_index, 0))) & (progress_array >= float(threshold))
        indices = np.where(active)[0]
        if indices.size == 0:
            return None
        if sustain_window <= 1 or sustain_count <= 1:
            return int(indices[0])
        active_int = active.astype(np.int32)
        for onset in indices.tolist():
            end = min(frame_count, int(onset) + int(max(sustain_window, 1)))
            if int(active_int[int(onset) : int(end)].sum()) >= int(max(sustain_count, 1)):
                return int(onset)
        return None

    def _suffix_from_score(
        score_array: np.ndarray,
        *,
        quantile: float,
        floor: float,
        min_index: int,
        fallback_start: int,
    ) -> np.ndarray:
        threshold_value = float(max(np.quantile(score_array[valid_mask], quantile), floor))
        search_mask = valid_mask & (np.arange(frame_count) >= int(max(min_index, 0)))
        active = np.where(search_mask & (score_array >= threshold_value))[0]
        if active.size == 0:
            active = np.where(valid_mask & (score_array >= threshold_value))[0]
        onset = int(active[0]) if active.size else int(fallback_start)
        suffix = np.zeros((frame_count,), dtype=bool)
        suffix[max(0, onset) :] = True
        metadata["threshold"] = threshold_value
        return suffix

    if state_mode == "full":
        score = np.asarray(series["fill_score"], dtype=np.float32)
        default_start = visible_segments[-1][0] if visible_segments else max(frame_count - 1, 0)
        min_index = max(int(frame_count * 0.65), int(plan_start or 0))
        derived_mask = _suffix_from_score(
            score,
            quantile=0.85,
            floor=0.08,
            min_index=min_index,
            fallback_start=default_start,
        )
    elif state_mode == "above_midpoint":
        score = np.asarray(series["fill_score"], dtype=np.float32)
        default_start = visible_segments[1][0] if len(visible_segments) >= 2 else (visible_segments[-1][0] if visible_segments else 0)
        min_index = max(int(frame_count * 0.45), int(plan_start or 0))
        derived_mask = _suffix_from_score(
            score,
            quantile=0.70,
            floor=0.04,
            min_index=min_index,
            fallback_start=default_start,
        )
    elif state_mode == "dark":
        score = np.asarray(series["dark_score"], dtype=np.float32)
        default_start = visible_segments[1][0] if len(visible_segments) >= 2 else (visible_segments[-1][0] if visible_segments else 0)
        progress, progress_low, progress_high = _normalized_progress(score)
        min_index = max(int(frame_count * 0.15), int(plan_start or 0))
        onset = _first_progress_onset(
            progress,
            threshold=0.18,
            min_index=min_index,
        )
        if onset is None:
            onset = int(default_start)
        derived_mask = np.zeros((frame_count,), dtype=bool)
        derived_mask[max(0, int(onset)) :] = True
        metadata["threshold"] = 0.18
        metadata["progress_low"] = progress_low
        metadata["progress_high"] = progress_high
    elif state_mode == "empty":
        score = np.asarray(series["fill_score"], dtype=np.float32)
        progress, progress_low, progress_high = _normalized_progress(score)
        if plan_end is not None:
            # bisect directly tells us where the empty period ends; trust it
            prefix_end = int(plan_end)
            metadata["strategy"] = "bisect_window_prefix"
        else:
            onset = _first_progress_onset(
                progress,
                threshold=0.15,
                min_index=int(plan_start or 0),
            )
            if onset is None:
                if not visible_segments:
                    return [], {**metadata, "fallback": "support"}
                prefix_end = int(visible_segments[0][1])
                metadata["strategy"] = "leading_visible_prefix_fallback"
            else:
                prefix_end = max(0, int(onset) - 1)
                metadata["strategy"] = "prefix_until_fill_onset"
        metadata["threshold"] = 0.15
        metadata["progress_low"] = progress_low
        metadata["progress_high"] = progress_high
        derived_mask = np.zeros((frame_count,), dtype=bool)
        derived_mask[: prefix_end + 1] = True
    elif state_mode == "light":
        score = np.asarray(series["dark_score"], dtype=np.float32)
        progress, progress_low, progress_high = _normalized_progress(score)
        onset = _first_progress_onset(
            progress,
            threshold=0.30,
            min_index=int(plan_start or 0),
        )
        if onset is None:
            if not visible_segments:
                return [], {**metadata, "fallback": "support"}
            prefix_end = int(visible_segments[0][1])
            metadata["strategy"] = "leading_visible_prefix_fallback"
        else:
            prefix_end = max(0, int(onset) - 1)
            metadata["strategy"] = "prefix_until_dark_onset"
        metadata["threshold"] = 0.30
        metadata["progress_low"] = progress_low
        metadata["progress_high"] = progress_high
        derived_mask = np.zeros((frame_count,), dtype=bool)
        derived_mask[: prefix_end + 1] = True
    elif state_mode in {"opened", "closed"}:
        score = np.asarray(series["open_score"], dtype=np.float32)
        threshold = float(np.quantile(score[valid_mask], 0.50))
        raw_mask = valid_mask & (score >= threshold)
        smoothed_mask = np.convolve(raw_mask.astype(np.int32), np.ones((3,), dtype=np.int32), mode="same") >= 2
        derived_mask = smoothed_mask & valid_mask
        if state_mode == "closed":
            derived_mask = valid_mask & (~derived_mask)
    else:
        return visible_segments, metadata

    derived_segments = _ranges_from_mask(derived_mask)
    if score is not None:
        metadata["score_min"] = float(score[valid_mask].min())
        metadata["score_max"] = float(score[valid_mask].max())
    metadata["derived_segments_test"] = derived_segments
    return derived_segments, metadata


def _entities_payload_to_assignments(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert new entities.json format to legacy assignments format expected by _build_candidates."""
    assignments = []
    for entity in payload.get("entities", []):
        keyframes = entity.get("keyframes") or []
        segs = entity.get("segments") or []
        # Use the entity's active slot range as the support window.
        # When visibility_ratio is high (entity present in most frames), extend to
        # the full frame_count range so downstream segment mapping covers all test frames.
        # support_frame_start/end only captures the peak-support sub-range, which
        # under-reports temporal coverage for sustained actions.
        frame_count = int(payload.get("frame_count", 0))
        visibility_ratio = float(entity.get("visibility_ratio", 0.0))
        if visibility_ratio >= 0.7 and frame_count > 0:
            support_start = 0
            support_end = frame_count - 1
        else:
            support_start = int(
                entity["support_frame_start"] if entity.get("support_frame_start") is not None
                else (min(keyframes) if keyframes else 0)
            )
            support_end = int(
                entity["support_frame_end"] if entity.get("support_frame_end") is not None
                else (max(keyframes) if keyframes else 0)
            )
        assignments.append({
            "entity_id": int(entity["id"]),
            "support_window": {
                "start": support_start,
                "end": support_end,
            },
            "phase_segments": {
                "moving":     [[int(s["t0"]), int(s["t1"])] for s in segs if s.get("label") == "moving"],
                "stationary": [[int(s["t0"]), int(s["t1"])] for s in segs if s.get("label") == "stationary"],
            },
            "entity_type": entity.get("entity_type", "object"),
            "quality":      float(entity.get("quality", 0.0)),
            "concept_tags": entity.get("concept_tags", []),
            "native_text": {
                "static_text": entity.get("static_text", ""),
                "global_desc": entity.get("global_desc", ""),
            },
            "qwen_text": {},
        })
    return {"assignments": assignments, "pair_interactions": payload.get("pair_interactions", [])}


def _build_candidates(
    assignments_payload: dict[str, Any],
    run_dir: Path,
    dataset_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Normalize new entities.json format → legacy assignments structure
    if "assignments" not in assignments_payload and "entities" in assignments_payload:
        assignments_payload = _entities_payload_to_assignments(assignments_payload)
    sample_times = _sample_time_values(run_dir)
    test_times = _test_time_values(run_dir, dataset_dir=dataset_dir)
    entity_phrase_map = _entity_phrase_map(run_dir)
    pair_rows = []
    query_segments_by_entity: dict[int, list[list[int]]] = {}
    for pair in assignments_payload.get("pair_interactions", []):
        test_segments = _map_segments_to_test(pair.get("contact_segments", []), sample_times, test_times)
        entity_a = int(pair["entity_a"])
        entity_b = int(pair["entity_b"])
        pair_rows.append(
            {
                "entity_a": entity_a,
                "entity_b": entity_b,
                "entity_a_phrase": str(entity_phrase_map.get(entity_a, {}).get("static_text", "")),
                "entity_b_phrase": str(entity_phrase_map.get(entity_b, {}).get("static_text", "")),
                "interaction_score": float(pair.get("interaction_score", 0.0)),
                "contact_ratio": float(pair.get("contact_ratio", 0.0)),
                "event_contrast": float(pair.get("event_contrast", 0.0)),
                "contact_segments_test": test_segments,
            }
        )
        if test_segments:
            query_segments_by_entity.setdefault(entity_a, []).extend(test_segments)
            query_segments_by_entity.setdefault(entity_b, []).extend(test_segments)

    rows = []
    for assignment in assignments_payload.get("assignments", []):
        entity_id = int(assignment["entity_id"])
        support = assignment.get("support_window", {})
        support_segments_test = _map_segments_to_test(_support_segment_from_window(support), sample_times, test_times)
        moving_segments_test = _map_segments_to_test(assignment.get("phase_segments", {}).get("moving", []), sample_times, test_times)
        stationary_segments_test = _map_segments_to_test(assignment.get("phase_segments", {}).get("stationary", []), sample_times, test_times)
        qwen_text = assignment.get("qwen_text", {})
        proposal_entity = entity_phrase_map.get(entity_id, {})
        rows.append(
            {
                "id": entity_id,
                "proposal_alias": str(proposal_entity.get("proposal_alias", "")).strip(),
                "proposal_phase": str(proposal_entity.get("proposal_phase", "")).strip(),
                "proposal_variant": str(proposal_entity.get("proposal_variant", "")).strip(),
                "proposal_phrase": str(proposal_entity.get("static_text", "")).strip(),
                "proposal_desc": str(proposal_entity.get("global_desc", "")).strip(),
                "entity_type": assignment.get("entity_type"),
                "semantic_head": assignment.get("semantic_head"),
                "quality": float(assignment.get("quality", 0.0)),
                "query_relevant_segments_test": _merge_ranges(query_segments_by_entity.get(entity_id, [])),
                "support_segments_test": support_segments_test,
                "moving_segments_test": moving_segments_test,
                "stationary_segments_test": stationary_segments_test,
                "global_desc": qwen_text.get("global_desc") or assignment.get("native_text", {}).get("global_desc", ""),
                "static_text": qwen_text.get("static_text") or assignment.get("native_text", {}).get("static_text", ""),
                "concept_tags": assignment.get("concept_tags", [])[:12],
                "interaction_partners": assignment.get("interaction_partners", [])[:4],
            }
        )
    pair_rows.sort(key=lambda item: (-float(item["interaction_score"]), int(item["entity_a"]), int(item["entity_b"])))
    return rows, pair_rows


def _normalize_selected(raw_payload: dict[str, Any], valid_ids: set[int], query: str) -> dict[str, Any]:
    selected_rows = []
    for item in raw_payload.get("selected", []):
        try:
            entity_id = int(item["id"])
        except Exception:
            continue
        if entity_id not in valid_ids:
            continue
        segments = []
        for segment in item.get("segments", []):
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                continue
            start = int(segment[0])
            end = int(segment[1])
            if end < start:
                start, end = end, start
            segments.append([start, end])
        selected_rows.append(
            {
                "id": entity_id,
                "role": "entity",
                "confidence": 1.0,
                "reason": " ".join(str(item.get("reason", "")).strip().split()),
                "segments": segments,
            }
        )
    return {
        "query": query,
        "selected": selected_rows,
        "empty": bool(raw_payload.get("empty", False)) and not selected_rows,
        "notes": " ".join(str(raw_payload.get("notes", "")).strip().split()),
    }


def _compose_phrase_grounded_selection(
    *,
    query: str,
    query_plan_payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    pair_candidates: list[dict[str, Any]],
    test_times: np.ndarray,
    tracks_payload: dict[str, Any] | None,
    raw_phrase_payload: dict[str, Any] | None = None,
    raw_output: str = "",
) -> dict[str, Any]:
    qwen_subjects = [str(item).strip() for item in (raw_phrase_payload or {}).get("subject_phrases", []) if str(item).strip()]
    qwen_successors = [str(item).strip() for item in (raw_phrase_payload or {}).get("successor_phrases", []) if str(item).strip()]
    plan_subjects = [str(item).strip() for item in query_plan_payload.get("query_subject_phrases", []) if str(item).strip()]
    plan_successors = [str(item).strip() for item in query_plan_payload.get("query_successor_phrases", []) if str(item).strip()]
    # If Qwen actually ran (raw_phrase_payload is not None) and explicitly returned empty subject_phrases,
    # respect that as "no matching entity" — do NOT fall back to plan_subjects.
    qwen_ran = raw_phrase_payload is not None
    query_state_mode = _query_state_mode(_normalize_phrase(query))
    # Also check query_plan_payload for explicit state mode (set by query plan editor)
    if query_state_mode is None:
        plan_state_mode = str(query_plan_payload.get("query_state_mode") or "").strip()
        if plan_state_mode:
            query_state_mode = plan_state_mode
    if query_state_mode is not None and len(plan_subjects) >= 2 and len(qwen_subjects) < len(plan_subjects):
        subject_phrases = plan_subjects
    elif qwen_ran and not qwen_subjects:
        # Qwen explicitly returned empty — treat as "entity does not satisfy query conditions"
        subject_phrases = []
    else:
        subject_phrases = qwen_subjects if qwen_ran else (qwen_subjects or plan_subjects)
    successor_phrases = [phrase for phrase in (qwen_successors or plan_successors) if _normalize_phrase(phrase) not in {_normalize_phrase(value) for value in subject_phrases}]
    # Early exit: Qwen explicitly returned no subjects → entity does not satisfy query
    if qwen_ran and not subject_phrases:
        notes = raw_phrase_payload.get("notes", "") if raw_phrase_payload else ""
        return {
            "query": query,
            "selected": [],
            "empty": True,
            "notes": f"Qwen returned no matching subject — entity does not satisfy query conditions. {notes}".strip(),
            "selection_mode": "qwen_plan_empty",
            "subject_phrases": [],
            "successor_phrases": [],
            "subject_phrase_matches": {},
            "successor_phrase_matches": {},
            "contact_pair": None,
            "raw_output": raw_output,
        }
    _allow_missing = (query_state_mode == "static")
    # Build normalized phrase → entity map from query_plan phrase_entities for richer matching
    _plan_phrase_entities: dict[str, dict[str, Any]] = {
        _normalize_phrase(str(e.get("phrase", ""))): e
        for e in query_plan_payload.get("phrase_entities", [])
        if isinstance(e, dict) and str(e.get("phrase", "")).strip()
    }
    subject_ids, subject_matches = _select_phrase_ids(
        candidates, subject_phrases, allow_missing=_allow_missing, phrase_entities=_plan_phrase_entities
    )
    successor_ids, successor_matches = (
        _select_phrase_ids(candidates, successor_phrases, allow_missing=True, phrase_entities=_plan_phrase_entities)
        if successor_phrases else ([], {})
    )
    query_norm = _normalize_phrase(query)
    split_keywords = ("broken", "pieces", "halves", "split", "cracked")
    intact_keywords = ("complete", "whole", "intact", "unbroken")
    if len(subject_ids) == 1:
        subject_id = int(subject_ids[0])
        subject_row = next((candidate for candidate in candidates if int(candidate["id"]) == subject_id), None)
        if subject_row is None:
            raise ValueError("Unable to resolve the selected subject entity.")
        normalized_subject = _normalize_phrase(subject_phrases[0]) if subject_phrases else ""
        matched_ids = subject_matches.get(normalized_subject, [subject_id]) if normalized_subject else [subject_id]
        matched_rows = [candidate for candidate in candidates if int(candidate["id"]) in {int(item) for item in matched_ids}]
        family_phrase = _normalize_phrase(subject_row.get("proposal_phrase", "") or subject_row.get("static_text", ""))
        if family_phrase:
            family_rows = [
                candidate
                for candidate in candidates
                if _normalize_phrase(candidate.get("proposal_phrase", "") or candidate.get("static_text", "")) == family_phrase
            ]
            if family_rows:
                matched_rows = family_rows

        def _variant_row(variant: str) -> dict[str, Any] | None:
            rows = [row for row in matched_rows if str(row.get("proposal_variant", "")).strip() == variant]
            if not rows:
                return None
            rows.sort(
                key=lambda item: (
                    -float(item.get("quality", 0.0)),
                    _first_range_start(item.get("support_segments_test", [])),
                    int(item.get("id", -1)),
                )
            )
            return rows[0]

        pre_split_row = _variant_row("pre_split")
        post_union_row = _variant_row("post_split_union")
        post_part_rows = [row for row in matched_rows if str(row.get("proposal_variant", "")).strip() == "post_split_part"]
        post_part_rows.sort(
            key=lambda item: (
                -float(item.get("quality", 0.0)),
                _first_range_start(item.get("support_segments_test", [])),
                int(item.get("id", -1)),
            )
        )
        plan_start_frame = _query_plan_boundary_test_index(
            query_plan_payload,
            test_times=test_times,
            boundary_kind="start",
            phrase=str(subject_row.get("proposal_phrase", "") or subject_row.get("static_text", "") or subject_phrases[0]),
        )
        plan_end_frame = _query_plan_boundary_test_index(
            query_plan_payload,
            test_times=test_times,
            boundary_kind="end",
            phrase=str(subject_row.get("proposal_phrase", "") or subject_row.get("static_text", "") or subject_phrases[0]),
        )
        split_frame = _track_split_start_frame(
            tracks_payload,
            phrase=str(subject_row.get("proposal_phrase", "") or subject_row.get("static_text", "") or subject_phrases[0]),
            test_times=test_times,
            start_frame=0,
        )
        subject_phrase_key = str(
            subject_row.get("proposal_phrase", "") or subject_row.get("static_text", "") or subject_phrases[0]
        )
        track_state_segments, track_state_meta = _track_state_segments_test(
            query=query,
            query_plan_payload=query_plan_payload,
            tracks_payload=tracks_payload,
            phrase=subject_phrase_key,
            test_times=test_times,
        )
        selected_rows = []
        selection_source = "single_subject_support"

        def _row_segments(row: dict[str, Any], default_ranges: list[list[int]]) -> list[list[int]]:
            return _merge_ranges(
                row.get("support_segments_test", [])
                or row.get("query_relevant_segments_test", [])
                or row.get("moving_segments_test", [])
                or default_ranges
            )

        def _select_rows_for_target_segments(
            target_segments: list[list[int]],
            reason: str,
        ) -> list[dict[str, Any]]:
            chosen_rows = []
            for row in matched_rows:
                available_segments = _row_segments(
                    row,
                    [[0, int(len(test_times) - 1)]],
                )
                row_segments = (
                    _intersect_ranges(available_segments, target_segments, len(test_times))
                    if target_segments
                    else available_segments
                )
                if not row_segments:
                    continue
                chosen_rows.append(
                    {
                        "id": int(row["id"]),
                        "role": "entity",
                        "confidence": 1.0,
                        "reason": reason,
                        "segments": row_segments,
                    }
                )
            if chosen_rows:
                return chosen_rows
            fallback_segments = target_segments or _row_segments(
                subject_row,
                [[0, int(len(test_times) - 1)]],
            )
            if not fallback_segments:
                return []
            return [
                {
                    "id": int(subject_row["id"]),
                    "role": "entity",
                    "confidence": 1.0,
                    "reason": reason,
                    "segments": fallback_segments,
                }
            ]

        if any(keyword in query_norm for keyword in split_keywords):
            effective_start = plan_start_frame if plan_start_frame is not None else split_frame
            if effective_start is None:
                effective_start = 0
            if split_frame is None and plan_start_frame is not None:
                effective_start = max(0, int(effective_start) - max(2, int(round(0.08 * len(test_times)))))
            segment_ranges = [[int(effective_start), int(len(test_times) - 1)]]
            target_row: dict[str, Any] | None = None
            if post_union_row is not None:
                target_row = post_union_row
            elif post_part_rows:
                target_row = post_part_rows[0]
            else:
                for successor_id in successor_ids:
                    row = next((candidate for candidate in candidates if int(candidate["id"]) == int(successor_id)), None)
                    if row is not None:
                        target_row = row
                        break
            if target_row is None:
                target_row = subject_row
            selected_rows.append(
                {
                    "id": int(target_row["id"]),
                    "role": "entity",
                    "confidence": 1.0,
                    "reason": f"Selected post-split variant for subject phrase '{subject_phrases[0]}'.",
                    "segments": segment_ranges,
                }
            )
            selection_source = "single_subject_split_after"
        elif any(keyword in query_norm for keyword in intact_keywords):
            effective_end = plan_end_frame if plan_end_frame is not None else split_frame
            if effective_end is None:
                effective_end = len(test_times) - 1
            if split_frame is None and plan_end_frame is not None:
                effective_end = max(0, int(effective_end) - max(2, int(round(0.08 * len(test_times)))))
            target_row = pre_split_row or subject_row
            segment_ranges = [[0, int(max(0, int(effective_end)))]]
            selected_rows.append(
                {
                    "id": int(target_row["id"]),
                    "role": "entity",
                    "confidence": 1.0,
                    "reason": f"Selected pre-split variant for subject phrase '{subject_phrases[0]}'.",
                    "segments": segment_ranges,
                }
            )
            selection_source = "single_subject_split_before"
        elif split_frame is not None and (pre_split_row is not None or post_union_row is not None or post_part_rows):
            pre_end = max(0, int(split_frame) - 1)
            post_start = max(0, int(split_frame))
            if pre_split_row is not None and pre_end >= 0:
                pre_segments = _intersect_ranges(
                    _row_segments(pre_split_row, [[0, pre_end]]),
                    [[0, pre_end]],
                    len(test_times),
                ) or [[0, pre_end]]
                selected_rows.append(
                    {
                        "id": int(pre_split_row["id"]),
                        "role": "entity",
                        "confidence": 1.0,
                        "reason": f"Selected pre-split phase for subject phrase '{subject_phrases[0]}'.",
                        "segments": pre_segments,
                    }
                )
            post_rows: list[dict[str, Any]] = []
            if post_union_row is not None:
                post_rows.append(post_union_row)
            elif post_part_rows:
                post_rows.extend(post_part_rows[:2])
            if not post_rows:
                post_rows.append(subject_row)
            for row in post_rows:
                post_segments = _intersect_ranges(
                    _row_segments(row, [[post_start, int(len(test_times) - 1)]]),
                    [[post_start, int(len(test_times) - 1)]],
                    len(test_times),
                ) or [[post_start, int(len(test_times) - 1)]]
                selected_rows.append(
                    {
                        "id": int(row["id"]),
                        "role": "entity",
                        "confidence": 1.0,
                        "reason": f"Selected phase-aware composition for subject phrase '{subject_phrases[0]}'.",
                        "segments": post_segments,
                    }
                )
            selection_source = "single_subject_phase_composition"
        else:
            target_segments = track_state_segments
            if target_segments:
                state_mode = (track_state_meta or {}).get("state_mode", "state")
                selected_rows = _select_rows_for_target_segments(
                    target_segments,
                    f"Selected from track-derived {state_mode} support for subject phrase '{subject_phrases[0]}'.",
                )
                selection_source = f"single_subject_track_{state_mode}"
            else:
                selected_rows = _select_rows_for_target_segments(
                    [],
                    f"Selected from Qwen-planned subject phrase '{subject_phrases[0]}'.",
                )
                selection_source = "single_subject_support"
            if not selected_rows:
                segment_ranges = _row_segments(
                    subject_row,
                    [[0, int(len(test_times) - 1)]],
                )
                selected_rows = [
                    {
                        "id": subject_id,
                        "role": "entity",
                        "confidence": 1.0,
                        "reason": f"Selected from Qwen-planned subject phrase '{subject_phrases[0]}'.",
                        "segments": segment_ranges,
                    }
                ]
                selection_source = "single_subject_support"
        merged_selected_ranges = _merge_ranges(
            [
                [int(segment[0]), int(segment[1])]
                for row in selected_rows
                for segment in row.get("segments", [])
                if isinstance(segment, (list, tuple)) and len(segment) == 2
            ]
        )
        notes_parts = [
            f"Subjects={subject_phrases}",
            f"Successors={successor_phrases}",
            f"Selection source={selection_source}",
            f"Segments={merged_selected_ranges}",
        ]
        if split_frame is not None:
            notes_parts.append(f"Mask split frame={int(split_frame)}")
        if track_state_meta:
            notes_parts.append(f"Track state mode={track_state_meta.get('state_mode')}")
            if "threshold" in track_state_meta:
                notes_parts.append(f"Track state threshold={float(track_state_meta['threshold']):.4f}")
        if query_plan_payload.get("start_condition"):
            notes_parts.append(f"Start condition: {query_plan_payload['start_condition']}")
        if query_plan_payload.get("stop_condition"):
            notes_parts.append(f"Stop condition: {query_plan_payload['stop_condition']}")
        return {
            "query": query,
            "selected": selected_rows,
            "empty": False,
            "notes": "; ".join(notes_parts),
            "selection_mode": "qwen_plan_single_subject",
            "subject_phrases": subject_phrases,
            "successor_phrases": successor_phrases,
            "subject_phrase_matches": subject_matches,
            "successor_phrase_matches": successor_matches,
            "contact_pair": {
                "entity_a": subject_id,
                "entity_b": -1,
                "entity_a_phrase": str(subject_phrases[0]) if subject_phrases else "",
                "entity_b_phrase": "",
                "contact_segments_test": [],
                "source": selection_source,
            },
            "raw_output": raw_output,
        }
    pair_row = None
    pair_segments: list[list[int]] = []
    contact_source = "worldtube_pair"
    subject_rows = {
        int(candidate["id"]): candidate
        for candidate in candidates
        if int(candidate["id"]) in {int(value) for value in subject_ids}
    }

    def _available_segments_for_subject(entity_id: int) -> list[list[int]]:
        row = subject_rows.get(int(entity_id))
        if row is None:
            return []
        return _merge_ranges(
            row.get("support_segments_test", [])
            or row.get("query_relevant_segments_test", [])
            or row.get("moving_segments_test", [])
            or [[0, int(len(test_times) - 1)]]
        )

    def _state_pair_selection() -> dict[str, Any] | None:
        if query_state_mode not in {"dark", "light", "full", "empty", "above_midpoint"}:
            return None
        if not tracks_payload or len(subject_phrases) < 2:
            return None
        carrier_index = None
        for index, phrase in enumerate(subject_phrases):
            phrase_norm = _normalize_phrase(phrase)
            if any(token in phrase_norm for token in ("liquid", "coffee", "espresso", "water", "juice", "tea")):
                carrier_index = index
                break
        if carrier_index is None:
            carrier_index = len(subject_phrases) - 1
        carrier_phrase = str(subject_phrases[int(carrier_index)])
        carrier_segments, carrier_meta = _track_state_segments_test(
            query=query,
            query_plan_payload=query_plan_payload,
            tracks_payload=tracks_payload,
            phrase=carrier_phrase,
            test_times=test_times,
        )
        if not carrier_segments:
            return None

        selected_rows_local = []
        for entity_id, phrase in zip(subject_ids, subject_phrases):
            available_segments = _available_segments_for_subject(int(entity_id))
            row_segments = _intersect_ranges(available_segments, carrier_segments, len(test_times))
            if not row_segments:
                row_segments = carrier_segments
            selected_rows_local.append(
                {
                    "id": int(entity_id),
                    "role": "entity",
                    "confidence": 1.0,
                    "reason": f"Selected from carrier state '{query_state_mode}' driven by phrase '{carrier_phrase}'.",
                    "segments": row_segments,
                }
            )
        merged_selected_ranges = _merge_ranges(
            [
                [int(segment[0]), int(segment[1])]
                for row in selected_rows_local
                for segment in row.get("segments", [])
                if isinstance(segment, (list, tuple)) and len(segment) == 2
            ]
        )
        notes_parts = [
            f"Subjects={subject_phrases}",
            f"Successors={successor_phrases}",
            f"State carrier={carrier_phrase}",
            f"State mode={query_state_mode}",
            f"Segments={merged_selected_ranges}",
        ]
        if carrier_meta and "threshold" in carrier_meta:
            notes_parts.append(f"Carrier threshold={float(carrier_meta['threshold']):.4f}")
        return {
            "query": query,
            "selected": selected_rows_local,
            "empty": False,
            "notes": "; ".join(notes_parts),
            "selection_mode": "qwen_plan_pair_state_grounded",
            "subject_phrases": subject_phrases,
            "successor_phrases": successor_phrases,
            "subject_phrase_matches": subject_matches,
            "successor_phrase_matches": successor_matches,
            "contact_pair": {
                "entity_a": int(subject_ids[0]) if subject_ids else -1,
                "entity_b": int(subject_ids[1]) if len(subject_ids) > 1 else -1,
                "entity_a_phrase": str(subject_phrases[0]) if subject_phrases else "",
                "entity_b_phrase": str(subject_phrases[1]) if len(subject_phrases) > 1 else "",
                "contact_segments_test": carrier_segments,
                "source": f"state_carrier:{carrier_phrase}",
            },
            "raw_output": raw_output,
        }

    state_pair_payload = _state_pair_selection()
    if state_pair_payload is not None:
        return state_pair_payload

    # --- Static-throughout-video queries ---
    # When query_state_mode == "static", the subjects should be active for the full video.
    if query_state_mode == "static":
        # If no phrases mapped to entities (entitybank lacks named objects), use ALL candidates
        if not subject_ids and candidates:
            subject_ids = [int(c["id"]) for c in candidates]
            subject_phrases = [str(c.get("proposal_phrase") or c.get("static_text") or f"entity_{c['id']}") for c in candidates]
        selected_rows_static = []
        full_range = [[0, int(len(test_times) - 1)]]
        for entity_id, phrase in zip(subject_ids, subject_phrases):
            selected_rows_static.append({
                "id": int(entity_id),
                "role": "entity",
                "confidence": 1.0,
                "reason": f"Static-throughout-video query: entity '{phrase}' selected for full video range.",
                "segments": full_range,
            })
        return {
            "query": query,
            "selected": selected_rows_static,
            "empty": not bool(selected_rows_static),
            "notes": f"Static query: selected {len(selected_rows_static)} entities for full video [{full_range}]. Subjects={subject_phrases}",
            "selection_mode": "qwen_plan_static_full_video",
            "subject_phrases": subject_phrases,
            "successor_phrases": successor_phrases,
            "subject_phrase_matches": subject_matches,
            "successor_phrase_matches": successor_matches,
            "contact_pair": {
                "entity_a": int(subject_ids[0]) if subject_ids else -1,
                "entity_b": -1,
                "entity_a_phrase": str(subject_phrases[0]) if subject_phrases else "",
                "entity_b_phrase": "",
                "contact_segments_test": full_range,
                "source": "single_subject_track_static",
            },
            "raw_output": raw_output,
        }

    # --- Exclusion query: "all objects EXCEPT X" — no interaction required ---
    # For exclusion queries, each selected entity is active in its own support window.
    if _is_exclusion_query(_normalize_phrase(query)) and subject_ids:
        selected_rows_excl = []
        full_range = [[0, int(len(test_times) - 1)]]
        for entity_id, phrase in zip(subject_ids, subject_phrases):
            subject_row_excl = next(
                (candidate for candidate in candidates if int(candidate["id"]) == int(entity_id)), None
            )
            if subject_row_excl is not None:
                segs = _merge_ranges(
                    subject_row_excl.get("support_segments_test", [])
                    or subject_row_excl.get("query_relevant_segments_test", [])
                    or subject_row_excl.get("moving_segments_test", [])
                    or full_range
                )
            else:
                segs = full_range
            selected_rows_excl.append({
                "id": int(entity_id),
                "role": "entity",
                "confidence": 1.0,
                "reason": f"Exclusion query: all objects except excluded. Subject phrase='{phrase}'.",
                "segments": segs,
            })
        merged_excl = _merge_ranges([
            [int(s[0]), int(s[1])]
            for row in selected_rows_excl
            for s in row.get("segments", [])
            if isinstance(s, (list, tuple)) and len(s) == 2
        ])
        return {
            "query": query,
            "selected": selected_rows_excl,
            "empty": False,
            "notes": f"Exclusion query: selected {len(selected_rows_excl)} entities; Subjects={subject_phrases}; Segments={merged_excl}",
            "selection_mode": "qwen_plan_exclusion",
            "subject_phrases": subject_phrases,
            "successor_phrases": successor_phrases,
            "subject_phrase_matches": subject_matches,
            "successor_phrase_matches": successor_matches,
            "contact_pair": None,
            "raw_output": raw_output,
        }

    if tracks_payload and len(subject_phrases) >= 2:
        pair_segments = _track_contact_segments_test(
            tracks_payload,
            phrase_a=subject_phrases[0],
            phrase_b=subject_phrases[1],
            test_times=test_times,
        )
        if not pair_segments:
            raise ValueError("Grounded SAM3 tracks did not produce any usable contact window for the selected subject phrases.")
        contact_source = "grounded_sam2_masks"
    elif len(set(int(value) for value in subject_ids)) >= 2:
        pair_row = _choose_subject_pair(pair_candidates, subject_ids)
        pair_segments = _merge_ranges(pair_row.get("contact_segments_test", []))
        if not pair_segments:
            # No contact segments — fall back to the union of both entities' support windows
            print("[select] WARNING: empty pair contact segments; falling back to subject support union.")
            fallback_segs: list[list[int]] = []
            for eid in subject_ids:
                cand = next((c for c in candidates if int(c["id"]) == int(eid)), None)
                if cand is None:
                    continue
                fallback_segs.extend(
                    cand.get("support_segments_test", [])
                    or cand.get("query_relevant_segments_test", [])
                    or cand.get("moving_segments_test", [])
                    or [[0, int(len(test_times) - 1)]]
                )
            pair_segments = _merge_ranges(fallback_segs) or [[0, int(len(test_times) - 1)]]
            contact_source = "subject_support_union_fallback"
    else:
        # Single subject or no subject with no tracks — fall back to full range support segments
        if subject_ids:
            single_row = next((c for c in candidates if int(c["id"]) == int(subject_ids[0])), None)
            segs = (
                _merge_ranges(
                    single_row.get("support_segments_test", [])
                    or single_row.get("query_relevant_segments_test", [])
                    or single_row.get("moving_segments_test", [])
                    or [[0, int(len(test_times) - 1)]]
                )
                if single_row is not None
                else [[0, int(len(test_times) - 1)]]
            )
            return {
                "query": query,
                "selected": [{"id": int(subject_ids[0]), "role": "entity", "confidence": 1.0,
                               "reason": f"Single subject fallback (no pair/tracks); phrase='{subject_phrases[0] if subject_phrases else ''}'.",
                               "segments": segs}],
                "empty": False,
                "notes": f"Single subject, no pair/tracks; using support segments.",
                "selection_mode": "qwen_plan_single_subject_fallback",
                "subject_phrases": subject_phrases,
                "successor_phrases": successor_phrases,
                "subject_phrase_matches": subject_matches,
                "successor_phrase_matches": successor_matches,
                "contact_pair": {"entity_a": int(subject_ids[0]), "entity_b": -1,
                                 "entity_a_phrase": subject_phrases[0] if subject_phrases else "",
                                 "entity_b_phrase": "", "contact_segments_test": [], "source": "single_subject_support"},
                "raw_output": raw_output,
            }
        else:
            return {
                "query": query,
                "selected": [],
                "empty": True,
                "notes": "No subject entities matched; cannot determine contact window.",
                "selection_mode": "qwen_plan_empty",
                "subject_phrases": subject_phrases,
                "successor_phrases": successor_phrases,
                "subject_phrase_matches": subject_matches,
                "successor_phrase_matches": successor_matches,
                "contact_pair": None,
                "raw_output": raw_output,
            }
    start_frame = int(pair_segments[0][0])
    contact_end = int(pair_segments[-1][1])
    split_stop = None
    # Only apply mask-split-based stop if query plan has an explicit stop condition.
    # If stop_condition is empty, the action is continuous and split detection should not clip the window.
    plan_has_stop_condition = bool(str(query_plan_payload.get("stop_condition") or "").strip())
    if plan_has_stop_condition:
        for phrase in subject_phrases:
            split_frame = _track_split_start_frame(
                tracks_payload,
                phrase=phrase,
                test_times=test_times,
                start_frame=start_frame,
            )
            if split_frame is None or int(split_frame) < int(start_frame):
                continue
            split_stop = int(split_frame) if split_stop is None else min(int(split_stop), int(split_frame))
    if split_stop is not None:
        stop_frame = int(split_stop)
    else:
        stop_frame = _successor_stop_frame(candidates, successor_ids, start_frame=start_frame, contact_end=contact_end)
    base_start = int(start_frame)
    base_stop = int(max(stop_frame, start_frame))
    base_length = int(base_stop - base_start + 1)
    target_length = int(max(round(float(base_length) * TEMPORAL_EXPANSION_FACTOR), base_length))
    extra = max(0, target_length - base_length)
    extend_before = extra // 2
    extend_after = extra - extend_before
    segment = [
        int(max(0, base_start - extend_before)),
        int(min(len(test_times) - 1, base_stop + extend_after)),
    ]
    segment = _clip_segment_to_plan_window(segment, query_plan_payload=query_plan_payload, test_times=test_times)
    selected_rows = []
    for entity_id, phrase in zip(subject_ids, subject_phrases):
        selected_rows.append(
            {
                "id": int(entity_id),
                "role": "entity",
                "confidence": 1.0,
                "reason": f"Selected from Qwen-planned subject phrase '{phrase}'.",
                "segments": [segment],
            }
        )
    notes_parts = [
        f"Subjects={subject_phrases}",
        f"Successors={successor_phrases}",
        f"Contact source={contact_source}",
        f"Window={segment[0]}-{segment[1]}",
        f"Base window={base_start}-{base_stop}",
        f"Expansion factor={TEMPORAL_EXPANSION_FACTOR:.1f}x",
    ]
    if pair_row is not None:
        notes_parts.append(f"Used contact pair=({pair_row.get('entity_a_phrase')}, {pair_row.get('entity_b_phrase')})")
    if split_stop is not None:
        notes_parts.append(f"Mask split stop={int(split_stop)}")
    plan_start, plan_end = _query_plan_window_test_range(query_plan_payload, test_times)
    if plan_start is not None or plan_end is not None:
        notes_parts.append(f"Plan window={plan_start}-{plan_end}")
    if query_plan_payload.get("start_condition"):
        notes_parts.append(f"Start condition: {query_plan_payload['start_condition']}")
    if query_plan_payload.get("stop_condition"):
        notes_parts.append(f"Stop condition: {query_plan_payload['stop_condition']}")
    payload = {
        "query": query,
        "selected": selected_rows,
        "empty": not bool(selected_rows),
        "notes": "; ".join(notes_parts),
        "selection_mode": "qwen_plan_phrase_grounded",
        "subject_phrases": subject_phrases,
        "successor_phrases": successor_phrases,
        "subject_phrase_matches": subject_matches,
        "successor_phrase_matches": successor_matches,
        "contact_pair": {
            "entity_a": int(subject_ids[0]) if subject_ids else -1,
            "entity_b": int(subject_ids[1]) if len(subject_ids) > 1 else -1,
            "entity_a_phrase": str(subject_phrases[0]) if subject_phrases else "",
            "entity_b_phrase": str(subject_phrases[1]) if len(subject_phrases) > 1 else "",
            "contact_segments_test": pair_segments,
            "source": contact_source,
        },
        "raw_output": raw_output,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments-path", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--query-plan-path", default=None)
    parser.add_argument("--dataset-dir", default=None,
                        help="Fallback dataset dir when run_dir lacks config.yaml (e.g. scenes with only cfg_args).")
    args = parser.parse_args()

    assignments_path = Path(args.assignments_path)
    run_dir = assignments_path.parents[1]
    _dataset_dir_arg = Path(args.dataset_dir) if args.dataset_dir else None
    assignments_payload = _read_json(assignments_path)
    candidates, pair_candidates = _build_candidates(assignments_payload, run_dir=run_dir, dataset_dir=_dataset_dir_arg)
    valid_ids = {int(item["id"]) for item in candidates}
    query_plan_payload = _read_json(Path(args.query_plan_path)) if args.query_plan_path else {}
    tracks_payload = _resolve_query_tracks_payload(Path(args.query_plan_path)) if args.query_plan_path else None
    test_times = _test_time_values(run_dir, dataset_dir=_dataset_dir_arg)
    query = str(args.query).strip()
    raw_payload: dict[str, Any] | None = None
    raw_output = ""
    use_qwen_selection = os.environ.get("QUERY_USE_QWEN_SELECTION", "0") == "1"
    subject_phrases = query_plan_payload.get("subjects") or query_plan_payload.get("query_subject_phrases") or []
    if subject_phrases:
        if use_qwen_selection:
            prompt = PROMPT_TEMPLATE.format(
                query=query,
                query_plan_json=json.dumps(query_plan_payload, ensure_ascii=False, indent=2),
                candidate_json=json.dumps(candidates, ensure_ascii=False, indent=2),
                pair_json=json.dumps(pair_candidates[:16], ensure_ascii=False, indent=2),
                total_frames=int(test_times.shape[0]) if test_times.size else "unknown",
            )
            teacher = QwenQueryPlanner(_resolve_qwen_model(args.qwen_model))
            raw_payload, raw_output = teacher.generate_json(prompt=prompt, images=None)
        else:
            raw_payload = {
                "query": query,
                "subject_phrases": subject_phrases,
                "successor_phrases": query_plan_payload.get("query_successor_phrases", []),
                "notes": "Using query-plan phrases directly (Qwen selection disabled).",
            }
            raw_output = "query_plan_phrase_fallback"
        selection_payload = _compose_phrase_grounded_selection(
            query=query,
            query_plan_payload=query_plan_payload,
            candidates=candidates,
            pair_candidates=pair_candidates,
            test_times=test_times,
            tracks_payload=tracks_payload,
            raw_phrase_payload=raw_payload,
            raw_output=raw_output,
        )
    else:
        raw_payload = {
            "query": query,
            "subject_phrases": [],
            "successor_phrases": [],
            "notes": "No subject phrases in query plan; cannot select entity without Qwen.",
        }
        raw_output = "no_plan_phrases"
        selection_payload = _compose_phrase_grounded_selection(
            query=query,
            query_plan_payload=query_plan_payload,
            candidates=candidates,
            pair_candidates=pair_candidates,
            test_times=test_times,
            tracks_payload=tracks_payload,
            raw_phrase_payload=raw_payload,
            raw_output=raw_output,
        )
    track_state_mode = _selection_track_state_mode(selection_payload)
    if track_state_mode is not None:
        selection_payload["track_state_mode"] = track_state_mode
    _write_json(Path(args.output_path), selection_payload)
    print(args.output_path)


if __name__ == "__main__":
    main()
