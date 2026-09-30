from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .source_images import resolve_dataset_image_entries

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _expand_qwen_model_path(path: Path) -> Path | None:
    path = Path(path)
    if (path / "config.json").exists():
        return path
    snapshots_dir = path / "snapshots"
    if snapshots_dir.exists():
        refs_main = path / "refs" / "main"
        if refs_main.exists():
            snapshot_name = refs_main.read_text(encoding="utf-8").strip()
            candidate = snapshots_dir / snapshot_name
            if (candidate / "config.json").exists():
                return candidate
        snapshot_candidates = sorted(
            candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir() and (candidate / "config.json").exists()
        )
        if snapshot_candidates:
            return snapshot_candidates[-1]
    return None


def _default_qwen_model() -> Path:
    env_override = os.environ.get("D4RESPLAT_QWEN_MODEL")
    candidates = [
        Path(env_override) if env_override else None,
        _PROJECT_ROOT / "Qwen3-VL-8B-Instruct",
        _PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct",
        _PROJECT_ROOT.parent / "hyperGS" / "models--Qwen--Qwen3-VL-8B-Instruct",
        Path("/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = _expand_qwen_model_path(candidate)
        if resolved is not None:
            return resolved
    return candidates[0] if candidates[0] is not None else (_PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct")


DEFAULT_QWEN_MODEL = _default_qwen_model()


# ---------------------------------------------------------------------------
# Scene knowledge base helpers
# ---------------------------------------------------------------------------

def _load_scene_knowledge(dataset_dir: Path) -> dict[str, Any] | None:
    """Load scene_knowledge.json from dataset_dir if it exists."""
    kb_path = Path(dataset_dir) / "scene_knowledge.json"
    if not kb_path.exists():
        return None
    try:
        with open(kb_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[qwen_planner] warn: failed to load KB {kb_path}: {exc}")
        return None


def _format_kb_context(kb: dict[str, Any]) -> str:
    """Format knowledge base entities into a compact prompt context string."""
    lines: list[str] = []
    for ent in kb.get("entities", []):
        attrs = ent.get("attributes", {})
        attr_str = ", ".join(v for v in [
            attrs.get("color", ""), attrs.get("material", ""),
            attrs.get("shape", ""), attrs.get("texture", ""),
        ] if v)
        pos = ent.get("spatial_position", "")
        pos_str = f" @ {pos}" if pos else ""
        motion_descs = list(dict.fromkeys(
            m["description"] for m in ent.get("motions", [])
            if m.get("description")
        ))
        # deduplicate very similar motion descriptions (keep first 3)
        motion_str = "; ".join(motion_descs[:3])
        lines.append(f'- "{ent["name"]}" [{attr_str}]{pos_str}')
        if motion_str:
            lines.append(f'    actions seen: {motion_str}')
    return "\n".join(lines)


def _kb_lookup_time_window(
    matched_entities: list[str],
    kb: dict[str, Any],
) -> dict[str, Any] | None:
    """Return existence_window from KB for the first matched entity."""
    entity_map = {e["name"]: e for e in kb.get("entities", [])}
    for name in matched_entities:
        ent = entity_map.get(name)
        if ent and ent.get("existence_window"):
            w = ent["existence_window"]
            if len(w) >= 2:
                return {"start_frame": int(w[0]), "end_frame": int(w[1]),
                        "confidence": "kb_derived", "source": name}
    return None


SCENE_DESCRIPTION_TEMPLATE = """You are a 4D video scene analyst. You are given one frame from a video.
List the most important visible objects and describe the scene.

Return exactly one JSON object:
- objects: JSON array of up to 10 strings, each ≤12 words, describing one visible object with color/material/shape (e.g. ["brown wooden cutting board", "silver metal tongs", "hand holding cast iron pan lid"])
- scene_context: one sentence ≤20 words describing the overall scene and ongoing action

Rules:
- FIRST list important objects, prioritizing those being interacted with.
- Be specific.
- Output valid JSON only.
"""

COARSE_QUERY_TEMPLATE = """You are a 4D video understanding assistant. You are given a natural language query about a video scene.

Scene context (objects visible in the video):
{scene_context}

Query:
{query}

Return exactly one JSON object:
- query: original query
- subjects: JSON array of visually grounded noun phrases, one per distinct physical object the query refers to. Use the scene context above to resolve implicit references. Each entry is a single object only.
- action: short action phrase describing what the subject(s) do, or null if the query is about static objects
- notes: short string

Rules:
- Each element of subjects must refer to exactly one physical object.
- subjects entries are VISUALLY RICH GROUNDING PHRASES designed to help SAM/GroundingDINO locate the exact object in the video. Build each entry by combining three layers:
  (1) NOUN TYPE from the query (e.g. "tray", "coaster", "glass cup")
  (2) VISUAL ATTRIBUTES from scene context: color, material, texture, shape (e.g. "bamboo wicker", "braided straw", "clear transparent", "round flat")
  (3) SPATIAL POSITION in the frame: use terms like "lower left", "bottom of frame", "upper right", "center", "beneath X", "on top of Y", "in front of Z"
  Combine them into a concise descriptive phrase of 4–10 words. Examples:
    query "the bottle" + scene "clear glass bottle on table center" → "clear glass bottle on table center"
    query "the plate" + scene "white ceramic plate lower frame" → "white ceramic plate lower frame"
    query "the book" + scene "red hardcover book upper left" → "red hardcover book upper left"
    query "the pen" + scene "blue ballpoint pen next to notebook right" → "blue ballpoint pen next to notebook right"
  Do NOT use ONLY the bare noun — always add at least one visual attribute AND one spatial hint. Maximum 12 words per entry.
- Do NOT include action verbs in subjects entries — subjects describe what the object IS and WHERE it is, not what it does. Exception: relational descriptors ("supporting", "holding", "under", "on top of") are allowed to disambiguate spatial relationships.
- FILL-LEVEL QUERIES: When the query describes a fill or quantity state, map it to a precise visual descriptor — do NOT copy spatial measurement phrases verbatim. Use the following mapping strictly:
  * "empty" / "without liquid" / "unfilled" / "no liquid" → "[object] empty [position]" (e.g. "empty bottle center"). NEVER omit "empty" for empty-state queries. For transparent/clear containers that are hard to see when empty, include the specific support device as spatial hint (e.g. "clear glass cup empty on scale center", "clear bottle empty on tray right") — this helps grounding models locate the transparent object.
  * "full" / "completely filled" / "completely full" → "[object] full of liquid" (e.g. "bottle full of liquid")
  * "above midpoint" / "above the midpoint" / "more than half" / "over half" / "over halfway" → "[object] more than half full [position]" (e.g. "bottle more than half full center"). NEVER use "half full" for above-midpoint queries — "half full" means exactly 50%, which is semantically wrong. ALWAYS append the spatial position from rule 3 — omitting position causes SAM to detect the wrong object.
  * "half full" / "halfway" / "exactly half" → "[object] half full [position]" (e.g. "bottle half full center")
  * "below midpoint" / "below the midpoint" / "less than half" → "[object] less than half full [position]"
  * "partially filled" / "some liquid" / "partial" → "[object] partially filled [position]"
  * "nearly empty" / "almost empty" → "[object] nearly empty [position]"
  Never output spatial measurements like "above midpoint center" or "filled to X level" in subjects — use only the mapped natural fill phrases above.
  IMPORTANT: Fill-level subjects still MUST follow rule 3 (SPATIAL POSITION) — always append a position hint from context (e.g. "center", "right side", "left of frame"). Example: "bottle more than half full center", NOT "bottle more than half full".
- For hand queries, follow these rules strictly:
  * If the query word is "hands" or "both hands" (plural, no further qualifier), output exactly ["left hand", "right hand"] — always resolve to both camera-frame hands.
  * If the query refers to a specific hand by relation or negation (e.g. "the hand holding X", "the hand that does not hold X"), use scene context to resolve to exactly "left hand" or "right hand" based on camera-frame position. Output ONLY "left hand" or "right hand" — no additional description.
  * If the scene context does not contain enough information to resolve which hand (e.g. only one hand visible, or cannot determine left/right), output ["hand"] — never output [] for a hand query when a hand is clearly present in the scene.
- For reverse/hypothetical/counterfactual queries (e.g. "played in reverse", "without this item"): use scene context to identify the concrete object being referred to. For reverse queries, look for objects being actively held or manipulated (not the hands themselves). Do NOT reject based on current frame state — an object being gripped or positioned may be in mid-action.
- Near-synonym matching: query vocabulary and scene context vocabulary often differ. Use semantic/visual equivalence to match: "container" ≈ "vessel" ≈ "holder"; "cloth" ≈ "fabric piece" ≈ "mat"; "stand" ≈ "support" ≈ "platform"; "box" ≈ "case" ≈ "crate"; "rod" ≈ "stick" ≈ "bar". When the query noun is semantically the same as a scene object by a different name, treat them as matched and output the noun phrase from the query.
- Attribute mismatches (e.g. query says "red wooden chair" but scene has "black metal chair"): if the NOUN TYPE matches, use the visually correct attributes from scene context (not from the query) — accurate visual description helps grounding more than faithfully copying query wording.
- COLOR / CONTENT ATTRIBUTE QUERIES: When the query specifies a color or content attribute of an object or its contents (e.g. "light-colored liquid", "dark liquid", "red surface", "blue dye"), always include that color/content descriptor in the subjects phrase. Example: query "vessel containing light-colored liquid" → subjects=["vessel with light-colored liquid center"], NOT ["vessel center"]. This applies to both static and temporal queries.
- TEMPORAL STATE CHANGE QUERIES: If the action describes a visual state change of the subject itself (e.g. "becomes darker", "fills with liquid", "turns brown/cloudy", "breaks apart", "changes color/appearance", "becomes complete/whole", "is whole before breaking"), describe the subject in its TARGET STATE — the final/changed visual appearance — not the neutral or initial appearance. Include the target-state descriptor as a visual attribute in the subjects phrase. Examples:
    query "apple turning brown" → action="turning brown", subject="apple with brown surface center" (not "green apple")
    query "the vase is still intact" → action="is still intact (before breaking)", subject="complete unbroken vase center"
    query "water filling the bottle" → action="filling", subject="bottle nearly full of water center"
  This ensures SAM3.1 locates the object when it is in the TARGET state described by the query, not throughout the entire video.
- BEFORE-STATE QUERIES: When the query asks about a physical object in its INTACT / COMPLETE / ORIGINAL state before a physical change (e.g. "complete cookie", "whole cookie", "unbroken cookie", "intact object"), treat this as a before-state query:
  * Output EXACTLY ONE subjects entry describing the intact/complete object. Do NOT split into multiple subjects even if the object is simultaneously held by multiple hands in the scene.
  * Set action = "is whole (before breaking)" (or equivalent phrasing for the specific object).
  * Use the target-state descriptor in the subjects phrase to help SAM find frames where the object is still intact: e.g. "complete unbroken chocolate chip cookie center".
  * Examples:
      query "complete apple" → subjects=["complete unbruised apple center"], action="is whole (before decaying)"
      query "whole vase" → subjects=["intact unbroken ceramic vase center"], action="is whole (before shattering)"
- TRAP QUERIES (absent objects): Some queries describe objects that do NOT exist in this specific video. Output subjects: [] — signaling "nothing to find" — when ALL of the following hold:
  (a) The query specifies an attribute (color, material, clothing item, modifier) that you can CONFIRM is absent from both the video frame and scene context. Examples: query says "red wooden chair" but only metal chairs visible; "hand wearing purple glove" but no gloves visible; "green ceramic mug" but mug is white/porcelain; "person sitting" but no seated person visible.
  (b) There is NO plausible alternative object in the scene that the query could be referring to despite the attribute mismatch.
  Do NOT output subjects: [] just because an object is small, partially occluded, or temporarily absent — only when the described object is clearly impossible or non-existent in this scene. When uncertain, err on the side of extracting a subject.
- Output valid JSON only.
"""

KB_COARSE_QUERY_TEMPLATE = """You are a 4D video understanding assistant. You are given a natural language query about a video scene.

Scene knowledge base — objects CONFIRMED present in the video with their visual attributes and observed actions:
{kb_context}

Query:
{query}

Return exactly one JSON object:
- query: original query
- subjects: JSON array of the PRIMARY physical object(s) the query is asking about. Only include the object being directly queried — NOT reference objects used merely to describe a relationship. Extract noun phrases directly from the query text; do NOT add KB color/material/texture attributes. Keep each entry 1–4 words. You MAY prepend a 1–2 word spatial position BEFORE the object name when it helps SAM locate the object (e.g. "center wooden tray", "lower left hand").
- action: short action phrase or null if static object query
- empty: true if the query's description is CONFIRMED impossible in this scene, false otherwise
- matched_entities: JSON array of entity names from the knowledge base that this query refers to (exact names as listed above)
- notes: short string

Rules for subjects:
- SHORT and SIMPLE — SAM uses these as text prompts; keep them concise (1–6 words max)
- ONLY the primary subject. In relational queries "A holding B", "A supporting B", "A on top of B": subjects = [A], B must NOT be a separate entry
- Spatial position prefix: prepend position BEFORE the noun when helpful: "center wooden tray", "top hand". Position word comes first.
- Examples:
  * "The hand holding the ceramic mug" → ["hand"]
  * "The wooden shelf supporting the book" → ["wooden shelf"]
  * "The mat under the bowl" → ["mat"]
  * "The bowl on the table" → ["bowl"]
- For hand queries: always use "hand" — do NOT resolve to "left hand" or "right hand" unless the query explicitly names left or right
- Do NOT include action verbs as standalone subjects
- Do NOT add KB color/material/texture attributes to subjects UNLESS the query is a temporal state change query (see below)
- TEMPORAL STATE CHANGE QUERIES: If the action describes a visual state change of the subject itself (e.g. "becomes darker", "fills with liquid", "turns brown", "breaks apart", "changes color/appearance", "is whole before breaking"), you MUST describe the subject in its TARGET STATE by appending 1–3 words for the target visual appearance. Examples:
    query "apple turning brown" → action="turning brown", subject="apple brown surface" (not "apple")
    query "the vase is still intact" → action="is still intact", subject="complete vase" (not "vase")
    query "water filling the bottle" → action="filling", subject="bottle full water"
  This ensures SAM3.1 locates the object at the MOMENT it is in the queried state, not throughout the entire video.
- BEFORE-STATE QUERIES: When the query asks about an object in its INTACT / COMPLETE / ORIGINAL state before a physical change (e.g. "complete cookie", "whole cookie", "unbroken cookie"), output EXACTLY ONE subjects entry for the intact object. Do NOT split into multiple subjects even if the object is held by multiple hands. Set action = "is whole (before breaking)". Examples:
    query "complete apple" → subjects=["complete apple"], action="is whole (before decaying)"
    query "whole vase" → subjects=["intact vase"], action="is whole (before shattering)"
- FILL-LEVEL QUERIES: For queries describing fill/quantity states, use precise visual fill descriptors (keep under 6 words):
  * "empty" / "no liquid" → "empty [object]" (e.g. "empty glass cup"). NEVER omit "empty".
  * "full" / "completely filled" → "glass cup full" or "[object] full liquid"
  * "above midpoint" / "more than half" / "over half" → "[position] [object] more than half full" (e.g. "center glass cup more than half full"). NEVER use "half full". Always prepend the spatial position — omitting it causes SAM to detect the wrong object.
  * "half full" / "halfway" → "[position] [object] half full" (e.g. "center glass cup half full")
  * "below midpoint" / "less than half" → "[position] [object] less than half full"
  * "partially filled" → "[position] [object] partial liquid"
  Do NOT output spatial measurements like "above midpoint" literally in subjects. Always include spatial position (prepend before object name).

Rules for empty:
- NAMING VARIATIONS are NOT traps: match objects by function/material/context, not exact wording. "woven mat" ≈ "fabric pad" ≈ "flat cloth piece"; "metal container" ≈ "metal vessel" ≈ "tin box"; "drink vessel" ≈ "cup" ≈ "glass" ≈ "mug"; "stand" ≈ "support" ≈ "platform". If the described object plausibly matches a KB entity, do NOT set empty=true.
- COLOR/MATERIAL trap: query specifies a color or material that CLEARLY contradicts KB with no plausible match (e.g. "purple wooden chair" but KB has only metal chairs; "green ceramic mug" but KB has only white mugs) → empty=true
- ACCESSORY trap: query mentions clothing or accessories (gloves, rings, sleeves) worn by a person that are NOT present in KB → empty=true
- NONEXISTENT OBJECT trap: query names an object type that has NO plausible KB match even accounting for naming variations (e.g. "dark metal tray" — no tray-like dark metal object in KB) → empty=true
- WRONG ROLE trap: only when KB EXPLICITLY shows the object does the OPPOSITE action with NO exceptions (e.g. "object A pouring liquid" — KB shows object A only receives liquid, never pours; "object B changing color" — KB shows only object C changes color, not object B; "object D moving" — KB shows object D is always static) → empty=true. Do NOT apply this trap to approximate spatial/proximity descriptions ("pressed against", "close to", "touching", "next to") — the primary object still exists even if the exact spatial relationship is imprecise.
- IMPOSSIBLE CONFIGURATION trap: query describes a physical arrangement confirmed NEVER to occur in KB (e.g. "object A balanced on top of object B" — KB shows object A always rests on a fixed surface, never on object B) → empty=true. Do NOT apply to descriptions of objects that are genuinely nearby or interacting even if the phrasing is approximate (e.g. "pressed against" when objects are close but not literally pressed).
- Do NOT set empty=true for temporal queries about valid moments (e.g. "pitcher moving away", "pitcher after pouring", "pitcher before pouring", "metal cup before it starts pouring", "hand suspended in midair") — the object exists even if the exact temporal boundary is at the video edge or ambiguous.
- Do NOT set empty=true for multi-subject queries — even if one subject is uncertain, extract the subjects you CAN resolve. Only set empty=true when the PRIMARY described object has a CONFIRMED impossible attribute (wrong color, non-existent accessory, etc.).
- When uncertain → empty=false

Rules for matched_entities:
- List the KB entity names the query refers to using EXACT names from the knowledge base

- Output valid JSON only.
"""

FINE_QUERY_TEMPLATE = """You are a 4D video understanding assistant.
A coarse temporal window has been identified. You are given dense frames within that window.
Find the precise start and end frame indices for the query event.

Query:
{query}

Subject: {subject}
Action:  {action}

Dense frames in candidate window (slot → frame_index):
{frame_summary}

Return exactly one JSON object:
- query: original query
- start_frame: integer frame_index of the first active frame, or null
- end_frame: integer frame_index of the last active frame, or null
- start_slot: 0-based slot index for start_frame in this window
- end_slot: 0-based slot index for end_frame in this window
- confidence: "high", "medium", or "low"
- notes: short string

Rules:
- Use frame_index values (not slot indices) for start_frame and end_frame.
- start_frame = first frame where the event semantically begins.
- end_frame   = last frame where the event is still active.
- Prefer slightly earlier start and slightly later end when uncertain.
- Output valid JSON only.
"""

SCAN_QUERY_TEMPLATE = """You are a 4D video understanding assistant.
You are scanning a short video segment to find a specific event.

Query:
{query}

Subject: {subject}
Action:  {action}

Frames in this segment (slot → frame_index):
{frame_summary}

Return exactly one JSON object:
- is_active: true if the query event occurs in this segment, false otherwise
- start_frame: integer frame_index of the first active frame in this segment, or null
- end_frame: integer frame_index of the last active frame in this segment, or null
- confidence: "high", "medium", or "low"
- notes: short string

Rules:
- is_active=true only if you can clearly see the event happening in these frames.
- Use frame_index values for start_frame and end_frame.
- Output valid JSON only.
"""


# Standard template: used for threshold/discrete states (e.g. "full glass cup", "empty glass cup").
STATE_CLASSIFY_TEMPLATE = """You are analyzing a single video frame.

Query: {query}
Target state: {state_description}

Is the target state visible in this frame?
Reply with one word only: YES or NO."""

# Fill-level template: for queries about a liquid fill threshold (e.g. "full glass cup",
# "above midpoint"). Fires when the liquid level clearly reaches or approaches the described
# threshold — looser than STRICT ("must be completely full") but tighter than TRANSITION
# ("any change from empty counts").
STATE_CLASSIFY_TEMPLATE_FILL_LEVEL = """You are analyzing a single video frame.

Query: {query}
Container: {state_description}

Is the container in this frame filled with liquid — does the liquid level approach or exceed the halfway point?
Answer YES if the container has a noticeable amount of liquid, even if not yet at the halfway point — as long as it is clearly not empty.
Answer NO only if the container is empty or has only a very small trace of liquid.
Reply with one word only: YES or NO."""

# Transition template: used when the action is a gradual/monotonic change (e.g. "become darker",
# "breaks apart"). Asks whether the transformation has AT LEAST BEGUN, so the window starts as
# soon as any progress is visible rather than only at the final/peak state.
STATE_CLASSIFY_TEMPLATE_TRANSITION = """You are analyzing a single video frame.

Query: {query}
Target state: {state_description}

Has the transformation described in the query at least partially started in this frame?
Answer YES if the object shows any visible departure from its original intact appearance — even early-stage or partial changes count.
Answer NO only if the object appears completely unchanged from its original state.
Reply with one word only: YES or NO."""

# Before-state template: used when the action describes an initial/before state
# (e.g. "is whole (before breaking)"). Asks whether the ORIGINAL state is still intact.
STATE_CLASSIFY_TEMPLATE_BEFORE = """You are analyzing a single video frame.

Query: {query}
Target state: {state_description}

Is the original/intact state still present in this frame — i.e., has the change described in the query NOT yet started?
Answer YES if the object still looks as described. Answer NO if the change has already begun or completed.
Reply with one word only: YES or NO."""


# NOTE: old multi-pass (coarse→fine→boundary-refine) functions removed;
#       precise temporal boundaries are now derived from SAM3.1 active-frame range.


def get_scene_description(
    dataset_dir: str | Path,
    teacher: "QwenQueryPlanner",
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run Qwen on the first frame of dataset_dir to get a scene description.

    Result is cached to cache_path (default: dataset_dir/qwen_plan/scene_description.json).
    Subsequent calls with the same cache_path skip the model call.
    """
    dataset_dir = Path(dataset_dir)
    if cache_path is None:
        cache_path = dataset_dir / "qwen_plan" / "scene_description.json"
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"[qwen_planner] scene description cache hit: {cache_path}")
        return _read_json(cache_path)

    entries = resolve_dataset_image_entries(dataset_dir)
    if not entries:
        return {"objects": [], "scene_context": ""}

    first_entry = entries[0]
    with Image.open(first_entry["image_path"]) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        longest = max(w, h)
        if longest > 896:
            scale = 896.0 / longest
            rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)
        first_frame = rgb.copy()

    print(f"[qwen_planner] running scene description for {dataset_dir.name} (frame {first_entry['frame_index']})")
    raw, _ = teacher.generate_json(prompt=SCENE_DESCRIPTION_TEMPLATE, images=[first_frame])

    objects = raw.get("objects") or []
    if isinstance(objects, list):
        objects = [str(o).strip() for o in objects if str(o).strip()]
    else:
        objects = []
    scene_context = " ".join(str(raw.get("scene_context", "")).strip().split())

    result = {"objects": objects, "scene_context": scene_context,
              "frame_index": int(first_entry["frame_index"])}
    _write_json(cache_path, result)
    print(f"[qwen_planner] scene description: {len(objects)} objects, context='{scene_context[:80]}'")
    return result


def _scene_context_text(scene_desc: dict[str, Any]) -> str:
    objects = scene_desc.get("objects") or []
    context = str(scene_desc.get("scene_context", "")).strip()
    parts = []
    if objects:
        parts.append("Objects: " + "; ".join(objects))
    if context:
        parts.append("Context: " + context)
    return "\n".join(parts) if parts else "No scene context available."


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _clean_llm_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_first_json(text: str) -> dict[str, Any]:
    cleaned = _clean_llm_text(text)
    if not cleaned:
        raise ValueError("Unable to parse JSON object from empty model output.")

    start_index = cleaned.find("{")
    if start_index < 0:
        raise ValueError(f"Unable to find top-level JSON object in model output: {text!r}")

    in_string = False
    escape = False
    depth = 0
    end_index: int | None = None
    for index in range(start_index, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                end_index = index + 1
                break

    if end_index is None:
        snippet = cleaned[start_index : min(len(cleaned), start_index + 400)]
        raise ValueError(f"Top-level JSON object is incomplete or truncated: {snippet!r}")

    candidate = cleaned[start_index:end_index]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse top-level JSON object from model output: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON payload must be an object, got {type(payload).__name__}.")
    return payload


def _resolve_qwen_model(explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if DEFAULT_QWEN_MODEL.exists():
        return DEFAULT_QWEN_MODEL
    raise FileNotFoundError(
        f"Unable to resolve a local Qwen model. Expected {DEFAULT_QWEN_MODEL} or pass an explicit path."
    )


def _import_transformers():
    try:
        import types
        import torch  # type: ignore

        if not hasattr(torch, "compiler"):
            torch.compiler = types.SimpleNamespace()  # type: ignore[attr-defined]
        if not hasattr(torch.compiler, "is_compiling"):
            torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        import torch.utils._pytree as _torch_pytree  # type: ignore

        if not hasattr(_torch_pytree, "register_pytree_node"):
            fallback = getattr(_torch_pytree, "_register_pytree_node", None)
            if fallback is not None:
                def _compat_register_pytree_node(typ, flatten_fn, unflatten_fn, **kwargs):
                    supported_kwargs = {}
                    if "to_dumpable_context" in kwargs:
                        supported_kwargs["to_dumpable_context"] = kwargs["to_dumpable_context"]
                    if "from_dumpable_context" in kwargs:
                        supported_kwargs["from_dumpable_context"] = kwargs["from_dumpable_context"]
                    return fallback(typ, flatten_fn, unflatten_fn, **supported_kwargs)

                _torch_pytree.register_pytree_node = _compat_register_pytree_node  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        import transformers  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependency 'transformers'. Install it in the query-planning environment."
        ) from exc
    return transformers


def _qwen_model_load_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    try:
        import flash_attn  # type: ignore  # noqa: F401
        kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gib = max(int(free_bytes // (1024 ** 3)), 1)
            total_gib = int(total_bytes // (1024 ** 3))
            gpu_budget = min(max(free_gib - 2, 8), total_gib - 2)
            max_memory = {index: f"{gpu_budget}GiB" for index in range(torch.cuda.device_count())}
            max_memory["cpu"] = "160GiB"
            kwargs["max_memory"] = max_memory
    except Exception:
        pass
    return kwargs


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _subsample_entries(entries: list[dict[str, Any]], frame_subsample_stride: int) -> list[dict[str, Any]]:
    stride = max(int(frame_subsample_stride), 1)
    sampled = entries[::stride]
    if entries and sampled and sampled[-1]["frame_index"] != entries[-1]["frame_index"]:
        sampled.append(entries[-1])
    return sampled


def _sample_context_entries(entries: list[dict[str, Any]], num_sampled_frames: int) -> list[dict[str, Any]]:
    if not entries:
        raise ValueError("No image entries available for Qwen query planning.")
    count = max(int(num_sampled_frames), 1)
    if len(entries) <= count:
        return list(entries)
    indices = np.linspace(0, len(entries) - 1, num=count, dtype=np.int32)
    return [entries[int(index)] for index in indices.tolist()]


def _load_context_images(
    dataset_dir: Path,
    frame_subsample_stride: int,
    num_sampled_frames: int,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    sampled_entries = _load_subsampled_entries(dataset_dir, frame_subsample_stride=frame_subsample_stride)
    context_entries = _sample_context_entries(sampled_entries, num_sampled_frames=num_sampled_frames)
    images = _load_images_for_entries(context_entries)
    return context_entries, images


def _load_subsampled_entries(dataset_dir: Path, frame_subsample_stride: int) -> list[dict[str, Any]]:
    all_entries = resolve_dataset_image_entries(dataset_dir)
    return _subsample_entries(all_entries, frame_subsample_stride=frame_subsample_stride)


def _load_images_for_entries(entries: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for entry in entries:
        with Image.open(entry["image_path"]) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            longest = max(width, height)
            if longest > 896:
                scale = 896.0 / float(longest)
                rgb = rgb.resize(
                    (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                    Image.Resampling.BICUBIC,
                )
            images.append(rgb)
    return images


def _frame_summary(entries: list[dict[str, Any]]) -> str:
    summary = [
        {
            "slot": int(index),
            "frame_index": int(entry["frame_index"]),
            "image_id": str(entry["image_id"]),
            "time_value": round(float(entry["time_value"]), 6),
        }
        for index, entry in enumerate(entries)
    ]
    return json.dumps(summary, ensure_ascii=False)


def _normalize_phrase_list(values: Any) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        phrase = " ".join(str(value).strip().lower().split())
        if not phrase:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        normalized.append(phrase)
    return normalized


class QwenQueryPlanner:
    def __init__(self, model_name_or_path: str | Path):
        transformers = _import_transformers()
        processor_cls = getattr(transformers, "AutoProcessor", None)
        if processor_cls is None:
            raise RuntimeError("transformers.AutoProcessor is unavailable.")

        model_cls = None
        for candidate in (
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
        ):
            model_cls = getattr(transformers, candidate, None)
            if model_cls is not None:
                break
        if model_cls is None:
            raise RuntimeError("Unable to find a compatible Qwen vision-language model class.")

        self.processor = processor_cls.from_pretrained(str(model_name_or_path), trust_remote_code=True)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        elif hasattr(self.processor, "padding_side"):
            self.processor.padding_side = "left"
        self.model = model_cls.from_pretrained(
            str(model_name_or_path),
            **_qwen_model_load_kwargs(),
        )

    def _format_text(self, prompt: str, images: list[Image.Image]) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": prompt}],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate_json(self, prompt: str, images: list[Image.Image] | None = None, max_new_tokens: int = 512) -> tuple[dict[str, Any], str]:
        images = images or []
        text = self._format_text(prompt, images)
        processor_kwargs: dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        model_inputs = self.processor(**processor_kwargs)
        model_inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}
        generated = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = generated[:, model_inputs["input_ids"].shape[1]:]
        output = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return _extract_first_json(output), output

    def generate_text(self, prompt: str, images: list[Image.Image] | None = None, max_new_tokens: int = 8) -> str:
        """Generate a short raw text response (no JSON parsing)."""
        images = images or []
        text = self._format_text(prompt, images)
        processor_kwargs: dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        model_inputs = self.processor(**processor_kwargs)
        model_inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}
        generated = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = generated[:, model_inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def classify_batch_with_image(
        self,
        prompts: list[str],
        image: "Image.Image",
        max_new_tokens: int = 8,
    ) -> list[str]:
        """One GPU forward pass: all prompts share the same image (true batching).

        Each prompt gets the same image prepended. Returns raw decoded strings.
        """
        texts = [self._format_text(p, [image]) for p in prompts]
        processor_kwargs: dict[str, Any] = {
            "text":   texts,
            "images": [image] * len(prompts),
            "padding": True,
            "return_tensors": "pt",
        }
        model_inputs = self.processor(**processor_kwargs)
        model_inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}
        generated = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = generated[:, model_inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def generate_json_batch(self, prompts: list[str]) -> list[tuple[dict[str, Any], str]]:
        """Process multiple text-only prompts in a single model.generate call."""
        texts = [self._format_text(p, []) for p in prompts]
        model_inputs = self.processor(text=texts, padding=True, return_tensors="pt")
        model_inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}
        generated = self.model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
        trimmed = generated[:, model_inputs["input_ids"].shape[1]:]
        outputs = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results: list[tuple[dict[str, Any], str]] = []
        for raw in outputs:
            raw = raw.strip()
            try:
                results.append((_extract_first_json(raw), raw))
            except Exception:
                results.append(({}, raw))
        return results

    def generate_json_batch_with_image(
        self,
        prompts: list[str],
        shared_image: "Image.Image",
    ) -> list[tuple[dict[str, Any], str]]:
        """Process multiple prompts each paired with the same image.

        Each prompt gets its own model.generate call (images prevent true batching
        across items with different image tensors), but the image is encoded once and
        the visual tokens are reused via the same PIL object reference.
        """
        results: list[tuple[dict[str, Any], str]] = []
        for prompt in prompts:
            try:
                result = self.generate_json(prompt=prompt, images=[shared_image], max_new_tokens=256)
                results.append(result)
            except Exception as exc:
                print(f"[qwen_planner] generate_json_batch_with_image error: {exc}")
                results.append(({}, ""))
        return results


def _clamp_to_entries(val: Any, entries: list[dict[str, Any]]) -> int | None:
    """Snap an arbitrary frame_index to the nearest valid frame_index in entries."""
    if val is None:
        return None
    try:
        v = int(val)
    except Exception:
        return None
    valid = {int(e["frame_index"]) for e in entries}
    if v in valid:
        return v
    return min(valid, key=lambda x: abs(x - v))


_STATIC_ACTION_TOKENS = frozenset({
    "stationary", "still", "static", "motionless", "immobile", "unmoving",
    "not moving", "not move", "never move", "never moves",
})

# Verbs that indicate gradual/monotonic physical transformation.
# Queries with these action verbs use STATE_CLASSIFY_TEMPLATE_TRANSITION so the window
# starts when the change first becomes visible, not only at the peak/final state.
_TRANSITION_ACTION_TOKENS = frozenset({
    "become", "becomes", "becoming",
    "darken", "darkens", "darkening",
    "lighten", "lightens", "lightening",
    "fill", "fills", "filling",
    "empty", "empties", "emptying",
    "break", "breaks", "breaking", "broken",
    "crack", "cracks", "cracking", "cracked",
    "split", "splits", "splitting", "splitted",
    "pour", "pours", "pouring",
    "melt", "melts", "melting",
    "dissolve", "dissolves", "dissolving",
    "burn", "burns", "burning",
    "fade", "fades", "fading",
    "spread", "spreads", "spreading",
    "grow", "grows", "growing",
    "shrink", "shrinks", "shrinking",
    "turn", "turns", "turning",
    "change", "changes", "changing",
    "transform", "transforms", "transforming",
})


def _classify_action(action: str, query: str = "") -> str:
    """Return 'static' | 'before' | 'transition' | 'dynamic' | 'null' for the given action string."""
    a = action.strip().lower() if action else ""
    if not a or a in ("null", "none"):
        return "null"
    if any(tok in a for tok in _STATIC_ACTION_TOKENS):
        return "static"
    # Before-state: action DESCRIBES the initial state, not a transformation.
    # Patterns: "is X (before Y)", "is whole", "is intact", "was X", "contain" (initial contents)
    if (a.startswith("is ") or a.startswith("was ")
            or "before " in a or " intact" in a or " whole" in a):
        return "before"
    # Transition: action contains a change/transformation verb AND query also describes the process.
    # If query describes a result state (not the process), use dynamic template instead.
    words = set(a.split())
    if words & _TRANSITION_ACTION_TOKENS or any(a.startswith(v) for v in _TRANSITION_ACTION_TOKENS):
        q = query.strip().lower()
        q_words = set(q.split())
        if q_words & _TRANSITION_ACTION_TOKENS or any(f" {v} " in f" {q} " for v in _TRANSITION_ACTION_TOKENS):
            return "transition"
        return "dynamic"
    # Fallback: treat as a generic dynamic action → standard template
    return "dynamic"

_DIRECTIONAL_HAND_PATTERNS = (
    re.compile(r"\bleft hand\b", flags=re.IGNORECASE),
    re.compile(r"\bright hand\b", flags=re.IGNORECASE),
)


def _is_fill_approaching(qtext: str) -> bool:
    """True for fill-level queries where the target state is reached gradually (full/above-midpoint).

    For these queries STATE_CLASSIFY_TEMPLATE is too strict and may produce no active region.
    """
    q = qtext.strip().lower()
    _full = (
        ("full" in q or "completely filled" in q)
        and "half" not in q
        and "partial" not in q
        and "empty" not in q
        and "nearly empty" not in q
    )
    _above = any(kw in q for kw in (
        "above midpoint", "above the midpoint", "midpoint of the cup",
        "more than half", "over half", "over halfway",
    ))
    return _full or _above


def scan_state_windows_batch(
    queries: list[dict[str, Any]],
    dataset_dir: Path,
    teacher: "QwenQueryPlanner",
    frame_stride: int = 10,
) -> list["tuple[int | None, int | None]"]:
    """Frame-first batch state scan: load each frame ONCE, classify ALL queries in one GPU pass.

    Args:
        queries: list of {"query": str, "subjects": [str], "action": str}
        dataset_dir: path to HyperNeRF scene directory
        teacher: loaded QwenQueryPlanner
        frame_stride: sample every N-th entry (by sorted time_id order)

    Returns:
        List of (start_time_id, end_time_id) per query. None means no active region.

    Total GPU passes = ceil(n_frames / frame_stride)  (vs n_queries × that without batching).
    """
    import json as _json

    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        return [(None, None)] * len(queries)

    meta: dict[str, Any] = _json.loads(meta_path.read_text())
    entries: list[tuple[int, str]] = sorted(
        [(int(v["time_id"]), k) for k, v in meta.items()], key=lambda x: x[0]
    )
    if not entries:
        return [(None, None)] * len(queries)

    # Build one prompt per query using action-type-specific templates:
    #
    # "transition" (become darker, breaks apart, filling → gradual change):
    #   template = STATE_CLASSIFY_TEMPLATE_TRANSITION  ("has it at least begun?")
    #   state_desc = subjects  (3a encodes the END state, e.g. "glass with dark liquid")
    #   Rationale: widest window — YES as soon as change first appears, not only at peak.
    #
    # "before" (is whole before breaking → query describes the INITIAL state):
    #   template = STATE_CLASSIFY_TEMPLATE_BEFORE  ("is original state still intact?")
    #   state_desc = query text  (subjects encode neutral object, not the before-state)
    #
    # "null" / "dynamic" (full glass cup, empty glass cup, or generic):
    #   template = STATE_CLASSIFY_TEMPLATE  (standard visibility check)
    #   state_desc = query text
    def _is_fill_approaching(qtext: str) -> bool:
        """True for fill-level queries where the target state is reached gradually.

        For these queries STATE_CLASSIFY_TEMPLATE is too strict ("is 100% full visible?")
        and produces no active region.  TRANSITION template ("has it at least begun?")
        gives a wider bisect window; the precise segment is identified later by
        fill_score analysis in select_qwen_query_entities.py.
        """
        q = qtext.strip().lower()
        # "full" without a hedging qualifier (exclude "half full", "partially full", "empty")
        _full = (
            ("full" in q or "completely filled" in q)
            and "half" not in q
            and "partial" not in q
            and "empty" not in q
            and "nearly empty" not in q
        )
        # "above midpoint" family
        _above = any(kw in q for kw in (
            "above midpoint", "above the midpoint", "midpoint of the cup",
            "more than half", "over half", "over halfway",
        ))
        return _full or _above

    prompts: list[str] = []
    for q in queries:
        action_type = _classify_action(q.get("action") or "", q.get("query", ""))
        query_text  = q.get("query", "")
        if action_type == "transition":
            state_desc = " ".join(q.get("subjects", [])) or query_text
            tpl = STATE_CLASSIFY_TEMPLATE_TRANSITION
        elif action_type == "before":
            state_desc = query_text
            tpl = STATE_CLASSIFY_TEMPLATE_BEFORE
        elif _is_fill_approaching(query_text):
            # Fill-level queries split into two sub-cases:
            #   "full" / "completely filled": use strict template + subjects.
            #     subjects encode the complete state; strict "is this visible?" gives a
            #     tight start (YES only when truly full). End forced to last_tid below.
            #   "above midpoint" / "more than half": use TRANSITION + subjects.
            #     strict template fires too late for threshold queries; TRANSITION
            #     ("has it started to look like this?") gives an earlier, correct start.
            _qn = query_text.strip().lower()
            _is_full_state = (
                ("full" in _qn or "completely filled" in _qn)
                and "half" not in _qn
                and "above" not in _qn
                and "partial" not in _qn
            )
            if _is_full_state:
                # Use query_text (e.g. "full glass cup") as state_desc — it reads as a
                # natural container description. Subjects (e.g. "clear glass cup full of
                # liquid center") are noun phrases that confuse the fill-level question.
                state_desc = query_text
                tpl = STATE_CLASSIFY_TEMPLATE_FILL_LEVEL
            else:
                state_desc = " ".join(q.get("subjects", [])) or query_text
                tpl = STATE_CLASSIFY_TEMPLATE_TRANSITION
        else:
            # null / dynamic / static — use subjects (richer 3a description) when available;
            # subjects give Qwen concrete visual attributes (color, material, fill level)
            # rather than abstract query words like "full glass cup" which map to never-true.
            state_desc = " ".join(q.get("subjects", [])) or query_text
            tpl = STATE_CLASSIFY_TEMPLATE
        prompts.append(tpl.format(query=query_text, state_description=state_desc))

    sampled = entries[::frame_stride]
    n_q = len(queries)
    print(f"[scan_state_batch] {len(sampled)} frames × {n_q} queries  (stride={frame_stride})")

    # active_matrix[frame_i][query_i] = bool
    active_matrix: list[list[bool]] = []

    for fi, (tid, iid) in enumerate(sampled):
        frame_path = _find_frame_image(iid, dataset_dir)
        if frame_path is None:
            active_matrix.append([False] * n_q)
            continue
        img = Image.open(frame_path).convert("RGB")
        raw_outputs = teacher.classify_batch_with_image(prompts, img, max_new_tokens=8)
        row = ["YES" in r.strip().upper()[:10] for r in raw_outputs]
        active_matrix.append(row)
        flags = " ".join("Y" if a else "N" for a in row)
        print(f"[scan_state_batch]   frame[{fi}] tid={tid}: [{flags}]")

    # Derive action types per query (used to determine end-boundary strategy)
    _action_types = [
        _classify_action(q.get("action") or "", q.get("query", "")) for q in queries
    ]
    last_tid = entries[-1][0]

    # Derive (start, end) per query as the longest contiguous YES block.
    # For TRANSITION queries the YES block ends when the action completes (not when
    # the object disappears), so we force end = last scene frame once any YES block
    # is found — the post-transition state (e.g. broken cookie) is still the target.
    results: list[tuple[int | None, int | None]] = []
    for qi in range(n_q):
        best_start: int | None = None
        best_end:   int | None = None
        best_len = 0
        run_start: int | None = None
        run_end:   int | None = None
        run_len = 0
        for fi, row in enumerate(active_matrix):
            tid = sampled[fi][0]
            if row[qi]:
                if run_start is None:
                    run_start = tid
                run_end = tid
                run_len += 1
                if run_len > best_len:
                    best_len, best_start, best_end = run_len, run_start, run_end
            else:
                run_start = run_end = None
                run_len = 0
        if best_start is not None and (
            _action_types[qi] == "transition"
            or _is_fill_approaching(queries[qi].get("query", ""))
        ):
            best_end = last_tid
        results.append((best_start, best_end))
        qid_info = queries[qi].get("query_id", f"q{qi}")
        print(f"[scan_state_batch]   {qid_info}: {results[-1]}")

    return results


def _action_to_state_mode(action: str) -> str | None:
    """Derive a query_state_mode string from Qwen's inferred action phrase."""
    action_lower = action.strip().lower()
    if any(token in action_lower for token in _STATIC_ACTION_TOKENS):
        return "static"
    return None


def _find_frame_image(image_id: str, dataset_dir: Path) -> "Path | None":
    for res in ("1x", "2x", "4x"):
        p = dataset_dir / "rgb" / res / f"{image_id}.png"
        if p.exists():
            return p
    return None


def _classify_frame_state(
    image_id: str,
    dataset_dir: Path,
    query: str,
    state_desc: str,
    teacher: "QwenQueryPlanner",
    template=None,
) -> bool | None:
    """Ask Qwen (with image) whether the target state is active in this frame.
    Returns True/False, or None if frame not found."""
    frame_path = _find_frame_image(image_id, dataset_dir)
    if frame_path is None:
        return None
    if template is None:
        template = STATE_CLASSIFY_TEMPLATE
    img = Image.open(frame_path).convert("RGB")
    prompt = template.format(query=query, state_description=state_desc)
    raw = teacher.generate_text(prompt=prompt, images=[img], max_new_tokens=8)
    return "YES" in raw.strip().upper()[:10]


def bisect_state_window(
    query: str,
    subjects: list[str],
    action: str,
    dataset_dir: Path,
    teacher: "QwenQueryPlanner",
    n_coarse: int = 10,
    frame_stride: int = 10,
) -> "tuple[int | None, int | None]":
    """Binary-search temporal window where the query's state condition is active.

    Algorithm:
      1. Coarse scan: classify n_coarse uniformly sampled frames → find YES region.
      2. Binary refine start/end, stopping when interval <= frame_stride entries.

    Returns (start_time_id, end_time_id) or (None, None) if no active region found.
    Total Qwen calls ≈ n_coarse + 2 * log2(n_frames / frame_stride).
    """
    import json as _json

    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[bisect_state] no metadata.json in {dataset_dir}")
        return None, None

    meta: dict[str, Any] = _json.loads(meta_path.read_text())
    entries: list[tuple[int, str]] = sorted(
        [(int(v["time_id"]), k) for k, v in meta.items()],
        key=lambda x: x[0],
    )
    if not entries:
        return None, None

    n = len(entries)
    action_type = _classify_action(action, query)
    if action_type == "transition":
        state_desc = " ".join(subjects) if subjects else query
        _bisect_template = STATE_CLASSIFY_TEMPLATE_TRANSITION
        # End boundary: object visibility (transition may be "done" but object persists)
        _bisect_end_template = STATE_CLASSIFY_TEMPLATE
        _end_state_desc = " ".join(subjects) if subjects else query
    elif action_type == "before":
        state_desc = query
        _bisect_template = STATE_CLASSIFY_TEMPLATE_BEFORE
        _bisect_end_template = STATE_CLASSIFY_TEMPLATE_BEFORE
        _end_state_desc = state_desc
    else:
        state_desc = " ".join(subjects) if subjects else query
        if action and action.lower() not in ("null", "none", ""):
            state_desc = f"{state_desc} — {action}"
        _bisect_template = STATE_CLASSIFY_TEMPLATE
        _bisect_end_template = STATE_CLASSIFY_TEMPLATE
        _end_state_desc = state_desc
    print(f"[bisect_state] action_type={action_type!r}, start_template=TRANSITION end_template=STANDARD" if action_type == "transition" else f"[bisect_state] action_type={action_type!r}")

    call_count = 0

    def classify(entry_idx: int, template=None, desc=None) -> bool:
        nonlocal call_count
        tid, iid = entries[entry_idx]
        result = _classify_frame_state(iid, dataset_dir, query,
                                       desc if desc is not None else state_desc,
                                       teacher,
                                       template=template if template is not None else _bisect_template)
        call_count += 1
        yesno = "YES" if result else ("NO" if result is False else "?")
        print(f"[bisect_state]   [{call_count}] entry[{entry_idx}] time_id={tid}: {yesno}")
        return bool(result)

    # ── Coarse scan ──────────────────────────────────────────────
    coarse_step = max(1, (n - 1) // (n_coarse - 1))
    coarse_indices: list[int] = list(range(0, n, coarse_step))
    if coarse_indices[-1] != n - 1:
        coarse_indices.append(n - 1)
    print(f"[bisect_state] coarse scan: {len(coarse_indices)} frames (n={n})")
    coarse_results: list[tuple[int, bool]] = [(idx, classify(idx)) for idx in coarse_indices]

    yes_positions = [i for i, (_, active) in enumerate(coarse_results) if active]
    if not yes_positions:
        print(f"[bisect_state] no active frames found in coarse scan")
        return None, None

    first_yes_pos = min(yes_positions)
    last_yes_pos  = max(yes_positions)
    first_yes_idx = coarse_results[first_yes_pos][0]
    last_yes_idx  = coarse_results[last_yes_pos][0]

    # ── Binary refine: start boundary ────────────────────────────
    lo = coarse_results[first_yes_pos - 1][0] if first_yes_pos > 0 else 0
    hi = first_yes_idx
    print(f"[bisect_state] refine start in entry range [{lo}, {hi}]")
    while hi - lo > frame_stride:
        mid = (lo + hi) // 2
        if classify(mid):
            hi = mid
        else:
            lo = mid
    start_tid = entries[hi][0]

    # ── Binary refine: end boundary ──────────────────────────────
    if action_type == "transition":
        # Transition persists until the object leaves the scene; use last frame.
        end_tid = entries[n - 1][0]
        print(f"[bisect_state] transition end: using last frame time_id={end_tid}")
    else:
        lo = last_yes_idx
        hi = coarse_results[last_yes_pos + 1][0] if last_yes_pos < len(coarse_results) - 1 else n - 1
        print(f"[bisect_state] refine end in entry range [{lo}, {hi}]")
        while hi - lo > frame_stride:
            mid = (lo + hi) // 2
            if classify(mid, template=_bisect_end_template, desc=_end_state_desc):
                lo = mid
            else:
                hi = mid
        end_tid = entries[lo][0]

    print(f"[bisect_state] result: [{start_tid}, {end_tid}]  (total Qwen calls: {call_count})")
    return start_tid, end_tid




def _strip_query_prefix(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    cleaned = re.sub(r"[.?!]+$", "", cleaned)
    cleaned = re.sub(
        r"^(find|locate|identify|show|select|track|get|give me|return)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _scene_hand_observations(scene_desc: dict[str, Any] | None) -> tuple[bool, bool, int]:
    objects = scene_desc.get("objects") if isinstance(scene_desc, dict) else []
    left_visible = False
    right_visible = False
    hand_count = 0
    for obj in objects or []:
        text = " ".join(str(obj).strip().lower().split())
        if "hand" not in text:
            continue
        hand_count += 1
        if "left hand" in text:
            left_visible = True
        if "right hand" in text:
            right_visible = True
    return left_visible, right_visible, hand_count


def _extract_hand_feature_phrase(query: str) -> str:
    cleaned = _strip_query_prefix(query)
    lowered = cleaned.lower()
    hand_pos = lowered.find("hand")
    if hand_pos < 0:
        return "hand"
    phrase = cleaned[hand_pos:]
    phrase = re.sub(r"^(hands?)\b", "hand", phrase, flags=re.IGNORECASE)
    phrase = phrase.strip(" ,;:.")
    words = phrase.split()
    if not words:
        return "hand"
    return " ".join(words[:12])


def _canonicalize_hand_subjects(
    subjects: list[str],
    query: str,
    scene_desc: dict[str, Any] | None = None,
) -> list[str]:
    left_visible, right_visible, hand_count = _scene_hand_observations(scene_desc)
    two_hands_resolved = left_visible and right_visible
    hand_feature_phrase = _extract_hand_feature_phrase(query)
    query_lower = " ".join(str(query).strip().lower().split())
    asks_both_hands = ("both hands" in query_lower) or bool(re.search(r"\bhands\b", query_lower))

    normalized: list[str] = []
    for subject in subjects:
        subj = " ".join(str(subject).strip().split())
        subj_lower = subj.lower()
        if subj_lower in {"left hand", "right hand"}:
            if asks_both_hands:
                normalized.append(subj_lower)
            elif hand_count <= 1:
                normalized.append("hand")
            elif two_hands_resolved:
                normalized.append(subj_lower)
            else:
                normalized.append(hand_feature_phrase)
            continue
        normalized.append(subj)

    deduped: list[str] = []
    seen: set[str] = set()
    for subject in normalized:
        key = subject.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(subject)
    return deduped


def _normalize_coarse(
    raw: dict[str, Any],
    coarse_entries: list[dict[str, Any]],
    query: str,
    scene_desc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_subjects = raw.get("subjects")
    if isinstance(raw_subjects, list):
        subjects = [" ".join(str(s).strip().split()) for s in raw_subjects if str(s).strip()]
    else:
        single = " ".join(str(raw.get("subject", raw_subjects or "")).strip().split())
        subjects = [single] if single else []
    subjects = _canonicalize_hand_subjects(subjects, query=query, scene_desc=scene_desc)
    action = " ".join(str(raw.get("action", "")).strip().split())

    # KB-aware fields
    empty = bool(raw.get("empty", False))
    if empty:
        subjects = []  # force empty prediction when Qwen confirms trap
    matched_entities: list[str] = []
    if isinstance(raw.get("matched_entities"), list):
        matched_entities = [str(e).strip() for e in raw["matched_entities"] if str(e).strip()]

    return {
        "query":            query,
        "subjects":         subjects,
        "action":           action,
        "empty":            empty,
        "matched_entities": matched_entities,
        "notes":            " ".join(str(raw.get("notes", "")).strip().split()),
    }


def _normalize_fine(raw: dict[str, Any], fine_entries: list[dict[str, Any]]) -> dict[str, Any]:
    start_frame = _clamp_to_entries(raw.get("start_frame"), fine_entries)
    end_frame   = _clamp_to_entries(raw.get("end_frame"),   fine_entries)
    if start_frame is not None and end_frame is not None and end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame
    return {
        "start_frame": start_frame,
        "end_frame":   end_frame,
        "confidence":  str(raw.get("confidence", "medium")).strip().lower(),
        "notes":       " ".join(str(raw.get("notes", "")).strip().split()),
    }


def _normalize_scan(raw: dict[str, Any], seg_entries: list[dict[str, Any]]) -> dict[str, Any]:
    is_active   = bool(raw.get("is_active", False))
    start_frame = _clamp_to_entries(raw.get("start_frame"), seg_entries) if is_active else None
    end_frame   = _clamp_to_entries(raw.get("end_frame"),   seg_entries) if is_active else None
    if start_frame is not None and end_frame is not None and end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame
    return {
        "is_active":   is_active,
        "start_frame": start_frame,
        "end_frame":   end_frame,
        "confidence":  str(raw.get("confidence", "low")).strip().lower(),
        "notes":       " ".join(str(raw.get("notes", "")).strip().split()),
    }


def _coarse_to_frame_range(
    coarse_start_slot: int | None,
    coarse_end_slot:   int | None,
    coarse_entries:    list[dict[str, Any]],
    all_entries:       list[dict[str, Any]],
    margin_ratio:      float = 0.10,
) -> tuple[int, int]:
    """Convert coarse slot indices to a frame-index range with margin, clamped to all_entries."""
    all_frame_indices = [int(e["frame_index"]) for e in all_entries]
    first_fi, last_fi = all_frame_indices[0], all_frame_indices[-1]
    margin = max(1, int(len(all_entries) * margin_ratio))

    if coarse_start_slot is not None and coarse_start_slot < len(coarse_entries):
        anchor_start = int(coarse_entries[coarse_start_slot]["frame_index"])
        # find position in all_entries and step back by margin
        pos_start = next((i for i, e in enumerate(all_entries) if int(e["frame_index"]) >= anchor_start), 0)
        fi_start  = all_frame_indices[max(0, pos_start - margin)]
    else:
        fi_start = first_fi

    if coarse_end_slot is not None and coarse_end_slot < len(coarse_entries):
        anchor_end = int(coarse_entries[coarse_end_slot]["frame_index"])
        pos_end = next((i for i, e in enumerate(all_entries) if int(e["frame_index"]) >= anchor_end), len(all_entries) - 1)
        fi_end  = all_frame_indices[min(len(all_entries) - 1, pos_end + margin)]
    else:
        fi_end = last_fi

    return fi_start, fi_end


def plan_query_entities(
    query: str,
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
    qwen_model: str | None = None,
    coarse_stride: int = 16,
    fine_frames: int = 24,       # kept for API compat, unused
    scan_seg_size: int = 16,     # kept for API compat, unused
    _teacher: "QwenQueryPlanner | None" = None,
) -> dict[str, Any]:
    """
    Coarse-only Qwen planner.

    Runs a single coarse pass (every `coarse_stride` frames) to extract subject,
    action, event_duration, and a rough seed window for SAM3.1.  Precise temporal
    boundaries are no longer derived here — they are determined downstream from
    SAM3.1 tracking output (active-frame range), which is more accurate and avoids
    the VRAM cost of a dense fine pass.

    Pass `_teacher` to reuse an already-loaded QwenQueryPlanner (batch mode).
    """
    query = " ".join(str(query).strip().split())
    if not query:
        raise ValueError("query must be non-empty")
    dataset_dir = Path(dataset_dir)

    teacher = _teacher if _teacher is not None else QwenQueryPlanner(_resolve_qwen_model(qwen_model))

    # ── Scene description (first frame, cached per dataset) ────────────────────
    scene_desc = get_scene_description(dataset_dir, teacher)

    # ── Load scene knowledge base if available ─────────────────────────────────
    kb = _load_scene_knowledge(dataset_dir)

    # ── Coarse pass ────────────────────────────────────────────────────────────
    if kb is not None:
        kb_ctx = _format_kb_context(kb)
        prompt = KB_COARSE_QUERY_TEMPLATE.format(query=query, kb_context=kb_ctx)
        print(f"[qwen_planner] KB-aware pass for query: '{query}'")
    else:
        prompt = COARSE_QUERY_TEMPLATE.format(
            query=query, scene_context=_scene_context_text(scene_desc)
        )
        print(f"[qwen_planner] text-only pass for query: '{query}'")

    coarse_raw, _ = teacher.generate_json(prompt=prompt, images=[])
    coarse = _normalize_coarse(coarse_raw, [], query, scene_desc=scene_desc)
    print(f"[qwen_planner] coarse → subjects={coarse['subjects']} "
          f"action='{coarse['action']}' empty={coarse['empty']} "
          f"matched={coarse['matched_entities']}")

    # ── Resolve time_window seed from KB ───────────────────────────────────────
    time_window: dict[str, Any] | None = None
    if kb is not None and coarse["matched_entities"]:
        time_window = _kb_lookup_time_window(coarse["matched_entities"], kb)

    # ── Assemble plan ──────────────────────────────────────────────────────────
    plan: dict[str, Any] = {
        "query":            query,
        "subjects":         coarse["subjects"],
        "action":           coarse["action"],
        "query_state_mode": _action_to_state_mode(coarse["action"]),
        "dataset_dir":      str(dataset_dir),
    }
    if coarse["empty"]:
        plan["negative_query"] = True
    if time_window is not None:
        plan["time_window"] = time_window

    if output_path is not None:
        _write_json(Path(output_path), plan)
    return plan


def plan_query_entities_batch(
    items: list[dict[str, Any]],
    qwen_model: str | None = None,
    batch_size: int = 8,
    _teacher: "QwenQueryPlanner | None" = None,
) -> list[dict[str, Any]]:
    """
    Batch version of plan_query_entities.

    Each item must have: query_text, dataset_dir, output_path (optional).
    Processes `batch_size` queries per model.generate call.
    Returns list of plan dicts in the same order as items.
    """
    teacher = _teacher if _teacher is not None else QwenQueryPlanner(_resolve_qwen_model(qwen_model))

    # ── Scene description + KB: one call per unique dataset_dir ───────────────
    unique_dataset_dirs = list(dict.fromkeys(
        str(Path(item["dataset_dir"]).resolve()) for item in items
    ))
    scene_desc_cache: dict[str, dict[str, Any]] = {}
    kb_cache: dict[str, dict[str, Any] | None] = {}
    for ddir in unique_dataset_dirs:
        scene_desc_cache[ddir] = get_scene_description(Path(ddir), teacher)
        kb_cache[ddir] = _load_scene_knowledge(Path(ddir))
        if kb_cache[ddir] is not None:
            print(f"[qwen_planner] KB loaded for {Path(ddir).name}: "
                  f"{len(kb_cache[ddir].get('entities', []))} entities")
        else:
            print(f"[qwen_planner] no KB for {Path(ddir).name}, using scene description")

    results: list[dict[str, Any]] = [{}] * len(items)
    total_queries = len(items)
    num_batches = (total_queries + batch_size - 1) // batch_size
    t_all_start = time.time()

    for chunk_start in range(0, total_queries, batch_size):
        chunk = items[chunk_start: chunk_start + batch_size]
        batch_idx = chunk_start // batch_size + 1

        prompts = []
        for item in chunk:
            ddir_key = str(Path(item["dataset_dir"]).resolve())
            kb = kb_cache.get(ddir_key)
            query = " ".join(str(item["query_text"]).strip().split())
            if kb is not None:
                prompts.append(KB_COARSE_QUERY_TEMPLATE.format(
                    query=query,
                    kb_context=_format_kb_context(kb),
                ))
            else:
                prompts.append(COARSE_QUERY_TEMPLATE.format(
                    query=query,
                    scene_context=_scene_context_text(
                        scene_desc_cache.get(ddir_key, {})
                    ),
                ))

        print(f"\n[qwen_planner] ── batch {batch_idx}/{num_batches}  ({len(chunk)} queries) ──")
        t_gen = time.time()
        batch_outputs = teacher.generate_json_batch(prompts)
        gen_elapsed = time.time() - t_gen
        print(f"[qwen_planner]   generate: {gen_elapsed:.2f}s  ({gen_elapsed/len(chunk):.2f}s/query)")

        for offset, (item, (coarse_raw, _)) in enumerate(zip(chunk, batch_outputs)):
            query = " ".join(str(item["query_text"]).strip().split())
            dataset_dir = Path(item["dataset_dir"])
            ddir_key = str(dataset_dir.resolve())
            scene_desc = scene_desc_cache.get(ddir_key, {})
            kb = kb_cache.get(ddir_key)

            coarse = _normalize_coarse(coarse_raw, [], query, scene_desc=scene_desc)
            print(f"[qwen_planner]   [{chunk_start+offset+1}/{total_queries}] {item.get('query_id','?')}: "
                  f"subjects={coarse['subjects']} action='{coarse['action']}' "
                  f"empty={coarse['empty']} matched={coarse['matched_entities']}")

            # resolve time_window seed from KB
            time_window: dict[str, Any] | None = None
            if kb is not None and coarse["matched_entities"]:
                time_window = _kb_lookup_time_window(coarse["matched_entities"], kb)

            plan: dict[str, Any] = {
                "query":            query,
                "subjects":         coarse["subjects"],
                "action":           coarse["action"],
                "query_state_mode": _action_to_state_mode(coarse["action"]),
                "dataset_dir":      str(dataset_dir),
            }
            if coarse["empty"]:
                plan["negative_query"] = True
            if time_window is not None:
                plan["time_window"] = time_window

            output_path = item.get("output_path")
            if output_path is not None:
                _write_json(Path(output_path), plan)
            results[chunk_start + offset] = plan

    total_elapsed = time.time() - t_all_start
    print(f"\n[qwen_planner] done: {total_queries} queries in {total_elapsed:.1f}s  "
          f"({total_elapsed/max(total_queries,1):.2f}s/query avg)")
    return results
