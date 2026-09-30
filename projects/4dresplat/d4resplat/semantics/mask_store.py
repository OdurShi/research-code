from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def save_mask_store_npz(
    path: str | Path,
    frame_indices: list[int] | np.ndarray,
    masks: list[np.ndarray] | np.ndarray,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    frame_index_array = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    if frame_index_array.size == 0:
        np.savez_compressed(
            target,
            frame_indices=frame_index_array,
            mask_shape=np.asarray([0, 0, 0], dtype=np.int32),
            mask_bits=np.zeros((0, 0), dtype=np.uint8),
        )
        return

    mask_array = np.asarray(masks, dtype=np.uint8)
    if mask_array.ndim != 3:
        raise ValueError(f"Expected 3D mask array, got shape={mask_array.shape!r}")
    if mask_array.shape[0] != frame_index_array.shape[0]:
        raise ValueError(
            f"Mask/frame count mismatch: masks={mask_array.shape[0]} frames={frame_index_array.shape[0]}"
        )

    num_masks, height, width = [int(value) for value in mask_array.shape]
    flat_masks = mask_array.reshape(num_masks, height * width).astype(np.uint8)
    packed_bits = np.packbits(flat_masks, axis=1)
    np.savez_compressed(
        target,
        frame_indices=frame_index_array,
        mask_shape=np.asarray([num_masks, height, width], dtype=np.int32),
        mask_bits=packed_bits,
    )


@lru_cache(maxsize=256)
def _load_mask_store_payload(path_value: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
        mask_shape_array = np.asarray(payload["mask_shape"], dtype=np.int32).reshape(-1)
        if mask_shape_array.size != 3:
            raise ValueError(f"Invalid mask_shape in {path}")
        mask_shape = tuple(int(value) for value in mask_shape_array.tolist())
        mask_bits = np.asarray(payload["mask_bits"], dtype=np.uint8)
    return frame_indices, mask_bits, mask_shape


def frame_has_mask(frame: dict[str, Any] | None) -> bool:
    if not isinstance(frame, dict):
        return False
    return bool(frame.get("mask_path") or frame.get("mask_store_path"))


def load_frame_mask(frame: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(frame, dict):
        return None

    mask_path = frame.get("mask_path")
    if mask_path:
        path = Path(str(mask_path))
        if not path.exists():
            return None
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        return mask if mask.any() else None

    store_path = frame.get("mask_store_path")
    if not store_path:
        return None
    store_format = str(frame.get("mask_store_format", "npz")).strip().lower()
    if store_format != "npz":
        raise ValueError(f"Unsupported mask store format: {store_format}")

    try:
        frame_indices, mask_bits, mask_shape = _load_mask_store_payload(str(store_path))
    except FileNotFoundError:
        return None

    num_masks, height, width = mask_shape
    if num_masks <= 0 or height <= 0 or width <= 0:
        return None

    item_index_value = frame.get("mask_store_item_index")
    if item_index_value is None:
        frame_index_value = frame.get("frame_index")
        if frame_index_value is None:
            return None
        matches = np.where(frame_indices == int(frame_index_value))[0]
        if matches.size == 0:
            return None
        item_index = int(matches[0])
    else:
        item_index = int(item_index_value)
    if item_index < 0 or item_index >= num_masks:
        return None

    flat_count = int(height * width)
    unpacked = np.unpackbits(mask_bits[item_index], count=flat_count).astype(bool)
    mask = unpacked.reshape(height, width)
    return mask if mask.any() else None
