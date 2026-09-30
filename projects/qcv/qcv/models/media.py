from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def _uniform_indices(length: int, count: int) -> np.ndarray:
    if length < 1:
        raise ValueError("Video contains no frames")
    if count < 1:
        raise ValueError("frame_count must be positive")
    if length <= count:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=count, dtype=np.int64)


def load_video_frames(path: str | Path, frame_count: int = 32) -> list[Image.Image]:
    path = Path(path)
    if path.is_dir():
        frame_paths = sorted(
            item
            for item in path.iterdir()
            if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        indices = _uniform_indices(len(frame_paths), frame_count)
        return [load_image(frame_paths[int(index)]) for index in indices]
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError(
            "Video loading requires imageio with an ffmpeg backend: pip install 'imageio[ffmpeg]'"
        ) from exc
    metadata = iio.immeta(path, plugin="ffmpeg")
    nframes = int(metadata.get("nframes", 0))
    if nframes <= 0:
        frames_array = list(iio.imiter(path, plugin="ffmpeg"))
        indices = _uniform_indices(len(frames_array), frame_count)
        return [Image.fromarray(frames_array[int(index)]).convert("RGB") for index in indices]
    indices = _uniform_indices(nframes, frame_count)
    return [Image.fromarray(iio.imread(path, index=int(index), plugin="ffmpeg")).convert("RGB") for index in indices]


def load_media(media: Mapping[str, Any]) -> tuple[str, Any]:
    media_type = str(media.get("type", "image")).lower()
    path = media.get("path")
    if path is None:
        raise ValueError("Media record requires a path")
    if media_type == "image":
        return "image", load_image(str(path))
    if media_type == "video":
        frame_count = int(media.get("frame_count", 32))
        return "video", load_video_frames(str(path), frame_count=frame_count)
    if media_type == "frames":
        paths = [str(item) for item in media.get("paths", [])]
        if not paths:
            raise ValueError("frames media requires a non-empty paths list")
        return "video", [load_image(item) for item in paths]
    raise ValueError(f"Unsupported media type {media_type!r}")
