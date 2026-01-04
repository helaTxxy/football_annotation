from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameInfo:
    frame_idx: int
    image_id: str
    path: Path


def build_frame_index(frame_dir: Path) -> list[FrameInfo]:
    if not frame_dir.exists():
        raise FileNotFoundError(f"frame dir not found: {frame_dir}")

    frames: list[FrameInfo] = []
    jpgs = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    for idx, p in enumerate(jpgs):
        frames.append(FrameInfo(frame_idx=idx, image_id=p.stem, path=p))
    if not frames:
        raise RuntimeError(f"no jpg frames found in: {frame_dir}")
    return frames
