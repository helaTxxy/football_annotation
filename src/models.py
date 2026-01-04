from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Detection:
    image_id: str
    track_id: int
    bbox_ltwh: tuple[float, float, float, float]
    bbox_conf: float
    role: str


@dataclass(frozen=True)
class IdRemapRule:
    from_frame_idx: int
    old_id: int
    new_id: int


@dataclass(frozen=True)
class SamRequest:
    frame_idx: int
    image_id: str
    bbox_ltwh: tuple[float, float, float, float]
    forward_frames: int


@dataclass(frozen=True)
class ManualBboxAnnotation:
    frame_idx: int
    image_id: str
    bbox_ltwh: tuple[float, float, float, float]
    track_id: float


@dataclass(frozen=True)
class BboxDeleteRule:
    kind: str  # 'single' | 'track_id' | 'frame_range'
    track_id: float
    image_id: Optional[str] = None
    bbox_ltwh: Optional[tuple[float, float, float, float]] = None
    from_frame_idx: Optional[int] = None  # 从哪帧开始删除（含）
    to_frame_idx: Optional[int] = None    # 到哪帧结束删除（含），None表示到分片结束


@dataclass
class TargetMissingEvent:
    frame_idx: int
    image_id: str
    reason: str  # out_of_view | not_detected | id_switched
    notes: Optional[str] = None
