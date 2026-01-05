from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import BboxDeleteRule, IdRemapRule, ManualBboxAnnotation, SamRequest, TargetMissingEvent


class AnnotationStore:
    def __init__(self, path: Path):
        self.path = path
        self.target_id: int | None = None
        self.id_remap_rules: list[IdRemapRule] = []
        self.sam_requests: list[SamRequest] = []
        self.manual_bboxes: list[ManualBboxAnnotation] = []
        self.delete_rules: list[BboxDeleteRule] = []
        self.missing_events: list[TargetMissingEvent] = []

        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.target_id = data.get("target_id")
        self.id_remap_rules = [IdRemapRule(**x) for x in data.get("id_remap_rules", [])]
        self.sam_requests = [SamRequest(**x) for x in data.get("sam_requests", [])]
        self.manual_bboxes = []
        for x in data.get("manual_bboxes", []):
            if not isinstance(x, dict):
                continue
            if "track_id" in x:
                try:
                    self.manual_bboxes.append(ManualBboxAnnotation(**{**x, "track_id": float(x.get("track_id"))}))
                except Exception:
                    continue
            elif "target_id" in x:
                # Backward-compat: older files used target_id; treat it as track_id.
                try:
                    self.manual_bboxes.append(
                        ManualBboxAnnotation(
                            frame_idx=int(x.get("frame_idx")),
                            image_id=str(x.get("image_id")),
                            bbox_ltwh=tuple(x.get("bbox_ltwh")),
                            track_id=float(x.get("target_id")),
                        )
                    )
                except Exception:
                    continue
        self.missing_events = [TargetMissingEvent(**x) for x in data.get("missing_events", [])]

        self.delete_rules = []
        for x in data.get("delete_rules", []):
            if not isinstance(x, dict):
                continue
            try:
                self.delete_rules.append(
                    BboxDeleteRule(
                        kind=str(x.get("kind")),
                        track_id=float(x.get("track_id")),
                        image_id=(str(x.get("image_id")) if x.get("image_id") is not None else None),
                        bbox_ltwh=(tuple(x.get("bbox_ltwh")) if x.get("bbox_ltwh") is not None else None),
                        from_frame_idx=(int(x.get("from_frame_idx")) if x.get("from_frame_idx") is not None else None),
                        to_frame_idx=(int(x.get("to_frame_idx")) if x.get("to_frame_idx") is not None else None),
                    )
                )
            except Exception:
                continue

    def save(self) -> None:
        payload: dict[str, Any] = {
            "target_id": self.target_id,
            "id_remap_rules": [asdict(r) for r in self.id_remap_rules],
            "sam_requests": [asdict(r) for r in self.sam_requests],
            "manual_bboxes": [asdict(b) for b in self.manual_bboxes],
            "delete_rules": [asdict(r) for r in self.delete_rules],
            "missing_events": [asdict(e) for e in self.missing_events],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_remap_rule(self, rule: IdRemapRule) -> None:
        self.id_remap_rules.append(rule)
        self.save()

    def add_sam_request(self, req: SamRequest) -> None:
        self.sam_requests.append(req)
        self.save()

    def add_manual_bbox(self, ann: ManualBboxAnnotation) -> None:
        self.manual_bboxes.append(ann)
        self.save()

    def add_missing_event(self, ev: TargetMissingEvent) -> None:
        self.missing_events.append(ev)
        self.save()

    def add_delete_rule(self, rule: BboxDeleteRule) -> None:
        self.delete_rules.append(rule)
        self.save()

    def delete_manual_bbox_single(self, frame_idx: int, image_id: str, track_id: float, bbox_ltwh: tuple[float, float, float, float]) -> int:
        def _close(a: float, b: float) -> bool:
            return abs(a - b) <= 1e-3

        before = len(self.manual_bboxes)
        kept: list[ManualBboxAnnotation] = []
        for ann in self.manual_bboxes:
            if ann.frame_idx != frame_idx:
                kept.append(ann)
                continue
            if ann.image_id != image_id:
                kept.append(ann)
                continue
            if float(ann.track_id) != float(track_id):
                kept.append(ann)
                continue
            ax, ay, aw, ah = ann.bbox_ltwh
            bx, by, bw, bh = bbox_ltwh
            if _close(ax, bx) and _close(ay, by) and _close(aw, bw) and _close(ah, bh):
                continue
            kept.append(ann)

        self.manual_bboxes = kept
        removed = before - len(self.manual_bboxes)
        if removed:
            self.save()
        return removed

    def delete_manual_bboxes_by_track_id(self, track_id: float) -> int:
        before = len(self.manual_bboxes)
        self.manual_bboxes = [b for b in self.manual_bboxes if float(b.track_id) != float(track_id)]
        removed = before - len(self.manual_bboxes)
        if removed:
            self.save()
        return removed

    def delete_manual_bboxes_from_frame(self, track_id: float, from_frame_idx: int) -> int:
        """删除指定 track_id 从 from_frame_idx 开始（含）的所有 manual_bboxes"""
        before = len(self.manual_bboxes)
        self.manual_bboxes = [
            b for b in self.manual_bboxes
            if not (float(b.track_id) == float(track_id) and b.frame_idx >= from_frame_idx)
        ]
        removed = before - len(self.manual_bboxes)
        if removed:
            self.save()
        return removed

    def remap_manual_bbox_single(self, frame_idx: int, image_id: str, old_track_id: float, 
                                  bbox_ltwh: tuple[float, float, float, float], new_track_id: float) -> bool:
        """将指定的单个 manual_bbox 的 track_id 修改为 new_track_id"""
        from .models import ManualBboxAnnotation
        
        def _close(a: float, b: float) -> bool:
            return abs(a - b) <= 1e-3
        
        found = False
        new_list: list[ManualBboxAnnotation] = []
        for b in self.manual_bboxes:
            if (b.frame_idx == frame_idx and 
                b.image_id == image_id and 
                float(b.track_id) == float(old_track_id)):
                # 检查 bbox 是否匹配
                bx, by, bw, bh = b.bbox_ltwh
                tx, ty, tw, th = bbox_ltwh
                if _close(bx, tx) and _close(by, ty) and _close(bw, tw) and _close(bh, th):
                    # 找到匹配的，修改 track_id
                    new_list.append(
                        ManualBboxAnnotation(
                            frame_idx=b.frame_idx,
                            image_id=b.image_id,
                            bbox_ltwh=b.bbox_ltwh,
                            track_id=float(new_track_id),
                        )
                    )
                    found = True
                    continue
            new_list.append(b)
        
        if found:
            self.manual_bboxes = new_list
            self.save()
        return found

    def remap_manual_bboxes_from_frame(self, old_track_id: float, new_track_id: float, from_frame_idx: int) -> int:
        """将指定 track_id 从 from_frame_idx 开始（含）的所有 manual_bboxes 的 track_id 修改为 new_track_id"""
        from .models import ManualBboxAnnotation
        
        count = 0
        new_list: list[ManualBboxAnnotation] = []
        for b in self.manual_bboxes:
            if float(b.track_id) == float(old_track_id) and b.frame_idx >= from_frame_idx:
                # 创建新的 ManualBboxAnnotation，修改 track_id
                new_list.append(
                    ManualBboxAnnotation(
                        frame_idx=b.frame_idx,
                        image_id=b.image_id,
                        bbox_ltwh=b.bbox_ltwh,
                        track_id=float(new_track_id),
                    )
                )
                count += 1
            else:
                new_list.append(b)
        
        if count > 0:
            self.manual_bboxes = new_list
            self.save()
        return count

