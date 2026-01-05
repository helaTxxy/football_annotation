from __future__ import annotations

from decimal import Decimal
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import ijson

from .models import BboxDeleteRule, IdRemapRule, ManualBboxAnnotation


@dataclass(frozen=True)
class CommitSummary:
    track_id: float
    total_manual: int
    added: int
    removed: int
    replaced: int  # 原有记录被手动标注替换的数量
    backup_path: Path


@dataclass(frozen=True)
class CommitAllSummary:
    total_manual: int
    added: int
    removed: int
    replaced: int  # 原有记录被手动标注替换的数量
    backup_path: Path


def _json_default(o):
    # ijson commonly uses Decimal for numbers; convert to float for JSON.
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _bbox_close(a: list[float] | tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> bool:
    if a is None or b is None:
        return False
    try:
        ax, ay, aw, ah = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx, by, bw, bh = b
    except Exception:
        return False
    eps = 1e-3
    return abs(ax - bx) <= eps and abs(ay - by) <= eps and abs(aw - bw) <= eps and abs(ah - bh) <= eps


def _should_delete_item(item: dict, delete_rules: list[BboxDeleteRule], track_id: float) -> bool:
    image_id = str(item.get("image_id"))
    try:
        item_track_id = float(item.get("track_id"))
    except Exception:
        return False
    if item_track_id != float(track_id):
        return False

    bbox_ltwh = item.get("bbox_ltwh")
    for r in delete_rules:
        if float(r.track_id) != float(track_id):
            continue
        if r.kind == "track_id":
            return True
        if r.kind == "single":
            if r.image_id is not None and str(r.image_id) != image_id:
                continue
            if _bbox_close(bbox_ltwh if isinstance(bbox_ltwh, list) else None, r.bbox_ltwh):
                return True
    return False


def _apply_id_remap(frame_idx: int | None, track_id: int, rules: list[IdRemapRule]) -> int:
    if frame_idx is None:
        return track_id
    tid = track_id
    for r in rules:
        if frame_idx >= int(r.from_frame_idx) and tid == int(r.old_id):
            tid = int(r.new_id)
    return tid


def _frame_idx_for_image_id(image_id: str, frame_idx_by_image_id: Mapping[str, int] | None) -> int | None:
    if frame_idx_by_image_id is not None:
        v = frame_idx_by_image_id.get(str(image_id))
        if v is not None:
            return int(v)
    # fallback: if image_id is numeric, treat it as a sortable index
    try:
        return int(str(image_id))
    except Exception:
        return None


def _next_backup_path(original: Path) -> Path:
    base = original.with_suffix(original.suffix + ".bak")
    if not base.exists():
        return base
    for i in range(1, 1000):
        p = original.with_suffix(original.suffix + f".bak{i}")
        if not p.exists():
            return p
    raise RuntimeError("too many backup files")


def commit_manual_bboxes_to_tracking_json(
    tracking_json: Path,
    manual_bboxes: Iterable[ManualBboxAnnotation],
    track_id: float,
    delete_rules: Iterable[BboxDeleteRule] | None = None,
    id_remap_rules: Iterable[IdRemapRule] | None = None,
    frame_idx_by_image_id: Mapping[str, int] | None = None,
) -> CommitSummary:
    """
    将指定 track_id 的手动标注写回到 tracking JSON。
    
    重要说明：
    - manual_bbox 的 track_id 是用户指定的最终 ID，不受 remap 规则影响
    - remap 规则用于处理错误的检测数据
    - 如果原始记录的 track_id 与目标 track_id 相同，会被手动标注替换
    - 如果原始记录被 remap 后的 ID 与目标 track_id 相同，该记录会被删除（避免冲突）
    """
    target_bboxes = [b for b in manual_bboxes if float(b.track_id) == float(track_id)]
    delete_rules_list = list(delete_rules or [])
    remap_rules_list = list(id_remap_rules or [])

    if not target_bboxes and not any(float(r.track_id) == float(track_id) for r in delete_rules_list):
        raise ValueError("no manual bboxes or delete rules for track_id")

    # Read columns (small) so we can preserve the schema header.
    with tracking_json.open("rb") as f:
        columns = list(ijson.items(f, "columns.item"))

    desired_keys = {(b.image_id, float(track_id)) for b in target_bboxes}
    existing_keys: set[tuple[str, float]] = set()

    video_id: str | None = None
    default_costs = None

    tmp_path = tracking_json.with_suffix(tracking_json.suffix + ".tmp")

    added = 0
    skipped = 0
    removed = 0
    replaced = 0

    # 构建手动标注的查找表：(image_id, track_id) -> bbox_ltwh（只保留最后一个/最新的）
    manual_bbox_map: dict[tuple[str, float], tuple[float, float, float, float]] = {}
    for b in target_bboxes:
        key = (str(b.image_id), float(track_id))
        manual_bbox_map[key] = b.bbox_ltwh  # 后面的会覆盖前面的
    
    # 记录有手动标注的 image_id 集合
    manual_image_ids = {str(b.image_id) for b in target_bboxes}

    with tracking_json.open("rb") as fin, tmp_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("{\n  \"columns\": ")
        out.write(json.dumps(columns, ensure_ascii=False, indent=2, default=_json_default))
        out.write(",\n  \"data\": [\n")

        first = True
        for item in ijson.items(fin, "data.item"):
            if video_id is None:
                video_id = str(item.get("video_id") or "")
                default_costs = item.get("costs")

            image_id = str(item.get("image_id"))
            track_id_raw = item.get("track_id")
            try:
                original_track_id = float(track_id_raw)
            except Exception:
                original_track_id = None

            # 检查原始记录是否与目标 track_id 相同
            # 如果相同且有手动标注，则跳过（稍后写入手动标注）
            if original_track_id == float(track_id):
                item_key = (image_id, float(track_id))
                if item_key in manual_bbox_map:
                    # 原始记录会被手动标注替换，跳过
                    replaced += 1
                    continue

            # Apply id-remap rules
            track_id_val = original_track_id
            if remap_rules_list and original_track_id is not None:
                frame_idx = _frame_idx_for_image_id(image_id, frame_idx_by_image_id)
                mapped_tid = _apply_id_remap(frame_idx, int(original_track_id), remap_rules_list)
                if mapped_tid != int(original_track_id):
                    item["track_id"] = int(mapped_tid)
                    track_id_val = float(mapped_tid)
                    
                    # 如果 remap 后的 ID 与目标 track_id 相同，
                    # 且该 image_id 有手动标注，则删除这条原始记录（避免冲突）
                    if track_id_val == float(track_id) and image_id in manual_image_ids:
                        removed += 1
                        continue

            if delete_rules_list and _should_delete_item(item, delete_rules_list, track_id=float(track_id)):
                removed += 1
                continue

            if not first:
                out.write(",\n")
            out.write(json.dumps(item, ensure_ascii=False, indent=4, default=_json_default))
            first = False

        if video_id is None:
            raise RuntimeError("could not read video_id from tracking json")

        # Append ALL manual bboxes（手动标注直接写入）
        for key, bbox in manual_bbox_map.items():
            img_id, tid = key

            new_item = {
                "video_id": video_id,
                "bbox_conf": 1.0,
                "bbox_ltwh": [float(x) for x in bbox],
                "category_id": 1.0,
                "image_id": str(img_id),
                "embeddings": [],
                "role_confidence": None,
                "role_detection": "player",
                "visibility_scores": [True],
                "age": 1.0,
                "costs": default_costs,
                "hits": 1.0,
                "matched_with": None,
                "state": "c",
                "time_since_update": 0.0,
                "track_bbox_kf_ltwh": [float(x) for x in bbox],
                "track_bbox_pred_kf_ltwh": None,
                "track_id": float(tid),
                "bbox_pitch": None,
            }

            if not first:
                out.write(",\n")
            out.write(json.dumps(new_item, ensure_ascii=False, indent=4, default=_json_default))
            first = False
            added += 1

        out.write("\n  ]\n}\n")

    backup_path = _next_backup_path(tracking_json)
    shutil.copy2(tracking_json, backup_path)
    tmp_path.replace(tracking_json)

    return CommitSummary(
        track_id=float(track_id),
        total_manual=len(target_bboxes),
        added=added,
        removed=removed,
        replaced=replaced,
        backup_path=backup_path,
    )


def commit_annotations_to_tracking_json(
    tracking_json: Path,
    manual_bboxes: Iterable[ManualBboxAnnotation],
    delete_rules: Iterable[BboxDeleteRule] | None = None,
    id_remap_rules: Iterable[IdRemapRule] | None = None,
    frame_idx_by_image_id: Mapping[str, int] | None = None,
) -> CommitAllSummary:
    """Commit ALL manual bboxes + delete rules + id-remap rules into the given tracking JSON.

    This rewrites the (segment) JSON once, which is much faster than rewriting per-track_id.
    
    重要说明：
    - manual_bbox 的 track_id 是用户指定的最终 ID，不受 remap 规则影响
    - remap 规则用于处理错误的检测数据（将其 ID 改为 -1 或其他）
    - 如果原始记录被 remap 后的 ID 与某个 manual_bbox 的 ID 相同，
      则该原始记录会被删除（因为 manual_bbox 是用户的修正，应该覆盖）
    """
    manual_list = list(manual_bboxes)
    delete_rules_list = list(delete_rules or [])
    remap_rules_list = list(id_remap_rules or [])

    if not manual_list and not delete_rules_list and not remap_rules_list:
        # Nothing to do.
        raise ValueError("no annotations to commit")

    # Read columns (small) so we can preserve the schema header.
    with tracking_json.open("rb") as f:
        columns = list(ijson.items(f, "columns.item"))

    video_id: str | None = None
    default_costs = None

    tmp_path = tracking_json.with_suffix(tracking_json.suffix + ".tmp")
    added = 0
    removed = 0
    replaced = 0

    # 构建手动标注的查找表：(image_id, track_id) -> bbox_ltwh（只保留最后一个/最新的）
    # 注意：这里的 track_id 是用户指定的最终 ID
    manual_bbox_map: dict[tuple[str, float], tuple[float, float, float, float]] = {}
    for b in manual_list:
        key = (str(b.image_id), float(b.track_id))
        manual_bbox_map[key] = b.bbox_ltwh  # 后面的会覆盖前面的
    
    # 记录哪些 (image_id, track_id) 有手动标注，用于判断是否删除被 remap 后的原始记录
    manual_bbox_keys = set(manual_bbox_map.keys())

    with tracking_json.open("rb") as fin, tmp_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("{\n  \"columns\": ")
        out.write(json.dumps(columns, ensure_ascii=False, indent=2, default=_json_default))
        out.write(",\n  \"data\": [\n")

        first = True
        for item in ijson.items(fin, "data.item"):
            if not isinstance(item, dict):
                continue
            if video_id is None:
                video_id = str(item.get("video_id") or "")
                default_costs = item.get("costs")

            image_id = str(item.get("image_id"))
            track_id_raw = item.get("track_id")
            try:
                original_track_id = float(track_id_raw)
            except Exception:
                original_track_id = None

            # 检查是否有手动标注要覆盖这条原始记录（使用原始 track_id 匹配）
            # 如果原始记录的 track_id 与某个 manual_bbox 相同，则跳过原始记录（稍后会写入 manual_bbox）
            original_key = (image_id, float(original_track_id)) if original_track_id is not None else None
            if original_key and original_key in manual_bbox_keys:
                # 原始记录会被手动标注替换，跳过
                replaced += 1
                continue

            # Apply id remap（只改变 track_id，用于标记错误的检测）
            track_id_val = original_track_id
            if remap_rules_list and original_track_id is not None:
                frame_idx = _frame_idx_for_image_id(image_id, frame_idx_by_image_id)
                mapped_tid = _apply_id_remap(frame_idx, int(original_track_id), remap_rules_list)
                if mapped_tid != int(original_track_id):
                    item["track_id"] = int(mapped_tid)
                    track_id_val = float(mapped_tid)
                    
                    # 如果 remap 后的 ID 与某个 manual_bbox 的 ID 冲突，
                    # 则删除这条原始记录（因为 manual_bbox 是用户的修正）
                    remapped_key = (image_id, float(mapped_tid))
                    if remapped_key in manual_bbox_keys:
                        removed += 1
                        continue

            # Apply delete rules (delete rules are keyed by track_id; treat them after remap).
            if delete_rules_list:
                # We must check against ALL delete rules; re-use _should_delete_item by passing the rule's track_id.
                # Efficient enough because delete_rules is expected small.
                should_delete = False
                for r in delete_rules_list:
                    if _should_delete_item(item, delete_rules_list, track_id=float(r.track_id)):
                        should_delete = True
                        break
                if should_delete:
                    removed += 1
                    continue

            if not first:
                out.write(",\n")
            out.write(json.dumps(item, ensure_ascii=False, indent=4, default=_json_default))
            first = False

        if video_id is None:
            raise RuntimeError("could not read video_id from tracking json")

        # Append ALL manual bboxes（手动标注直接写入，不受 remap 影响）
        for key, bbox in manual_bbox_map.items():
            img_id, tid = key
            new_item = {
                "video_id": video_id,
                "bbox_conf": 1.0,
                "bbox_ltwh": [float(x) for x in bbox],
                "category_id": 1.0,
                "image_id": str(img_id),
                "embeddings": [],
                "role_confidence": None,
                "role_detection": "player",
                "visibility_scores": [True],
                "age": 1.0,
                "costs": default_costs,
                "hits": 1.0,
                "matched_with": None,
                "state": "c",
                "time_since_update": 0.0,
                "track_bbox_kf_ltwh": [float(x) for x in bbox],
                "track_bbox_pred_kf_ltwh": None,
                "track_id": float(tid),
                "bbox_pitch": None,
            }
            if not first:
                out.write(",\n")
            out.write(json.dumps(new_item, ensure_ascii=False, indent=4, default=_json_default))
            first = False
            added += 1

        out.write("\n  ]\n}\n")

    backup_path = _next_backup_path(tracking_json)
    shutil.copy2(tracking_json, backup_path)
    tmp_path.replace(tracking_json)

    return CommitAllSummary(
        total_manual=len(manual_list),
        added=added,
        removed=removed,
        replaced=replaced,
        backup_path=backup_path,
    )
