from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Iterable

import ijson

from .models import Detection


class TrackStore:
    """Lightweight per-image_id detection store.

    - Streaming-parses the original tracking JSON
    - Drops embeddings to keep memory low
    - Builds a gzip+pickle cache for fast subsequent loads
    """

    def __init__(self, tracking_json: Path, cache_path: Path, force_rebuild: bool = False):
        self.tracking_json = tracking_json
        self.cache_path = cache_path
        self._by_image: dict[str, list[Detection]] | None = None
        self._force_rebuild = force_rebuild

    def invalidate_cache(self) -> None:
        """删除缓存文件，强制下次 load 时重新从 JSON 读取"""
        if self.cache_path.exists():
            try:
                self.cache_path.unlink()
                print(f"[TrackStore] 已删除缓存: {self.cache_path.name}")
            except Exception as e:
                print(f"[TrackStore] 删除缓存失败: {e}")
        self._by_image = None

    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效：缓存必须存在，且修改时间严格大于源 JSON"""
        if not self.cache_path.exists():
            return False
        if not self.tracking_json.exists():
            return False
        # 使用严格大于（>）而非大于等于（>=），确保 JSON 有任何修改都会重建缓存
        cache_mtime = self.cache_path.stat().st_mtime
        json_mtime = self.tracking_json.stat().st_mtime
        # 加一个小的容差（0.1秒），避免时间戳精度问题
        return cache_mtime > json_mtime + 0.1

    def load(self) -> None:
        if self._by_image is not None:
            return
        if not self._force_rebuild and self._is_cache_valid():
            try:
                self._by_image = self._load_cache()
                print(f"[TrackStore] 从缓存加载: {self.cache_path.name}")
                return
            except Exception as e:
                print(f"[TrackStore] 缓存加载失败，将重建: {e}")
        # 重建缓存
        print(f"[TrackStore] 从 JSON 重建: {self.tracking_json.name}")
        self._by_image = self._build_from_json_stream()
        self._save_cache(self._by_image)

    def _load_cache(self) -> dict[str, list[Detection]]:
        with gzip.open(self.cache_path, "rb") as f:
            return pickle.load(f)

    def _save_cache(self, by_image: dict[str, list[Detection]]) -> None:
        with gzip.open(self.cache_path, "wb") as f:
            pickle.dump(by_image, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _build_from_json_stream(self) -> dict[str, list[Detection]]:
        by_image: dict[str, list[Detection]] = {}
        with self.tracking_json.open("rb") as f:
            for item in ijson.items(f, "data.item"):
                image_id = str(item.get("image_id"))
                track_id_raw = item.get("track_id")
                if image_id is None or track_id_raw is None:
                    continue

                bbox_ltwh = item.get("bbox_ltwh")
                if not isinstance(bbox_ltwh, list) or len(bbox_ltwh) != 4:
                    continue

                det = Detection(
                    image_id=image_id,
                    track_id=int(track_id_raw),
                    bbox_ltwh=(float(bbox_ltwh[0]), float(bbox_ltwh[1]), float(bbox_ltwh[2]), float(bbox_ltwh[3])),
                    bbox_conf=float(item.get("bbox_conf") or 0.0),
                    role=str(item.get("role_detection") or ""),
                )
                by_image.setdefault(image_id, []).append(det)

        # stable order: by track_id for nicer UI
        for k in list(by_image.keys()):
            by_image[k].sort(key=lambda d: d.track_id)
        return by_image

    def detections_for(self, image_id: str) -> list[Detection]:
        self.load()
        assert self._by_image is not None
        return self._by_image.get(image_id, [])

    def all_track_ids_for(self, image_id: str) -> list[int]:
        return [d.track_id for d in self.detections_for(image_id)]

    def iter_image_ids(self) -> Iterable[str]:
        self.load()
        assert self._by_image is not None
        return self._by_image.keys()
