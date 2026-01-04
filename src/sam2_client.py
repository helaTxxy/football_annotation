from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TrackResponse:
    result_file_path: str
    processing_time: float
    frame_count: int


class Sam2Client:
    def __init__(self, base_url: str):
        self.base_url = str(base_url).rstrip("/")

    def health(self, timeout_s: float = 0.2) -> bool:
        try:
            data = self._get_json("/health", timeout_s=timeout_s)
            return bool((data or {}).get("ok"))
        except Exception:
            return False

    def init_video(self, video_path: str, timeout_s: float = 5.0) -> dict[str, Any]:
        return self._post_json(
            "/init_video",
            {"video_path": str(video_path)},
            timeout_s=timeout_s,
        )

    def track(
        self,
        *,
        video_path: str,
        frame_idx: int,
        obj_id: int,
        bbox_xyxy: Iterable[float],
        forward_frames: int,
        frame_paths: list[str],
        timeout_s: float = 300.0,
    ) -> TrackResponse:
        payload = {
            "video_path": str(video_path),
            "start_frame_idx": int(frame_idx),
            "obj_id": int(obj_id),
            "bbox_xyxy": [float(x) for x in bbox_xyxy],
            "forward_frames": int(forward_frames),
            "frame_paths": [str(p) for p in frame_paths],
        }
        data = self._post_json("/track", payload, timeout_s=timeout_s)
        return TrackResponse(
            result_file_path=str((data or {}).get("result_file_path") or ""),
            processing_time=float((data or {}).get("processing_time") or 0.0),
            frame_count=int((data or {}).get("frame_count") or 0),
        )

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def _get_json(self, path: str, *, timeout_s: float) -> dict[str, Any]:
        req = urllib.request.Request(self._url(path), method="GET")
        return self._do_json(req, timeout_s=timeout_s)

    def _post_json(self, path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._do_json(req, timeout_s=timeout_s)

    def _do_json(self, req: urllib.request.Request, *, timeout_s: float) -> dict[str, Any]:
        # urllib uses seconds timeout; also harden DNS/socket timeouts.
        timeout_s = max(0.05, float(timeout_s))
        old = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(timeout_s)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            return data if isinstance(data, dict) else {"data": data}
        except urllib.error.HTTPError as e:
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                raw = ""
            raise RuntimeError(f"HTTP {e.code}: {raw or e.reason}")
        finally:
            socket.setdefaulttimeout(old)
