from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Mapping

import ijson


def _json_default(o):
    # ijson commonly uses Decimal for numbers; convert to float for JSON.
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


@dataclass(frozen=True)
class SplitSegment:
    seg_idx: int
    start_frame_idx: int
    end_frame_idx: int
    path: str


@dataclass(frozen=True)
class SplitManifest:
    version: int
    source_tracking_json: str
    source_tracking_mtime: float
    frame_dir: str
    segment_size: int
    video_id: str
    columns: list
    segments: list[SplitSegment]


def _guess_video_id(tracking_json: Path) -> str:
    # Typical names: 1128_21128000001_21128000010.json
    stem = tracking_json.stem
    head = stem.split("_")[0]
    if head.isdigit():
        return head
    return stem


def _read_video_id_from_json(tracking_json: Path) -> str | None:
    # Try to read the first item's video_id via streaming (fast even for huge files)
    try:
        with tracking_json.open("rb") as f:
            first = next(ijson.items(f, "data.item"), None)
        if isinstance(first, dict) and first.get("video_id") is not None:
            return str(first.get("video_id"))
    except Exception:
        return None
    return None


def load_manifest(path: Path) -> SplitManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SplitManifest(
        version=int(data.get("version") or 1),
        source_tracking_json=str(data.get("source_tracking_json")),
        source_tracking_mtime=float(data.get("source_tracking_mtime") or 0.0),
        frame_dir=str(data.get("frame_dir")),
        segment_size=int(data.get("segment_size")),
        video_id=str(data.get("video_id")),
        columns=list(data.get("columns") or []),
        segments=[SplitSegment(**s) for s in data.get("segments") or []],
    )


def _manifest_is_usable(manifest: SplitManifest, tracking_json: Path, frame_dir: Path, segment_size: int) -> bool:
    if manifest.segment_size != int(segment_size):
        return False
    if Path(manifest.source_tracking_json) != tracking_json:
        return False
    if Path(manifest.frame_dir) != frame_dir:
        return False
    try:
        if abs(float(manifest.source_tracking_mtime) - float(tracking_json.stat().st_mtime)) > 1e-6:
            return False
    except Exception:
        return False
    return True


class _SegmentWriter:
    def __init__(self, path: Path, columns: list):
        self.path = path
        self._out = path.open("w", encoding="utf-8", newline="\n")
        self._first = True
        self._out.write("{\n  \"columns\": ")
        self._out.write(json.dumps(columns, ensure_ascii=False, indent=2, default=_json_default))
        self._out.write(",\n  \"data\": [\n")

    def write_item(self, item: dict) -> None:
        if not self._first:
            self._out.write(",\n")
        self._out.write(json.dumps(item, ensure_ascii=False, indent=4, default=_json_default))
        self._first = False

    def close(self) -> None:
        self._out.write("\n  ]\n}\n")
        self._out.close()


def ensure_split(
    tracking_json: Path,
    frame_dir: Path,
    frame_idx_by_image_id: Mapping[str, int],
    segment_size: int,
    output_dir: Path | None = None,
    progress_cb=None,
    cancel_cb=None,
    progress_every: int = 200000,
) -> Path:
    """Split a huge tracking JSON into ordered segment JSON files.

    Output:
      <output_dir>/manifest.json
      <output_dir>/seg_00000.json, seg_00001.json, ...

    The function is idempotent: if a usable manifest exists, it returns it.
    """
    segment_size = int(segment_size)
    if segment_size <= 0:
        raise ValueError("segment_size must be > 0")

    video_id = _read_video_id_from_json(tracking_json) or _guess_video_id(tracking_json)
    out_dir = output_dir or (tracking_json.parent / f"frame_{video_id}_split")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        try:
            m = load_manifest(manifest_path)
            if _manifest_is_usable(m, tracking_json=tracking_json, frame_dir=frame_dir, segment_size=segment_size):
                return manifest_path
        except Exception:
            pass

    # Build fresh split into a temp directory then replace (so partial outputs don't break future loads).
    tmp_dir = out_dir.with_name(out_dir.name + "_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Read columns once.
    with tracking_json.open("rb") as f:
        columns = list(ijson.items(f, "columns.item"))

    # Writers are created lazily per seg_idx.
    writers: dict[int, _SegmentWriter] = {}
    seg_ranges: dict[int, list[int]] = {}

    def get_writer(seg_idx: int) -> _SegmentWriter:
        w = writers.get(seg_idx)
        if w is not None:
            return w
        p = tmp_dir / f"seg_{seg_idx:05d}.json"
        w = _SegmentWriter(p, columns)
        writers[seg_idx] = w
        seg_ranges.setdefault(seg_idx, [10**18, -1])
        return w

    processed = 0
    with tracking_json.open("rb") as fin:
        for item in ijson.items(fin, "data.item"):
            if cancel_cb is not None and cancel_cb():
                raise RuntimeError("split cancelled")
            if not isinstance(item, dict):
                continue
            image_id = str(item.get("image_id"))
            frame_idx = frame_idx_by_image_id.get(image_id)
            if frame_idx is None:
                # fallback: if numeric, treat as frame idx (best-effort)
                try:
                    frame_idx = int(image_id)
                except Exception:
                    continue
            seg_idx = int(frame_idx) // segment_size
            get_writer(seg_idx).write_item(item)
            r = seg_ranges[seg_idx]
            r[0] = min(r[0], int(frame_idx))
            r[1] = max(r[1], int(frame_idx))

            processed += 1
            if progress_cb is not None and processed % int(progress_every) == 0:
                try:
                    progress_cb(int(processed))
                except Exception:
                    pass

    for w in writers.values():
        w.close()

    segments: list[SplitSegment] = []
    for seg_idx in sorted(seg_ranges.keys()):
        start_f, end_f = seg_ranges[seg_idx]
        segments.append(
            SplitSegment(
                seg_idx=int(seg_idx),
                start_frame_idx=int(start_f),
                end_frame_idx=int(end_f),
                path=str((out_dir / f"seg_{seg_idx:05d}.json").name),
            )
        )

    manifest = SplitManifest(
        version=1,
        source_tracking_json=str(tracking_json),
        source_tracking_mtime=float(tracking_json.stat().st_mtime),
        frame_dir=str(frame_dir),
        segment_size=int(segment_size),
        video_id=str(video_id),
        columns=columns,
        segments=segments,
    )

    (tmp_dir / "manifest.json").write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")

    # Move tmp outputs into place atomically-ish.
    # Remove old seg files/manifest first to avoid mixing versions.
    for p in out_dir.glob("seg_*.json"):
        try:
            p.unlink()
        except Exception:
            pass
    if manifest_path.exists():
        try:
            manifest_path.unlink()
        except Exception:
            pass

    # Rename each generated file into out_dir
    for p in tmp_dir.glob("seg_*.json"):
        p.replace(out_dir / p.name)
    (tmp_dir / "manifest.json").replace(manifest_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return manifest_path
