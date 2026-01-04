from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt, QThread, QObject, pyqtSignal
from PyQt6.QtCore import QProcess
from PyQt6.QtGui import QAction, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QProgressDialog,
)

from .annotation_store import AnnotationStore
from .frame_index import FrameInfo, build_frame_index
from .models import BboxDeleteRule, Detection, IdRemapRule, ManualBboxAnnotation, SamRequest, TargetMissingEvent
from .track_store import TrackStore
from .tracking_json_writer import commit_annotations_to_tracking_json, commit_manual_bboxes_to_tracking_json
from .tracking_splitter import ensure_split, load_manifest
from .video_canvas import VideoCanvas
from .sam2_client import Sam2Client


class _SamTrackWorker(QObject):
    finished = pyqtSignal(object, object)  # result, error

    def __init__(
        self,
        *,
        base_url: str,
        video_path: str,
        frame_idx: int,
        obj_id: int,
        bbox_ltwh: tuple[float, float, float, float],
        forward_frames: int,
        frame_paths: list[str],
        frames: list[FrameInfo],
        track_id: float,
    ):
        super().__init__()
        self.client = Sam2Client(base_url=base_url)
        self.video_path = video_path
        self.frame_idx = int(frame_idx)
        self.obj_id = int(obj_id)
        self.bbox_ltwh = bbox_ltwh
        self.forward_frames = int(forward_frames)
        self.frame_paths = list(frame_paths)
        self.frames = frames
        self.track_id = float(track_id)

    def run(self) -> None:
        try:
            # Initialize video on server (idempotent)
            self.client.init_video(self.video_path, timeout_s=30.0)

            x, y, w, h = self.bbox_ltwh
            bbox_xyxy = (float(x), float(y), float(x + w), float(y + h))
            resp = self.client.track(
                video_path=self.video_path,
                frame_idx=self.frame_idx,
                obj_id=self.obj_id,
                bbox_xyxy=bbox_xyxy,
                forward_frames=self.forward_frames,
                frame_paths=self.frame_paths,
                timeout_s=300.0,
            )

            # Load result json written by local server process.
            import json
            from pathlib import Path

            p = Path(resp.result_file_path)
            if not p.exists():
                raise RuntimeError(f"SAM结果文件不存在: {p}")
            data = json.loads(p.read_text(encoding="utf-8"))

            # Convert to ManualBboxAnnotation list.
            from .models import ManualBboxAnnotation

            out: list[ManualBboxAnnotation] = []
            # Keys are strings of frame_idx -> obj_id -> {bbox:[x,y,w,h]}
            for k_frame, v in (data or {}).items():
                try:
                    fidx = int(k_frame)
                except Exception:
                    continue
                if fidx < self.frame_idx:
                    continue
                if fidx >= len(self.frames):
                    continue
                obj_map = v or {}
                obj_entry = obj_map.get(str(self.obj_id)) or obj_map.get(self.obj_id)
                if not isinstance(obj_entry, dict):
                    continue
                bb = obj_entry.get("bbox")
                if not (isinstance(bb, list) and len(bb) == 4):
                    continue
                x, y, w, h = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                fi = self.frames[fidx]
                out.append(
                    ManualBboxAnnotation(
                        frame_idx=int(fidx),
                        image_id=fi.image_id,
                        bbox_ltwh=(x, y, w, h),
                        track_id=float(self.track_id),
                    )
                )

            self.finished.emit(
                {
                    "bboxes": out,
                    "meta": resp,
                    "start_frame_idx": int(self.frame_idx),
                    "forward_frames": int(self.forward_frames),
                    "track_id": float(self.track_id),
                },
                None,
            )
        except Exception as e:
            self.finished.emit(None, e)


class _SplitWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object)  # manifest_path, error

    def __init__(self, tracking_json: Path, frame_dir: Path, frame_idx_by_image_id: dict[str, int], segment_size: int):
        super().__init__()
        self.tracking_json = tracking_json
        self.frame_dir = frame_dir
        self.frame_idx_by_image_id = frame_idx_by_image_id
        self.segment_size = segment_size
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            manifest_path = ensure_split(
                tracking_json=self.tracking_json,
                frame_dir=self.frame_dir,
                frame_idx_by_image_id=self.frame_idx_by_image_id,
                segment_size=self.segment_size,
                progress_cb=lambda n: self.progress.emit(int(n)),
                cancel_cb=lambda: bool(self._cancelled),
            )
            self.finished.emit(manifest_path, None)
        except Exception as e:
            self.finished.emit(None, e)


class _AutoCommitWorker(QObject):
    finished = pyqtSignal(object, object, object)  # summary, error, segment_json

    def __init__(
        self,
        segment_json: Path,
        manual_bboxes,
        delete_rules,
        id_remap_rules,
        frame_idx_by_image_id: dict[str, int],
    ):
        super().__init__()
        self.segment_json = segment_json
        self.manual_bboxes = manual_bboxes
        self.delete_rules = delete_rules
        self.id_remap_rules = id_remap_rules
        self.frame_idx_by_image_id = frame_idx_by_image_id

    def run(self) -> None:
        try:
            summary = commit_annotations_to_tracking_json(
                tracking_json=self.segment_json,
                manual_bboxes=self.manual_bboxes,
                delete_rules=self.delete_rules,
                id_remap_rules=self.id_remap_rules,
                frame_idx_by_image_id=self.frame_idx_by_image_id,
            )
            self.finished.emit(summary, None, self.segment_json)
        except Exception as e:
            self.finished.emit(None, e, self.segment_json)


class MainWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.setWindowTitle("jh4player - 标注交互")

        self.root = root
        self.settings_path = root / "jh4player_settings.json"
        self.annotations = AnnotationStore(root / "annotations.json")
        
        # 程序启动时备份 annotation（两级防护）
        self._backup_annotations_on_startup()

        # Split settings (30fps video: half-minute = 900 frames)
        self.segment_size: int = 900
        self._split_ready: bool = False
        self._split_manifest_path: Path | None = None
        self._segment_paths_by_idx: dict[int, Path] = {}
        self._current_segment_idx: int = 0
        self._split_thread: QThread | None = None
        self._split_worker: _SplitWorker | None = None
        self._split_progress: QProgressDialog | None = None

        self._autocommit_thread: QThread | None = None
        self._autocommit_worker: _AutoCommitWorker | None = None
        self._autocommit_inflight: bool = False

        # Data sources (tracking json + frame directory)
        self.frame_dir: Path
        self.tracking_json: Path
        self.frames: list[FrameInfo]
        self.frame_idx_by_image_id: dict[str, int]
        self.track_store: TrackStore

        # Resolve defaults first (workspace root), then allow a persisted override via settings.
        tracking_json, frame_dir = self._resolve_initial_sources()
        self._load_sources(tracking_json=tracking_json, frame_dir=frame_dir, save_settings=False, render=False)

        self.target_id: int | None = self.annotations.target_id
        self.current_frame_idx: int = 0
        self.playing: bool = False
        self.base_fps: float = 30
        self.speed: float = 1.0
        self.fps: float = self.base_fps * self.speed

        self._needs_initial_target_prompt: bool = self.target_id is None

        self._manual_bbox_mode: bool = False
        self._manual_bbox_frame_idx: int | None = None
        self._manual_bbox_track_id: float | None = None
        self._manual_bbox_prompt_track_id_on_confirm: bool = False

        # SAM2 integration state
        self._sam_base_url: str = "http://127.0.0.1:8848"
        self._sam_proc: QProcess | None = None
        self._sam_track_thread: QThread | None = None
        self._sam_track_worker: _SamTrackWorker | None = None
        self._sam_pending: dict | None = None
        self._sam_handoff_bboxes: list[ManualBboxAnnotation] = []

        # "不在画面内"模式：标记后停止自动停止/弹窗，直到用户手动添加标注
        self._target_out_of_view: bool = False
        
        # 上一帧 target 的 bbox，用于检测跟踪跳变
        self._last_target_bbox: tuple[float, float, float, float] | None = None
        self._last_target_frame_idx: int | None = None

        self._selected_det: Detection | None = None
        self._awaiting_remap_click: bool = False

        self._deleted_track_ids: set[int] = set()
        self._deleted_single: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
        self._deleted_frame_range: list[tuple[int, int | None, int | None]] = []
        self._rebuild_delete_index()

        self.id_remap_rules: list[IdRemapRule] = list(self.annotations.id_remap_rules)

        self.canvas = VideoCanvas(self)
        self.canvas.detectionClicked.connect(self.on_detection_clicked)
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.frames) - 1)
        self.slider.valueChanged.connect(self.on_slider_changed)

        self.speed_box = QDoubleSpinBox()
        self.speed_box.setRange(0.5, 2.0)
        self.speed_box.setSingleStep(0.1)
        self.speed_box.setValue(1.0)
        self.speed_box.setKeyboardTracking(False)
        self.speed_box.lineEdit().setReadOnly(True)  # 禁用手动输入，只能用上下按钮
        self.speed_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # 不获取键盘焦点
        self.speed_box.valueChanged.connect(self.on_speed_changed)

        self.setStatusBar(QStatusBar(self))
        self.set_status("Ready")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.canvas, stretch=1)
        
        # 进度条和帧数显示
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider, stretch=1)
        self.frame_label = QLabel(f"0/{len(self.frames)}")
        self.frame_label.setMinimumWidth(100)
        slider_layout.addWidget(self.frame_label)
        layout.addLayout(slider_layout)
        
        self.setCentralWidget(central)

        # Start SAM service early (non-blocking). It can take time to load model.
        QTimer.singleShot(0, self._ensure_sam_server_running)

        tb = QToolBar("Main")
        self.addToolBar(tb)

        act_select_data = QAction("选择数据", self)
        act_select_data.triggered.connect(self.action_select_data_sources)
        tb.addAction(act_select_data)

        tb.addSeparator()

        act_set_target = QAction("设置Target", self)
        act_set_target.triggered.connect(self.action_set_target)
        tb.addAction(act_set_target)

        act_unlock_target = QAction("取消锁定", self)
        act_unlock_target.triggered.connect(self.action_unlock_target)
        tb.addAction(act_unlock_target)

        act_add_ann = QAction("添加标注", self)
        act_add_ann.triggered.connect(self.action_add_annotation)
        tb.addAction(act_add_ann)

        tb.addSeparator()
        tb.addWidget(QLabel("track_id"))
        self.track_id_box = QDoubleSpinBox()
        self.track_id_box.setRange(1.0, 1e12)
        self.track_id_box.setDecimals(1)
        self.track_id_box.setSingleStep(1.0)
        # If a target exists, default to it; otherwise default to 1.0
        self.track_id_box.setValue(float(self.target_id) if self.target_id is not None else 1.0)
        self.track_id_box.setEnabled(self.target_id is None)
        tb.addWidget(self.track_id_box)

        act_commit = QAction("写回原JSON", self)
        act_commit.triggered.connect(self.action_commit_to_original_json)
        tb.addAction(act_commit)

        self.act_delete_one = QAction("删除当前框", self)
        self.act_delete_one.triggered.connect(self.action_delete_selected_one)
        self.act_delete_one.setEnabled(False)
        tb.addAction(self.act_delete_one)

        self.act_delete_id = QAction("删除该ID所有框", self)
        self.act_delete_id.triggered.connect(self.action_delete_selected_track)
        self.act_delete_id.setEnabled(False)
        tb.addAction(self.act_delete_id)

        self.act_remap_to_target = QAction("映射ID", self)
        self.act_remap_to_target.triggered.connect(self.action_remap_selected_to_target)
        self.act_remap_to_target.setEnabled(False)
        tb.addAction(self.act_remap_to_target)

        self.act_play = QAction("播放", self)
        self.act_play.triggered.connect(self.action_play)
        tb.addAction(self.act_play)

        self.act_pause = QAction("暂停", self)
        self.act_pause.triggered.connect(self.action_pause)
        tb.addAction(self.act_pause)

        act_prev = QAction("上一帧", self)
        act_prev.triggered.connect(lambda: self.goto_frame(self.current_frame_idx - 1))
        tb.addAction(act_prev)

        act_next = QAction("下一帧", self)
        act_next.triggered.connect(lambda: self.goto_frame(self.current_frame_idx + 1))
        tb.addAction(act_next)

        tb.addSeparator()
        tb.addWidget(QLabel("倍速"))
        tb.addWidget(self.speed_box)

        act_jump = QAction("跳到帧", self)
        act_jump.triggered.connect(self.action_jump)
        tb.addAction(act_jump)

        tb.addSeparator()
        self.act_confirm_bbox = QAction("确认标注", self)
        self.act_confirm_bbox.triggered.connect(self.on_confirm_manual_bbox)
        self.act_confirm_bbox.setEnabled(False)
        self.act_confirm_bbox.setVisible(False)
        tb.addAction(self.act_confirm_bbox)

        self.act_cancel_bbox = QAction("取消标注", self)
        self.act_cancel_bbox.triggered.connect(self.on_cancel_manual_bbox)
        self.act_cancel_bbox.setEnabled(False)
        self.act_cancel_bbox.setVisible(False)
        tb.addAction(self.act_cancel_bbox)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)

        # Show the first frame image immediately (no need to parse the large tracking JSON yet).
        self.render_frame_image_only(0)

        # After UI is visible, split in background and then load overlays.
        QTimer.singleShot(0, self._start_split_job)

    # --- data sources ---
    def _resolve_default_tracking_json(self) -> Path:
        candidates = sorted(self.root.glob("*_21128000001_21128000010.json"))
        tracking_candidates = [p for p in candidates if "_image_" not in p.name]
        if len(tracking_candidates) == 1:
            return tracking_candidates[0]
        if len(tracking_candidates) > 1:
            items = [p.name for p in tracking_candidates]
            choice, ok = QInputDialog.getItem(self, "选择Tracking JSON", "检测到多个可能的tracking json，请选择：", items, 0, False)
            if not ok:
                raise FileNotFoundError("tracking json selection cancelled")
            return self.root / str(choice)
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError("tracking json not found in workspace root")

    def _load_settings(self) -> dict:
        if not self.settings_path.exists():
            return {}
        try:
            return json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self, tracking_json: Path, frame_dir: Path) -> None:
        payload = {
            "tracking_json": str(tracking_json),
            "frame_dir": str(frame_dir),
            "segment_size": int(self.segment_size),
        }
        self.settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _resolve_initial_sources(self) -> tuple[Path, Path]:
        # Defaults: relative to code workspace
        default_frame_dir = self.root / "frame"
        default_tracking_json = self._resolve_default_tracking_json()

        # Optional persisted override
        s = self._load_settings()
        tj = Path(s.get("tracking_json")) if s.get("tracking_json") else default_tracking_json
        fd = Path(s.get("frame_dir")) if s.get("frame_dir") else default_frame_dir
        try:
            if s.get("segment_size") is not None:
                self.segment_size = int(s.get("segment_size"))
        except Exception:
            self.segment_size = 900

        # Validate; fallback to defaults if invalid
        if not tj.exists():
            tj = default_tracking_json
        if not fd.exists():
            fd = default_frame_dir
        return tj, fd

    def _cache_path_for(self, tracking_json: Path) -> Path:
        # Keep caches under workspace root, but avoid collisions between different json files.
        safe_stem = tracking_json.stem.replace(" ", "_")
        return self.root / f"tracks_cache_{safe_stem}.pkl.gz"

    def _load_sources(self, tracking_json: Path, frame_dir: Path, save_settings: bool, render: bool) -> None:
        self.tracking_json = tracking_json
        self.frame_dir = frame_dir
        self.frames = build_frame_index(self.frame_dir)
        self.frame_idx_by_image_id = {f.image_id: f.frame_idx for f in self.frames}
        # TrackStore will be replaced by split-segment TrackStore after split completes.
        self.track_store = TrackStore(self.tracking_json, self._cache_path_for(self.tracking_json))
        self._split_ready = False
        self._split_manifest_path = None
        self._segment_paths_by_idx = {}
        self._current_segment_idx = 0

        # Slider range depends on frame count
        if hasattr(self, "slider"):
            self.slider.blockSignals(True)
            self.slider.setMinimum(0)
            self.slider.setMaximum(max(0, len(self.frames) - 1))
            self.slider.blockSignals(False)

        if save_settings:
            self._save_settings(tracking_json=self.tracking_json, frame_dir=self.frame_dir)

        # Keep current frame in range
        if hasattr(self, "current_frame_idx"):
            self.current_frame_idx = min(self.current_frame_idx, max(0, len(self.frames) - 1))

        if render and hasattr(self, "canvas"):
            # During split we can show image-only; overlays will appear after split is ready.
            self.render_frame_image_only(self.current_frame_idx)

    def action_select_data_sources(self) -> None:
        json_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 tracking JSON",
            str(self.tracking_json.parent if hasattr(self, "tracking_json") else self.root),
            "JSON (*.json);;All Files (*)",
        )
        if not json_path:
            return

        frame_dir = QFileDialog.getExistingDirectory(
            self,
            "选择 frame 目录（包含 *.jpg，文件名为 image_id）",
            str(self.frame_dir if hasattr(self, "frame_dir") else (self.root / "frame")),
        )
        if not frame_dir:
            return

        try:
            self._load_sources(tracking_json=Path(json_path), frame_dir=Path(frame_dir), save_settings=True, render=True)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载数据失败：{e}")
            return
        self.set_status(f"已加载：json={Path(json_path).name}  frame_dir={frame_dir}")
        self._start_split_job()

    # --- split loading ---
    def _start_split_job(self) -> None:
        if self._split_ready:
            return
        if self._split_thread is not None:
            return

        self.set_status(f"正在分片（{self.segment_size}帧/片）…大文件首次分片会较慢")
        self._split_progress = QProgressDialog("正在分片…", "取消", 0, 0, self)
        self._split_progress.setWindowTitle("分片")
        self._split_progress.setMinimumDuration(0)
        self._split_progress.setValue(0)
        self._split_progress.canceled.connect(self._cancel_split_job)
        self._split_progress.show()

        self._split_thread = QThread(self)
        self._split_worker = _SplitWorker(
            tracking_json=self.tracking_json,
            frame_dir=self.frame_dir,
            frame_idx_by_image_id=self.frame_idx_by_image_id,
            segment_size=self.segment_size,
        )
        self._split_worker.moveToThread(self._split_thread)
        self._split_thread.started.connect(self._split_worker.run)
        self._split_worker.progress.connect(self._on_split_progress)
        self._split_worker.finished.connect(self._on_split_finished)
        self._split_worker.finished.connect(self._split_thread.quit)
        self._split_worker.finished.connect(self._split_worker.deleteLater)
        self._split_thread.finished.connect(self._split_thread.deleteLater)
        self._split_thread.start()

    def _cancel_split_job(self) -> None:
        if self._split_worker is not None:
            self._split_worker.cancel()

    def _on_split_progress(self, n: int) -> None:
        # Indeterminate progress dialog; just update status every so often.
        self.set_status(f"正在分片…已处理 {n} 条记录")

    def _on_split_finished(self, manifest_path: object, err: object) -> None:
        if self._split_progress is not None:
            self._split_progress.close()
            self._split_progress = None

        # Tear down thread refs
        self._split_thread = None
        self._split_worker = None

        if err is not None:
            QMessageBox.critical(self, "分片失败", f"分片失败：{err}")
            self.set_status("分片失败")
            return
        if manifest_path is None:
            QMessageBox.critical(self, "分片失败", "分片失败：未知错误")
            self.set_status("分片失败")
            return

        self._split_manifest_path = Path(str(manifest_path))
        try:
            m = load_manifest(self._split_manifest_path)
        except Exception as e:
            QMessageBox.critical(self, "分片失败", f"读取 manifest 失败：{e}")
            return

        split_dir = self._split_manifest_path.parent
        self._segment_paths_by_idx = {int(s.seg_idx): (split_dir / s.path) for s in m.segments}
        self._split_ready = True

        self._current_segment_idx = self.current_frame_idx // int(self.segment_size)
        self._switch_segment(self._current_segment_idx, do_autocommit=False)
        self.render_frame(self.current_frame_idx)
        self.set_status(f"分片完成：共 {len(self._segment_paths_by_idx)} 片（{self.segment_size}帧/片）")

    def _segment_idx_for_frame(self, frame_idx: int) -> int:
        return int(frame_idx) // int(self.segment_size)

    def _segment_path(self, seg_idx: int) -> Path | None:
        return self._segment_paths_by_idx.get(int(seg_idx))

    def _switch_segment(self, seg_idx: int, do_autocommit: bool = True) -> None:
        seg_idx = int(seg_idx)
        if not self._split_ready:
            return
        if seg_idx == self._current_segment_idx and self.track_store.tracking_json == self._segment_path(seg_idx):
            return

        # Auto-commit previous segment in background.
        if do_autocommit and self._segment_path(self._current_segment_idx) is not None:
            self._autocommit_segment_async(self._segment_path(self._current_segment_idx), target_seg_idx=self._current_segment_idx)

        p = self._segment_path(seg_idx)
        if p is None or not p.exists():
            # Segment missing: just keep old store.
            return
        self._current_segment_idx = seg_idx
        self.track_store = TrackStore(p, self._cache_path_for(p))

        # If we have a SAM handoff bbox (segment_end+1), attach it to the new segment's annotation pool.
        if self._sam_handoff_bboxes:
            try:
                keep: list[ManualBboxAnnotation] = []
                for b in self._sam_handoff_bboxes:
                    if self._segment_idx_for_frame(int(b.frame_idx)) == int(self._current_segment_idx):
                        self.annotations.add_manual_bbox(b)
                    else:
                        keep.append(b)
                self._sam_handoff_bboxes = keep
            except Exception:
                pass

    def set_status(self, msg: str) -> None:
        sb = self.statusBar()
        if sb is not None:
            sb.showMessage(msg)

    def _rebuild_delete_index(self) -> None:
        self._deleted_track_ids = set()
        self._deleted_single = {}
        self._deleted_frame_range: list[tuple[int, int | None, int | None]] = []  # (track_id, from_frame, to_frame)
        for r in self.annotations.delete_rules:
            try:
                tid = int(float(r.track_id))
            except Exception:
                continue
            if r.kind == "track_id":
                self._deleted_track_ids.add(tid)
                continue
            if r.kind == "single" and r.image_id is not None and r.bbox_ltwh is not None:
                key = (str(r.image_id), tid)
                self._deleted_single.setdefault(key, []).append(r.bbox_ltwh)
                continue
            if r.kind == "frame_range":
                self._deleted_frame_range.append((tid, r.from_frame_idx, r.to_frame_idx))

    def _is_deleted(self, image_id: str, track_id: int, bbox_ltwh: tuple[float, float, float, float], frame_idx: int | None = None) -> bool:
        result = track_id in self._deleted_track_ids
        if result:
            print(f"[DEBUG] _is_deleted: track_id={track_id} FOUND in _deleted_track_ids={self._deleted_track_ids}")
        if result:
            return True
        
        # 检查帧范围删除规则
        if frame_idx is not None:
            for tid, from_frame, to_frame in self._deleted_frame_range:
                if tid != track_id:
                    continue
                if from_frame is not None and frame_idx < from_frame:
                    continue
                if to_frame is not None and frame_idx > to_frame:
                    continue
                print(f"[DEBUG] _is_deleted: track_id={track_id} frame_idx={frame_idx} FOUND in frame_range ({from_frame}, {to_frame})")
                return True
        
        key = (image_id, track_id)
        boxes = self._deleted_single.get(key)
        if not boxes:
            return False
        bx, by, bw, bh = bbox_ltwh
        eps = 1e-3
        for x, y, w, h in boxes:
            if abs(x - bx) <= eps and abs(y - by) <= eps and abs(w - bw) <= eps and abs(h - bh) <= eps:
                return True
        return False

    def _set_selected_det(self, det: Detection | None) -> None:
        self._selected_det = det
        self.act_delete_one.setEnabled(det is not None)
        self.act_delete_id.setEnabled(det is not None)
        # 映射按钮只要有选中的检测框就启用
        self.act_remap_to_target.setEnabled(det is not None)
        if det is None:
            self.canvas.set_selected_track(None)
        else:
            self.canvas.set_selected_track(det.track_id)

    def _backup_annotations_on_startup(self) -> None:
        """程序启动时备份 annotation.json（两级防护：startup + exit）"""
        try:
            ann_path = self.root / "annotations.json"
            if not ann_path.exists():
                return
            backup_dir = self.root / "annotation_backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"annotations_startup_{timestamp}.json"
            
            import shutil
            shutil.copy2(ann_path, backup_path)
            print(f"[INFO] 启动时备份 annotation: {backup_path}")
            
            # 清理旧备份，只保留最近 20 个
            self._cleanup_old_backups(backup_dir, prefix="annotations_startup_", keep=20)
        except Exception as e:
            print(f"[WARN] 启动时备份 annotation 失败: {e}")

    def _backup_annotations_on_exit(self) -> None:
        """程序退出时备份 annotation.json（两级防护：startup + exit）"""
        try:
            ann_path = self.root / "annotations.json"
            if not ann_path.exists():
                return
            # 检查是否有未写回的数据
            if (not self.annotations.manual_bboxes and 
                not self.annotations.delete_rules and 
                not self.annotations.sam_requests and 
                not self.annotations.missing_events):
                print("[INFO] 退出时无未写回数据，跳过备份")
                return
                
            backup_dir = self.root / "annotation_backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"annotations_exit_{timestamp}.json"
            
            import shutil
            shutil.copy2(ann_path, backup_path)
            print(f"[INFO] 退出时备份 annotation: {backup_path}")
            
            # 清理旧备份，只保留最近 20 个
            self._cleanup_old_backups(backup_dir, prefix="annotations_exit_", keep=20)
        except Exception as e:
            print(f"[WARN] 退出时备份 annotation 失败: {e}")

    def _cleanup_old_backups(self, backup_dir: Path, prefix: str, keep: int) -> None:
        """清理旧备份文件，只保留最近的 keep 个"""
        try:
            backups = sorted(backup_dir.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in backups[keep:]:
                old.unlink()
                print(f"[INFO] 删除旧备份: {old.name}")
        except Exception:
            pass

    def keyPressEvent(self, e):
        # 空格键控制播放/暂停
        if e.key() == Qt.Key.Key_Space:
            if self.playing:
                self.action_pause()
            else:
                self.action_play()
            return
        super().keyPressEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self._post_show_init)

    def closeEvent(self, e):
        # 程序关闭时备份 annotation（两级防护）
        self._backup_annotations_on_exit()
        
        # Ensure background threads stop cleanly to avoid:
        # "QThread: Destroyed while thread is still running"
        try:
            if self._split_worker is not None:
                self._split_worker.cancel()
            if self._split_thread is not None:
                self._split_thread.quit()
                self._split_thread.wait(5000)
        except Exception:
            pass

        try:
            if self._autocommit_thread is not None:
                self._autocommit_thread.quit()
                self._autocommit_thread.wait(5000)
        except Exception:
            pass

        # 关闭 SAM 服务进程
        try:
            if self._sam_proc is not None:
                self._sam_proc.terminate()
                if not self._sam_proc.waitForFinished(3000):
                    self._sam_proc.kill()
                    self._sam_proc.waitForFinished(2000)
                self._sam_proc = None
        except Exception:
            pass

        super().closeEvent(e)

    def _post_show_init(self) -> None:
        # Now that the window is visible, load detections/overlay for the current frame.
        self.render_frame(self.current_frame_idx)
        # Start SAM2 service eagerly so it's ready when the operator clicks.
        # This is non-blocking (QProcess) and won't freeze UI.
        try:
            self._ensure_sam_server_running()
        except Exception:
            pass
        if self._needs_initial_target_prompt:
            self._needs_initial_target_prompt = False
            self.action_set_target(initial=True)

    # --- bbox utils ---
    def _bbox_intersection(self, a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        """计算两个 bbox (ltwh) 的交集面积"""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        # 转换为 xyxy
        ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
        bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
        # 计算交集
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    def _bbox_iou(self, a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        """计算两个 bbox (ltwh) 的 IoU"""
        inter = self._bbox_intersection(a, b)
        if inter <= 0:
            return 0.0
        area_a = a[2] * a[3]
        area_b = b[2] * b[3]
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    # --- mapping ---
    def apply_id_remap(self, frame_idx: int, track_id: int) -> int:
        tid = track_id
        for r in self.id_remap_rules:
            if frame_idx >= r.from_frame_idx and tid == r.old_id:
                tid = r.new_id
        return tid

    def detections_for_frame(self, frame_idx: int) -> list[Detection]:
        image_id = self.frames[frame_idx].image_id
        dets = self.track_store.detections_for(image_id)
        
        # 收集手动/SAM标注的 (image_id, track_id) 集合，用于覆盖原有检测
        manual_keys: set[tuple[str, int]] = set()
        for ann in self.annotations.manual_bboxes:
            if ann.frame_idx == frame_idx and ann.image_id == image_id:
                manual_keys.add((ann.image_id, int(float(ann.track_id))))
        
        mapped: list[Detection] = []
        for d in dets:
            new_id = self.apply_id_remap(frame_idx, d.track_id)
            if new_id != d.track_id:
                display_det = Detection(image_id=d.image_id, track_id=new_id, bbox_ltwh=d.bbox_ltwh, bbox_conf=d.bbox_conf, role=d.role)
            else:
                display_det = d

            # Apply deletion filter AFTER remap so it matches what the user sees/clicks.
            if self._is_deleted(image_id, display_det.track_id, display_det.bbox_ltwh, frame_idx):
                continue
            
            # 如果该 (image_id, track_id) 已有手动/SAM标注，跳过原有检测，只显示手动的
            if (image_id, display_det.track_id) in manual_keys:
                continue

            mapped.append(display_det)

        # Merge manual bbox annotations into overlay so newly added data shows immediately.
        # These annotations are stored in annotations.json and treated as a detection for that frame.
        # 如果同一个 (image_id, track_id) 有多个手动标注，只取最后一个（最新的）
        # 注意：手动标注（包括SAM生成的）是用户的"修正"，不应用 ID remap 规则
        # 因为用户添加这些标注就是为了覆盖错误的检测
        # 同样，手动标注不受 frame_range 删除规则影响（只受 single/track_id 删除规则影响）
        seen_manual_keys: set[tuple[str, int]] = set()
        manual_dets_reversed: list[Detection] = []
        for ann in reversed(self.annotations.manual_bboxes):
            if ann.frame_idx != frame_idx:
                continue
            if ann.image_id != image_id:
                continue
            
            # 手动标注直接使用原始 track_id，不应用 remap
            # 这样用户添加的标注不会被之前的 remap 规则过滤掉
            original_track_id = int(float(ann.track_id))
            
            key = (ann.image_id, original_track_id)
            if key in seen_manual_keys:
                continue  # 跳过旧的，只保留最新的
            seen_manual_keys.add(key)
            # 手动标注只检查 single 和 track_id 删除规则，不检查 frame_range
            # 因为 frame_range 是用来删除错误的自动跟踪，手动标注是用户的修正
            if self._is_deleted(image_id, original_track_id, ann.bbox_ltwh, frame_idx=None):
                continue
            manual_dets_reversed.append(
                Detection(
                    image_id=ann.image_id,
                    track_id=original_track_id,
                    bbox_ltwh=ann.bbox_ltwh,
                    bbox_conf=1.0,
                    role="manual",
                )
            )
        # 反转回正序添加
        mapped.extend(reversed(manual_dets_reversed))
        return mapped

    # --- UI actions ---
    def action_set_target(self, initial: bool = False) -> None:
        first_dets = self.detections_for_frame(0)
        ids = sorted({d.track_id for d in first_dets})
        items = [str(x) for x in ids]
        items.append("手动输入…")
        choice, ok = QInputDialog.getItem(self, "选择Target ID", "在第一帧选择已有 track_id，或手动输入", items, 0, False)
        if not ok:
            if initial:
                self.set_status("未设置 target_id")
            return
        if choice == "手动输入…":
            v, ok2 = QInputDialog.getInt(self, "手动输入", "target_id", value=1, min=1, max=10**9)
            if not ok2:
                return
            self.target_id = int(v)
        else:
            self.target_id = int(choice)

        self.annotations.target_id = self.target_id
        self.annotations.save()
        # Lock the toolbar track_id to the chosen target.
        if self.target_id is not None:
            self.track_id_box.blockSignals(True)
            self.track_id_box.setValue(float(self.target_id))
            self.track_id_box.blockSignals(False)
            self.track_id_box.setEnabled(False)
        # 重新设置 target 时清除"不在画面内"状态和上一帧 bbox
        self._target_out_of_view = False
        self._last_target_bbox = None
        self._last_target_frame_idx = None
        # 初始化当前帧的 target bbox
        dets = self.detections_for_frame(self.current_frame_idx)
        target_det = next((d for d in dets if d.track_id == self.target_id), None)
        if target_det is not None:
            self._last_target_bbox = target_det.bbox_ltwh
            self._last_target_frame_idx = self.current_frame_idx
        self.set_status(f"target_id = {self.target_id}")
        self.render_frame(self.current_frame_idx)

    def action_unlock_target(self) -> None:
        # Unlock target: stop auto-missing checks and allow manual track_id input.
        if self.target_id is None:
            self.set_status("target_id 未锁定")
            self.track_id_box.setEnabled(True)
            return

        self.target_id = None
        self.annotations.target_id = None
        self.annotations.save()

        self._awaiting_remap_click = False
        self._target_out_of_view = False  # 取消锁定时也清除该状态
        self._last_target_bbox = None
        self._last_target_frame_idx = None
        self.track_id_box.setEnabled(True)
        self.set_status("已取消锁定 target_id")
        self.render_frame(self.current_frame_idx)

    def action_commit_to_original_json(self) -> None:
        # In split mode, "original" for performance means the current segment file.
        commit_path = self.tracking_json
        if self._split_ready:
            seg_path = self._segment_path(self._current_segment_idx)
            if seg_path is not None and seg_path.exists():
                commit_path = seg_path

        # If target is not locked, allow selecting which track_id to write back.
        if self.target_id is None:
            track_ids = sorted({float(b.track_id) for b in self.annotations.manual_bboxes})
            if not track_ids:
                self.set_status("没有手动标注数据，无需写回")
                return
            items = [str(x) for x in track_ids]
            choice, ok = QInputDialog.getItem(self, "选择写回的track_id", "未锁定target，请选择要写回原JSON的track_id：", items, 0, False)
            if not ok:
                return
            commit_track_id = float(choice)
        else:
            commit_track_id = float(self.target_id)
        try:
            summary = commit_manual_bboxes_to_tracking_json(
                tracking_json=commit_path,
                manual_bboxes=self.annotations.manual_bboxes,
                delete_rules=self.annotations.delete_rules,
                track_id=float(commit_track_id),
                id_remap_rules=self.annotations.id_remap_rules,
                frame_idx_by_image_id=self.frame_idx_by_image_id,
            )
        except ValueError:
            self.set_status("该 track_id 没有手动标注数据，无需写回")
            return
        except Exception as e:
            QMessageBox.critical(self, "写回失败", f"写回原JSON失败：{e}")
            return

        self.set_status(
            f"写回完成：track_id={summary.track_id} 新增={summary.added} 删除={summary.removed} 替换={summary.replaced}  备份={summary.backup_path.name}"
        )

    def action_add_annotation(self) -> None:
        # Allow adding annotation even when target_id is not locked.
        self.start_add_annotation_flow(self.current_frame_idx)

    def action_jump(self) -> None:
        v, ok = QInputDialog.getInt(self, "跳到帧", f"0 ~ {len(self.frames)-1}", value=self.current_frame_idx, min=0, max=len(self.frames)-1)
        if ok:
            self.goto_frame(int(v))

    def on_speed_changed(self, v: int) -> None:
        self.speed = float(v)
        self.fps = self.base_fps * self.speed
        if self.playing:
            self.timer.start(int(1000 / self.fps))

    def action_play(self) -> None:
        if self.playing:
            return
        self.playing = True
        self.timer.start(int(1000 / self.fps))

    def action_pause(self) -> None:
        if not self.playing:
            return
        self.playing = False
        self.timer.stop()

    def on_tick(self) -> None:
        self.goto_frame(self.current_frame_idx + 1)

    def on_slider_changed(self, v: int) -> None:
        if v != self.current_frame_idx:
            self.goto_frame(v, from_slider=True)

    def goto_frame(self, frame_idx: int, from_slider: bool = False) -> None:
        if self._manual_bbox_mode and self._manual_bbox_frame_idx is not None and frame_idx != self._manual_bbox_frame_idx:
            self.exit_manual_bbox_mode()

        if frame_idx < 0:
            frame_idx = 0
        if frame_idx >= len(self.frames):
            frame_idx = len(self.frames) - 1
            self.action_pause()

        self.current_frame_idx = frame_idx
        if not from_slider:
            self.slider.blockSignals(True)
            self.slider.setValue(frame_idx)
            self.slider.blockSignals(False)
        
        # 更新帧数显示
        self.frame_label.setText(f"{frame_idx + 1}/{len(self.frames)}")

        # If split ready, switch segment on demand.
        if self._split_ready:
            new_seg = self._segment_idx_for_frame(frame_idx)
            if new_seg != self._current_segment_idx:
                self._switch_segment(new_seg, do_autocommit=True)
        # If split not ready yet, render image-only to keep UI responsive.
        if not self._split_ready:
            self.render_frame_image_only(frame_idx)
        else:
            self.render_frame(frame_idx)

        # auto-stop if target missing or tracking jump detected
        if self.target_id is not None:
            dets = self.detections_for_frame(frame_idx)
            target_det = next((d for d in dets if d.track_id == self.target_id), None)
            
            # 如果处于"不在画面内"模式
            if self._target_out_of_view:
                # 检查 target 是否重新出现
                if target_det is not None:
                    if self.playing:
                        self.action_pause()
                    self.prompt_target_reappear(frame_idx, target_det)
                return  # 在 out_of_view 模式下不进行其他检测
            
            if target_det is None:
                # target 不在当前帧
                self._last_target_bbox = None
                self._last_target_frame_idx = None
                if self.playing:
                    self.action_pause()
                self.prompt_target_missing(frame_idx)
            else:
                # target 存在，检查是否发生跟踪跳变
                current_bbox = target_det.bbox_ltwh
                if (self._last_target_bbox is not None and 
                    self._last_target_frame_idx is not None and
                    abs(frame_idx - self._last_target_frame_idx) == 1):
                    # 只在连续帧之间检测跳变
                    iou = self._bbox_iou(self._last_target_bbox, current_bbox)
                    if iou < 0.01:  # IoU < 1% 认为没有交集，发生了跳变
                        if self.playing:
                            self.action_pause()
                        self.prompt_tracking_jump(frame_idx, target_det)
                        return
                
                # 更新上一帧的 target bbox
                self._last_target_bbox = current_bbox
                self._last_target_frame_idx = frame_idx

    def render_frame(self, frame_idx: int) -> None:
        fi = self.frames[frame_idx]
        pm = QPixmap(str(fi.path))
        dets = self.detections_for_frame(frame_idx)
        self.canvas.set_frame(pm, dets)
        track_ids = sorted(set(d.track_id for d in dets))
        has_target = self.target_id in track_ids if self.target_id is not None else None
        self.set_status(f"frame={frame_idx} image_id={fi.image_id}  dets={len(dets)}  target={self.target_id}  has_target={has_target}")

    # --- background auto commit ---
    def _autocommit_segment_async(self, segment_json: Path, target_seg_idx: int | None = None) -> None:
        if self._autocommit_inflight:
            return

        # 确定当前要写回的分片索引
        commit_seg_idx = target_seg_idx if target_seg_idx is not None else self._current_segment_idx
        seg_start = commit_seg_idx * self.segment_size
        seg_end = seg_start + self.segment_size - 1

        # 过滤：只保留属于当前分片的数据
        all_manual_bboxes = list(self.annotations.manual_bboxes)
        all_delete_rules = list(self.annotations.delete_rules)
        all_sam_requests = list(self.annotations.sam_requests)
        all_missing_events = list(self.annotations.missing_events)

        # 筛选属于当前分片的 manual_bboxes
        manual_bboxes = [b for b in all_manual_bboxes if seg_start <= int(b.frame_idx) <= seg_end]
        
        # 筛选属于当前分片的 delete_rules
        delete_rules = []
        delete_rules_remaining = []
        for r in all_delete_rules:
            if r.kind == "track_id":
                # track_id 删除规则应用于所有分片，但只在当前分片执行一次
                delete_rules.append(r)
            elif r.kind == "single" and r.image_id is not None:
                fidx = self.frame_idx_by_image_id.get(str(r.image_id))
                if fidx is not None and seg_start <= fidx <= seg_end:
                    delete_rules.append(r)
                else:
                    delete_rules_remaining.append(r)
            elif r.kind == "frame_range":
                r_from = r.from_frame_idx if r.from_frame_idx is not None else 0
                r_to = r.to_frame_idx if r.to_frame_idx is not None else seg_end
                if r_from <= seg_end and r_to >= seg_start:
                    delete_rules.append(r)
                else:
                    delete_rules_remaining.append(r)
            else:
                delete_rules_remaining.append(r)
        
        # 筛选属于当前分片的 sam_requests 和 missing_events
        sam_requests = [s for s in all_sam_requests if seg_start <= int(s.frame_idx) <= seg_end]
        missing_events = [e for e in all_missing_events if seg_start <= int(e.frame_idx) <= seg_end]

        # 保留不属于当前分片的数据（稍后写回 annotation）
        remaining_bboxes = [b for b in all_manual_bboxes if not (seg_start <= int(b.frame_idx) <= seg_end)]
        remaining_sam_requests = [s for s in all_sam_requests if not (seg_start <= int(s.frame_idx) <= seg_end)]
        remaining_missing_events = [e for e in all_missing_events if not (seg_start <= int(e.frame_idx) <= seg_end)]

        if not manual_bboxes and not delete_rules and not sam_requests and not missing_events:
            return

        # Archive snapshot for safety
        try:
            archive_dir = self.root / "annotation_archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            snap_path = archive_dir / f"seg_{commit_seg_idx:05d}.json"
            payload = {
                "target_id": self.annotations.target_id,
                "id_remap_rules": [r.__dict__ for r in self.annotations.id_remap_rules],
                "sam_requests": [r.__dict__ for r in sam_requests],
                "manual_bboxes": [b.__dict__ for b in manual_bboxes],
                "delete_rules": [r.__dict__ for r in delete_rules],
                "missing_events": [e.__dict__ for e in missing_events],
            }
            snap_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        # 只清除已写回的数据，保留未写回的数据
        self.annotations.manual_bboxes = remaining_bboxes
        self.annotations.delete_rules = delete_rules_remaining
        self.annotations.sam_requests = remaining_sam_requests
        self.annotations.missing_events = remaining_missing_events
        self.annotations.save()

        self._autocommit_inflight = True
        self.set_status(f"后台写回上一片：{segment_json.name} …")

        self._autocommit_thread = QThread(self)
        self._autocommit_worker = _AutoCommitWorker(
            segment_json=segment_json,
            manual_bboxes=manual_bboxes,
            delete_rules=delete_rules,
            id_remap_rules=list(self.annotations.id_remap_rules),
            frame_idx_by_image_id=self.frame_idx_by_image_id,
        )
        self._autocommit_worker.moveToThread(self._autocommit_thread)
        self._autocommit_thread.started.connect(self._autocommit_worker.run)
        self._autocommit_worker.finished.connect(self._on_autocommit_finished)
        self._autocommit_worker.finished.connect(self._autocommit_thread.quit)
        self._autocommit_worker.finished.connect(self._autocommit_worker.deleteLater)
        self._autocommit_thread.finished.connect(self._autocommit_thread.deleteLater)
        self._autocommit_thread.start()

    def _on_autocommit_finished(self, summary: object, err: object, segment_json: object) -> None:
        self._autocommit_inflight = False
        self._autocommit_thread = None
        self._autocommit_worker = None

        # 删除该 segment 的缓存文件，强制下次访问时重新解析更新后的 JSON
        if segment_json is not None:
            try:
                cache_path = self._cache_path_for(Path(segment_json))
                if cache_path.exists():
                    cache_path.unlink()
                    print(f"[DEBUG] 已删除缓存文件: {cache_path}")
                # 如果当前正在查看这个 segment，刷新 track_store
                if self.track_store.tracking_json == Path(segment_json):
                    self.track_store = TrackStore(Path(segment_json), cache_path)
                    self.render_frame(self.current_frame_idx)
            except Exception as e:
                print(f"[DEBUG] 删除缓存文件失败: {e}")

        # 重建删除索引（因为 delete_rules 已被清空）
        self._rebuild_delete_index()

        if err is not None:
            QMessageBox.warning(self, "后台写回失败", f"后台写回失败：{err}\n已将本片标注快照保存到 annotation_archive")
            self.set_status("后台写回失败")
            return
        if summary is not None:
            try:
                self.set_status(
                    f"后台写回完成：新增={summary.added} 删除={summary.removed} 替换={summary.replaced} 备份={summary.backup_path.name}"
                )
            except Exception:
                self.set_status("后台写回完成")

    def render_frame_image_only(self, frame_idx: int) -> None:
        fi = self.frames[frame_idx]
        pm = QPixmap(str(fi.path))
        self.canvas.set_frame(pm, [])
        self.set_status(f"frame={frame_idx} image_id={fi.image_id}  dets=?  target={self.target_id}")

    # --- interactions ---
    def prompt_target_missing(self, frame_idx: int) -> None:
        fi = self.frames[frame_idx]
        msg = QMessageBox(self)
        msg.setWindowTitle("Target 缺失")
        msg.setText(f"当前帧缺少 target_id={self.target_id}\nframe={frame_idx} image_id={fi.image_id}\n请选择原因与操作")

        btn_out = msg.addButton("1) 不在画面内（我来跳帧）", QMessageBox.ButtonRole.ActionRole)
        btn_not = msg.addButton("2) 未检测到（手动标注 / SAM）", QMessageBox.ButtonRole.ActionRole)
        btn_sw = msg.addButton("3) ID变了（点人改ID）", QMessageBox.ButtonRole.ActionRole)
        msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == btn_out:
            self.annotations.add_missing_event(TargetMissingEvent(frame_idx=frame_idx, image_id=fi.image_id, reason="out_of_view"))
            self._target_out_of_view = True
            self.set_status("不在画面内：请跳帧到目标重新出现的位置，然后点击「添加标注」")
            return

        if clicked == btn_not:
            self.annotations.add_missing_event(TargetMissingEvent(frame_idx=frame_idx, image_id=fi.image_id, reason="not_detected"))
            self.start_not_detected_flow(frame_idx)
            return

        if clicked == btn_sw:
            self.annotations.add_missing_event(TargetMissingEvent(frame_idx=frame_idx, image_id=fi.image_id, reason="id_switched"))
            self._awaiting_remap_click = True
            self.set_status("ID变了：请点击目标球员框来做ID映射")
            return

    def prompt_target_reappear(self, frame_idx: int, target_det: Detection) -> None:
        """Target 在"不在画面内"模式下重新出现，让标注员确认是否正确"""
        fi = self.frames[frame_idx]
        msg = QMessageBox(self)
        msg.setWindowTitle("Target 重新出现")
        msg.setText(
            f"target_id={self.target_id} 在帧 {frame_idx} 重新出现了。\n\n"
            f"请确认这是否是正确的目标球员？"
        )

        btn_correct = msg.addButton("正确（继续跟踪）", QMessageBox.ButtonRole.AcceptRole)
        btn_wrong = msg.addButton("错误（映射为404 + 手动标注）", QMessageBox.ButtonRole.DestructiveRole)
        btn_cancel = msg.addButton("取消（保持不在画面内状态）", QMessageBox.ButtonRole.RejectRole)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == btn_correct:
            # 正确：退出 out_of_view 模式，继续跟踪
            self._target_out_of_view = False
            self._last_target_bbox = target_det.bbox_ltwh
            self._last_target_frame_idx = frame_idx
            self.set_status(f"已确认 target_id={self.target_id} 正确，继续跟踪")
            # 继续播放
            self.action_play()
            return

        if clicked == btn_wrong:
            # 错误：将当前帧起的 target_id 映射为 404（这样之后的都会被视为 404）
            if self.target_id is None:
                self.set_status("错误：target_id 未设置")
                return
            
            target_id_int = int(self.target_id)
            
            # 1. 添加 ID 映射规则：将当前帧起的 target_id 映射为 404
            # 这会让原始跟踪数据中的错误ID显示为404
            rule = IdRemapRule(from_frame_idx=frame_idx, old_id=target_id_int, new_id=404)
            self.id_remap_rules.append(rule)
            self.annotations.add_remap_rule(rule)
            
            # 2. 删除当前帧中这个错误的 bbox（使用 single 删除规则）
            # 这样当前帧的错误bbox会被彻底删除，不会显示为404
            delete_rule = BboxDeleteRule(
                kind="single",
                track_id=float(404),  # 映射后的ID是404
                image_id=fi.image_id,
                bbox_ltwh=target_det.bbox_ltwh,
            )
            self.annotations.add_delete_rule(delete_rule)
            self._rebuild_delete_index()
            
            # 记录事件
            self.annotations.add_missing_event(
                TargetMissingEvent(frame_idx=frame_idx, image_id=fi.image_id, reason="wrong_reappear")
            )
            
            # 注意：不退出 out_of_view 模式，因为目标可能仍然不在画面内
            # 用户需要继续跳帧查找真正的目标重新出现的位置
            # self._target_out_of_view = False  # 保持 True
            self._last_target_bbox = None
            self._last_target_frame_idx = None
            
            # 保存到 annotations.json
            self.annotations.save()
            
            # 重新渲染（应用 ID 映射后 target 会变成 404）
            self.render_frame(frame_idx)
            
            # 不弹出手动标注弹窗，让用户继续跳帧查找
            self.set_status(f"已删除帧 {frame_idx} 的错误bbox，并将之后的 ID {target_id_int} 映射为 404。请继续跳帧查找目标真正重新出现的位置。")
            return

        # 取消或关闭对话框：保持 out_of_view 状态，用户可以继续跳帧查找
        self.set_status("已取消，保持不在画面内状态。可继续跳帧查找目标重新出现的位置。")

    def prompt_tracking_jump(self, frame_idx: int, wrong_det: Detection) -> None:
        """检测到跟踪跳变：target_id 的 bbox 与上一帧没有交集"""
        fi = self.frames[frame_idx]
        msg = QMessageBox(self)
        msg.setWindowTitle("跟踪跳变检测")
        msg.setText(
            f"检测到可能的跟踪错误！\n"
            f"target_id={self.target_id} 在帧 {frame_idx} 的位置与上一帧没有交集。\n"
            f"这通常意味着 ID 跳到了另一个球员身上。\n\n"
            f"是否将当前帧起的错误 ID 映射为 404，并手动添加正确的标注？"
        )

        btn_fix = msg.addButton("修正（映射为404 + 手动标注）", QMessageBox.ButtonRole.ActionRole)
        btn_ignore = msg.addButton("忽略（继续跟踪）", QMessageBox.ButtonRole.RejectRole)
        btn_cancel = msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == btn_fix:
            if self.target_id is None:
                self.set_status("错误：target_id 未设置")
                return
            
            target_id_int = int(self.target_id)
            
            # 1. 将当前帧起的 target_id 映射为 404
            rule = IdRemapRule(from_frame_idx=frame_idx, old_id=target_id_int, new_id=404)
            self.id_remap_rules.append(rule)
            self.annotations.add_remap_rule(rule)
            
            # 2. 记录跳变事件
            self.annotations.add_missing_event(
                TargetMissingEvent(frame_idx=frame_idx, image_id=fi.image_id, reason="tracking_jump")
            )
            
            # 3. 清除上一帧的 bbox 记录
            self._last_target_bbox = None
            self._last_target_frame_idx = None
            
            # 4. 保存到 annotations.json
            self.annotations.save()
            
            # 5. 重新渲染（应用 ID 映射后 target 会消失）
            self.render_frame(frame_idx)
            
            # 6. 进入手动标注流程
            self.set_status(f"已将帧 {frame_idx} 起的 ID {target_id_int} 映射为 404，请手动标注正确的目标")
            self.start_not_detected_flow(frame_idx)
            return

        if clicked == btn_ignore:
            # 更新 bbox 继续跟踪
            self._last_target_bbox = wrong_det.bbox_ltwh
            self._last_target_frame_idx = frame_idx
            self.set_status("已忽略跳变警告，继续跟踪")
            self.action_play()  # 继续播放
            return
        
        # 取消：暂停在当前帧，不做任何处理
        self.set_status("已取消，暂停在当前帧")

    def start_not_detected_flow(self, frame_idx: int) -> None:
        # In target-missing flow we already have a locked target_id.
        if self.target_id is None:
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("未检测到")
        msg.setText("请选择处理方式：")
        btn_manual = msg.addButton("手动标注当前帧bbox", QMessageBox.ButtonRole.ActionRole)
        btn_sam = msg.addButton("SAM自动跟踪（向后）", QMessageBox.ButtonRole.ActionRole)
        msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_manual:
            # In not-detected flow, target_id is already locked and should be used.
            self.enter_manual_bbox_mode(frame_idx, prompt_track_id_on_confirm=False)
            self.set_status("手动标注：拖拽框选bbox后点击“确认标注”")
            return

        if clicked == btn_sam:
            self.start_sam_flow(frame_idx=frame_idx, default_forward_frames=150, prompt_track_id_on_confirm=False)
            return

    def start_add_annotation_flow(self, frame_idx: int) -> None:
        msg = QMessageBox(self)
        msg.setWindowTitle("添加标注")
        msg.setText("请选择添加方式：")
        btn_manual = msg.addButton("单帧添加（手动框bbox）", QMessageBox.ButtonRole.ActionRole)
        btn_sam = msg.addButton("SAM添加（向后跟踪）", QMessageBox.ButtonRole.ActionRole)
        msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_manual:
            # For ad-hoc adding, ALWAYS ask operator for track_id on confirm.
            self.enter_manual_bbox_mode(frame_idx, prompt_track_id_on_confirm=True)
            self.set_status("手动标注：拖拽框选bbox后点击“确认标注”（将弹窗输入track_id）")
            return

        if clicked == btn_sam:
            # For ad-hoc SAM, ALWAYS prompt track_id on confirm (even if target is locked).
            self.start_sam_flow(frame_idx=frame_idx, default_forward_frames=150, prompt_track_id_on_confirm=True)
            return

    def _ensure_sam_server_running(self) -> None:
        # If already healthy, do nothing.
        try:
            if Sam2Client(self._sam_base_url).health(timeout_s=0.2):
                return
        except Exception:
            pass

        if self._sam_proc is not None:
            # Already starting.
            return

        # Start local SAM2 server (python sam2-server.py).
        self._sam_proc = QProcess(self)
        self._sam_proc.setProgram(sys.executable)
        self._sam_proc.setArguments([str(self.root / "sam2-server.py")])
        self._sam_proc.setWorkingDirectory(str(self.root))
        self._sam_proc.start()
        self.set_status("正在启动 SAM 服务…（首次加载模型可能需要 10-30 秒）")

    def _sam_forward_frames_within_segment(self, start_frame_idx: int, requested: int, *, allow_handoff_frame: bool = True) -> int:
        # UX requirement: SAM tracking must be bounded for responsiveness.
        # Hard cap: 150 frames total (including the start frame).
        req = max(1, int(requested))
        req = min(req, 150)

        # Always clamp to remaining frames in the dataset.
        remaining = (len(self.frames) - 1) - int(start_frame_idx) + 1
        if remaining <= 0:
            return 1
        req = min(req, int(remaining))

        # Enforce segment boundary even before split is ready:
        # segment idx is based on segment_size.
        try:
            seg_idx = int(start_frame_idx) // int(self.segment_size)
            seg_end = (seg_idx + 1) * int(self.segment_size) - 1
            boundary_end = int(seg_end)
            if allow_handoff_frame:
                boundary_end = min(boundary_end + 1, len(self.frames) - 1)
            max_in_seg = int(boundary_end) - int(start_frame_idx) + 1
            if max_in_seg > 0:
                req = min(req, int(max_in_seg))
        except Exception:
            pass

        # If split is ready, further clamp to manifest's actual end_frame_idx (in case the last segment is shorter).
        if not self._split_ready or self._split_manifest_path is None:
            return req

        try:
            m = load_manifest(self._split_manifest_path)
            seg_idx = self._segment_idx_for_frame(int(start_frame_idx))
            seg = next((s for s in m.segments if int(s.seg_idx) == int(seg_idx)), None)
            if seg is None:
                return req
            boundary_end = int(seg.end_frame_idx)
            if allow_handoff_frame:
                boundary_end = min(boundary_end + 1, len(self.frames) - 1)
            max_in_seg = int(boundary_end) - int(start_frame_idx) + 1
            if max_in_seg <= 0:
                return 1
            return min(req, max_in_seg)
        except Exception:
            return req

    def start_sam_flow(self, *, frame_idx: int, default_forward_frames: int = 150, prompt_track_id_on_confirm: bool = False) -> None:
        # UX requirement: SAM is fixed to a 150-frame window (or less near boundaries/end).
        requested = int(default_forward_frames)
        requested = min(max(1, requested), 150)
        safe_forward = self._sam_forward_frames_within_segment(frame_idx, requested, allow_handoff_frame=True)

        # Enter bbox draw mode and mark as SAM pending.
        self._sam_pending = {
            "frame_idx": int(frame_idx),
            "forward_frames": int(safe_forward),
        }
        self.enter_manual_bbox_mode(frame_idx, prompt_track_id_on_confirm=bool(prompt_track_id_on_confirm))
        self.set_status(f"[SAM] 请在当前帧拖拽框选目标bbox，然后点“确认标注”开始向后跟踪 {safe_forward} 帧")

    def enter_manual_bbox_mode(self, frame_idx: int, prompt_track_id_on_confirm: bool = False) -> None:
        self.action_pause()

        # Determine track_id for this manual annotation.
        # - If this flow requires prompting (e.g. "添加标注"), defer to confirm-time input.
        # - Otherwise, if target is locked, use it.
        self._manual_bbox_prompt_track_id_on_confirm = bool(prompt_track_id_on_confirm)

        chosen_track_id: float | None = None
        if not self._manual_bbox_prompt_track_id_on_confirm and self.target_id is not None:
            chosen_track_id = float(self.target_id)

        self._manual_bbox_mode = True
        self._manual_bbox_frame_idx = frame_idx
        self._manual_bbox_track_id = chosen_track_id
        self.canvas.enable_bbox_mode(True)
        self.act_confirm_bbox.setVisible(True)
        self.act_confirm_bbox.setEnabled(True)
        self.act_cancel_bbox.setVisible(True)
        self.act_cancel_bbox.setEnabled(True)
        if self._manual_bbox_prompt_track_id_on_confirm or self.target_id is None:
            self.set_status("[手动标注] 拖拽框选后确认（将弹窗输入track_id）")
        else:
            self.set_status("[手动标注] 拖拽框选后确认")

    def exit_manual_bbox_mode(self) -> None:
        self._manual_bbox_mode = False
        self._manual_bbox_frame_idx = None
        self._manual_bbox_track_id = None
        self._manual_bbox_prompt_track_id_on_confirm = False
        self.canvas.enable_bbox_mode(False)
        self.act_confirm_bbox.setEnabled(False)
        self.act_confirm_bbox.setVisible(False)
        self.act_cancel_bbox.setEnabled(False)
        self.act_cancel_bbox.setVisible(False)

    def on_confirm_manual_bbox(self) -> None:
        if not self._manual_bbox_mode or self._manual_bbox_frame_idx is None:
            return

        # Ask operator for track_id when:
        # - adding ad-hoc annotation ("添加标注")
        # - OR target is not locked
        if self._manual_bbox_prompt_track_id_on_confirm or self.target_id is None:
            default_id = int(float(self.track_id_box.value()))
            # If currently locked, default to target_id for convenience, but do not force it.
            if self.target_id is not None:
                default_id = int(self.target_id)
            v, ok = QInputDialog.getInt(self, "输入 track_id", "track_id", value=default_id, min=1, max=10**9)
            if not ok:
                self.set_status("已取消：未输入 track_id")
                return
            self._manual_bbox_track_id = float(v)
            # Keep toolbar in sync for next annotation.
            self.track_id_box.blockSignals(True)
            self.track_id_box.setValue(float(v))
            self.track_id_box.blockSignals(False)
        else:
            self._manual_bbox_track_id = float(self.target_id)

        if self._manual_bbox_track_id is None:
            self.set_status("未设置 track_id，无法记录")
            return

        bbox = self.canvas.take_drawn_bbox_ltwh_image()
        if bbox is None:
            self.set_status("还没有框选bbox：请先拖拽框选再确认")
            return

        # If SAM flow is pending, kick off SAM tracking instead of single-frame write.
        if self._sam_pending is not None:
            pending = dict(self._sam_pending)
            self._sam_pending = None

            chosen_track_id = float(self._manual_bbox_track_id)

            start_frame_idx = int(pending.get("frame_idx", self._manual_bbox_frame_idx))
            forward_frames = int(pending.get("forward_frames", 150))
            forward_frames = self._sam_forward_frames_within_segment(start_frame_idx, forward_frames, allow_handoff_frame=True)

            fi = self.frames[start_frame_idx]
            # Log request for auditing/debug.
            try:
                self.annotations.add_sam_request(
                    SamRequest(
                        frame_idx=int(start_frame_idx),
                        image_id=fi.image_id,
                        bbox_ltwh=bbox,
                        forward_frames=int(forward_frames),
                    )
                )
            except Exception:
                pass

            self.exit_manual_bbox_mode()
            self.render_frame(self.current_frame_idx)

            self._run_sam_tracking(
                start_frame_idx=start_frame_idx,
                bbox_ltwh=bbox,
                forward_frames=forward_frames,
                track_id=float(chosen_track_id),
            )
            return

        fi = self.frames[self._manual_bbox_frame_idx]
        added_track_id = float(self._manual_bbox_track_id)
        self.annotations.add_manual_bbox(
            ManualBboxAnnotation(
                frame_idx=self._manual_bbox_frame_idx,
                image_id=fi.image_id,
                bbox_ltwh=bbox,
                track_id=added_track_id,
            )
        )
        self.exit_manual_bbox_mode()
        # 如果是为锁定的 target 添加标注，退出"不在画面内"模式，恢复正常检测
        if self.target_id is not None and int(added_track_id) == int(self.target_id):
            self._target_out_of_view = False
            # 同时更新 target bbox 记录
            self._last_target_bbox = bbox
            self._last_target_frame_idx = self.current_frame_idx
        # Refresh current frame so the newly written annotation is visible immediately.
        self.render_frame(self.current_frame_idx)
        self.set_status("已记录手动bbox标注（annotations.json）")

    def _run_sam_tracking(self, *, start_frame_idx: int, bbox_ltwh: tuple[float, float, float, float], forward_frames: int, track_id: float) -> None:
        # Ensure server process started (non-blocking). If server is still loading, request will fail and we will show error.
        self._ensure_sam_server_running()

        if self._sam_track_thread is not None:
            QMessageBox.warning(self, "SAM 忙", "SAM 正在运行中，请等待完成后再发起新的请求。")
            return

        self.set_status("SAM运行中：正在向后跟踪并生成bbox…")

        self._sam_track_thread = QThread(self)

        # Build the exact frame list for this request (keeps server memory bounded).
        start = int(start_frame_idx)
        end = min(len(self.frames), start + int(forward_frames))
        frame_paths = [str(fi.path) for fi in self.frames[start:end]]

        self._sam_track_worker = _SamTrackWorker(
            base_url=self._sam_base_url,
            video_path=str(self.frame_dir),
            frame_idx=int(start_frame_idx),
            obj_id=int(track_id),
            bbox_ltwh=bbox_ltwh,
            forward_frames=int(forward_frames),
            frame_paths=frame_paths,
            frames=list(self.frames),
            track_id=float(track_id),
        )
        self._sam_track_worker.moveToThread(self._sam_track_thread)
        self._sam_track_thread.started.connect(self._sam_track_worker.run)
        self._sam_track_worker.finished.connect(self._on_sam_tracking_finished)
        self._sam_track_worker.finished.connect(self._sam_track_thread.quit)
        self._sam_track_worker.finished.connect(self._sam_track_worker.deleteLater)
        self._sam_track_thread.finished.connect(self._sam_track_thread.deleteLater)
        self._sam_track_thread.start()

    def _on_sam_tracking_finished(self, result: object, err: object) -> None:
        self._sam_track_thread = None
        self._sam_track_worker = None

        if err is not None:
            QMessageBox.warning(self, "SAM 失败", f"SAM 跟踪失败：{err}")
            self.set_status("SAM 跟踪失败")
            return

        try:
            payload = result or {}
            bboxes = list(payload.get("bboxes") or [])
            start_frame_idx = int(payload.get("start_frame_idx"))
            forward_frames = int(payload.get("forward_frames") or 0)
        except Exception:
            bboxes = []
            start_frame_idx = self.current_frame_idx
            forward_frames = 0

        if not bboxes:
            self.set_status("SAM 完成：未生成任何bbox")
            return

        # If split is ready, keep the segment-end+1 handoff bbox for the next segment
        # so that auto-commit of the current segment doesn't write it into the wrong seg JSON.
        seg_end = None
        start_seg_idx = None
        if self._split_ready:
            try:
                start_seg_idx = self._segment_idx_for_frame(int(start_frame_idx))
                seg_end = (int(start_seg_idx) + 1) * int(self.segment_size) - 1
                if self._split_manifest_path is not None:
                    m = load_manifest(self._split_manifest_path)
                    seg = next((s for s in m.segments if int(s.seg_idx) == int(start_seg_idx)), None)
                    if seg is not None:
                        seg_end = int(seg.end_frame_idx)
            except Exception:
                seg_end = None
                start_seg_idx = None

        bboxes_curr: list[ManualBboxAnnotation] = []
        bboxes_next: list[ManualBboxAnnotation] = []
        handoff_intended = False
        if seg_end is not None:
            try:
                end_frame_idx = int(start_frame_idx) + int(max(1, forward_frames)) - 1
                handoff_intended = end_frame_idx > int(seg_end)
            except Exception:
                handoff_intended = False
            for ann in bboxes:
                try:
                    if int(ann.frame_idx) <= int(seg_end):
                        bboxes_curr.append(ann)
                    else:
                        bboxes_next.append(ann)
                except Exception:
                    continue
        else:
            bboxes_curr = list(bboxes)

        for ann in bboxes_curr:
            try:
                self.annotations.add_manual_bbox(ann)
            except Exception:
                continue

        if bboxes_next:
            self._sam_handoff_bboxes.extend(bboxes_next)

        # SAM 跟踪完成后，如果是为锁定的 target 添加的标注，退出"不在画面内"模式
        if bboxes_curr and self.target_id is not None:
            try:
                track_id = float(payload.get("track_id"))
                if int(track_id) == int(self.target_id):
                    self._target_out_of_view = False
            except Exception:
                pass

        # Start background commit for the segment we just finished tracking.
        if handoff_intended and self._split_ready and start_seg_idx is not None:
            seg_path = self._segment_path(int(start_seg_idx))
            if seg_path is not None:
                self._autocommit_segment_async(seg_path, target_seg_idx=int(start_seg_idx))

        self.render_frame(self.current_frame_idx)
        if handoff_intended:
            self.set_status(
                f"SAM 完成：本片 {len(bboxes_curr)} 帧bbox；跨片handoff {len(bboxes_next)} 帧bbox（已启动后台写回）"
            )
        else:
            self.set_status(f"SAM 完成：已生成 {len(bboxes_curr)} 帧bbox（写入 annotations.json）")

    def on_cancel_manual_bbox(self) -> None:
        if not self._manual_bbox_mode:
            return
        self.exit_manual_bbox_mode()

    def on_detection_clicked(self, det: Detection) -> None:
        self._set_selected_det(det)
        self.set_status(f"已选中：image_id={det.image_id} track_id={det.track_id} role={det.role}")

        if not self._awaiting_remap_click:
            return
        self._awaiting_remap_click = False

        if self.target_id is None:
            return
        if det.track_id == self.target_id:
            return

        r = QMessageBox.question(
            self,
            "修改track id",
            f"将当前点击的 track_id={det.track_id} 从当前帧起映射为 target_id={self.target_id}？\n（不会改原始大JSON，只写 annotations.json）",
        )
        if r != QMessageBox.StandardButton.Yes:
            return

        rule = IdRemapRule(from_frame_idx=self.current_frame_idx, old_id=det.track_id, new_id=int(self.target_id))
        self.id_remap_rules.append(rule)
        self.annotations.add_remap_rule(rule)
        self.render_frame(self.current_frame_idx)

    def action_delete_selected_one(self) -> None:
        det = self._selected_det
        if det is None:
            return

        # If it's a manual bbox, remove it from annotations immediately.
        if det.role == "manual":
            removed = self.annotations.delete_manual_bbox_single(
                frame_idx=self.current_frame_idx,
                image_id=det.image_id,
                track_id=float(det.track_id),
                bbox_ltwh=det.bbox_ltwh,
            )
            self.set_status(f"已删除手动标注 {removed} 条（当前框）")
        else:
            self.annotations.add_delete_rule(
                BboxDeleteRule(kind="single", track_id=float(det.track_id), image_id=det.image_id, bbox_ltwh=det.bbox_ltwh)
            )
            self.set_status("已记录删除（当前框）。点“写回原JSON”后会从原文件删除")

        self._rebuild_delete_index()
        self.render_frame(self.current_frame_idx)

    def action_delete_selected_track(self) -> None:
        det = self._selected_det
        if det is None:
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("删除该ID的框")
        msg.setText(f"选择删除 track_id={det.track_id} 的范围：")
        
        btn_all = msg.addButton("删除所有帧中该ID的框", QMessageBox.ButtonRole.DestructiveRole)
        btn_from_current = msg.addButton("仅删除当前帧及之后的", QMessageBox.ButtonRole.ActionRole)
        msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        
        msg.exec()
        clicked = msg.clickedButton()
        
        if clicked == btn_all:
            # 删除所有帧
            removed_manual = self.annotations.delete_manual_bboxes_by_track_id(float(det.track_id))
            self.annotations.add_delete_rule(BboxDeleteRule(kind="track_id", track_id=float(det.track_id)))
            self._rebuild_delete_index()
            print(f"[DEBUG] action_delete_selected_track (all): track_id={det.track_id}, _deleted_track_ids={self._deleted_track_ids}")
            self.render_frame(self.current_frame_idx)
            self.set_status(f"已删除 track_id={det.track_id} 的所有框（手动标注 {removed_manual} 条）")
            return
        
        if clicked == btn_from_current:
            # 只删除当前帧及之后：使用 ID remap 机制（映射为 404）
            rule = IdRemapRule(from_frame_idx=self.current_frame_idx, old_id=det.track_id, new_id=404)
            self.id_remap_rules.append(rule)
            self.annotations.add_remap_rule(rule)
            self.annotations.save()
            self.render_frame(self.current_frame_idx)
            self.set_status(f"已将 track_id={det.track_id} 从帧 {self.current_frame_idx} 起映射为 404（之前的帧不受影响）")
            return

    def action_remap_selected_to_target(self) -> None:
        """将选中的 bbox 从当前帧起映射为用户输入的 ID"""
        det = self._selected_det
        if det is None:
            QMessageBox.warning(self, "提示", "请先选择一个bbox")
            return

        # 默认目标ID为 target_id（如果已设置），否则为 404
        default_id = self.target_id if self.target_id is not None else 404
        
        new_id, ok = QInputDialog.getInt(
            self,
            "映射ID",
            f"将 track_id={det.track_id} 从当前帧（帧 {self.current_frame_idx}）起映射为：",
            value=int(default_id),
            min=1,
            max=10**9
        )
        if not ok:
            return
        
        if new_id == det.track_id:
            QMessageBox.information(self, "提示", "目标ID与当前ID相同，无需映射")
            return

        rule = IdRemapRule(from_frame_idx=self.current_frame_idx, old_id=det.track_id, new_id=int(new_id))
        self.id_remap_rules.append(rule)
        self.annotations.add_remap_rule(rule)
        self.render_frame(self.current_frame_idx)
        self.set_status(f"已将 track_id={det.track_id} 从帧 {self.current_frame_idx} 起映射为 {new_id}")
