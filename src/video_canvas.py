from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from .models import Detection


@dataclass
class HitTestResult:
    det: Detection


class VideoCanvas(QWidget):
    detectionClicked = pyqtSignal(object)  # emits Detection

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._detections: list[Detection] = []
        self._selected_track_id: Optional[int] = None
        self._target_id: Optional[int] = None  # 锁定的目标ID

        self._drawing_bbox: bool = False
        self._bbox_start: Optional[QPoint] = None
        self._bbox_rect: Optional[QRect] = None
        self._bbox_mode_enabled: bool = False

        self.setMouseTracking(True)

    def set_frame(self, pixmap: QPixmap, detections: list[Detection]) -> None:
        self._pixmap = pixmap
        self._detections = detections
        self.update()

    def set_selected_track(self, track_id: Optional[int]) -> None:
        self._selected_track_id = track_id
        self.update()

    def set_target_id(self, target_id: Optional[int]) -> None:
        self._target_id = target_id
        self.update()

    def enable_bbox_mode(self, enabled: bool) -> None:
        self._bbox_mode_enabled = enabled
        self._drawing_bbox = False
        self._bbox_start = None
        self._bbox_rect = None
        self.update()

    def take_drawn_bbox_ltwh(self) -> Optional[tuple[float, float, float, float]]:
        if self._bbox_rect is None:
            return None
        r = self._bbox_rect.normalized()
        self._bbox_rect = None
        self.update()
        return (float(r.x()), float(r.y()), float(r.width()), float(r.height()))

    def take_drawn_bbox_ltwh_image(self) -> Optional[tuple[float, float, float, float]]:
        if self._bbox_rect is None or self._pixmap is None:
            return None
        if self._pixmap.width() <= 0 or self._pixmap.height() <= 0:
            return None

        scaled, xoff, yoff = self._scale_to_widget(self._pixmap)
        if scaled.width() <= 0:
            return None
        scale = scaled.width() / self._pixmap.width()
        if scale <= 0:
            return None

        r = self._bbox_rect.normalized()
        x1, y1 = self._widget_to_img_f(r.topLeft(), scale, xoff, yoff)
        x2, y2 = self._widget_to_img_f(r.bottomRight(), scale, xoff, yoff)

        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        # clamp to image bounds
        x = max(0.0, min(x, float(self._pixmap.width() - 1)))
        y = max(0.0, min(y, float(self._pixmap.height() - 1)))
        w = max(0.0, min(w, float(self._pixmap.width()) - x))
        h = max(0.0, min(h, float(self._pixmap.height()) - y))

        self._bbox_rect = None
        self.update()
        return (float(x), float(y), float(w), float(h))

    def _scale_to_widget(self, pm: QPixmap) -> tuple[QPixmap, float, float]:
        # Keep aspect ratio; compute offsets for letterboxing
        if self.width() <= 0 or self.height() <= 0:
            return pm, 0.0, 0.0
        scaled = pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        xoff = (self.width() - scaled.width()) / 2.0
        yoff = (self.height() - scaled.height()) / 2.0
        return scaled, xoff, yoff

    def _img_to_widget(self, x: float, y: float, scale: float, xoff: float, yoff: float) -> QPoint:
        return QPoint(int(x * scale + xoff), int(y * scale + yoff))

    def _widget_to_img(self, p: QPoint, scale: float, xoff: float, yoff: float) -> QPoint:
        return QPoint(int((p.x() - xoff) / scale), int((p.y() - yoff) / scale))

    def _widget_to_img_f(self, p: QPoint, scale: float, xoff: float, yoff: float) -> tuple[float, float]:
        return ((p.x() - xoff) / scale, (p.y() - yoff) / scale)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self._pixmap is None:
            painter.end()
            return

        scaled, xoff, yoff = self._scale_to_widget(self._pixmap)
        painter.drawPixmap(int(xoff), int(yoff), scaled)

        if self._pixmap.width() <= 0:
            painter.end()
            return
        scale = scaled.width() / self._pixmap.width()

        painter.setFont(QFont("Segoe UI", 10))

        # 分离目标ID的检测框，确保最后绘制（在顶层）
        target_dets = []
        other_dets = []
        for det in self._detections:
            if self._target_id is not None and det.track_id == self._target_id:
                target_dets.append(det)
            else:
                other_dets.append(det)

        # 先绘制非目标ID的检测框
        for det in other_dets:
            x, y, w, h = det.bbox_ltwh
            p1 = self._img_to_widget(x, y, scale, xoff, yoff)
            p2 = self._img_to_widget(x + w, y + h, scale, xoff, yoff)
            rect = QRect(p1, p2)

            is_sel = (self._selected_track_id is not None and det.track_id == self._selected_track_id)
            pen = QPen(Qt.GlobalColor.yellow if is_sel else Qt.GlobalColor.green)
            pen.setWidth(3 if is_sel else 2)
            painter.setPen(pen)
            painter.drawRect(rect)

            painter.setPen(QPen(Qt.GlobalColor.white))
            painter.drawText(rect.topLeft() + QPoint(2, 12), f"{det.track_id} {det.role}")

        # 最后绘制目标ID的检测框（红色、加粗、顶层）
        for det in target_dets:
            x, y, w, h = det.bbox_ltwh
            p1 = self._img_to_widget(x, y, scale, xoff, yoff)
            p2 = self._img_to_widget(x + w, y + h, scale, xoff, yoff)
            rect = QRect(p1, p2)

            pen = QPen(Qt.GlobalColor.red)
            pen.setWidth(6)  # 加粗一倍（原来是3，现在是6）
            painter.setPen(pen)
            painter.drawRect(rect)

            painter.setPen(QPen(Qt.GlobalColor.white))
            painter.drawText(rect.topLeft() + QPoint(2, 12), f"{det.track_id} {det.role}")

        if self._bbox_mode_enabled and self._bbox_rect is not None:
            pen = QPen(Qt.GlobalColor.cyan)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self._bbox_rect.normalized())

        painter.end()

    def _hit_test(self, pos: QPoint) -> Optional[HitTestResult]:
        if self._pixmap is None:
            return None
        scaled, xoff, yoff = self._scale_to_widget(self._pixmap)
        if self._pixmap.width() <= 0:
            return None
        scale = scaled.width() / self._pixmap.width()

        img_p = self._widget_to_img(pos, scale, xoff, yoff)
        x0, y0 = img_p.x(), img_p.y()

        for det in reversed(self._detections):
            x, y, w, h = det.bbox_ltwh
            if x <= x0 <= x + w and y <= y0 <= y + h:
                return HitTestResult(det=det)
        return None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._bbox_mode_enabled:
            self._drawing_bbox = True
            self._bbox_start = e.position().toPoint()
            self._bbox_rect = QRect(self._bbox_start, self._bbox_start)
            self.update()
            return

        if e.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_test(e.position().toPoint())
            if hit is not None:
                self.detectionClicked.emit(hit.det)

    def mouseMoveEvent(self, e):
        if self._drawing_bbox and self._bbox_start is not None:
            self._bbox_rect = QRect(self._bbox_start, e.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._drawing_bbox:
            self._drawing_bbox = False
            self.update()
