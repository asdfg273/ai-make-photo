# crop_overlay.py
"""
裁剪交互：使用 QRubberBand 在 QLabel 上绘制裁剪框。
- 支持鼠标按下-拖拽-释放来生成矩形。
- 支持固定比例约束（1:1 / 4:3 / 16:9）。
"""
from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLabel, QRubberBand


class CropLabel(QLabel):
    """扩展 QLabel，承载裁剪选框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._crop_mode = False
        self._origin = QPoint()
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._aspect_ratio = None  # None 或 (w, h)

    def set_crop_mode(self, enabled: bool):
        self._crop_mode = enabled
        if not enabled:
            self._rubber.hide()
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_aspect_ratio(self, ratio):
        """ratio: None 表示自由，或传入 (w, h)"""
        self._aspect_ratio = ratio

    def get_selection_rect(self) -> QRect:
        """返回当前选框在 QLabel 坐标系中的 QRect（像素）。"""
        if self._rubber.isVisible():
            return self._rubber.geometry()
        return QRect()

    # ---------- 事件 ----------
    def mousePressEvent(self, event):
        if self._crop_mode and event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._rubber.setGeometry(QRect(self._origin, QSize()))
            self._rubber.show()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._crop_mode and self._rubber.isVisible():
            current = event.position().toPoint()
            rect = QRect(self._origin, current).normalized()

            if self._aspect_ratio:
                aw, ah = self._aspect_ratio
                w = rect.width()
                h = int(w * ah / aw)
                if current.y() >= self._origin.y():
                    rect.setHeight(h)
                else:
                    rect.setTop(rect.bottom() - h)

            # 限制在 label 区域内
            rect = rect.intersected(self.rect())
            self._rubber.setGeometry(rect)
        super().mouseMoveEvent(event)