# ui/widgets.py
# ============================================================
#  自定义 PyQt6 控件 — 从 ui_builder.py 提取
# ============================================================

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap


import math

class FloatSlider(QWidget):
    """浮点滑块控件"""
    valueChanged = pyqtSignal(float)

    def __init__(self, minimum=0.0, maximum=1.0, step=0.01,
                 value=0.5, parent=None):
        super().__init__(parent)
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        self._factor = round(1 / step)
        self._minimum = minimum
        self._maximum = maximum
        # 根据 step 自适应小数位数
        self._decimals = max(0, -int(math.log10(step))) if step < 1 else 0

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(round(minimum * self._factor))
        self._slider.setMaximum(round(maximum * self._factor))
        self._slider.setValue(round(value * self._factor))
        self._slider.setFixedHeight(22)

        self._label = QLabel(f"{value:.{self._decimals}f}")
        self._label.setFixedWidth(40)
        self._label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._label.setStyleSheet(
            "color:#ffffff; font-family:Consolas; background:transparent;")

        self._layout.addWidget(self._slider)
        self._layout.addWidget(self._label)
        self._slider.valueChanged.connect(self._on_change)

    def _on_change(self, int_val):
        fval = int_val / self._factor
        self._label.setText(f"{fval:.{self._decimals}f}")
        self.valueChanged.emit(fval)

    def float_value(self) -> float:
        return self._slider.value() / self._factor

    def value(self) -> float:
        return self.float_value()

    def setValue(self, v: float):
        v = max(self._minimum, min(self._maximum, v))  # 限制在合法范围内
        self._slider.setValue(round(v * self._factor))

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)
        color = "#ffffff" if enabled else "#7d8187"
        self._label.setStyleSheet(
            f"color:{color}; font-family:Consolas; background:transparent;")


class GpuCanvas(QLabel):
    """自适应缩放画布（GPU 加速）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._raw_pixmap = None

    def set_pixmap(self, pixmap: QPixmap):
        self._raw_pixmap = pixmap
        self._refresh()

    def _refresh(self):
        if (self._raw_pixmap and not self._raw_pixmap.isNull()
                and self.width() > 0 and self.height() > 0):
            scaled = self._raw_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()


class CompareCanvas(QWidget):
    """修前/修后对比画布：左半显示原图、右半显示结果，拖拽分割线（类 PS before/after）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        self._ratio = 0.5          # 分割线位置 0..1
        self._dragging = False
        self.setMinimumHeight(200)

    def set_images(self, before: QPixmap, after: QPixmap):
        self._before = before
        self._after = after
        self.update()

    def has_images(self) -> bool:
        return bool(self._before and self._after
                    and not self._before.isNull() and not self._after.isNull())

    # ---------- 绘制 ----------
    def _fitted_rect(self, pm: QPixmap):
        from PyQt6.QtCore import QRect
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0 or pm.isNull():
            return None
        scaled = pm.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio)
        return QRect((w - scaled.width()) // 2, (h - scaled.height()) // 2,
                     scaled.width(), scaled.height())

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QColor, QPen
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(10, 14, 20))
        if not self.has_images():
            p.setPen(QColor("#8fa1b3"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "无对比图（需开启 Hires.fix / ADetailer 生成一次）")
            p.end()
            return
        rect = self._fitted_rect(self._after)
        before_scaled = self._before.scaled(
            rect.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        after_scaled = self._after.scaled(
            rect.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        split_x = rect.x() + int(rect.width() * self._ratio)
        # 左：原图
        p.setClipRect(rect.x(), rect.y(), split_x - rect.x(), rect.height())
        p.drawPixmap(rect.x(), rect.y(), before_scaled)
        # 右：结果
        p.setClipRect(split_x, rect.y(), rect.right() - split_x + 1, rect.height())
        p.drawPixmap(rect.x(), rect.y(), after_scaled)
        p.setClipping(False)
        # 分割线 + 手柄
        p.setPen(QPen(QColor("#4a9eff"), 2))
        p.drawLine(split_x, rect.y(), split_x, rect.bottom())
        p.setBrush(QColor("#4a9eff"))
        p.drawEllipse(split_x - 8, rect.center().y() - 8, 16, 16)
        p.setPen(QColor("#ffffff"))
        f = p.font(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        p.drawText(split_x - 6, rect.center().y() + 3, "◀▶")
        # 角落标签
        p.setPen(QColor("#ffffff"))
        f.setPointSize(10); p.setFont(f)
        p.drawText(rect.x() + 8, rect.y() + 20, "修前")
        p.drawText(rect.right() - 40, rect.y() + 20, "修后")
        p.end()

    # ---------- 拖拽 ----------
    def _set_ratio_from_x(self, x: int):
        self._ratio = max(0.0, min(1.0, x / max(1, self.width())))
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._set_ratio_from_x(e.position().x())

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._set_ratio_from_x(e.position().x())

    def mouseReleaseEvent(self, e):
        self._dragging = False