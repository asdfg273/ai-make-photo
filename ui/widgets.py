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