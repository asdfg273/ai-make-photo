# photo_turn/components.py
# ============================================================
#  PyQt6 GPU 加速画布（QOpenGLWidget）
# ============================================================

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtCore    import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui     import QPainter, QPixmap, QColor, QPen, QCursor


class ImageCanvas(QOpenGLWidget):
    """
    OpenGL 加速画布
    • GPU 渲染，彻底消除画笔卡顿
    • 正确处理单击出水
    """

    sig_press       = pyqtSignal(int, int)   # 鼠标按下
    sig_drag        = pyqtSignal(int, int)   # 拖拽移动
    sig_release     = pyqtSignal(int, int)   # 鼠标释放
    sig_right_click = pyqtSignal(int, int)   # 右键

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap        = None
        self._last_pos      = None
        self._is_pressing   = False

        self.setMouseTracking(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(400, 300)

    # ----------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.update()   # 触发 paintGL → GPU 渲染

    # ----------------------------------------------------------
    #  GPU 渲染
    # ----------------------------------------------------------
    def paintGL(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#1e1e2e"))

        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width()  - scaled.width())  // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        painter.end()

    # ----------------------------------------------------------
    #  坐标映射：屏幕坐标 → 图像坐标
    # ----------------------------------------------------------
    def _to_image_pos(self, sx: int, sy: int):
        if not self._pixmap:
            return sx, sy

        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        ox = (self.width()  - scaled.width())  // 2
        oy = (self.height() - scaled.height()) // 2

        rx = (sx - ox) / scaled.width()  * self._pixmap.width()
        ry = (sy - oy) / scaled.height() * self._pixmap.height()
        return int(rx), int(ry)

    # ----------------------------------------------------------
    #  鼠标事件
    # ----------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            x, y = self._to_image_pos(e.position().x(), e.position().y())
            self.sig_right_click.emit(x, y)
            return

        if e.button() == Qt.MouseButton.LeftButton:
            self._is_pressing = True
            x, y = self._to_image_pos(e.position().x(), e.position().y())
            self._last_pos = (x, y)
            # ✅ 单击也触发 press 信号（修复单击不出水）
            self.sig_press.emit(x, y)

    def mouseMoveEvent(self, e):
        if self._is_pressing:
            x, y = self._to_image_pos(e.position().x(), e.position().y())
            self.sig_drag.emit(x, y)
            self._last_pos = (x, y)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._is_pressing:
            self._is_pressing = False
            x, y = self._to_image_pos(e.position().x(), e.position().y())
            self.sig_release.emit(x, y)
            self._last_pos = None

    # ----------------------------------------------------------
    #  裁剪预览框（可选）
    # ----------------------------------------------------------
    class CropOverlay:
        pass


# ── 裁剪遮罩（独立类，供 mixin_tools 使用）───────────────────
class CropOverlay(QOpenGLWidget):
    sig_crop_done = pyqtSignal(int, int, int, int)   # x1,y1,x2,y2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._start = None
        self._end   = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background: transparent;")

    def set_rect(self, x1, y1, x2, y2):
        self._start = QPoint(x1, y1)
        self._end   = QPoint(x2, y2)
        self.update()

    def paintGL(self):
        if not (self._start and self._end):
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))
        pen = QPen(QColor("#cdd6f4"), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(QRect(self._start, self._end))
        painter.end()