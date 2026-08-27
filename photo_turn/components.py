# photo_turn/components.py
# ============================================================
#  PyQt6 GPU 加速画布（QOpenGLWidget）
#  • 滚轮缩放 / 中键平移 / 适配窗口
#  • drawPixmap(目标矩形) 由 GPU 缩放,不再每次 CPU 重采样
# ============================================================

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtCore    import Qt, QPoint, QRect, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui     import QPainter, QPixmap, QColor, QPen, QCursor


class ImageCanvas(QOpenGLWidget):
    """
    OpenGL 加速画布
    • GPU 渲染 + 目标矩形绘制,消除画笔卡顿
    • 滚轮:以光标为锚点缩放
    • 中键拖拽:平移视图
    • reset_view(): 恢复适配窗口
    """

    sig_press       = pyqtSignal(int, int)   # 鼠标按下（图像坐标）
    sig_drag        = pyqtSignal(int, int)   # 拖拽移动
    sig_release     = pyqtSignal(int, int)   # 鼠标释放
    sig_right_click = pyqtSignal(int, int)   # 右键

    MIN_SCALE = 0.05
    MAX_SCALE = 40.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap      = None
        self._is_pressing = False
        self._last_pos    = None

        # 视图状态
        self._fit    = True              # True=适配窗口;False=自由缩放/平移
        self._scale  = 1.0               # 自由模式下的缩放倍率
        self._offset = QPointF(0, 0)     # 自由模式下图像左上角在控件中的位置
        self._pan_anchor = None          # 中键平移起点(屏幕坐标)
        self._show_grid  = False         # 三分构图网格

        self.setMouseTracking(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(400, 300)

    # ----------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap):
        size_changed = (
            self._pixmap is None
            or self._pixmap.size() != pixmap.size()
        )
        self._pixmap = pixmap
        if size_changed:
            self._fit = True   # 换了图(裁剪/旋转)回到适配窗口
        self.update()

    def reset_view(self):
        """恢复适配窗口"""
        self._fit = True
        self.update()

    def toggle_grid(self) -> bool:
        """切换三分构图网格,返回当前状态"""
        self._show_grid = not self._show_grid
        self.update()
        return self._show_grid

    @property
    def is_fit_mode(self) -> bool:
        return self._fit

    # ----------------------------------------------------------
    #  视图几何(唯一出口,渲染与坐标映射共用)
    # ----------------------------------------------------------
    def _target_rect(self):
        """图像在控件坐标系中的目标矩形"""
        if not self._pixmap or self._pixmap.isNull():
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return None

        if self._fit:
            scale = min(self.width() / pw, self.height() / ph)
            tw, th = pw * scale, ph * scale
            return QRectF((self.width() - tw) / 2,
                          (self.height() - th) / 2, tw, th)
        return QRectF(self._offset.x(), self._offset.y(),
                      pw * self._scale, ph * self._scale)

    # ----------------------------------------------------------
    #  GPU 渲染
    # ----------------------------------------------------------
    def paintGL(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#1e1e2e"))

        rect = self._target_rect()
        if rect is not None:
            painter.drawPixmap(rect, self._pixmap,
                               QRectF(self._pixmap.rect()))
            if self._show_grid:
                self._draw_grid(painter, rect)
        painter.end()

    def _draw_grid(self, painter: QPainter, rect: QRectF):
        """三分构图网格 + 边框"""
        pen = QPen(QColor(255, 255, 255, 70), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for i in (1, 2):
            x = rect.x() + rect.width()  * i / 3
            y = rect.y() + rect.height() * i / 3
            painter.drawLine(QPointF(x, rect.y()),
                             QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.x(), y),
                             QPointF(rect.right(), y))
        painter.setPen(QPen(QColor(255, 255, 255, 110), 1))
        painter.drawRect(rect)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update()   # fit 模式下目标矩形随窗口重算

    # ----------------------------------------------------------
    #  坐标映射：屏幕坐标 → 图像坐标
    # ----------------------------------------------------------
    def _to_image_pos(self, sx: float, sy: float):
        rect = self._target_rect()
        if rect is None or rect.width() <= 0 or rect.height() <= 0:
            return int(sx), int(sy)
        rx = (sx - rect.x()) / rect.width()  * self._pixmap.width()
        ry = (sy - rect.y()) / rect.height() * self._pixmap.height()
        return int(rx), int(ry)

    # ----------------------------------------------------------
    #  缩放 / 平移
    # ----------------------------------------------------------
    def wheelEvent(self, e):
        if not self._pixmap or self._pixmap.isNull():
            return
        rect = self._target_rect()
        if rect is None:
            return

        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        cur_scale = rect.width() / self._pixmap.width()
        new_scale = max(self.MIN_SCALE,
                        min(self.MAX_SCALE, cur_scale * factor))
        if abs(new_scale - cur_scale) < 1e-6:
            return

        # 以光标下的图像点为锚点,缩放后保持其屏幕位置不动
        mx, my = e.position().x(), e.position().y()
        ix = (mx - rect.x()) / cur_scale
        iy = (my - rect.y()) / cur_scale

        self._scale  = new_scale
        self._offset = QPointF(mx - ix * new_scale,
                               my - iy * new_scale)
        self._fit = False
        self.update()

    def _start_pan(self, pos):
        self._pan_anchor = pos
        self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def _end_pan(self):
        self._pan_anchor = None
        self.unsetCursor()

    # ----------------------------------------------------------
    #  鼠标事件
    # ----------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            if self._pixmap is not None:
                # 从 fit 切自由模式时同步当前几何,避免视图跳变
                if self._fit:
                    rect = self._target_rect()
                    if rect is not None:
                        self._scale  = rect.width() / self._pixmap.width()
                        self._offset = rect.topLeft()
                    self._fit = False
                self._start_pan(e.position())
            return

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
        if self._pan_anchor is not None:
            delta = e.position() - self._pan_anchor
            self._offset += delta
            self._pan_anchor = e.position()
            self.update()
            return
        if self._is_pressing:
            x, y = self._to_image_pos(e.position().x(), e.position().y())
            self.sig_drag.emit(x, y)
            self._last_pos = (x, y)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton and self._pan_anchor is not None:
            self._end_pan()
            return
        if e.button() == Qt.MouseButton.LeftButton and self._is_pressing:
            self._is_pressing = False
            x, y = self._to_image_pos(e.position().x(), e.position().y())
            self.sig_release.emit(x, y)
            self._last_pos = None

    # ----------------------------------------------------------
    #  双击 = 适配窗口
    # ----------------------------------------------------------
    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self.reset_view()
        super().mouseDoubleClickEvent(e)


# ── 裁剪遮罩（独立类，可供框选预览扩展使用）──────────────────
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
