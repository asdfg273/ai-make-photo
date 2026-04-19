# main_window.py
"""
主窗口：集成菜单栏、工具栏、调整面板、图像显示区、状态栏。
核心流程：
1. 打开图片 -> 存入 self.base_image（原始调整基准）和 self.current_image。
2. 调整滑块 -> QTimer 防抖 150ms -> 启动 AdjustmentWorker。
3. 滤镜按钮 -> 应用到 current_image 并压入历史栈。
4. 撤销/重做 -> 双栈切换 current_image。
5. 裁剪/旋转 -> 修改 current_image 并作为新的 base_image（因为改变了像素尺寸）。
"""
import os
from PIL import Image

from PyQt6.QtCore import Qt, QTimer, QSize, QSettings, QEvent
from PyQt6.QtGui import (QAction, QPixmap, QIcon, QKeySequence,
                         QShortcut, QImageReader)
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QLabel, QScrollArea, QDockWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QPushButton,
    QToolBar, QStatusBar, QProgressBar, QComboBox, QGroupBox,
    QGridLayout, QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
    QSizePolicy, QApplication
)

from utils import pil_to_qpixmap, format_file_size
from worker_thread import AdjustmentWorker, FilterWorker, run_in_thread
from image_processor import ImageProcessor
from crop_overlay import CropLabel


MAX_HISTORY = 20


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("本地修图软件 - Photo Editor")
        self.resize(1280, 820)

        # ---------- 状态 ----------
        self.base_image: Image.Image | None = None       # 调整前基础图
        self.current_image: Image.Image | None = None    # 当前显示
        self.current_pixmap: QPixmap | None = None
        self.file_path: str = ""
        self.file_size: int = 0

        self.scale_factor: float = 1.0
        self._space_pressed = False
        self._dragging = False
        self._drag_start = None
        self._drag_scroll_start = None

        # 历史栈（保存 PIL Image）
        self.history_stack: list[Image.Image] = []
        self.future_stack: list[Image.Image] = []

        # 调整参数
        self._init_params()

        # 防抖计时器
        self.adjust_timer = QTimer(self)
        self.adjust_timer.setSingleShot(True)
        self.adjust_timer.setInterval(150)
        self.adjust_timer.timeout.connect(self._run_adjustments)

        # 工作线程引用
        self._adjust_thread = None
        self._adjust_worker = None
        self._filter_thread = None
        self._filter_worker = None
        self._pending_adjust = False

        # QSettings
        self.settings = QSettings()

        # ---------- UI ----------
        self._create_central()
        self._create_actions()
        self.act_quick_save = QAction("✅ 保存并返回主界面", self)
        self.act_quick_save.triggered.connect(self.quick_save_and_exit)
        self._create_menubar()
        self._create_toolbar()
        self._create_dock()
        self._create_statusbar()
        self._create_shortcuts()

        self._restore_state()
        self._update_ui_state()
        if input_path and os.path.exists(input_path):
            self._load_from_path(input_path)

    # ==================== 参数初始化 ====================
    def _init_params(self):
        self.params = {
            "brightness": 0, "contrast": 0, "saturation": 0,
            "sharpness": 0, "temperature": 0,
        }

    # ==================== UI 构建 ====================
    def _create_central(self):
        self.image_label = CropLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e;")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(False)
        self.scroll.viewport().installEventFilter(self)

        self.setCentralWidget(self.scroll)

    def _create_actions(self):
        self.act_open = QAction("打开图片", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_image)

        self.act_save_as = QAction("另存为...", self)
        self.act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_as.triggered.connect(self.save_as)

        self.act_exit = QAction("退出", self)
        self.act_exit.triggered.connect(self.close)

        self.act_undo = QAction("撤销", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self.undo)

        self.act_redo = QAction("重做", self)
        self.act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        self.act_redo.triggered.connect(self.redo)

        self.act_rotate_left = QAction("左转90°", self)
        self.act_rotate_left.triggered.connect(lambda: self._apply_transform("rotate_left"))

        self.act_rotate_right = QAction("右转90°", self)
        self.act_rotate_right.triggered.connect(lambda: self._apply_transform("rotate_right"))

        self.act_flip_h = QAction("水平镜像", self)
        self.act_flip_h.triggered.connect(lambda: self._apply_transform("flip_horizontal"))

        self.act_flip_v = QAction("垂直镜像", self)
        self.act_flip_v.triggered.connect(lambda: self._apply_transform("flip_vertical"))

        self.act_crop_mode = QAction("裁剪", self)
        self.act_crop_mode.setCheckable(True)
        self.act_crop_mode.triggered.connect(self._toggle_crop_mode)

        self.act_apply_crop = QAction("应用裁剪", self)
        self.act_apply_crop.triggered.connect(self._apply_crop)
        self.act_apply_crop.setEnabled(False)

        self.act_zoom_in = QAction("放大", self)
        self.act_zoom_in.setShortcut(QKeySequence("Ctrl++"))
        self.act_zoom_in.triggered.connect(lambda: self._zoom(1.25))

        self.act_zoom_out = QAction("缩小", self)
        self.act_zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        self.act_zoom_out.triggered.connect(lambda: self._zoom(0.8))

        self.act_fit = QAction("适合窗口", self)
        self.act_fit.triggered.connect(self._fit_to_window)

        self.act_actual = QAction("实际大小", self)
        self.act_actual.setShortcut(QKeySequence("Ctrl+0"))
        self.act_actual.triggered.connect(lambda: self._set_scale(1.0))

    def _create_menubar(self):
        mb = self.menuBar()
        mf = mb.addMenu("文件(&F)")
        mf.addAction(self.act_open)
        mf.addAction(self.act_save_as)
        mf.addSeparator()
        mf.addAction(self.act_exit)

        me = mb.addMenu("编辑(&E)")
        me.addAction(self.act_undo)
        me.addAction(self.act_redo)
        me.addSeparator()
        me.addAction(self.act_rotate_left)
        me.addAction(self.act_rotate_right)
        me.addAction(self.act_flip_h)
        me.addAction(self.act_flip_v)
        me.addSeparator()
        me.addAction(self.act_crop_mode)
        me.addAction(self.act_apply_crop)

        mv = mb.addMenu("视图(&V)")
        mv.addAction(self.act_zoom_in)
        mv.addAction(self.act_zoom_out)
        mv.addAction(self.act_fit)
        mv.addAction(self.act_actual)

    def _create_toolbar(self):
        tb = QToolBar("主工具栏")
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)
        tb.addAction(self.act_quick_save)
        tb.addSeparator()
        tb.addAction(self.act_open)
        tb.addAction(self.act_save_as)
        tb.addSeparator()
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)
        tb.addSeparator()
        tb.addAction(self.act_rotate_left)
        tb.addAction(self.act_rotate_right)
        tb.addAction(self.act_flip_h)
        tb.addAction(self.act_flip_v)
        tb.addSeparator()
        tb.addAction(self.act_crop_mode)
        tb.addAction(self.act_apply_crop)
        tb.addSeparator()
        tb.addAction(self.act_zoom_in)
        tb.addAction(self.act_zoom_out)
        tb.addAction(self.act_fit)

    def _create_dock(self):
        # -------- 调整面板 --------
        dock = QDockWidget("调整", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        # 调整滑块组
        adjust_box = QGroupBox("基础调整")
        grid = QGridLayout(adjust_box)

        self.sliders = {}
        specs = [
            ("brightness", "亮度",   -100, 100, 0),
            ("contrast",   "对比度", -100, 100, 0),
            ("saturation", "饱和度", -100, 100, 0),
            ("sharpness",  "锐化",      0, 100, 0),
            ("temperature","色温",   -100, 100, 0),
        ]
        for row, (key, label, mn, mx, default) in enumerate(specs):
            lbl = QLabel(label)
            lbl.setMinimumWidth(50)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(mn, mx)
            sl.setValue(default)
            sb = QSpinBox()
            sb.setRange(mn, mx)
            sb.setValue(default)
            sb.setFixedWidth(64)
            sl.valueChanged.connect(sb.setValue)
            sb.valueChanged.connect(sl.setValue)
            sl.valueChanged.connect(lambda v, k=key: self._on_slider_change(k, v))
            grid.addWidget(lbl, row, 0)
            grid.addWidget(sl, row, 1)
            grid.addWidget(sb, row, 2)
            self.sliders[key] = (sl, sb)

        btn_reset = QPushButton("重置调整")
        btn_reset.clicked.connect(self._reset_adjustments)
        grid.addWidget(btn_reset, len(specs), 0, 1, 3)

        lay.addWidget(adjust_box)

        # -------- 滤镜 --------
        filter_box = QGroupBox("预设滤镜")
        fl = QVBoxLayout(filter_box)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "（选择滤镜）", "黑白", "复古", "冷色调",
            "暖色调", "胶片颗粒", "高斯模糊"
        ])
        fl.addWidget(self.filter_combo)

        blur_row = QHBoxLayout()
        blur_row.addWidget(QLabel("模糊半径"))
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(0, 100)     # 映射为 0.0 ~ 10.0
        self.blur_slider.setValue(20)
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(0, 100)
        self.blur_spin.setValue(20)
        self.blur_slider.valueChanged.connect(self.blur_spin.setValue)
        self.blur_spin.valueChanged.connect(self.blur_slider.setValue)
        blur_row.addWidget(self.blur_slider)
        blur_row.addWidget(self.blur_spin)
        fl.addLayout(blur_row)

        btn_apply_filter = QPushButton("应用滤镜")
        btn_apply_filter.clicked.connect(self._apply_filter)
        fl.addWidget(btn_apply_filter)

        lay.addWidget(filter_box)

        # -------- 裁剪比例 --------
        crop_box = QGroupBox("裁剪比例")
        cl = QHBoxLayout(crop_box)
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(["自由", "1:1", "4:3", "16:9"])
        self.aspect_combo.currentTextChanged.connect(self._on_aspect_change)
        cl.addWidget(self.aspect_combo)
        lay.addWidget(crop_box)

        lay.addStretch(1)

        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.adjust_dock = dock

    def _create_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_info = QLabel("未打开图片")
        sb.addWidget(self.lbl_info, 1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setRange(0, 0)   # 不确定进度
        self.progress.setVisible(False)
        sb.addPermanentWidget(self.progress)

    def _create_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)

    # ==================== 图片读取 / 保存 ====================
    def open_image(self):
        last_dir = self.settings.value("last_open_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", last_dir,
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*.*)"
        )
        if not path:
            return
        try:
            # 使用 PIL 打开，转成 RGBA 以支持透明度
            img = Image.open(path)
            img.load()
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.mode else "RGB")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开图片：{e}")
            return

        self.settings.setValue("last_open_dir", os.path.dirname(path))
        self.file_path = path
        self.file_size = os.path.getsize(path)

        # 重置状态
        self.base_image = img.copy()
        self.current_image = img.copy()
        self.history_stack.clear()
        self.future_stack.clear()
        self.history_stack.append(self.current_image.copy())
        self._reset_adjustments(silent=True)

        self._display_image(self.current_image)
        self._fit_to_window()
        self._update_ui_state()
        self._update_statusbar()

    def save_as(self):
        if self.current_image is None:
            return

        # 若还有待处理的调整，先同步处理
        if self.adjust_timer.isActive():
            self.adjust_timer.stop()
            self._run_adjustments(sync=True)

        last_dir = self.settings.value("last_save_dir", "")
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "另存为", last_dir,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if not path:
            return
        self.settings.setValue("last_save_dir", os.path.dirname(path))

        ext = os.path.splitext(path)[1].lower()
        fmt = None
        quality = 90

        if ext in (".jpg", ".jpeg") or "JPEG" in selected_filter:
            fmt = "JPEG"
            if not ext:
                path += ".jpg"
            # 询问质量
            q = self._ask_jpeg_quality()
            if q is None:
                return
            quality = q
        elif ext == ".png" or "PNG" in selected_filter:
            fmt = "PNG"
            if not ext:
                path += ".png"
        elif ext == ".bmp" or "BMP" in selected_filter:
            fmt = "BMP"
            if not ext:
                path += ".bmp"
        else:
            fmt = "PNG"
            path += ".png"

        try:
            img = self.current_image
            if fmt == "JPEG" and img.mode == "RGBA":
                img = img.convert("RGB")
            if fmt == "JPEG":
                img.save(path, fmt, quality=quality)
            else:
                img.save(path, fmt)
            self.statusBar().showMessage(f"已保存到 {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{e}")

    def _ask_jpeg_quality(self) -> int | None:
        dlg = QDialog(self)
        dlg.setWindowTitle("JPEG 导出设置")
        form = QFormLayout(dlg)
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(0, 100)
        sl.setValue(90)
        sb = QSpinBox()
        sb.setRange(0, 100)
        sb.setValue(90)
        sl.valueChanged.connect(sb.setValue)
        sb.valueChanged.connect(sl.setValue)
        row = QWidget()
        rowL = QHBoxLayout(row)
        rowL.setContentsMargins(0, 0, 0, 0)
        rowL.addWidget(sl)
        rowL.addWidget(sb)
        form.addRow("质量：", row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return sb.value()
        return None

    # ==================== 显示 ====================
    def _display_image(self, pil_img: Image.Image):
        """把 PIL Image 显示到 label。"""
        self.current_image = pil_img
        # 旧 pixmap 会被 Python GC 自动回收
        self.current_pixmap = pil_to_qpixmap(pil_img)
        self._apply_scale_to_label()

    def _apply_scale_to_label(self):
        if self.current_pixmap is None:
            return
        w = int(self.current_pixmap.width() * self.scale_factor)
        h = int(self.current_pixmap.height() * self.scale_factor)
        scaled = self.current_pixmap.scaled(
            w, h, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def _set_scale(self, factor: float):
        self.scale_factor = max(0.05, min(8.0, factor))
        self._apply_scale_to_label()

    def _zoom(self, step: float):
        self._set_scale(self.scale_factor * step)

    def _fit_to_window(self):
        if self.current_pixmap is None:
            return
        vp = self.scroll.viewport().size()
        if self.current_pixmap.width() == 0 or self.current_pixmap.height() == 0:
            return
        sx = vp.width() / self.current_pixmap.width()
        sy = vp.height() / self.current_pixmap.height()
        self._set_scale(min(sx, sy) * 0.98)

    # ==================== 调整 ====================
    def _on_slider_change(self, key: str, value: int):
        self.params[key] = value
        if self.base_image is None:
            return
        # 防抖：150ms 内的连续变化合并为一次处理
        self.adjust_timer.start()

    def _reset_adjustments(self, silent=False):
        self._init_params()
        for key, (sl, sb) in self.sliders.items():
            sl.blockSignals(True)
            sb.blockSignals(True)
            sl.setValue(0)
            sb.setValue(0)
            sl.blockSignals(False)
            sb.blockSignals(False)
        if not silent and self.base_image is not None:
            self._display_image(self.base_image.copy())

    def _run_adjustments(self, sync=False):
        """启动后台线程做亮度/对比度等调整。"""
        if self.base_image is None:
            return

        # 若已有任务在运行，标记 pending 待结束再跑一次
        if self._adjust_thread is not None and self._adjust_thread.isRunning():
            self._pending_adjust = True
            return

        params = dict(self.params)

        if sync:
            # 同步模式（保存前确保应用最新参数）
            result = ImageProcessor.process_adjustments(self.base_image, params)
            self._on_adjust_finished(result, from_sync=True)
            return

        self.progress.setVisible(True)
        self.statusBar().showMessage("正在处理…")

        worker = AdjustmentWorker(self.base_image.copy(), params)
        self._adjust_thread, self._adjust_worker = run_in_thread(
            worker, self._on_adjust_finished, self._on_worker_error
        )

    def _on_adjust_finished(self, pil_img: Image.Image, from_sync=False):
        # 调整结果不进入历史栈（滑块调整视为“活动编辑”，仅裁剪/旋转/滤镜入栈）
        self.current_image = pil_img
        self.current_pixmap = pil_to_qpixmap(pil_img)
        self._apply_scale_to_label()

        self.progress.setVisible(False)
        self.statusBar().clearMessage()

        if not from_sync:
            self._adjust_thread = None
            self._adjust_worker = None
            if self._pending_adjust:
                self._pending_adjust = False
                self._run_adjustments()

    def _on_worker_error(self, msg: str):
        self.progress.setVisible(False)
        self.statusBar().clearMessage()
        QMessageBox.warning(self, "处理失败", msg)

    # ==================== 滤镜 ====================
    def _apply_filter(self):
        if self.current_image is None:
            return
        name = self.filter_combo.currentText()
        if name == "（选择滤镜）":
            return

        # 等待当前调整任务
        if self.adjust_timer.isActive():
            self.adjust_timer.stop()
            self._run_adjustments(sync=True)

        radius = self.blur_slider.value() / 10.0

        self.progress.setVisible(True)
        self.statusBar().showMessage(f"正在应用滤镜：{name}…")

        # 基于当前显示图像应用滤镜
        worker = FilterWorker(self.current_image.copy(), name, radius)
        self._filter_thread, self._filter_worker = run_in_thread(
            worker, self._on_filter_finished, self._on_worker_error
        )

    def _on_filter_finished(self, pil_img: Image.Image):
        # 滤镜作为新的基础图（相当于"确认"当前调整结果 + 滤镜）
        self._push_history(pil_img)
        self.base_image = pil_img.copy()
        self.current_image = pil_img
        self._reset_adjustments(silent=True)
        self._display_image(pil_img)

        self.progress.setVisible(False)
        self.statusBar().clearMessage()
        self._filter_thread = None
        self._filter_worker = None
        self._update_ui_state()

    # ==================== 变换（旋转、镜像）====================
    def _apply_transform(self, name: str):
        if self.current_image is None:
            return
        # 先同步调整
        if self.adjust_timer.isActive():
            self.adjust_timer.stop()
            self._run_adjustments(sync=True)

        fn = getattr(ImageProcessor, name)
        new_img = fn(self.current_image)
        self._push_history(new_img)
        self.base_image = new_img.copy()
        self.current_image = new_img
        self._reset_adjustments(silent=True)
        self._display_image(new_img)
        self._update_ui_state()

    # ==================== 裁剪 ====================
    def _toggle_crop_mode(self, checked: bool):
        self.image_label.set_crop_mode(checked)
        self.act_apply_crop.setEnabled(checked)

    def _on_aspect_change(self, text: str):
        mapping = {"自由": None, "1:1": (1, 1), "4:3": (4, 3), "16:9": (16, 9)}
        self.image_label.set_aspect_ratio(mapping.get(text))

    def _apply_crop(self):
        if self.current_image is None:
            return
        rect = self.image_label.get_selection_rect()
        if not rect.isValid() or rect.width() < 5 or rect.height() < 5:
            QMessageBox.information(self, "提示", "请先在图上框选裁剪区域。")
            return

        # 先同步当前调整
        if self.adjust_timer.isActive():
            self.adjust_timer.stop()
            self._run_adjustments(sync=True)

        # label 上的选框是相对于 scaled pixmap 的。
        # label 的 pixmap 居中显示，需要计算相对于 pixmap 左上角的偏移。
        label = self.image_label
        pm = label.pixmap()
        if pm is None or pm.isNull():
            return

        # pixmap 在 label 内居中
        lx = (label.width() - pm.width()) // 2
        ly = (label.height() - pm.height()) // 2

        x1 = rect.x() - lx
        y1 = rect.y() - ly
        x2 = x1 + rect.width()
        y2 = y1 + rect.height()

        # 根据缩放因子映射到原始图像坐标
        sf = self.scale_factor if self.scale_factor > 0 else 1.0
        box = (x1 / sf, y1 / sf, x2 / sf, y2 / sf)

        cropped = ImageProcessor.crop(self.current_image, box)
        self._push_history(cropped)
        self.base_image = cropped.copy()
        self.current_image = cropped
        self._reset_adjustments(silent=True)
        self._display_image(cropped)

        # 退出裁剪模式
        self.act_crop_mode.setChecked(False)
        self._toggle_crop_mode(False)
        self._fit_to_window()
        self._update_ui_state()

    # ==================== 历史记录 ====================
    def _push_history(self, pil_img: Image.Image):
        self.history_stack.append(pil_img.copy())
        if len(self.history_stack) > MAX_HISTORY:
            self.history_stack.pop(0)
        self.future_stack.clear()
        self._update_ui_state()

    def undo(self):
        if len(self.history_stack) < 2:
            return
        current = self.history_stack.pop()
        self.future_stack.append(current)
        prev = self.history_stack[-1]
        self.base_image = prev.copy()
        self.current_image = prev.copy()
        self._reset_adjustments(silent=True)
        self._display_image(self.current_image)
        self._update_ui_state()

    def redo(self):
        if not self.future_stack:
            return
        img = self.future_stack.pop()
        self.history_stack.append(img.copy())
        self.base_image = img.copy()
        self.current_image = img.copy()
        self._reset_adjustments(silent=True)
        self._display_image(self.current_image)
        self._update_ui_state()

    # ==================== UI 状态 ====================
    def _update_ui_state(self):
        has = self.current_image is not None
        self.act_save_as.setEnabled(has)
        self.act_rotate_left.setEnabled(has)
        self.act_rotate_right.setEnabled(has)
        self.act_flip_h.setEnabled(has)
        self.act_flip_v.setEnabled(has)
        self.act_crop_mode.setEnabled(has)
        self.act_zoom_in.setEnabled(has)
        self.act_zoom_out.setEnabled(has)
        self.act_fit.setEnabled(has)
        self.act_actual.setEnabled(has)

        self.act_undo.setEnabled(len(self.history_stack) >= 2)
        self.act_redo.setEnabled(len(self.future_stack) > 0)

    def _update_statusbar(self):
        if self.current_image is None:
            self.lbl_info.setText("未打开图片")
            return
        w, h = self.current_image.size
        size_str = format_file_size(self.file_size) if self.file_size else "-"
        self.lbl_info.setText(
            f"{self.file_path}    |    {w} × {h} px    |    {size_str}"
        )

    # ==================== 事件 ====================
    def eventFilter(self, obj, event):
        # 鼠标滚轮缩放
        if obj is self.scroll.viewport():
            if event.type() == QEvent.Type.Wheel:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier or True:
                    delta = event.angleDelta().y()
                    if delta > 0:
                        self._zoom(1.1)
                    else:
                        self._zoom(0.9)
                    return True
            # 空格拖拽
            if event.type() == QEvent.Type.MouseButtonPress and self._space_pressed:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._dragging = True
                    self._drag_start = event.position().toPoint()
                    self._drag_scroll_start = (
                        self.scroll.horizontalScrollBar().value(),
                        self.scroll.verticalScrollBar().value(),
                    )
                    self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
            if event.type() == QEvent.Type.MouseMove and self._dragging:
                cur = event.position().toPoint()
                dx = cur.x() - self._drag_start.x()
                dy = cur.y() - self._drag_start.y()
                self.scroll.horizontalScrollBar().setValue(self._drag_scroll_start[0] - dx)
                self.scroll.verticalScrollBar().setValue(self._drag_scroll_start[1] - dy)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and self._dragging:
                self._dragging = False
                self.scroll.viewport().setCursor(
                    Qt.CursorShape.OpenHandCursor if self._space_pressed else Qt.CursorShape.ArrowCursor
                )
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self.scroll.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        super().keyReleaseEvent(event)

    def _load_from_path(self, path):
        img = Image.open(path)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.mode else "RGB")
        self.file_path = path
        self.base_image = img.copy()
        self.current_image = img.copy()
        self.history_stack.append(self.current_image.copy())
        self._display_image(self.current_image)
        self._fit_to_window()
        self._update_ui_state()

    def quick_save_and_exit(self):
        if self.current_image is None: return
        if self.adjust_timer.isActive():
            self.adjust_timer.stop()
            self._run_adjustments(sync=True)
        # 保存到主界面指定的路径
        save_path = self.target_output_path if self.target_output_path else "edited_temp.png"
        self.current_image.save(save_path)
        self.close() # 关闭自己，主程序会自动恢复

    # ==================== 状态持久化 ====================
    def _restore_state(self):
        geom = self.settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        super().closeEvent(event)