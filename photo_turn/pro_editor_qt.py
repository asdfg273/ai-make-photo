# photo_turn/pro_editor_qt.py
# ============================================================
#  PyQt6 专业修图窗口 — 全屏 / GPU / 滑块画笔 / 完整滤镜面板
# ============================================================

import io
from PIL import Image

from PyQt6.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QPushButton, QSlider,
    QSpinBox, QComboBox, QSizePolicy, QScrollArea,
)
from PyQt6.QtCore  import Qt, QTimer, QSize
from PyQt6.QtGui   import (
    QPixmap, QKeySequence, QImage,
    QColor, QShortcut,
)

from .components    import ImageCanvas
from .mixin_history import HistoryMixin
from .mixin_filters import FiltersMixin
from .mixin_ai      import AIToolsMixin
from .mixin_tools   import ToolsEventsMixin


# ── PIL → QPixmap ────────────────────────────────────────────
def _pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    """
    直接用原始 RGBA 字节构造 QImage，
    比 PNG 编码快 10x 以上，彻底消除画笔卡顿
    """
    img  = pil_img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(
        data,
        img.width,
        img.height,
        img.width * 4,                      # bytes per line
        QImage.Format.Format_RGBA8888,
    )
    return QPixmap.fromImage(qimg.copy())   


# ── 通用按钮工厂 ─────────────────────────────────────────────
def _make_btn(text: str, tip: str = "", color: str = "#313244") -> QPushButton:
    btn = QPushButton(text)
    btn.setToolTip(tip)
    btn.setStyleSheet(
        f"QPushButton {{ background:{color}; color:#cdd6f4; "
        f"border-radius:6px; border:1px solid #45475a; "
        f"padding:5px 8px; font-size:12px; }}"
        f"QPushButton:hover {{ background:#45475a; }}"
        f"QPushButton:disabled {{ color:#585b70; }}"
    )
    return btn


# ── 滑块工厂 ─────────────────────────────────────────────────
def _make_slider(min_v: int, max_v: int, default: int,
                 parent=None) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal, parent)
    s.setRange(min_v, max_v)
    s.setValue(default)
    s.setStyleSheet(
        "QSlider::groove:horizontal { height:4px; background:#45475a; border-radius:2px; }"
        "QSlider::handle:horizontal  { width:14px; height:14px; background:#cdd6f4; "
        "                              border-radius:7px; margin:-5px 0; }"
        "QSlider::sub-page:horizontal{ background:#89b4fa; border-radius:2px; }"
    )
    return s


# ============================================================
class ProImageEditor(
    QDialog,
    HistoryMixin,
    FiltersMixin,
    AIToolsMixin,
    ToolsEventsMixin,
):
    def __init__(self, parent, image_path: str, callback_on_save=None):
        super().__init__(parent)
        self.setWindowTitle("✨ 专业修图 & AI 遮罩引擎")
        self.setMinimumSize(900, 600) 

        # ✅ 1. 默认最大化全屏
        
        
        self.setModal(True)

        self.callback_on_save = callback_on_save
        self.image_path       = image_path

        # ── 图像初始化 ─────────────────────────────────────
        self.original_full_img = Image.open(image_path).convert("RGB")
        self.original_full_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        self.base_img        = self.original_full_img.copy()
        self.current_img     = self.base_img.copy()
        self.filter_base_img = self.base_img.copy()
        self.original_img    = self.original_full_img.copy()
        self.mask_img        = Image.new("L", self.base_img.size, 0)

        # ── 状态变量 ───────────────────────────────────────
        self.history             = []
        self.future              = []
        self.crop_mode           = False
        self.crop_overlay        = None
        self.text_mode           = False
        self.text_element        = None
        self.current_text_string = ""
        self.text_color          = "#ffffff"
        self.text_size           = 40
        self.draw_mode           = False
        self.is_mask_brush       = False
        self.brush_color         = "#ff0000"
        self.is_eraser           = False
        self.adjust_vars         = {}
        self.adjust_timer: QTimer | None = None
        self._adetailer_running  = False
        self._last_draw_pos      = None

        # ── 构建 UI ────────────────────────────────────────
        self._setup_ui()
        self._bind_shortcuts()
        self.push_history()
        self.update_canvas(self.current_img, force=True)
        self.showMaximized()

    # ----------------------------------------------------------
    #  快捷键
    # ----------------------------------------------------------
    def _bind_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"),  self).activated.connect(self.undo)
        QShortcut(QKeySequence("Ctrl+Y"),  self).activated.connect(self.redo)
        QShortcut(QKeySequence("Ctrl+S"),  self).activated.connect(self.save_and_return)
        QShortcut(QKeySequence("Escape"),  self).activated.connect(self._cancel_any_mode)
        QShortcut(QKeySequence("F11"),     self).activated.connect(self._toggle_fullscreen)

    def _toggle_fullscreen(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # ----------------------------------------------------------
    #  主布局
    # ----------------------------------------------------------
    def _setup_ui(self):
        self.setStyleSheet("background:#1e1e2e; color:#cdd6f4;")

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # 中间：左面板 + 画布
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # 左侧可滚动工具面板
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setStyleSheet(
            "QScrollArea { border:none; background:#181825; }"
            "QScrollBar:vertical { background:#181825; width:6px; }"
            "QScrollBar::handle:vertical { background:#45475a; border-radius:3px; }"
        )
        left_panel = self._build_left_panel()
        scroll.setWidget(left_panel)
        splitter.addWidget(scroll)

        # 画布（GPU）
        self.canvas = ImageCanvas(self)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.sig_press.connect(self.on_mouse_press)
        self.canvas.sig_drag.connect(self.on_mouse_drag)
        self.canvas.sig_release.connect(self.on_mouse_release)
        self.canvas.sig_right_click.connect(self.on_mouse_right_click)
        splitter.addWidget(self.canvas)
        splitter.setSizes([300, 1200])

        root.addWidget(splitter)

        # 底部状态栏
        root.addLayout(self._build_status_bar())

    # ----------------------------------------------------------
    #  底部状态栏
    # ----------------------------------------------------------
    def _build_status_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 0, 4, 0)

        self.lbl_status = QLabel("✅ 就绪")
        self.lbl_status.setStyleSheet("color:#585b70; font-size:13px; padding:4px;")
        bar.addWidget(self.lbl_status)
        bar.addStretch()

        # 全屏按钮
        btn_fs = _make_btn("⛶ 全屏 (F11)", color="#313244")
        btn_fs.clicked.connect(self._toggle_fullscreen)
        bar.addWidget(btn_fs)

        btn_save = QPushButton("💾 保存并返回 (Ctrl+S)")
        btn_save.setStyleSheet(
            "QPushButton { background:#a6e3a1; color:#1e1e2e; "
            "border-radius:6px; font-weight:bold; padding:6px 18px; font-size:13px; }"
            "QPushButton:hover { background:#94e2d5; }"
        )
        btn_save.clicked.connect(self.save_and_return)
        bar.addWidget(btn_save)

        btn_close = _make_btn("✖ 取消")
        btn_close.clicked.connect(self.reject)
        bar.addWidget(btn_close)

        return bar

    # ----------------------------------------------------------
    #  左侧工具面板
    # ----------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background:#181825;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        # ── 绘图工具 ───────────────────────────────────────
        layout.addWidget(self._sec("🛠 绘图工具"))

        row1 = QHBoxLayout()
        self.btn_brush  = _make_btn("🖌 画笔")
        self.btn_eraser = _make_btn("🧽 橡皮")
        self.btn_mask   = _make_btn("🔴 遮罩")
        for b in [self.btn_brush, self.btn_eraser, self.btn_mask]:
            row1.addWidget(b)
        self.btn_brush.clicked.connect(self.toggle_brush)
        self.btn_eraser.clicked.connect(self.toggle_eraser)
        self.btn_mask.clicked.connect(self.toggle_mask_brush)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_crop = _make_btn("✂ 裁剪")
        self.btn_text = _make_btn("📝 文字")
        self.btn_crop.clicked.connect(self.toggle_crop)
        self.btn_text.clicked.connect(self.toggle_text)
        row2.addWidget(self.btn_crop)
        row2.addWidget(self.btn_text)
        layout.addLayout(row2)

        # ── ✅ 画笔大小 — 改为滑块 ────────────────────────
        layout.addWidget(self._sec("🔘 画笔大小"))
        brush_row = QHBoxLayout()
        self.sld_brush_size = _make_slider(1, 150, 20)
        self.lbl_brush_val  = QLabel("20 px")
        self.lbl_brush_val.setFixedWidth(45)
        self.lbl_brush_val.setStyleSheet("color:#cdd6f4; font-size:12px;")
        self.sld_brush_size.valueChanged.connect(
            lambda v: self.lbl_brush_val.setText(f"{v} px")
        )
        brush_row.addWidget(self.sld_brush_size)
        brush_row.addWidget(self.lbl_brush_val)
        layout.addLayout(brush_row)

        # ── 文字大小 ───────────────────────────────────────
        layout.addWidget(self._sec("🔤 文字大小"))
        text_row = QHBoxLayout()
        self.spin_text_size = QSpinBox()
        self.spin_text_size.setRange(8, 200)
        self.spin_text_size.setValue(40)
        self.spin_text_size.setStyleSheet(self._input_style())
        text_row.addWidget(self.spin_text_size)
        layout.addLayout(text_row)

        # ── 颜色 ───────────────────────────────────────────
        layout.addWidget(self._sec("🎨 颜色"))
        cr = QHBoxLayout()
        btn_pc = _make_btn("画笔颜色")
        btn_pc.clicked.connect(self.pick_color)
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(26, 26)
        self.color_preview.setStyleSheet(
            f"background:{self.brush_color}; border-radius:4px;")
        cr.addWidget(btn_pc)
        cr.addWidget(self.color_preview)
        layout.addLayout(cr)

        tr = QHBoxLayout()
        btn_tc = _make_btn("文字颜色")
        btn_tc.clicked.connect(self.pick_text_color)
        self.text_color_preview = QLabel()
        self.text_color_preview.setFixedSize(26, 26)
        self.text_color_preview.setStyleSheet(
            f"background:{self.text_color}; border-radius:4px;")
        tr.addWidget(btn_tc)
        tr.addWidget(self.text_color_preview)
        layout.addLayout(tr)

        # ── 变换 ───────────────────────────────────────────
        layout.addWidget(self._sec("🔄 变换"))
        t1 = QHBoxLayout()
        bfh = _make_btn("↔ 水平")
        bfv = _make_btn("↕ 垂直")
        bfh.clicked.connect(lambda: self.flip_image("horizontal"))
        bfv.clicked.connect(lambda: self.flip_image("vertical"))
        t1.addWidget(bfh); t1.addWidget(bfv)
        layout.addLayout(t1)

        t2 = QHBoxLayout()
        brl = _make_btn("↺ 左 90°")
        brr = _make_btn("↻ 右 90°")
        brl.clicked.connect(lambda: self.rotate_image(90))
        brr.clicked.connect(lambda: self.rotate_image(-90))
        t2.addWidget(brl); t2.addWidget(brr)
        layout.addLayout(t2)

        # ── 历史 ───────────────────────────────────────────
        layout.addWidget(self._sec("⏪ 历史 (Ctrl+Z/Y)"))
        h1 = QHBoxLayout()
        bu = _make_btn("↩ 撤销")
        br = _make_btn("↪ 重做")
        bu.clicked.connect(self.undo)
        br.clicked.connect(self.redo)
        h1.addWidget(bu); h1.addWidget(br)
        layout.addLayout(h1)

        # ── ✅ 调色滑块（完整 5 项）─────────────────────────
        layout.addWidget(self._sec("🎛 色彩调整"))
        adj_cfg = [
            ("brightness", "☀ 亮度",   -100, 100, 0),
            ("contrast",   "◑ 对比度", -100, 100, 0),
            ("saturation", "🌈 饱和度", -100, 100, 0),
            ("sharpness",  "🔪 锐度",   -100, 100, 0),
            ("temperature","🌡 色温",   -100, 100, 0),
        ]
        for key, label, mn, mx, dv in adj_cfg:
            layout.addWidget(self._sec_small(label))
            row = QHBoxLayout()
            sld = _make_slider(mn, mx, dv)
            lbl = QLabel(f"{dv}")
            lbl.setFixedWidth(35)
            lbl.setStyleSheet("color:#cdd6f4; font-size:11px;")
            sld.valueChanged.connect(
                lambda v, l=lbl, k=key: (
                    l.setText(str(v)),
                    self.on_adjust_change(k, v),
                )
            )
            row.addWidget(sld)
            row.addWidget(lbl)
            layout.addLayout(row)
            self.adjust_vars[key] = sld

        btn_reset_adj = _make_btn("🔄 重置调色")
        btn_reset_adj.clicked.connect(self.reset_adjustments)
        layout.addWidget(btn_reset_adj)

        # ── ✅ 滤镜面板（完整 12 种）──────────────────────
        layout.addWidget(self._sec("✨ 预设滤镜"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "无", "黑白", "复古", "冷色调", "暖色调",
            "胶片颗粒", "模糊", "浮雕", "边缘检测",
            "轮廓", "锐化", "油画",
        ])
        self.filter_combo.setStyleSheet(
            "QComboBox { background:#313244; color:#cdd6f4; "
            "border:1px solid #45475a; border-radius:4px; "
            "padding:4px; font-size:12px; }"
            "QComboBox::drop-down { border:none; }"
            "QComboBox QAbstractItemView { background:#313244; color:#cdd6f4; "
            "selection-background-color:#45475a; }"
        )
        layout.addWidget(self.filter_combo)

        # 模糊半径（模糊滤镜专用）
        layout.addWidget(self._sec_small("🌫 模糊半径"))
        blur_row = QHBoxLayout()
        self.blur_scale = _make_slider(0, 20, 2)
        self.lbl_blur_val = QLabel("2")
        self.lbl_blur_val.setFixedWidth(25)
        self.lbl_blur_val.setStyleSheet("color:#cdd6f4; font-size:11px;")
        self.blur_scale.valueChanged.connect(
            lambda v: self.lbl_blur_val.setText(str(v))
        )
        blur_row.addWidget(self.blur_scale)
        blur_row.addWidget(self.lbl_blur_val)
        layout.addLayout(blur_row)

        btn_apply_filter = _make_btn("▶ 应用滤镜", color="#45475a")
        btn_apply_filter.clicked.connect(self.apply_selected_filter)
        layout.addWidget(btn_apply_filter)

        # ── AI 工具 ────────────────────────────────────────
        layout.addWidget(self._sec("🤖 AI 工具"))
        btn_ad = _make_btn("✨ ADetailer 人脸修复", color="#45475a")
        btn_ad.clicked.connect(self.run_adetailer)
        layout.addWidget(btn_ad)

        layout.addStretch()
        return panel

    # ----------------------------------------------------------
    #  样式工具
    # ----------------------------------------------------------
    def _sec(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color:#7f849c; font-size:11px; font-weight:bold; "
            "padding:6px 0 2px 0; border-top:1px solid #313244;"
        )
        return lbl

    def _sec_small(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#9399b2; font-size:11px; padding:2px 0;")
        return lbl

    def _input_style(self) -> str:
        return (
            "QSpinBox { background:#313244; color:#cdd6f4; "
            "border:1px solid #45475a; border-radius:4px; "
            "padding:3px; font-size:12px; }"
        )

    # ----------------------------------------------------------
    #  画布更新
    # ----------------------------------------------------------
    def update_canvas(self, pil_img: Image.Image, force: bool = False):
        self.canvas.set_pixmap(_pil_to_qpixmap(pil_img))

    # ----------------------------------------------------------
    #  保存
    # ----------------------------------------------------------
    def save_and_return(self):
        if self.text_element:
            self._commit_text_to_image()

        print(f"[DEBUG] mask extrema = {self.mask_img.getextrema()}, "
              f"size = {self.mask_img.size}")
        self.mask_img.save("debug_mask.png")

        print(f"[DEBUG] callback_on_save = {self.callback_on_save}")

        if self.callback_on_save:
            try:
                print("[DEBUG] ▶ 即将调用 callback_on_save ...")
                self.callback_on_save(self.current_img, self.mask_img)
                print("[DEBUG] ✅ callback_on_save 正常返回")
            except Exception as e:
                import traceback
                print(f"[DEBUG] ❌ callback_on_save 抛异常: {e}")
                print(traceback.format_exc())
        else:
            print("[DEBUG] ⚠️ callback_on_save 为 None,不会触发后续流程")

        self.accept()