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
import logging

logger = logging.getLogger(__name__)


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
        self.pick_mode           = False
        self.brush_color         = "#ff0000"
        self._mask_preview_on    = True
        self._filter_preview     = None
        self._stroke_end         = None
        self._dirty              = False
        self.is_eraser           = False
        self.adjust_vars         = {}
        self.adjust_timer: QTimer | None = None
        self._adetailer_running  = False
        self._last_draw_pos      = None
        self._filter_anchor      = None   # 滤镜叠加前的基图(选「无」时还原)

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
        QShortcut(QKeySequence("Ctrl+0"),  self).activated.connect(self._reset_view)
        QShortcut(QKeySequence("["),       self).activated.connect(
            lambda: self._nudge_brush(-2))
        QShortcut(QKeySequence("]"),       self).activated.connect(
            lambda: self._nudge_brush(+2))

    def _reset_view(self):
        if hasattr(self, "canvas"):
            self.canvas.reset_view()
            self.lbl_status.setText("⛶ 视图已适配窗口")
            self.lbl_status.setStyleSheet("color:#89dceb; font-size:13px;")

    def _toggle_grid(self):
        on = self.canvas.toggle_grid()
        self.lbl_status.setText("▦ 构图网格已开启" if on else "▦ 构图网格已关闭")
        self.lbl_status.setStyleSheet("color:#89dceb; font-size:13px;")

    def _compare_on(self):
        """按住:显示原图"""
        self.update_canvas(self.original_img, force=True)

    def _compare_off(self):
        """松开:回到当前编辑结果(遮罩模式下恢复红色叠加预览)"""
        if self.is_mask_brush and self.mask_img.getextrema() != (0, 0):
            self._overlay_mask()
        else:
            self.update_canvas(self.current_img, force=True)

    def _nudge_brush(self, delta: int):
        sld = self.sld_brush_size
        sld.setValue(max(sld.minimum(), min(sld.maximum(),
                                            sld.value() + delta)))

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
        self.canvas.sig_hover.connect(self._on_canvas_hover)
        self.canvas.sig_view_changed.connect(self._refresh_view_info)
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

        # 光标坐标 / 图像尺寸信息
        self.lbl_cursor = QLabel("")
        self.lbl_cursor.setStyleSheet("color:#7f849c; font-size:11px; padding:4px;")
        self.lbl_cursor.setMinimumWidth(150)
        bar.addWidget(self.lbl_cursor)
        self.lbl_img_info = QLabel("")
        self.lbl_img_info.setStyleSheet("color:#7f849c; font-size:11px; padding:4px;")
        bar.addWidget(self.lbl_img_info)
        bar.addStretch()

        # 按住对比原图
        btn_compare = _make_btn("👁 按住对比原图", color="#313244")
        btn_compare.pressed.connect(self._compare_on)
        btn_compare.released.connect(self._compare_off)
        bar.addWidget(btn_compare)

        # 构图网格
        btn_grid = _make_btn("▦ 网格", color="#313244")
        btn_grid.clicked.connect(self._toggle_grid)
        bar.addWidget(btn_grid)

        # 全屏按钮
        btn_fs = _make_btn("⛶ 全屏 (F11)", color="#313244")
        btn_fs.clicked.connect(self._toggle_fullscreen)
        bar.addWidget(btn_fs)

        # 视图适配
        btn_fit = _make_btn("🖼 适配 (Ctrl+0)", color="#313244")
        btn_fit.clicked.connect(self._reset_view)
        bar.addWidget(btn_fit)

        # 另存为
        btn_export = _make_btn("📤 另存为…", color="#313244")
        btn_export.clicked.connect(self.export_image_as)
        bar.addWidget(btn_export)

        btn_save = QPushButton("💾 保存并返回 (Ctrl+S)")
        btn_save.setStyleSheet(
            "QPushButton { background:#a6e3a1; color:#1e1e2e; "
            "border-radius:6px; font-weight:bold; padding:6px 18px; font-size:13px; }"
            "QPushButton:hover { background:#94e2d5; }"
        )
        btn_save.clicked.connect(self.save_and_return)
        bar.addWidget(btn_save)

        btn_close = _make_btn("✖ 取消")
        btn_close.clicked.connect(self._try_close)
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

        # 裁剪比例锁定
        ratio_row = QHBoxLayout()
        self.combo_crop_aspect = QComboBox()
        self.combo_crop_aspect.addItems(
            ["自由", "1:1", "4:3", "3:4", "16:9", "9:16"])
        self.combo_crop_aspect.setStyleSheet(self._combo_style())
        ratio_row.addWidget(QLabel("比例:"))
        ratio_row.addWidget(self.combo_crop_aspect)
        layout.addLayout(ratio_row)

        # ── ✅ 画笔大小 — 改为滑块 ────────────────────────
        layout.addWidget(self._sec("🔘 画笔大小  ( [ / ] 微调 )"))
        brush_row = QHBoxLayout()
        self.sld_brush_size = _make_slider(1, 150, 20)
        self.lbl_brush_val  = QLabel("20 px")
        self.lbl_brush_val.setFixedWidth(45)
        self.lbl_brush_val.setStyleSheet("color:#cdd6f4; font-size:12px;")
        self.sld_brush_size.valueChanged.connect(
            lambda v: (self.lbl_brush_val.setText(f"{v} px"),
                       self._refresh_brush_cursor())
        )
        brush_row.addWidget(self.sld_brush_size)
        brush_row.addWidget(self.lbl_brush_val)
        layout.addLayout(brush_row)

        # ── 画笔不透明度 ──────────────────────────────────
        layout.addWidget(self._sec("💧 画笔不透明度"))
        op_row = QHBoxLayout()
        self.sld_brush_opacity = _make_slider(5, 100, 100)
        self.lbl_opacity_val   = QLabel("100%")
        self.lbl_opacity_val.setFixedWidth(45)
        self.lbl_opacity_val.setStyleSheet("color:#cdd6f4; font-size:12px;")
        self.sld_brush_opacity.valueChanged.connect(
            lambda v: self.lbl_opacity_val.setText(f"{v}%")
        )
        op_row.addWidget(self.sld_brush_opacity)
        op_row.addWidget(self.lbl_opacity_val)
        layout.addLayout(op_row)

        # ── 画笔硬度(柔边) ────────────────────────────────
        layout.addWidget(self._sec("✒️ 画笔硬度"))
        hd_row = QHBoxLayout()
        self.sld_brush_hardness = _make_slider(0, 100, 100)
        self.lbl_hardness_val   = QLabel("硬边")
        self.lbl_hardness_val.setFixedWidth(45)
        self.lbl_hardness_val.setStyleSheet("color:#cdd6f4; font-size:12px;")
        self.sld_brush_hardness.valueChanged.connect(
            lambda v: self.lbl_hardness_val.setText(
                "硬边" if v >= 100 else ("柔边" if v <= 30 else f"{v}"))
        )
        hd_row.addWidget(self.sld_brush_hardness)
        hd_row.addWidget(self.lbl_hardness_val)
        layout.addLayout(hd_row)

        # ── 文字大小 / 描边 ─────────────────────────────
        layout.addWidget(self._sec("🔤 文字大小 / 描边"))
        text_row = QHBoxLayout()
        self.spin_text_size = QSpinBox()
        self.spin_text_size.setRange(8, 200)
        self.spin_text_size.setValue(40)
        self.spin_text_size.setStyleSheet(self._input_style())
        text_row.addWidget(self.spin_text_size)
        self.spin_text_stroke = QSpinBox()
        self.spin_text_stroke.setRange(0, 8)
        self.spin_text_stroke.setValue(0)
        self.spin_text_stroke.setPrefix("描边 ")
        self.spin_text_stroke.setStyleSheet(self._input_style())
        text_row.addWidget(self.spin_text_stroke)
        layout.addLayout(text_row)

        # ── 颜色 ───────────────────────────────────────────
        layout.addWidget(self._sec("🎨 颜色"))
        cr = QHBoxLayout()
        btn_pc = _make_btn("画笔颜色")
        btn_pc.clicked.connect(self.pick_color)
        btn_drop = _make_btn("💧 吸管", tip="点击画布取样画笔颜色")
        btn_drop.clicked.connect(self.toggle_eyedropper)
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(26, 26)
        self.color_preview.setStyleSheet(
            f"background:{self.brush_color}; border-radius:4px;")
        cr.addWidget(btn_pc)
        cr.addWidget(btn_drop)
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

        # 任意角度旋转
        t3 = QHBoxLayout()
        self.spin_rotate_angle = QSpinBox()
        self.spin_rotate_angle.setRange(-180, 180)
        self.spin_rotate_angle.setValue(0)
        self.spin_rotate_angle.setSuffix(" °")
        self.spin_rotate_angle.setStyleSheet(self._input_style())
        bra = _make_btn("↻ 应用角度")
        bra.clicked.connect(self.rotate_image_any)
        t3.addWidget(self.spin_rotate_angle)
        t3.addWidget(bra)
        layout.addLayout(t3)

        # 等比缩放
        brs = _make_btn("📐 缩放到长边…",
                        tip="等比缩放到 512/768/1024 等 SD 常用尺寸")
        brs.clicked.connect(self.resize_dialog)
        layout.addWidget(brs)

        # ── 遮罩操作 ───────────────────────────────────────
        layout.addWidget(self._sec("🔴 遮罩操作"))
        m1 = QHBoxLayout()
        bmc = _make_btn("🧹 清除遮罩")
        bmi = _make_btn("🔁 反转遮罩")
        bmc.clicked.connect(self.clear_mask)
        bmi.clicked.connect(self.invert_mask)
        m1.addWidget(bmc); m1.addWidget(bmi)
        layout.addLayout(m1)

        # 羽化半径 + 应用
        mf_row = QHBoxLayout()
        self.sld_mask_feather = _make_slider(0, 30, 4)
        self.lbl_mask_feather = QLabel("4 px")
        self.lbl_mask_feather.setFixedWidth(40)
        self.lbl_mask_feather.setStyleSheet("color:#cdd6f4; font-size:11px;")
        self.sld_mask_feather.valueChanged.connect(
            lambda v: self.lbl_mask_feather.setText(f"{v} px"))
        bmf = _make_btn("🌫 羽化")
        bmf.clicked.connect(lambda: self.feather_mask())
        mf_row.addWidget(self.sld_mask_feather)
        mf_row.addWidget(self.lbl_mask_feather)
        mf_row.addWidget(bmf)
        layout.addLayout(mf_row)

        m2 = QHBoxLayout()
        bmg = _make_btn("➕ 扩边 4px")
        bms = _make_btn("➖ 收缩 4px")
        bmg.clicked.connect(lambda: self.grow_mask(4))
        bms.clicked.connect(lambda: self.shrink_mask(4))
        m2.addWidget(bmg); m2.addWidget(bms)
        layout.addLayout(m2)

        m3 = QHBoxLayout()
        bmp = _make_btn("🙈 遮罩预览 开/关",
                        tip="临时隐藏红色叠加,查看原图效果")
        bmp.clicked.connect(self.toggle_mask_preview)
        m3.addWidget(bmp)
        layout.addLayout(m3)

        # ── 历史 ───────────────────────────────────────────
        layout.addWidget(self._sec("⏪ 历史 (Ctrl+Z/Y)"))
        h1 = QHBoxLayout()
        bu = _make_btn("↩ 撤销")
        br = _make_btn("↪ 重做")
        bu.clicked.connect(self.undo)
        br.clicked.connect(self.redo)
        h1.addWidget(bu); h1.addWidget(br)
        layout.addLayout(h1)

        # ── ✅ 调色滑块（7 项）─────────────────────────
        layout.addWidget(self._sec("🎛 色彩调整"))
        adj_cfg = [
            ("brightness", "☀ 亮度",   -100, 100, 0),
            ("contrast",   "◑ 对比度", -100, 100, 0),
            ("saturation", "🌈 饱和度", -100, 100, 0),
            ("exposure",   "🔆 曝光",   -100, 100, 0),
            ("hue",        "🎨 色相",   -100, 100, 0),
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

        # ── ✅ 滤镜面板（18 种）──────────────────────────
        layout.addWidget(self._sec("✨ 预设滤镜"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "无", "黑白", "复古", "冷色调", "暖色调",
            "胶片颗粒", "模糊", "浮雕", "边缘检测",
            "轮廓", "锐化", "油画",
            "负片", "像素化", "晕影", "色调分离",
            "素描", "卡通",
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

        # 滤镜强度（模糊半径 / 像素化块大小）
        layout.addWidget(self._sec_small("🌫 滤镜强度 (模糊/像素化)"))
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

        filter_btn_row = QHBoxLayout()
        btn_preview_filter = _make_btn("👁 预览滤镜",
                                       tip="先看效果,不入历史,Esc 取消")
        btn_preview_filter.clicked.connect(self.preview_filter)
        btn_apply_filter = _make_btn("▶ 应用滤镜", color="#45475a")
        btn_apply_filter.clicked.connect(self.apply_selected_filter)
        filter_btn_row.addWidget(btn_preview_filter)
        filter_btn_row.addWidget(btn_apply_filter)
        layout.addLayout(filter_btn_row)

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

    def _combo_style(self) -> str:
        return (
            "QComboBox { background:#313244; color:#cdd6f4; "
            "border:1px solid #45475a; border-radius:4px; "
            "padding:4px; font-size:12px; }"
            "QComboBox::drop-down { border:none; }"
            "QComboBox QAbstractItemView { background:#313244; color:#cdd6f4; "
            "selection-background-color:#45475a; }"
        )

    # ----------------------------------------------------------
    #  状态栏信息区
    # ----------------------------------------------------------
    def _on_canvas_hover(self, x: int, y: int):
        """光标位置 + 取色值"""
        w, h = self.current_img.size
        if 0 <= x < w and 0 <= y < h:
            r, g, b = self.current_img.convert("RGB").getpixel((x, y))
            self.lbl_cursor.setText(f"({x},{y}) #{r:02x}{g:02x}{b:02x}")
        else:
            self.lbl_cursor.setText("")

    def _refresh_view_info(self):
        """图像尺寸 + 缩放百分比"""
        if not hasattr(self, "lbl_img_info"):
            return
        w, h = self.current_img.size
        pct  = self.canvas.current_scale_pct()
        self.lbl_img_info.setText(f"{w}×{h} @ {pct}%")

    # ----------------------------------------------------------
    #  单键快捷键(带输入框焦点保护)
    # ----------------------------------------------------------
    def keyPressEvent(self, e):
        from PyQt6.QtWidgets import QApplication, QLineEdit, QTextEdit, QSpinBox, QComboBox
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QTextEdit, QSpinBox, QComboBox)):
            super().keyPressEvent(e)
            return
        k = e.key()
        if   k == Qt.Key.Key_B: self.toggle_brush()
        elif k == Qt.Key.Key_E: self.toggle_eraser()
        elif k == Qt.Key.Key_M: self.toggle_mask_brush()
        elif k == Qt.Key.Key_G: self._toggle_grid()
        else: super().keyPressEvent(e)

    # ----------------------------------------------------------
    #  退出确认(有未保存修改时)
    # ----------------------------------------------------------
    def _confirm_discard(self) -> bool:
        """返回 True 表示可以关闭"""
        if not getattr(self, "_dirty", False):
            return True
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "放弃修改",
            "有未保存的修改,确定放弃并关闭吗?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _try_close(self):
        if self._confirm_discard():
            self.reject()

    def closeEvent(self, e):
        if self._confirm_discard():
            e.accept()
        else:
            e.ignore()

    # ----------------------------------------------------------
    #  画布更新
    # ----------------------------------------------------------
    def update_canvas(self, pil_img: Image.Image, force: bool = False):
        self.canvas.set_pixmap(_pil_to_qpixmap(pil_img))

    # ----------------------------------------------------------
    #  另存为(直接导出当前编辑结果,不影响回调流程)
    # ----------------------------------------------------------
    def export_image_as(self):
        from PyQt6.QtWidgets import QFileDialog
        import os
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "另存为", f"{base}_edited.png",
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg);;所有文件 (*)",
        )
        if not path:
            return
        try:
            img = self.current_img
            if path.lower().endswith((".jpg", ".jpeg")):
                img = img.convert("RGB")
            img.save(path)
            self.lbl_status.setText(f"📤 已导出: {os.path.basename(path)}")
            self.lbl_status.setStyleSheet("color:#a6e3a1; font-size:13px;")
        except Exception as e:
            self.lbl_status.setText(f"❌ 导出失败: {e}")
            self.lbl_status.setStyleSheet("color:#f38ba8; font-size:13px;")

    # ----------------------------------------------------------
    #  保存
    # ----------------------------------------------------------
    def save_and_return(self):
        if self.text_element:
            self._commit_text_to_image()

        logger.debug(f"mask extrema = {self.mask_img.getextrema()}, "
                     f"size = {self.mask_img.size}")

        if self.callback_on_save:
            try:
                self.callback_on_save(self.current_img, self.mask_img)
                logger.debug("callback_on_save 正常返回")
            except Exception as e:
                logger.error(f"callback_on_save 抛异常: {e}", exc_info=True)
        else:
            logger.warning("callback_on_save 为 None,不会触发后续流程")

        self.accept()