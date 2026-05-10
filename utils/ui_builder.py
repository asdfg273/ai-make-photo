# utils/ui_builder.py
# ============================================================
#  PyQt6 完整重构版 UI — 修复版
# ============================================================

import os
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox,
    QTabWidget, QGroupBox, QScrollArea, QSplitter,
    QFileDialog, QMessageBox, QProgressBar, QSizePolicy,
    QFrame, QSpacerItem,QListWidget, QListWidgetItem, QMenu, QAbstractItemView,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal,QSize
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QIcon,
    QPixmap, QPainter, QSurfaceFormat, QAction
)


# ============================================================
#  GPU 加速初始化
# ============================================================
def enable_gpu_acceleration():
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setVersion(3, 3)
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)


# ============================================================
#  FloatSlider
# ============================================================
class FloatSlider(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, minimum=0.0, maximum=1.0, step=0.01,
                 value=0.5, parent=None):
        super().__init__(parent)
        self._factor = round(1 / step)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(minimum * self._factor))
        self._slider.setMaximum(int(maximum * self._factor))
        self._slider.setValue(int(value * self._factor))
        self._slider.setFixedHeight(22)

        self._label = QLabel(f"{value:.2f}")
        self._label.setFixedWidth(40)
        self._label.setAlignment(Qt.AlignmentFlag.AlignRight |
                                 Qt.AlignmentFlag.AlignVCenter)
        self._label.setStyleSheet("color:#cdd6f4; font-family:Consolas;")

        self._layout.addWidget(self._slider)
        self._layout.addWidget(self._label)
        self._slider.valueChanged.connect(self._on_change)

    def _on_change(self, int_val):
        fval = int_val / self._factor
        self._label.setText(f"{fval:.2f}")
        self.valueChanged.emit(fval)

    def float_value(self) -> float:
        return self._slider.value() / self._factor

    def value(self) -> float:
        return self.float_value()

    def setValue(self, v: float):
        self._slider.setValue(int(v * self._factor))

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)
        color = "#cdd6f4" if enabled else "#585b70"
        self._label.setStyleSheet(f"color:{color}; font-family:Consolas;")


# ============================================================
#  GPU 画布 (QLabel + set_pixmap 兼容接口)
# ============================================================
class GpuCanvas(QLabel):
    """兼容 set_pixmap 接口的画布，自动等比缩放"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._raw_pixmap = None

    def set_pixmap(self, pixmap: QPixmap):
        self._raw_pixmap = pixmap
        self._refresh()

    def _refresh(self):
        if self._raw_pixmap and not self._raw_pixmap.isNull():
            scaled = self._raw_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()


# ============================================================
#  启动画面（修复版 — 移除错误代码）
# ============================================================
class SplashScreen(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.SplashScreen |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedSize(480, 320)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._setup_ui()
        self._center()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setObjectName("splash_container")
        container.setStyleSheet("""
            #splash_container {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e1e2e, stop:1 #181825
                );
                border-radius: 18px;
                border: 1px solid #45475a;
            }
        """)
        inner = QVBoxLayout(container)
        inner.setContentsMargins(40, 40, 40, 30)
        inner.setSpacing(14)

        ico_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "logo", "dzbut-9fc5g-001.ico"
        )
        lbl_icon = QLabel()
        lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.exists(ico_path):
            pix = QPixmap(ico_path).scaled(
                80, 80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            lbl_icon.setPixmap(pix)
        else:
            lbl_icon.setText("🎨")
            lbl_icon.setStyleSheet("font-size:64px;")
        inner.addWidget(lbl_icon)

        lbl_title = QLabel("AI 绘画工作站")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet(
            "color:#cba6f7; font-size:22px; font-weight:bold;")
        inner.addWidget(lbl_title)

        lbl_sub = QLabel("v5.0  PyQt6 Edition")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet("color:#6c7086; font-size:12px;")
        inner.addWidget(lbl_sub)

        self.lbl_msg = QLabel("正在初始化...")
        self.lbl_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_msg.setStyleSheet("color:#a6adc8; font-size:11px;")
        inner.addWidget(self.lbl_msg)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar { background:#313244; border-radius:3px; }
            QProgressBar::chunk { background:#cba6f7; border-radius:3px; }
        """)
        inner.addWidget(self.progress)
        layout.addWidget(container)
        # ❌ 删除了原来错误的 combo_lora_0 代码块（属于主窗口而不是这里）

    def _center(self):
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )

    def set_message(self, msg: str):
        self.lbl_msg.setText(msg)
        QApplication.processEvents()

    def finish_loading(self, main_window):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        QTimer.singleShot(300, lambda: (main_window.show(), self.close()))


def create_splash() -> SplashScreen:
    splash = SplashScreen()
    splash.show()
    QApplication.processEvents()
    return splash


# ============================================================
#  全局样式
# ============================================================
DARK_STYLE = """
QMainWindow, QDialog, QWidget {
    background:#1e1e2e; color:#cdd6f4;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}
QTabWidget::pane { border:1px solid #45475a; border-radius:6px; background:#181825; }
QTabBar::tab { background:#313244; color:#a6adc8; padding:6px 18px; border-radius:4px 4px 0 0; margin-right:2px; }
QTabBar::tab:selected { background:#cba6f7; color:#1e1e2e; font-weight:bold; }
QTabBar::tab:hover { background:#45475a; color:#cdd6f4; }
QPushButton { background:#313244; color:#cdd6f4; border:1px solid #45475a; border-radius:6px; padding:5px 14px; }
QPushButton:hover { background:#45475a; }
QPushButton:pressed { background:#cba6f7; color:#1e1e2e; }
QPushButton:disabled { background:#1e1e2e; color:#585b70; border-color:#313244; }
QComboBox { background:#313244; color:#cdd6f4; border:1px solid #45475a; border-radius:5px; padding:3px 8px; }
QComboBox QAbstractItemView { background:#313244; color:#cdd6f4; selection-background-color:#cba6f7; selection-color:#1e1e2e; }
QTextEdit, QLineEdit { background:#181825; color:#cdd6f4; border:1px solid #45475a; border-radius:5px; padding:4px; }
QTextEdit:focus, QLineEdit:focus { border-color:#cba6f7; }
QSlider::groove:horizontal { height:4px; background:#45475a; border-radius:2px; }
QSlider::handle:horizontal { background:#cba6f7; width:14px; height:14px; margin:-5px 0; border-radius:7px; }
QSlider::sub-page:horizontal { background:#cba6f7; border-radius:2px; }
QCheckBox { color:#cdd6f4; spacing:6px; }
QCheckBox::indicator { width:16px; height:16px; border:1px solid #45475a; border-radius:3px; background:#313244; }
QCheckBox::indicator:checked { background:#cba6f7; border-color:#cba6f7; }
QScrollBar:vertical { background:#1e1e2e; width:8px; border-radius:4px; }
QScrollBar::handle:vertical { background:#45475a; border-radius:4px; min-height:20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QGroupBox { border:1px solid #45475a; border-radius:6px; margin-top:14px; padding:8px; color:#a6adc8; font-size:12px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; color:#cba6f7; }
QLabel { background:transparent; }
QProgressBar { background:#313244; border-radius:4px; border:none; text-align:center; color:#cdd6f4; }
QProgressBar::chunk { background:#cba6f7; border-radius:4px; }
"""


# ============================================================
#  UIBuilderMixin
# ============================================================
class UIBuilderMixin:

    def setup_ui(self):
        self.setMinimumSize(1280, 800)
        self.setWindowTitle("AI 绘画工作站 v5.0")
        self.setStyleSheet(DARK_STYLE)

        ico_path = os.path.join("logo", "dzbut-9fc5g-001.ico")
        if os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([480, 800])
        splitter.setHandleWidth(2)
        root.addWidget(splitter)

        self._build_menu()
        self._build_statusbar()

        # ============================================================
        #  统一在 setup_ui 末尾建立 LoRA 列表 + 别名兼容映射
        # ============================================================
        self.combo_loras = [
            self.combo_lora_0, self.combo_lora_1, self.combo_lora_2,
        ]
        self.scale_loras = [
            self.scale_lora_0, self.scale_lora_1, self.scale_lora_2,
        ]

        # —— 兼容 app_generation.py 的旧变量名 —— 
        self.btn_gen        = self.btn_generate
        self.btn_stop       = self.btn_interrupt
        self.scale_str      = self.scale_strength
        self.scale_hires    = self.scale_hires_denoise
        self.progress_total = self.progress_gen
        # 进度子条（用顶部状态条复用）
        self.progress       = self.progress_gen

        # —— 兼容 app_events.py —— 
        self.preview_canvas = self.lbl_preview
        self.pose_canvas    = self.lbl_cn_thumb
        try:
            from utils.app_utils import OUTPUT_DIR
            self.gallery.reload_from_dir(OUTPUT_DIR, limit=80)
        except Exception as e:
            print(f"⚠️ 画廊初始化失败: {e}")

        self._init_defaults()

    # ----------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(480)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(self._build_tab_basic(),  "🎨 基础")
        tabs.addTab(self._build_tab_adv(),    "⚙️ 高级")
        tabs.addTab(self._build_tab_lora(),   "🧩 LoRA")
        tabs.addTab(self._build_tab_ctrl(),   "🕹 ControlNet")
        tabs.addTab(self._build_tab_hires(),  "🔍 Hires.fix")
        tabs.addTab(self._build_tab_xy(),     "📊 X/Y 矩阵")
        layout.addWidget(tabs, 1)

        layout.addWidget(self._build_gen_button_area())
        layout.addWidget(self._build_status_bar_widget())
        return w

    # ----------------------------------------------------------
    def _build_tab_basic(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        w.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # 模型 + 设备
        grp_model = QGroupBox("模型")
        gm = QFormLayout(grp_model)
        self.combo_model = QComboBox()
        self.combo_model.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_model.currentIndexChanged.connect(self.on_model_selected)
        gm.addRow("SD 模型:", self.combo_model)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["AUTO", "CUDA", "MPS", "CPU"])
        gm.addRow("运行设备:", self.combo_device)

        self.lbl_model_info = QLabel("请选择模型")
        self.lbl_model_info.setWordWrap(True)
        self.lbl_model_info.setStyleSheet("color:#a6adc8; font-size:11px;")
        gm.addRow(self.lbl_model_info)

        # 预设
        self.combo_preset = QComboBox()
        self.combo_preset.addItem("（无）")
        for name in getattr(self, '_prompt_presets', {}).keys():
            self.combo_preset.addItem(name)
        self.combo_preset.currentIndexChanged.connect(self.apply_preset)
        gm.addRow("提示词预设:", self.combo_preset)

        layout.addWidget(grp_model)

        # 提示词
        grp_prompt = QGroupBox("提示词")
        gp = QVBoxLayout(grp_prompt)
        lbl_pos = QLabel("正向 (中/英 均可):")
        lbl_pos.setStyleSheet("color:#a6e3a1;")
        gp.addWidget(lbl_pos)
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setFixedHeight(100)
        self.txt_prompt.setPlaceholderText("在此输入正向提示词...")
        gp.addWidget(self.txt_prompt)

        lbl_neg = QLabel("负向提示词:")
        lbl_neg.setStyleSheet("color:#f38ba8;")
        gp.addWidget(lbl_neg)
        self.txt_neg = QTextEdit()
        self.txt_neg.setFixedHeight(70)
        self.txt_neg.setPlaceholderText("在此输入负向提示词...")
        gp.addWidget(self.txt_neg)
        layout.addWidget(grp_prompt)

        # 基础参数
        grp_params = QGroupBox("基础参数")
        gpa = QFormLayout(grp_params)

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 150)
        self.spin_steps.setValue(30)
        gpa.addRow("步数 Steps:", self.spin_steps)

        self.scale_cfg = FloatSlider(1.0, 20.0, 0.5, 7.0)
        gpa.addRow("CFG Scale:", self.scale_cfg)

        # 分辨率（统一用 combo_res）
        self.combo_res = QComboBox()
        self.combo_res.addItems([
            "512x512", "512x768", "768x512", "768x768",
            "1024x1024", "832x1216", "1216x832"
        ])
        self.combo_res.setCurrentText("512x768")
        gpa.addRow("分辨率:", self.combo_res)

        # spin_width / spin_height 仍保留，方便高级面板
        self.spin_width = QSpinBox()
        self.spin_width.setRange(256, 2048)
        self.spin_width.setSingleStep(64)
        self.spin_width.setValue(512)

        self.spin_height = QSpinBox()
        self.spin_height.setRange(256, 2048)
        self.spin_height.setSingleStep(64)
        self.spin_height.setValue(768)

        # 数量（兼容名 spin_count + spin_batch）
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 32)
        self.spin_count.setValue(1)
        gpa.addRow("生成数量:", self.spin_count)
        self.spin_batch = self.spin_count  # 别名

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 2147483647)
        self.spin_seed.setValue(-1)
        self.spin_seed.setSpecialValueText("随机")
        gpa.addRow("种子 Seed:", self.spin_seed)

        self.combo_sampler = QComboBox()
        self.combo_sampler.addItems([
            "DPM++ 2M Karras", "DPM++ SDE Karras",
            "Euler a", "Euler", "DDIM", "UniPC"
        ])
        gpa.addRow("采样器:", self.combo_sampler)

        # 连环画
        self.chk_make_comic = QCheckBox("生成完后拼合分镜连环画")
        gpa.addRow(self.chk_make_comic)
        layout.addWidget(grp_params)

        # 图生图
        grp_i2i = QGroupBox("图生图 (可选)")
        gi = QVBoxLayout(grp_i2i)
        btn_row = QHBoxLayout()
        self.btn_load_img = QPushButton("📂 加载参考图")
        self.btn_load_img.clicked.connect(self.select_image)
        self.btn_clear_img = QPushButton("🗑 清除")
        self.btn_clear_img.clicked.connect(self.clear_reference)
        btn_row.addWidget(self.btn_load_img)
        btn_row.addWidget(self.btn_clear_img)
        gi.addLayout(btn_row)

        self.lbl_img_path = QLabel("未选择参考图")
        self.lbl_img_path.setStyleSheet("color:#585b70; font-size:11px;")
        gi.addWidget(self.lbl_img_path)

        self.lbl_ref_thumb = QLabel("无参考图")
        self.lbl_ref_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ref_thumb.setFixedHeight(80)
        self.lbl_ref_thumb.setStyleSheet(
            "border:1px dashed #45475a; border-radius:4px; color:#585b70;")
        gi.addWidget(self.lbl_ref_thumb)

        gi.addWidget(QLabel("重绘强度:"))
        self.scale_strength = FloatSlider(0.05, 1.0, 0.05, 0.6)
        gi.addWidget(self.scale_strength)
        layout.addWidget(grp_i2i)

        layout.addStretch()
        return w

    # ----------------------------------------------------------
    def _build_tab_adv(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        w.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # 修脸
        grp_face = QGroupBox("ADetailer — 修脸")
        gf = QFormLayout(grp_face)
        self.chk_use_adetailer = QCheckBox("开启修脸")
        self.chk_use_adetailer.toggled.connect(self._toggle_adetailer)
        gf.addRow(self.chk_use_adetailer)

        # 模型分类（真人/二次元）
        self.combo_adetailer_model = QComboBox()
        self.combo_adetailer_model.addItems(["真人脸", "二次元脸"])
        gf.addRow("脸部类型:", self.combo_adetailer_model)

        self.combo_ad_target = QComboBox()
        self.combo_ad_target.addItems([
            "现实脸部", "二次元脸部"
        ])
        gf.addRow("检测目标:", self.combo_ad_target)

        self.lbl_ad_str = QLabel("修复强度:")
        self.lbl_ad_str.setStyleSheet("color:#585b70; font-family:Consolas;")
        self.scale_adetailer_strength = FloatSlider(0.1, 0.9, 0.05, 0.35)
        gf.addRow(self.lbl_ad_str, self.scale_adetailer_strength)
        layout.addWidget(grp_face)

        # 修手
        grp_hand = QGroupBox("ADetailer — 修手")
        gh = QFormLayout(grp_hand)
        self.chk_use_ad_hand = QCheckBox("开启修手")
        self.chk_use_ad_hand.toggled.connect(self._toggle_ad_hand)
        gh.addRow(self.chk_use_ad_hand)

        self.combo_ad_hand = QComboBox()
        self.combo_ad_hand.addItems(["现实手部", "二次元手部"])
        gh.addRow("检测目标:", self.combo_ad_hand)

        self.lbl_ad_hand_str = QLabel("重绘强度:")
        self.lbl_ad_hand_str.setStyleSheet(
            "color:#585b70; font-family:Consolas;")
        self.scale_ad_hand = FloatSlider(0.1, 0.6, 0.05, 0.25)
        gh.addRow(self.lbl_ad_hand_str, self.scale_ad_hand)

        self.lbl_ad_hand_blend = QLabel("融合度:")
        self.lbl_ad_hand_blend.setStyleSheet(
            "color:#585b70; font-family:Consolas;")
        self.scale_ad_hand_blend = FloatSlider(0.0, 1.0, 0.05, 0.65)
        gh.addRow(self.lbl_ad_hand_blend, self.scale_ad_hand_blend)
        layout.addWidget(grp_hand)

        # 输出
        grp_out = QGroupBox("输出设置")
        go = QFormLayout(grp_out)
        self.combo_output_dir = QComboBox()
        self.combo_output_dir.setEditable(True)
        self.combo_output_dir.addItem("outputs/")
        go.addRow("输出目录:", self.combo_output_dir)

        self.combo_img_format = QComboBox()
        self.combo_img_format.addItems(["PNG", "JPEG", "WEBP"])
        go.addRow("图片格式:", self.combo_img_format)
        layout.addWidget(grp_out)

        # PNG 信息读取
        self.btn_read_png = QPushButton("📥 读取 PNG 中的生成参数")
        self.btn_read_png.clicked.connect(self.read_png_info)
        layout.addWidget(self.btn_read_png)

        layout.addStretch()
        return w

    # ----------------------------------------------------------
    def _build_tab_lora(self) -> QWidget:
        """⚙️ 真正创建 3 个 LoRA 槽位"""
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        w.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("LoRA 槽位 (按主模型架构自动过滤):"))
        self.btn_refresh_lora = QPushButton("🔄 刷新")
        self.btn_refresh_lora.setFixedWidth(70)
        self.btn_refresh_lora.clicked.connect(self.refresh_lora_by_model)
        hdr.addWidget(self.btn_refresh_lora)
        layout.addLayout(hdr)

        # 创建 3 个槽位 —— 关键修复！
        for i in range(3):
            grp = QGroupBox(f"LoRA 槽位 {i + 1}")
            gv = QFormLayout(grp)
            combo = QComboBox()
            combo.addItem("无")
            scale = FloatSlider(0.0, 2.0, 0.05, 0.8)
            gv.addRow("模型:", combo)
            gv.addRow("权重:", scale)
            layout.addWidget(grp)

            setattr(self, f'combo_lora_{i}', combo)
            setattr(self, f'scale_lora_{i}', scale)
            combo.currentIndexChanged.connect(self.load_lora_info)

        lbl_info = QLabel("LoRA 备忘录:")
        layout.addWidget(lbl_info)
        self.text_lora_info = QTextEdit()
        self.text_lora_info.setReadOnly(True)
        self.text_lora_info.setFixedHeight(120)
        self.text_lora_info.setStyleSheet(
            "font-family:Consolas; font-size:11px;")
        layout.addWidget(self.text_lora_info)
        layout.addStretch()
        return w

    # ----------------------------------------------------------
    def _build_tab_ctrl(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        w.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        grp = QGroupBox("ControlNet")
        gv = QFormLayout(grp)

        self.chk_use_pose = QCheckBox("开启 ControlNet")
        self.chk_use_pose.toggled.connect(self._toggle_cn)
        gv.addRow(self.chk_use_pose)

        self.combo_cn_type = QComboBox()
        self.combo_cn_type.addItems([
            "OpenPose", "Canny", "Depth", "Scribble", "SoftEdge"
        ])
        gv.addRow("类型:", self.combo_cn_type)

        self.scale_cn_weight = FloatSlider(0.0, 2.0, 0.05, 1.0)
        gv.addRow("权重:", self.scale_cn_weight)

        self.btn_load_cn_img = QPushButton("📂 加载姿态图")
        self.btn_load_cn_img.clicked.connect(self.load_pose_image)
        gv.addRow(self.btn_load_cn_img)

        self.lbl_pose_path = QLabel("未加载动作图")
        self.lbl_pose_path.setStyleSheet("color:#585b70; font-size:11px;")
        gv.addRow(self.lbl_pose_path)

        self.lbl_cn_thumb = GpuCanvas()
        self.lbl_cn_thumb.setText("未加载")
        self.lbl_cn_thumb.setFixedHeight(180)
        self.lbl_cn_thumb.setStyleSheet(
            "border:1px dashed #45475a; border-radius:4px; color:#585b70;")
        gv.addRow(self.lbl_cn_thumb)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    # ----------------------------------------------------------
    def _build_tab_hires(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        w.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        grp = QGroupBox("Hires.fix 高清修复")
        gv = QFormLayout(grp)

        self.chk_hires = QCheckBox("开启 Hires.fix")
        self.chk_hires.toggled.connect(self._toggle_hires)
        gv.addRow(self.chk_hires)

        # XY 模式专用复选
        self.chk_enable_hires = QCheckBox("XY 矩阵中也启用 Hires.fix")
        gv.addRow(self.chk_enable_hires)

        self.combo_hires_scale = QComboBox()
        self.combo_hires_scale.addItems(["1.5", "2.0", "2.5", "3.0"])
        self.combo_hires_scale.setCurrentText("2.0")
        gv.addRow("放大倍率:", self.combo_hires_scale)

        self.scale_hires_denoise = FloatSlider(0.1, 0.9, 0.05, 0.45)
        gv.addRow("降噪强度:", self.scale_hires_denoise)

        self.combo_hires_upscaler = QComboBox()
        self.combo_hires_upscaler.addItems([
            "Latent", "ESRGAN_4x", "R-ESRGAN 4x+", "SwinIR"
        ])
        gv.addRow("Upscaler:", self.combo_hires_upscaler)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    # ----------------------------------------------------------
    def _build_tab_xy(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        w.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        grp = QGroupBox("X/Y 矩阵生成")
        gv = QFormLayout(grp)

        self.chk_enable_xy = QCheckBox("开启 X/Y 矩阵")
        self.chk_enable_xy.toggled.connect(self._toggle_xy)
        gv.addRow(self.chk_enable_xy)

        self.combo_x_type = QComboBox()
        self.combo_x_type.addItems([
            "Steps", "CFG Scale", "Sampler", "Seed", "LoRA 权重"
        ])
        gv.addRow("X 轴类型:", self.combo_x_type)
        self.entry_x_vals = QLineEdit()
        self.entry_x_vals.setPlaceholderText("如: 10,20,30 或 7,9,11")
        gv.addRow("X 轴值:", self.entry_x_vals)

        self.combo_y_type = QComboBox()
        self.combo_y_type.addItems([
            "Steps", "CFG Scale", "Sampler", "Seed", "LoRA 权重"
        ])
        gv.addRow("Y 轴类型:", self.combo_y_type)
        self.entry_y_vals = QLineEdit()
        self.entry_y_vals.setPlaceholderText("如: 0.4,0.6,0.8")
        gv.addRow("Y 轴值:", self.entry_y_vals)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    # ----------------------------------------------------------
    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.lbl_preview = GpuCanvas()
        self.lbl_preview.setText("等待生成...")
        self.lbl_preview.setMinimumHeight(400)
        self.lbl_preview.setStyleSheet(
            "background:#181825; border:1px solid #45475a;"
            "border-radius:8px; color:#585b70; font-size:16px;")
        self.lbl_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.lbl_preview, 3)

        btn_row = QHBoxLayout()
        self.btn_edit = QPushButton("✏️ 局部重绘")
        self.btn_edit.clicked.connect(self.open_editor)
        self.btn_edit.setEnabled(False)

        self.btn_upscale = QPushButton("🔍 高清放大")
        self.btn_upscale.clicked.connect(
            lambda: self.start_upscale()
            if hasattr(self, 'start_upscale') else None)
        self.btn_upscale.setEnabled(False)

        self.btn_open_editor = QPushButton("🎨 打开调色")
        self.btn_open_editor.clicked.connect(self.open_gallery_to_edit)

        self.btn_open_folder = QPushButton("📁 打开输出")
        self.btn_open_folder.clicked.connect(self._open_output_folder)

        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_upscale)
        btn_row.addWidget(self.btn_open_editor)
        btn_row.addWidget(self.btn_open_folder)
        layout.addLayout(btn_row)

        # ─────────── 🖼 历史画廊 ───────────
        lbl_gallery_title = QLabel("🖼 历史画廊 (双击载入 · 右键菜单)")
        lbl_gallery_title.setStyleSheet(
            "color:#cba6f7; font-weight:bold; padding:2px;")
        layout.addWidget(lbl_gallery_title)

        self.gallery = GalleryPanel()
        self.gallery.setMinimumHeight(140)
        self.gallery.setMaximumHeight(200)
        # 双击画廊里的图 → 载入预览
        self.gallery.image_selected.connect(self._on_gallery_picked)
        layout.addWidget(self.gallery, 1)

        # ─────────── 日志 ───────────
        lbl_log = QLabel("📋 生成日志:")
        lbl_log.setStyleSheet("color:#a6adc8;")
        layout.addWidget(lbl_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(140)
        self.txt_log.setStyleSheet(
            "background:#11111b; font-family:Consolas; font-size:11px;")
        layout.addWidget(self.txt_log, 1)
        return w

    # ----------------------------------------------------------
    def _on_gallery_picked(self, path: str):
        """画廊双击 → 把这张图设为当前预览 + last_generated_path"""
        if hasattr(self, 'show_preview'):
            self.show_preview(path)
        else:
            try:
                pix = QPixmap(path)
                if not pix.isNull():
                    self.lbl_preview.set_pixmap(pix)
            except Exception:
                pass
        self.last_generated_path = path
        # 启用后处理按钮
        if hasattr(self, 'btn_edit'):    self.btn_edit.setEnabled(True)
        if hasattr(self, 'btn_upscale'): self.btn_upscale.setEnabled(True)

    # ----------------------------------------------------------
    def _build_gen_button_area(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        self.btn_generate = QPushButton("🚀  开始生成")
        self.btn_generate.setFixedHeight(46)
        self.btn_generate.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #cba6f7, stop:1 #89b4fa
                );
                color:#1e1e2e; font-size:15px; font-weight:bold;
                border:none; border-radius:8px;
            }
            QPushButton:pressed { background:#a6e3a1; }
            QPushButton:disabled{ background:#313244; color:#585b70; }
        """)
        self.btn_generate.clicked.connect(self.start_generation)
        layout.addWidget(self.btn_generate)

        self.btn_interrupt = QPushButton("⏹  中断生成")
        self.btn_interrupt.setFixedHeight(32)
        self.btn_interrupt.setEnabled(False)
        self.btn_interrupt.setStyleSheet("""
            QPushButton {
                background:#313244; color:#f38ba8;
                border:1px solid #f38ba8; border-radius:6px;
            }
            QPushButton:hover { background:#45475a; }
        """)
        self.btn_interrupt.clicked.connect(self.stop_generation)
        layout.addWidget(self.btn_interrupt)
        return w

    # ----------------------------------------------------------
    def _build_status_bar_widget(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_status = QLabel("✅ 就绪")
        self.lbl_status.setStyleSheet(
            "color:#a6e3a1; font-family:Consolas;")
        layout.addWidget(self.lbl_status, 1)

        self.progress_gen = QProgressBar()
        self.progress_gen.setRange(0, 100)
        self.progress_gen.setValue(0)
        self.progress_gen.setFixedWidth(160)
        self.progress_gen.setFixedHeight(8)
        self.progress_gen.setTextVisible(False)
        layout.addWidget(self.progress_gen)
        return w

    # ----------------------------------------------------------
    def _build_menu(self):
        mb = self.menuBar()
        mb.setStyleSheet("""
            QMenuBar { background:#181825; color:#cdd6f4; border-bottom:1px solid #45475a; }
            QMenuBar::item:selected { background:#313244; }
            QMenu { background:#1e1e2e; color:#cdd6f4; border:1px solid #45475a; }
            QMenu::item:selected { background:#cba6f7; color:#1e1e2e; }
        """)
        m_file = mb.addMenu("📁 文件")
        a_open = QAction("加载图片", self); a_open.triggered.connect(self.select_image)
        m_file.addAction(a_open)
        m_file.addSeparator()
        a_quit = QAction("退出", self); a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_tool = mb.addMenu("🔧 工具")
        a_clear_log = QAction("清空日志", self)
        a_clear_log.triggered.connect(lambda: self.txt_log.clear())
        m_tool.addAction(a_clear_log)

        m_about = mb.addMenu("❓ 关于")
        a_about = QAction("关于本软件", self)
        a_about.triggered.connect(self._show_about)
        m_about.addAction(a_about)

    def _build_statusbar(self):
        self.statusBar().setStyleSheet(
            "background:#181825; color:#6c7086; font-size:11px;")
        self.statusBar().showMessage("AI 绘画工作站 v5.0 已就绪")

    # ----------------------------------------------------------
    def _init_defaults(self):
        if hasattr(self, 'refresh_models'):
            try: self.refresh_models()
            except Exception as e: print(f"refresh_models 失败: {e}")
        self._toggle_adetailer()
        self._toggle_ad_hand()
        self._toggle_hires()
        self._toggle_xy()
        self._toggle_cn()

    # ----------------------------------------------------------
    def _toggle_adetailer(self):
        on = self.chk_use_adetailer.isChecked()
        self.combo_ad_target.setEnabled(on)
        self.combo_adetailer_model.setEnabled(on)
        self.scale_adetailer_strength.setEnabled(on)
        color = "#a6e3a1" if on else "#585b70"
        self.lbl_ad_str.setStyleSheet(
            f"color:{color}; font-family:Consolas;")

    def _toggle_ad_hand(self):
        on = self.chk_use_ad_hand.isChecked()
        self.combo_ad_hand.setEnabled(on)
        self.scale_ad_hand.setEnabled(on)
        self.scale_ad_hand_blend.setEnabled(on)
        color = "#a6e3a1" if on else "#585b70"
        self.lbl_ad_hand_str.setStyleSheet(
            f"color:{color}; font-family:Consolas;")
        self.lbl_ad_hand_blend.setStyleSheet(
            f"color:{color}; font-family:Consolas;")

    def _toggle_hires(self):
        on = self.chk_hires.isChecked()
        self.combo_hires_scale.setEnabled(on)
        self.scale_hires_denoise.setEnabled(on)
        self.combo_hires_upscaler.setEnabled(on)

    def _toggle_xy(self):
        on = self.chk_enable_xy.isChecked()
        for w in (self.combo_x_type, self.entry_x_vals,
                  self.combo_y_type, self.entry_y_vals):
            w.setEnabled(on)

    def _toggle_cn(self):
        on = self.chk_use_pose.isChecked()
        self.combo_cn_type.setEnabled(on)
        self.scale_cn_weight.setEnabled(on)
        self.btn_load_cn_img.setEnabled(on)

    # ----------------------------------------------------------
    def _open_output_folder(self):
        import subprocess
        from utils.app_utils import OUTPUT_DIR
        out_dir = OUTPUT_DIR
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile(os.path.abspath(out_dir))
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', out_dir])
            else:
                subprocess.Popen(['xdg-open', out_dir])
        except Exception as e:
            print(f"打开目录失败: {e}")

    def _show_about(self):
        QMessageBox.about(
            self, "关于",
            "<b>AI 绘画工作站 v5.0</b><br>"
            "PyQt6 重构版 — GPU 加速<br><br>"
            "基于 Stable Diffusion + ADetailer<br>"
            "支持 LoRA / ControlNet / Hires.fix"
        )

    def append_log(self, text: str, color: str = "#cdd6f4"):
        self.txt_log.append(
            f'<span style="color:{color}; font-family:Consolas;">{text}</span>'
        )
        self.txt_log.verticalScrollBar().setValue(
            self.txt_log.verticalScrollBar().maximum()
        )

    def set_status(self, text: str, color: str = "#a6e3a1"):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(
            f"color:{color}; font-family:Consolas;")

    # 兼容老接口
    def _set_status(self, text: str, color: str = "#a6e3a1"):
        self.set_status(text, color)

    def set_progress(self, value: int):
        self.progress_gen.setValue(value)

class GalleryPanel(QListWidget):
    """生成历史缩略图列表,双击预览,右键菜单。"""

    image_selected = pyqtSignal(str)      
    image_deleted  = pyqtSignal(str)    

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(96, 96))
        self.setGridSize(QSize(108, 108))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.setSpacing(4)
        self.setUniformItemSizes(True)
        self.setStyleSheet("""
            QListWidget {
                background-color: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background-color: #45475a;
                border: 2px solid #cba6f7;
            }
        """)

        self.itemDoubleClicked.connect(self._on_dbl)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_menu)

    # -------------------------------------------
    def add_image(self, path: str, prepend: bool = True):
        """添加一张图;prepend=True 表示插在最前面。"""
        import os
        if not path or not os.path.exists(path):
            return
        pix = QPixmap(path)
        if pix.isNull():
            return
        icon = QIcon(pix)
        item = QListWidgetItem(icon, "")
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(os.path.basename(path))
        if prepend:
            self.insertItem(0, item)
        else:
            self.addItem(item)
        # 控制总量,避免缩略图太多
        max_items = 200
        while self.count() > max_items:
            self.takeItem(self.count() - 1)

    # -------------------------------------------
    def reload_from_dir(self, dir_path: str, limit: int = 80):
        """启动时扫目录回填历史。"""
        import os, glob
        self.clear()
        if not os.path.isdir(dir_path):
            return
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(dir_path, ext)))
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for p in files[:limit]:
            self.add_image(p, prepend=False)

    # -------------------------------------------
    def _on_dbl(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.image_selected.emit(path)

    def _on_menu(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)

        menu = QMenu(self)
        act_open   = menu.addAction("🖼  载入预览")
        act_folder = menu.addAction("📂 在文件夹中显示")
        menu.addSeparator()
        act_del    = menu.addAction("🗑  从画廊移除")

        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen == act_open:
            self.image_selected.emit(path)
        elif chosen == act_folder:
            self._reveal_in_folder(path)
        elif chosen == act_del:
            row = self.row(item)
            self.takeItem(row)
            self.image_deleted.emit(path)

    def _reveal_in_folder(self, path: str):
        import os, sys, subprocess
        if not os.path.exists(path):
            return
        try:
            if sys.platform == "win32":
                subprocess.Popen(['explorer', '/select,', path])
            elif sys.platform == "darwin":
                subprocess.Popen(['open', '-R', path])
            else:
                subprocess.Popen(['xdg-open', os.path.dirname(path)])
        except Exception as e:
            print(f"⚠️ 打开文件夹失败: {e}")