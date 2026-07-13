# file_name: ui_builder.py
import os
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox,
    QTabWidget, QGroupBox, QScrollArea, QSplitter,
    QFileDialog, QMessageBox, QProgressBar, QSizePolicy,
    QFrame, QSpacerItem, QListWidget, QListWidgetItem,
    QMenu, QAbstractItemView, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize, QUrl, QThread
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QIcon,
    QPixmap, QPainter, QSurfaceFormat, QAction,QImage
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QGroupBox
import core.presets
from utils.gallery_panel import *
from utils.gallery_panel import GalleryPanel
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
#  FloatSlider —— 浮点滑块
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
        self._label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._label.setStyleSheet("color:#ffffff; font-family:Consolas;")

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
        color = "#ffffff" if enabled else "#7d8187"
        self._label.setStyleSheet(f"color:{color}; font-family:Consolas;")


# ============================================================
#  GpuCanvas —— 自适应缩放画布
# ============================================================
class GpuCanvas(QLabel):
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
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()


# ============================================================
#  启动画面
# ============================================================
class SplashScreen(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
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
                background: #0a0a0a;
                border-radius: 8px;
                border: 1px solid #212327;
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
            "color:#ffffff; font-size:22px;")
        inner.addWidget(lbl_title)

        lbl_sub = QLabel("v5.0  PyQt6 Edition")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet("color:#7d8187; font-size:12px;")
        inner.addWidget(lbl_sub)

        self.lbl_msg = QLabel("正在初始化...")
        self.lbl_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_msg.setStyleSheet("color:#7d8187; font-size:11px;")
        inner.addWidget(self.lbl_msg)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar { background:#1a1c20; border-radius:3px; }
            QProgressBar::chunk { background:#ffffff; border-radius:3px; }
        """)
        inner.addWidget(self.progress)
        layout.addWidget(container)

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
#  xAI 设计令牌 (Design Tokens)
#  基于 DESIGN.md 规范定义的设计系统常量
# ============================================================
DESIGN_TOKENS = {
    # 颜色
    'colors': {
        'primary': '#ffffff',
        'on-primary': '#0a0a0a',
        'ink': '#ffffff',
        'ink-hover': '#fafaf7',
        'body': '#dadbdf',
        'body-mid': '#7d8187',
        'mute': '#7d8187',
        'hairline': '#212327',
        'canvas': '#0a0a0a',
        'canvas-soft': '#1a1c20',
        'canvas-card': '#191919',
        'canvas-mid': '#363a3f',
        'accent-sunset': '#ff7a17',
        'accent-sunset-soft': '#ffc285',
        'accent-dusk': '#7c3aed',
        'accent-twilight': '#c4b5fd',
        'accent-breeze': '#a0c3ec',
        'accent-midnight': '#0d1726',
    },
    # 圆角
    'rounded': {
        'none': 0,
        'sm': 8,
        'pill': 9999,
        'full': 9999,
    },
    # 间距
    'spacing': {
        'xxs': 2,
        'xs': 4,
        'sm': 8,
        'md': 12,
        'lg': 16,
        'xl': 24,
        '2xl': 32,
        '3xl': 48,
        '4xl': 64,
    },
    # 字体
    'fonts': {
        'display': '"Segoe UI", "Microsoft YaHei", sans-serif',
        'mono': 'Consolas, "JetBrains Mono", "IBM Plex Mono", monospace',
    },
}


# ============================================================
#  全局深色样式
# ============================================================
DARK_STYLE = """
QMainWindow, QDialog, QWidget {
    background:#0a0a0a; color:#ffffff;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 14px;
}
QTabWidget::pane { border:1px solid #212327; border-radius:8px; background:#191919; }
QTabBar::tab { background:#0a0a0a; color:#7d8187; padding:6px 16px; border:1px solid #212327; border-radius:9999px; margin-right:4px; }
QTabBar::tab:selected { background:#ffffff; color:#0a0a0a; border-color:#ffffff; }
QTabBar::tab:hover { color:#ffffff; border-color:#363a3f; }
QPushButton { background:#0a0a0a; color:#ffffff; border:1px solid #212327; border-radius:9999px; padding:6px 16px; }
QPushButton:hover { border-color:#363a3f; }
QPushButton:pressed { background:#1a1c20; }
QPushButton:disabled { background:#0a0a0a; color:#363a3f; border-color:#212327; }
QComboBox { background:#1a1c20; color:#ffffff; border:1px solid #212327; border-radius:8px; padding:4px 8px; }
QComboBox QAbstractItemView { background:#191919; color:#ffffff; selection-background-color:#363a3f; selection-color:#ffffff; border:1px solid #212327; }
QTextEdit, QLineEdit { background:#1a1c20; color:#ffffff; border:1px solid #212327; border-radius:8px; padding:8px 12px; }
QTextEdit:focus, QLineEdit:focus { border-color:#ffffff; }
QSpinBox, QDoubleSpinBox { background:#1a1c20; color:#ffffff; border:1px solid #212327; border-radius:8px; padding:4px 8px; }
QSlider::groove:horizontal { height:4px; background:#212327; border-radius:2px; }
QSlider::handle:horizontal { background:#ffffff; width:14px; height:14px; margin:-5px 0; border-radius:7px; }
QSlider::sub-page:horizontal { background:#ffffff; border-radius:2px; }
QCheckBox { color:#ffffff; spacing:8px; }
QCheckBox::indicator { width:16px; height:16px; border:1px solid #212327; border-radius:3px; background:#191919; }
QCheckBox::indicator:checked { background:#ffffff; border-color:#ffffff; }
QScrollBar:vertical { background:#0a0a0a; width:8px; border-radius:4px; }
QScrollBar::handle:vertical { background:#363a3f; border-radius:4px; min-height:20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QGroupBox { border:1px solid #212327; border-radius:8px; margin-top:16px; padding:12px; color:#dadbdf; font-size:13px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 8px; color:#ffffff; font-size:13px; }
QLabel { background:transparent; }
QProgressBar { background:#1a1c20; border-radius:4px; border:1px solid #212327; text-align:center; color:#ffffff; }
QProgressBar::chunk { background:#ffffff; border-radius:4px; }
QToolTip {
    background-color: #191919;
    color: #ffffff;
    border: 1px solid #212327;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    font-family: "Segoe UI", Consolas;
}
"""

# ============================================================
#  UIBuilderMixin —— 主窗口 UI 构造器
# ============================================================
class UIBuilderMixin:

    # ----------------------------------------------------------
    #  主入口
    # ----------------------------------------------------------
    def setup_ui(self):
        self.setMinimumSize(1320, 820)
        self.setWindowTitle("AI 绘画工作站 v5.0")
        self.setStyleSheet(DARK_STYLE)

        ico_path = os.path.join("logo", "dzbut-9fc5g-001.ico")
        if os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))

        # IP-Adapter 参考图路径(挂在主窗口上)
        self.ipa_image_path = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.left_panel = self._build_left_panel()
        self.right_panel = self._build_right_panel()
        self.video_right_panel = self._build_video_right_panel()

        self.right_stacked = QStackedWidget()
        self.right_stacked.addWidget(self.right_panel)
        self.right_stacked.addWidget(self.video_right_panel)
        self.right_stacked.setCurrentIndex(0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_stacked)
        self.main_splitter.setSizes([500, 820])
        self.main_splitter.setHandleWidth(2)
        root.addWidget(self.main_splitter)

        self._build_menu()
        self._build_statusbar()

        # ============== 兼容别名 ==============
        # LoRA 列表
        self.combo_loras = [
            self.combo_lora_0, self.combo_lora_1, self.combo_lora_2]
        self.scale_loras = [
            self.scale_lora_0, self.scale_lora_1, self.scale_lora_2]
        # app_generation.py 旧名
        self.btn_gen = self.btn_generate
        self.btn_stop = self.btn_interrupt
        self.scale_str = self.scale_strength
        self.scale_hires = self.scale_hires_denoise
        self.progress_total = self.progress_gen
        self.progress = self.progress_gen
        # app_events.py 旧名
        self.preview_canvas = self.lbl_preview
        self.pose_canvas = self.lbl_cn_thumb

        # 启动时回填画廊
        try:
            from utils.app_utils import OUTPUT_DIR
            self.gallery.reload_from_dir(OUTPUT_DIR, limit=80)
        except Exception as e:
            print(f"⚠️ 画廊初始化失败: {e}")

        self._init_defaults()
        print(f"[BUILD-5] setup_ui 完成, combo_preset={hasattr(self, 'combo_preset')}")
    # ============================================================
    #  左侧面板
    # ============================================================
    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(500)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(self._build_tab_basic(), "🎨 基础")
        tabs.addTab(self._build_tab_animation(), "🌀 动画")
        tabs.addTab(self._build_tab_img2img(), "🖼 图生图")
        tabs.addTab(self._build_tab_lora(), "🧩 LoRA")
        tabs.addTab(self._build_tab_ctrl(), "🕹 ControlNet")
        tabs.addTab(self._build_tab_advanced(), "⚙️ 高级")
        tabs.addTab(self._build_tab_xy(), "📊 X/Y 矩阵")
        tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs = tabs
        layout.addWidget(tabs, 1)

        layout.addWidget(self._build_gen_button_area())
        layout.addWidget(self._build_status_bar_widget())
        return w

    def _on_tab_changed(self, index: int):
        """标签页切换事件处理"""
        if index == 1:
            self._switch_to_video_mode()
        else:
            self._switch_to_image_mode()

    def _switch_to_video_mode(self):
        """切换到视频模式（切换右侧面板）"""
        try:
            if hasattr(self, 'right_stacked'):
                self.right_stacked.setCurrentIndex(1)
                print("✅ 已切换到视频模式")
                QTimer.singleShot(100, self._refresh_video_gallery)
            else:
                print("⚠️ right_stacked 不存在")
        except Exception as e:
            import traceback
            print(f"⚠️ 切换视频模式失败: {e}")
            print(traceback.format_exc())

    def _switch_to_image_mode(self):
        """切换回图片模式（恢复右侧面板）"""
        try:
            if hasattr(self, 'right_stacked'):
                self.right_stacked.setCurrentIndex(0)
                print("✅ 已切换到图片模式")
                try:
                    from utils.app_utils import OUTPUT_DIR
                    if hasattr(self, 'gallery'):
                        self.gallery.reload_from_dir(OUTPUT_DIR, limit=80)
                except Exception as e:
                    print(f"⚠️ 画廊刷新失败: {e}")
            else:
                print("⚠️ right_stacked 不存在")
        except Exception as e:
            import traceback
            print(f"⚠️ 切换图片模式失败: {e}")
            print(traceback.format_exc())

    # ============================================================
    #  Tab 1: 基础
    # ============================================================
    def _build_tab_basic(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        w.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # ============== 1. 模型与设备 ==============
        grp_model = QGroupBox("模型与设备")
        gm = QFormLayout(grp_model)
        gm.setSpacing(8)

        # === 模型类型选择 ===
        self.combo_model_type = QComboBox()
        self.combo_model_type.addItem("SD 1.5  (轻量,4GB)", "sd15")
        self.combo_model_type.addItem("SDXL  (高质量,8GB)", "sdxl")
        self.combo_model_type.addItem("SD3/SD3.5  (新一代,12GB+)", "sd3")
        self.combo_model_type.addItem("Flux  (强,需GGUF量化)", "flux")
        self.combo_model_type.currentIndexChanged.connect(self._on_model_type_changed)
        gm.addRow("模型类型:", self.combo_model_type)

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
        self.lbl_model_info.setStyleSheet("color:#7d8187; font-size:11px;")
        gm.addRow(self.lbl_model_info)

        # ---------- 场景预设行 ----------
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)

        # 下拉框
        self.combo_preset = QComboBox()
        self.combo_preset.addItem("（无）")
        self.combo_preset.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.combo_preset.customContextMenuRequested.connect(self.show_preset_menu)
        self.combo_preset.currentIndexChanged.connect(self.apply_preset)
        self.combo_preset.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        preset_row.addWidget(self.combo_preset, 1)

        # 改动徽章
        self.lbl_preset_badge = QLabel("")
        self.lbl_preset_badge.setStyleSheet(
            "color:#dadbdf; font-weight:bold; font-size:11px; padding:0 4px;")
        self.lbl_preset_badge.setMinimumWidth(60)
        preset_row.addWidget(self.lbl_preset_badge)

        # 通用按钮样式
        def _mini_btn_style(color: str) -> str:
            return f"""
                QPushButton {{
                    background:#0a0a0a; color:{color};
                    border:1px solid #212327; border-radius:9999px;
                    padding:2px; font-size:13px;
                }}
                QPushButton:hover {{ border-color:{color}; }}
                QPushButton:pressed {{ background:#1a1c20; }}
            """

        # 保存
        self.btn_save_preset = QPushButton("💾")
        self.btn_save_preset.setFixedSize(32, 28)
        self.btn_save_preset.setToolTip("把当前所有参数保存为新预设")
        self.btn_save_preset.clicked.connect(self.save_current_as_preset)
        self.btn_save_preset.setStyleSheet(_mini_btn_style("#dadbdf"))
        preset_row.addWidget(self.btn_save_preset)

        # ↩还原
        self.btn_restore_preset = QPushButton("↩️")
        self.btn_restore_preset.setFixedSize(32, 28)
        self.btn_restore_preset.setToolTip("还原到套用预设前的所有参数")
        self.btn_restore_preset.clicked.connect(self.restore_preset_backup)
        self.btn_restore_preset.setStyleSheet(_mini_btn_style("#dadbdf"))
        preset_row.addWidget(self.btn_restore_preset)

        # 更多
        self.btn_preset_menu = QPushButton("⋮")
        self.btn_preset_menu.setFixedSize(28, 28)
        self.btn_preset_menu.setToolTip("更多操作（删除/导入/导出）")
        self.btn_preset_menu.clicked.connect(
            lambda: self.show_preset_menu(self.btn_preset_menu.rect().bottomLeft()))
        self.btn_preset_menu.setStyleSheet(_mini_btn_style("#ffffff"))
        preset_row.addWidget(self.btn_preset_menu)

        gm.addRow("🎨 场景预设:", preset_row)

        layout.addWidget(grp_model)

        # ============== 2. 提示词 ==============
        grp_prompt = QGroupBox("提示词")
        gp = QVBoxLayout(grp_prompt)
        gp.setSpacing(6)

        lbl_pos = QLabel("正向 (中/英 均可):")
        lbl_pos.setStyleSheet("color:#dadbdf; font-weight:bold;")
        gp.addWidget(lbl_pos)

        self.txt_prompt = QTextEdit()
        self.txt_prompt.setFixedHeight(100)
        self.txt_prompt.setPlaceholderText("在此输入正向提示词...")
        self.lbl_dynamic_hint = QLabel(
            "💡 提示：使用 {红|蓝|白} 语法可批量生成所有组合"
        )
        self.lbl_dynamic_hint.setStyleSheet("color: #7d8187; font-size: 11px;")
        layout.addWidget(self.lbl_dynamic_hint)

        gp.addWidget(self.txt_prompt)

        lbl_neg = QLabel("负向提示词:")
        lbl_neg.setStyleSheet("color:#dadbdf; font-weight:bold;")
        gp.addWidget(lbl_neg)

        self.txt_neg = QTextEdit()
        self.txt_neg.setFixedHeight(70)
        self.txt_neg.setPlaceholderText("在此输入负向提示词...")
        gp.addWidget(self.txt_neg)

        # AI 工具按钮行
        prompt_btn_row = QHBoxLayout()
        prompt_btn_row.setSpacing(6)

        self.btn_enhance_prompt = QPushButton("✨ 智能改写")
        self.btn_enhance_prompt.setToolTip(
            "把自然语言描述自动转换为 AI 画图标准提示词\n"
            "首次使用会下载约 1.5GB 模型")
        self.btn_enhance_prompt.setStyleSheet("""
            QPushButton {
                background:#0a0a0a; color:#dadbdf;
                border:1px solid #212327; border-radius:9999px;
                padding:6px 14px;
            }
            QPushButton:hover { border-color:#7c3aed; }
            QPushButton:disabled { color:#363a3f; border-color:#212327; }
        """)
        prompt_btn_row.addWidget(self.btn_enhance_prompt)

        self.btn_vision_prompt = QPushButton("📷 识图生成")
        self.btn_vision_prompt.setToolTip(
            "上传一张图片 + 输入需求, AI 自动整合生成 SD 提示词")
        self.btn_vision_prompt.setStyleSheet("""
            QPushButton {
                background:#0a0a0a; color:#dadbdf;
                border:1px solid #212327; border-radius:9999px;
                padding:6px 12px;
            }
            QPushButton:hover { border-color:#ff7a17; }
        """)
        prompt_btn_row.addWidget(self.btn_vision_prompt)

        self.chk_auto_enhance = QCheckBox("生成前自动改写")
        self.chk_auto_enhance.setToolTip(
            "勾选后, 每次生成前都会调用 AI 智能改写提示词")
        prompt_btn_row.addWidget(self.chk_auto_enhance)
        prompt_btn_row.addStretch()
        gp.addLayout(prompt_btn_row)

        # ─── 翻译模式选择 ───
        row_trans = QHBoxLayout()
        row_trans.setSpacing(6)

        lbl_trans = QLabel("🌐 翻译模式:")
        lbl_trans.setStyleSheet("color:#dadbdf; font-weight:bold;")
        row_trans.addWidget(lbl_trans)

        self.combo_trans_mode = QComboBox()
        self.combo_trans_mode.addItems([
            " 纯词典",
            "AI 智能改写",
            " 词典优先 + AI 兜底 ",
        ])
        self.combo_trans_mode.setCurrentIndex(2)  
        self.combo_trans_mode.setToolTip(
            " 纯词典: 只用 JSON 词典查词, 速度快但不通顺\n"
            " AI 改写: 每次都调 Qwen, 效果最好但慢\n"
            "混合模式: 词典命中直接用, 未命中才调 AI "
        )
        row_trans.addWidget(self.combo_trans_mode, 1)
        gp.addLayout(row_trans)

        layout.addWidget(grp_prompt)


        grp_preview = QGroupBox("🎨 实时预览")
        gp = QFormLayout(grp_preview)

        self.chk_use_preview = QCheckBox("生成时显示实时预览")
        self.chk_use_preview.setChecked(False)  
        self.chk_use_preview.setToolTip(
            "每 N 步解码一次 latent 显示到画布\n"
            "GPU: 开启基本无影响\n"
            "CPU: 每次预览额外耗时 1-3 分钟, 慎用!\n"
            "用途: 看到生成过程, 早发现废图早中断"
        )
        gp.addRow(self.chk_use_preview)

        self.spin_preview_interval = QSpinBox()
        self.spin_preview_interval.setRange(1, 30)
        self.spin_preview_interval.setValue(10)
        self.spin_preview_interval.setSuffix(" 步")
        self.spin_preview_interval.setToolTip(
            "每 N 步刷新一次预览\n"
            "CPU 推荐 10-15\n"
            "GPU 推荐 3-5"
        )
        gp.addRow("预览间隔:", self.spin_preview_interval)

        layout.addWidget(grp_preview)

        # ============== 3. 基础参数 ==============
        grp_params = QGroupBox("基础参数")
        gpa = QFormLayout(grp_params)
        gpa.setSpacing(8)

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 150)
        self.spin_steps.setValue(30)
        gpa.addRow("步数 Steps:", self.spin_steps)

        self.scale_cfg = FloatSlider(1.0, 20.0, 0.5, 7.0)
        gpa.addRow("CFG Scale:", self.scale_cfg)

        self.combo_res = QComboBox()
        self.combo_res.addItems([
            "512x512", "512x768", "768x512", "768x768",
            "1024x1024", "832x1216", "1216x832"
        ])
        self.combo_res.setCurrentText("512x768")
        gpa.addRow("分辨率:", self.combo_res)

        # 兼容隐藏字段
        self.spin_width = QSpinBox()
        self.spin_width.setRange(256, 2048)
        self.spin_width.setSingleStep(64)
        self.spin_width.setValue(512)
        self.spin_height = QSpinBox()
        self.spin_height.setRange(256, 2048)
        self.spin_height.setSingleStep(64)
        self.spin_height.setValue(768)

        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 32)
        self.spin_count.setValue(1)
        gpa.addRow("生成数量:", self.spin_count)
        self.spin_batch = self.spin_count

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

        self.chk_make_comic = QCheckBox("生成完后拼合分镜连环画")
        gpa.addRow(self.chk_make_comic)

        layout.addWidget(grp_params)
        layout.addStretch()

        return w

    # ============================================================
    #  Tab 2: 动画
    # ============================================================
    def _build_tab_animation(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        root = QVBoxLayout(w)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        _label_style = "color:#ffffff; font-weight:bold; font-size:13px;"
        _hint_style = "color:#7d8187; font-size:12px;"
        _input_style = "font-size:13px;"

        # ============ 💡 使用提示 ============
        grp_tips = QGroupBox("💡 使用提示")
        tips_lay = QVBoxLayout(grp_tips)
        tips_text = QLabel(
            "<ul style='margin:0; padding-left:20px;'>"
            "<li><b>文生视频:</b> 直接用提示词生成视频，无需输入文件</li>"
            "<li><b>图生视频:</b> 选择一张图片作为首帧，AI 延续动画</li>"
            "<li><b>视频转绘:</b> 选择视频文件，AI 改变画风</li>"
            "<li><b>提示词旅行:</b> 在不同帧使用不同提示词，制作剧情视频</li>"
            "</ul>"
        )
        tips_text.setStyleSheet("color:#dadbdf; font-size:12px;")
        tips_text.setWordWrap(True)
        tips_lay.addWidget(tips_text)
        root.addWidget(grp_tips)

        # ============ 🎯 生成模式 ============
        grp_mode = QGroupBox("🎯 生成模式")
        grp_mode.setStyleSheet("QGroupBox { color:#ff7a17; font-weight:bold; font-size:14px; }")
        mode_lay = QVBoxLayout(grp_mode)

        self.combo_video_mode = QComboBox()
        self.combo_video_mode.addItems([
            "📝 文生视频 (txt2video)",
            "🖼️ 图生视频 (img2video) - 首帧引导",
            "🎞️ 视频转绘 (vid2vid) - 改画风",
            "✨ 提示词旅行 (Prompt Travel) - 剧情视频",
        ])
        self.combo_video_mode.setStyleSheet(_input_style)
        self.combo_video_mode.setMinimumHeight(36)
        mode_lay.addWidget(self.combo_video_mode)

        mode_desc = QLabel("选择生成模式后，按提示准备输入内容")
        mode_desc.setStyleSheet(_hint_style)
        mode_lay.addWidget(mode_desc)

        root.addWidget(grp_mode)

        # ============ 📥 输入文件 ============
        grp_input = QGroupBox("📥 输入文件")
        grp_input.setStyleSheet("QGroupBox { color:#dadbdf; font-weight:bold; font-size:14px; }")

        input_vbox = QVBoxLayout(grp_input)
        input_vbox.setSpacing(6)

        input_lay = QHBoxLayout()
        self.lbl_video_input = QLabel("未选择文件")
        self.lbl_video_input.setStyleSheet("color:#7d8187; font-size:13px;")
        self.lbl_video_input.setMinimumHeight(36)
        input_lay.addWidget(self.lbl_video_input, 1)

        btn_pick = QPushButton("📂 选择")
        btn_pick.setStyleSheet(
            "background:#0a0a0a; color:#dadbdf; padding:8px 16px; "
            "border:1px solid #212327; border-radius:9999px; font-size:13px;"
        )
        btn_pick.clicked.connect(self.on_pick_video_input)
        input_lay.addWidget(btn_pick)

        input_vbox.addLayout(input_lay)

        input_hint = QLabel("图生视频/视频转绘模式下需要选择首帧图或输入视频")
        input_hint.setStyleSheet(_hint_style)
        input_hint.setWordWrap(True)
        input_vbox.addWidget(input_hint)

        root.addWidget(grp_input)

        # ============ 💬 提示词 ============
        grp_prompt = QGroupBox("💬 提示词")
        grp_prompt.setStyleSheet("QGroupBox { color:#dadbdf; font-weight:bold; font-size:14px; }")
        prompt_lay = QVBoxLayout(grp_prompt)

        lbl_pos = QLabel("正面提示词")
        lbl_pos.setStyleSheet(_label_style)
        prompt_lay.addWidget(lbl_pos)

        self.txt_video_prompt = QTextEdit()
        self.txt_video_prompt.setFixedHeight(80)
        self.txt_video_prompt.setPlaceholderText("输入生成视频的描述，例如：一只可爱的小猫在草地上奔跑")
        self.txt_video_prompt.setStyleSheet(_input_style)
        prompt_lay.addWidget(self.txt_video_prompt)

        lbl_neg = QLabel("负面提示词")
        lbl_neg.setStyleSheet(_label_style)
        prompt_lay.addWidget(lbl_neg)

        self.txt_video_neg = QTextEdit()
        self.txt_video_neg.setFixedHeight(60)
        self.txt_video_neg.setPlaceholderText("输入要避免的内容，例如：模糊、低质量、文字")
        self.txt_video_neg.setStyleSheet(_input_style)
        prompt_lay.addWidget(self.txt_video_neg)

        self.grp_prompt_travel = QGroupBox("✨ 提示词旅行")
        self.grp_prompt_travel.setStyleSheet("QGroupBox { color:#ff7a17; font-weight:bold; font-size:13px; }")
        travel_text_lay = QVBoxLayout(self.grp_prompt_travel)

        travel_hint = QLabel("格式: 帧号|提示词（每行一个关键帧）")
        travel_hint.setStyleSheet(_hint_style)
        travel_text_lay.addWidget(travel_hint)

        self.txt_prompt_travel = QTextEdit()
        self.txt_prompt_travel.setFixedHeight(100)
        self.txt_prompt_travel.setPlaceholderText(
            "示例:\n0|1girl, smiling, sunny day\n8|1girl, surprised, wind blowing\n16|1girl, crying, rain falling"
        )
        self.txt_prompt_travel.setStyleSheet(_input_style)
        travel_text_lay.addWidget(self.txt_prompt_travel)

        prompt_lay.addWidget(self.grp_prompt_travel)
        self.grp_prompt_travel.setVisible(False)
        root.addWidget(grp_prompt)

        # ============ 🎞️ 视频参数 ============
        grp_video = QGroupBox("🎞️ 视频参数")
        grp_video.setStyleSheet("QGroupBox { color:#dadbdf; font-weight:bold; font-size:14px; }")
        video_lay = QVBoxLayout(grp_video)

        # === 帧数与时长 ===
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl_frames = QLabel("帧数:")
        lbl_frames.setStyleSheet(_label_style)
        lbl_frames.setFixedWidth(60)
        row.addWidget(lbl_frames)

        self.spin_video_frames = QSpinBox()
        self.spin_video_frames.setRange(8, 80)
        self.spin_video_frames.setValue(16)
        self.spin_video_frames.setFixedWidth(80)
        self.spin_video_frames.setStyleSheet(_input_style)
        row.addWidget(self.spin_video_frames)

        lbl_duration = QLabel("快捷时长:")
        lbl_duration.setStyleSheet(_label_style)
        lbl_duration.setFixedWidth(80)
        row.addWidget(lbl_duration)

        for sec in [2, 4, 6, 8, 10]:
            btn = QPushButton(f"{sec}秒")
            btn.setFixedWidth(50)
            btn.setStyleSheet(
                "background:#0a0a0a; color:#dadbdf; border:1px solid #212327; "
                "border-radius:9999px; font-size:12px; padding:4px;"
            )
            btn.clicked.connect(lambda checked, s=sec: self._set_video_duration(s))
            row.addWidget(btn)

        row.addStretch()
        video_lay.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(8)

        lbl_fps = QLabel("FPS:")
        lbl_fps.setStyleSheet(_label_style)
        lbl_fps.setFixedWidth(60)
        row.addWidget(lbl_fps)

        self.spin_video_fps = QSpinBox()
        self.spin_video_fps.setRange(4, 30)
        self.spin_video_fps.setValue(8)
        self.spin_video_fps.setFixedWidth(80)
        self.spin_video_fps.setStyleSheet(_input_style)
        row.addWidget(self.spin_video_fps)

        fps_hint = QLabel("建议 8-12 FPS（更高更流畅但更慢）")
        fps_hint.setStyleSheet(_hint_style)
        row.addWidget(fps_hint)

        row.addStretch()
        video_lay.addLayout(row)

        row = QHBoxLayout()
        self.chk_long_video = QCheckBox("🎬 长视频模式 (>32 帧)")
        self.chk_long_video.setStyleSheet("color:#dadbdf; font-size:13px;")
        self.chk_long_video.setToolTip("启用 Context Window,支持 64+ 帧")
        row.addWidget(self.chk_long_video)
        row.addStretch()
        video_lay.addLayout(row)

        # === 提示词旅行分段 ===
        grp_travel = QGroupBox("🎞️ 提示词旅行分段")
        grp_travel.setStyleSheet("QGroupBox { color:#dadbdf; font-weight:bold; font-size:13px; }")
        grp_travel.setCheckable(True)
        grp_travel.setChecked(False)
        travel_lay = QVBoxLayout(grp_travel)

        travel_hint2 = QLabel("启用后可在不同帧使用不同提示词，制作剧情变化")
        travel_hint2.setStyleSheet(_hint_style)
        travel_lay.addWidget(travel_hint2)

        self.travel_container = QVBoxLayout()
        self.travel_container.setSpacing(6)
        travel_lay.addLayout(self.travel_container)

        btn_add_segment = QPushButton("➕ 添加段")
        btn_add_segment.setStyleSheet(
            "background:#0a0a0a; color:#dadbdf; border:1px solid #212327; "
            "border-radius:9999px; font-size:12px; padding:6px;"
        )
        btn_add_segment.clicked.connect(self._add_travel_segment)
        travel_lay.addWidget(btn_add_segment)

        video_lay.addWidget(grp_travel)
        self.grp_travel = grp_travel

        self.travel_segments = []
        self._add_travel_segment()
        self._add_travel_segment()

        # === 其他参数 ===
        row = QHBoxLayout()
        row.setSpacing(16)

        lbl_steps = QLabel("步数:")
        lbl_steps.setStyleSheet(_label_style)
        lbl_steps.setFixedWidth(60)
        row.addWidget(lbl_steps)

        self.spin_video_steps = QSpinBox()
        self.spin_video_steps.setRange(10, 100)
        self.spin_video_steps.setValue(25)
        self.spin_video_steps.setFixedWidth(80)
        self.spin_video_steps.setStyleSheet(_input_style)
        row.addWidget(self.spin_video_steps)

        lbl_cfg = QLabel("CFG:")
        lbl_cfg.setStyleSheet(_label_style)
        lbl_cfg.setFixedWidth(60)
        row.addWidget(lbl_cfg)

        self.spin_video_cfg = QDoubleSpinBox()
        self.spin_video_cfg.setRange(1.0, 20.0)
        self.spin_video_cfg.setValue(7.5)
        self.spin_video_cfg.setSingleStep(0.5)
        self.spin_video_cfg.setFixedWidth(100)
        self.spin_video_cfg.setStyleSheet(_input_style)
        row.addWidget(self.spin_video_cfg)

        row.addStretch()
        video_lay.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(16)

        lbl_w = QLabel("宽度:")
        lbl_w.setStyleSheet(_label_style)
        lbl_w.setFixedWidth(60)
        row.addWidget(lbl_w)

        self.spin_video_w = QSpinBox()
        self.spin_video_w.setRange(256, 1024)
        self.spin_video_w.setValue(512)
        self.spin_video_w.setSingleStep(64)
        self.spin_video_w.setFixedWidth(80)
        self.spin_video_w.setStyleSheet(_input_style)
        row.addWidget(self.spin_video_w)

        lbl_h = QLabel("高度:")
        lbl_h.setStyleSheet(_label_style)
        lbl_h.setFixedWidth(60)
        row.addWidget(lbl_h)

        self.spin_video_h = QSpinBox()
        self.spin_video_h.setRange(256, 1024)
        self.spin_video_h.setValue(512)
        self.spin_video_h.setSingleStep(64)
        self.spin_video_h.setFixedWidth(80)
        self.spin_video_h.setStyleSheet(_input_style)
        row.addWidget(self.spin_video_h)

        row.addStretch()
        video_lay.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(8)

        lbl_sampler = QLabel("采样器:")
        lbl_sampler.setStyleSheet(_label_style)
        lbl_sampler.setFixedWidth(80)
        row.addWidget(lbl_sampler)

        self.combo_video_sched = QComboBox()
        self.combo_video_sched.addItems([
            "EulerDiscrete (推荐)",
            "DPM++ 2M",
            "LCM (快速)",
            "DDIM",
        ])
        self.combo_video_sched.setStyleSheet(_input_style)
        row.addWidget(self.combo_video_sched)

        row.addStretch()
        video_lay.addLayout(row)

        root.addWidget(grp_video)

        # ---------- Motion LoRA (多选) ----------
        lora_group = QGroupBox("🎭 Motion LoRA (可多选)")
        lora_layout = QVBoxLayout(lora_group)
        lora_layout.setContentsMargins(8, 12, 8, 8)
        lora_layout.setSpacing(6)

        # 添加按钮行
        add_row = QHBoxLayout()
        self.cmb_motion_lora_pick = QComboBox()
        self.cmb_motion_lora_pick.addItem("-- 选择 Motion LoRA --")
        # 填充列表
        try:
            from utils.video_gen import VideoGenerator
            lora_dir = "models/motion_lora"
            if os.path.isdir(lora_dir):
                for d in sorted(os.listdir(lora_dir)):
                    if os.path.isdir(os.path.join(lora_dir, d)):
                        self.cmb_motion_lora_pick.addItem(d)
        except Exception:
            pass

        btn_add_lora = QPushButton("➕ 添加")
        btn_add_lora.setFixedWidth(70)
        btn_add_lora.clicked.connect(self._add_motion_lora_item)

        add_row.addWidget(self.cmb_motion_lora_pick, 1)
        add_row.addWidget(btn_add_lora)
        lora_layout.addLayout(add_row)

        # 已选 LoRA 容器
        self.motion_lora_container = QVBoxLayout()
        self.motion_lora_container.setSpacing(4)
        lora_layout.addLayout(self.motion_lora_container)

        # 记录已选项 [{'name':str, 'widget':QWidget, 'slider':QSlider, 'label':QLabel}]
        self.motion_lora_items = []

        root.addWidget(lora_group)

        # ============ ✨ 后处理 ============
        grp_post = QGroupBox("✨ 后处理 (可选)")
        grp_post.setStyleSheet("QGroupBox { color:#dadbdf; font-weight:bold; font-size:14px; }")
        post_lay = QVBoxLayout(grp_post)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.chk_frame_interp = QCheckBox("帧插值 (RIFE)")
        self.chk_frame_interp.setStyleSheet("color:#dadbdf; font-size:13px;")
        row.addWidget(self.chk_frame_interp)

        self.combo_frame_interp = QComboBox()
        self.combo_frame_interp.addItems(["2x", "4x", "8x"])
        self.combo_frame_interp.setCurrentText("2x")
        self.combo_frame_interp.setEnabled(False)
        self.combo_frame_interp.setStyleSheet(_input_style)
        row.addWidget(self.combo_frame_interp)

        interp_hint = QLabel("增加帧数使视频更流畅")
        interp_hint.setStyleSheet(_hint_style)
        row.addWidget(interp_hint)

        row.addStretch()
        post_lay.addLayout(row)
        self.chk_frame_interp.toggled.connect(self.combo_frame_interp.setEnabled)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.chk_video_upscale = QCheckBox("🔍 视频放大 (Real-ESRGAN)")
        self.chk_video_upscale.setStyleSheet("color:#dadbdf; font-size:13px;")
        self.chk_video_upscale.setToolTip("512 → 1024/2048")
        row.addWidget(self.chk_video_upscale)

        self.combo_upscale_factor = QComboBox()
        self.combo_upscale_factor.addItems(["2x", "4x"])
        self.combo_upscale_factor.setStyleSheet(_input_style)
        row.addWidget(self.combo_upscale_factor)

        upscale_hint = QLabel("提高视频分辨率")
        upscale_hint.setStyleSheet(_hint_style)
        row.addWidget(upscale_hint)

        row.addStretch()
        post_lay.addLayout(row)

        root.addWidget(grp_post)

        # ============ 💾 输出格式 ============
        grp_out = QGroupBox("💾 输出设置")
        grp_out.setStyleSheet("QGroupBox { color:#dadbdf; font-weight:bold; font-size:14px; }")
        out_lay = QVBoxLayout(grp_out)

        row = QHBoxLayout()
        row.setSpacing(8)

        lbl_format = QLabel("格式:")
        lbl_format.setStyleSheet(_label_style)
        lbl_format.setFixedWidth(60)
        row.addWidget(lbl_format)

        self.combo_video_fmt = QComboBox()
        self.combo_video_fmt.addItems(["MP4", "GIF", "MP4 + GIF"])
        self.combo_video_fmt.setStyleSheet(_input_style)
        row.addWidget(self.combo_video_fmt)

        row.addStretch()
        out_lay.addLayout(row)

        format_hint = QLabel("MP4 适合分享，GIF 适合社交媒体")
        format_hint.setStyleSheet(_hint_style)
        out_lay.addWidget(format_hint)

        root.addWidget(grp_out)

        # ============ 🎬 生成按钮 + 状态 ============
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self.btn_gen_video = QPushButton("🎬 生成视频")
        self.btn_gen_video.setStyleSheet(
            "background:#ff7a17;color:#0a0a0a;font-weight:bold;padding:14px 32px;font-size:15px;border-radius:9999px;"
        )
        self.btn_gen_video.setFixedHeight(48)
        self.btn_gen_video.clicked.connect(self.on_generate_video)
        btn_row.addWidget(self.btn_gen_video, 1)

        root.addLayout(btn_row)

        self.lbl_video_status = QLabel("💤 待命中 - 请设置参数后点击生成")
        self.lbl_video_status.setStyleSheet("color:#7d8187; font-size:13px; padding:6px;")
        root.addWidget(self.lbl_video_status)

        scroll.setWidget(w)
        return scroll

    # ============================================================
    #  Tab 3: 图生图(参考图 + IP-Adapter + Pose Transfer)
    # ============================================================
    def _build_tab_img2img(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        w.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # ---------- 参考图 ----------
        grp_i2i = QGroupBox("参考图 (img2img / inpaint)")
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
        self.lbl_img_path.setStyleSheet("color:#7d8187; font-size:11px;")
        gi.addWidget(self.lbl_img_path)

        self.lbl_ref_thumb = QLabel("无参考图")
        self.lbl_ref_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ref_thumb.setFixedHeight(120)
        self.lbl_ref_thumb.setStyleSheet(
            "border:1px dashed #212327; border-radius:8px; color:#7d8187;")
        gi.addWidget(self.lbl_ref_thumb)

        gi.addWidget(QLabel("重绘强度 (Denoise):"))
        self.scale_strength = FloatSlider(0.05, 1.0, 0.05, 0.6)
        gi.addWidget(self.scale_strength)

        layout.addWidget(grp_i2i)

        # ---------- IP-Adapter ----------
        grp_ipa = QGroupBox("🎭 IP-Adapter — 角色一致性")
        grp_ipa.setStyleSheet(
            "QGroupBox::title { color:#dadbdf; }")
        g_ipa = QGridLayout(grp_ipa)

        self.chk_use_ipa = QCheckBox("启用 IP-Adapter (锁定角色样貌)")
        g_ipa.addWidget(self.chk_use_ipa, 0, 0, 1, 4)

        btn_load_ipa = QPushButton("📷 加载角色参考图")
        btn_load_ipa.clicked.connect(self.load_ipa_image)
        g_ipa.addWidget(btn_load_ipa, 1, 0)

        self.lbl_ipa_image = QLabel("未选择")
        self.lbl_ipa_image.setStyleSheet("color:#7d8187; padding:4px;")
        g_ipa.addWidget(self.lbl_ipa_image, 1, 1, 1, 3)

        g_ipa.addWidget(QLabel("影响力:"), 2, 0)
        self.spin_ipa_scale = QDoubleSpinBox()
        self.spin_ipa_scale.setRange(0.0, 1.5)
        self.spin_ipa_scale.setSingleStep(0.05)
        self.spin_ipa_scale.setValue(0.6)
        self.spin_ipa_scale.setDecimals(2)
        g_ipa.addWidget(self.spin_ipa_scale, 2, 1)

        g_ipa.addWidget(QLabel("版本:"), 2, 2)
        self.combo_ipa_variant = QComboBox()
        self.combo_ipa_variant.addItems(["plus (推荐)", "standard (轻量)"])
        g_ipa.addWidget(self.combo_ipa_variant, 2, 3)

        layout.addWidget(grp_ipa)

        # ---------- 🎬 Pose Transfer ----------
        grp_pt = QGroupBox("🎬 Pose Transfer — 智能姿势迁移 (推荐)")
        grp_pt.setStyleSheet(
            "QGroupBox::title { color:#dadbdf; }")
        g_pt = QVBoxLayout(grp_pt)

        self.chk_pose_transfer = QCheckBox("启用 Pose Transfer (3 阶段流水线)")
        self.chk_pose_transfer.setToolTip(
            "🎬 自动 3 阶段流水线:\n"
            "1️⃣ 用提示词生成动作参考图\n"
            "2️⃣ 自动提取 OpenPose 骨架\n"
            "3️⃣ 骨架(锁动作) + 角色图(锁角色) → 最终图\n\n"
            "✅ 完美解决「图生图看不懂提示词」问题\n"
            "⚠️ 需要在上方上传 IP-Adapter 角色参考图\n"
            "⏱ 总耗时约普通生成的 1.5~2 倍"
        )
        self.chk_pose_transfer.toggled.connect(self._on_pose_transfer_toggled)
        g_pt.addWidget(self.chk_pose_transfer)

        # 提示行
        self.lbl_pt_tip = QLabel(
            "💡 启用后会自动:\n"
            "   • 强制开启 IP-Adapter (用上方角色图锁人物)\n"
            "   • 强制使用 OpenPose ControlNet (锁动作)\n"
            "   • 忽略「重绘强度」(走 ControlNet 通道)"
        )
        self.lbl_pt_tip.setStyleSheet(
            "color:#7d8187; padding:6px; background:#191919;"
            "border-radius:8px; font-size:11px;")
        self.lbl_pt_tip.setWordWrap(True)
        g_pt.addWidget(self.lbl_pt_tip)

        row_cn = QHBoxLayout()
        row_cn.addWidget(QLabel("姿势约束强度:"))
        self.slider_pt_cn = QSlider(Qt.Orientation.Horizontal)
        self.slider_pt_cn.setRange(30, 120)   # 0.30 ~ 1.20
        self.slider_pt_cn.setValue(65)        # 默认 0.65
        self.slider_pt_cn.setFixedWidth(220)
        self.lbl_pt_cn = QLabel("0.65")
        self.lbl_pt_cn.setFixedWidth(50)
        self.slider_pt_cn.valueChanged.connect(
            lambda v: self.lbl_pt_cn.setText(f"{v/100:.2f}"))
        row_cn.addWidget(self.slider_pt_cn)
        row_cn.addWidget(self.lbl_pt_cn)
        row_cn.addStretch()

        # 子提示
        hint_cn = QLabel("(越低 = 越像角色; 越高 = 越像动作)")
        hint_cn.setStyleSheet("color: #7d8187; font-size: 11px; padding-left: 20px;")

        g_pt.addLayout(row_cn)
        g_pt.addWidget(hint_cn)

        # 启用/禁用联动
        def _toggle_pt(checked):
            self.slider_pt_cn.setEnabled(checked)
            self.lbl_pt_cn.setEnabled(checked)
        self.chk_pose_transfer.toggled.connect(_toggle_pt)
        _toggle_pt(False)  # 初始禁用

        

        g_consist = QGroupBox("🎯 单图角色一致性增强")
        g_consist.setStyleSheet(
            "QGroupBox::title { color:#dadbdf; }")
        v_consist = QVBoxLayout(g_consist)

        self.chk_auto_features = QCheckBox(
            " 自动提取角色特征 (Qwen 识别发色/瞳色/兽耳并注入 prompt)")
        self.chk_auto_features.setChecked(True)
        self.chk_auto_features.setToolTip(
            "启用后,生成前会用 Qwen2-VL 分析参考图,\n"
            "自动提取发色/瞳色/兽耳/服装等关键特征,\n"
            "并以最高权重注入 prompt 最前端。\n"
            "✅ 单图角色一致性必备"
        )
        v_consist.addWidget(self.chk_auto_features)

        self.chk_reference_only = QCheckBox(
            "🪞 启用 Reference-Only (锁定参考图细节,与 Pose 互斥)")
        self.chk_reference_only.setChecked(False)
        v_consist.addWidget(self.chk_reference_only)

        row_ref = QHBoxLayout()
        row_ref.addWidget(QLabel("参考强度:"))
        self.scale_ref_fidelity = QSlider(Qt.Orientation.Horizontal)
        self.scale_ref_fidelity.setRange(50, 100)   # 0.50 ~ 1.00
        self.scale_ref_fidelity.setValue(70)
        self.scale_ref_fidelity.setFixedWidth(220)
        self.lbl_ref_fidelity = QLabel("0.70")
        self.lbl_ref_fidelity.setFixedWidth(50)
        self.scale_ref_fidelity.valueChanged.connect(
            lambda v: self.lbl_ref_fidelity.setText(f"{v/100:.2f}"))
        row_ref.addWidget(self.scale_ref_fidelity)
        row_ref.addWidget(self.lbl_ref_fidelity)
        row_ref.addStretch()
        v_consist.addLayout(row_ref)

        hint_ref = QLabel("(0.50=自由发挥, 0.70=平衡推荐, 1.00=完全复刻)")
        hint_ref.setStyleSheet("color:#7d8187; font-size:11px; padding-left:10px;")
        v_consist.addWidget(hint_ref)

        layout.addWidget(grp_pt)
        layout.addWidget(g_consist)        
        layout.addStretch()
        return w


    # ============================================================
    #  Tab 4: LoRA
    # ============================================================
    def _build_tab_lora(self) -> QWidget:
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

        layout.addWidget(QLabel("LoRA 备忘录:"))
        self.text_lora_info = QTextEdit()
        self.text_lora_info.setReadOnly(True)
        self.text_lora_info.setFixedHeight(120)
        self.text_lora_info.setStyleSheet(
            "font-family:Consolas; font-size:11px; background:#191919;")
        layout.addWidget(self.text_lora_info)
        # ============ 触发词插入按钮行 ============
        btn_row = QHBoxLayout()

        self.btn_insert_lora_all = QPushButton("📋 插入全部触发词")
        self.btn_insert_lora_all.setStyleSheet(
            "background:#0a0a0a; color:#dadbdf; border:1px solid #212327; "
            "border-radius:9999px; padding:4px 10px;"
        )
        self.btn_insert_lora_all.clicked.connect(lambda: self._insert_lora_triggers(None))
        btn_row.addWidget(self.btn_insert_lora_all)

        for i in range(3):
            btn = QPushButton(f"槽{i+1}")
            btn.setFixedWidth(45)
            btn.setStyleSheet(
                "background:#0a0a0a; color:#dadbdf; border:1px solid #212327; "
                "border-radius:9999px; padding:4px;"
            )
            btn.clicked.connect(lambda _, idx=i: self._insert_lora_triggers(idx))
            btn_row.addWidget(btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addStretch()
        return w

    # ============================================================
    #  Tab 5: ControlNet
    # ============================================================
    def _build_tab_ctrl(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        w.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        grp = QGroupBox("ControlNet — 手动模式")
        gv = QFormLayout(grp)

        self.chk_use_pose = QCheckBox("开启 ControlNet")
        self.chk_use_pose.toggled.connect(self._toggle_cn)
        gv.addRow(self.chk_use_pose)

        self.combo_cn_type = QComboBox()
        self.combo_cn_type.addItems([
            "OpenPose", "Canny", "Depth", "Scribble", "SoftEdge"
        ])
        gv.addRow("类型:", self.combo_cn_type)

        self.scale_cn_strength = FloatSlider(0.0, 2.0, 0.05, 1.0)
        gv.addRow("条件强度:", self.scale_cn_strength)

        # 兼容老名 scale_cn_weight
        self.scale_cn_weight = self.scale_cn_strength

        self.btn_load_cn_img = QPushButton("📂 加载姿态图")
        self.btn_load_cn_img.clicked.connect(self.load_pose_image)
        gv.addRow(self.btn_load_cn_img)

        self.lbl_pose_path = QLabel("未加载动作图")
        self.lbl_pose_path.setStyleSheet("color:#7d8187; font-size:11px;")
        gv.addRow(self.lbl_pose_path)

        self.lbl_cn_thumb = GpuCanvas()
        self.lbl_cn_thumb.setText("未加载")
        self.lbl_cn_thumb.setFixedHeight(180)
        self.lbl_cn_thumb.setStyleSheet(
            "border:1px dashed #212327; border-radius:8px; color:#7d8187;")
        gv.addRow(self.lbl_cn_thumb)
        layout.addWidget(grp)

        # 提示
        tip = QLabel(
            "💡 提示: 如果想用「提示词→自动生成动作」,\n"
            "   请到 [图生图] Tab 启用 🎬 Pose Transfer。"
        )
        tip.setStyleSheet(
            "color:#7d8187; padding:8px; background:#191919;"
            "border-radius:8px; font-size:11px;")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        layout.addStretch()
        return w

    # ============================================================
    #  Tab 6: 高级 (ADetailer + Hires.fix + 输出)
    # ============================================================
    def _build_tab_advanced(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        w.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # ---------- 修脸 ----------
        grp_face = QGroupBox("ADetailer — 修脸")
        gf = QFormLayout(grp_face)
        self.chk_use_adetailer = QCheckBox("开启修脸")
        self.chk_use_adetailer.toggled.connect(self._toggle_adetailer)
        gf.addRow(self.chk_use_adetailer)

        self.combo_adetailer_model = QComboBox()
        self.combo_adetailer_model.addItems(["真人脸", "二次元脸"])
        gf.addRow("脸部类型:", self.combo_adetailer_model)

        self.combo_ad_target = QComboBox()
        self.combo_ad_target.addItems(["现实脸部", "二次元脸部"])
        gf.addRow("检测目标:", self.combo_ad_target)

        self.lbl_ad_str = QLabel("修复强度:")
        self.lbl_ad_str.setStyleSheet("color:#7d8187; font-family:Consolas;")
        self.scale_adetailer_strength = FloatSlider(0.1, 0.9, 0.05, 0.35)
        gf.addRow(self.lbl_ad_str, self.scale_adetailer_strength)
        layout.addWidget(grp_face)

        # ---------- 修手 ----------
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
            "color:#7d8187; font-family:Consolas;")
        self.scale_ad_hand = FloatSlider(0.1, 0.6, 0.05, 0.25)
        gh.addRow(self.lbl_ad_hand_str, self.scale_ad_hand)

        self.lbl_ad_hand_blend = QLabel("融合度:")
        self.lbl_ad_hand_blend.setStyleSheet(
            "color:#7d8187; font-family:Consolas;")
        self.scale_ad_hand_blend = FloatSlider(0.0, 1.0, 0.05, 0.65)
        gh.addRow(self.lbl_ad_hand_blend, self.scale_ad_hand_blend)
        layout.addWidget(grp_hand)

        # ---------- Hires.fix ----------
        grp_hr = QGroupBox("Hires.fix — 高清修复")
        ghr = QFormLayout(grp_hr)

        self.chk_hires = QCheckBox("开启 Hires.fix")
        self.chk_hires.toggled.connect(self._toggle_hires)
        ghr.addRow(self.chk_hires)

        self.chk_enable_hires = QCheckBox("XY 矩阵中也启用 Hires.fix")
        ghr.addRow(self.chk_enable_hires)

        self.combo_hires_scale = QComboBox()
        self.combo_hires_scale.addItems(["1.5", "2.0", "2.5", "3.0"])
        self.combo_hires_scale.setCurrentText("2.0")
        ghr.addRow("放大倍率:", self.combo_hires_scale)

        self.scale_hires_denoise = FloatSlider(0.1, 0.9, 0.05, 0.45)
        ghr.addRow("降噪强度:", self.scale_hires_denoise)

        self.combo_hires_upscaler = QComboBox()
        self.combo_hires_upscaler.addItems([
            "Latent", "ESRGAN_4x", "R-ESRGAN 4x+", "SwinIR"
        ])
        ghr.addRow("Upscaler:", self.combo_hires_upscaler)
        layout.addWidget(grp_hr)

       # ---------- 大图生成 ----------

        grp_photo = QGroupBox("🖼️ 大图生成 (Tiled Diffusion)")
        fl_tiled = QFormLayout(grp_photo)
    
        # 总开关
        self.chk_use_tiled = QCheckBox("启用大图生成(对当前图后处理)")
        self.chk_use_tiled.setToolTip(
            "Tiled Diffusion: 将大图分块生成后融合\n"
            "突破显存限制，可出 2048-4096 分辨率\n"
            "⚠️ CPU 用户慎用：一张 2K 图约需 2-4 小时"
        )
        fl_tiled.addRow(self.chk_use_tiled)
    
        # 目标尺寸
        size_row = QHBoxLayout()
        self.spin_tiled_w = QSpinBox()
        self.spin_tiled_w.setRange(768, 8192)
        self.spin_tiled_w.setSingleStep(64)
        self.spin_tiled_w.setValue(2048)
        self.spin_tiled_h = QSpinBox()
        self.spin_tiled_h.setRange(768, 8192)
        self.spin_tiled_h.setSingleStep(64)
        self.spin_tiled_h.setValue(2048)
        size_row.addWidget(self.spin_tiled_w)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.spin_tiled_h)
        wrap = QWidget(); wrap.setLayout(size_row)
        fl_tiled.addRow("目标分辨率:", wrap)
    
        # Tile 大小
        self.combo_tile_size = QComboBox()
        self.combo_tile_size.addItems(["512", "640", "768", "1024"])
        self.combo_tile_size.setCurrentText("768")
        self.combo_tile_size.setToolTip("单块大小，越大越慢但接缝越少")
        fl_tiled.addRow("Tile 大小:", self.combo_tile_size)
    
        # 重叠
        self.spin_tile_overlap = QSpinBox()
        self.spin_tile_overlap.setRange(32, 256)
        self.spin_tile_overlap.setSingleStep(16)
        self.spin_tile_overlap.setValue(96)
        self.spin_tile_overlap.setToolTip("重叠像素，消接缝必需，建议 64-128")
        fl_tiled.addRow("Tile 重叠:", self.spin_tile_overlap)
    
        # img2img 强度
        self.scale_tile_strength = FloatSlider(0.2, 0.8, 0.05, 0.4)
        self.scale_tile_strength.setToolTip(
            "0.3-0.4: 仅放大细化(推荐)\n"
            "0.5-0.6: 中度重绘\n"
            "0.7+: 大幅改变原图"
        )
        fl_tiled.addRow("重绘强度:", self.scale_tile_strength)
    
        # 执行按钮(独立触发，不污染主生成流程)
        self.btn_run_tiled = QPushButton("🚀 对最后一张图执行大图生成")
        self.btn_run_tiled.clicked.connect(self.run_tiled_diffusion)
        fl_tiled.addRow(self.btn_run_tiled)
        layout.addWidget(grp_photo)

        # ---------- 输出 ----------
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

    # ============================================================
    #  Tab 7: X/Y 矩阵
    # ============================================================
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

    # ============================================================
    #  右侧面板
    # ============================================================
    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── 上半区: 预览画布 ──
        self.lbl_preview = GpuCanvas()
        self.lbl_preview.setText("等待生成...")
        self.lbl_preview.setStyleSheet(
            "background:#191919; color:#7d8187; "
            "border:1px dashed #212327; border-radius:8px; font-size:14px;"
        )
        self.lbl_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview_canvas = self.lbl_preview

        # ── 上半区: 4 个操作按钮 ──
        preview_wrap = QWidget()
        preview_layout = QVBoxLayout(preview_wrap)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)
        preview_layout.addWidget(self.lbl_preview, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_open_editor = QPushButton("🖌️ 编辑")
        self.btn_save_as = QPushButton("💾 另存为")
        self.btn_send_img2img = QPushButton("🔄 转图生图")
        self.btn_send_inpaint = QPushButton("🎭 转重绘")
        for b in (self.btn_open_editor, self.btn_save_as,
                  self.btn_send_img2img, self.btn_send_inpaint):
            b.setStyleSheet(
                "background:#0a0a0a; color:#dadbdf; padding:6px; "
                "border:1px solid #212327; border-radius:9999px; font-size:11px;"
            )
            btn_row.addWidget(b)
        preview_layout.addLayout(btn_row)
        self.btn_open_editor.clicked.connect(self.open_gallery_to_edit)
        self.btn_save_as.clicked.connect(self.save_current_image_as)         
        self.btn_send_img2img.clicked.connect(self.send_preview_to_img2img) 
        self.btn_send_inpaint.clicked.connect(self.send_preview_to_inpaint)  
        # ── 下半区: 画廊标题 + 画廊 ──
        gallery_wrap = QWidget()
        gallery_layout = QVBoxLayout(gallery_wrap)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        gallery_layout.setSpacing(2)

        lbl_gallery_title = QLabel("🖼️ 历史画廊 (双击大图 · 右键菜单)")
        lbl_gallery_title.setStyleSheet(
            "color:#ffffff; font-weight:bold; padding:2px;"
        )
        gallery_layout.addWidget(lbl_gallery_title)

        self.gallery = GalleryPanel()
        # ⭐ 关键: 取消最大高度限制,让 QSplitter 自由分配
        self.gallery.setMinimumHeight(180)
        self.gallery.image_selected.connect(self._on_gallery_picked)
        # G6: 元数据面板的"套用参数"信号连到 main.py
        if hasattr(self, 'apply_meta_params'):
            self.gallery.apply_params_signal.connect(self.apply_meta_params)
        gallery_layout.addWidget(self.gallery, 1)

        # ── QSplitter: 上下分割,可拖动 ──
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(preview_wrap)
        right_splitter.addWidget(gallery_wrap)
        right_splitter.setSizes([500, 400])       # 初始 5:4
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setChildrenCollapsible(False)  # 不允许折叠隐藏
        right_splitter.setHandleWidth(4)
        right_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #212327;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background: #363a3f;
            }
        """)
        layout.addWidget(right_splitter, 1)

        # 暴露给外部
        self.right_splitter = right_splitter

        # 日志
        lbl_log = QLabel("📋 生成日志:")
        lbl_log.setStyleSheet("color:#7d8187;")
        layout.addWidget(lbl_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(140)
        self.txt_log.setStyleSheet(
            "background:#191919; font-family:Consolas; font-size:11px;")
        layout.addWidget(self.txt_log, 1)
        return w

    def _build_video_right_panel(self) -> QWidget:
        """视频模式专用右侧面板（视频预览 + 视频画廊）"""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── 上半区: 视频预览 ──
        video_preview_wrap = QWidget()
        video_preview_layout = QVBoxLayout(video_preview_wrap)
        video_preview_layout.setContentsMargins(0, 0, 0, 0)
        video_preview_layout.setSpacing(4)

        self.video_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.video_player.setAudioOutput(self.audio_output)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)
        self.video_widget.setStyleSheet("background:#0a0a0a;")
        self.video_player.setVideoOutput(self.video_widget)

        self.video_player.mediaStatusChanged.connect(self._on_video_media_changed)
        self.video_player.errorOccurred.connect(self._on_video_player_error)

        self.lbl_video_placeholder = QLabel("🎥 视频生成后自动播放\n或从下方历史列表双击选择")
        self.lbl_video_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video_placeholder.setStyleSheet("color:#7d8187;padding:40px;font-size:14px;")
        self.lbl_video_placeholder.setMinimumHeight(300)
        self.lbl_video_placeholder.setStyleSheet("background:#0a0a0a;color:#7d8187;padding:40px;font-size:14px;border-radius:8px;")

        video_stacked = QStackedWidget()
        video_stacked.setMinimumHeight(300)
        video_stacked.addWidget(self.lbl_video_placeholder)
        video_stacked.addWidget(self.video_widget)
        video_stacked.setCurrentIndex(0)

        self.video_stacked = video_stacked

        video_preview_layout.addWidget(video_stacked)

        video_btn_row = QHBoxLayout()
        video_btn_row.setSpacing(4)
        self.btn_video_save = QPushButton("💾 保存")
        self.btn_video_refresh = QPushButton("🔄 刷新")
        for b in (self.btn_video_save, self.btn_video_refresh):
            b.setStyleSheet(
                "background:#0a0a0a; color:#dadbdf; padding:6px; "
                "border:1px solid #212327; border-radius:9999px; font-size:11px;"
            )
            video_btn_row.addWidget(b)
        self.btn_video_save.clicked.connect(self._save_current_video)
        self.btn_video_refresh.clicked.connect(self._refresh_video_gallery)
        video_preview_layout.addLayout(video_btn_row)

        # ── 下半区: 视频画廊 ──
        video_gallery_wrap = QWidget()
        video_gallery_layout = QVBoxLayout(video_gallery_wrap)
        video_gallery_layout.setContentsMargins(0, 0, 0, 0)
        video_gallery_layout.setSpacing(2)

        lbl_video_gallery_title = QLabel("📂 视频历史 (双击播放)")
        lbl_video_gallery_title.setStyleSheet(
            "color:#ffffff; font-weight:bold; padding:2px;font-size:13px;"
        )
        video_gallery_layout.addWidget(lbl_video_gallery_title)

        self.video_list = QListWidget()
        self.video_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.video_list.setIconSize(QSize(160, 90))
        self.video_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.video_list.setSpacing(10)
        self.video_list.itemDoubleClicked.connect(self._on_video_item_clicked)
        video_gallery_layout.addWidget(self.video_list, 1)

        # ── QSplitter: 上下分割 ──
        video_splitter = QSplitter(Qt.Orientation.Vertical)
        video_splitter.addWidget(video_preview_wrap)
        video_splitter.addWidget(video_gallery_wrap)
        video_splitter.setSizes([500, 300])
        video_splitter.setStretchFactor(0, 1)
        video_splitter.setStretchFactor(1, 1)
        video_splitter.setChildrenCollapsible(False)
        video_splitter.setHandleWidth(4)
        video_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #212327;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background: #363a3f;
            }
        """)
        layout.addWidget(video_splitter, 1)

        # 日志
        lbl_log = QLabel("📋 生成日志:")
        lbl_log.setStyleSheet("color:#7d8187;")
        layout.addWidget(lbl_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(140)
        self.txt_log.setStyleSheet(
            "background:#191919; font-family:Consolas; font-size:11px;")
        layout.addWidget(self.txt_log, 1)

        return w

    def play_video(self, video_path: str):
        """播放指定路径的视频"""
        print(f"🎥 尝试播放视频: {video_path}")
        if not os.path.exists(video_path):
            self._set_status(f"⚠️ 视频文件不存在: {video_path}", "#ff7a17")
            return

        try:
            self.video_player.stop()
            self.video_player.setSource(QUrl.fromLocalFile(video_path))
            
            if hasattr(self, 'video_stacked'):
                self.video_stacked.setCurrentIndex(1)
                print("✅ 切换到视频播放界面")
            else:
                self.lbl_video_placeholder.hide()
                self.video_widget.show()
            
            self.video_player.play()
            self._set_status(f"🎥 正在播放: {os.path.basename(video_path)}", "#dadbdf")
            self.current_video_path = video_path
            print(f"✅ 视频播放开始: {os.path.basename(video_path)}")
        except Exception as e:
            import traceback
            self._set_status(f"⚠️ 视频播放失败: {e}", "#ff7a17")
            print(f"❌ 视频播放失败: {e}")
            print(traceback.format_exc())
            if hasattr(self, 'video_stacked'):
                self.video_stacked.setCurrentIndex(0)
            else:
                self.lbl_video_placeholder.show()
                self.video_widget.hide()

    def _on_video_media_changed(self, status):
        """视频媒体状态变化回调"""
        from PyQt6.QtMultimedia import QMediaPlayer
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.video_player.play()
        elif status == QMediaPlayer.MediaStatus.LoadedMedia:
            pass
        elif status == QMediaPlayer.MediaStatus.NoMedia:
            if hasattr(self, 'video_stacked'):
                self.video_stacked.setCurrentIndex(0)
            else:
                self.lbl_video_placeholder.show()
                self.video_widget.hide()

    def _on_video_error(self, error, error_string):
        """视频播放错误回调"""
        self._set_status(f"⚠️ 视频播放错误: {error_string}", "#ff7a17")
        if hasattr(self, 'video_stacked'):
            self.video_stacked.setCurrentIndex(0)
        else:
            self.lbl_video_placeholder.show()
            self.video_widget.hide()

    def _save_current_video(self):
        """保存当前播放的视频"""
        if not hasattr(self, 'current_video_path') or not self.current_video_path:
            self._set_status("⚠️ 没有正在播放的视频", "#ff7a17")
            return

        try:
            from PyQt6.QtWidgets import QFileDialog
            import shutil

            current_path = self.current_video_path
            ext = os.path.splitext(current_path)[1]
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存视频",
                os.path.basename(current_path),
                f"视频文件 (*{ext});;所有文件 (*)"
            )

            if save_path:
                shutil.copy2(current_path, save_path)
                self._set_status(f"✅ 视频已保存: {os.path.basename(save_path)}", "#dadbdf")
        except Exception as e:
            self._set_status(f"⚠️ 保存失败: {e}", "#ff7a17")

    # ============================================================
    #  生成按钮 + 状态条
    # ============================================================
    def _build_gen_button_area(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        self.btn_generate = QPushButton("🚀  开始生成")
        self.btn_generate.setFixedHeight(46)
        self.btn_generate.setStyleSheet("""
            QPushButton {
                background:#ffffff;
                color:#0a0a0a; font-size:15px; font-weight:bold;
                border:none; border-radius:9999px;
            }
            QPushButton:pressed { background:#fafaf7; }
            QPushButton:disabled { background:#1a1c20; color:#363a3f; }
        """)
        self.btn_generate.clicked.connect(self.start_generation)
        layout.addWidget(self.btn_generate)

        self.btn_interrupt = QPushButton("⏹  中断生成")
        self.btn_interrupt.setFixedHeight(32)
        self.btn_interrupt.setEnabled(False)
        self.btn_interrupt.setStyleSheet("""
            QPushButton {
                background:#0a0a0a; color:#dadbdf;
                border:1px solid #212327; border-radius:9999px;
            }
            QPushButton:hover { border-color:#363a3f; }
        """)
        self.btn_interrupt.clicked.connect(self.stop_generation)
        layout.addWidget(self.btn_interrupt)
        return w

    def _build_status_bar_widget(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_status = QLabel("✅ 就绪")
        self.lbl_status.setStyleSheet("color:#dadbdf; font-family:Consolas;")
        layout.addWidget(self.lbl_status, 1)

        self.progress_gen = QProgressBar()
        self.progress_gen.setRange(0, 100)
        self.progress_gen.setValue(0)
        self.progress_gen.setFixedWidth(160)
        self.progress_gen.setFixedHeight(8)
        self.progress_gen.setTextVisible(False)
        layout.addWidget(self.progress_gen)
        return w

    # ============================================================
    #  菜单
    # ============================================================
    def _build_menu(self):
        mb = self.menuBar()
        mb.setStyleSheet("""
            QMenuBar { background:#0a0a0a; color:#ffffff;
                       border-bottom:1px solid #212327; }
            QMenuBar::item:selected { background:#1a1c20; }
            QMenu { background:#191919; color:#ffffff;
                    border:1px solid #212327; }
            QMenu::item:selected { background:#363a3f; color:#ffffff; }
        """)
        m_file = mb.addMenu("📁 文件")
        a_open = QAction("加载图片", self)
        a_open.triggered.connect(self.select_image)
        m_file.addAction(a_open)
        m_file.addSeparator()
        a_quit = QAction("退出", self)
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_tool = mb.addMenu("🔧 工具")
        a_clear_log = QAction("清空日志", self)
        a_clear_log.triggered.connect(lambda: self.txt_log.clear())
        m_tool.addAction(a_clear_log)

        m_about = mb.addMenu("❓ 关于")
        a_about = QAction("关于本软件", self)
        a_about.triggered.connect(self._show_about)
        m_about.addAction(a_about)

        m_memory = mb.addMenu("🧹 内存")
        a_release = QAction("释放内存", self)
        a_release.triggered.connect(self.on_unload_models)
        m_memory.addAction(a_release)
        a_show = QAction("查看当前内存", self)
        a_show.triggered.connect(self._show_memory)
        m_memory.addAction(a_show)

    def _build_statusbar(self):
        self.statusBar().setStyleSheet(
            "background:#0a0a0a; color:#7d8187; font-size:11px;")
        self.statusBar().showMessage("AI 绘画工作站 v5.0 已就绪")

    # ============================================================
    #  默认值 + 控件联动
    # ============================================================
    def _init_defaults(self):
        if hasattr(self, 'refresh_models'):
            try:
                self.refresh_models()
            except Exception as e:
                print(f"refresh_models 失败: {e}")
        self._toggle_adetailer()
        self._toggle_ad_hand()
        self._toggle_hires()
        self._toggle_xy()
        self._toggle_cn()

    def _toggle_adetailer(self):
        on = self.chk_use_adetailer.isChecked()
        for c in (self.combo_ad_target, self.combo_adetailer_model,
                  self.scale_adetailer_strength):
            c.setEnabled(on)
        color = "#dadbdf" if on else "#7d8187"
        self.lbl_ad_str.setStyleSheet(f"color:{color}; font-family:Consolas;")

    def _toggle_ad_hand(self):
        on = self.chk_use_ad_hand.isChecked()
        for c in (self.combo_ad_hand, self.scale_ad_hand,
                  self.scale_ad_hand_blend):
            c.setEnabled(on)
        color = "#dadbdf" if on else "#7d8187"
        self.lbl_ad_hand_str.setStyleSheet(
            f"color:{color}; font-family:Consolas;")
        self.lbl_ad_hand_blend.setStyleSheet(
            f"color:{color}; font-family:Consolas;")

    def _toggle_hires(self):
        on = self.chk_hires.isChecked()
        for c in (self.combo_hires_scale, self.scale_hires_denoise,
                  self.combo_hires_upscaler):
            c.setEnabled(on)

    def _toggle_xy(self):
        on = self.chk_enable_xy.isChecked()
        for w in (self.combo_x_type, self.entry_x_vals,
                  self.combo_y_type, self.entry_y_vals):
            w.setEnabled(on)

    def _toggle_cn(self):
        on = self.chk_use_pose.isChecked()
        for c in (self.combo_cn_type, self.scale_cn_strength,
                  self.btn_load_cn_img):
            c.setEnabled(on)

    def _on_pose_transfer_toggled(self, checked: bool):
        """Pose Transfer 开关切换 → 自动联动其他控件"""
        if checked:
            # 自动配置 ControlNet 为 OpenPose
            if hasattr(self, 'combo_cn_type'):
                idx = self.combo_cn_type.findText("OpenPose")
                if idx >= 0:
                    self.combo_cn_type.setCurrentIndex(idx)
            QMessageBox.information(
                self, "Pose Transfer 已启用",
                "✅ 工作流程:\n\n"
                "1️⃣ AI 用提示词生成动作参考图\n"
                "2️⃣ 自动提取 OpenPose 骨架\n"
                "3️⃣ 骨架 + IP-Adapter 角色图 → 最终图\n\n"
                "⚠️ 请确保已上传【IP-Adapter 角色参考图】\n"
                "💡 推荐: 影响力 0.6~0.8"
            )
            self.lbl_pt_tip.setStyleSheet(
                "color:#dadbdf; padding:6px; background:#191919;"
                "border-radius:8px; font-size:11px;")
        else:
            self.lbl_pt_tip.setStyleSheet(
                "color:#7d8187; padding:6px; background:#191919;"
                "border-radius:8px; font-size:11px;")


    def _safe_set_check(self, name, val):
        w = getattr(self, name, None)
        if w is None or val is None:
            return
        try:
            w.setChecked(bool(val))
        except Exception as e:
            print(f"[preset] setChecked {name} 失败: {e}")

    def _safe_set_combo(self, name, text):
        w = getattr(self, name, None)
        if w is None or text is None:
            return
        try:
            idx = w.findText(str(text))
            if idx >= 0:
                w.setCurrentIndex(idx)
            else:
                # 模糊匹配（比如 "plus" 命中 "plus (推荐)"）
                for i in range(w.count()):
                    if str(text).lower() in w.itemText(i).lower():
                        w.setCurrentIndex(i)
                        return
        except Exception as e:
            print(f"[preset] setCombo {name} 失败: {e}")

    def _safe_set_float(self, name, val):
        """适配 FloatSlider / QDoubleSpinBox / QSlider(整数*100)"""
        w = getattr(self, name, None)
        if w is None or val is None:
            return
        try:
            # FloatSlider 一般有 set_value / setValue
            if hasattr(w, 'set_value'):
                w.set_value(float(val))
            elif hasattr(w, 'setValue'):
                # QSlider 是整数 → 推断是否要 *100
                from PyQt6.QtWidgets import QSlider
                if isinstance(w, QSlider):
                    w.setValue(int(round(float(val) * 100)))
                else:
                    w.setValue(float(val))
        except Exception as e:
            print(f"[preset] setFloat {name} 失败: {e}")

    def _safe_set_int(self, name, val):
        w = getattr(self, name, None)
        if w is None or val is None:
            return
        try:
            w.setValue(int(val))
        except Exception as e:
            print(f"[preset] setInt {name} 失败: {e}")

    # --- 快照：应用预设前备份当前参数，方便"还原" ---
    def _snapshot_current_params(self):
        """把当前所有可被预设修改的参数存到 self._preset_backup"""
        try:
            self._preset_backup = {
                "prompt": self.txt_prompt.toPlainText(),
                "neg":    self.txt_neg.toPlainText(),
                "steps":  self.spin_steps.value(),
                "cfg":    self._read_float(self.scale_cfg),
                "res":    self.combo_res.currentText(),
                "sampler":self.combo_sampler.currentText(),
                "strength": self._read_float(self.scale_strength),
                # adetailer
                "ad_face_on":  self.chk_use_adetailer.isChecked(),
                "ad_face_target": self.combo_ad_target.currentText(),
                "ad_face_model":  self.combo_adetailer_model.currentText(),
                "ad_face_str": self._read_float(self.scale_adetailer_strength),
                "ad_hand_on":  self.chk_use_ad_hand.isChecked(),
                "ad_hand_target": self.combo_ad_hand.currentText(),
                "ad_hand_str": self._read_float(self.scale_ad_hand),
                "ad_hand_blend": self._read_float(self.scale_ad_hand_blend),
                # hires
                "hires_on":   self.chk_hires.isChecked(),
                "hires_scale":self.combo_hires_scale.currentText(),
                "hires_denoise": self._read_float(self.scale_hires_denoise),
                "hires_upscaler": self.combo_hires_upscaler.currentText(),
                # cn
                "cn_on":     self.chk_use_pose.isChecked(),
                "cn_type":   self.combo_cn_type.currentText(),
                "cn_strength": self._read_float(self.scale_cn_strength),
                # ipa
                "ipa_on":    self.chk_use_ipa.isChecked(),
                "ipa_scale": self.spin_ipa_scale.value(),
                "ipa_variant": self.combo_ipa_variant.currentText(),
                # pose transfer
                "pt_on":     self.chk_pose_transfer.isChecked(),
                "pt_cn":     self.slider_pt_cn.value(),
                # consistency
                "auto_features": self.chk_auto_features.isChecked(),
                "ref_only":  self.chk_reference_only.isChecked(),
                "ref_fidelity": self.scale_ref_fidelity.value(),
            }
        except Exception as e:
            print(f"[preset] 快照失败: {e}")
            self._preset_backup = None

    def _read_float(self, w):
        """读取 FloatSlider / QDoubleSpinBox / QSlider 的当前值"""
        try:
            for m in ('value', 'get_value'):
                if hasattr(w, m):
                    v = getattr(w, m)()
                    return float(v)
        except Exception:
            pass
        return None

    def _update_preset_badge(self, n: int, lines: list):
        """更新还原按钮旁边的徽章 + tooltip"""
        # 徽章
        if hasattr(self, "lbl_preset_badge"):
            if n > 0:
                self.lbl_preset_badge.setText(f"● {n} 项已改")
                self.lbl_preset_badge.setStyleSheet(
                    "color:#dadbdf; font-weight:bold; font-size:11px;"
                    "padding:0 4px;")
            else:
                self.lbl_preset_badge.setText("")

        # 还原按钮 tooltip = 完整 diff
        if hasattr(self, "btn_restore_preset"):
            if n > 0:
                # tooltip 用纯文本（QToolTip 支持简单 html）
                plain_lines = []
                for ln in lines:
                    # 去 html 标签
                    import re
                    txt = re.sub(r'<[^>]+>', '', ln).strip()
                    plain_lines.append(txt)
                tip = ("<b>↩️ 点击还原以下改动：</b><br>"
                       + "<br>".join(plain_lines[:30]))
                if len(plain_lines) > 30:
                    tip += f"<br>...还有 {len(plain_lines)-30} 项"
                self.btn_restore_preset.setToolTip(tip)
            else:
                self.btn_restore_preset.setToolTip("还原到套用预设前的所有参数")


    # --- 还原预设前参数 ---
    def restore_preset_backup(self):
        bk = getattr(self, "_preset_backup", None)
        if not bk:
            self._set_status("⚠️ 没有可还原的快照", "#ff7a17")
            return
        try:
            self.txt_prompt.setPlainText(bk["prompt"])
            self.txt_neg.setPlainText(bk["neg"])
            self._safe_set_int("spin_steps", bk["steps"])
            self._safe_set_float("scale_cfg", bk["cfg"])
            self._safe_set_combo("combo_res", bk["res"])
            self._safe_set_combo("combo_sampler", bk["sampler"])
            self._safe_set_float("scale_strength", bk["strength"])
            self._safe_set_check("chk_use_adetailer", bk["ad_face_on"])
            self._safe_set_combo("combo_ad_target", bk["ad_face_target"])
            self._safe_set_combo("combo_adetailer_model", bk["ad_face_model"])
            self._safe_set_float("scale_adetailer_strength", bk["ad_face_str"])
            self._safe_set_check("chk_use_ad_hand", bk["ad_hand_on"])
            self._safe_set_combo("combo_ad_hand", bk["ad_hand_target"])
            self._safe_set_float("scale_ad_hand", bk["ad_hand_str"])
            self._safe_set_float("scale_ad_hand_blend", bk["ad_hand_blend"])
            self._safe_set_check("chk_hires", bk["hires_on"])
            self._safe_set_combo("combo_hires_scale", bk["hires_scale"])
            self._safe_set_float("scale_hires_denoise", bk["hires_denoise"])
            self._safe_set_combo("combo_hires_upscaler", bk["hires_upscaler"])
            self._safe_set_check("chk_use_pose", bk["cn_on"])
            self._safe_set_combo("combo_cn_type", bk["cn_type"])
            self._safe_set_float("scale_cn_strength", bk["cn_strength"])
            self._safe_set_check("chk_use_ipa", bk["ipa_on"])
            self.spin_ipa_scale.setValue(float(bk["ipa_scale"]))
            self._safe_set_combo("combo_ipa_variant", bk["ipa_variant"])
            self._safe_set_check("chk_pose_transfer", bk["pt_on"])
            self.slider_pt_cn.setValue(int(bk["pt_cn"]))
            self._safe_set_check("chk_auto_features", bk["auto_features"])
            self._safe_set_check("chk_reference_only", bk["ref_only"])
            self.scale_ref_fidelity.setValue(int(bk["ref_fidelity"]))
            self._toggle_adetailer(); self._toggle_ad_hand()
            self._toggle_hires(); self._toggle_cn()
        except Exception as e:
            self._set_status(f"⚠️ 还原失败: {e}", "#ff7a17")
            if hasattr(self, "lbl_preset_badge"):
                self.lbl_preset_badge.setText("")
            if hasattr(self, "btn_restore_preset"):
                self.btn_restore_preset.setToolTip("还原到套用预设前的所有参数")

    _CONTROL_LABELS = {
        "spin_steps":                ("步数 Steps",        "🎨 基础"),
        "scale_cfg":                 ("CFG Scale",         "🎨 基础"),
        "combo_res":                 ("分辨率",            "🎨 基础"),
        "combo_sampler":             ("采样器",            "🎨 基础"),
        "spin_count":                ("生成数量",          "🎨 基础"),
        "spin_seed":                 ("种子",              "🎨 基础"),
        "scale_strength":            ("重绘强度",          "🖼 图生图"),
        "chk_use_ipa":               ("IP-Adapter",       "🖼 图生图"),
        "spin_ipa_scale":            ("IPA 影响力",        "🖼 图生图"),
        "combo_ipa_variant":         ("IPA 版本",          "🖼 图生图"),
        "chk_pose_transfer":         ("Pose Transfer",    "🖼 图生图"),
        "slider_pt_cn":              ("姿势约束",          "🖼 图生图"),
        "chk_auto_features":         ("自动提取特征",      "🖼 图生图"),
        "chk_reference_only":        ("Reference-Only",   "🖼 图生图"),
        "scale_ref_fidelity":        ("参考强度",          "🖼 图生图"),
        "chk_use_pose":              ("ControlNet",        "🕹 ControlNet"),
        "combo_cn_type":             ("CN 类型",           "🕹 ControlNet"),
        "scale_cn_strength":         ("CN 条件强度",       "🕹 ControlNet"),
        "chk_use_adetailer":         ("修脸",              "⚙️ 高级"),
        "combo_ad_target":           ("脸部检测目标",      "⚙️ 高级"),
        "combo_adetailer_model":     ("脸部模型",          "⚙️ 高级"),
        "scale_adetailer_strength":  ("脸部修复强度",      "⚙️ 高级"),
        "chk_use_ad_hand":           ("修手",              "⚙️ 高级"),
        "combo_ad_hand":             ("手部检测目标",      "⚙️ 高级"),
        "scale_ad_hand":             ("手部重绘强度",      "⚙️ 高级"),
        "scale_ad_hand_blend":       ("手部融合度",        "⚙️ 高级"),
        "chk_hires":                 ("Hires.fix",         "⚙️ 高级"),
        "combo_hires_scale":         ("放大倍率",          "⚙️ 高级"),
        "scale_hires_denoise":       ("Hires 降噪",        "⚙️ 高级"),
        "combo_hires_upscaler":      ("Upscaler",          "⚙️ 高级"),
        "txt_prompt":                ("正向提示词",        "🎨 基础"),
        "txt_neg":                   ("负向提示词",        "🎨 基础"),
    }

    # --- 读取控件当前值（统一接口） ---
    def _get_widget_value(self, name):
        from PyQt6.QtWidgets import (
            QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QSlider, QTextEdit
        )
        w = getattr(self, name, None)
        if w is None:
            return None
        try:
            if isinstance(w, QCheckBox):     return w.isChecked()
            if isinstance(w, QComboBox):     return w.currentText()
            if isinstance(w, (QSpinBox, QDoubleSpinBox)): return w.value()
            if isinstance(w, QSlider):       return w.value()
            if isinstance(w, QTextEdit):     return w.toPlainText()
            if hasattr(w, 'value'):          return w.value()  # FloatSlider
            if hasattr(w, 'get_value'):      return w.get_value()
        except Exception:
            pass
        return None

    def _flash_widget(self, name, color="#dadbdf"):
        from PyQt6.QtWidgets import QGraphicsColorizeEffect
        from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
        from PyQt6.QtGui import QColor

        w = getattr(self, name, None)
        if w is None:
            return
        try:
            # 已有 effect 就跳过（避免叠加）
            if w.graphicsEffect() is not None:
                return

            effect = QGraphicsColorizeEffect(w)
            effect.setColor(QColor(color))
            effect.setStrength(0.0)
            w.setGraphicsEffect(effect)

            anim = QPropertyAnimation(effect, b"strength", self)
            anim.setDuration(2500)
            anim.setKeyValueAt(0.0, 0.0)
            anim.setKeyValueAt(0.15, 0.85)   # 快速点亮
            anim.setKeyValueAt(0.50, 0.85)   # 保持
            anim.setKeyValueAt(1.0, 0.0)     # 淡出
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

            def _cleanup():
                try: w.setGraphicsEffect(None)
                except: pass

            anim.finished.connect(_cleanup)
            anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

            # 防 GC
            if not hasattr(self, '_flash_anims'):
                self._flash_anims = []
            self._flash_anims.append(anim)
            # 限长，避免无限增长
            self._flash_anims = self._flash_anims[-50:]
        except Exception as e:
            print(f"[flash] {name}: {e}")

    def _build_diff_report(self, before: dict, after: dict):
        """返回 (改动列表, 受影响 Tab 集合)"""
        lines = []
        tabs_hit = set()
        for key, (cn_name, tab_name) in self._CONTROL_LABELS.items():
            b = before.get(key)
            a = after.get(key)
            if b is None and a is None:
                continue
            # 浮点数容差
            try:
                if isinstance(b, float) and isinstance(a, float):
                    if abs(b - a) < 1e-4:
                        continue
            except: pass
            if b == a:
                continue
            # 文本太长截一下
            def _fmt(v):
                if v is None: return "—"
                s = str(v)
                return s if len(s) < 40 else s[:37] + "..."
            lines.append(f"  • {cn_name}: <span style='color:#7d8187'>{_fmt(b)}</span> "
                         f"→ <span style='color:#dadbdf'>{_fmt(a)}</span>")
            tabs_hit.add(tab_name)
        return lines, tabs_hit

    # ============================================================
    #  画廊回调
    # ============================================================
    def _on_gallery_picked(self, path: str):
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
        if hasattr(self, 'btn_edit'):
            self.btn_edit.setEnabled(True)
        if hasattr(self, 'btn_upscale'):
            self.btn_upscale.setEnabled(True)

    # ============================================================
    #  辅助
    # ============================================================

    def _insert_lora_triggers(self, slot_idx=None):
        """插入 LoRA 触发词到提示词框
        slot_idx=None → 插入所有槽的触发词
        slot_idx=0/1/2 → 只插入指定槽
        """
        import os
        triggers_list = []
    
        for i, combo in enumerate(self.combo_loras):
            if slot_idx is not None and i != slot_idx:
                continue
        
            lora_name = combo.currentText().strip()
            if not lora_name or lora_name in ("无", "None", ""):
                continue
        
            # 去除可能的 [大小] 后缀
            if "[" in lora_name:
                lora_name = lora_name.split("[")[0].strip()
        
            base = os.path.splitext(lora_name)[0]
        
            # 尝试多个可能路径
            for sub in ["sdxl", "sd1.5", "sd15", ""]:
                txt_path = os.path.join("loras", sub, base + ".txt") if sub else os.path.join("loras", base + ".txt")
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                triggers_list.append(content)
                        break
                    except Exception as e:
                        print(f"⚠️ 读取 {txt_path} 失败: {e}")
    
        if not triggers_list:
            self._set_status("⚠️ 没有可插入的触发词", "#ff7a17")
            return

        all_triggers = ", ".join(triggers_list)
        cur = self.txt_prompt.toPlainText().strip()
        new_text = f"{all_triggers}, {cur}" if cur else all_triggers
        self.txt_prompt.setPlainText(new_text)

        self._set_status(f"✅ 已插入 {len(triggers_list)} 组触发词", "#dadbdf")

    def _open_output_folder(self):
        import subprocess
        from utils.app_utils import OUTPUT_DIR
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile(os.path.abspath(OUTPUT_DIR))
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', OUTPUT_DIR])
            else:
                subprocess.Popen(['xdg-open', OUTPUT_DIR])
        except Exception as e:
            print(f"打开目录失败: {e}")

    def _show_about(self):
        QMessageBox.about(
            self, "关于",
            "<b>AI 绘画工作站 v5.0</b><br>"
            "PyQt6 重构版 — GPU 加速<br><br>"
            "基于 Stable Diffusion + ADetailer<br>"
            "支持 LoRA / ControlNet / Hires.fix / IP-Adapter / Pose Transfer"
        )

    def on_unload_models(self):
        if getattr(self, 'is_generating', False):
            QMessageBox.warning(self, "提示", "请先停止当前生成任务")
            return
        try:
            self._set_status("🧹 正在释放模型...", "#ff7a17")
            if hasattr(self, 'ai'):
                self.ai.unload_all()
            try:
                import psutil
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                self._set_status(
                    f"✅ 模型已释放 (当前内存 {mem:.0f} MB)", "#dadbdf")
            except ImportError:
                self._set_status("✅ 模型已释放", "#dadbdf")
        except Exception as e:
            QMessageBox.critical(self, "释放失败", str(e))

    def _show_memory(self):
        try:
            import psutil
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            QMessageBox.information(
                self, "内存使用情况",
                f"当前进程内存: {mem:.1f} MB\n\n"
                f"如果数值过大,可以点'释放内存'清理。"
            )
        except ImportError:
            QMessageBox.information(
                self, "提示", "请安装 psutil: pip install psutil"
            )

    def append_log(self, text: str, color: str = "#ffffff"):
        self.txt_log.append(
            f'<span style="color:{color}; font-family:Consolas;">'
            f'{text}</span>'
        )
        self.txt_log.verticalScrollBar().setValue(
            self.txt_log.verticalScrollBar().maximum()
        )

    def set_status(self, text: str, color: str = "#dadbdf"):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color:{color}; font-family:Consolas;")

    def _set_status(self, text: str, color: str = "#dadbdf"):
        self.set_status(text, color)

    def set_progress(self, value: int):
        self.progress_gen.setValue(value)

    def _on_video_mode_changed(self, idx: int):
        """切换生成模式时刷新 UI"""
        is_travel = (idx == 3)
        self.grp_prompt_travel.setVisible(is_travel)


    def on_pick_video_input(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "选择首帧图/输入视频",
            "", "图片/视频 (*.png *.jpg *.jpeg *.mp4 *.gif)"
        )
        if path:
            self._video_input_path = path
            self.lbl_video_input.setText(os.path.basename(path))


    def _create_video_preview(self):
        """创建视频预览播放器组件"""
        container = QWidget()
        container.setStyleSheet("background:#0a0a0a;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.video_player.setAudioOutput(self.audio_output)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(240)
        self.video_widget.setStyleSheet("background:#0a0a0a;")
        self.video_player.setVideoOutput(self.video_widget)

        self.video_player.mediaStatusChanged.connect(self._on_video_media_changed)
        self.video_player.errorOccurred.connect(self._on_video_player_error)

        layout.addWidget(self.video_widget)

        self.lbl_video_placeholder = QLabel("🎥 视频生成后自动播放\n或从下方历史列表双击选择")
        self.lbl_video_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video_placeholder.setStyleSheet("color:#7d8187;padding:40px;font-size:12px;")
        self.lbl_video_placeholder.setMinimumHeight(240)
        layout.addWidget(self.lbl_video_placeholder)
        self.lbl_video_placeholder.show()
        self.video_widget.hide()

        return container

    def play_video(self, video_path: str):
        """播放指定路径的视频"""
        if not os.path.exists(video_path):
            self._set_status(f"⚠️ 视频文件不存在: {video_path}", "#ff7a17")
            return

        try:
            self.video_player.stop()
            self.video_player.setSource(QUrl.fromLocalFile(video_path))
            self.lbl_video_placeholder.hide()
            self.video_widget.show()
            self.video_player.play()
            self._set_status(f"🎥 正在播放: {os.path.basename(video_path)}", "#dadbdf")
        except Exception as e:
            self._set_status(f"⚠️ 视频播放失败: {e}", "#ff7a17")
            self.lbl_video_placeholder.show()
            self.video_widget.hide()

    def _on_video_media_changed(self, status):
        """视频媒体状态变化回调"""
        from PyQt6.QtMultimedia import QMediaPlayer
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.video_player.play()
        elif status == QMediaPlayer.MediaStatus.LoadedMedia:
            pass
        elif status == QMediaPlayer.MediaStatus.NoMedia:
            if hasattr(self, 'video_stacked'):
                self.video_stacked.setCurrentIndex(0)
            else:
                self.lbl_video_placeholder.show()
                self.video_widget.hide()

    def _on_video_error(self, error, error_string):
        """视频播放错误回调"""
        self._set_status(f"⚠️ 视频播放错误: {error_string}", "#ff7a17")
        if hasattr(self, 'video_stacked'):
            self.video_stacked.setCurrentIndex(0)
        else:
            self.lbl_video_placeholder.show()
            self.video_widget.hide()

    def on_generate_video(self):
        """触发视频生成流程"""
        if getattr(self, 'is_generating', False):
            self._set_status("⚠️ 正在生成中，请等待", "#ff7a17")
            return

        try:
            self.is_generating = True
            self.btn_gen_video.setEnabled(False)
            self.btn_gen_video.setText("生成中...")

            mode = self.combo_video_mode.currentIndex()
            prompt = self.txt_video_prompt.toPlainText().strip()
            negative = self.txt_video_neg.toPlainText().strip() or (
                "bad hands, bad fingers, extra fingers, missing fingers, "
                "deformed hands, orange tint, warm color cast, oversaturated, "
                "lowres, worst quality, low quality, jpeg artifacts, blurry"
            )

            if not prompt:
                self._set_status("⚠️ 请输入正面提示词", "#ff7a17")
                self.is_generating = False
                self.btn_gen_video.setEnabled(True)
                self.btn_gen_video.setText("🎬 生成视频")
                return

            input_path = getattr(self, '_video_input_path', None)
            if mode in (1, 2) and not input_path:
                self._set_status("⚠️ 图生视频/视频转绘需要选择输入文件", "#ff7a17")
                self.is_generating = False
                self.btn_gen_video.setEnabled(True)
                self.btn_gen_video.setText("🎬 生成视频")
                return

            num_frames = self.spin_video_frames.value()
            num_steps = self.spin_video_steps.value()
            guidance = self.spin_video_cfg.value()
            width = self.spin_video_w.value()
            height = self.spin_video_h.value()
            fps = self.spin_video_fps.value()
            scheduler_map = {"EulerDiscrete (推荐)": "euler", "DPM++ 2M": "dpm++", "LCM (快速)": "lcm", "DDIM": "euler"}
            scheduler = scheduler_map.get(self.combo_video_sched.currentText(), "euler")
            motion_lora = self.combo_motion_lora.currentText() if self.combo_motion_lora.currentText() != "无" else None
            motion_scale = self.spin_motion_scale.value()
            use_context_window = self.chk_long_video.isChecked() and num_frames > 32
            output_format = self.combo_video_fmt.currentText()

            prompt_travel = []
            if mode == 3 and self.grp_prompt_travel.isVisible():
                travel_text = self.txt_prompt_travel.toPlainText().strip()
                if travel_text:
                    for line in travel_text.split('\n'):
                        line = line.strip()
                        if line and '|' in line:
                            parts = line.split('|', 1)
                            if len(parts) == 2:
                                try:
                                    frame_idx = int(parts[0].strip())
                                    pt_prompt = parts[1].strip()
                                    prompt_travel.append((frame_idx, pt_prompt))
                                except ValueError:
                                    continue

            self._set_status(f"🎬 开始视频生成: {num_frames}帧 {width}x{height}", "#ff7a17")
            self.lbl_video_status.setText("🎬 生成中...")
            self.lbl_video_status.setStyleSheet("color:#ff7a17; padding:4px;")

            def progress_callback(step, timestep, callback_kwargs):
                try:
                    progress = int((step / max(num_steps, 1)) * 100)
                    self._app_bridge.status_msg.emit(f"🎬 生成中 {progress}%", "#ff7a17")
                except Exception:
                    pass
                return callback_kwargs
                try:
                    total = num_steps * num_frames if use_context_window else num_steps
                    progress = int((step / total) * 100)
                    self.set_progress(progress)
                except Exception:
                    pass

            import threading
            def generate_task():
                try:
                    if not hasattr(self, 'video_generator'):
                        from utils.video_gen import VideoGenerator
                        self.video_generator = VideoGenerator(self.ai)

                    result = self.video_generator.generate(
                        prompt=prompt,
                        negative=negative,
                        num_frames=num_frames,
                        num_steps=num_steps,
                        guidance=guidance,
                        width=width,
                        height=height,
                        fps=fps,
                        scheduler=scheduler,
                        motion_lora=motion_lora,
                        motion_scale=motion_scale,
                        use_context_window=use_context_window,
                        prompt_travel=prompt_travel if prompt_travel else None,
                        output_format=output_format,
                        output_dir="photo/videos",
                        progress_callback=progress_callback,
                    )

                    video_path, seed = result

                    from PyQt6.QtCore import QMetaObject, Q_ARG, Qt
                    QMetaObject.invokeMethod(
                        self, "_on_video_generated",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, video_path),
                        Q_ARG(int, seed),
                    )

                except Exception as e:
                    import traceback
                    error_msg = f"❌ 视频生成失败: {str(e)}"
                    print(f"[VIDEO GEN ERROR]\n{traceback.format_exc()}")
                    from PyQt6.QtCore import QMetaObject, Q_ARG, Qt
                    QMetaObject.invokeMethod(
                        self, "_on_video_error",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(int, 0),
                        Q_ARG(str, error_msg),
                    )

            threading.Thread(target=generate_task, daemon=True).start()

        except Exception as e:
            self._set_status(f"⚠️ 参数校验失败: {e}", "#ff7a17")
            self.is_generating = False
            self.btn_gen_video.setEnabled(True)
            self.btn_gen_video.setText("🎬 生成视频")

    @pyqtSlot(str, int)
    def _on_video_generate_done(self, video_path: str, seed: int):
        """视频生成完成回调（主线程）"""
        try:
            self.is_generating = False
            self.btn_gen_video.setEnabled(True)
            self.btn_gen_video.setText("🎬 生成视频")
            self.set_progress(100)

            self._set_status(f"✅ 视频生成完成: {os.path.basename(video_path)} (seed={seed})", "#dadbdf")
            self.lbl_video_status.setText(f"✅ 已完成 (seed={seed})")
            self.lbl_video_status.setStyleSheet("color:#dadbdf; padding:4px;")

            if hasattr(self, 'video_player') and hasattr(self, 'video_widget'):
                self.play_video(video_path)

            if hasattr(self, 'video_list'):
                self._refresh_video_gallery()

            if hasattr(self, 'gallery'):
                if not hasattr(self, '_gallery_seen_paths'):
                    self._gallery_seen_paths = set()
                abs_path = os.path.abspath(video_path)
                if abs_path not in self._gallery_seen_paths:
                    self._gallery_seen_paths.add(abs_path)
                    self.gallery.add_image(video_path, prepend=True)

        except Exception as e:
            self._set_status(f"⚠️ 视频生成后处理失败: {e}", "#ff7a17")

    def _refresh_video_gallery(self):
        """刷新视频历史画廊"""
        try:
            if not hasattr(self, 'video_list'):
                return

            video_dir = "photo/videos"
            if not os.path.isdir(video_dir):
                os.makedirs(video_dir, exist_ok=True)
                return

            self.video_list.clear()

            video_extensions = ('.mp4', '.gif', '.webm', '.mov')
            video_files = []
            for f in os.listdir(video_dir):
                if f.lower().endswith(video_extensions):
                    full_path = os.path.join(video_dir, f)
                    if os.path.isfile(full_path):
                        video_files.append((full_path, f))

            video_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)

            for full_path, filename in video_files:
                try:
                    item = QListWidgetItem()
                    item.setText(filename)
                    item.setData(Qt.ItemDataRole.UserRole, full_path)

                    pixmap = self._generate_video_thumbnail(full_path)
                    if pixmap:
                        item.setIcon(QIcon(pixmap))

                    self.video_list.addItem(item)
                except Exception as e:
                    print(f"⚠️ 添加视频项失败: {filename} - {e}")

        except Exception as e:
            print(f"⚠️ 刷新视频画廊失败: {e}")

    def _generate_video_thumbnail(self, video_path: str) -> QPixmap:
        """生成视频缩略图"""
        try:
            if video_path.lower().endswith('.gif'):
                from PIL import Image
                img = Image.open(video_path)
                img.seek(0)
                frame = img.convert("RGB")
                from PIL.ImageQt import ImageQt
                qimg = ImageQt(frame)
                pixmap = QPixmap.fromImage(QImage(qimg))
                return pixmap.scaled(160, 90, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

            else:
                pixmap = QPixmap(160, 90)
                pixmap.fill(QColor("#191919"))
                painter = QPainter(pixmap)
                painter.setPen(QColor("#7d8187"))
                painter.setFont(QFont("Segoe UI", 10))
                painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🎬")
                painter.end()
                return pixmap

        except Exception as e:
            print(f"⚠️ 生成缩略图失败: {video_path} - {e}")
            return None

    def _on_video_item_clicked(self, item: QListWidgetItem):
        """双击视频项播放"""
        try:
            video_path = item.data(Qt.ItemDataRole.UserRole)
            if video_path and os.path.exists(video_path):
                self.play_video(video_path)
        except Exception as e:
            self._set_status(f"⚠️ 播放视频失败: {e}", "#ff7a17")

    def _set_video_duration(self, seconds: int):
        """设置视频时长（自动计算帧数）"""
        try:
            fps = self.spin_video_fps.value()
            num_frames = seconds * fps
            if num_frames < 8:
                num_frames = 8
            elif num_frames > 80:
                num_frames = 80
            self.spin_video_frames.setValue(num_frames)
        except Exception as e:
            print(f"⚠️ 设置视频时长失败: {e}")

    def _add_travel_segment(self):
        """添加提示词旅行分段"""
        try:
            segment_idx = len(self.travel_segments) + 1
            segment_widget = QWidget()
            segment_lay = QHBoxLayout(segment_widget)
            segment_lay.setContentsMargins(0, 0, 0, 0)

            lbl = QLabel(f"段 {segment_idx}:")
            lbl.setFixedWidth(40)
            segment_lay.addWidget(lbl)

            spin_frame = QSpinBox()
            spin_frame.setRange(0, 100)
            spin_frame.setValue(segment_idx * 8)
            segment_lay.addWidget(spin_frame)

            txt_prompt = QLineEdit()
            txt_prompt.setPlaceholderText("提示词...")
            segment_lay.addWidget(txt_prompt, 1)

            btn_remove = QPushButton("✕")
            btn_remove.setFixedWidth(28)
            btn_remove.clicked.connect(
                lambda: self._remove_travel_segment(segment_widget)
            )
            segment_lay.addWidget(btn_remove)

            self.travel_container.addWidget(segment_widget)
            self.travel_segments.append({
                'widget': segment_widget,
                'frame_spin': spin_frame,
                'prompt_edit': txt_prompt
            })
        except Exception as e:
            print(f"⚠️ 添加旅行分段失败: {e}")

    def _remove_travel_segment(self, widget: QWidget):
        """移除提示词旅行分段"""
        try:
            self.travel_container.removeWidget(widget)
            widget.deleteLater()
            self.travel_segments = [
                s for s in self.travel_segments if s['widget'] != widget
            ]
            for i, seg in enumerate(self.travel_segments, 1):
                seg['widget'].findChild(QLabel).setText(f"段 {i}:")
        except Exception as e:
            print(f"⚠️ 移除旅行分段失败: {e}")