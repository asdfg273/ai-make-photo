# file_name: ui_builder.py
import os
import sys

from PyQt6.QtWidgets import (
    QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox,
    QTabWidget, QGroupBox, QScrollArea, QSplitter,
    QFileDialog, QMessageBox, QProgressBar, QSizePolicy,
    QListWidget, QStackedWidget,QFrame
)
from PyQt6.QtCore import Qt, QTimer, QSize, QUrl
from PyQt6.QtGui import QColor, QIcon, QPixmap, QAction
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
import core.presets
from ui.extension_market import ExtensionMarketDialog
from ui.gallery_panel import GalleryPanel

# 从独立文件导入 — 已从 ui_builder.py 提取
from utils.gpu_init import enable_gpu_acceleration
from ui.widgets import FloatSlider, GpuCanvas
from ui.splash import SplashScreen, create_splash
from ui.design_tokens import DARK_STYLE,VIDEO_TAB_QSS

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
                QTimer.singleShot(100, self._refresh_video_gallery)
            else:
                print("⚠️ right_stacked 不存在")
        except Exception as e:
            import traceback
            print(f"⚠️ 切换视频模式失败: {e}")
            print(traceback.format_exc())

    def _switch_to_image_mode(self):
        """切换回图片模式（恢复右侧面板 + 停止视频播放）"""
        try:
            if hasattr(self, 'right_stacked'):
                self.stop_video()
                self.right_stacked.setCurrentIndex(0)
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

    # ==================================================================
    #  动画 / 视频 标签页
    # ==================================================================

    def _build_tab_animation(self) -> QWidget:
        # ---------------- 局部构件工厂（消除重复样式代码） ----------------
        LABEL_W = 88

        def field(text: str, w: int = LABEL_W) -> QLabel:
            lb = QLabel(text)
            lb.setProperty("role", "field")
            if w:
                lb.setMinimumWidth(w)
            return lb

        def hint(text: str) -> QLabel:
            lb = QLabel(text)
            lb.setProperty("role", "hint")
            lb.setWordWrap(True)
            return lb

        def group(title: str, accent: bool = False) -> QGroupBox:
            g = QGroupBox(title)
            if accent:
                g.setProperty("accent", True)
            return g

        def spin(lo, hi, val, step=1, w=96) -> QSpinBox:
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setValue(val)
            sp.setMinimumWidth(w)          # ← 不再 setFixedWidth，避免数字被截断
            sp.setMinimumHeight(32)
            sp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return sp

        def dspin(lo, hi, val, step=0.5, dec=2, w=104) -> QDoubleSpinBox:
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
            sp.setValue(val)
            sp.setMinimumWidth(w)
            sp.setMinimumHeight(32)
            sp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return sp

        def pill(text: str, w: int | None = None) -> QPushButton:
            b = QPushButton(text)
            b.setProperty("role", "pill")
            b.setMinimumHeight(30)
            if w:
                b.setMinimumWidth(w)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            return b

        def grid() -> QGridLayout:
            g = QGridLayout()
            g.setHorizontalSpacing(10)
            g.setVerticalSpacing(8)
            g.setContentsMargins(0, 0, 0, 0)
            return g

        # ---------------- 滚动容器 ----------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        w = QWidget()
        w.setObjectName("animRoot")
        w.setMinimumWidth(452)                 # 留出滚动条空间，避免横向压缩
        root = QVBoxLayout(w)
        root.setSpacing(4)
        root.setContentsMargins(12, 8, 12, 16)

        # ============ 💡 使用提示 ============
        grp_tips = group("💡 使用提示")
        tips_lay = QVBoxLayout(grp_tips)
        tips = QLabel(
            "<ul style='margin:0; padding-left:18px; line-height:150%;'>"
            "<li><b>文生视频</b>：仅用提示词生成，无需输入文件</li>"
            "<li><b>图生视频</b>：选一张图作首帧，AI 延续动画</li>"
            "<li><b>视频转绘</b>：选视频文件，AI 改变画风</li>"
            "<li><b>提示词旅行</b>：不同帧用不同提示词，做剧情变化</li>"
            "</ul>"
        )
        tips.setProperty("role", "body")
        tips.setWordWrap(True)
        tips_lay.addWidget(tips)
        root.addWidget(grp_tips)

        # ============ 🎯 生成模式 ============
        grp_mode = group("🎯 生成模式", accent=True)
        mode_lay = QVBoxLayout(grp_mode)
        self.combo_video_mode = QComboBox()
        self.combo_video_mode.addItems([
            "📝 文生视频 (txt2video)",
            "🖼️ 图生视频 (img2video) — 首帧引导",
            "🎞️ 视频转绘 (vid2vid) — 改画风",
            "✨ 提示词旅行 (Prompt Travel) — 剧情视频",
        ])
        self.combo_video_mode.setMinimumHeight(36)
        mode_lay.addWidget(self.combo_video_mode)
        self.lbl_video_mode_desc = hint("无需输入文件，直接填写提示词即可生成。")
        mode_lay.addWidget(self.lbl_video_mode_desc)
        root.addWidget(grp_mode)

        # ============ 📥 输入文件 ============
        self.grp_video_input = group("📥 输入文件")
        in_lay = QVBoxLayout(self.grp_video_input)
        in_lay.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.lbl_video_input = QLabel("未选择文件")
        self.lbl_video_input.setProperty("role", "hint")
        self.lbl_video_input.setMinimumHeight(34)
        self.lbl_video_input.setToolTip("未选择文件")
        row.addWidget(self.lbl_video_input, 1)

        self.btn_pick_video_input = pill("📂 选择", 84)
        self.btn_pick_video_input.clicked.connect(self.on_pick_video_input)
        row.addWidget(self.btn_pick_video_input)

        self.btn_clear_video_input = pill("✖", 40)
        self.btn_clear_video_input.setToolTip("清除已选文件")
        self.btn_clear_video_input.clicked.connect(self._clear_video_input)
        row.addWidget(self.btn_clear_video_input)
        in_lay.addLayout(row)

        in_lay.addWidget(hint("图生视频 → 选择首帧图片；视频转绘 → 选择输入视频。"))
        root.addWidget(self.grp_video_input)

        # ============ 💬 提示词 ============
        grp_prompt = group("💬 提示词")
        p_lay = QVBoxLayout(grp_prompt)
        p_lay.setSpacing(6)

        p_lay.addWidget(field("正面提示词", 0))
        self.txt_video_prompt = QTextEdit()
        self.txt_video_prompt.setFixedHeight(92)
        self.txt_video_prompt.setPlaceholderText("例如：一只可爱的小猫在草地上奔跑, best quality, masterpiece")
        p_lay.addWidget(self.txt_video_prompt)

        p_lay.addWidget(field("负面提示词", 0))
        self.txt_video_neg = QTextEdit()
        self.txt_video_neg.setFixedHeight(68)
        self.txt_video_neg.setPlaceholderText("例如：blurry, lowres, worst quality, text, watermark")
        p_lay.addWidget(self.txt_video_neg)

        # ── AI 工具按钮行（与基础 Tab 一致）──
        prompt_btn_row = QHBoxLayout()
        prompt_btn_row.setSpacing(6)

        self.btn_enhance_video_prompt = pill("✨ 智能改写", 108)
        self.btn_enhance_video_prompt.setToolTip(
            "将自然语言提示词自动转换为英文 Danbooru 标签\n"
            "同时改写正面和负面提示词\n"
            "首次使用会下载 Qwen2-VL 模型 (~4.5GB)")
        prompt_btn_row.addWidget(self.btn_enhance_video_prompt)

        self.btn_vision_video_prompt = pill("📷 识图生成", 108)
        self.btn_vision_video_prompt.setToolTip(
            "上传一张图片，AI 自动识别内容并生成提示词")
        prompt_btn_row.addWidget(self.btn_vision_video_prompt)

        self.btn_enhance_travel = pill("✨ 改写旅行段", 120)
        self.btn_enhance_travel.setToolTip(
            "用 AI 改写所有旅行分段的提示词")
        prompt_btn_row.addWidget(self.btn_enhance_travel)
        prompt_btn_row.addStretch()
        p_lay.addLayout(prompt_btn_row)

        root.addWidget(grp_prompt)

        # ============ ✨ 提示词旅行（唯一入口，两种编辑方式二选一）============
        self.grp_prompt_travel = group("✨ 提示词旅行", accent=True)
        self.grp_prompt_travel.setCheckable(True)
        self.grp_prompt_travel.setChecked(False)
        self.grp_prompt_travel.setToolTip("在指定帧切换提示词，实现剧情/动作变化")
        tv_lay = QVBoxLayout(self.grp_prompt_travel)
        tv_lay.setSpacing(8)

        tv_lay.addWidget(hint("在不同帧使用不同提示词。可用「分段编辑」或「文本格式」，两者填写其一即可。"))

        # -- 编辑方式切换 --
        sw_row = QHBoxLayout()
        sw_row.setSpacing(8)
        sw_row.addWidget(field("编辑方式:", 72))
        self.combo_travel_mode = QComboBox()
        self.combo_travel_mode.addItems(["🧩 分段编辑（推荐）", "⌨️ 文本格式"])
        sw_row.addWidget(self.combo_travel_mode, 1)
        tv_lay.addLayout(sw_row)

        # -- ① 分段编辑 --
        self.wrap_travel_segments = QWidget()
        seg_lay = QVBoxLayout(self.wrap_travel_segments)
        seg_lay.setContentsMargins(0, 0, 0, 0)
        seg_lay.setSpacing(6)

        self.travel_container = QVBoxLayout()
        self.travel_container.setSpacing(6)
        seg_lay.addLayout(self.travel_container)

        seg_btn_row = QHBoxLayout()
        seg_btn_row.setSpacing(6)
        btn_add_segment = pill("➕ 添加段", 96)
        btn_add_segment.clicked.connect(self._add_travel_segment)
        seg_btn_row.addWidget(btn_add_segment)
        btn_auto_spread = pill("⇄ 均匀分布帧号", 130)
        btn_auto_spread.setToolTip("按当前总帧数自动重排各段起始帧")
        btn_auto_spread.clicked.connect(self._spread_travel_frames)
        seg_btn_row.addWidget(btn_auto_spread)
        seg_btn_row.addStretch()
        seg_lay.addLayout(seg_btn_row)
        tv_lay.addWidget(self.wrap_travel_segments)

        # -- ② 文本格式 --
        self.wrap_travel_text = QWidget()
        txt_lay = QVBoxLayout(self.wrap_travel_text)
        txt_lay.setContentsMargins(0, 0, 0, 0)
        txt_lay.setSpacing(4)
        txt_lay.addWidget(hint("格式：帧号|提示词（每行一个关键帧）"))
        self.txt_prompt_travel = QTextEdit()
        self.txt_prompt_travel.setFixedHeight(100)
        self.txt_prompt_travel.setPlaceholderText(
            "0|1girl, smiling, sunny day\n8|1girl, surprised, wind blowing\n16|1girl, crying, rain falling"
        )
        txt_lay.addWidget(self.txt_prompt_travel)
        self.wrap_travel_text.setVisible(False)
        tv_lay.addWidget(self.wrap_travel_text)

        self.combo_travel_mode.currentIndexChanged.connect(self._on_travel_edit_mode_changed)
        root.addWidget(self.grp_prompt_travel)

        self.travel_segments = []

        # 兼容旧代码引用
        self.grp_travel = self.grp_prompt_travel

        # ============ 🎞️ 视频参数 ============
        grp_video = group("🎞️ 视频参数")
        v_lay = QVBoxLayout(grp_video)
        v_lay.setSpacing(8)

        g = grid()
        g.setColumnStretch(4, 1)

        # 帧数 / FPS / 时长
        g.addWidget(field("帧数:"), 0, 0)
        self.spin_video_frames = spin(8, 80, 16)
        self.spin_video_frames.setToolTip("总生成帧数；开启长视频模式后上限提升")
        g.addWidget(self.spin_video_frames, 0, 1)
        g.addWidget(field("FPS:", 52), 0, 2)
        self.spin_video_fps = spin(4, 30, 8)
        g.addWidget(self.spin_video_fps, 0, 3)
        self.lbl_video_duration = QLabel("≈ 2.0 秒")
        self.lbl_video_duration.setProperty("role", "value")
        g.addWidget(self.lbl_video_duration, 0, 4)

        # 步数 / CFG
        g.addWidget(field("步数:"), 1, 0)
        self.spin_video_steps = spin(10, 100, 25)
        g.addWidget(self.spin_video_steps, 1, 1)
        g.addWidget(field("CFG:", 52), 1, 2)
        self.spin_video_cfg = dspin(1.0, 20.0, 7.5, 0.5, 1)
        g.addWidget(self.spin_video_cfg, 1, 3)

        # 分辨率
        g.addWidget(field("宽 × 高:"), 2, 0)
        res_row = QHBoxLayout()
        res_row.setSpacing(6)
        self.spin_video_w = spin(256, 1024, 512, 64, 88)
        self.spin_video_h = spin(256, 1024, 512, 64, 88)
        x_lbl = QLabel("×")
        x_lbl.setProperty("role", "field")
        res_row.addWidget(self.spin_video_w)
        res_row.addWidget(x_lbl)
        res_row.addWidget(self.spin_video_h)
        res_row.addStretch()
        g.addLayout(res_row, 2, 1, 1, 4)

        # 采样器
        g.addWidget(field("采样器:"), 3, 0)
        self.combo_video_sched = QComboBox()
        self.combo_video_sched.addItems(["EulerDiscrete (推荐)", "DPM++ 2M", "LCM (快速)", "DDIM"])
        self.combo_video_sched.setMinimumHeight(32)
        g.addWidget(self.combo_video_sched, 3, 1, 1, 4)
        v_lay.addLayout(g)

        # 快捷时长
        dur_row = QHBoxLayout()
        dur_row.setSpacing(6)
        dur_row.addWidget(field("快捷时长:"))
        for sec in (2, 4, 6, 8, 10):
            b = pill(f"{sec}秒", 50)
            b.clicked.connect(lambda _=False, s=sec: self._set_video_duration(s))
            dur_row.addWidget(b)
        dur_row.addStretch()
        v_lay.addLayout(dur_row)

        v_lay.addWidget(hint("建议 8–12 FPS：更高更流畅，但显存与耗时线性增加。"))

        # 长视频模式
        self.chk_long_video = QCheckBox("🎬 长视频模式 (>32 帧)")
        self.chk_long_video.setToolTip("启用 Context Window，帧数上限扩至 150")
        self.chk_long_video.toggled.connect(self._on_long_video_toggled)
        v_lay.addWidget(self.chk_long_video)

        root.addWidget(grp_video)

        # ============ 🎭 Motion LoRA ============
        grp_lora = group("🎭 Motion LoRA (可多选)")
        l_lay = QVBoxLayout(grp_lora)
        l_lay.setSpacing(6)

        add_row = QHBoxLayout()
        add_row.setSpacing(8)
        self.cmb_motion_lora_pick = QComboBox()
        self.cmb_motion_lora_pick.setMinimumHeight(32)
        self.cmb_motion_lora_pick.addItem("-- 选择 Motion LoRA --")
        for name in self._scan_motion_loras():
            self.cmb_motion_lora_pick.addItem(name)

        btn_add_lora = pill("➕ 添加", 78)
        btn_add_lora.clicked.connect(self._add_motion_lora_item)
        add_row.addWidget(self.cmb_motion_lora_pick, 1)
        add_row.addWidget(btn_add_lora)
        l_lay.addLayout(add_row)

        self.motion_lora_container = QVBoxLayout()
        self.motion_lora_container.setSpacing(4)
        l_lay.addLayout(self.motion_lora_container)
        self.motion_lora_items = []

        self.lbl_motion_lora_hint = hint(
            "未检测到 LoRA，请将模型放入 models/motion_lora/"
            if self.cmb_motion_lora_pick.count() <= 1 else
            "可叠加多个运镜 LoRA（如 ZoomIn / PanLeft），权重建议 ≤ 0.8。"
        )
        l_lay.addWidget(self.lbl_motion_lora_hint)
        root.addWidget(grp_lora)

        # ============ 🎙️ 配音 ============
        grp_voice = group("🎙️ 配音 (可选)")
        vo_lay = QVBoxLayout(grp_voice)
        vo_lay.setSpacing(6)

        self.chk_video_voice = QCheckBox("为视频添加配音")
        self.chk_video_voice.setToolTip("生成完成后自动合成语音并合并进视频")
        vo_lay.addWidget(self.chk_video_voice)

        # 引擎
        eng_row = QHBoxLayout()
        eng_row.setSpacing(8)
        eng_row.addWidget(field("引擎:", 72))
        self.combo_tts_engine = QComboBox()
        self.combo_tts_engine.addItems(["ChatTTS (中文)", "GPT-SoVITS (日语)"])
        self.combo_tts_engine.setMinimumHeight(32)
        eng_row.addWidget(self.combo_tts_engine, 1)
        vo_lay.addLayout(eng_row)

        vo_lay.addWidget(field("配音文本:", 0))
        self.txt_video_voice = QTextEdit()
        self.txt_video_voice.setFixedHeight(76)
        self.txt_video_voice.setPlaceholderText("旁白文字，例如：清晨的阳光洒在草地上，一只小猫追逐着蝴蝶。")
        vo_lay.addWidget(self.txt_video_voice)

        # -- ChatTTS --
        self.wrap_chattts = QWidget()
        c_lay = QVBoxLayout(self.wrap_chattts)
        c_lay.setContentsMargins(0, 0, 0, 0)
        r = QHBoxLayout()
        r.setSpacing(6)
        r.addWidget(field("说话人 Seed:", 92))
        self.spin_video_voice_seed = spin(0, 999999, 2222, 1, 92)
        r.addWidget(self.spin_video_voice_seed)
        for txt, sd in (("👨 男1", 2222), ("👨 男2", 7869), ("👩 女1", 1983), ("👩 女2", 4099)):
            b = pill(txt, 62)
            b.clicked.connect(lambda _=False, v=sd: self.spin_video_voice_seed.setValue(v))
            r.addWidget(b)
        r.addStretch()
        c_lay.addLayout(r)
        vo_lay.addWidget(self.wrap_chattts)

        # -- GPT-SoVITS --
        self.wrap_sovits = QWidget()
        s_lay = QVBoxLayout(self.wrap_sovits)
        s_lay.setContentsMargins(0, 0, 0, 0)
        s_lay.setSpacing(6)

        r = QHBoxLayout()
        r.setSpacing(8)
        r.addWidget(field("参考音频:", 92))
        self.combo_sovits_ref = QComboBox()
        self.combo_sovits_ref.setMinimumHeight(32)
        self.combo_sovits_ref.addItems(["默认女声 (Nanami)"])
        r.addWidget(self.combo_sovits_ref, 1)
        btn_pick_ref = pill("📂 自定义", 88)
        btn_pick_ref.clicked.connect(self._on_pick_sovits_ref)
        r.addWidget(btn_pick_ref)
        s_lay.addLayout(r)

        r = QHBoxLayout()
        r.setSpacing(8)
        lbl_rt = field("参考文本:", 92)
        lbl_rt.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        r.addWidget(lbl_rt)
        self.txt_sovits_reftext = QTextEdit()
        self.txt_sovits_reftext.setFixedHeight(50)
        self.txt_sovits_reftext.setPlaceholderText("参考音频对应文字（留空使用默认）")
        r.addWidget(self.txt_sovits_reftext, 1)
        s_lay.addLayout(r)

        r = QHBoxLayout()
        r.setSpacing(8)
        r.addWidget(field("语速:", 92))
        self.spin_sovits_speed = dspin(0.5, 2.0, 1.0, 0.05, 2, 84)
        r.addWidget(self.spin_sovits_speed)
        self.chk_sovits_auto_translate = QCheckBox("自动中 → 日翻译")
        self.chk_sovits_auto_translate.setChecked(True)
        r.addWidget(self.chk_sovits_auto_translate)
        r.addStretch()
        s_lay.addLayout(r)

        self.wrap_sovits.setVisible(False)
        vo_lay.addWidget(self.wrap_sovits)

        self.lbl_voice_hint = hint("首次使用会自动下载 ChatTTS 模型（约 1.1 GB）。")
        vo_lay.addWidget(self.lbl_voice_hint)
        root.addWidget(grp_voice)

        # ============ ✨ 后处理 ============
        grp_post = group("✨ 后处理 (可选)")
        po_lay = QVBoxLayout(grp_post)
        po_lay.setSpacing(6)

        r = QHBoxLayout()
        r.setSpacing(8)
        self.chk_frame_interp = QCheckBox("帧插值 (RIFE)")
        self.chk_frame_interp.setToolTip("补帧让动作更连贯，不改变时长（提高 FPS）")
        r.addWidget(self.chk_frame_interp)
        self.combo_frame_interp = QComboBox()
        self.combo_frame_interp.addItems(["2x", "4x", "8x"])
        self.combo_frame_interp.setEnabled(False)
        self.combo_frame_interp.setMinimumWidth(72)
        r.addWidget(self.combo_frame_interp)
        r.addWidget(hint("使视频更流畅"))
        r.addStretch()
        po_lay.addLayout(r)

        r = QHBoxLayout()
        r.setSpacing(8)
        self.chk_video_upscale = QCheckBox("🔍 视频放大 (Real-ESRGAN)")
        self.chk_video_upscale.setToolTip("512 → 1024 / 2048，显著增加耗时")
        r.addWidget(self.chk_video_upscale)
        self.combo_upscale_factor = QComboBox()
        self.combo_upscale_factor.addItems(["2x", "4x"])
        self.combo_upscale_factor.setEnabled(False)
        self.combo_upscale_factor.setMinimumWidth(72)
        r.addWidget(self.combo_upscale_factor)
        r.addStretch()
        po_lay.addLayout(r)

        self.chk_frame_interp.toggled.connect(self.combo_frame_interp.setEnabled)
        self.chk_video_upscale.toggled.connect(self.combo_upscale_factor.setEnabled)
        root.addWidget(grp_post)

        # ============ 💾 输出设置 ============
        grp_out = group("💾 输出设置")
        o_lay = QVBoxLayout(grp_out)
        r = QHBoxLayout()
        r.setSpacing(8)
        r.addWidget(field("格式:", 72))
        self.combo_video_fmt = QComboBox()
        self.combo_video_fmt.addItems(["MP4", "GIF", "MP4 + GIF"])
        self.combo_video_fmt.setMinimumHeight(32)
        r.addWidget(self.combo_video_fmt, 1)
        r.addStretch()
        o_lay.addLayout(r)
        o_lay.addWidget(hint("MP4 适合分享与二次剪辑，GIF 适合社交媒体。"))
        root.addWidget(grp_out)

        # ============ 🎬 生成 ============
        self.btn_gen_video = QPushButton("🎬 生成视频")
        self.btn_gen_video.setObjectName("btnGenVideo")
        self.btn_gen_video.setMinimumHeight(48)
        self.btn_gen_video.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_gen_video.clicked.connect(self.on_generate_video)
        root.addSpacing(6)
        root.addWidget(self.btn_gen_video)

        self.lbl_video_status = QLabel("💤 待命中 — 设置参数后点击生成")
        self.lbl_video_status.setProperty("role", "hint")
        self.lbl_video_status.setWordWrap(True)
        root.addWidget(self.lbl_video_status)
        root.addStretch()

        # ---------------- 信号联动（集中管理） ----------------
        self.combo_video_mode.currentIndexChanged.connect(self._on_video_mode_changed)
        self.btn_enhance_video_prompt.clicked.connect(self.on_enhance_video_prompt)
        self.btn_vision_video_prompt.clicked.connect(self.on_vision_video_prompt)
        self.btn_enhance_travel.clicked.connect(self.on_enhance_travel_prompts)
        self.spin_video_frames.valueChanged.connect(self._update_video_duration_hint)
        self.spin_video_fps.valueChanged.connect(self._update_video_duration_hint)
        self.combo_tts_engine.currentIndexChanged.connect(self._on_tts_engine_changed)
        for wdg in (self.txt_video_voice, self.combo_tts_engine,
                    self.wrap_chattts, self.wrap_sovits):
            self.chk_video_voice.toggled.connect(wdg.setEnabled)
            wdg.setEnabled(False)

        # ---------------- 一次性套用样式并初始化状态 ----------------
        w.setStyleSheet(VIDEO_TAB_QSS)
        self._on_video_mode_changed(self.combo_video_mode.currentIndex())
        self._update_video_duration_hint()

        # 初始化旅行分段（必须在 spin_video_frames 创建之后）
        self._add_travel_segment()
        self._add_travel_segment()

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
        self.txt_log_image = QTextEdit()
        self.txt_log_image.setReadOnly(True)
        self.txt_log_image.setMaximumHeight(140)
        self.txt_log_image.setStyleSheet(
            "background:#191919; font-family:Consolas; font-size:11px;")
        layout.addWidget(self.txt_log_image, 1)
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
        self.btn_video_pause = QPushButton("⏯️ 暂停")
        self.btn_video_stop = QPushButton("⏹️ 停止")
        for b in (self.btn_video_save, self.btn_video_refresh,
                  self.btn_video_pause, self.btn_video_stop):
            b.setStyleSheet(
                "background:#0a0a0a; color:#dadbdf; padding:6px; "
                "border:1px solid #212327; border-radius:9999px; font-size:11px;"
            )
            video_btn_row.addWidget(b)
        self.btn_video_save.clicked.connect(self._save_current_video)
        self.btn_video_refresh.clicked.connect(self._refresh_video_gallery)
        self.btn_video_pause.clicked.connect(self.pause_video)
        self.btn_video_stop.clicked.connect(self.stop_video)
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
        video_splitter.setSizes([400, 350])  # 预览:画廊 = ~1:1 可拖拽
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
        self.txt_log_video = QTextEdit()
        self.txt_log_video.setReadOnly(True)
        self.txt_log_video.setMaximumHeight(140)
        self.txt_log_video.setStyleSheet(
            "background:#191919; font-family:Consolas; font-size:11px;")
        layout.addWidget(self.txt_log_video, 1)

        return w

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
            # 播放结束 → 停在最后一帧（不再自动循环）
            self.video_player.pause()
            if hasattr(self, 'video_stacked'):
                self.video_stacked.setCurrentIndex(1)
        elif status == QMediaPlayer.MediaStatus.LoadedMedia:
            pass
        elif status == QMediaPlayer.MediaStatus.NoMedia:
            if hasattr(self, 'video_stacked'):
                self.video_stacked.setCurrentIndex(0)
            else:
                self.lbl_video_placeholder.show()
                self.video_widget.hide()

    def stop_video(self):
        """停止当前视频播放"""
        if hasattr(self, 'video_player') and self.video_player:
            self.video_player.stop()
        if hasattr(self, 'video_stacked'):
            self.video_stacked.setCurrentIndex(0)
        elif hasattr(self, 'lbl_video_placeholder'):
            self.lbl_video_placeholder.show()
            if hasattr(self, 'video_widget'):
                self.video_widget.hide()

    def pause_video(self):
        """暂停/恢复当前视频"""
        if not hasattr(self, 'video_player') or not self.video_player:
            return
        if self.video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.video_player.pause()
        else:
            self.video_player.play()

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
        a_clear_log.triggered.connect(lambda: (
            getattr(self, 'txt_log_image', None) and self.txt_log_image.clear(),
            getattr(self, 'txt_log_video', None) and self.txt_log_video.clear()))
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
    def _on_long_video_toggled(self, checked: bool):
        """长视频模式：勾选后帧数上限扩展至 150，不勾选恢复 80"""
        if not hasattr(self, 'spin_video_frames'):
            return
        if checked:
            self.spin_video_frames.setRange(8, 150)
            if self.spin_video_frames.value() <= 80:
                self.spin_video_frames.setValue(64)  # 建议默认 64+ 帧
        else:
            self.spin_video_frames.setRange(8, 80)
            if self.spin_video_frames.value() > 80:
                self.spin_video_frames.setValue(16)

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
        html = (
            f'<span style="color:{color}; font-family:Consolas;">'
            f'{text}</span>'
        )
        for attr in ("txt_log_image", "txt_log_video"):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.append(html)
                sb = widget.verticalScrollBar()
                sb.setValue(sb.maximum())

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

    def _on_travel_edit_mode_changed(self, idx: int):
        """切换旅行编辑方式：分段编辑 / 文本格式"""
        self.wrap_travel_segments.setVisible(idx == 0)
        self.wrap_travel_text.setVisible(idx == 1)

    def _spread_travel_frames(self):
        """均匀分布旅行分段帧号（调用已有方法，加空列表保护）"""
        if not self.travel_segments:
            return
        self._auto_distribute_frames()
        self._set_status("✅ 已均匀分布旅行分段帧号", "#dadbdf")

    def _scan_motion_loras(self):
        """扫描 models/motion_lora 目录"""
        import os
        result = []
        lora_dir = "models/motion_lora"
        try:
            if os.path.isdir(lora_dir):
                for d in sorted(os.listdir(lora_dir)):
                    if os.path.isdir(os.path.join(lora_dir, d)):
                        result.append(d)
        except Exception:
            pass
        return result

    def _update_video_duration_hint(self):
        """更新视频时长提示标签"""
        if not hasattr(self, 'lbl_video_duration'):
            return
        try:
            frames = self.spin_video_frames.value()
            fps = self.spin_video_fps.value()
            sec = frames / max(fps, 1)
            self.lbl_video_duration.setText(f"≈ {sec:.1f} 秒")
        except Exception:
            self.lbl_video_duration.setText("—")

    def _clear_video_input(self):
        """清除已选视频/图片输入文件"""
        self._video_input_path = None
        self.lbl_video_input.setText("未选择文件")
        self._set_status("🗑️ 已清除输入文件", "#dadbdf")


    def on_pick_video_input(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "选择首帧图/输入视频",
            "", "图片/视频 (*.png *.jpg *.jpeg *.mp4 *.gif)"
        )
        if path:
            self._video_input_path = path
            self.lbl_video_input.setText(os.path.basename(path))


    
    def _on_tts_engine_changed(self, idx):
        """引擎切换"""
        is_sovits = (idx == 1)
        self.wrap_chattts.setVisible(not is_sovits)
        self.wrap_sovits.setVisible(is_sovits)
        if is_sovits:
            self.lbl_voice_hint.setText("首次使用会加载 GPT-SoVITS (~2GB 显存,常驻)")
            self.txt_video_voice.setPlaceholderText("输入中文或日文,例如:今日はとても楽しかったです")
        else:
            self.lbl_voice_hint.setText("首次使用会自动下载 ChatTTS 模型 (~1.1GB)")
            self.txt_video_voice.setPlaceholderText("输入要配音的旁白文字,例如:清晨的阳光洒在草地上")

    def _on_pick_sovits_ref(self):
        """选择自定义参考音频"""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "选择参考音频 (3-10秒)", "", "音频文件 (*.wav *.mp3 *.flac)"
        )
        if path:
            name = os.path.basename(path)
            # 移除旧的自定义项(如果有)
            for i in range(self.combo_sovits_ref.count() - 1, 0, -1):
                if self.combo_sovits_ref.itemText(i).startswith("🎵 "):
                    self.combo_sovits_ref.removeItem(i)
            self.combo_sovits_ref.addItem(f"🎵 {name}", path)
            self.combo_sovits_ref.setCurrentIndex(self.combo_sovits_ref.count() - 1)


    # ==========================================================
    #  菜单栏 + 扩展计数状态栏 (v5.0 扩展市场)
    # ==========================================================
    def _setup_menu_and_statusbar(self):
        """初始化菜单栏 + 状态栏扩展计数"""
        from PyQt6.QtGui import QAction
        from PyQt6.QtWidgets import QLabel

        # ---- 菜单栏 ----
        menubar = self.menuBar()
        tools_menu = menubar.addMenu("🛠️ 工具")

        act_market = QAction("🛒 扩展市场...", self)
        act_market.setShortcut("Ctrl+E")
        act_market.triggered.connect(self._open_extension_market)
        tools_menu.addAction(act_market)

        tools_menu.addSeparator()

        act_refresh = QAction("🔄 刷新扩展状态", self)
        act_refresh.triggered.connect(self._refresh_extension_count)
        tools_menu.addAction(act_refresh)

        # ---- 状态栏扩展计数 ----
        self.lbl_ext_count = QLabel()
        self.lbl_ext_count.setStyleSheet(
            "color:#8ab4ff; padding:2px 10px; font-size:12px;"
        )
        self.statusBar().addPermanentWidget(self.lbl_ext_count)
        self._refresh_extension_count()

    def _refresh_extension_count(self):
        """刷新状态栏的扩展计数"""
        try:
            from utils.extension_manager import get_status_summary
            s = get_status_summary()
            self.lbl_ext_count.setText(f"🧩 扩展: {s['installed']}/{s['total']}")
        except Exception as e:
            self.lbl_ext_count.setText("🧩 扩展: --")
            print(f"[EXT-COUNT] 刷新失败: {e}")

    def _open_extension_market(self):
        """打开扩展市场对话框"""
        try:
            from ui.extension_market import ExtensionMarketDialog
            dlg = ExtensionMarketDialog(self)
            dlg.exec()
            self._refresh_extension_count()
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"打开扩展市场失败:\n{e}")