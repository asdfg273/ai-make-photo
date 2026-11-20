# ui/settings_dialog.py
# ============================================================
#  设置对话框 — 快捷键改键 + 组件默认值
#  只读写 AppConfig；应用动作由宿主（shell._open_settings）执行
# ============================================================
import logging
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget, QWidget,
                             QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                             QKeySequenceEdit, QDialogButtonBox, QGroupBox, QLineEdit, QMessageBox)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence

from core.config_manager import DEFAULT_SHORTCUTS, SHORTCUT_LABELS
from ui.i18n import LANGUAGE_ORDER, SUPPORTED_LANGUAGES, ui_i18n

logger = logging.getLogger(__name__)

RES_OPTIONS = ["512x512", "512x768", "768x512", "768x768",
               "1024x1024", "832x1216", "1216x832"]
from core.samplers import sampler_names, sigma_styles  # noqa: E402

SAMPLER_OPTIONS = list(sampler_names())
SIGMA_OPTIONS = list(sigma_styles())
TRANS_MODES = ["纯词典", "AI 智能改写", "词典优先 + AI 兜底"]

#: 设置页那个「点这里」按钮打开的页面。别问，问就是经典曲目。
EASTER_EGG_URL = "https://www.bilibili.com/video/BV1GJ411x7h7"


class SettingsDialog(QDialog):
    """host 需提供 .config (AppConfig)。"""

    def __init__(self, host):
        super().__init__(host)
        self._host = host
        self.cfg = host.config
        self.setWindowTitle("设置")
        self.resize(560, 600)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_general_tab()
        self._build_shortcut_tab()
        self._build_defaults_tab()
        self._build_performance_tab()

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)
        ui_i18n.translate_widget_tree(self)

    # ============ Tab 0: 常规 ============
    def _build_general_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(12, 16, 12, 12)
        form.setSpacing(12)

        grp = QGroupBox("界面与语言")
        gl = QFormLayout(grp)
        self.combo_ui_language = QComboBox()
        for code in LANGUAGE_ORDER:
            self.combo_ui_language.addItem(SUPPORTED_LANGUAGES[code], code)
        current = ui_i18n.normalize(getattr(self.cfg, "ui_language", "zh"))
        self.combo_ui_language.setCurrentIndex(
            max(0, self.combo_ui_language.findData(current)))
        gl.addRow("界面语言:", self.combo_ui_language)
        form.addRow(grp)

        hint = QLabel("选择界面使用的语言。点击「确定」后立即生效，并保存到本地配置。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        form.addRow(hint)

        # 彩蛋：一个不解释的按钮
        grp_egg = QGroupBox("其他")
        egg_row = QHBoxLayout(grp_egg)
        egg_row.setContentsMargins(10, 8, 10, 8)
        egg_lbl = QLabel("这里好像藏了点什么。")
        egg_lbl.setProperty("role", "hint")
        egg_lbl.setWordWrap(True)
        self.btn_click_here = QPushButton("点这里")
        self.btn_click_here.setToolTip("在浏览器里打开一个视频（真的只是视频）")
        self.btn_click_here.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_click_here.clicked.connect(self._open_easter_egg)
        egg_row.addWidget(egg_lbl, 1)
        egg_row.addWidget(self.btn_click_here, 0)
        form.addRow(grp_egg)

        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.tabs.addTab(w, "常规")

    def _open_easter_egg(self):
        """用系统默认浏览器打开经典曲目；打不开就把链接摆出来让人自己点。"""
        try:
            opened = QDesktopServices.openUrl(QUrl(EASTER_EGG_URL))
        except Exception as exc:  # noqa: BLE001 - 没有浏览器/没有默认程序
            logger.warning("打开彩蛋链接失败: %s", exc)
            opened = False
        if opened:
            logger.info("彩蛋：已在浏览器中打开 %s", EASTER_EGG_URL)
            return
        box = QMessageBox(self)
        box.setWindowTitle("没能自动打开浏览器")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("系统没有自动打开浏览器，你可以手动访问：")
        box.setInformativeText(EASTER_EGG_URL)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.exec()

    # ============ Tab 1: 快捷键 ============
    def _build_shortcut_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(10)

        self.seq_edits = {}
        current = dict(DEFAULT_SHORTCUTS)
        current.update(getattr(self.cfg, "shortcuts", None) or {})
        for action_id in DEFAULT_SHORTCUTS:      # 固定顺序
            edit = QKeySequenceEdit(QKeySequence(current.get(action_id, "")))
            edit.setClearButtonEnabled(True)
            form.addRow(SHORTCUT_LABELS.get(action_id, action_id), edit)
            self.seq_edits[action_id] = edit

        hint = QLabel("点击输入框后按下新键位；清空即禁用该快捷键。\n"
                      "修改点「确定」后立即生效。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        form.addRow(hint)

        btn_reset = QPushButton("恢复默认键位")
        btn_reset.clicked.connect(self._reset_shortcuts)
        form.addRow(btn_reset)

        self.tabs.addTab(w, "快捷键")

    def _reset_shortcuts(self):
        for action_id, edit in self.seq_edits.items():
            edit.setKeySequence(QKeySequence(DEFAULT_SHORTCUTS[action_id]))

    # ============ Tab 2: 默认参数 ============
    def _build_defaults_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        cfg = self.cfg

        # ── 生成默认值 ──
        grp_gen = QGroupBox("生成默认值")
        f = QFormLayout(grp_gen)

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 150)
        self.spin_steps.setValue(int(getattr(cfg, "default_steps", 30)))
        f.addRow("步数 Steps:", self.spin_steps)

        self.spin_cfg = QDoubleSpinBox()
        self.spin_cfg.setRange(1.0, 30.0)
        self.spin_cfg.setSingleStep(0.5)
        self.spin_cfg.setValue(float(getattr(cfg, "default_cfg", 7.0)))
        f.addRow("CFG Scale:", self.spin_cfg)

        self.combo_res = QComboBox()
        self.combo_res.addItems(RES_OPTIONS)
        res = f"{getattr(cfg, 'default_width', 512)}x{getattr(cfg, 'default_height', 768)}"
        idx = self.combo_res.findText(res)
        self.combo_res.setCurrentIndex(idx if idx >= 0 else 0)
        f.addRow("分辨率:", self.combo_res)

        self.combo_sampler = QComboBox()
        self.combo_sampler.addItems(SAMPLER_OPTIONS)
        idx = self.combo_sampler.findText(getattr(cfg, "default_sampler", ""))
        self.combo_sampler.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_scheduler = QComboBox()
        self.combo_scheduler.addItems(SIGMA_OPTIONS)
        idx_s = self.combo_scheduler.findText(
            getattr(cfg, "sampler_scheduler", "默认") or "默认")
        self.combo_scheduler.setCurrentIndex(idx_s if idx_s >= 0 else 0)
        f.addRow("sigma 风格:", self.combo_scheduler)
        f.addRow("采样器:", self.combo_sampler)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 32)
        self.spin_batch.setValue(int(getattr(cfg, "default_batch", 1)))
        f.addRow("生成数量:", self.spin_batch)

        self.spin_strength = QDoubleSpinBox()
        self.spin_strength.setRange(0.05, 1.0)
        self.spin_strength.setSingleStep(0.05)
        self.spin_strength.setValue(float(getattr(cfg, "default_strength", 0.6)))
        f.addRow("图生图重绘强度:", self.spin_strength)
        v.addWidget(grp_gen)

        # ── 精修默认开关 ──
        grp_fix = QGroupBox("精修默认开关")
        f2 = QFormLayout(grp_fix)
        self.chk_hires = QCheckBox("默认开启 Hires.fix")
        self.chk_hires.setChecked(bool(getattr(cfg, "use_hires", False)))
        f2.addRow(self.chk_hires)
        self.chk_adetailer = QCheckBox("默认开启 ADetailer 脸部精修")
        self.chk_adetailer.setChecked(bool(getattr(cfg, "use_adetailer", False)))
        f2.addRow(self.chk_adetailer)
        self.chk_ad_hand = QCheckBox("默认开启 ADetailer 手部精修")
        self.chk_ad_hand.setChecked(bool(getattr(cfg, "use_ad_hand", False)))
        f2.addRow(self.chk_ad_hand)
        self.chk_freeu = QCheckBox("默认开启 FreeU 细节增强")
        self.chk_freeu.setChecked(bool(getattr(cfg, "freeu_enabled", False)))
        f2.addRow(self.chk_freeu)
        v.addWidget(grp_fix)

        # ── AI / 设备 ──
        grp_ai = QGroupBox("AI 与设备")
        f3 = QFormLayout(grp_ai)

        self.combo_trans = QComboBox()
        self.combo_trans.setProperty("i18nItems", True)  # 业务层读取索引
        self.combo_trans.addItems(TRANS_MODES)
        self.combo_trans.setCurrentIndex(
            max(0, min(2, int(getattr(cfg, "default_trans_mode", 2)))))
        f3.addRow("默认翻译模式:", self.combo_trans)

        self.combo_qwen = QComboBox()
        try:
            from utils.prompt_enhancer import PromptEnhancer
            for _k, _c in PromptEnhancer.MODEL_REGISTRY.items():
                self.combo_qwen.addItem(_c["label"], _k)
        except Exception:
            pass
        idx = self.combo_qwen.findData(getattr(cfg, "qwen_model_key", "qwen2vl_2b"))
        if idx >= 0:
            self.combo_qwen.setCurrentIndex(idx)
        f3.addRow("AI 模型档位:", self.combo_qwen)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["AUTO", "CUDA", "MPS", "CPU"])
        dev = str(getattr(cfg, "device_preference", "auto")).upper()
        idx = self.combo_device.findText(dev)
        self.combo_device.setCurrentIndex(idx if idx >= 0 else 0)
        f3.addRow("默认设备:", self.combo_device)
        v.addWidget(grp_ai)

        hint = QLabel("点「确定」后立即写入配置并应用到当前界面；"
                      "设备偏好在下次加载模型时生效。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addStretch()

        self.tabs.addTab(w, "默认参数")

    # ============ Tab 3: 推理性能 ============
    def _build_performance_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        cfg = self.cfg

        grp_dit = QGroupBox("Flux / SD3 / MOSS-Image 低显存推理")
        f = QFormLayout(grp_dit)
        self.combo_dit_quant = QComboBox()
        self.combo_dit_quant.setProperty("i18nItems", True)
        self.combo_dit_quant.addItem("关闭", "off")
        self.combo_dit_quant.addItem("8bit（bitsandbytes，8GB 显示卡推荐）", "bnb8")
        self.combo_dit_quant.addItem("FP8 权重量化（torchao）", "fp8")
        self.combo_dit_quant.addItem("NVFP4（NVIDIA ModelOpt）", "nvfp4")
        self.combo_dit_quant.setToolTip(
            "MOSS-Image 的 DiT 是 9.7GB bf16，8GB 显示卡装不下，只能每步把权重\n"
            "从内存搬到显卡（慢 2~3 倍，任务管理器里会看到「共享 GPU 内存」很高）。\n"
            "选 8bit 后权重缩到约 5.2GB，可以整块常驻显存 —— 又快又省内存。\n"
            "只作用于 transformer（DiT），不影响 VAE 与文本编码器。\n"
            "注意：MOSS-Image 的自定义 transformer 不支持量化（会把激活当 int8，\n"
            "出图报 silu_cuda 未实现），选了也会按原精度加载；8bit 对 Flux / SD3 有效。")
        idx = self.combo_dit_quant.findData(
            getattr(cfg, "dit_quantization", "off"))
        self.combo_dit_quant.setCurrentIndex(max(0, idx))
        f.addRow("DiT 量化:", self.combo_dit_quant)

        self.chk_dit_group_offload = QCheckBox("启用 DiT 分块 CPU offload")
        self.chk_dit_group_offload.setChecked(
            bool(getattr(cfg, "dit_group_offload", True)))
        f.addRow(self.chk_dit_group_offload)
        self.spin_dit_offload_blocks = QSpinBox()
        self.spin_dit_offload_blocks.setRange(1, 32)
        self.spin_dit_offload_blocks.setValue(
            int(getattr(cfg, "dit_offload_blocks", 1)))
        f.addRow("每组 block 数:", self.spin_dit_offload_blocks)

        # MOSS-Image 8bit 常驻显存（bitsandbytes）：整条链路（补丁 + 量化 + 显存摆放）
        # 已实现，但在本机（bnb 0.49.2 + torch 2.11 + RTX 5060 / sm_120）采样阶段会
        # 掉进 bitsandbytes 的兜底实现并报 Char 相关错误，暂不开放界面开关。
        # 需要试的人可以手动在 app_config.json 里写 "moss_accel": true，
        # 加载器会先自检、不满足就自动退回 bf16 并打日志。
        self.combo_moss_quant = QComboBox()
        self.combo_moss_quant.addItem("关闭（原样 bf16）", "off")
        self.combo_moss_quant.addItem("int8 只压 DiT（快、显存紧）", "int8")
        self.combo_moss_quant.addItem("int8 压 DiT + 文本编码器（最省内存）", "int8te")
        self.combo_moss_quant.setToolTip(
            "只对 MOSS-Image 生效。用纯 torch 的 int8 权重量化，不依赖 bitsandbytes。\n\n"
            "关闭：DiT 9.5GB 装不进 8GB 显存，只能在内存里、每步搬到显卡 ——\n"
            "      内存峰值实测 23.8GB（1024²/20 步，同种子对比）。\n"
            "int8（只压 DiT）：9.5GB → 4.7GB 常驻显存，速度反而快 16%（130 → 110 秒），\n"
            "      内存峰值 17.4GB。\n"
            "int8 + 文本编码器：文本编码器 16GB → 6.9GB，内存峰值降到 11.0GB ——\n"
            "      代价是换提示词时那一步文本编码变慢（154 秒/张 vs 110 秒）。\n\n"
            "提示词不变、连续出图时，编码只做一次，选「int8 + 文本编码器」几乎不亏。")
        cur = str(getattr(cfg, "moss_quant", "off") or "off").lower()
        idx = self.combo_moss_quant.findData("int8te" if cur in ("int8te", "int8-te") else cur)
        self.combo_moss_quant.setCurrentIndex(idx if idx >= 0 else 0)
        f.addRow("MOSS-Image 量化:", self.combo_moss_quant)
        v.addWidget(grp_dit)

        grp_cache = QGroupBox("TeaCache 风格步间缓存")
        fc = QFormLayout(grp_cache)
        self.chk_step_cache = QCheckBox("启用步间缓存（任何 UNet / DiT 架构）")
        self.chk_step_cache.setToolTip(
            "相邻采样步的输入足够接近时，直接复用上一步的 UNet/Transformer\n"
            "输出，跳过这次计算。阈值越大越省、但越容易出现细节糊掉。\n"
            "生成结束后日志里会写明「N 步里复用了几步」，0 步命中说明阈值偏小。")
        self.chk_step_cache.setChecked(bool(getattr(cfg, "step_cache_enabled", False)))
        fc.addRow(self.chk_step_cache)
        self.spin_step_cache_threshold = QDoubleSpinBox()
        self.spin_step_cache_threshold.setRange(0.01, 1.0)
        self.spin_step_cache_threshold.setSingleStep(0.01)
        self.spin_step_cache_threshold.setDecimals(2)
        self.spin_step_cache_threshold.setValue(
            float(getattr(cfg, "step_cache_threshold", 0.18)))
        fc.addRow("复用阈值:", self.spin_step_cache_threshold)
        self.spin_step_cache_max_skips = QSpinBox()
        self.spin_step_cache_max_skips.setRange(1, 4)
        self.spin_step_cache_max_skips.setValue(
            int(getattr(cfg, "step_cache_max_skips", 2)))
        fc.addRow("最多连续复用:", self.spin_step_cache_max_skips)

        grp_py = QGroupBox("高级（能执行任意代码，默认关闭）")
        fp = QFormLayout(grp_py)
        self.chk_allow_python = QCheckBox(
            "允许「任意 Python 节点」（会执行任意代码，只用于你自己写的工作流）")
        self.chk_allow_python.setToolTip(
            "打开后，工作流里的 PythonScript 节点会真的执行你写的 Python 代码。\n"
            "这是全程序唯一能执行任意代码的开关。\n\n"
            "风险：工作流文件不再可移植 —— 别人给你的工作流（含从 ComfyUI 导入的）\n"
            "可能带任意代码，打开就跑；节点写错也不会被预检发现。\n\n"
            "节点参数里的「允许 import」关掉时，会禁用 open / exec / eval / compile\n"
            "这些顺手就能读写文件的入口，但**那不是沙箱**（Python 层面绕得过去）。\n"
            "默认关闭。")
        self.chk_allow_python.setChecked(
            bool(getattr(cfg, "allow_python_nodes", False)))
        fp.addRow(self.chk_allow_python)
        v.addWidget(grp_py)

        # ── 命中率：把"勾了但一步没省"直接摆出来 ──
        # 只写"已启用"是不够的：用户勾了加速、看到 enabled=true，却感觉没变快，
        # 只会怀疑功能坏了。这里读的是真实的 computed/reused 计数。
        self.lbl_step_cache_stats = QLabel("")
        self.lbl_step_cache_stats.setWordWrap(True)
        fc.addRow("当前命中率:", self.lbl_step_cache_stats)
        self.btn_step_cache_refresh = QPushButton("刷新命中率")
        self.btn_step_cache_refresh.setToolTip(
            "生成一张图之后点这里：会显示这次采样里真正复用了几步。\n"
            "显示「0 步命中」说明阈值太小，把上面的「复用阈值」调大。")
        self.btn_step_cache_refresh.clicked.connect(
            self._refresh_step_cache_stats)
        fc.addRow("", self.btn_step_cache_refresh)
        self._refresh_step_cache_stats()
        v.addWidget(grp_cache)

        grp_clip = QGroupBox("画廊语义搜索")
        fclip = QFormLayout(grp_clip)
        self.txt_clip_model_path = QLineEdit(
            str(getattr(cfg, "clip_model_path", "") or ""))
        self.txt_clip_model_path.setPlaceholderText(
            "留空使用 Hugging Face 缓存，或填写本地 CLIP 目录")
        fclip.addRow("本地 CLIP 目录:", self.txt_clip_model_path)
        v.addWidget(grp_clip)

        hint = QLabel("量化与缓存会改变推理结果或性能，建议先用固定种子对比。"
                      "8bit 用 bitsandbytes（本项目环境已装），FP8 需要 torchao，"
                      "NVFP4 需要兼容 GPU 和 ModelOpt；修改量化后下次生成会重载模型。"
                      "语义搜索只读取本地目录或已有缓存，不会自动下载模型。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addStretch()
        self.tabs.addTab(w, "推理性能")

    def _refresh_step_cache_stats(self):
        """把真实的复用/重算计数写进设置页（一眼看出"0 命中"）。"""
        lbl = getattr(self, "lbl_step_cache_stats", None)
        if lbl is None:
            return
        try:
            from utils.step_cache import step_cache_stats, step_cache_status_text
            st = step_cache_stats()
            text = step_cache_status_text()
        except Exception as exc:
            lbl.setText(f"读取命中率失败：{exc}")
            lbl.setStyleSheet("color: #f38ba8;")
            return
        lbl.setText(text)
        if not st.get("enabled"):
            color = "#7f849c"
        elif st.get("calls") and not st.get("reused"):
            color = "#f38ba8"        # 已启用但 0 命中 → 红
        elif st.get("reused"):
            color = "#a6e3a1"        # 有命中 → 绿
        else:
            color = "#f9e2af"        # 启用了但还没采样 → 黄
        lbl.setStyleSheet(f"color: {color};")

    # ============ 收集结果 ============
    def accept(self):
        cfg = self.cfg

        # 快捷键
        cfg.shortcuts = {aid: edit.keySequence().toString()
                         for aid, edit in self.seq_edits.items()}

        # 默认参数
        cfg.default_steps = self.spin_steps.value()
        cfg.default_cfg = float(self.spin_cfg.value())
        w, h = self.combo_res.currentText().split("x", 1)
        cfg.default_width, cfg.default_height = int(w), int(h)
        cfg.default_sampler = self.combo_sampler.currentText()
        cfg.sampler_scheduler = self.combo_scheduler.currentText()
        cfg.default_batch = self.spin_batch.value()
        cfg.default_strength = float(self.spin_strength.value())
        cfg.use_hires = self.chk_hires.isChecked()
        cfg.use_adetailer = self.chk_adetailer.isChecked()
        cfg.use_ad_hand = self.chk_ad_hand.isChecked()
        cfg.freeu_enabled = self.chk_freeu.isChecked()
        cfg.default_trans_mode = self.combo_trans.currentIndex()
        if self.combo_qwen.currentData():
            cfg.qwen_model_key = self.combo_qwen.currentData()
        cfg.device_preference = self.combo_device.currentText().lower()
        cfg.ui_language = self.combo_ui_language.currentData()
        cfg.dit_quantization = self.combo_dit_quant.currentData()
        cfg.dit_group_offload = self.chk_dit_group_offload.isChecked()
        cfg.dit_offload_blocks = self.spin_dit_offload_blocks.value()
        cfg.moss_quant = str(self.combo_moss_quant.currentData() or "off")
        cfg.step_cache_enabled = self.chk_step_cache.isChecked()
        cfg.step_cache_threshold = float(self.spin_step_cache_threshold.value())
        cfg.step_cache_max_skips = self.spin_step_cache_max_skips.value()
        cfg.clip_model_path = self.txt_clip_model_path.text().strip()
        _want_py = self.chk_allow_python.isChecked()
        if _want_py and not getattr(cfg, "allow_python_nodes", False):
            # 从"关"变"开"必须再问一次：这是全程序唯一能执行任意代码的开关
            from PyQt6.QtWidgets import QMessageBox
            _ans = QMessageBox.warning(
                self, "确认打开任意 Python 节点",
                "打开后，工作流里的「任意 Python 节点」会执行你写的 Python 代码。\n\n"
                "· 别人给你的工作流文件（包括从 ComfyUI 导入的）可能带任意代码，打开就跑；\n"
                "· 这类节点不受预检与缓存保护，写错会直接报错；\n"
                "· 工作流文件将不再可移植；\n"
                "· 节点参数里的「允许 import」只是禁用 open/exec/eval 等入口，不是沙箱。\n\n"
                "只在你自己的、可信的工作流里使用。确定打开吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            _want_py = _ans == QMessageBox.StandardButton.Yes
            self.chk_allow_python.setChecked(_want_py)
        cfg.allow_python_nodes = _want_py

        cfg.save()
        ui_i18n.set_language(cfg.ui_language)
        logger.info(f"设置已保存，界面语言: {cfg.ui_language}")
        super().accept()
