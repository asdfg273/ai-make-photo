# ui/settings_dialog.py
# ============================================================
#  设置对话框 — 快捷键改键 + 组件默认值
#  只读写 AppConfig；应用动作由宿主（shell._open_settings）执行
# ============================================================
import logging
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QTabWidget, QWidget, QLabel, QPushButton,
                             QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                             QKeySequenceEdit, QDialogButtonBox, QGroupBox, QLineEdit)
from PyQt6.QtGui import QKeySequence

from core.config_manager import DEFAULT_SHORTCUTS, SHORTCUT_LABELS
from ui.i18n import LANGUAGE_ORDER, SUPPORTED_LANGUAGES, ui_i18n

logger = logging.getLogger(__name__)

RES_OPTIONS = ["512x512", "512x768", "768x512", "768x768",
               "1024x1024", "832x1216", "1216x832"]
SAMPLER_OPTIONS = ["DPM++ 2M Karras", "DPM++ SDE Karras",
                   "Euler a", "Euler", "DDIM", "UniPC"]
TRANS_MODES = ["纯词典", "AI 智能改写", "词典优先 + AI 兜底"]


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
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.tabs.addTab(w, "常规")

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

        grp_dit = QGroupBox("Flux / SD3 低显存推理")
        f = QFormLayout(grp_dit)
        self.combo_dit_quant = QComboBox()
        self.combo_dit_quant.setProperty("i18nItems", True)
        self.combo_dit_quant.addItem("关闭", "off")
        self.combo_dit_quant.addItem("FP8 权重量化（torchao）", "fp8")
        self.combo_dit_quant.addItem("NVFP4（NVIDIA ModelOpt）", "nvfp4")
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
        v.addWidget(grp_dit)

        grp_cache = QGroupBox("TeaCache 风格步间缓存")
        fc = QFormLayout(grp_cache)
        self.chk_step_cache = QCheckBox("启用步间缓存（Flux / SDXL）")
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
                      "NVFP4 需要兼容 GPU 和 ModelOpt；修改量化后下次生成会重载模型。"
                      "语义搜索只读取本地目录或已有缓存，不会自动下载模型。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addStretch()
        self.tabs.addTab(w, "推理性能")

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
        cfg.step_cache_enabled = self.chk_step_cache.isChecked()
        cfg.step_cache_threshold = float(self.spin_step_cache_threshold.value())
        cfg.step_cache_max_skips = self.spin_step_cache_max_skips.value()
        cfg.clip_model_path = self.txt_clip_model_path.text().strip()

        cfg.save()
        ui_i18n.set_language(cfg.ui_language)
        logger.info(f"设置已保存，界面语言: {cfg.ui_language}")
        super().accept()
