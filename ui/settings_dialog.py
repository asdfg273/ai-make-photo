# ui/settings_dialog.py
# ============================================================
#  设置对话框 — 快捷键改键 + 组件默认值
#  只读写 AppConfig；应用动作由宿主（shell._open_settings）执行
# ============================================================
import logging
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QHBoxLayout,
                             QTabWidget, QWidget, QLabel, QPushButton,
                             QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                             QKeySequenceEdit, QDialogButtonBox, QGroupBox)
from PyQt6.QtGui import QKeySequence

from core.config_manager import DEFAULT_SHORTCUTS, SHORTCUT_LABELS

logger = logging.getLogger(__name__)

RES_OPTIONS = ["512x512", "512x768", "768x512", "768x768",
               "1024x1024", "832x1216", "1216x832"]
SAMPLER_OPTIONS = ["DPM++ 2M Karras", "DPM++ SDE Karras",
                   "Euler a", "Euler", "DDIM", "UniPC"]
TRANS_MODES = [" 纯词典", "AI 智能改写", " 词典优先 + AI 兜底 "]


class SettingsDialog(QDialog):
    """host 需提供 .config (AppConfig)。"""

    def __init__(self, host):
        super().__init__(host)
        self._host = host
        self.cfg = host.config
        self.setWindowTitle("⚙️ 设置")
        self.resize(520, 560)

        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_shortcut_tab()
        self._build_defaults_tab()

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

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

        hint = QLabel("💡 点击输入框后按下新键位；清空即禁用该快捷键。\n"
                      "修改点「确定」后立即生效。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        form.addRow(hint)

        btn_reset = QPushButton("↩️ 恢复默认键位")
        btn_reset.clicked.connect(self._reset_shortcuts)
        form.addRow(btn_reset)

        self.tabs.addTab(w, "⌨️ 快捷键")

    def _reset_shortcuts(self):
        for action_id, edit in self.seq_edits.items():
            edit.setKeySequence(QKeySequence(DEFAULT_SHORTCUTS[action_id]))

    # ============ Tab 2: 默认参数 ============
    def _build_defaults_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        cfg = self.cfg

        # ── 生成默认值 ──
        grp_gen = QGroupBox("🎨 生成默认值")
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
        grp_fix = QGroupBox("✨ 精修默认开关")
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
        v.addWidget(grp_fix)

        # ── AI / 设备 ──
        grp_ai = QGroupBox("🧠 AI 与设备")
        f3 = QFormLayout(grp_ai)

        self.combo_trans = QComboBox()
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

        hint = QLabel("💡 点「确定」后立即写入配置并应用到当前界面；"
                      "设备偏好在下次加载模型时生效。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addStretch()

        self.tabs.addTab(w, "🎛️ 默认参数")

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
        cfg.default_trans_mode = self.combo_trans.currentIndex()
        if self.combo_qwen.currentData():
            cfg.qwen_model_key = self.combo_qwen.currentData()
        cfg.device_preference = self.combo_device.currentText().lower()

        cfg.save()
        logger.info("⚙️ 设置已保存")
        super().accept()
