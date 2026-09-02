# ui/pages/img2img_page.py
# ============================================================
#  图生图页 — 专属区：参考图/蒙版/强度/IP-Adapter/Pose Transfer
#  从 ui_builder._build_tab_img2img（995-1171 行）迁入，属性名不变
# ============================================================
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QComboBox, QCheckBox,
                             QSlider, QDoubleSpinBox, QGroupBox)
from PyQt6.QtCore import Qt

from ui.pages.base import PageBase
from ui.widgets import FloatSlider
from ui.core_panel import _wire

logger = logging.getLogger(__name__)


class Img2ImgPage(PageBase):
    page_id, title, icon = "img2img", "图生图", "🖼️"

    def build(self, host):
        self._host = host
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # ---------- 参考图 ----------
        grp_i2i = QGroupBox("参考图 (img2img / inpaint)")
        gi = QVBoxLayout(grp_i2i)

        btn_row = QHBoxLayout()
        host.btn_load_img = QPushButton("📂 加载参考图")
        _wire(host, host.btn_load_img.clicked, "select_image")
        host.btn_clear_img = QPushButton("🗑 清除")
        _wire(host, host.btn_clear_img.clicked, "clear_reference")
        btn_row.addWidget(host.btn_load_img)
        btn_row.addWidget(host.btn_clear_img)
        gi.addLayout(btn_row)

        host.lbl_img_path = QLabel("未选择参考图")
        host.lbl_img_path.setProperty("role", "hint")
        gi.addWidget(host.lbl_img_path)

        host.lbl_ref_thumb = QLabel("无参考图")
        host.lbl_ref_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        host.lbl_ref_thumb.setFixedHeight(120)
        host.lbl_ref_thumb.setProperty("role", "hint")
        gi.addWidget(host.lbl_ref_thumb)

        gi.addWidget(QLabel("重绘强度 (Denoise):"))
        host.scale_strength = FloatSlider(0.05, 1.0, 0.05, 0.6)
        gi.addWidget(host.scale_strength)

        layout.addWidget(grp_i2i)

        # ---------- IP-Adapter ----------
        grp_ipa = QGroupBox("🎭 IP-Adapter — 角色一致性")
        g_ipa = QGridLayout(grp_ipa)

        host.chk_use_ipa = QCheckBox("启用 IP-Adapter (锁定角色样貌)")
        g_ipa.addWidget(host.chk_use_ipa, 0, 0, 1, 4)

        btn_load_ipa = QPushButton("📷 加载角色参考图")
        _wire(host, btn_load_ipa.clicked, "load_ipa_image")
        g_ipa.addWidget(btn_load_ipa, 1, 0)

        host.lbl_ipa_image = QLabel("未选择")
        host.lbl_ipa_image.setProperty("role", "hint")
        g_ipa.addWidget(host.lbl_ipa_image, 1, 1, 1, 3)

        g_ipa.addWidget(QLabel("影响力:"), 2, 0)
        host.spin_ipa_scale = QDoubleSpinBox()
        host.spin_ipa_scale.setRange(0.0, 1.5)
        host.spin_ipa_scale.setSingleStep(0.05)
        host.spin_ipa_scale.setValue(0.6)
        host.spin_ipa_scale.setDecimals(2)
        g_ipa.addWidget(host.spin_ipa_scale, 2, 1)

        g_ipa.addWidget(QLabel("版本:"), 2, 2)
        host.combo_ipa_variant = QComboBox()
        host.combo_ipa_variant.addItems(["plus (推荐)", "standard (轻量)"])
        g_ipa.addWidget(host.combo_ipa_variant, 2, 3)

        layout.addWidget(grp_ipa)

        # ---------- Pose Transfer ----------
        grp_pt = QGroupBox("🎬 Pose Transfer — 智能姿势迁移 (推荐)")
        g_pt = QVBoxLayout(grp_pt)

        host.chk_pose_transfer = QCheckBox("启用 Pose Transfer (3 阶段流水线)")
        host.chk_pose_transfer.setToolTip(
            "🎬 自动 3 阶段流水线:\n"
            "1️⃣ 用提示词生成动作参考图\n"
            "2️⃣ 自动提取 OpenPose 骨架\n"
            "3️⃣ 骨架(锁动作) + 角色图(锁角色) → 最终图\n\n"
            "✅ 完美解决「图生图看不懂提示词」问题\n"
            "⚠️ 需要在上方上传 IP-Adapter 角色参考图\n"
            "⏱ 总耗时约普通生成的 1.5~2 倍")
        _wire(host, host.chk_pose_transfer.toggled, "_on_pose_transfer_toggled")
        g_pt.addWidget(host.chk_pose_transfer)

        host.lbl_pt_tip = QLabel(
            "💡 启用后会自动:\n"
            "   • 强制开启 IP-Adapter (用上方角色图锁人物)\n"
            "   • 强制使用 OpenPose ControlNet (锁动作)\n"
            "   • 忽略「重绘强度」(走 ControlNet 通道)")
        host.lbl_pt_tip.setProperty("role", "hint")
        host.lbl_pt_tip.setWordWrap(True)
        g_pt.addWidget(host.lbl_pt_tip)

        row_cn = QHBoxLayout()
        row_cn.addWidget(QLabel("姿势约束强度:"))
        host.slider_pt_cn = QSlider(Qt.Orientation.Horizontal)
        host.slider_pt_cn.setRange(30, 120)   # 0.30 ~ 1.20
        host.slider_pt_cn.setValue(65)        # 默认 0.65
        host.slider_pt_cn.setFixedWidth(220)
        host.lbl_pt_cn = QLabel("0.65")
        host.lbl_pt_cn.setFixedWidth(50)
        host.slider_pt_cn.valueChanged.connect(
            lambda v: host.lbl_pt_cn.setText(f"{v/100:.2f}"))
        row_cn.addWidget(host.slider_pt_cn)
        row_cn.addWidget(host.lbl_pt_cn)
        row_cn.addStretch()

        hint_cn = QLabel("(越低 = 越像角色; 越高 = 越像动作)")
        hint_cn.setProperty("role", "hint")
        g_pt.addLayout(row_cn)
        g_pt.addWidget(hint_cn)

        def _toggle_pt(checked):
            host.slider_pt_cn.setEnabled(checked)
            host.lbl_pt_cn.setEnabled(checked)
        host.chk_pose_transfer.toggled.connect(_toggle_pt)
        _toggle_pt(False)  # 初始禁用

        layout.addWidget(grp_pt)

        # ---------- 单图角色一致性增强 ----------
        g_consist = QGroupBox("🎯 单图角色一致性增强")
        v_consist = QVBoxLayout(g_consist)

        host.chk_auto_features = QCheckBox(
            " 自动提取角色特征 (Qwen 识别发色/瞳色/兽耳并注入 prompt)")
        host.chk_auto_features.setChecked(True)
        host.chk_auto_features.setToolTip(
            "启用后,生成前会用 Qwen2-VL 分析参考图,\n"
            "自动提取发色/瞳色/兽耳/服装等关键特征,\n"
            "并以最高权重注入 prompt 最前端。\n"
            "✅ 单图角色一致性必备")
        v_consist.addWidget(host.chk_auto_features)

        host.chk_reference_only = QCheckBox(
            "🪞 启用 Reference-Only (锁定参考图细节,与 Pose 互斥)")
        host.chk_reference_only.setChecked(False)
        v_consist.addWidget(host.chk_reference_only)

        row_ref = QHBoxLayout()
        row_ref.addWidget(QLabel("参考强度:"))
        host.scale_ref_fidelity = QSlider(Qt.Orientation.Horizontal)
        host.scale_ref_fidelity.setRange(50, 100)   # 0.50 ~ 1.00
        host.scale_ref_fidelity.setValue(70)
        host.scale_ref_fidelity.setFixedWidth(220)
        host.lbl_ref_fidelity = QLabel("0.70")
        host.lbl_ref_fidelity.setFixedWidth(50)
        host.scale_ref_fidelity.valueChanged.connect(
            lambda v: host.lbl_ref_fidelity.setText(f"{v/100:.2f}"))
        row_ref.addWidget(host.scale_ref_fidelity)
        row_ref.addWidget(host.lbl_ref_fidelity)
        row_ref.addStretch()
        v_consist.addLayout(row_ref)

        hint_ref = QLabel("(0.50=自由发挥, 0.70=平衡推荐, 1.00=完全复刻)")
        hint_ref.setProperty("role", "hint")
        v_consist.addWidget(hint_ref)

        layout.addWidget(g_consist)
        layout.addStretch()

        self._params = w

    def workspace(self) -> QWidget:
        # 与文生图共享中央预览区（lbl_preview 全局单例）
        return self._host._pages["txt2img"].workspace() \
            if "txt2img" in getattr(self._host, "_pages", {}) else QWidget()

    def params_widget(self) -> QWidget:
        return self._params
