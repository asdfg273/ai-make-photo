# ui/shared_groups.py
# ============================================================
#  共享折叠分组 — LoRA / ControlNet / 高级 / X-Y 矩阵
#  从 ui_builder 的 _build_tab_lora/_ctrl/_advanced/_xy 迁入
#  属性名不变；默认全部折叠；动画页时整体隐藏（shell 控制）
# ============================================================
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLabel, QPushButton, QComboBox, QCheckBox,
                             QTextEdit, QLineEdit, QSpinBox, QGroupBox)

from ui.widgets import FloatSlider, GpuCanvas
from ui.components.collapsible import CollapsibleSection
from ui.core_panel import _wire

logger = logging.getLogger(__name__)


def build_shared_groups(host, layout: QVBoxLayout) -> dict:
    """构建 4 个共享折叠分组，返回 {key: CollapsibleSection}。"""
    sections = {}
    sections["lora"] = _build_lora(host)
    sections["ctrl"] = _build_ctrl(host)
    sections["advanced"] = _build_advanced(host)
    sections["xy"] = _build_xy(host)
    for sec in sections.values():
        layout.addWidget(sec)
    return sections


# ============================================================
#  LoRA（原 _build_tab_lora）
# ============================================================
def _build_lora(host) -> CollapsibleSection:
    sec = CollapsibleSection("🧩 LoRA", collapsed=True)
    layout = sec.content_layout

    hdr = QHBoxLayout()
    hdr.addWidget(QLabel("LoRA 槽位 (按主模型架构自动过滤):"))
    host.btn_refresh_lora = QPushButton("🔄 刷新")
    host.btn_refresh_lora.setFixedWidth(70)
    _wire(host, host.btn_refresh_lora.clicked, "refresh_lora_by_model")
    hdr.addWidget(host.btn_refresh_lora)
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
        setattr(host, f'combo_lora_{i}', combo)
        setattr(host, f'scale_lora_{i}', scale)
        _wire(host, combo.currentIndexChanged, "load_lora_info")

    layout.addWidget(QLabel("LoRA 备忘录:"))
    host.text_lora_info = QTextEdit()
    host.text_lora_info.setReadOnly(True)
    host.text_lora_info.setFixedHeight(120)
    layout.addWidget(host.text_lora_info)

    btn_row = QHBoxLayout()
    host.btn_insert_lora_all = QPushButton("📋 插入全部触发词")
    host.btn_insert_lora_all.clicked.connect(
        lambda: host._insert_lora_triggers(None))
    btn_row.addWidget(host.btn_insert_lora_all)

    for i in range(3):
        btn = QPushButton(f"槽{i+1}")
        btn.setFixedWidth(45)
        btn.clicked.connect(
            lambda _, idx=i: host._insert_lora_triggers(idx))
        btn_row.addWidget(btn)

    btn_row.addStretch()
    layout.addLayout(btn_row)
    return sec


# ============================================================
#  ControlNet（原 _build_tab_ctrl）
# ============================================================
def _build_ctrl(host) -> CollapsibleSection:
    sec = CollapsibleSection("🕹 ControlNet", collapsed=True)
    layout = sec.content_layout

    grp = QGroupBox("ControlNet — 手动模式")
    gv = QFormLayout(grp)

    host.chk_use_pose = QCheckBox("开启 ControlNet")
    _wire(host, host.chk_use_pose.toggled, "_toggle_cn")
    gv.addRow(host.chk_use_pose)

    host.combo_cn_type = QComboBox()
    host.combo_cn_type.addItems([
        "OpenPose", "Canny", "Depth", "Scribble", "SoftEdge"])
    gv.addRow("类型:", host.combo_cn_type)

    host.scale_cn_strength = FloatSlider(0.0, 2.0, 0.05, 1.0)
    gv.addRow("条件强度:", host.scale_cn_strength)
    host.scale_cn_weight = host.scale_cn_strength   # 兼容老名

    host.btn_load_cn_img = QPushButton("📂 加载姿态图")
    _wire(host, host.btn_load_cn_img.clicked, "load_pose_image")
    gv.addRow(host.btn_load_cn_img)

    host.lbl_pose_path = QLabel("未加载动作图")
    host.lbl_pose_path.setProperty("role", "hint")
    gv.addRow(host.lbl_pose_path)

    host.lbl_cn_thumb = GpuCanvas()
    host.lbl_cn_thumb.setText("未加载")
    host.lbl_cn_thumb.setFixedHeight(180)
    gv.addRow(host.lbl_cn_thumb)
    layout.addWidget(grp)

    tip = QLabel(
        "💡 提示: 如果想用「提示词→自动生成动作」,\n"
        "   请到 [图生图] 页启用 🎬 Pose Transfer。")
    tip.setProperty("role", "hint")
    tip.setWordWrap(True)
    layout.addWidget(tip)
    return sec


# ============================================================
#  高级（原 _build_tab_advanced）
# ============================================================
def _build_advanced(host) -> CollapsibleSection:
    sec = CollapsibleSection("⚙️ 高级选项", collapsed=True)
    layout = sec.content_layout

    # ---------- 修脸 ----------
    grp_face = QGroupBox("ADetailer — 修脸")
    gf = QFormLayout(grp_face)
    host.chk_use_adetailer = QCheckBox("开启修脸")
    _wire(host, host.chk_use_adetailer.toggled, "_toggle_adetailer")
    gf.addRow(host.chk_use_adetailer)

    host.combo_adetailer_model = QComboBox()
    host.combo_adetailer_model.addItems(["真人脸", "二次元脸"])
    gf.addRow("脸部类型:", host.combo_adetailer_model)

    host.combo_ad_target = QComboBox()
    host.combo_ad_target.addItems(["现实脸部", "二次元脸部"])
    gf.addRow("检测目标:", host.combo_ad_target)

    host.lbl_ad_str = QLabel("修复强度:")
    host.scale_adetailer_strength = FloatSlider(0.1, 0.9, 0.05, 0.35)
    gf.addRow(host.lbl_ad_str, host.scale_adetailer_strength)
    layout.addWidget(grp_face)

    # ---------- 修手 ----------
    grp_hand = QGroupBox("ADetailer — 修手")
    gh = QFormLayout(grp_hand)
    host.chk_use_ad_hand = QCheckBox("开启修手")
    _wire(host, host.chk_use_ad_hand.toggled, "_toggle_ad_hand")
    gh.addRow(host.chk_use_ad_hand)

    host.combo_ad_hand = QComboBox()
    host.combo_ad_hand.addItems(["现实手部", "二次元手部"])
    gh.addRow("检测目标:", host.combo_ad_hand)

    host.lbl_ad_hand_str = QLabel("重绘强度:")
    host.scale_ad_hand = FloatSlider(0.1, 0.6, 0.05, 0.25)
    gh.addRow(host.lbl_ad_hand_str, host.scale_ad_hand)

    host.lbl_ad_hand_blend = QLabel("融合度:")
    host.scale_ad_hand_blend = FloatSlider(0.0, 1.0, 0.05, 0.65)
    gh.addRow(host.lbl_ad_hand_blend, host.scale_ad_hand_blend)
    layout.addWidget(grp_hand)

    # ---------- FreeU ----------
    host.chk_freeu = QCheckBox("✨ FreeU 细节增强")
    host.chk_freeu.setToolTip(
        "免训练的细节增强技术，提升图像清晰度和纹理细节\n"
        "SD1.5 / SDXL 系可用；FLUX / SD3 架构不支持")
    _wire(host, host.chk_freeu.toggled, "_toggle_freeu")
    layout.addWidget(host.chk_freeu)

    # ---------- CLIP Skip ----------
    host.combo_clip_skip = QComboBox()
    host.combo_clip_skip.addItems(["关闭 (最后一层)", "跳过 1 层 (推荐动漫)",
                                   "跳过 2 层", "跳过 3 层"])
    host.combo_clip_skip.setCurrentIndex(0)
    host.combo_clip_skip.setToolTip(
        "🎚️ CLIP Skip —— 文本编码器用倒数第 N 层输出。\n\n"
        "• Pony / Illustrious / NoobAI 以及多数 SD1.5 二次元底模\n"
        "  都是在【倒数第二层】上训练的 → 选「跳过 1 层」。\n"
        "• 用默认的最后一层常见症状：线条发脏、色彩发灰、脸更容易崩。\n"
        "• 真实系底模一般保持「关闭」。\n"
        "• 仅对 CLIP 系生效（SD1.5 / SDXL）；FLUX/SD3/Anima 会自动跳过。\n"
        "• 切换后在下一次生成时生效；X/Y 矩阵可用它对比效果。")
    _wire(host, host.combo_clip_skip.currentIndexChanged, "_on_clip_skip_changed")
    grp_cs = QGroupBox("🎚️ CLIP Skip — 文本编码层")
    gcs = QFormLayout(grp_cs)
    gcs.addRow("跳过层数:", host.combo_clip_skip)
    layout.addWidget(grp_cs)

    # ---------- Hires.fix ----------
    grp_hr = QGroupBox("Hires.fix — 高清修复")
    ghr = QFormLayout(grp_hr)

    host.chk_hires = QCheckBox("开启 Hires.fix")
    _wire(host, host.chk_hires.toggled, "_toggle_hires")
    ghr.addRow(host.chk_hires)

    host.chk_enable_hires = QCheckBox("XY 矩阵中也启用 Hires.fix")
    ghr.addRow(host.chk_enable_hires)

    host.combo_hires_scale = QComboBox()
    host.combo_hires_scale.addItems(["1.5", "2.0", "2.5", "3.0"])
    host.combo_hires_scale.setCurrentText("2.0")
    ghr.addRow("放大倍率:", host.combo_hires_scale)

    host.scale_hires_denoise = FloatSlider(0.1, 0.9, 0.05, 0.45)
    ghr.addRow("降噪强度:", host.scale_hires_denoise)

    host.combo_hires_upscaler = QComboBox()
    host.combo_hires_upscaler.addItems([
        "Latent", "ESRGAN_4x", "R-ESRGAN 4x+", "SwinIR"])
    ghr.addRow("Upscaler:", host.combo_hires_upscaler)
    layout.addWidget(grp_hr)

    # ---------- 大图生成 ----------
    grp_photo = QGroupBox("🖼️ 大图生成 (Tiled Diffusion)")
    fl_tiled = QFormLayout(grp_photo)

    host.chk_use_tiled = QCheckBox("启用大图生成(对当前图后处理)")
    host.chk_use_tiled.setToolTip(
        "Tiled Diffusion: 将大图分块生成后融合\n"
        "突破显存限制，可出 2048-4096 分辨率\n"
        "⚠️ CPU 用户慎用：一张 2K 图约需 2-4 小时")
    fl_tiled.addRow(host.chk_use_tiled)

    size_row = QHBoxLayout()
    host.spin_tiled_w = QSpinBox()
    host.spin_tiled_w.setRange(768, 8192)
    host.spin_tiled_w.setSingleStep(64)
    host.spin_tiled_w.setValue(2048)
    host.spin_tiled_h = QSpinBox()
    host.spin_tiled_h.setRange(768, 8192)
    host.spin_tiled_h.setSingleStep(64)
    host.spin_tiled_h.setValue(2048)
    size_row.addWidget(host.spin_tiled_w)
    size_row.addWidget(QLabel("×"))
    size_row.addWidget(host.spin_tiled_h)
    wrap = QWidget()
    wrap.setLayout(size_row)
    fl_tiled.addRow("目标分辨率:", wrap)

    host.combo_tile_size = QComboBox()
    host.combo_tile_size.addItems(["512", "640", "768", "1024"])
    host.combo_tile_size.setCurrentText("768")
    host.combo_tile_size.setToolTip("单块大小，越大越慢但接缝越少")
    fl_tiled.addRow("Tile 大小:", host.combo_tile_size)

    host.spin_tile_overlap = QSpinBox()
    host.spin_tile_overlap.setRange(32, 256)
    host.spin_tile_overlap.setSingleStep(16)
    host.spin_tile_overlap.setValue(96)
    host.spin_tile_overlap.setToolTip("重叠像素，消接缝必需，建议 64-128")
    fl_tiled.addRow("Tile 重叠:", host.spin_tile_overlap)

    host.scale_tile_strength = FloatSlider(0.2, 0.8, 0.05, 0.4)
    host.scale_tile_strength.setToolTip(
        "0.3-0.4: 仅放大细化(推荐)\n0.5-0.6: 中度重绘\n0.7+: 大幅改变原图")
    fl_tiled.addRow("重绘强度:", host.scale_tile_strength)

    host.btn_run_tiled = QPushButton("🚀 对最后一张图执行大图生成")
    _wire(host, host.btn_run_tiled.clicked, "run_tiled_diffusion")
    fl_tiled.addRow(host.btn_run_tiled)
    layout.addWidget(grp_photo)

    # ---------- 输出 ----------
    grp_out = QGroupBox("输出设置")
    go = QFormLayout(grp_out)
    host.combo_output_dir = QComboBox()
    host.combo_output_dir.setEditable(True)
    # 默认显示真实目录（photo/），不再用占位值 "outputs/"：
    # 那个值既与实际目录不符，又会被回写进配置，造成"设置不生效"的观感。
    from utils.paths import default_output_dir
    host.combo_output_dir.addItem(default_output_dir())
    go.addRow("输出目录:", host.combo_output_dir)

    host.combo_img_format = QComboBox()
    host.combo_img_format.addItems(["PNG", "JPEG", "WEBP"])
    go.addRow("图片格式:", host.combo_img_format)
    layout.addWidget(grp_out)

    # ---------- 水印 ----------
    grp_wm = QGroupBox("💧 水印")
    gw = QFormLayout(grp_wm)

    host.chk_watermark = QCheckBox("给生成图加水印")
    host.chk_watermark.setToolTip(
        "启用水印后，所有生成图（含放大/大图）都会加上水印。\n"
        "XY 参数矩阵是诊断图，不会加水印。")
    gw.addRow(host.chk_watermark)

    host.edit_watermark_text = QLineEdit()
    host.edit_watermark_text.setPlaceholderText("文字水印，例如：© 我的名字")
    gw.addRow("文字:", host.edit_watermark_text)

    wm_row = QHBoxLayout()
    host.btn_watermark_image = QPushButton("选择图片…")
    host.btn_watermark_image.setToolTip(
        "选择 PNG/JPG 作为图片水印（建议带透明通道）。\n"
        "同时配置了文字和图片时，优先使用图片；图片读取失败会自动退回文字。")
    _wire(host, host.btn_watermark_image.clicked, "_pick_watermark_image")
    host.lbl_watermark_image = QLabel("未选择")
    host.lbl_watermark_image.setStyleSheet("color:#7f849c;")
    host.btn_watermark_clear = QPushButton("清除")
    _wire(host, host.btn_watermark_clear.clicked, "_clear_watermark_image")
    wm_row.addWidget(host.btn_watermark_image)
    wm_row.addWidget(host.lbl_watermark_image, 1)
    wm_row.addWidget(host.btn_watermark_clear)
    wm_wrap = QWidget()
    wm_wrap.setLayout(wm_row)
    gw.addRow("图片:", wm_wrap)

    host.combo_watermark_pos = QComboBox()
    from utils.watermark import POSITIONS
    host.combo_watermark_pos.addItems(list(POSITIONS))
    gw.addRow("位置:", host.combo_watermark_pos)

    host.scale_watermark_opacity = FloatSlider(0.05, 1.0, 0.05, 0.6)
    gw.addRow("透明度:", host.scale_watermark_opacity)

    host.scale_watermark_size = FloatSlider(0.3, 3.0, 0.1, 1.0)
    gw.addRow("大小:", host.scale_watermark_size)

    host.lbl_watermark_hint = QLabel(
        "提示：水印会烧进位图，导出后无法移除，建议保留一份无水印原图。")
    host.lbl_watermark_hint.setWordWrap(True)
    host.lbl_watermark_hint.setStyleSheet("color:#7f849c; font-size:11px;")
    gw.addRow(host.lbl_watermark_hint)
    layout.addWidget(grp_wm)

    host.btn_read_png = QPushButton("📥 读取 PNG 中的生成参数")
    _wire(host, host.btn_read_png.clicked, "read_png_info")
    layout.addWidget(host.btn_read_png)
    return sec


# ============================================================
#  X/Y 矩阵（原 _build_tab_xy）
# ============================================================
def _build_xy(host) -> CollapsibleSection:
    sec = CollapsibleSection("📊 X/Y 矩阵", collapsed=True)
    layout = sec.content_layout

    grp = QGroupBox("X/Y 矩阵生成")
    gv = QFormLayout(grp)

    host.chk_enable_xy = QCheckBox("开启 X/Y 矩阵")
    _wire(host, host.chk_enable_xy.toggled, "_toggle_xy")
    gv.addRow(host.chk_enable_xy)

    host.combo_x_type = QComboBox()
    host.combo_x_type.addItems([
        "Steps", "CFG Scale", "Sampler", "Seed", "LoRA 权重"])
    gv.addRow("X 轴类型:", host.combo_x_type)
    host.entry_x_vals = QLineEdit()
    host.entry_x_vals.setPlaceholderText("如: 10,20,30 或 7,9,11")
    gv.addRow("X 轴值:", host.entry_x_vals)

    host.combo_y_type = QComboBox()
    host.combo_y_type.addItems([
        "Steps", "CFG Scale", "Sampler", "Seed", "LoRA 权重"])
    gv.addRow("Y 轴类型:", host.combo_y_type)
    host.entry_y_vals = QLineEdit()
    host.entry_y_vals.setPlaceholderText("如: 0.4,0.6,0.8")
    gv.addRow("Y 轴值:", host.entry_y_vals)

    layout.addWidget(grp)
    return sec
