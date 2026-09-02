# ui/pages/video_page.py
# ============================================================
#  动画页 — 专属区：动画参数组（原 _build_tab_animation）
#          中央工作区：视频预览 + 视频历史（原 _build_video_right_panel）
#  属性名不变；样式改用语义 property（theme.py 统一）
# ============================================================
import logging
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QComboBox, QCheckBox,
                             QTextEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QScrollArea, QFrame, QListWidget, QStackedWidget,
                             QSplitter)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from ui.pages.base import PageBase
from ui.core_panel import _wire

logger = logging.getLogger(__name__)

LABEL_W = 88


def _field(text: str, w: int = LABEL_W) -> QLabel:
    lb = QLabel(text)
    lb.setProperty("role", "field")
    if w:
        lb.setMinimumWidth(w)
    return lb


def _hint(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setProperty("role", "hint")
    lb.setWordWrap(True)
    return lb


def _group(title: str, accent: bool = False) -> QGroupBox:
    g = QGroupBox(title)
    if accent:
        g.setProperty("accent", True)
    return g


def _spin(lo, hi, val, step=1, w=96) -> QSpinBox:
    sp = QSpinBox()
    sp.setRange(lo, hi)
    sp.setSingleStep(step)
    sp.setValue(val)
    sp.setMinimumWidth(w)
    sp.setMinimumHeight(32)
    sp.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return sp


def _dspin(lo, hi, val, step=0.5, dec=2, w=104) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setSingleStep(step)
    sp.setDecimals(dec)
    sp.setValue(val)
    sp.setMinimumWidth(w)
    sp.setMinimumHeight(32)
    sp.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return sp


def _pill(text: str, w: int | None = None) -> QPushButton:
    b = QPushButton(text)
    b.setProperty("role", "pill")
    b.setMinimumHeight(30)
    if w:
        b.setMinimumWidth(w)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def _grid() -> QGridLayout:
    g = QGridLayout()
    g.setHorizontalSpacing(10)
    g.setVerticalSpacing(8)
    g.setContentsMargins(0, 0, 0, 0)
    return g


class VideoPage(PageBase):
    page_id, title, icon = "video", "动画", "🎬"

    def build(self, host):
        self._host = host
        self._params = self._build_params(host)
        self._workspace = self._build_workspace(host)

    def workspace(self) -> QWidget:
        return self._workspace

    def params_widget(self) -> QWidget:
        return self._params

    # ========================================================
    #  专属参数区（原 _build_tab_animation）
    # ========================================================
    def _build_params(self, host) -> QWidget:
        w = QWidget()
        w.setObjectName("animRoot")
        root = QVBoxLayout(w)
        root.setSpacing(4)
        root.setContentsMargins(12, 8, 12, 16)

        # ============ 💡 使用提示 ============
        grp_tips = _group("💡 使用提示")
        tips_lay = QVBoxLayout(grp_tips)
        tips = QLabel(
            "<ul style='margin:0; padding-left:18px; line-height:150%;'>"
            "<li><b>文生视频</b>：仅用提示词生成，无需输入文件</li>"
            "<li><b>图生视频</b>：选一张图作首帧，AI 延续动画</li>"
            "<li><b>视频转绘</b>：选视频文件，AI 改变画风</li>"
            "<li><b>提示词旅行</b>：不同帧用不同提示词，做剧情变化</li>"
            "</ul>")
        tips.setProperty("role", "body")
        tips.setWordWrap(True)
        tips_lay.addWidget(tips)
        root.addWidget(grp_tips)

        # ============ 🎯 生成模式 ============
        grp_mode = _group("🎯 生成模式", accent=True)
        mode_lay = QVBoxLayout(grp_mode)
        host.combo_video_mode = QComboBox()
        host.combo_video_mode.addItems([
            "📝 文生视频 (txt2video)",
            "🖼️ 图生视频 (img2video) — 首帧引导",
            "🎞️ 视频转绘 (vid2vid) — 改画风",
            "✨ 提示词旅行 (Prompt Travel) — 剧情视频",
        ])
        host.combo_video_mode.setMinimumHeight(36)
        mode_lay.addWidget(host.combo_video_mode)
        host.lbl_video_mode_desc = _hint("无需输入文件，直接填写提示词即可生成。")
        mode_lay.addWidget(host.lbl_video_mode_desc)
        root.addWidget(grp_mode)

        # ============ 📥 输入文件 ============
        host.grp_video_input = _group("📥 输入文件")
        in_lay = QVBoxLayout(host.grp_video_input)
        in_lay.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(8)
        host.lbl_video_input = QLabel("未选择文件")
        host.lbl_video_input.setProperty("role", "hint")
        host.lbl_video_input.setMinimumHeight(34)
        host.lbl_video_input.setToolTip("未选择文件")
        row.addWidget(host.lbl_video_input, 1)

        host.btn_pick_video_input = _pill("📂 选择", 84)
        _wire(host, host.btn_pick_video_input.clicked, "on_pick_video_input")
        row.addWidget(host.btn_pick_video_input)

        host.btn_clear_video_input = _pill("✖", 40)
        host.btn_clear_video_input.setToolTip("清除已选文件")
        _wire(host, host.btn_clear_video_input.clicked, "_clear_video_input")
        row.addWidget(host.btn_clear_video_input)
        in_lay.addLayout(row)

        in_lay.addWidget(_hint("图生视频 → 选择首帧图片；视频转绘 → 选择输入视频。"))
        root.addWidget(host.grp_video_input)

        # ============ 💬 提示词 ============
        grp_prompt = _group("💬 提示词")
        p_lay = QVBoxLayout(grp_prompt)
        p_lay.setSpacing(6)

        p_lay.addWidget(_field("正面提示词", 0))
        host.txt_video_prompt = QTextEdit()
        host.txt_video_prompt.setFixedHeight(92)
        host.txt_video_prompt.setPlaceholderText(
            "例如：一只可爱的小猫在草地上奔跑, best quality, masterpiece")
        p_lay.addWidget(host.txt_video_prompt)

        p_lay.addWidget(_field("负面提示词", 0))
        host.txt_video_neg = QTextEdit()
        host.txt_video_neg.setFixedHeight(68)
        host.txt_video_neg.setPlaceholderText(
            "例如：blurry, lowres, worst quality, text, watermark")
        p_lay.addWidget(host.txt_video_neg)

        prompt_btn_row = QHBoxLayout()
        prompt_btn_row.setSpacing(6)

        host.btn_enhance_video_prompt = _pill("✨ 智能改写", 108)
        host.btn_enhance_video_prompt.setToolTip(
            "将自然语言提示词自动转换为英文 Danbooru 标签\n"
            "同时改写正面和负面提示词\n"
            "首次使用会下载 Qwen2-VL 模型 (~4.5GB)")
        prompt_btn_row.addWidget(host.btn_enhance_video_prompt)

        host.btn_vision_video_prompt = _pill("📷 识图生成", 108)
        host.btn_vision_video_prompt.setToolTip(
            "上传一张图片，AI 自动识别内容并生成提示词")
        prompt_btn_row.addWidget(host.btn_vision_video_prompt)

        host.btn_enhance_travel = _pill("✨ 改写旅行段", 120)
        host.btn_enhance_travel.setToolTip("用 AI 改写所有旅行分段的提示词")
        prompt_btn_row.addWidget(host.btn_enhance_travel)
        prompt_btn_row.addStretch()
        p_lay.addLayout(prompt_btn_row)
        root.addWidget(grp_prompt)

        # ============ ✨ 提示词旅行 ============
        host.grp_prompt_travel = _group("✨ 提示词旅行", accent=True)
        host.grp_prompt_travel.setCheckable(True)
        host.grp_prompt_travel.setChecked(False)
        host.grp_prompt_travel.setToolTip("在指定帧切换提示词，实现剧情/动作变化")
        tv_lay = QVBoxLayout(host.grp_prompt_travel)
        tv_lay.setSpacing(8)

        tv_lay.addWidget(_hint(
            "在不同帧使用不同提示词。可用「分段编辑」或「文本格式」，两者填写其一即可。"))

        sw_row = QHBoxLayout()
        sw_row.setSpacing(8)
        sw_row.addWidget(_field("编辑方式:", 72))
        host.combo_travel_mode = QComboBox()
        host.combo_travel_mode.addItems(["🧩 分段编辑（推荐）", "⌨️ 文本格式"])
        sw_row.addWidget(host.combo_travel_mode, 1)
        tv_lay.addLayout(sw_row)

        host.wrap_travel_segments = QWidget()
        seg_lay = QVBoxLayout(host.wrap_travel_segments)
        seg_lay.setContentsMargins(0, 0, 0, 0)
        seg_lay.setSpacing(6)

        host.travel_container = QVBoxLayout()
        host.travel_container.setSpacing(6)
        seg_lay.addLayout(host.travel_container)

        seg_btn_row = QHBoxLayout()
        seg_btn_row.setSpacing(6)
        btn_add_segment = _pill("➕ 添加段", 96)
        _wire(host, btn_add_segment.clicked, "_add_travel_segment")
        seg_btn_row.addWidget(btn_add_segment)
        btn_auto_spread = _pill("⇄ 均匀分布帧号", 130)
        btn_auto_spread.setToolTip("按当前总帧数自动重排各段起始帧")
        _wire(host, btn_auto_spread.clicked, "_spread_travel_frames")
        seg_btn_row.addWidget(btn_auto_spread)
        seg_btn_row.addStretch()
        seg_lay.addLayout(seg_btn_row)
        tv_lay.addWidget(host.wrap_travel_segments)

        host.wrap_travel_text = QWidget()
        txt_lay = QVBoxLayout(host.wrap_travel_text)
        txt_lay.setContentsMargins(0, 0, 0, 0)
        txt_lay.setSpacing(4)
        txt_lay.addWidget(_hint("格式：帧号|提示词（每行一个关键帧）"))
        host.txt_prompt_travel = QTextEdit()
        host.txt_prompt_travel.setFixedHeight(100)
        host.txt_prompt_travel.setPlaceholderText(
            "0|1girl, smiling, sunny day\n8|1girl, surprised, wind blowing\n"
            "16|1girl, crying, rain falling")
        txt_lay.addWidget(host.txt_prompt_travel)
        host.wrap_travel_text.setVisible(False)
        tv_lay.addWidget(host.wrap_travel_text)

        _wire(host, host.combo_travel_mode.currentIndexChanged,
              "_on_travel_edit_mode_changed")
        root.addWidget(host.grp_prompt_travel)

        host.travel_segments = getattr(host, "travel_segments", None) or []
        host.grp_travel = host.grp_prompt_travel   # 兼容旧引用

        # ============ 🎞️ 视频参数 ============
        grp_video = _group("🎞️ 视频参数")
        v_lay = QVBoxLayout(grp_video)
        v_lay.setSpacing(8)

        g = _grid()
        g.setColumnStretch(4, 1)

        g.addWidget(_field("帧数:"), 0, 0)
        host.spin_video_frames = _spin(8, 80, 16)
        host.spin_video_frames.setToolTip("总生成帧数；开启长视频模式后上限提升")
        g.addWidget(host.spin_video_frames, 0, 1)
        g.addWidget(_field("FPS:", 52), 0, 2)
        host.spin_video_fps = _spin(4, 30, 8)
        g.addWidget(host.spin_video_fps, 0, 3)
        host.lbl_video_duration = QLabel("≈ 2.0 秒")
        host.lbl_video_duration.setProperty("role", "value")
        g.addWidget(host.lbl_video_duration, 0, 4)

        g.addWidget(_field("步数:"), 1, 0)
        host.spin_video_steps = _spin(10, 100, 25)
        g.addWidget(host.spin_video_steps, 1, 1)
        g.addWidget(_field("CFG:", 52), 1, 2)
        host.spin_video_cfg = _dspin(1.0, 20.0, 7.5, 0.5, 1)
        g.addWidget(host.spin_video_cfg, 1, 3)

        g.addWidget(_field("宽 × 高:"), 2, 0)
        res_row = QHBoxLayout()
        res_row.setSpacing(6)
        host.spin_video_w = _spin(256, 1024, 512, 64, 88)
        host.spin_video_h = _spin(256, 1024, 512, 64, 88)
        x_lbl = QLabel("×")
        x_lbl.setProperty("role", "field")
        res_row.addWidget(host.spin_video_w)
        res_row.addWidget(x_lbl)
        res_row.addWidget(host.spin_video_h)
        res_row.addStretch()
        g.addLayout(res_row, 2, 1, 1, 4)

        g.addWidget(_field("采样器:"), 3, 0)
        host.combo_video_sched = QComboBox()
        host.combo_video_sched.addItems(
            ["EulerDiscrete (推荐)", "DPM++ 2M", "LCM (快速)", "DDIM"])
        host.combo_video_sched.setMinimumHeight(32)
        g.addWidget(host.combo_video_sched, 3, 1, 1, 4)
        v_lay.addLayout(g)

        dur_row = QHBoxLayout()
        dur_row.setSpacing(6)
        dur_row.addWidget(_field("快捷时长:"))
        for sec in (2, 4, 6, 8, 10):
            b = _pill(f"{sec}秒", 50)
            b.clicked.connect(
                lambda _=False, s=sec: getattr(host, "_set_video_duration", lambda *_: None)(s))
            dur_row.addWidget(b)
        dur_row.addStretch()
        v_lay.addLayout(dur_row)

        v_lay.addWidget(_hint("建议 8–12 FPS：更高更流畅，但显存与耗时线性增加。"))

        host.chk_long_video = QCheckBox("🎬 长视频模式 (>32 帧)")
        host.chk_long_video.setToolTip("启用 Context Window，帧数上限扩至 150")
        _wire(host, host.chk_long_video.toggled, "_on_long_video_toggled")
        v_lay.addWidget(host.chk_long_video)
        root.addWidget(grp_video)

        # ============ 🎭 Motion LoRA ============
        grp_lora = _group("🎭 Motion LoRA (可多选)")
        l_lay = QVBoxLayout(grp_lora)
        l_lay.setSpacing(6)

        add_row = QHBoxLayout()
        add_row.setSpacing(8)
        host.cmb_motion_lora_pick = QComboBox()
        host.cmb_motion_lora_pick.setMinimumHeight(32)
        host.cmb_motion_lora_pick.addItem("-- 选择 Motion LoRA --")
        _scan = getattr(host, "_scan_motion_loras", None)
        for name in (_scan() if callable(_scan) else []):
            host.cmb_motion_lora_pick.addItem(name)

        btn_add_lora = _pill("➕ 添加", 78)
        _wire(host, btn_add_lora.clicked, "_add_motion_lora_item")
        add_row.addWidget(host.cmb_motion_lora_pick, 1)
        add_row.addWidget(btn_add_lora)
        l_lay.addLayout(add_row)

        host.motion_lora_container = QVBoxLayout()
        host.motion_lora_container.setSpacing(4)
        l_lay.addLayout(host.motion_lora_container)
        host.motion_lora_items = []

        host.lbl_motion_lora_hint = _hint(
            "未检测到 LoRA，请将模型放入 models/motion_lora/"
            if host.cmb_motion_lora_pick.count() <= 1 else
            "可叠加多个运镜 LoRA（如 ZoomIn / PanLeft），权重建议 ≤ 0.8。")
        l_lay.addWidget(host.lbl_motion_lora_hint)
        root.addWidget(grp_lora)

        # ============ 🎙️ 配音 ============
        grp_voice = _group("🎙️ 配音 (可选)")
        vo_lay = QVBoxLayout(grp_voice)
        vo_lay.setSpacing(6)

        host.chk_video_voice = QCheckBox("为视频添加配音")
        host.chk_video_voice.setToolTip("生成完成后自动合成语音并合并进视频")
        vo_lay.addWidget(host.chk_video_voice)

        eng_row = QHBoxLayout()
        eng_row.setSpacing(8)
        eng_row.addWidget(_field("引擎:", 72))
        host.combo_tts_engine = QComboBox()
        host.combo_tts_engine.addItems(["ChatTTS (中文)", "GPT-SoVITS (日语)"])
        host.combo_tts_engine.setMinimumHeight(32)
        eng_row.addWidget(host.combo_tts_engine, 1)
        vo_lay.addLayout(eng_row)

        vo_lay.addWidget(_field("配音文本:", 0))
        host.txt_video_voice = QTextEdit()
        host.txt_video_voice.setFixedHeight(76)
        host.txt_video_voice.setPlaceholderText(
            "旁白文字，例如：清晨的阳光洒在草地上，一只小猫追逐着蝴蝶。")
        vo_lay.addWidget(host.txt_video_voice)

        host.wrap_chattts = QWidget()
        c_lay = QVBoxLayout(host.wrap_chattts)
        c_lay.setContentsMargins(0, 0, 0, 0)
        r = QHBoxLayout()
        r.setSpacing(6)
        r.addWidget(_field("说话人 Seed:", 92))
        host.spin_video_voice_seed = _spin(0, 999999, 2222, 1, 92)
        r.addWidget(host.spin_video_voice_seed)
        for txt, sd in (("👨 男1", 2222), ("👨 男2", 7869),
                        ("👩 女1", 1983), ("👩 女2", 4099)):
            b = _pill(txt, 62)
            b.clicked.connect(
                lambda _=False, v=sd: host.spin_video_voice_seed.setValue(v))
            r.addWidget(b)
        r.addStretch()
        c_lay.addLayout(r)
        vo_lay.addWidget(host.wrap_chattts)

        host.wrap_sovits = QWidget()
        s_lay = QVBoxLayout(host.wrap_sovits)
        s_lay.setContentsMargins(0, 0, 0, 0)
        s_lay.setSpacing(6)

        r = QHBoxLayout()
        r.setSpacing(8)
        r.addWidget(_field("参考音频:", 92))
        host.combo_sovits_ref = QComboBox()
        host.combo_sovits_ref.setMinimumHeight(32)
        host.combo_sovits_ref.addItems(["默认女声 (Nanami)"])
        r.addWidget(host.combo_sovits_ref, 1)
        btn_pick_ref = _pill("📂 自定义", 88)
        _wire(host, btn_pick_ref.clicked, "_on_pick_sovits_ref")
        r.addWidget(btn_pick_ref)
        s_lay.addLayout(r)

        r = QHBoxLayout()
        r.setSpacing(8)
        lbl_rt = _field("参考文本:", 92)
        lbl_rt.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        r.addWidget(lbl_rt)
        host.txt_sovits_reftext = QTextEdit()
        host.txt_sovits_reftext.setFixedHeight(50)
        host.txt_sovits_reftext.setPlaceholderText("参考音频对应文字（留空使用默认）")
        r.addWidget(host.txt_sovits_reftext, 1)
        s_lay.addLayout(r)

        r = QHBoxLayout()
        r.setSpacing(8)
        r.addWidget(_field("语速:", 92))
        host.spin_sovits_speed = _dspin(0.5, 2.0, 1.0, 0.05, 2, 84)
        r.addWidget(host.spin_sovits_speed)
        host.chk_sovits_auto_translate = QCheckBox("自动中 → 日翻译")
        host.chk_sovits_auto_translate.setChecked(True)
        r.addWidget(host.chk_sovits_auto_translate)
        r.addStretch()
        s_lay.addLayout(r)

        host.wrap_sovits.setVisible(False)
        vo_lay.addWidget(host.wrap_sovits)

        host.lbl_voice_hint = _hint("首次使用会自动下载 ChatTTS 模型（约 1.1 GB）。")
        vo_lay.addWidget(host.lbl_voice_hint)
        root.addWidget(grp_voice)

        # ============ ✨ 后处理 ============
        grp_post = _group("✨ 后处理 (可选)")
        po_lay = QVBoxLayout(grp_post)
        po_lay.setSpacing(6)

        r = QHBoxLayout()
        r.setSpacing(8)
        host.chk_frame_interp = QCheckBox("帧插值 (RIFE)")
        host.chk_frame_interp.setToolTip("补帧让动作更连贯，不改变时长（提高 FPS）")
        r.addWidget(host.chk_frame_interp)
        host.combo_frame_interp = QComboBox()
        host.combo_frame_interp.addItems(["2x", "4x", "8x"])
        host.combo_frame_interp.setEnabled(False)
        host.combo_frame_interp.setMinimumWidth(72)
        r.addWidget(host.combo_frame_interp)
        r.addWidget(_hint("使视频更流畅"))
        r.addStretch()
        po_lay.addLayout(r)

        r = QHBoxLayout()
        r.setSpacing(8)
        host.chk_video_upscale = QCheckBox("🔍 视频放大 (Real-ESRGAN)")
        host.chk_video_upscale.setToolTip("512 → 1024 / 2048，显著增加耗时")
        r.addWidget(host.chk_video_upscale)
        host.combo_upscale_factor = QComboBox()
        host.combo_upscale_factor.addItems(["2x", "4x"])
        host.combo_upscale_factor.setEnabled(False)
        host.combo_upscale_factor.setMinimumWidth(72)
        r.addWidget(host.combo_upscale_factor)
        r.addStretch()
        po_lay.addLayout(r)

        host.chk_frame_interp.toggled.connect(host.combo_frame_interp.setEnabled)
        host.chk_video_upscale.toggled.connect(host.combo_upscale_factor.setEnabled)
        root.addWidget(grp_post)

        # ============ 💾 输出设置 ============
        grp_out = _group("💾 输出设置")
        o_lay = QVBoxLayout(grp_out)
        r = QHBoxLayout()
        r.setSpacing(8)
        r.addWidget(_field("格式:", 72))
        host.combo_video_fmt = QComboBox()
        host.combo_video_fmt.addItems(["MP4", "GIF", "MP4 + GIF"])
        host.combo_video_fmt.setMinimumHeight(32)
        r.addWidget(host.combo_video_fmt, 1)
        r.addStretch()
        o_lay.addLayout(r)
        o_lay.addWidget(_hint("MP4 适合分享与二次剪辑，GIF 适合社交媒体。"))
        root.addWidget(grp_out)

        # ============ 🎬 生成 ============
        host.btn_gen_video = QPushButton("🎬 生成视频")
        host.btn_gen_video.setObjectName("btnGenVideo")
        host.btn_gen_video.setMinimumHeight(48)
        host.btn_gen_video.setCursor(Qt.CursorShape.PointingHandCursor)
        _wire(host, host.btn_gen_video.clicked, "on_generate_video")
        root.addSpacing(6)
        root.addWidget(host.btn_gen_video)

        host.lbl_video_status = QLabel("💤 待命中 — 设置参数后点击生成")
        host.lbl_video_status.setProperty("role", "hint")
        host.lbl_video_status.setWordWrap(True)
        root.addWidget(host.lbl_video_status)
        root.addStretch()

        # ---------------- 信号联动 ----------------
        _wire(host, host.combo_video_mode.currentIndexChanged, "_on_video_mode_changed")
        _wire(host, host.btn_enhance_video_prompt.clicked, "on_enhance_video_prompt")
        _wire(host, host.btn_vision_video_prompt.clicked, "on_vision_video_prompt")
        _wire(host, host.btn_enhance_travel.clicked, "on_enhance_travel_prompts")
        _wire(host, host.spin_video_frames.valueChanged, "_update_video_duration_hint")
        _wire(host, host.spin_video_fps.valueChanged, "_update_video_duration_hint")
        _wire(host, host.combo_tts_engine.currentIndexChanged, "_on_tts_engine_changed")
        for wdg in (host.txt_video_voice, host.combo_tts_engine,
                    host.wrap_chattts, host.wrap_sovits):
            host.chk_video_voice.toggled.connect(wdg.setEnabled)
            wdg.setEnabled(False)

        # 初始化状态（有宿主方法才调）
        if hasattr(host, "_on_video_mode_changed"):
            host._on_video_mode_changed(host.combo_video_mode.currentIndex())
        if hasattr(host, "_update_video_duration_hint"):
            host._update_video_duration_hint()
        if hasattr(host, "_add_travel_segment"):
            host._add_travel_segment()
            host._add_travel_segment()

        # 包一层滚动区（原结构）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(w)
        return scroll

    # ========================================================
    #  中央工作区（原 _build_video_right_panel）
    # ========================================================
    def _build_workspace(self, host) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        video_preview_wrap = QWidget()
        video_preview_layout = QVBoxLayout(video_preview_wrap)
        video_preview_layout.setContentsMargins(0, 0, 0, 0)
        video_preview_layout.setSpacing(4)

        host.video_player = QMediaPlayer()
        # offscreen 测试环境下 QAudioOutput 会挂住进程，跳过音频输出
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            host.audio_output = None
        else:
            host.audio_output = QAudioOutput()
            host.video_player.setAudioOutput(host.audio_output)

        host.video_widget = QVideoWidget()
        host.video_widget.setMinimumHeight(300)
        host.video_player.setVideoOutput(host.video_widget)

        _wire(host, host.video_player.mediaStatusChanged, "_on_video_media_changed")
        _wire(host, host.video_player.errorOccurred, "_on_video_player_error")

        host.lbl_video_placeholder = QLabel(
            "🎥 视频生成后自动播放\n或从下方历史列表双击选择")
        host.lbl_video_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        host.lbl_video_placeholder.setMinimumHeight(300)
        host.lbl_video_placeholder.setProperty("role", "hint")

        video_stacked = QStackedWidget()
        video_stacked.setMinimumHeight(300)
        video_stacked.addWidget(host.lbl_video_placeholder)
        video_stacked.addWidget(host.video_widget)
        video_stacked.setCurrentIndex(0)
        host.video_stacked = video_stacked
        video_preview_layout.addWidget(video_stacked)

        video_btn_row = QHBoxLayout()
        video_btn_row.setSpacing(4)
        host.btn_video_save = QPushButton("💾 保存")
        host.btn_video_refresh = QPushButton("🔄 刷新")
        host.btn_video_pause = QPushButton("⏯️ 暂停")
        host.btn_video_stop = QPushButton("⏹️ 停止")
        for b in (host.btn_video_save, host.btn_video_refresh,
                  host.btn_video_pause, host.btn_video_stop):
            video_btn_row.addWidget(b)
        _wire(host, host.btn_video_save.clicked, "_save_current_video")
        _wire(host, host.btn_video_refresh.clicked, "_refresh_video_gallery")
        _wire(host, host.btn_video_pause.clicked, "pause_video")
        _wire(host, host.btn_video_stop.clicked, "stop_video")
        video_preview_layout.addLayout(video_btn_row)

        video_gallery_wrap = QWidget()
        video_gallery_layout = QVBoxLayout(video_gallery_wrap)
        video_gallery_layout.setContentsMargins(0, 0, 0, 0)
        video_gallery_layout.setSpacing(2)

        lbl_video_gallery_title = QLabel("📂 视频历史 (双击播放)")
        lbl_video_gallery_title.setProperty("role", "title")
        video_gallery_layout.addWidget(lbl_video_gallery_title)

        host.video_list = QListWidget()
        host.video_list.setViewMode(QListWidget.ViewMode.IconMode)
        host.video_list.setIconSize(QSize(160, 90))
        host.video_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        host.video_list.setSpacing(10)
        _wire(host, host.video_list.itemDoubleClicked, "_on_video_item_clicked")
        video_gallery_layout.addWidget(host.video_list, 1)

        video_splitter = QSplitter(Qt.Orientation.Vertical)
        video_splitter.addWidget(video_preview_wrap)
        video_splitter.addWidget(video_gallery_wrap)
        video_splitter.setSizes([400, 350])
        video_splitter.setStretchFactor(0, 1)
        video_splitter.setStretchFactor(1, 1)
        video_splitter.setChildrenCollapsible(False)
        video_splitter.setHandleWidth(4)
        layout.addWidget(video_splitter, 1)

        lbl_log = QLabel("📋 生成日志:")
        lbl_log.setProperty("role", "hint")
        layout.addWidget(lbl_log)
        host.txt_log_video = QTextEdit()
        host.txt_log_video.setReadOnly(True)
        host.txt_log_video.setMaximumHeight(140)
        layout.addWidget(host.txt_log_video, 1)
        return w
