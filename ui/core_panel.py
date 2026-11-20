# ui/core_panel.py
# ============================================================
#  生成核心区（全局单例）— 从 ui_builder._build_tab_basic /
#  _build_gen_button_area / _build_status_bar_widget 迁出
#  硬约束：所有属性名与原代码完全一致；页面禁止重建同名控件
# ============================================================
import logging
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLabel, QPushButton, QComboBox, QCheckBox,
                             QTextEdit, QSpinBox, QGroupBox, QProgressBar,
                             QSizePolicy)
from PyQt6.QtCore import Qt

from ui.widgets import FloatSlider
from core.arch import REGISTRY

logger = logging.getLogger(__name__)


def _wire(host, signal, method_name: str) -> None:
    """防御性信号连接：宿主缺方法时告警而不是崩溃（页面隔离需要）。"""
    fn = getattr(host, method_name, None)
    if callable(fn):
        signal.connect(fn)
    else:
        logger.warning(f"信号未连接（宿主缺方法）: {method_name}")


def build_core(host, layout: QVBoxLayout) -> None:
    """生成核心区：模型与设备 / 提示词 / 实时预览 / 基础参数。"""

    # ============== 1. 模型与设备 ==============
    grp_model = QGroupBox("模型与设备")
    gm = QFormLayout(grp_model)
    gm.setSpacing(8)

    host.combo_model_type = QComboBox()
    host.combo_model_type.setMinimumContentsLength(8)
    host.combo_model_type.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    _seen_subdirs = set()
    for _arch_id, _info in REGISTRY.items():
        if not _info.caps.is_base_model:
            continue
        _key = _info.model_subdir or _arch_id   # model_subdir 可能为 None，回退到架构 id
        if _key in _seen_subdirs:
            continue
        _seen_subdirs.add(_key)
        _label = _info.display_name
        if not _info.supported:
            _label += "（暂不支持）"
        host.combo_model_type.addItem(_label, _arch_id)
    _wire(host, host.combo_model_type.currentIndexChanged, "_on_model_type_changed")
    gm.addRow("模型类型:", host.combo_model_type)

    host.combo_model = QComboBox()
    host.combo_model.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    # 模型文件名很长，最小宽度按 8 个字符算，防止撑爆面板
    host.combo_model.setMinimumContentsLength(8)
    host.combo_model.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    _wire(host, host.combo_model.currentIndexChanged, "on_model_selected")
    # 模型诊断按钮：查看已加载模型的架构/v-pred/调度器/显存
    model_row = QHBoxLayout()
    model_row.setSpacing(6)
    model_row.addWidget(host.combo_model, 1)
    host.btn_model_diag = QPushButton("诊断")
    host.btn_model_diag.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    host.btn_model_diag.setToolTip(
        "诊断当前已加载模型的架构、v-pred、调度器、显存等信息")
    _wire(host, host.btn_model_diag.clicked, "_show_model_diagnostics")
    model_row.addWidget(host.btn_model_diag)
    gm.addRow("SD 模型:", model_row)

    host.combo_device = QComboBox()
    host.combo_device.addItems(["AUTO", "CUDA", "MPS", "CPU"])
    gm.addRow("运行设备:", host.combo_device)

    host.lbl_model_info = QLabel("请选择模型")
    host.lbl_model_info.setWordWrap(True)
    host.lbl_model_info.setProperty("role", "hint")
    gm.addRow(host.lbl_model_info)
    host.lbl_model_preview = QLabel("无示例图")
    host.lbl_model_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
    host.lbl_model_preview.setMinimumHeight(40)
    host.lbl_model_preview.setProperty("role", "hint")
    host.lbl_model_preview.setVisible(False)
    gm.addRow(host.lbl_model_preview)

    # ---------- 场景预设行 ----------
    preset_row = QHBoxLayout()
    preset_row.setSpacing(6)

    host.combo_preset = QComboBox()
    host.combo_preset.addItem("（无）")
    host.combo_preset.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    _wire(host, host.combo_preset.customContextMenuRequested, "show_preset_menu")
    _wire(host, host.combo_preset.currentIndexChanged, "apply_preset")
    host.combo_preset.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    preset_row.addWidget(host.combo_preset, 1)

    host.lbl_preset_badge = QLabel("")
    host.lbl_preset_badge.setProperty("role", "value")
    preset_row.addWidget(host.lbl_preset_badge)

    host.btn_save_preset = QPushButton("保存")
    host.btn_save_preset.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    host.btn_save_preset.setToolTip("把当前所有参数保存为新预设")
    _wire(host, host.btn_save_preset.clicked, "save_current_as_preset")
    preset_row.addWidget(host.btn_save_preset)

    host.btn_restore_preset = QPushButton("还原")
    host.btn_restore_preset.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    host.btn_restore_preset.setToolTip("还原到套用预设前的所有参数")
    _wire(host, host.btn_restore_preset.clicked, "restore_preset_backup")
    preset_row.addWidget(host.btn_restore_preset)

    host.btn_preset_menu = QPushButton("⋮")
    host.btn_preset_menu.setFixedSize(28, 28)
    host.btn_preset_menu.setToolTip("更多操作（删除/导入/导出）")
    host.btn_preset_menu.clicked.connect(
        lambda: host.show_preset_menu(host.btn_preset_menu.rect().bottomLeft()))
    preset_row.addWidget(host.btn_preset_menu)

    gm.addRow("场景预设:", preset_row)
    layout.addWidget(grp_model)

    # ============== 2. 提示词 ==============
    grp_prompt = QGroupBox("提示词")
    gp = QVBoxLayout(grp_prompt)
    gp.setSpacing(6)

    lbl_pos = QLabel("正向 (中/英 均可):")
    lbl_pos.setProperty("role", "title")
    gp.addWidget(lbl_pos)

    host.txt_prompt = QTextEdit()
    host.txt_prompt.setFixedHeight(100)
    host.txt_prompt.setPlaceholderText("在此输入正向提示词...")
    gp.addWidget(host.txt_prompt)

    host.lbl_dynamic_hint = QLabel("提示：使用 {红|蓝|白} 语法可批量生成所有组合")
    host.lbl_dynamic_hint.setProperty("role", "hint")
    gp.addWidget(host.lbl_dynamic_hint)

    lbl_neg = QLabel("负向提示词:")
    lbl_neg.setProperty("role", "title")
    gp.addWidget(lbl_neg)

    host.txt_neg = QTextEdit()
    host.txt_neg.setFixedHeight(70)
    host.txt_neg.setPlaceholderText("在此输入负向提示词...")
    gp.addWidget(host.txt_neg)

    # AI 工具分两行：操作与改写模式一行，自动开关与模型档位一行。
    # 英/日文本比中文长，全部塞进一行会把按钮截断。
    prompt_btn_row = QHBoxLayout()
    prompt_btn_row.setSpacing(6)

    host.btn_enhance_prompt = QPushButton("智能改写")
    host.btn_enhance_prompt.setToolTip(
        "按下方「改写模式」把输入整理成 AI 画图标准提示词。\n"
        "三种模式：\n"
        "① 严格翻译 —— 忠实转换，不增不减，完全不改变你的原意\n"
        "② 适当润色 —— 修语法/统一用词/规范化排序，不新增内容\n"
        "③ 关键词帮写 —— 只给几个关键词，帮你扩写成完整提示词\n"
        "首次使用会自动下载所选档位的模型")
    _wire(host, host.btn_enhance_prompt.clicked, "on_enhance_prompt")
    prompt_btn_row.addWidget(host.btn_enhance_prompt)

    # ─── 改写模式选择（①严格翻译 / ②适当润色 / ③关键词帮写）───
    host.combo_enhance_mode = QComboBox()
    host.combo_enhance_mode.addItem("① 严格翻译", "strict")
    host.combo_enhance_mode.addItem("② 适当润色", "polish")
    host.combo_enhance_mode.addItem("③ 关键词帮写", "keywords")
    host.combo_enhance_mode.setToolTip(
        "① 严格翻译：把中文/混合语言忠实转成英文 Danbooru 标签，\n"
        "不新增、不删除、不改写，适合你已经写好提示词的场景。\n\n"
        "② 适当润色：修语法、统一用词、按标准顺序重排，\n"
        "但不会凭空添加你没写过的内容。\n\n"
        "③ 关键词帮写：你只给几个关键词（比如「樱花 少女」），\n"
        "由 AI 扩写成含服装/姿势/构图/光线/背景的完整提示词。")
    host.combo_enhance_mode.setMinimumContentsLength(6)
    host.combo_enhance_mode.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    prompt_btn_row.addWidget(host.combo_enhance_mode, 1)

    host.btn_vision_prompt = QPushButton("识图生成")
    host.btn_vision_prompt.setToolTip(
        "上传一张图片 + 输入需求, AI 自动整合生成 SD 提示词")
    _wire(host, host.btn_vision_prompt.clicked, "on_vision_prompt")
    prompt_btn_row.addWidget(host.btn_vision_prompt)
    gp.addLayout(prompt_btn_row)

    enhance_model_row = QHBoxLayout()
    enhance_model_row.setSpacing(6)
    host.chk_auto_enhance = QCheckBox("生成前自动改写")
    host.chk_auto_enhance.setToolTip(
        "勾选后, 每次生成前都会调用 AI 智能改写提示词")
    enhance_model_row.addWidget(host.chk_auto_enhance)

    from utils.prompt_enhancer import PromptEnhancer
    host.combo_ai_model = QComboBox()
    for _k, _c in PromptEnhancer.MODEL_REGISTRY.items():
        host.combo_ai_model.addItem(_c["label"], _k)
    host.combo_ai_model.setToolTip("改写/识图模型档位，切换后下次调用生效")
    # 关键：最小宽度按 4 个字符算，否则长档位名会撑爆右侧面板（右侧控件被裁掉）
    host.combo_ai_model.setMinimumContentsLength(4)
    host.combo_ai_model.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    _wire(host, host.combo_ai_model.currentIndexChanged, "_on_ai_model_changed")
    enhance_model_row.addWidget(host.combo_ai_model, 1)

    host.btn_save_character = QPushButton("存为角色")
    host.btn_save_character.setToolTip(
        "把当前提示词 / LoRA / IP-Adapter / ControlNet / 采样参数\n"
        "一键保存为角色卡（在角色库中可继续编辑、批量出图）")
    _wire(host, host.btn_save_character.clicked, "on_save_character")
    enhance_model_row.addWidget(host.btn_save_character)
    gp.addLayout(enhance_model_row)

    # ─── 翻译模式选择：标题/回译按钮与模式下拉分行，保留下拉完整宽度 ───
    row_trans = QHBoxLayout()
    row_trans.setSpacing(6)
    lbl_trans = QLabel("翻译模式:")
    lbl_trans.setProperty("role", "title")
    row_trans.addWidget(lbl_trans)
    row_trans.addStretch(1)

    host.btn_trans_compare = QPushButton("回译对比")
    host.btn_trans_compare.setToolTip(
        "中→英→中 回译对比\n"
        "把当前提示词按所选模式翻成英文（即实际送入模型的文本），\n"
        "再让 AI 把英文翻回中文，检查翻译是否有幻觉/漏词，\n"
        "避免词不达意导致生图质量不佳")
    _wire(host, host.btn_trans_compare.clicked, "_on_trans_compare")
    row_trans.addWidget(host.btn_trans_compare)
    gp.addLayout(row_trans)

    host.combo_trans_mode = QComboBox()
    host.combo_trans_mode.addItems([
        "纯词典",
        "AI 智能改写",
        "词典优先 + AI 兜底 ",
    ])
    host.combo_trans_mode.setMinimumContentsLength(6)
    host.combo_trans_mode.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    host.combo_trans_mode.setCurrentIndex(2)
    host.combo_trans_mode.setToolTip(
        "纯词典: 只用 JSON 词典查词, 速度快但不通顺\n"
        "AI 改写: 每次都调 Qwen, 效果最好但慢\n"
        "混合模式: 词典命中直接用, 未命中才调 AI ")
    gp.addWidget(host.combo_trans_mode)

    # ─── 是否翻译：关掉就直接把中文原文投递给模型 ───
    host.chk_translate_prompt = QCheckBox("翻译成英文再生成")
    host.chk_translate_prompt.setChecked(True)
    host.chk_translate_prompt.setToolTip(
        "勾选（默认）：提示词先翻成英文再送进模型 —— SD1.5 / SDXL 这类"
        "用 CLIP 做编码器的模型只认英文，中文会被当成乱码。\n"
        "取消：直接把原文（中文）投递给模型。\n"
        "  适合 MOSS-Image 这类文本编码器本身懂中文的模型 ——\n"
        "  翻成英文反而会丢掉中文排版/文字渲染的能力。")
    _wire(host, host.chk_translate_prompt.toggled, "_on_translate_toggled")
    gp.addWidget(host.chk_translate_prompt)

    # 中文文本理解提示：按当前模型能力动态变文案
    host.lbl_prompt_lang_hint = QLabel(
        "ℹ️ 当前模型用 CLIP 编码器，只认英文 —— 建议保持上面的「翻译成英文再生成」")
    host.lbl_prompt_lang_hint.setWordWrap(True)
    host.lbl_prompt_lang_hint.setProperty("role", "hint")
    gp.addWidget(host.lbl_prompt_lang_hint)

    layout.addWidget(grp_prompt)

    # ============== 3. 实时预览 ==============
    grp_preview = QGroupBox("实时预览")
    gpv = QFormLayout(grp_preview)

    host.chk_use_preview = QCheckBox("生成时显示实时预览")
    host.chk_use_preview.setChecked(False)
    host.chk_use_preview.setToolTip(
        "每 N 步解码一次 latent 显示到画布\n"
        "GPU: 开启基本无影响\n"
        "CPU: 每次预览额外耗时 1-3 分钟, 慎用!\n"
        "用途: 看到生成过程, 早发现废图早中断")
    gpv.addRow(host.chk_use_preview)

    host.spin_preview_interval = QSpinBox()
    host.spin_preview_interval.setRange(1, 30)
    host.spin_preview_interval.setValue(10)
    host.spin_preview_interval.setSuffix("步")
    host.spin_preview_interval.setToolTip(
        "每 N 步刷新一次预览\nCPU 推荐 10-15\nGPU 推荐 3-5")
    gpv.addRow("预览间隔:", host.spin_preview_interval)

    layout.addWidget(grp_preview)

    # ============== 4. 基础参数 ==============
    grp_params = QGroupBox("基础参数")
    gpa = QFormLayout(grp_params)
    gpa.setSpacing(8)

    host.spin_steps = QSpinBox()
    host.spin_steps.setRange(1, 150)
    host.spin_steps.setValue(30)
    gpa.addRow("步数 Steps:", host.spin_steps)

    host.scale_cfg = FloatSlider(1.0, 20.0, 0.5, 7.0)
    gpa.addRow("CFG Scale:", host.scale_cfg)

    host.combo_res = QComboBox()
    #  分辨率是全局单例（切页不重造），所以这里要同时覆盖文生图与图生图需求。
    # 图生图/局部重绘最常见的诉求是"和参考图同比例"，因此补齐常见竖构图与方图，
    # 避免用户只能选到 1:1 之外的横构图而被迫形变（见 _fit_to_target 的说明）。
    host.combo_res.addItems([
        "512x512", "512x768", "768x512", "768x768",
        "640x960", "960x640",
        "768x1024", "1024x768",
        "832x1216", "1216x832",
        "1024x1024", "1024x1536", "1536x1024",
    ])
    host.combo_res.setCurrentText("512x768")
    host.combo_res.setToolTip(
        "生成分辨率。\n"
        "• 图生图建议选与参考图相近的比例（会按比例适配，不做拉伸）\n"
        "• SD1.5 建议短边 ≤768；SDXL 建议短边 ≥1024\n"
        "• 竖构图（512x768 / 832x1216）适合人物立绘")
    gpa.addRow("分辨率:", host.combo_res)

    # 兼容隐藏字段（不加入布局，与旧行为一致）
    host.spin_width = QSpinBox()
    host.spin_width.setRange(256, 2048)
    host.spin_width.setSingleStep(64)
    host.spin_width.setValue(512)
    host.spin_height = QSpinBox()
    host.spin_height.setRange(256, 2048)
    host.spin_height.setSingleStep(64)
    host.spin_height.setValue(768)

    host.spin_count = QSpinBox()
    host.spin_count.setRange(1, 32)
    host.spin_count.setValue(1)
    gpa.addRow("生成数量:", host.spin_count)
    host.spin_batch = host.spin_count

    host.spin_seed = QSpinBox()
    host.spin_seed.setRange(-1, 2147483647)
    host.spin_seed.setValue(-1)
    host.spin_seed.setSpecialValueText("随机")
    gpa.addRow("种子 Seed:", host.spin_seed)

    # 采样器与 sigma 风格的候选项来自 core/samplers.py（唯一来源）：
    # 以前这里写死 6 个名字，底层只认 4 个，其余悄悄变成 Euler a。
    from core.samplers import sampler_names, sigma_styles
    host.combo_sampler = QComboBox()
    host.combo_sampler.addItems(list(sampler_names()))
    host.combo_sampler.setToolTip(
        "选择后由 core/samplers.py 真正构造对应的 diffusers 调度器")
    gpa.addRow("采样器:", host.combo_sampler)

    host.combo_scheduler = QComboBox()
    host.combo_scheduler.addItems(list(sigma_styles()))
    host.combo_scheduler.setToolTip(
        "sigma 风格：默认 / Karras / Exponential / Beta（DDIM、PNDM、LCM "
        "没有这些选项，会自动忽略）")
    gpa.addRow("sigma 风格:", host.combo_scheduler)

    host.chk_make_comic = QCheckBox("生成完后拼合分镜连环画")
    gpa.addRow(host.chk_make_comic)

    layout.addWidget(grp_params)


def build_gen_area(host, layout: QVBoxLayout) -> None:
    """生成/停止按钮（从 _build_gen_button_area 迁入）。"""
    from ui.icons import apply_icon
    host.btn_generate = QPushButton("开始生成")
    host.btn_generate.setFixedHeight(46)
    host.btn_generate.setProperty("kind", "primary")
    apply_icon(host.btn_generate, "generate")   # 图标接口：注册 key="generate" 即生效
    _wire(host, host.btn_generate.clicked, "start_generation")
    layout.addWidget(host.btn_generate)

    host.btn_generation_queue = QPushButton("生成队列")
    host.btn_generation_queue.setFixedHeight(30)
    host.btn_generation_queue.setToolTip("查看、编辑、插队和暂停等待中的生成任务")
    _wire(host, host.btn_generation_queue.clicked, "show_generation_queue")
    layout.addWidget(host.btn_generation_queue)

    host.btn_interrupt = QPushButton("中断生成")
    host.btn_interrupt.setFixedHeight(32)
    host.btn_interrupt.setEnabled(False)
    _wire(host, host.btn_interrupt.clicked, "stop_generation")
    layout.addWidget(host.btn_interrupt)


def build_status_widgets(host) -> None:
    """状态/进度控件（从 _build_status_bar_widget 迁入）。
    只创建控件，不摆布局——shell 的状态栏负责摆放。"""
    host.lbl_status = QLabel("就绪")
    host.progress_gen = QProgressBar()
    host.progress_gen.setRange(0, 100)
    host.progress_gen.setValue(0)
    host.progress_gen.setFixedWidth(160)
    host.progress_gen.setFixedHeight(8)
    host.progress_gen.setTextVisible(False)
