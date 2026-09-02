# ui/contracts.py
# ============================================================
#  UI 控件/方法契约 — 业务 mixin 依赖的命名契约集中在此
#  启动自检: check_contract(); 关键缺失降级: apply_degradation()
# ============================================================
import logging

logger = logging.getLogger(__name__)

# ── 全局单例控件（shell/core_panel 创建一次，页面禁止重建）──
GLOBAL_WIDGETS = [
    # 生成核心
    "combo_model", "combo_model_type", "combo_sampler", "combo_res",
    "txt_prompt", "txt_neg", "spin_steps", "spin_width", "spin_height",
    "spin_seed", "spin_count", "combo_img_format", "combo_device",
    "chk_auto_enhance", "chk_auto_features", "btn_enhance_prompt",
    "btn_vision_prompt", "lbl_model_info",
    # 生成控制
    "btn_generate", "btn_interrupt", "progress_gen", "lbl_status",
    "btn_preset_menu", "btn_save_preset", "btn_restore_preset",
    "combo_preset", "lbl_preset_badge",
    # 预览区
    "lbl_preview", "btn_open_editor", "btn_save_as",
    "btn_send_img2img", "btn_send_inpaint", "txt_log_image",
    # 画廊
    "gallery",
    # 共享折叠分组（LoRA/ControlNet/高级/X-Y）
    "combo_lora_0", "combo_lora_1", "combo_lora_2",
    "scale_lora_0", "scale_lora_1", "scale_lora_2",
    "btn_refresh_lora", "btn_insert_lora_all", "text_lora_info",
    "combo_cn_type", "btn_load_cn_img", "lbl_cn_thumb",
    "chk_reference_only", "chk_use_pose", "chk_pose_transfer",
    "chk_enable_hires", "chk_hires", "combo_hires_scale",
    "combo_hires_upscaler", "chk_use_adetailer", "chk_use_ad_hand",
    "combo_adetailer_model", "combo_ad_target", "combo_ad_hand",
    "chk_enable_xy", "combo_x_type", "combo_y_type",
    "entry_x_vals", "entry_y_vals",
    "chk_use_tiled", "spin_tiled_w", "spin_tiled_h",
    "spin_tile_overlap", "combo_tile_size", "btn_run_tiled",
    "chk_use_ipa", "combo_ipa_variant", "spin_ipa_scale", "lbl_ipa_image",
    "chk_use_preview", "spin_preview_interval",
]

# ── 页面专属控件 ──
PAGE_WIDGETS = {
    "txt2img": [],  # 专属区为空，核心控件全在全局区
    "img2img": [
        "btn_load_img", "btn_clear_img", "lbl_img_path", "lbl_ref_thumb",
        "scale_strength", "lbl_ref_fidelity", "scale_ref_fidelity",
    ],
    "video": [
        "btn_gen_video", "video_player", "video_widget", "video_list",
        "txt_video_prompt", "txt_video_neg", "txt_log_video",
        "combo_video_mode", "combo_video_fmt", "combo_video_sched",
        "chk_long_video", "chk_frame_interp", "combo_frame_interp",
        "chk_video_upscale", "chk_video_voice", "combo_tts_engine",
        "chk_make_comic", "cmb_motion_lora_pick", "motion_lora_container",
        "travel_container", "wrap_travel_segments", "wrap_travel_text",
        "txt_neg_prompt_travel", "combo_travel_mode",
        "wrap_chattts", "wrap_sovits", "combo_sovits_ref",
        "txt_sovits_reftext", "chk_sovits_auto_translate",
        "txt_video_voice", "audio_output", "lbl_video_status",
        "lbl_video_duration", "lbl_video_input", "lbl_video_placeholder",
        "btn_video_pause", "btn_video_stop", "btn_video_save",
        "btn_video_refresh", "lbl_dynamic_hint",
    ],
    "gallery": [],  # 复用全局 self.gallery
}

# ── 方法契约（业务 mixin 调用的、定义在 UI 层的方法）──
METHOD_CONTRACT = ["append_log", "set_status", "set_progress", "play_video"]

# ── 兼容别名 ──
ALIASES = {
    "btn_gen": "btn_generate",
    "btn_stop": "btn_interrupt",
    "scale_str": "scale_strength",
    "scale_hires": "scale_hires_denoise",
    "progress_total": "progress_gen",
    "progress": "progress_gen",
    "preview_canvas": "lbl_preview",
    "pose_canvas": "lbl_cn_thumb",
}
LIST_ALIASES = {
    "combo_loras": ["combo_lora_0", "combo_lora_1", "combo_lora_2"],
    "scale_loras": ["scale_lora_0", "scale_lora_1", "scale_lora_2"],
}

# ── 关键契约：缺失则禁用生成入口 ──
CRITICAL = {
    "btn_generate", "btn_interrupt", "txt_prompt", "txt_neg",
    "combo_model", "lbl_preview", "progress_gen",
    *METHOD_CONTRACT,
}


def install_aliases(host) -> None:
    """集中安装兼容别名。"""
    for alias, real in ALIASES.items():
        if hasattr(host, real):
            setattr(host, alias, getattr(host, real))
        else:
            logger.warning(f"⚠️ 别名跳过（目标缺失）: {alias} -> {real}")
    for alias, names in LIST_ALIASES.items():
        setattr(host, alias, [getattr(host, n, None) for n in names])


def check_contract(host) -> tuple[list[str], list[str]]:
    """返回 (critical_missing, minor_missing)。方法用 callable 检查。"""
    all_widgets = GLOBAL_WIDGETS + [w for ws in PAGE_WIDGETS.values() for w in ws]
    critical, minor = [], []
    for name in all_widgets:
        if hasattr(host, name) and getattr(host, name) is not None:
            continue
        (critical if name in CRITICAL else minor).append(name)
    for name in METHOD_CONTRACT:
        if not callable(getattr(host, name, None)):
            if name not in critical:
                critical.append(name)
    for alias, real in ALIASES.items():
        if hasattr(host, real) and getattr(host, alias, None) is not getattr(host, real):
            minor.append(f"alias:{alias}")
    return critical, minor


def apply_degradation(host, critical_missing: list[str]) -> None:
    """关键契约缺失：置灰生成入口并说明原因。"""
    if not critical_missing:
        return
    reason = "关键组件缺失: " + ", ".join(critical_missing[:6])
    logger.error(f"❌ 契约自检失败（关键），生成入口禁用 — {reason}")
    for btn_name in ("btn_generate", "btn_gen_video"):
        btn = getattr(host, btn_name, None)
        if btn is not None:
            btn.setEnabled(False)
            btn.setToolTip(f"⚠️ {reason}")
