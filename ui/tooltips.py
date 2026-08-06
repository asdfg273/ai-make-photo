# ui/tooltips.py
"""
🌐 参数中文解释库
统一管理，方便后续翻译/维护
"""

PARAM_TOOLTIPS = {
    # ─── 基础参数 ───
    "steps": (
        "🔢 采样步数 (Steps)\n"
        "推荐: 20-40\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "去噪迭代次数，越多细节越精细但越慢。\n"
        "• 20-25: 草图速览\n"
        "• 28-35: ⭐ 日常推荐\n"
        "• 40+: 精修出图（边际收益递减）"
    ),
    "cfg": (
        "🎯 CFG 引导强度 (Classifier-Free Guidance)\n"
        "推荐: 6-9\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "AI 听话程度。\n"
        "• <5: 创意发散、容易跑偏\n"
        "• 6-9: ⭐ 平衡，写实/二次元通用\n"
        "• >12: 过曝、颜色饱和度爆炸"
    ),
    "sampler": (
        "🎲 采样器算法\n"
        "推荐: DPM++ 2M Karras\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• DPM++ 2M Karras: ⭐ 综合最佳\n"
        "• DPM++ SDE Karras: 细节丰富但慢\n"
        "• Euler a: 二次元友好、风格化\n"
        "• DDIM: 老牌稳定、收敛快\n"
        "• UniPC: 步数少时质量高"
    ),
    "resolution": (
        "📐 输出分辨率\n"
        "推荐: 512×768 (竖) / 768×512 (横)\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "SD 1.5 训练分辨率为 512，建议长边不超过 768。\n"
        "更大分辨率建议先生成 512，再用 Hires.fix 放大。"
    ),
    "count": (
        "🔁 生成数量\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "一次提交跑几张（同 prompt 不同 seed）。\n"
        "显存不够时会自动顺序生成。"
    ),
    "seed": (
        "🌱 随机种子 (Seed)\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• -1: 每次随机（探索）\n"
        "• 固定值: 复现结果（微调 prompt 时用）\n"
        "记下喜欢的图的 seed，可以再生同款构图。"
    ),

    # ─── 提示词 ───
    "prompt_positive": (
        "✅ 正向提示词\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "描述你想要的内容。\n"
        "支持中/英文，权重语法: (word:1.3)\n"
        "动态组合: {白发|黑发} 少女"
    ),
    "prompt_negative": (
        "🚫 负向提示词\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "描述你不想要的内容。\n"
        "常用: lowres, bad anatomy, extra fingers, blurry"
    ),

    # ─── img2img ───
    "strength": (
        "💪 重绘强度 (Denoising Strength)\n"
        "推荐: 0.3-0.7\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• <0.3: 几乎不变\n"
        "• 0.5: ⭐ 平衡\n"
        "• >0.8: 大幅重绘，原图只作构图参考"
    ),

    # ─── Hires.fix ───
    "hires_scale": (
        "🔍 放大倍数\n"
        "推荐: 1.5×\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "1.5× 速度/质量平衡；2× 显存吃紧。"
    ),
    "hires_denoise": (
        "🎨 放大重绘强度\n"
        "推荐: 0.3-0.5\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• <0.3: 仅补细节\n"
        "• 0.4-0.5: ⭐ 增加细节\n"
        "• >0.6: 可能与原图差异过大"
    ),
    "hires_upscaler": (
        "⬆️ 放大算法\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• Latent: ⭐ 速度快、细节足\n"
        "• ESRGAN_4x: 真实系最佳\n"
        "• R-ESRGAN Anime6B: 二次元专用"
    ),

    # ─── ADetailer ───
    "adetailer_strength": (
        "🎭 修脸重绘强度\n"
        "推荐: 0.4-0.5\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "ADetailer 检测脸部后局部重绘的强度。\n"
        "过高会让脸与身体不协调。"
    ),
    "adetailer_model": (
        "🤖 脸部检测模型\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• face_yolov8n: ⭐ 通用人脸\n"
        "• mediapipe_face_full: 真人专用\n"
        "• anime_face: 二次元专用"
    ),

    # ─── ControlNet ───
    "cn_strength": (
        "🎛️ ControlNet 控制强度\n"
        "推荐: 0.7-1.0\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• <0.5: 弱引导，AI 自由发挥\n"
        "• 0.7-1.0: ⭐ 严格遵循参考\n"
        "• >1.2: 可能过拟合到参考图"
    ),
    "cn_type": (
        "🔌 ControlNet 类型\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• Canny: 边缘线稿\n"
        "• OpenPose: ⭐ 人体姿态\n"
        "• Depth: 深度图（构图）\n"
        "• Lineart: 漫画线稿上色"
    ),

    # ─── 设备 ───
    "device": (
        "💻 运行设备\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• AUTO: ⭐ 自动选择最快\n"
        "• CUDA: NVIDIA 显卡（最快）\n"
        "• MPS: Apple M 芯片\n"
        "• CPU: 兜底（极慢，不推荐）"
    ),
}


def tip(key: str) -> str:
    """安全获取提示文案"""
    result = PARAM_TOOLTIPS.get(key, "")
    if not result:
        import logging
        logging.getLogger("ui.tooltips").warning("No tooltip found for key: %s", key)
    return result