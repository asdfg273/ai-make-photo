# core/arch/registry.py
"""arch_id → ArchInfo 映射表。新增架构只改这里 + signatures.py"""
from .base import ArchInfo, Capabilities

_SD_FULL = Capabilities(
    txt2img=True, img2img=True, inpaint=True, controlnet=True,
    ip_adapter=True, lora=True, compel_weighting=True,
    scheduler_switch=True, clip_skip=True, hires_fix=True,
    adetailer=True, vpred_toggle=True,
)

_TXT2IMG_ONLY = Capabilities(
    txt2img=True, img2img=False, lora=True,
    compel_weighting=False, scheduler_switch=False,
)

_NOT_A_MODEL = Capabilities(is_base_model=False)
_NOTHING = Capabilities()

REGISTRY: dict[str, ArchInfo] = {
    "sd15": ArchInfo(
        "sd15", "Stable Diffusion 1.5", "sd", True, _SD_FULL,
        model_subdir="sd15", lora_subdir="sd1.5",
    ),
    "sd21": ArchInfo(
        "sd21", "Stable Diffusion 2.1", "sd", True, _SD_FULL,
        model_subdir="sd15", lora_subdir="sd1.5",
    ),
    "sdxl": ArchInfo(
        "sdxl", "Stable Diffusion XL", "sd", True, _SD_FULL,
        model_subdir="sdxl", lora_subdir="sdxl",
    ),
    "anima": ArchInfo(
        arch_id="anima",
        display_name="Anima (Cosmos DiT 2B)",
        engine_id="anima",
        supported=True,
        lora_subdir="anima",
        extra_components=["Qwen3-0.6B text_encoder", "LLM adapter", "Qwen VAE"],
        extra_download_size="约 1.7GB",
        caps=Capabilities(
        is_base_model=True,
        supports_lora=False,     # 先关，能出图再说
        supports_compel=False,
        supports_vpred=False,
        needs_fp32_vae=False,
        heavy_vram=True,
        txt2img=True,
    ),
    ),
    "mossimage": ArchInfo(
        arch_id="mossimage",
        display_name="MOSS-Image (5B DiT)",
        engine_id="mossimage",
        supported=True,
        model_subdir="mossimage",
        # 官方权重是 **diffusers 目录**（带自定义管线 pipeline_moss_image.py），
        # 不是单文件检查点；把它整个目录放进 models/mossimage/ 即可。
        extra_components=("Diffusers 目录形式权重（含 pipeline_moss_image.py）",),
        extra_download_size="约 25GB（bf16：transformer 9.7GB + 文本编码器 16GB）",
        caps=Capabilities(
            is_base_model=True,
            # 唯一一个「原生吃中文」的架构：文本编码器是 Qwen 系，
            # 官方示例直接写中文，连画面里的中文字都能画准。
            supports_zh_prompt=True,
            supports_lora=False,
            supports_compel=False,       # 不是 CLIP，权重语法无效
            supports_vpred=False,
            needs_fp32_vae=True,         # 官方 torch_dtype 里 vae 明确用 float32
            heavy_vram=True,
            txt2img=True,
            img2img=False,               # 官方 API 只有文生图 + layout 控制
            inpaint=False,
            controlnet=False,
            ip_adapter=False,
            lora=False,
            compel_weighting=False,
            scheduler_switch=False,      # 用官方 Flow-matching Euler + shift 5
            clip_skip=False,
            hires_fix=False,
            adetailer=False,
            vpred_toggle=False,
        ),
    ),
    "flux": ArchInfo(
        "flux", "FLUX.1", "diffusers", True, _TXT2IMG_ONLY,
        model_subdir="flux", lora_subdir="flux",
        unsupported_reason=(
            "Flux 引擎尚未接通：需要先在「扩展市场」安装 Flux 引擎组件"
            "（T5-XXL / CLIP-L / VAE，约 10GB），"
            "transformer 主权重（约 24GB）请自行下载后放入 models/flux/ 或 "
            "models/puton/。\n"
            "⚠️ 许可证差异很大：schnell 为 Apache-2.0（可商用）；"
            "dev 为非商用许可（不可商用）。"
        ),
        extra_components=("T5-XXL 文本编码器", "CLIP-L", "Flux VAE"),
        extra_download_size="约 10GB（扩展市场）",
    ),
    "sd3": ArchInfo(
        "sd3", "Stable Diffusion 3", "sd3", True, _TXT2IMG_ONLY,
        model_subdir="sd3", lora_subdir="sd3",
        unsupported_reason="SD3 单文件加载依赖本地 diffusers 配置与完整检查点。",
        extra_components=("T5-XXL 文本编码器", "双 CLIP 编码器"),
    ),
    # ---------------- 现代视频架构：只留「入口」，不接线 ----------------
    # 和 Flux 同一套做法：用户能选中、能看到要下什么、能自己放进目录，
    # 但引擎不接通，选了会给出明确原因（而不是一句"未识别的架构"）。
    "ltxvideo": ArchInfo(
        "ltxvideo", "LTX-Video", "ltxvideo", False, _NOTHING,
        model_subdir="ltxvideo",
        unsupported_reason=(
            "LTX-Video 入口已预留，但引擎尚未接通（本版不提供加载）。\n"
            "它属于 DiT 视频架构，8GB 显存需要 fp8/量化 + 分块 offload 才有希望跑起来，\n"
            "而本项目现有视频链路（AnimateDiff + SD1.5 motion module）是另一套。\n"
            "现在能做的：把 diffusers 目录形式的权重放到 models/ltxvideo/，\n"
            "等引擎接通后即可直接使用。"
        ),
        extra_components=("diffusers 目录形式权重（含 model_index.json）",),
        extra_download_size="约 8~20GB（视版本与精度）",
    ),
    "wan21": ArchInfo(
        "wan21", "Wan2.1", "wan21", False, _NOTHING,
        model_subdir="wan21",
        unsupported_reason=(
            "Wan2.1 入口已预留，但引擎尚未接通（本版不提供加载）。\n"
            "它是 1.3B/14B 的视频 DiT，14B 在 8GB 卡上只能靠 CPU offload 慢慢跑；\n"
            "1.3B 相对可行，但仍需要新的采样/VAE 通路。\n"
            "现在能做的：把 diffusers 目录形式的权重放到 models/wan21/，\n"
            "等引擎接通后即可直接使用。"
        ),
        extra_components=("diffusers 目录形式权重（含 model_index.json）",),
        extra_download_size="1.3B 约 5GB / 14B 约 30GB",
    ),
    "controlnet": ArchInfo(
        "controlnet", "ControlNet 模型（非底模）", "none", False, _NOT_A_MODEL,
        unsupported_reason=(
            "这是一个 ControlNet 权重文件，不能作为底模加载。"
            "请将其放入 controlnets/ 目录，然后在 ControlNet 面板中选择。"
        ),
    ),
    "vae": ArchInfo(
        "vae", "独立 VAE（非底模）", "none", False, _NOT_A_MODEL,
        unsupported_reason="这是一个独立 VAE 文件，不能作为底模加载。",
    ),
    "lora": ArchInfo(
        "lora", "LoRA 权重（非底模）", "none", False, _NOT_A_MODEL,
        unsupported_reason="这是一个 LoRA 文件，请放入 loras/ 对应子目录。",
    ),
   "lora_sd15": ArchInfo(
        "lora_sd15", "SD1.5 LoRA（非底模）", "none", False, _NOT_A_MODEL,
        lora_subdir="sd1.5",
        unsupported_reason="这是 SD1.5 LoRA 文件，请放入 loras/sd1.5/。",
    ),
    "lora_sd21": ArchInfo(
        "lora_sd21", "SD2.1 LoRA（非底模）", "none", False, _NOT_A_MODEL,
        lora_subdir="sd1.5",
        unsupported_reason="这是 SD2.1 LoRA 文件。",
    ),
    "lora_sdxl": ArchInfo(
        "lora_sdxl", "SDXL LoRA（非底模）", "none", False, _NOT_A_MODEL,
        lora_subdir="sdxl",
        unsupported_reason="这是 SDXL LoRA 文件，请放入 loras/sdxl/。",
    ),
    "lora_dit": ArchInfo(
        "lora_dit", "DiT 架构 LoRA（底模未知）", "none", False, _NOT_A_MODEL,
        unsupported_reason=(
            "这是 DiT 架构的 LoRA，不能用于 SD1.5/SDXL。"
            "请确认它对应的底模后再放入相应目录。"
        ),
    ),
    "unknown": ArchInfo(
        "unknown", "未识别的架构", "none", False, _NOTHING,
        unsupported_reason="无法识别该文件的架构，可能是损坏文件或全新架构。",
    ),
}


def get_arch(arch_id: str) -> ArchInfo:
    return REGISTRY.get(arch_id, REGISTRY["unknown"])
