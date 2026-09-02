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
    "flux": ArchInfo(
        "flux", "FLUX.1", "diffusers", False, _TXT2IMG_ONLY,
        model_subdir="flux", lora_subdir="flux",
        unsupported_reason="Flux 引擎尚未实现。",
        extra_components=("T5-XXL 文本编码器", "CLIP-L", "Flux VAE"),
        extra_download_size="约 10GB",
    ),
    "sd3": ArchInfo(
        "sd3", "Stable Diffusion 3", "sd3", False, _TXT2IMG_ONLY,
        model_subdir="sd3", lora_subdir="sd3",
        unsupported_reason="SD3 引擎尚未实现。",
        extra_components=("T5-XXL 文本编码器", "双 CLIP 编码器"),
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