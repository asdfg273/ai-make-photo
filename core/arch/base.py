# core/arch/base.py
"""架构能力声明与数据结构（不含加载逻辑）"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Capabilities:
    """某架构支持哪些功能。UI 直接读这个决定控件启用状态。"""
    is_base_model: bool = True      # 能否作为底模加载（ControlNet 文件为 False）
    supports_lora:   bool = True
    supports_compel: bool = False   # 仅 CLIP 文本编码器
    supports_vpred:  bool = False   # 仅 eps/v 可切的 UNet 系
    needs_fp32_vae:  bool = False
    heavy_vram:      bool = False   # 走 SDXL 级显存策略
    txt2img: bool = False
    img2img: bool = False
    inpaint: bool = False
    controlnet: bool = False
    ip_adapter: bool = False
    lora: bool = False
    compel_weighting: bool = False  # (word:1.2) 语法，仅 CLIP 系
    scheduler_switch: bool = False  # 采样器可切换
    clip_skip: bool = False
    hires_fix: bool = False
    adetailer: bool = False
    vpred_toggle: bool = False


@dataclass(frozen=True)
class ArchInfo:
    arch_id: str
    display_name: str
    engine_id: str                  # 将来对应的 Engine 实现
    supported: bool
    caps: Capabilities
    model_subdir: str | None = None  # models/ 下的子目录名
    lora_subdir: str | None = None   # loras/ 下的子目录名（可能与上不同！）
    unsupported_reason: str | None = None
    extra_components: tuple[str, ...] = ()   # 需额外下载的组件
    extra_download_size: str | None = None


@dataclass
class ArchProfile:
    prompt_mode: str          # "compel" | "raw" | "flux"
    supports_negative: bool
    guidance_kwarg: str       # "guidance_scale" | "guidance"
    uses_transformer: bool    # True => 用 pipe.transformer 而非 pipe.unet
    supports_ip_adapter: bool
    supports_image_encoder: bool
    default_guidance: float   # Flux 用 4.0（无 cfg 刻度）

@dataclass
class DetectResult:
    arch_id: str
    info: ArchInfo
    evidence: str                   # 命中了什么特征，用于日志排错
    key_count: int = 0
    size_gb: float = 0.0
    top_prefixes: tuple[str, ...] = ()  # unknown 时帮助诊断

def build_arch_profile(arch_id: str) -> "ArchProfile":
    """按 arch_id 构建 prompt 行为档案（ArchProfile 的唯一定义与实例化入口）。"""
    if arch_id == "anima":
        return ArchProfile(
            prompt_mode="raw",
            supports_negative=True,
            guidance_kwarg="guidance_scale", 
            uses_transformer=True,
            supports_ip_adapter=False,
            supports_image_encoder=False,
            default_guidance=7.0,
        )
    if arch_id == "flux":
        return ArchProfile(
            prompt_mode="flux",
            supports_negative=False,
            guidance_kwarg="guidance_scale",   # FluxPipeline 收 guidance_scale
            uses_transformer=True,
            supports_ip_adapter=False,
            supports_image_encoder=False,
            default_guidance=4.0,
        )
    if arch_id == "sd3":
        return ArchProfile(
            prompt_mode="raw",
            supports_negative=True,
            guidance_kwarg="guidance_scale",
            uses_transformer=True,
            supports_ip_adapter=False,
            supports_image_encoder=False,
            default_guidance=7.0,
        )
    # sd15 / sdxl / pony 及其余 CLIP 系 → Compel
    return ArchProfile(
        prompt_mode="compel",
        supports_negative=True,
        guidance_kwarg="guidance_scale",
        uses_transformer=False,
        supports_ip_adapter=True,
        supports_image_encoder=False,
        default_guidance=7.0,
    )