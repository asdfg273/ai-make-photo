# core/samplers.py
# ============================================================
#  采样器 / sigma 调度的**唯一实现**
# ------------------------------------------------------------
#  以前这个映射写在 core/model_manager.switch_sampler 里，只有四个分支：
#      Euler a / Euler / "DPM++ 2M"（**无条件** use_karras_sigmas=True）/ DDIM
#  其它名字全部掉进 `return EulerAncestralDiscreteScheduler(...)` —— 也就是说
#  界面上的 "DPM++ SDE Karras"、"UniPC"、"LCM"、"PNDM" 点了以后**悄悄变成
#  Euler a**，出图看起来"怎么调都一样"。这里把它变成一张真实的表：
#
#    · 名字（界面显示名）→ diffusers 调度器类 + 额外构造参数；
#    · ComfyUI / A1111 的写法（euler_ancestral、dpmpp_2m …）作为别名；
#    · "Karras / Exponential / Beta" 作为 **sigma 风格**单独一个参数，
#      不再塞进名字里 —— 但老配置里 "DPM++ 2M Karras" 这种写法仍然解析得开。
#
#  只保留 diffusers 真正支持的风格：`use_karras_sigmas` /
#  `use_exponential_sigmas` / `use_beta_sigmas`。A1111 的 "SGM Uniform" 在
#  diffusers 里没有等价开关，**故意不提供**，免得给一个"看起来能选、实际无效"
#  的选项（这个项目已经因为这类假选项返工过）。
# ============================================================
from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:  # diffusers 是硬依赖，但模块导入期不炸更好排查
    from diffusers import (
        DDIMScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSDEScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        LCMScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        UniPCMultistepScheduler,
    )
except Exception as exc:  # pragma: no cover - 只有装坏了才会这样
    logger.error("diffusers 采样器导入失败: %s", exc)
    raise


@dataclass(frozen=True)
class SamplerDef:
    """一个采样器：界面名字 + 调度器类 + 别名 + 固定构造参数。"""

    name: str
    cls: type
    aliases: tuple = ()
    extra: dict = field(default_factory=dict)
    # 需要 eta（随机性）的采样器：DDIM 以及 LMS 这类 ancestral 的近似
    uses_eta: bool = False
    note: str = ""
    # ComfyUI 的 sampler_name（导出/导入用）
    comfy: str = ""
    # 额外依赖（缺了就不该出现在下拉框里，而不是点了报错）
    requires: tuple = ()

    def available(self) -> bool:
        """依赖是否齐（例如 DPMSolverSDEScheduler 需要 torchsde）。"""
        for mod in self.requires:
            try:
                if importlib.util.find_spec(mod) is None:
                    return False
            except (ImportError, ValueError):
                return False
        return True


# 顺序 = 界面下拉的顺序：先放最常用的
SAMPLERS: tuple = (
    SamplerDef("Euler a", EulerAncestralDiscreteScheduler,
               aliases=("euler_ancestral", "euler a", "欧拉a"),
               comfy="euler_ancestral",
               note="每步都加噪声，风格最活泼，同种子不同步数差别大"),
    SamplerDef("Euler", EulerDiscreteScheduler,
               aliases=("euler", "欧拉"), comfy="euler",
               note="确定性，同种子同参数完全可复现"),
    SamplerDef("DPM++ 2M", DPMSolverMultistepScheduler,
               aliases=("dpmpp_2m", "dpm++ 2m"), comfy="dpmpp_2m",
               note="二阶多步，步数少也稳，默认推荐"),
    SamplerDef("DPM++ 2M SDE", DPMSolverMultistepScheduler,
               aliases=("dpmpp_2m_sde", "dpm++ 2m sde"), comfy="dpmpp_2m_sde",
               extra={"algorithm_type": "sde-dpmsolver++"},
               note="DPM++ 2M 的 SDE 版本，细节多但更慢"),
    SamplerDef("DPM++ 3M SDE", DPMSolverMultistepScheduler,
               aliases=("dpmpp_3m_sde", "dpm++ 3m sde"), comfy="dpmpp_3m_sde",
               extra={"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
               note="三阶 SDE，20 步以上才划算"),
    SamplerDef("DPM++ SDE", DPMSolverSDEScheduler,
               aliases=("dpmpp_sde", "dpm++ sde"), comfy="dpmpp_sde",
               requires=("torchsde",),
               note="单步 SDE，需要额外安装 torchsde"),
    SamplerDef("DPM++ 2S a", DPMSolverSinglestepScheduler,
               aliases=("dpmpp_2s_ancestral", "dpm++ 2s a"),
               comfy="dpmpp_2s_ancestral",
               note="单步 ancestral"),
    SamplerDef("DPM2", KDPM2DiscreteScheduler,
               aliases=("dpm_2", "dpm2"), comfy="dpm_2",
               note="二阶，慢但平滑"),
    SamplerDef("DPM2 a", KDPM2AncestralDiscreteScheduler,
               aliases=("dpm_2_ancestral", "dpm2 a"), comfy="dpm_2_ancestral",
               note="二阶 ancestral"),
    SamplerDef("Heun", HeunDiscreteScheduler,
               aliases=("heun", "k_heun"), comfy="heun",
               note="两步并一步的 Heun，耗步数"),
    SamplerDef("LMS", LMSDiscreteScheduler,
               aliases=("lms", "k_lms"), comfy="lms",
               uses_eta=True,
               note="线性多步，老派稳定"),
    SamplerDef("UniPC", UniPCMultistepScheduler,
               aliases=("uni_pc", "unipc"), comfy="uni_pc",
               note="预测-校正，5~10 步可用"),
    SamplerDef("DEIS", DEISMultistepScheduler,
               aliases=("deis",), comfy="deis",
               note="指数积分器，20 步以上"),
    SamplerDef("DDIM", DDIMScheduler,
               aliases=("ddim",), comfy="ddim",
               uses_eta=True,
               note="经典 DDIM，eta>0 才有随机性"),
    SamplerDef("PNDM", PNDMScheduler,
               aliases=("pndm", "plms"), comfy="pndm",
               note="老管线默认，步数取整到 50 的倍数才准"),
    SamplerDef("LCM", LCMScheduler,
               aliases=("lcm",), comfy="lcm",
               note="Latent Consistency，4~8 步；需要 LCM/蒸馏 LoRA 才有效"),
)


# sigma 风格：名字只描述"时间步怎么排"，与采样器自身无关
SIGMA_STYLES: tuple = ("默认", "Karras", "Exponential", "Beta")

_SIGMA_KWARGS = {
    "karras": {"use_karras_sigmas": True},
    "exponential": {"use_exponential_sigmas": True},
    "beta": {"use_beta_sigmas": True},
}

_SIGMA_ALIASES = {
    "默认": "默认", "default": "默认", "": "默认", "-": "默认",
    "karras": "Karras", "卡拉斯": "Karras",
    "exponential": "Exponential", "指数": "Exponential",
    "beta": "Beta", "贝塔": "Beta",
}

# 老名字里把风格写在末尾（"DPM++ 2M Karras"），解析时剥掉
_STYLE_SUFFIXES = ("karras", "exponential", "beta")


def sampler_names(available_only: bool = True) -> tuple:
    """界面/节点用的采样器名列表（有序）。

    `available_only=True` 时过滤掉缺依赖的（例如没装 torchsde 就没有
    DPM++ SDE）—— 与其让用户选一个点了报错的项，不如不显示。
    """
    return tuple(d.name for d in SAMPLERS if d.available() or not available_only)


def unavailable_samplers() -> tuple:
    """缺依赖、因此没出现在下拉框里的采样器（写明原因给用户看）。"""
    return tuple((d.name, ",".join(d.requires)) for d in SAMPLERS
                 if not d.available())


def sigma_styles() -> tuple:
    return SIGMA_STYLES


def normalize_sigma(text) -> str:
    raw = ("" if text is None else str(text)).strip()
    return _SIGMA_ALIASES.get(raw.lower(), _SIGMA_ALIASES.get(raw, "默认"))


def _split_style(name: str) -> tuple:
    """把 "DPM++ 2M Karras" 拆成 ("DPM++ 2M", "Karras")。"""
    text = (name or "").strip()
    low = text.lower()
    for suffix in _STYLE_SUFFIXES:
        if low.endswith(" " + suffix):
            return text[: -len(suffix) - 1].strip(), suffix.capitalize()
    return text, ""


def find_sampler(name) -> Optional[SamplerDef]:
    """按显示名 / 别名 / 中文 找定义；找不到返回 None（调用方保留现状）。"""
    text = (name or "").strip()
    if not text:
        return None
    low = text.lower()
    for d in SAMPLERS:
        if d.name.lower() == low:
            return d
    for d in SAMPLERS:
        if low in {a.lower() for a in d.aliases}:
            return d
    # 宽松匹配："DPM++ 2M Karras" 之外的写法，例如 "dpmpp2m" / "Euler A"
    squashed = re.sub(r"[\s_+\-]+", "", low)
    for d in SAMPLERS:
        cand = {re.sub(r"[\s_+\-]+", "", d.name.lower())}
        cand |= {re.sub(r"[\s_+\-]+", "", a.lower()) for a in d.aliases}
        if squashed in cand:
            return d
    return None


def resolve(name, sigma=None) -> tuple:
    """解析成 (SamplerDef|None, sigma 风格, 是否连名字里带风格)。"""
    base, inline = _split_style(name)
    d = find_sampler(base)
    style = normalize_sigma(sigma)
    if style == "默认" and inline:
        style = normalize_sigma(inline)
    return d, style, bool(inline)


def _supported_kwargs(cls, kwargs: dict) -> dict:
    """只保留该类 __init__ 真正接受的参数。

    diffusers 各调度器的 sigma 开关并不统一（LCM/PNDM/DDIM 就没有
    use_karras_sigmas），硬塞会 TypeError —— 所以按签名过滤，而不是写死
    "哪些类支持哪个开关"。
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return {}
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def build_scheduler(base_config, name, sigma=None, **overrides):
    """按名字 + sigma 风格构造调度器实例；不认识的名字返回 None。

    参数
    ----
    base_config : 现有管线的 `scheduler.config`（保留 beta/alpha 等训练侧设置）
    name        : 采样器名（可以是别名，也可以带 " Karras" 后缀）
    sigma       : "默认" / "Karras" / "Exponential" / "Beta"
    overrides   : 显式 kwargs（v-pred 的 prediction_type、zero-SNR 等），**优先**
    """
    d, style, inline = resolve(name, sigma)
    if d is None:
        return None
    if not d.available():
        need = "、".join(d.requires) or "未知依赖"
        raise RuntimeError(
            f"采样器「{d.name}」需要额外依赖 {need}（安装："
            f"venv\\Scripts\\pip install {' '.join(d.requires)}），"
            f"当前不可用 —— 请改选其它采样器。")

    kwargs = {}
    kwargs.update(_SIGMA_KWARGS.get(style.lower(), {}))
    kwargs.update(d.extra)
    kwargs.update(overrides or {})
    kwargs = _supported_kwargs(d.cls, kwargs)

    cls = d.cls.from_config
    try:
        return cls(base_config, **kwargs)
    except TypeError:
        # from_config 的签名和 __init__ 不同：退一步只传 config
        logger.debug("采样器 %s 的构造参数被拒，改按配置原样构造", d.name)
        return d.cls.from_config(base_config)


def describe(name, sigma=None) -> str:
    """一行中文说明（日志 / 状态栏用）。"""
    d, style, inline = resolve(name, sigma)
    if d is None:
        return f"未知采样器「{name}」"
    tag = f" · {style}" if style != "默认" else ""
    note = f"（{d.note}）" if d.note else ""
    return f"{d.name}{tag}{note}"


def to_comfy(name) -> Optional[str]:
    """映射到 ComfyUI 的 sampler_name（导出用）；无对应返回 None。"""
    d = find_sampler(name)
    return (d.comfy or None) if d is not None else None


def sigma_to_comfy(style) -> str:
    s = normalize_sigma(style)
    return {"默认": "normal", "Karras": "karras",
            "Exponential": "exponential", "Beta": "beta"}.get(s, "normal")
