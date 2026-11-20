# workflow/adapters.py
"""模型族适配器：节点只描述"做什么"，适配器决定"怎么在本模型族上做"。

为什么不给每个模型写一套节点
----------------------------
LoadCheckpoint / Sampler / TextEncode 这些行为对所有扩散模型都是同一件事，
差别只在：用哪个 Pipeline 类、文本编码器长什么样、有没有 pooled 输出、
scheduler 怎么配、显存策略如何。把这些差异收进适配器后，节点代码只有一份，
将来加模型主要就是加适配器。

已实现：SD15(SD1.5/SD2.1)、SDXL、Anima(Cosmos DiT)、SD3、Flux
VideoAdapter 目前只做能力声明（视频生成仍走原有 video_gen 路径）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class AdapterCaps:
    """适配器能力声明 —— 节点与 UI 据此决定开放哪些功能。"""
    txt2img: bool = True
    img2img: bool = True
    inpaint: bool = True
    compel: bool = False              # 支持权重语法 (word:1.2)
    negative: bool = True             # 支持负面提示词
    pooled: bool = False              # 需要 pooled embeds（SDXL）
    clip_skip: bool = False
    controlnet: bool = True
    ip_adapter: bool = True
    lora: bool = True
    hires: bool = True
    adetailer: bool = True
    vpred: bool = True
    guidance_kwarg: str = "guidance_scale"
    default_guidance: float = 7.0
    uses_transformer: bool = False    # True => pipe.transformer 而非 pipe.unet
    heavy_vram: bool = False
    precision: str = "fp16"           # 推荐精度（Flux/SD3 用 bf16 更稳）
    zh_prompt: bool = False           # 文本编码器原生理解中文（如 MOSS-Image）
    notes: str = ""


# ======================================================================
#  统一能力描述（对外稳定接口）
# ======================================================================

class UnsupportedFeatureError(RuntimeError):
    def __init__(self, feature: str, architecture: str = "", detail: str = ""):
        self.feature = feature
        self.architecture = architecture or "unknown"
        msg = f"当前模型架构「{self.architecture}」不支持 {feature}"
        if detail:
            msg += f"（{detail}）"
        super().__init__(msg)

_FEATURE_FIELDS = {
    "txt2img": "supports_txt2img",
    "img2img": "supports_img2img",
    "inpaint": "supports_inpaint",
    "controlnet": "supports_controlnet",
    "ip_adapter": "supports_ip_adapter",
    "lora": "supports_lora",
    "hires_fix": "supports_hires_fix",
    "adetailer": "supports_adetailer",
    "compel": "supports_compel",
    "negative": "supports_negative",
    "clip_skip": "supports_clip_skip",
}

# 旧名 / 简写 → 正式能力名。兼容既有调用（如 adapter.supports("hires")），
# 同时让"写错名字"这件事不再静默失败得那么隐蔽。
_FEATURE_ALIASES = {
    "hires": "hires_fix",
    "hiresfix": "hires_fix",
    "cn": "controlnet",
    "ipadapter": "ip_adapter",
    "t2i": "txt2img",
    "i2i": "img2img",
}

# 中文名，给报错和 UI 用
FEATURE_LABELS = {
    "txt2img": "文生图", "img2img": "图生图", "inpaint": "局部重绘",
    "controlnet": "ControlNet", "ip_adapter": "IP-Adapter", "lora": "LoRA",
    "hires_fix": "高清修复", "adetailer": "ADetailer(修脸/修手)",
    "compel": "提示词权重语法", "negative": "负面提示词",
    "clip_skip": "CLIP Skip",
}

# 架构 → 推荐精度。放在这里而不是散落在加载代码里，
# 换 dtype 策略时只改这一处。
_ARCH_PRECISION = {
    "flux": "bf16", "sd3": "bf16", "sd35": "bf16",
    "anima": "bf16", "mossimage": "bf16",
    "video": "fp16", "unknown": "fp16",
}


@dataclass
class ModelCapabilities:
    """模型加载后生成的统一能力描述（节点只读它，不读模型名）。

    ⚠️ 不要手工构造本对象去"覆盖"真实能力 —— 请用 `capabilities_for()`
    从适配器派生，否则又会出现两套说法。
    """
    architecture: str = "sd15"
    supports_txt2img: bool = True
    supports_img2img: bool = True
    supports_inpaint: bool = True
    supports_controlnet: bool = True
    supports_ip_adapter: bool = True
    supports_lora: bool = True
    supports_hires_fix: bool = True
    supports_adetailer: bool = True
    recommended_precision: str = "fp16"
    # 附加信息（不算"能力"，但节点/UI 常用）
    supports_compel: bool = False
    supports_negative: bool = True
    supports_clip_skip: bool = False
    needs_pooled_embeds: bool = False
    uses_transformer: bool = False
    heavy_vram: bool = False
    guidance_kwarg: str = "guidance_scale"
    default_guidance: float = 7.0
    # 文本编码器是否原生理解中文（MOSS-Image 这类用多语种 LLM 做编码器的）
    zh_prompt: bool = False
    notes: str = ""

    # ---- 查询 ----
    def supports(self, feature: str) -> bool:
        """按能力名查询（如 `"controlnet"` / `"hires_fix"`）。

        未知能力名返回 False —— 不猜、不默认放行。
        旧的 AdapterCaps 字段名（`"hires"`）通过别名表继续可用。
        """
        name = _FEATURE_ALIASES.get(feature, feature)
        attr = _FEATURE_FIELDS.get(name)
        if not attr:
            return False
        return bool(getattr(self, attr, False))

    def require(self, feature: str) -> None:
        """不具备该能力就抛 `UnsupportedFeatureError`（节点里这样用最省事）。"""
        if not self.supports(feature):
            raise UnsupportedFeatureError(
                FEATURE_LABELS.get(feature, feature), self.architecture)

    def unsupported(self) -> List[str]:
        """列出所有不支持的能力（UI 用来灰掉功能）。"""
        return [k for k in FEATURE_FIELDS if not self.supports(k)]

    # ---- 输出 ----
    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "supports_txt2img": self.supports_txt2img,
            "supports_img2img": self.supports_img2img,
            "supports_inpaint": self.supports_inpaint,
            "supports_controlnet": self.supports_controlnet,
            "supports_ip_adapter": self.supports_ip_adapter,
            "supports_lora": self.supports_lora,
            "supports_hires_fix": self.supports_hires_fix,
            "supports_adetailer": self.supports_adetailer,
            "recommended_precision": self.recommended_precision,
        }

    def describe(self) -> str:
        ok = [FEATURE_LABELS[k] for k in FEATURE_FIELDS if self.supports(k)]
        no = [FEATURE_LABELS[k] for k in FEATURE_FIELDS if not self.supports(k)]
        s = f"{self.architecture} · {self.recommended_precision} · 支持: " + \
            ("、".join(ok) if ok else "无")
        if no:
            s += " · 不支持: " + "、".join(no)
        return s


# 对外能力名 → 字段（供模块级使用）
FEATURE_FIELDS = _FEATURE_FIELDS


def _caps_from_adapter(adapter: "ModelAdapter") -> ModelCapabilities:
    c = adapter.caps
    return ModelCapabilities(
        architecture=adapter.arch_id,
        supports_txt2img=bool(c.txt2img),
        supports_img2img=bool(c.img2img),
        supports_inpaint=bool(c.inpaint),
        supports_controlnet=bool(c.controlnet),
        supports_ip_adapter=bool(c.ip_adapter),
        supports_lora=bool(c.lora),
        supports_hires_fix=bool(c.hires),
        supports_adetailer=bool(c.adetailer),
        recommended_precision=getattr(
            c, "precision", "") or _ARCH_PRECISION.get(adapter.arch_id, "fp16"),
        supports_compel=bool(c.compel),
        supports_negative=bool(c.negative),
        supports_clip_skip=bool(c.clip_skip),
        needs_pooled_embeds=bool(c.pooled),
        uses_transformer=bool(c.uses_transformer),
        heavy_vram=bool(c.heavy_vram),
        guidance_kwarg=c.guidance_kwarg,
        default_guidance=c.default_guidance,
        zh_prompt=bool(getattr(c, "zh_prompt", False)),
        notes=c.notes,
    )


def capabilities_for(arch_id: str) -> ModelCapabilities:
    """由架构 id 得到统一能力描述（节点/UI 的入口）。"""
    return _caps_from_adapter(get_adapter(arch_id or "sd15"))


def capabilities_of(model_ref) -> ModelCapabilities:
    """由一个已加载的 ModelRef 得到能力描述。"""
    adapter = getattr(model_ref, "adapter", None)
    if adapter is not None:
        return _caps_from_adapter(adapter)
    return capabilities_for(getattr(model_ref, "arch_id", "") or "sd15")


def capabilities_of_manager(ai) -> ModelCapabilities:
    """由模型管理器（加载完模型之后）得到能力描述。"""
    try:
        caps = getattr(ai, "capabilities", None)
        if isinstance(caps, ModelCapabilities):
            return caps
    except Exception:
        pass
    return capabilities_for(getattr(ai, "current_arch_id", "") or "sd15")


@dataclass
class ModelAdapter:
    """所有适配器的基类。默认行为 = SD1.5 式全功能。"""
    arch_id: str = "sd15"
    caps: AdapterCaps = field(default_factory=AdapterCaps)

    # ---- 能力查询 ----
    def supports(self, feature: str) -> bool:
        """按能力名查询。

        与 `ModelCapabilities.supports()` **用同一套能力名**（`"hires_fix"`、
        `"controlnet"`…），内部委托给它，避免出现"两套名字"：
        以前这里是 `getattr(self.caps, feature)`，于是同一个概念要看
        `"hires"`，而 ModelCapabilities 看 `"hires_fix"`，
        写错一个就静默返回 False（功能被悄悄锁死）。
        旧名（`"hires"` 等）由别名表兼容，不会立刻炸掉现有调用。
        """
        return capabilities_for(self.arch_id).supports(feature)

    def guidance_kwarg(self) -> str:
        return self.caps.guidance_kwarg

    def default_guidance(self) -> float:
        return self.caps.default_guidance

    # ---- 管线 ----
    def pipe(self, model_ref, kind: str = "txt2img"):
        """取对应用途的管线。kind: txt2img | img2img | inpaint"""
        if kind == "img2img":
            return getattr(model_ref, "img2img_pipe", None) or model_ref.pipe
        if kind == "inpaint":
            return getattr(model_ref, "inpaint_pipe", None) or model_ref.pipe
        return model_ref.pipe

    def sample_kwargs(self, conditioning, *, steps: int, cfg: float,
                      seed: int, width: int, height: int, **extra) -> dict:
        """组装采样调用参数（把 Conditioning 展开成管道认识的键）。"""
        kw: dict = {"num_inference_steps": int(steps), **extra}
        if self.caps.negative:
            kw[self.guidance_kwarg()] = float(cfg)

        c = conditioning
        if c is not None and getattr(c, "ready", False):
            kw["prompt_embeds"] = c.embeds
            if self.caps.negative and getattr(c, "negative_embeds", None) is not None:
                kw["negative_prompt_embeds"] = c.negative_embeds
            if self.caps.pooled and getattr(c, "pooled", None) is not None:
                kw["pooled_prompt_embeds"] = c.pooled
                if getattr(c, "negative_pooled", None) is not None:
                    kw["negative_prompt_embeds"] = c.negative_embeds
                    kw["negative_pooled_prompt_embeds"] = c.negative_pooled
        else:
            kw["prompt"] = getattr(c, "text", "") or ""
            if self.caps.negative:
                kw["negative_prompt"] = getattr(c, "negative_text", "") or ""
            kw["width"] = int(width)
            kw["height"] = int(height)
        return kw

    def describe(self) -> str:
        c = self.caps
        feats = [n for n, v in (("txt2img", c.txt2img), ("img2img", c.img2img),
                                ("inpaint", c.inpaint), ("compel", c.compel),
                                ("controlnet", c.controlnet),
                                ("ipa", c.ip_adapter), ("lora", c.lora),
                                ("hires", c.hires), ("adetailer", c.adetailer))
                 if v]
        return f"{self.arch_id}: {', '.join(feats)}"


# ---------------------------------------------------------------- 具体适配器
class SD15Adapter(ModelAdapter):
    def __init__(self):
        super().__init__(arch_id="sd15", caps=AdapterCaps(
            compel=True, pooled=False, clip_skip=True, vpred=True,
            guidance_kwarg="guidance_scale", default_guidance=7.0,
            heavy_vram=False,
            notes="SD1.5/SD2.1：单 CLIP，compel 权重语法可用"))


class SDXLAdapter(ModelAdapter):
    def __init__(self):
        super().__init__(arch_id="sdxl", caps=AdapterCaps(
            compel=True, pooled=True, clip_skip=False, vpred=False,
            guidance_kwarg="guidance_scale", default_guidance=6.0,
            heavy_vram=True,
            notes="SDXL：双编码器，需要 pooled；clip skip 无效（管线固定倒数第二层）"))

    def sample_kwargs(self, conditioning, **kw) -> dict:
        out = super().sample_kwargs(conditioning, **kw)
        # SDXL 的 pooled 走单独键名，super() 已处理；这里只确保不残留 width/height
        out.pop("width", None)
        out.pop("height", None)
        return out


class AnimaAdapter(ModelAdapter):
    """Anima（Cosmos DiT 2B，ModularPipeline）。"""

    def __init__(self):
        super().__init__(arch_id="anima", caps=AdapterCaps(
            compel=False, negative=False, pooled=False, clip_skip=False,
            vpred=False, controlnet=False, ip_adapter=False, lora=False,
            hires=False, adetailer=True, heavy_vram=True, precision="bf16",
            guidance_kwarg="guidance", default_guidance=4.0,
            uses_transformer=True,
            # AnimaAutoBlocks 的 block 输入只有 latents/尺寸/strength，
            # 没有 mask/mask_image —— 跟 SD3 是同一类问题：以前默认
            # inpaint=True，UI 放行了局部重绘，跑到调用才 TypeError。
            # 要做局部重绘请用「区域提示词」蒙版贴回那条路。
            inpaint=False,
            notes="Anima：ModularPipeline，无 Compel/负面提示词，走原始提示词；"
                  "支持 img2img（需显式 height/width），无 inpaint；"
                  "修脸/修手走「裁切图生图」（不是蒙版 inpaint）"))

    def sample_kwargs(self, conditioning, **kw) -> dict:
        # 非 compel 架构：直接用原始文本，由管线内部编码
        out = {"num_inference_steps": int(kw.pop("steps", 20)),
               "guidance": float(kw.pop("cfg", 4.0)),
               "prompt": getattr(conditioning, "text", "") or "",
               **kw}
        return out


class SD3Adapter(ModelAdapter):
    def __init__(self):
        super().__init__(arch_id="sd3", caps=AdapterCaps(
            compel=False, pooled=True, clip_skip=False, vpred=False,
            controlnet=False, ip_adapter=False, heavy_vram=True, precision="bf16",
            guidance_kwarg="guidance_scale", default_guidance=4.5,
            uses_transformer=True,
            # diffusers 没有 StableDiffusion3InpaintPipeline。
            # 以前 single_file.py 拿 StableDiffusion3Pipeline 顶替 inpaint 槽位：
            # 管线能建起来、能力表也说支持，但真调用传 image/mask_image/strength
            # 就 TypeError（SD3 的 __call__ 没有这几个参数）。
            # 这里如实声明不支持，让 UI/预检提前拦住，而不是跑到一半报错。
            inpaint=False,
            notes="SD3：MMDiT + T5，引擎尚未接通；无 inpaint 管线，"
                  "局部重绘请用「区域提示词 / 局部重绘」或蒙版贴回"))


class FluxAdapter(ModelAdapter):
    def __init__(self):
        super().__init__(arch_id="flux", caps=AdapterCaps(
            compel=False, negative=False, pooled=True, clip_skip=False,
            vpred=False, controlnet=False, ip_adapter=False, lora=False,
            inpaint=False, hires=False, adetailer=False, heavy_vram=True,
            precision="bf16",
            guidance_kwarg="guidance_scale", default_guidance=3.5,
            uses_transformer=True,
            notes="Flux：无负面提示词，需引擎组件；当前未接通"))


class MossAdapter(ModelAdapter):
    """MOSS-Image（OpenMOSS 5B DiT）—— 唯一的「原生吃中文」架构。"""

    def __init__(self):
        super().__init__(arch_id="mossimage", caps=AdapterCaps(
            compel=False, negative=True, pooled=False, clip_skip=False,
            vpred=False, controlnet=False, ip_adapter=False, lora=False,
            # 官方只有文生图 + layout 控制：没有 img2img/inpaint/hires/adetailer
            txt2img=True, img2img=False, inpaint=False, hires=False,
            adetailer=False, heavy_vram=True, precision="bf16",
            guidance_kwarg="guidance_scale", default_guidance=3.0,
            uses_transformer=True,
            zh_prompt=True,
            notes="MOSS-Image：Qwen 系文本编码器，原生理解中文（可不翻译）；"
                  "官方 50 步 / CFG 3.0 / shift 5 / uncond_skip_layers=[22]；"
                  "仅文生图"))

    def sample_kwargs(self, conditioning, **kw) -> dict:
        # 非 compel 架构：直接给原始文本，管线内部编码
        out = {"num_inference_steps": int(kw.pop("steps", 50)),
               "guidance_scale": float(kw.pop("cfg", 3.0)),
               "prompt": getattr(conditioning, "text", "") or "",
               **kw}
        return out


class VideoAdapter(ModelAdapter):
    def __init__(self):
        super().__init__(arch_id="video", caps=AdapterCaps(
            compel=False, negative=False, pooled=False, clip_skip=False,
            vpred=False, controlnet=False, ip_adapter=True, lora=False,
            hires=False, adetailer=False, inpaint=False, heavy_vram=True,
            notes="视频（SVD/AnimateDiff）：走原有 video_gen 路径"))


_ADAPTERS = {
    "sd15": SD15Adapter, "sd21": SD15Adapter,
    "sdxl": SDXLAdapter, "pony": SDXLAdapter,
    "anima": AnimaAdapter, "sd3": SD3Adapter, "flux": FluxAdapter,
    "mossimage": MossAdapter,
    "video": VideoAdapter,
}

_CACHE: dict = {}


def get_adapter(arch_id: str) -> ModelAdapter:
    """按架构取适配器（单例复用，适配器无状态）。"""
    key = (arch_id or "sd15").lower()
    if key in _CACHE:
        return _CACHE[key]
    cls = _ADAPTERS.get(key, SD15Adapter)
    inst = cls()
    if key not in _ADAPTERS:
        logger.warning(f"⚠️ 未知架构 {arch_id!r}，按 SD1.5 处理")
    _CACHE[key] = inst
    return inst


def adapter_for(model_ref) -> ModelAdapter:
    """从 ModelRef（或 ModelManager）推断适配器。"""
    a = getattr(model_ref, "adapter", None)
    if a is not None:
        return a
    arch = getattr(model_ref, "arch_id", None) or getattr(
        model_ref, "current_arch_id", "") or "sd15"
    return get_adapter(arch)


def supported_archs() -> List[str]:
    return sorted(set(_ADAPTERS))


def catalog() -> List[dict]:
    seen, out = set(), []
    for cls in _ADAPTERS.values():
        if cls in seen:
            continue
        seen.add(cls)
        inst = cls()
        c = inst.caps
        out.append({"arch_id": inst.arch_id, "notes": c.notes,
                    "features": [n for n in
                                 ("txt2img", "img2img", "inpaint", "compel",
                                  "negative", "pooled", "clip_skip",
                                  "controlnet", "ip_adapter", "lora",
                                  "hires", "adetailer")
                                 if getattr(c, n, False)],
                    "heavy_vram": c.heavy_vram})
    return out
