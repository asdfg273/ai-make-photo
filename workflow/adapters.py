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
    notes: str = ""


@dataclass
class ModelAdapter:
    """所有适配器的基类。默认行为 = SD1.5 式全功能。"""
    arch_id: str = "sd15"
    caps: AdapterCaps = field(default_factory=AdapterCaps)

    # ---- 能力查询 ----
    def supports(self, feature: str) -> bool:
        return bool(getattr(self.caps, feature, False))

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
            hires=False, adetailer=False, heavy_vram=True,
            guidance_kwarg="guidance", default_guidance=4.0,
            uses_transformer=True,
            notes="Anima：ModularPipeline，无 Compel/负面提示词，走原始提示词"))

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
            controlnet=False, ip_adapter=False, heavy_vram=True,
            guidance_kwarg="guidance_scale", default_guidance=4.5,
            uses_transformer=True,
            notes="SD3：MMDiT + T5，引擎尚未接通"))


class FluxAdapter(ModelAdapter):
    def __init__(self):
        super().__init__(arch_id="flux", caps=AdapterCaps(
            compel=False, negative=False, pooled=True, clip_skip=False,
            vpred=False, controlnet=False, ip_adapter=False, lora=False,
            inpaint=False, hires=False, adetailer=False, heavy_vram=True,
            guidance_kwarg="guidance_scale", default_guidance=3.5,
            uses_transformer=True,
            notes="Flux：无负面提示词，需引擎组件；当前未接通"))


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
