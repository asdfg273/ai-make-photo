# nodes/upscale_nodes.py
"""放大节点：HiresFix（二次采样）/ Upscale（纯图像放大）。

Hires.fix 在 ComfyUI 里的等价物
-------------------------------
    image --(resize)--> image2 --(img2img @ strength)--> image

关键在于**先把图放大再重采样**：直接在原尺寸上低强度重采样只会变糊，
先放大再补细节才叫 "hires"。所以这里：
· 目标尺寸必须能被 8 整除（VAE 的下采样倍率），否则管线内部会报 shape 错；
  这里主动取整并把取整后的事实写进元数据，而不是让它跑到一半炸；
· 缩放在 **PIL** 里做（LANCZOS），扩散只负责补细节；ESRGAN 之类的超分
  模型是"可选增强"：项目里没有 utils/upscaler 时降级回 LANCZOS 并告警，
  绝不让"少装一个模型"变成整张工作流跑不出来。

`Upscale` 则是纯图像节点（不碰扩散、不吃显存），用于"先把小图放大再送
Hires"这种前置整理，或最后把结果图放大到交付尺寸。
"""
from __future__ import annotations

import logging
import math

from workflow import adapters as adapters_mod
from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import as_image

logger = logging.getLogger(__name__)

# 放大粒度：VAE 的 8 倍下采样要求宽高都能被 8 整除
_SIZE_MULTIPLE = 8

_RESAMPLE = {
    "LANCZOS": "LANCZOS",
    "BICUBIC": "BICUBIC",
    "NEAREST": "NEAREST",
}

# 超分模型名 -> utils.upscaler 内部可能使用的别名（各家命名不一，多做几种尝试）
_ESRGAN_ALIASES = {
    "ESRGAN_4x": ("RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B",
                  "ESRGAN_4x", "4x-UltraSharp"),
    "R-ESRGAN 4x+": ("RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B",
                     "R-ESRGAN 4x+", "R-ESRGAN_4x+"),
    "SwinIR": ("SwinIR", "SwinIR_4x", "swinir"),
}


# ----------------------------------------------------------------------
def _resample(name) -> "object":
    """把方法名转成 PIL 重采样常量；未知名字按 LANCZOS 处理。"""
    from PIL import Image

    key = str(name or "LANCZOS").strip().upper()
    if key not in _RESAMPLE:
        raise ValueError(
            f"未知的放大方法 {name!r}，支持: {', '.join(_RESAMPLE)}")
    return getattr(Image.Resampling, key, None) or getattr(Image, key)


def _resize(image, size, method: str = "LANCZOS"):
    """按 (宽, 高) 缩放；尺寸没变则原样返回（避免无意义的像素重采样）。"""
    if image.size == (int(size[0]), int(size[1])):
        return image
    return image.resize((int(size[0]), int(size[1])), _resample(method))


def _hires_size(image, scale) -> tuple:
    """目标尺寸：原尺寸 × scale，再向上取整到 8 的倍数。

    VAE 要求宽高都能被 8 整除，这里主动对齐并把取整结果告知调用方，
    免得出现"用户填 2.0 却得到 1024x767"这种无法解释的结果。
    """
    scale = max(float(scale), 1.0)
    tw = max(_SIZE_MULTIPLE,
             int(math.ceil(image.width * scale / _SIZE_MULTIPLE)
                 * _SIZE_MULTIPLE))
    th = max(_SIZE_MULTIPLE,
             int(math.ceil(image.height * scale / _SIZE_MULTIPLE)
                 * _SIZE_MULTIPLE))
    return tw, th


def _run_external_upscaler(image, size, upscaler: str, scale):
    """尝试用 utils.upscaler 做超分；不可用返回 None（由调用方降级）。

    utils/upscaler 是可选的：不同版本函数名不一致，所以按常见名字逐个试，
    并接受多种调用签名。任何异常都只告警 —— "没有超分模型" 不该让工作流失败。
    """
    try:
        from utils import upscaler as upscaler_mod          # type: ignore
    except Exception as e:
        logger.warning(f"⚠️ 超分模块 utils.upscaler 不可用，改用 LANCZOS: {e}")
        return None

    fn = None
    for name in ("upscale_image", "upscale", "run_upscaler",
                 "apply_upscaler", "upscale_with_model"):
        cand = getattr(upscaler_mod, name, None)
        if callable(cand):
            fn = cand
            break
    if fn is None:
        logger.warning("⚠️ utils.upscaler 没有可用的放大函数，改用 LANCZOS")
        return None

    names = _ESRGAN_ALIASES.get(upscaler, (upscaler,))
    for model_name in names:
        for call in (
            lambda m=model_name: fn(image, m, scale=scale, target_size=size),
            lambda m=model_name: fn(image, m, target_size=size),
            lambda m=model_name: fn(image, m, scale),
            lambda m=model_name: fn(image, m),
        ):
            try:
                out = call()
            except TypeError:
                continue                 # 签名不匹配，换一种调法
            except Exception as e:
                # ⚠️ 这里必须用 model_name（循环变量）。以前写的是 lambda 的
                # 默认参数 m，它**不在本作用域内**，异常分支自己会抛
                # NameError 把真正的失败原因盖掉。
                logger.warning("⚠️ 超分模型 %s 执行失败，改用 LANCZOS: %s",
                               model_name, e)
                break                    # 该模型真的跑不动，不必再试别的签名
            img = as_image(out)
            if img is not None:
                # 超分模型通常固定 4x，最终尺寸以目标尺寸为准
                return _resize(img, size, "LANCZOS")
    return None


# ----------------------------------------------------------------------
@node(
    "HiresFix",
    category="放大",
    description="高清修复（二次采样）：先按倍率放大到 8 的倍数尺寸，"
                "再做一次低强度图生图补细节（Hires.fix）。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "image": PortSpec(DataType.IMAGE, "图像"),
            "positive": PortSpec(DataType.TEXT_COND, "正面条件"),
            "negative": PortSpec(DataType.TEXT_COND, "负面条件", optional=True)},
    outputs={"image": PortSpec(DataType.IMAGE, "图像"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "scale": ParamSpec("choice", default="2.0", label="放大倍率",
                           choices=("1.5", "2.0", "2.5", "3.0")),
        "denoise": ParamSpec("float", default=0.45, label="重绘强度",
                             minimum=0.05, maximum=1.0, step=0.05,
                             help="越低越贴近原图，越高细节越多但越容易变样"),
        "upscaler": ParamSpec("choice", default="Latent", label="放大器",
                              choices=("Latent", "ESRGAN_4x",
                                       "R-ESRGAN 4x+", "SwinIR"),
                              help="Latent = 仅用 LANCZOS 缩放；"
                                   "其余需 utils.upscaler，缺失时自动降级"),
        "steps": ParamSpec("int", default=24, label="步数",
                           minimum=1, maximum=150),
        "cfg": ParamSpec("float", default=7.0, label="CFG",
                         minimum=0.0, maximum=30.0, step=0.5),
        "seed": ParamSpec("seed", default=-1, label="种子",
                          help="-1 表示随机"),
    },
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=2200,
                                  supports_cpu_offload=True,
                                  supports_tiled=True),
)
def run_hires_fix(ctx, node, inputs):
    import torch

    # 复用采样节点的种子工具，避免两处随机语义不一致
    from nodes.sampling_nodes import _generator, _resolve_seed

    model = inputs.get("model")
    image = as_image(inputs.get("image"))
    if model is None:
        raise ValueError("缺少输入 model")
    if image is None:
        raise ValueError("缺少输入 image（HiresFix 需要一张底图）")

    adapter = getattr(model, "adapter", None)
    if adapter is None:
        raise RuntimeError("模型没有适配器，无法确定 img2img 调用方式")
    if not getattr(adapter.caps, "hires", True):
        raise adapters_mod.UnsupportedFeatureError(
            "高清修复", model.arch_id)

    pipe = adapter.pipe(model, "img2img")
    if pipe is None:
        raise adapters_mod.UnsupportedFeatureError(
            "高清修复", model.arch_id,
            "该架构没有可用的 img2img 管线")

    scale = float(node.params.get("scale", "2.0"))
    denoise = float(node.params.get("denoise", 0.45))
    upscaler = str(node.params.get("upscaler", "Latent") or "Latent").strip()
    steps = int(node.params.get("steps", 24))
    cfg = float(node.params.get("cfg", adapter.default_guidance()))
    seed = _resolve_seed(node.params.get("seed", -1))
    pos = inputs.get("positive")
    neg = inputs.get("negative")

    tw, th = _hires_size(image, scale)

    # ---- 1) 先把图缩放到目标尺寸（扩散只负责补细节） ----
    upscaled = image
    if upscaler != "Latent":
        ctx.status(f"🔍 超分 {upscaler} → {tw}x{th}")
        upscaled = _run_external_upscaler(image, (tw, th), upscaler, scale)
        if upscaled is None:
            logger.warning(
                f"⚠️ 超分器 {upscaler} 不可用，已降级为 LANCZOS 缩放 "
                f"({image.width}x{image.height} -> {tw}x{th})")
            upscaled = image          # 降级：退回纯 LANCZOS 缩放
    upscaled = as_image(upscaled) or image
    if upscaled.size != (tw, th):
        upscaled = _resize(upscaled, (tw, th), "LANCZOS")
    ctx.check_cancel()

    # ---- 2) 在放大后的图上做一次低强度 img2img ----
    kw = adapter.sample_kwargs(pos, steps=steps, cfg=cfg, seed=seed,
                               width=tw, height=th)
    if neg is not None and getattr(neg, "negative_embeds", None) is not None:
        kw["negative_prompt_embeds"] = neg.negative_embeds
        if getattr(neg, "negative_pooled", None) is not None:
            kw["negative_pooled_prompt_embeds"] = neg.negative_pooled
    kw["image"] = upscaled
    kw["strength"] = denoise
    # 图生图尺寸由输入图决定，显式宽高会被管线忽略甚至报冲突
    kw.pop("width", None)
    kw.pop("height", None)
    kw["generator"] = _generator(ctx.device, seed)

    ctx.status(f"✨ 高清修复 {steps} 步 · {tw}x{th} · 强度 {denoise} · seed {seed}")
    with torch.inference_mode():
        res = pipe(**kw)
    ctx.check_cancel()

    imgs = getattr(res, "images", None)
    if imgs is None:
        raise RuntimeError("img2img 管线没有返回图像")
    out = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
    out = as_image(out)
    if out is None:
        raise RuntimeError(
            f"img2img 管线返回了无法识别的图像类型: {type(out).__name__}")
    ctx.progress(steps, steps)

    meta = {"hires_enabled": True,
            "hires_scale": scale,
            "hires_denoise": denoise,
            "hires_upscaler": upscaler,
            "hires_size": f"{tw}x{th}"}
    ctx.metadata.update(meta)
    return {"image": out, "metadata": meta}


# ----------------------------------------------------------------------
@node(
    "Upscale",
    category="放大",
    description="纯图像放大（不跑扩散、不吃显存）：按倍率与插值方法缩放。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"image": PortSpec(DataType.IMAGE, "图像"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "scale": ParamSpec("float", default=2.0, label="放大倍率",
                           minimum=1.0, maximum=8.0, step=0.5),
        "method": ParamSpec("choice", default="LANCZOS", label="插值方法",
                            choices=("LANCZOS", "BICUBIC", "NEAREST")),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0),
)
def run_upscale(ctx, node, inputs):
    image = as_image(inputs.get("image"))
    if image is None:
        raise ValueError("缺少输入 image")

    scale = float(node.params.get("scale", 2.0))
    method = str(node.params.get("method", "LANCZOS") or "LANCZOS").strip()

    tw = max(1, int(image.width * scale))
    th = max(1, int(image.height * scale))
    ctx.status(f"🔍 放大 {scale}x · {image.width}x{image.height} → {tw}x{th} · {method}")

    out = _resize(image, (tw, th), method)
    ctx.check_cancel()

    meta = {"upscale_scale": scale, "upscale_method": method}
    ctx.metadata.update(meta)
    return {"image": out, "metadata": meta}
