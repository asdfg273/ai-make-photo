# nodes/sampling_nodes.py
"""采样与潜空间节点：Sampler / VAEEncode / VAEDecode。

为什么拆成"采样出 latent + 单独解码"
------------------------------------
这样"换 VAE 重解码""只重跑解码"“接 Hires 前先看 latent”都变成改一条线的事，
而不是重跑整个流程。Sampler 优先请求 `output_type="latent"`；
若某条管线不支持，则退回"直接解码"并同时给出 image，
下游 VAEDecode 发现已有 image 就透传 —— 保证图仍然跑得通。
"""
from __future__ import annotations

import logging
import random

from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import Latent, as_image

logger = logging.getLogger(__name__)

_SAMPLERS = ("Euler a", "Euler", "DPM++ 2M", "DPM++ 2M Karras",
             "DPM++ SDE Karras", "DDIM", "UniPC", "LCM", "PNDM")


def _resolve_seed(seed: int) -> int:
    if seed is None or int(seed) < 0:
        return random.randint(1, 2_147_483_647)
    return int(seed)


def _generator(device: str, seed: int):
    import torch
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        return torch.Generator().manual_seed(seed)


def _extract_latent(res):
    """从管线返回值里取 latent；取不到返回 None。"""
    try:
        import torch
    except Exception:
        return None
    lat = getattr(res, "latents", None)
    if isinstance(lat, torch.Tensor):
        return lat
    imgs = getattr(res, "images", None)
    if isinstance(imgs, torch.Tensor):
        return imgs
    return None


@node(
    "Sampler",
    category="采样",
    description="采样：文生图 / 图生图 / 局部重绘（有 image 即图生图，再有 mask 即重绘）。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "positive": PortSpec(DataType.TEXT_COND, "正面条件"),
            "negative": PortSpec(DataType.TEXT_COND, "负面条件", optional=True),
            "image": PortSpec(DataType.IMAGE, "初始图", optional=True),
            "mask": PortSpec(DataType.MASK, "遮罩", optional=True)},
    outputs={"latent": PortSpec(DataType.LATENT, "潜空间"),
             "image": PortSpec(DataType.IMAGE, "图像", optional=True),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "steps": ParamSpec("int", default=28, label="步数",
                           minimum=1, maximum=150),
        "cfg": ParamSpec("float", default=7.0, label="CFG",
                         minimum=0.0, maximum=30.0, step=0.5),
        "seed": ParamSpec("seed", default=-1, label="种子",
                          help="-1 表示随机"),
        "width": ParamSpec("int", default=512, label="宽",
                           minimum=64, maximum=8192),
        "height": ParamSpec("int", default=768, label="高",
                            minimum=64, maximum=8192),
        "sampler": ParamSpec("choice", default="Euler a", label="采样器",
                             choices=_SAMPLERS),
        "strength": ParamSpec("float", default=0.55, label="重绘强度",
                              minimum=0.0, maximum=1.0, step=0.05),
    },
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=2600,
                                  supports_cpu_offload=True,
                                  supports_tiled=True,
                                  exclusive_gpu=True, needs_model=True),
)
def run_sampler(ctx, node, inputs):
    import torch

    model = inputs.get("model")
    if model is None:
        raise ValueError("缺少输入 model")
    pos = inputs.get("positive")
    neg = inputs.get("negative")
    image = as_image(inputs.get("image"))
    mask = inputs.get("mask")

    adapter = model.adapter
    kind = "txt2img"
    if image is not None:
        kind = "inpaint" if mask is not None else "img2img"
    pipe = adapter.pipe(model, kind)
    if pipe is None:
        raise RuntimeError(f"模型没有可用的 {kind} 管线（该架构可能不支持）")
    if kind != "txt2img" and not getattr(adapter.caps, kind, True):
        raise RuntimeError(f"{model.arch_id} 架构不支持 {kind}")

    steps = int(node.params.get("steps", 28))
    cfg = float(node.params.get("cfg", adapter.default_guidance()))
    width = int(node.params.get("width", 512))
    height = int(node.params.get("height", 768))
    seed = _resolve_seed(node.params.get("seed", -1))
    sampler = (node.params.get("sampler") or "").strip()

    if sampler:
        try:
            ctx.ai.switch_sampler(sampler)
        except Exception as e:
            logger.warning(f"⚠️ 采样器切换失败（沿用当前）: {e}")

    kw = adapter.sample_kwargs(pos, steps=steps, cfg=cfg, seed=seed,
                               width=width, height=height)
    # 负面条件优先用独立输入
    if neg is not None and getattr(neg, "negative_embeds", None) is not None:
        kw["negative_prompt_embeds"] = neg.negative_embeds
        if getattr(neg, "negative_pooled", None) is not None:
            kw["negative_pooled_prompt_embeds"] = neg.negative_pooled

    if kind in ("img2img", "inpaint"):
        kw["image"] = image
        kw["strength"] = float(node.params.get("strength", 0.55))
        if kind == "inpaint":
            kw["mask_image"] = mask
        # 图生图尺寸跟输入图，避免被拉伸
        kw.pop("width", None)
        kw.pop("height", None)
        width, height = image.width, image.height

    gen = _generator(ctx.device, seed)
    kw["generator"] = gen

    ctx.status(f"🎨 采样 {steps} 步 · {width}x{height} · seed {seed}")
    latent = None
    out_image = None

    def _call(**extra):
        with torch.inference_mode():
            return pipe(**{**kw, **extra})

    try:
        res = _call(output_type="latent")
        latent = _extract_latent(res)
    except (TypeError, ValueError) as e:
        # 管线不认 output_type="latent"：下面用 latent is None 走直接解码
        logger.info(f"ℹ️ 该管线不支持 output_type=latent，改为直接解码: {e}")
    ctx.check_cancel()

    if latent is None:
        res = _call()
        imgs = getattr(res, "images", None)
        if imgs is None:
            raise RuntimeError("管线没有返回图像")
        if isinstance(imgs, torch.Tensor):
            latent = imgs
        else:
            out_image = imgs[0] if isinstance(imgs, (list, tuple)) else imgs

    meta = {"steps": steps, "cfg": cfg, "seed": seed, "sampler": sampler,
            "width": width, "height": height, "workflow_kind": kind,
            "strength": float(node.params.get("strength", 0.55))
            if kind != "txt2img" else ""}
    ctx.metadata.update(meta)
    ctx.progress(steps, steps)

    return {"latent": Latent(tensor=latent, is_noise=False),
            "image": out_image, "metadata": meta}


@node(
    "VAEEncode",
    category="潜空间",
    description="把图像编码到潜空间（图生图 / 重绘 / 分块放大的入口）。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"latent": PortSpec(DataType.LATENT, "潜空间")},
    params={},
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=1200,
                                  supports_cpu_offload=True,
                                  supports_tiled=True),
)
def run_vae_encode(ctx, node, inputs):
    import torch

    model = inputs.get("model")
    image = as_image(inputs.get("image"))
    if model is None or image is None:
        raise ValueError("缺少 model 或 image")

    vae = getattr(model, "vae", None) or getattr(model.pipe, "vae", None)
    if vae is None:
        raise RuntimeError("模型没有 VAE，无法编码")

    import numpy as np
    with torch.inference_mode():
        arr = np.asarray(image.convert("RGB"), dtype="float32") / 127.5 - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = t.to(device=vae.device, dtype=vae.dtype)
        lat = vae.encode(t).latent_dist.sample()
        if hasattr(vae.config, "scaling_factor"):
            lat = lat * vae.config.scaling_factor
    return {"latent": Latent(tensor=lat)}


@node(
    "VAEDecode",
    category="潜空间",
    description="把潜空间解码成图像。若上游已给出图像则直接透传。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "latent": PortSpec(DataType.LATENT, "潜空间", optional=True),
            "image": PortSpec(DataType.IMAGE, "图像", optional=True)},
    outputs={"image": PortSpec(DataType.IMAGE, "图像")},
    params={},
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=1400,
                                  supports_cpu_offload=True,
                                  supports_tiled=True),
)
def run_vae_decode(ctx, node, inputs):
    import torch

    passthrough = as_image(inputs.get("image"))
    if passthrough is not None:
        return {"image": passthrough}     # 上游已解码，无需再来一次

    model = inputs.get("model")
    lat = inputs.get("latent")
    if model is None or lat is None or getattr(lat, "tensor", None) is None:
        raise ValueError("缺少 latent（也没有可透传的 image）")

    vae = getattr(model, "vae", None) or getattr(model.pipe, "vae", None)
    if vae is None:
        raise RuntimeError("模型没有 VAE，无法解码")

    t = lat.tensor
    if hasattr(vae.config, "scaling_factor"):
        t = t / vae.config.scaling_factor
    with torch.inference_mode():
        t = t.to(device=vae.device, dtype=vae.dtype)
        img = vae.decode(t).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        arr = (img[0].permute(1, 2, 0).float().cpu().numpy() * 255).round()
    from PIL import Image
    return {"image": Image.fromarray(arr.astype("uint8"), "RGB")}
