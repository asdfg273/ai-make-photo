# nodes/detailer_nodes.py
"""修复节点：ADetailer（检测 + 局部重绘）。

为什么不在这里重写一遍检测/重绘
-------------------------------
项目里已经有一套跑通过的 ADetailer 实现（utils/image_processor.py）：
它负责 YOLO 检测、按框裁剪、只放大裁剪块重绘、羽化贴回、临时关掉 IPA，
这些都是踩过显存坑之后写出来的细节。节点层只做"翻译"：
把端口的 Conditioning 拆成 prompt / negative_prompt，把参数搬过去，
把结果写进元数据。两套实现只会带来两种不同的结果。

资源声明里的 `asset:adetailer_face` / `asset:adetailer_hand`
-----------------------------------------------------------
这是给资源管理器看的**声明**（不是运行期检查）：没装检测模型时界面
能据此提示"去下载"，而不是等执行到一半才发现。声明了就保留 —— 手部
检测器属于同一类资产，即使本次只修脸也一样列出。
"""
from __future__ import annotations

import logging

from workflow import adapters as adapters_mod
from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import as_image

logger = logging.getLogger(__name__)


def _conditioning_text(cond, attr: str = "text") -> str:
    """从 Conditioning（或裸字符串）里取文本；取不到给空串。"""
    if cond is None:
        return ""
    value = getattr(cond, attr, None)
    if isinstance(value, str):
        return value
    return str(cond) if isinstance(cond, str) else ""


def _find_model_key(models, needle: str):
    """在检测模型表里找第一个名字含 needle 的键（如 "脸" / "手"）。"""
    for key in models:
        if needle in str(key):
            return key
    return None


@node(
    "ADetailer",
    category="修复",
    description="ADetailer 局部精修：自动检测脸部/手部/人物并逐块重绘，"
                "再羽化贴回原图（不整图重绘）。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "image": PortSpec(DataType.IMAGE, "图像"),
            "positive": PortSpec(DataType.TEXT_COND, "正面条件"),
            "negative": PortSpec(DataType.TEXT_COND, "负面条件", optional=True)},
    outputs={"image": PortSpec(DataType.IMAGE, "图像"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "target": ParamSpec("choice", default="二次元脸部", label="修复目标",
                            choices=("二次元脸部", "二次元手部", "现实脸部",
                                     "现实手部", "全身人物")),
        "strength": ParamSpec("float", default=0.35, label="重绘强度",
                              minimum=0.05, maximum=0.95, step=0.05),
        "detail_size": ParamSpec("int", default=512, label="局部重绘尺寸",
                                 minimum=128, maximum=1536,
                                 help="裁剪块重绘的最长边，"
                                      "8GB 显存建议 512"),
        "padding": ParamSpec("float", default=0.15, label="边框留白",
                             minimum=0.0, maximum=0.6, step=0.05),
        "mask_blur": ParamSpec("int", default=8, label="遮罩羽化",
                               minimum=0, maximum=64),
        "both_face_and_hand": ParamSpec("bool", default=False,
                                        label="同时修脸和手",
                                        help="依次执行两次检测重绘，较慢"),
    },
    resources=ResourceRequirement(
        requires=("cuda", "model", "asset:adetailer_face",
                  "asset:adetailer_hand"),
        estimated_vram_mb=1600,
        supports_cpu_offload=True,
        supports_tiled=False),
)
def run_adetailer(ctx, node, inputs):
    # 重依赖 + 项目既有实现都在函数内部 import
    from utils.image_processor import ADETAILER_MODELS, process_adetailer

    model = inputs.get("model")
    image = as_image(inputs.get("image"))
    if model is None:
        raise ValueError("缺少输入 model")
    if image is None:
        raise ValueError("缺少输入 image（ADetailer 需要一张底图）")

    adapter = getattr(model, "adapter", None)
    if adapter is None:
        raise RuntimeError("模型没有适配器，无法确定 inpaint 调用方式")
    if not getattr(adapter.caps, "adetailer", True):
        raise adapters_mod.UnsupportedFeatureError(
            "ADetailer(修脸/修手)", model.arch_id)

    inpaint_pipe = adapter.pipe(model, "inpaint")
    i2i_pipe = adapter.pipe(model, "img2img")
    if inpaint_pipe is None and i2i_pipe is None:
        raise adapters_mod.UnsupportedFeatureError(
            "ADetailer(修脸/修手)", model.arch_id,
            "该架构既没有 inpaint 也没有 img2img 管线")

    target = str(node.params.get("target", "二次元脸部") or "二次元脸部").strip()
    strength = float(node.params.get("strength", 0.35))
    detail_size = int(node.params.get("detail_size", 512))
    padding = float(node.params.get("padding", 0.15))
    mask_blur = int(node.params.get("mask_blur", 8))
    both = bool(node.params.get("both_face_and_hand", False))

    if not ADETAILER_MODELS:
        raise RuntimeError("ADETAILER_MODELS 为空，没有可用的检测模型")
    if target not in ADETAILER_MODELS:
        raise ValueError(
            f"不支持的修复目标 {target!r}，可选: "
            f"{', '.join(str(k) for k in ADETAILER_MODELS)}")

    positive = inputs.get("positive")
    negative = inputs.get("negative")
    prompt = _conditioning_text(positive, "text")
    neg_prompt = (_conditioning_text(negative, "negative_text")
                  or _conditioning_text(negative, "text"))

    # 要跑几遍：普通模式一遍；"同时修脸和手" 时按脸/手各挑一个检测模型
    if not both:
        passes = [target]
    else:
        passes = [k for k in (_find_model_key(ADETAILER_MODELS, "脸"),
                              _find_model_key(ADETAILER_MODELS, "手"))
                  if k is not None]
        if not passes:
            logger.warning("⚠️ 检测模型表里没有脸/手模型，退回只修所选目标")
            passes = [target]
        elif target not in passes:
            # 用户选的目标（如全身人物）不在脸/手里，额外补一遍
            passes.append(target)

    out = image
    for cur_target in passes:
        ctx.check_cancel()
        ctx.status(f"🩹 ADetailer 精修 [{cur_target}] "
                   f"强度 {strength} 尺寸 {detail_size}")
        repaired = process_adetailer(
            out, inpaint_pipe, prompt, neg_prompt,
            strength=strength, target=cur_target,
            padding_ratio=padding, mask_blur=mask_blur,
            detail_size=detail_size,
            img2img_pipe=i2i_pipe)
        if repaired is None:
            raise RuntimeError(f"ADetailer 返回空结果（目标 {cur_target}）")
        out = as_image(repaired)
        if out is None:
            raise RuntimeError(
                "ADetailer 返回了无法识别的图像类型: "
                f"{type(repaired).__name__}")

    meta = {"adetailer_enabled": True,
            "adetailer_target": target,
            "adetailer_strength": strength,
            "adetailer_detail_size": detail_size,
            "adetailer_padding": padding,
            "adetailer_mask_blur": mask_blur,
            "adetailer_both": both}
    ctx.metadata.update(meta)
    return {"image": out, "metadata": meta}
