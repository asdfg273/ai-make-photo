# nodes/adapter_nodes.py
"""适配器节点：IPAdapterApply（参考图风格/人脸迁移）。

设计要点
--------
· IP-Adapter 的权重选择（SD1.5 vs SDXL、plus / plus-face / base）**全部**由
  ModelManager 决定，节点只负责"说清楚要哪个变体 + 强度"；
  权重安装/共享 UNet 这种脏活在 core/model_manager.py 里只写一次。
· 节点**不下载任何东西**：缺权重就抛错，让用户去扩展市场装。
· 只通过 ctx 访问宿主，不碰界面控件（工作流要能脱离界面跑）。
"""
from __future__ import annotations

import logging

from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import as_image

logger = logging.getLogger(__name__)

# 节点暴露的变体名 -> ModelManager 认的变体名
_VARIANT_ALIAS = {
    "plus": "plus",
    "plus_face": "plus-face",
    "light": "light",       # 管理器没有对应权重时会自行回退到同架构安全档位
    "standard": "base",
}


@node(
    "IPAdapterApply",
    category="适配",
    description="挂载 IP-Adapter：用参考图（可带遮罩）影响风格 / 人脸 / 构图。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "image": PortSpec(DataType.IMAGE, "参考图"),
            "mask": PortSpec(DataType.MASK, "遮罩", optional=True)},
    outputs={"model": PortSpec(DataType.MODEL, "模型"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "variant": ParamSpec("choice", default="plus", label="变体",
                             choices=("plus", "plus_face", "light", "standard"),
                             help="plus=通用风格，plus_face=人脸，"
                                  "standard=基础版，light=轻量"),
        "weight": ParamSpec("float", default=0.6, label="强度",
                            minimum=0.0, maximum=2.0, step=0.05),
    },
    resources=ResourceRequirement(requires=("cuda", "model",
                                            "asset:ip_adapter_sd15"),
                                  estimated_vram_mb=1500,
                                  supports_cpu_offload=True),
)
def run_ip_adapter_apply(ctx, node, inputs):
    model = inputs.get("model")
    if model is None:
        raise ValueError("缺少输入 model")
    if ctx.ai is None:
        raise RuntimeError("没有可用的模型管理器（ctx.ai 为空），无法加载 IP-Adapter")

    ref_image = as_image(inputs.get("image"))
    if ref_image is None:
        raise ValueError("缺少输入 image（IP-Adapter 需要一张参考图）")
    mask = inputs.get("mask")

    variant_raw = (node.params.get("variant") or "plus").strip().lower()
    variant = _VARIANT_ALIAS.get(variant_raw, variant_raw)
    weight = float(node.params.get("weight", 0.6))

    # ── 1) 安装 IP-Adapter（架构感知由 ModelManager 负责；这里不下载）──
    prepare = getattr(ctx.ai, "prepare_ip_adapter", None)
    if not callable(prepare):
        raise RuntimeError(
            "❌ 当前模型管理器不支持 IP-Adapter（缺少 prepare_ip_adapter 方法）。")
    ctx.status(f"🎭 挂载 IP-Adapter [{variant_raw}] 强度 {weight} ...")
    ctx.check_cancel()
    try:
        prepare(variant)
    except Exception as e:
        raise RuntimeError(
            f"❌ 挂载 IP-Adapter 失败（variant={variant_raw}）："
            f"{type(e).__name__}: {e}\n"
            "   常见原因：IP-Adapter 权重尚未安装 —— 请在**扩展市场**安装"
            "（asset: ip_adapter_sd15 / ip_adapter_sdxl），"
            "或手动把 ip-adapter-*.safetensors 放到 models/ip_adapter/ 下；"
            "也可能是变体与当前底模架构不匹配（SD1.5 ⇄ SDXL）。\n"
            "   本节点不会自动下载任何权重。"
        ) from e
    ctx.check_cancel()

    # ── 2) 强度（scale 会同步到全部管线）──
    setter = getattr(ctx.ai, "set_ip_adapter_scale", None)
    if callable(setter):
        try:
            setter(weight)
        except Exception as e:
            logger.warning(f"⚠️ 设置 IP-Adapter 强度失败（沿用当前值）: {e}")

    # ── 3) 把参考图挂到模型上（下游采样节点读取）──
    if getattr(model, "meta", None) is None:
        model.meta = {}
    model.meta["ipa_image"] = ref_image
    model.meta["ipa_scale"] = weight
    model.meta["ipa_variant"] = variant_raw
    if mask is not None:
        # 遮罩暂由采样层决定是否使用；保留下来便于调试与后续扩展
        model.meta["ipa_mask"] = mask

    meta = {"ip_adapter_enabled": "true",
            "ip_adapter_variant": variant_raw,
            "ip_adapter_weight": weight}
    ctx.metadata.update(dict(meta))
    logger.info(f"✅ IP-Adapter 已应用: {variant_raw}({variant}) "
                f"scale={weight}")
    return {"model": model, "metadata": meta}
