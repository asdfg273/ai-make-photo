# nodes/adapter_nodes.py
"""适配器节点：IPAdapterApply（参考图风格/人脸/构图，支持多参考图）。

设计要点
--------
· IP-Adapter 的权重选择（SD1.5 vs SDXL、plus / plus-face / base）**全部**由
  ModelManager 决定，节点只负责"说清楚要哪些参考图 + 各自强度"；
  权重安装/共享 UNet 这种脏活在 core/model_manager.py 里只写一次。
· 节点**不下载任何东西**：缺权重就抛错，让用户去扩展市场装。
· 只通过 ctx 访问宿主，不碰界面控件（工作流要能脱离界面跑）。
· **一图一角色**的精细控制请用 `nodes/ipa_nodes.py` 里的
  IPAdapterIdentity / IPAdapterStyle / IPAdapterComposition（可串联累加）；
  本节点保留的原因有两个：① 老工作流的兼容入口；② 一次性用
  `refs_json` 表达"一批参考图"的场景（例如界面把多个角色导出成一段 JSON）。

多参考图：`refs_json`
--------------------
形状与区域功能的 `regions_json` 完全一致（同一套 JSON 风格，用户学一次就够）::

    [
      {"role":"identity",    "path":"E:/refs/face.png",  "weight":0.8},
      {"role":"style",       "path":"E:/refs/style.jpg", "weight":0.6},
      {"role":"composition", "path":"E:/refs/sky.png",   "weight":0.45,
       "mask_path":"E:/masks/sky.png"},
      {"role":"style", "weight":0.5, "enabled":false}
    ]

· `role` 取 identity / style / composition（也认中文"身份/风格/构图"）；
· `path` 是参考图路径；`mask_path` 可选（该参考图只影响这块区域）；
· `role` 与 `path` 都为空、且连着单张 `image` 输入时 → 用**单图模式**
  （向后兼容：variant + weight 两个参数说了算）。
· 非法 JSON 永远按"空列表"处理并给出提示，绝不抛异常。

`refs_json` 里的条目与 `image` 输入**会合并**（json 在前、连线在后），
这样"连线给一张 + JSON 补几张"也能用。
"""
from __future__ import annotations

import logging

from core import ipa_multi as ipa_mod
from workflow import adapters as adapters_mod
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

_REFS_JSON_HELP = (
    "多参考图定义（JSON 数组），形如：\n"
    '[\n'
    '  {"role":"identity",    "path":"E:/refs/face.png",  "weight":0.8},\n'
    '  {"role":"style",       "path":"E:/refs/style.jpg", "weight":0.6},\n'
    '  {"role":"composition", "path":"E:/refs/sky.png",   "weight":0.45,\n'
    '   "mask_path":"E:/masks/sky.png"}\n'
    ']\n'
    "字段说明：\n"
    "· role：identity(身份=plus-face) / style(风格=plus) / "
    "composition(构图=base)，也认中文「身份/风格/构图」；\n"
    "· path：参考图文件路径；\n"
    "· weight：该参考图独立强度（0~2），最终是每个适配器一个 scale；\n"
    "· mask_path：可选。给了就让这张参考图**只影响该蒙版区域**"
    "（参考图会先按掩码抠图，再由注意力掩码限制）；\n"
    "· enabled=false 可临时停用一条（保留编号对齐，强度按 0 处理）。\n"
    "留空 [] 表示用上面连线的单张 image（旧行为）。\n"
    "JSON 非法时按空列表处理并提示，不会让节点失败。"
)


@node(
    "IPAdapterApply",
    category="适配",
    description="挂载 IP-Adapter：用参考图（可带遮罩）影响风格 / 人脸 / 构图。"
                "支持单张（连线 image）或多张（refs_json，每张独立角色与强度）。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "image": PortSpec(DataType.IMAGE, "参考图", optional=True),
            "mask": PortSpec(DataType.MASK, "遮罩", optional=True)},
    outputs={"model": PortSpec(DataType.MODEL, "模型"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "variant": ParamSpec("choice", default="plus", label="变体",
                             choices=("plus", "plus_face", "light", "standard"),
                             help="单图模式下的权重档位：plus=通用风格，"
                                  "plus_face=人脸，standard=基础版，"
                                  "light=轻量（管理器找不到时会安全回退）"),
        "weight": ParamSpec("float", default=0.6, label="强度",
                            minimum=0.0, maximum=2.0, step=0.05,
                            help="单图模式下的强度；多参考图时每条用自己的 "
                                 "weight 字段，这个值只作为兜底"),
        "refs_json": ParamSpec("text", default="[]", label="多参考图定义",
                               help=_REFS_JSON_HELP),
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

    # ── 能力门禁：看能力，不看模型名 ──
    caps = getattr(ctx, "capabilities", None)
    if caps is None:
        caps = adapters_mod.capabilities_of(model)
    caps.require("ip_adapter")

    ref_image = as_image(inputs.get("image"))
    mask = inputs.get("mask")

    variant_raw = (node.params.get("variant") or "plus").strip().lower()
    variant = _VARIANT_ALIAS.get(variant_raw, variant_raw)
    weight = float(node.params.get("weight", 0.6))

    raw_json = node.params.get("refs_json")
    warnings = []
    refs = []
    if raw_json is not None and str(raw_json).strip() not in ("", "[]", "{}"):
        refs = ipa_mod.references_from_json(raw_json)
        if not refs:
            warnings.append("多参考图 JSON 解析失败（不是合法的 JSON 数组），"
                            "已按空列表处理。")

    # 连线的单张参考图：作为一条 style 参考图追加在 JSON 条目之后
    if ref_image is not None:
        refs.append(ipa_mod.IpaReference(
            role="style", weight=weight,
            image=ref_image.convert("RGB").copy(),
            mask=mask, variant="",
            variant_requested=ipa_mod.normalize_variant(variant)))

    if not refs:
        raise ValueError(
            "缺少参考图：要么连一张 image，要么用 refs_json 给出参考图列表"
            "（IP-Adapter 需要至少一张参考图）")

    # ── 落进 model.meta，交给采样层一次性安装（与 ipa_nodes 同一套约定）──
    from nodes import ipa_nodes
    if getattr(model, "meta", None) is None:
        model.meta = {}
    existing = ipa_nodes.pending_references(model)
    pending = existing + refs
    model.meta[ipa_nodes.META_KEY] = pending
    model.meta["ipa_image"] = pending[0].image if pending[0].image is not None \
        else pending[0].path
    model.meta["ipa_scale"] = max(r.weight for r in pending)
    model.meta["ipa_variant"] = variant_raw

    summary = ipa_mod.describe_references(pending)
    meta = {
        "ip_adapter_enabled": "true",
        "ip_adapter_variant": variant_raw,
        "ip_adapter_weight": weight,
        "ip_adapter_refs": len(pending),
        "ip_adapter_roles": [r.role for r in pending],
        "ip_adapter_weights": [float(r.weight) for r in pending],
        "ip_adapter_summary": summary,
    }
    if warnings:
        meta["ip_adapter_warnings"] = list(warnings)
        existing_w = list(ctx.metadata.get("warnings") or [])
        existing_w.extend(warnings)
        ctx.metadata["warnings"] = existing_w

    ctx.metadata.update(dict(meta))
    ctx.status(f"🎭 IP-Adapter 参考图 {len(pending)} 条：{summary}")
    for w in warnings:
        logger.warning(f"⚠️ {w}")
    logger.info(f"✅ IP-Adapter 已登记: {summary}")
    return {"model": model, "metadata": meta}
