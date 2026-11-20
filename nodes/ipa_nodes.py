# nodes/ipa_nodes.py
"""多参考图 IP-Adapter 节点：Identity / Style / Composition（角色 + 权重）。

要解决的问题
------------
旧版 `IPAdapterApply` 只能表达"一张参考图 + 一个强度"。真实需求是：

    参考图 A：人物身份  → IPAdapterIdentity
    参考图 B：服装风格  → IPAdapterStyle
    参考图 C：画面构图  → IPAdapterComposition

三个角色在 h94/IP-Adapter 里对应**三个不同的权重文件**（已核对
`ModelManager._IPA_WEIGHTS` 与仓库布局，详见 `core/ipa_multi.py` 顶部注释）：

    identity(身份)    → plus-face  ip-adapter-plus-face_sd15.safetensors
    style(风格)       → plus       ip-adapter-plus_sd15.safetensors
    composition(构图) → base       ip-adapter_sd15.safetensors

设计要点（务必保持）
--------------------
1. **累加，而不是多次安装。**
   链式串接三个角色节点，最终必须变成**一次** IP-Adapter 应用（N 个适配器
   一次性装上 UNet、每适配器一个 scale），而不是跑三遍采样。
   所以这里的节点**只登记意图**（把参考图追加进 `model.meta["ipa_refs"]`），
   真正的安装发生在采样前一刻 —— 那时才知道一共有几条参考图。
   这与 LoRA 的做法一致：`LoadLoRA` 也只登记，`apply_multiple_loras` 一次挂上。

2. **缺权重文件要优雅降级，不能整体失败。**
   本机只装了 `base`（扩展市场 `ip_adapter_sd15` 装的就是它）。此时请求
   `identity` 拿不到 `plus-face` → 退用 `base`，**保留该角色的权重语义**，
   并把中文说明写进 `ctx.metadata["warnings"]`。绝不静默，也绝不失败整节点。

3. **能力门禁看能力，不看模型名。**
   `caps.require("ip_adapter")`：不支持的架构（Anima 等）在**运行前**就报清楚。

4. **区域（可选 mask 输入）。**
   带 mask 的参考图只影响该区域：一半在参考图侧（按掩码抠图，避免区域外
   内容进 CLIP），一半在注意力侧（`ip_adapter_masks`）。构造见
   `core/ipa_multi.build_ip_adapter_masks`，由采样层透传。

5. 节点**不下载任何东西**：缺权重就直接说清楚去哪装。
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

#: `model.meta` 里存放累积参考图的键（采样层读它）
META_KEY = "ipa_refs"


# ----------------------------------------------------------------------
#  共用实现
# ----------------------------------------------------------------------
def _collect_refs(model) -> list:
    """读取当前已累积的参考图（容错：meta 里可能是 JSON、dict 列表或空）。"""
    meta = getattr(model, "meta", None)
    if not isinstance(meta, dict):
        return []
    raw = meta.get(META_KEY)
    if raw is None:
        raw = meta.get("ipa_image")
        if raw is not None:
            # 兼容旧版单参考图写法（没有角色信息 → 按 style 处理）
            return [ipa_mod.IpaReference(role="style", weight=float(
                meta.get("ipa_scale", ipa_mod.DEFAULT_WEIGHT) or
                ipa_mod.DEFAULT_WEIGHT), image=raw,
                variant=ipa_mod.normalize_variant(meta.get("ipa_variant", ""))
                if meta.get("ipa_variant") else "")]
        return []
    try:
        return ipa_mod.coerce_references(raw)
    except Exception as e:                    # pragma: no cover - 防御
        logger.warning(f"⚠️ 已有参考图无法解析，按空处理: {e}")
        return []


def add_reference(ctx, node_obj, inputs, *, role: str):
    """三个角色节点共用的实现：登记一条参考图（**不安装权重**）。

    返回 `{"model": model, "metadata": meta}`。
    """
    role = ipa_mod.normalize_role(role)
    label = ipa_mod.role_label(role)

    model = inputs.get("model")
    if model is None:
        raise ValueError("缺少输入 model（IP-Adapter 需要一个模型）")

    # ── 能力门禁：看能力，不看模型名 ──
    caps = getattr(ctx, "capabilities", None)
    if caps is None:
        caps = adapters_mod.capabilities_of(model)
    caps.require("ip_adapter")

    if ctx.ai is None:
        raise RuntimeError(
            "没有可用的模型管理器（ctx.ai 为空），无法挂载 IP-Adapter")

    enabled = bool(node_obj.params.get("enabled", True))
    weight = float(node_obj.params.get("weight", ipa_mod.DEFAULT_WEIGHT))
    weight = max(0.0, min(2.0, weight))
    start_at = float(node_obj.params.get("start_at", 0.0))
    end_at = float(node_obj.params.get("end_at", 1.0))
    if end_at < start_at:
        start_at, end_at = 0.0, 1.0

    ref_image = as_image(inputs.get("image"))
    if ref_image is None:
        raise ValueError(f"缺少输入 image（{label}参考图）")
    # ⚠️ 绝不改动调用方的 PIL 对象：copy 一份再往下传
    ref_image = ref_image.convert("RGB").copy()
    mask = inputs.get("mask")

    warnings: list = []

    # ── 该角色**理想**该用哪个档位；本机装了没有 ──
    want_variant = ipa_mod.variant_for_role(role)
    try:
        installed = ctx.ai.installed_ipa_variants()
    except Exception as e:
        logger.warning(f"⚠️ 无法列出已安装的 IP-Adapter 档位: {e}")
        installed = []

    if enabled and installed and want_variant not in installed:
        warnings.append(
            f"{label}参考图需要档位「{want_variant}」"
            f"（{want_variant} 是 {label}专用权重），本机只装了 "
            f"{installed} → 已降级用「{installed[0]}」，"
            f"{label}角色语义未生效，仅权重 {weight:.2f} 生效。"
            f"想要完整效果请安装 {want_variant} 对应的权重文件到 "
            f"models/ip_adapter/。")
        # 记下"哪条参考图的哪个档位被降级了"：采样层据此去重，
        # 同一个问题不会在元数据里写成两条（用户会以为出了两个问题）。
        degraded = list(ctx.metadata.get("ipa_degraded") or [])
        degraded.append({"variant": want_variant, "role": role,
                         "ref_index": len(_collect_refs(model)) + 1})
        ctx.metadata["ipa_degraded"] = degraded

    # 起止步（start_at / end_at）：**如实说明现状**。
    # diffusers 0.40 里 IP-Adapter 的 scale 只能按"注意力层"设
    # （IPAdapterMixin.set_ip_adapter_scale），没有逐参考图的"从第几步到第几步"
    # 入参 —— 采样过程中也没有回调可以去改它（没有 callback_on_step_end）。
    # 所以这两个参数目前只被**记录**，不改变采样。既然做不到就说出来，
    # 不能让用户以为设了 0.2~0.8 就真的只在中段生效。
    if enabled and (start_at, end_at) != (0.0, 1.0):
        warnings.append(
            f"{label}参考图的起止步 {start_at:.2f}~{end_at:.2f} 已记录，"
            f"但当前 diffusers 0.40 的 IP-Adapter 只支持整段生效"
            f"（没有逐参考图的步数窗口，也没有逐步回调可改 scale）。"
            f"本次采样中该参考图**全程生效**。")

    ref = ipa_mod.IpaReference(
        role=role,
        weight=weight,
        image=ref_image,
        mask=mask,
        variant=want_variant if want_variant in installed else "",
        variant_requested=want_variant,
        enabled=enabled,
    )

    refs = _collect_refs(model)
    refs.append(ref)

    if getattr(model, "meta", None) is None:
        model.meta = {}
    model.meta[META_KEY] = refs
    # 时间范围：整个管线一次性调用时"第几个参考图"才有意义，
    # 这里只记录，真正生效由采样层决定（目前 SD1.5 只支持全时段生效，
    # 不支持逐参考图起止步 —— 见节点说明与 README）。
    ranges = list(model.meta.get("ipa_ranges") or [])
    ranges.append((start_at, end_at))
    model.meta["ipa_ranges"] = ranges
    # 兼容旧字段（下游老代码/元数据写入还会读它们）
    model.meta["ipa_image"] = refs[0].image
    model.meta["ipa_scale"] = max(r.weight for r in refs) if refs else weight
    model.meta["ipa_variant"] = ref.variant or want_variant

    # ── 元数据（人看得懂）──
    summary = ipa_mod.describe_references(refs)
    meta = {
        "ip_adapter_enabled": "true" if any(r.enabled for r in refs) else "false",
        "ip_adapter_roles": [r.role for r in refs],
        "ip_adapter_refs": len(refs),
        "ip_adapter_variants": [r.variant or r.variant_requested for r in refs],
        "ip_adapter_weights": [float(r.weight) for r in refs],
        "ip_adapter_summary": summary,
        "ip_adapter_last_role": role,
        "ip_adapter_weight": weight,
        "ip_adapter_variant": ref.variant or want_variant,
    }
    if warnings:
        existing = list(ctx.metadata.get("warnings") or [])
        existing.extend(warnings)
        ctx.metadata["warnings"] = existing
        meta["ip_adapter_warnings"] = list(warnings)

    ctx.metadata.update({k: v for k, v in meta.items()})
    ctx.metadata[META_KEY] = [r.to_dict() for r in refs]
    ctx.status(f"🎭 {label}参考图已加入（共 {len(refs)} 条，权重 {weight:.2f}"
               + ("，已停用" if not enabled else "") + "）")
    for w in warnings:
        logger.warning(f"⚠️ {w}")
        try:
            ctx.log(f"⚠️ {w}")
        except Exception:
            pass

    return {"model": model, "metadata": meta}


# ----------------------------------------------------------------------
#  三个角色节点
# ----------------------------------------------------------------------
_ROLE_PARAMS = {
    "weight": ParamSpec("float", default=0.6, label="强度",
                        minimum=0.0, maximum=2.0, step=0.05,
                        help="该参考图的影响强度；多个参考图各有各的强度"
                             "（最终是每个适配器一个独立 scale）"),
    "start_at": ParamSpec("float", default=0.0, label="起始步",
                          minimum=0.0, maximum=1.0, step=0.05,
                          help="从采样的百分之几开始生效。"
                               "⚠️ 诚实说明：diffusers 0.40 的 IP-Adapter 没有"
                               "逐参考图的步数窗口，这个值目前只被记录"
                               "（非默认值会在 warnings 里说明），采样时全程生效"),
    "end_at": ParamSpec("float", default=1.0, label="结束步",
                        minimum=0.0, maximum=1.0, step=0.05,
                        help="到采样的百分之几停止生效。"
                             "⚠️ 同上：当前 diffusers 不支持逐参考图步数窗口，"
                             "该值只被记录"),
    "enabled": ParamSpec("bool", default=True, label="启用",
                         help="关掉后这条参考图仍然占位（保持编号对齐），"
                              "但强度按 0 处理，不影响画面"),
}

_ROLE_RESOURCES = ResourceRequirement(
    requires=("cuda", "model", "asset:ip_adapter_sd15"),
    estimated_vram_mb=1500,
    supports_cpu_offload=True)

_IDENTITY_DOC = (
    "身份参考图：用人脸专用的 plus-face 权重把参考图里的人物身份带到画面里。\n"
    "· 角色 → 权重映射：身份 = plus-face（ip-adapter-plus-face_sd15.safetensors）；\n"
    "· 本机没装 plus-face 时会**降级**用已安装的档位（保留本节点的强度语义），"
    "并在 warnings 里说明，不会让工作流失败；\n"
    "· 多个角色节点串起来会**合并成一次** IP-Adapter 应用（不是跑多遍）。\n"
    "· 接一个 mask 可让这张参考图只影响蒙版区域（参考图会先按掩码抠图，"
    "再由注意力掩码限制，避免区域外内容进 CLIP）。"
)
_STYLE_DOC = (
    "风格参考图：用 plus 权重把参考图的画风/质感/配色带到画面里。\n"
    "· 角色 → 权重映射：风格 = plus（ip-adapter-plus_sd15.safetensors）；\n"
    "· 本机没装 plus 时降级用已安装的档位并写 warnings；\n"
    "· 与 Identity / Composition 串联时合并成一次安装、每张图一个独立强度。\n"
    "· 接 mask 可让风格只作用于指定区域。"
)
_COMPOSITION_DOC = (
    "构图参考图：用基础版 base 权重引导整体构图/布局/透视。\n"
    "· 角色 → 权重映射：构图 = base（ip-adapter_sd15.safetensors）；\n"
    "· base 是扩展市场 ip_adapter_sd15 默认装的档位，所以构图角色"
    "通常无需额外下载；\n"
    "· 与 Identity / Style 串联时合并成一次安装。\n"
    "· 接 mask 可让构图只作用于指定区域（例如只保留参考图的天空布局）。"
)


def _make_role_node(node_type: str, role: str, description: str):
    """生成一个角色节点（三个节点的输入/输出/参数**完全一致**，只有角色不同）。

    为什么用工厂：三个节点的接口必须严格一致，手抄三遍迟早会漂移
    （历史上 LoRA/IPA 都出现过"两个地方说法不一样"的问题）。注册出来的
    NodeSpec 与手写完全等价。
    """
    @node(
        node_type,
        category="适配",
        description=description,
        inputs={"model": PortSpec(DataType.MODEL, "模型"),
                "image": PortSpec(DataType.IMAGE, "参考图"),
                "mask": PortSpec(DataType.MASK, "遮罩", optional=True)},
        outputs={"model": PortSpec(DataType.MODEL, "模型"),
                 "metadata": PortSpec(DataType.METADATA, "信息")},
        params=dict(_ROLE_PARAMS),
        resources=_ROLE_RESOURCES,
    )
    def _run(ctx, node_obj, inputs):
        return add_reference(ctx, node_obj, inputs, role=role)

    _run.__name__ = f"run_ip_adapter_{role}"
    _run.__doc__ = description
    return _run


run_ip_adapter_identity = _make_role_node(
    "IPAdapterIdentity", "identity", _IDENTITY_DOC)
run_ip_adapter_style = _make_role_node(
    "IPAdapterStyle", "style", _STYLE_DOC)
run_ip_adapter_composition = _make_role_node(
    "IPAdapterComposition", "composition", _COMPOSITION_DOC)


# ----------------------------------------------------------------------
#  给采样层用的小工具（不在节点里做，因为它们要被 Sampler 调用）
# ----------------------------------------------------------------------
def clear_references(model):
    """清空模型上的累积参考图（LoadCheckpoint 用它保证"干净基线"）。

    为什么必须清：`ModelRef.meta` 是会跨节点传递的同一个 dict；如果上一轮
    运行留下的参考图不清掉，这一轮"没接 IP-Adapter"的工作流会**继承**上一轮
    的参考图 —— 同一组参数跑两次结果不同，用户完全无从判断。
    （LoRA 有完全一样的坑，见 model_nodes.load_checkpoint 里的注释。）
    """
    meta = getattr(model, "meta", None)
    if not isinstance(meta, dict):
        return 0
    n = len(ipa_mod.coerce_references(meta.get(META_KEY) or []))
    for k in (META_KEY, "ipa_ranges", "ipa_image", "ipa_scale", "ipa_variant",
              "ipa_mask"):
        meta.pop(k, None)
    return n


def pending_references(model) -> list:
    """采样层：取出本轮的参考图列表（已含角色/权重）。"""
    return _collect_refs(model)


__all__ = [
    "META_KEY", "add_reference", "clear_references",
    "pending_references", "run_ip_adapter_identity", "run_ip_adapter_style",
    "run_ip_adapter_composition",
]
