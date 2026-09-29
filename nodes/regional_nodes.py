# nodes/regional_nodes.py
"""区域节点：RegionalPrompt / RegionalRedraw（区域提示词与局部重绘）。

为什么分成"定义区域"和"执行区域"两步
------------------------------------
和 ControlNet 的"预处理 / 应用"同一个道理：区域定义是**纯数据**
（`list[Region]`，可缓存、可序列化、可只改一个区域重跑），而"逐个区域
采样"是重活（要模型、要显存、要时间）。拆开之后：

· 界面上的可视化区域编辑器只需要产出/读取一个 `REGIONS` 值，不必碰模型；
· 工作流里可以一个 `RegionalPrompt` 喂给多个 `RegionalRedraw`
  （例如先局部精修、再单独对某个区域跑一次不同参数）；
· 区域定义能直接存进工作流 JSON（`regions_json` 参数）。

真正的算法在 `core/regional.py`（无界面依赖，可脱离 PyQt 调用）：顺序蒙版
局部重绘。本模块只做"翻译"——参数 → 引擎调用 → 元数据，与其它节点模块的
约定保持一致。
"""
from __future__ import annotations

import logging

from core import regional as regional_mod
from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import as_image

logger = logging.getLogger(__name__)

# RegionalPrompt 没有输入图，做几何校验时用的基准分辨率
_DEFAULT_W = 512
_DEFAULT_H = 768

# JSON 形状示例：放在 help 里，用户不用翻文档就能填对
_JSON_HELP = (
    "区域定义的 JSON 数组，形如：\n"
    '[\n'
    '  {"name":"脸部","prompt":"beautiful detailed face",\n'
    '   "negative":"bad face","weight":0.55,"shape":"rect",\n'
    '   "rect":[0.3,0.1,0.7,0.45],"feather":16,\n'
    '   "controlnet":{"kind":"openpose","weight":0.8,"image_path":""},\n'
    '   "adetailer":{"target":"二次元脸部","strength":0.35}},\n'
    '  {"name":"衣服","prompt":"red silk dress","weight":0.45,\n'
    '   "shape":"rect","rect":[0.2,0.45,0.8,0.95]},\n'
    '  {"name":"徽章","prompt":"golden badge","weight":0.8,\n'
    '   "shape":"mask","mask_path":"E:/masks/badge.png"}\n'
    ']\n'
    "字段说明：\n"
    "· rect 是**归一化**坐标 (x0,y0,x1,y1)，0~1，相对整图（换尺寸不用改坐标）；\n"
    "· weight 就是该区域的重绘强度（0~1，越大改动越狠）；\n"
    "· shape=mask 时用 mask_path 指定的黑白图（白=重绘），rect 忽略；\n"
    "· negative 是**局部负面提示词**，留空则只用全局负面词；\n"
    "· feather 是该区域边缘羽化像素；enabled=false 可临时停用某个区域；\n"
    "· color 只给界面画框用，引擎完全忽略。\n"
    "主界面的区域编辑器会可视化地生成这段 JSON（画框/涂蒙版/填权重），"
    "在这里手写也可以，格式非法时会按空列表处理并给出提示。"
)


# ----------------------------------------------------------------------
@node(
    "RegionalPrompt",
    category="区域",
    description="定义区域提示词（矩形/蒙版、区域权重、局部负面词、"
                "局部 ControlNet、局部 ADetailer），供 RegionalRedraw 使用。",
    inputs={},
    outputs={"regions": PortSpec(DataType.REGIONS, "区域定义"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "regions_json": ParamSpec("text", default="[]", label="区域定义",
                                  help=_JSON_HELP),
        "preset_name": ParamSpec("str", default="", label="备注",
                                 help="给这组区域起个名字，只写进元数据，"
                                      "方便在图库里认出用了哪套区域"),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0,
                                  supports_cpu_offload=True,
                                  supports_tiled=True),
)
def run_regional_prompt(ctx, node, inputs):
    """解析区域定义 → REGIONS 值。纯 CPU，不碰模型，因此可以自由重跑。"""
    raw = node.params.get("regions_json")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = "[]"

    regions = regional_mod.regions_from_json(raw)
    note = (node.params.get("preset_name") or "").strip()

    # 没有底图，用默认分辨率做几何校验（能查出空矩形/越界/权重越界这些）
    issues = regional_mod.validate_regions(regions, _DEFAULT_W, _DEFAULT_H)
    text = regional_mod.describe(regions)

    # 解析失败时给出可诊断的提示，而不是静默变成一个空列表
    if isinstance(raw, str) and raw.strip() not in ("", "[]") and not regions:
        issues.insert(0, "区域 JSON 解析失败（不是合法的 JSON 数组），"
                         "已按空列表处理")

    if issues:
        for msg in issues:
            logger.warning(f"⚠️ 区域定义: {msg}")
    warning = "；".join(issues)
    ctx.status(f"🗺️ {('备注 ' + note + ' · ') if note else ''}{text}"
               + (f" ⚠️ {len(issues)} 个问题" if issues else ""))

    meta = {
        "regions": len(regions),
        "region_summary": text,
        "region_preset": note,
        "region_issues": issues,
        "region_warning": warning,
    }
    ctx.metadata["regions"] = regions
    ctx.metadata.update({k: v for k, v in meta.items() if k != "regions"})
    return {"regions": regional_mod.RegionsValue(regions), "metadata": meta}


# ----------------------------------------------------------------------
@node(
    "RegionalRedraw",
    category="区域",
    description="区域精修（局部重绘）：按区域逐个用各自的提示词/权重做蒙版"
                "重绘，再羽化合成回底图。",
    inputs={"image": PortSpec(DataType.IMAGE, "底图"),
            "model": PortSpec(DataType.MODEL, "模型"),
            "regions": PortSpec(DataType.REGIONS, "区域定义"),
            "positive": PortSpec(DataType.TEXT_COND, "正面条件",
                                 optional=True),
            "negative": PortSpec(DataType.TEXT_COND, "负面条件",
                                 optional=True)},
    outputs={"image": PortSpec(DataType.IMAGE, "图像"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "steps": ParamSpec("int", default=24, label="步数",
                           minimum=1, maximum=150),
        "cfg": ParamSpec("float", default=7.0, label="CFG",
                         minimum=0.0, maximum=30.0, step=0.5),
        "seed": ParamSpec("seed", default=-1, label="种子",
                          help="-1 表示随机；每个区域用 种子+序号，"
                               "所以区域之间不会互相抄同一份噪声"),
        "global_negative": ParamSpec("str", default="", label="全局负面提示词",
                                     help="区域自己没写局部负面词时用这一条；"
                                          "两边都空则不送负面条件"),
        "only_region": ParamSpec("str", default="", label="只跑指定区域",
                                 help="填区域名（多个用逗号分隔），"
                                      "留空=跑全部区域"),
        "max_regions": ParamSpec("int", default=8, label="最多区域数",
                                 minimum=0, maximum=64,
                                 help="安全阀：区域越多越慢（每个区域一次采样），"
                                      "0 表示不限制"),
        "feather": ParamSpec("int", default=-1, label="统一羽化",
                             minimum=-1, maximum=128,
                             help="填 0~128 会**覆盖**所有区域的羽化值；"
                                  "-1 表示尊重每个区域自己的羽化"),
    },
    # 显存：区域是**逐个**跑的（见 core/regional.py 的路线说明），峰值取决于
    # 单次 inpaint，与区域数量无关。所以这里按 Inpaint 节点量级声明 2600MB，
    # 而不是"2600 × 区域数"——那样 8GB 卡会被显存策略直接判成跑不动。
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=2600,
                                  supports_cpu_offload=True,
                                  supports_tiled=True,
                                  exclusive_gpu=True, needs_model=True),
)
def run_regional_redraw(ctx, node, inputs):
    """按区域逐个做蒙版重绘，返回合成后的图与整段元数据。"""
    from workflow import adapters as adapters_mod
    from workflow.schema import Cancelled

    image = as_image(inputs.get("image"))
    model = inputs.get("model")
    # 区域定义逐条规范化：外部（界面配置 / 旧工作流文件）给的是 dict，
    # 引擎认的是 Region；这一步做容错转换，缺字段用默认值（绝不抛）。
    regions = regional_mod.coerce_regions(inputs.get("regions"))

    if image is None:
        raise ValueError("缺少输入 image（区域重绘需要一张底图）")
    if model is None:
        raise ValueError("缺少输入 model")
    if not regions:
        raise ValueError("缺少输入 regions（请接 RegionalPrompt，"
                         "或至少定义一个 enabled 的区域）")
    image = image.convert("RGB")

    # 能力门禁：看**能力**，不看模型名（没有 inpaint 的架构直接报清楚）
    caps = getattr(ctx, "capabilities", None)
    if caps is None:
        caps = adapters_mod.capabilities_of(model)
    caps.require("inpaint")

    adapter = getattr(model, "adapter", None) or adapters_mod.adapter_for(model)
    if adapter.pipe(model, "inpaint") is None:
        raise adapters_mod.UnsupportedFeatureError(
            "局部重绘", getattr(model, "arch_id", "") or "",
            "该架构没有可用的 inpaint 管线")

    steps = int(node.params.get("steps", 24) or 24)
    cfg = float(node.params.get("cfg", adapter.default_guidance()))
    seed = node.params.get("seed", -1)
    try:
        seed = int(seed)
    except Exception:
        seed = -1
    global_negative = node.params.get("global_negative") or ""
    only_region = node.params.get("only_region") or ""
    max_regions = int(node.params.get("max_regions", 8) or 0)

    # 全局负面词：独立输入优先（与 Sampler / Inpaint 保持一致），其次参数
    neg = inputs.get("negative")
    neg_text = ""
    if neg is not None:
        neg_text = (getattr(neg, "negative_text", "") or
                    getattr(neg, "text", "") or "")
    if not neg_text:
        neg_text = global_negative

    feather = int(node.params.get("feather", -1) or -1)
    feather = feather if feather >= 0 else None

    ctx.status(f"🗺️ 区域精修准备：{regional_mod.describe(regions)}")
    ctx.check_cancel()

    out, meta = regional_mod.apply_regions(
        ctx.ai, model, image, regions,
        global_negative=neg_text,
        steps=steps, cfg=cfg, seed=seed,
        progress=lambda i, n: ctx.progress(i, n),
        cancel=lambda: ctx.cancelled(),
        status=lambda msg: ctx.status(msg),
        log=lambda msg: ctx.log(msg),
        device=getattr(ctx, "device", "") or "",
        only_region=only_region,
        max_regions=max_regions,
        feather=feather)

    if ctx.cancelled():
        raise Cancelled("用户已取消")

    errors = meta.get("region_errors") or []
    if errors:
        # 单个区域失败不算整节点失败（引擎已记录），但要在状态里说清楚
        ctx.status(f"⚠️ 区域精修完成，但有 {len(errors)} 个区域失败："
                   f"{errors[0]}")

    ctx.progress(steps, steps)
    summary = regional_mod.describe(regions)
    # 元数据端口上带一份"人能一眼看懂"的摘要：SaveImage 会原样写进 PNG，
    # 事后看图能知道用了哪几个区域、每个区域权重多少。
    meta_out = dict(meta)
    meta_out["region_summary"] = summary
    ctx.metadata.update({
        "regions": meta.get("regions"),
        "regions_applied": meta.get("regions_applied"),
        "region_summary": summary,
        "region_details": meta.get("region_details"),
        "region_errors": errors,
        "region_warnings": meta.get("region_warnings") or [],
        "regional_seconds": meta.get("seconds"),
        "seed": meta.get("seed"),
        "steps": steps,
        "cfg": cfg,
        "workflow_kind": "regional_redraw",
    })
    return {"image": out, "metadata": meta_out}
