# nodes/control_nodes.py
"""ControlNet 节点：ControlNetPreprocess / ControlNetApply。

为什么拆成"预处理"与"应用"两步
-------------------------------
预览/复用/换权重都是改一条线的事：预处理结果 `ControlImage` 是纯数据
（PIL + kind + weight + start/end），可以缓存、可以只重跑应用那一步；
而"把 ControlNet 装进管线"（prepare_controlnet）是重活，交给 ModelManager
按当前架构自己决定 SD1.5 / SDXL 权重，节点不重复造这套逻辑。

约定（与其它节点模块一致）
--------------------------
· 重依赖（torch / cv2 / controlnet_aux / diffusers）一律在函数**内部** import；
· 出错就抛异常，执行器会包成带 node_id 的 NodeError；
· 不碰界面控件（`ctx.host` 的长得像 UI 的属性一律不读不写），只走 ctx。
"""
from __future__ import annotations

import logging

from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import ControlImage, as_image

logger = logging.getLogger(__name__)

# 预处理类型（与 ControlImage.kind 取值一致）
_KINDS = ("openpose", "canny", "depth", "lineart", "softedge", "tile",
          "segmentation")

# controlnet_aux 的检测器类名映射（延迟 import，避免拖慢界面启动）
_AUX_DETECTORS = {
    "openpose": ("controlnet_aux", "OpenposeDetector"),
    "depth": ("controlnet_aux", "MidasDetector"),
    "lineart": ("controlnet_aux", "LineartDetector"),
    "softedge": ("controlnet_aux", "HEDdetector"),
    # 平铺（tile）没有专用检测器：ComfyUI 里就是"原图直通"
    "tile": None,
}

# 已实例化的检测器缓存：key -> 实例。模型只加载一次。
_DETECTORS: dict = {}

_INSTALL_HINT = (
    "❌ 缺少预处理依赖：{missing}\n"
    "   ControlNet 预处理器是**可选**组件：\n"
    "     · 安装 controlnet-aux（推荐）: pip install -U controlnet-aux\n"
    "     · 或把标注器权重放到 controlnets/Annotators 下（离线可用）\n"
    "   也可以先用 \"canny\"（只需 OpenCV）或 \"tile\"（原图直通）跑通流程。"
)


def _resolve_device():
    """优先用 torch 判断能否上 GPU；torch 不在就退回 CPU。"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_detector(kind: str):
    """按需构造并缓存 controlnet_aux 检测器；失败抛 RuntimeError（含安装提示）。"""
    if kind in _DETECTORS:
        return _DETECTORS[kind]
    entry = _AUX_DETECTORS.get(kind)
    if entry is None:
        raise RuntimeError(
            f"❌ 预处理类型 '{kind}' 没有可用的辅助模型。\n"
            f"   可用: {', '.join(_KINDS)}（tile = 原图直通）")

    mod_name, cls_name = entry
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
    except Exception as e:
        raise RuntimeError(
            _INSTALL_HINT.format(missing=f"{mod_name}.{cls_name}（{e}）")) from e

    try:
        detector = cls.from_pretrained("lllyasviel/Annotators")
    except Exception as e:
        raise RuntimeError(
            _INSTALL_HINT.format(
                missing=f"标注器权重 {cls_name}（{type(e).__name__}: {e}）")
            + "\n   权重会从 lllyasviel/Annotators 下载；离线环境请预先放好。"
        ) from e

    # 检测器不上 GPU：省显存，ControlNet 本体的显存才是瓶颈
    try:
        if hasattr(detector, "to"):
            detector.to(_resolve_device())
    except Exception as e:
        logger.info(f"ℹ️ 检测器 {cls_name} 未迁移设备（忽略）: {e}")

    _DETECTORS[kind] = detector
    return detector


def _preprocess(image, kind: str):
    """把一张 PIL 图变成 ControlNet 条件图。"""
    if kind == "tile":
        # 平铺控制不需要标注：直接把原图当条件
        return image
    if kind == "canny":
        return _canny(image)

    detector = _load_detector(kind)
    ctx_kw = {}
    if kind == "openpose":
        ctx_kw["hand_and_face"] = True
    elif kind == "softedge":
        ctx_kw["safe"] = True
    try:
        return detector(image, **ctx_kw)
    except TypeError:
        # 老版本检测器签名不同 → 用最朴素的调用方式再来一次
        return detector(image)


def _canny(image):
    """Canny 边缘：只依赖 OpenCV，不需要任何辅助权重。"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except Exception as e:
        raise RuntimeError(
            f"❌ Canny 预处理需要 opencv-python 与 numpy：{e}\n"
            "   可执行: pip install opencv-python") from e
    try:
        arr = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(edges, "L").convert("RGB")
    except Exception as e:
        raise RuntimeError(
            f"❌ Canny 预处理失败（需要 opencv-python）：{e}\n"
            "   可执行: pip install opencv-python") from e


def _r8(v: int) -> int:
    """对齐到 8 的倍数（多数 ControlNet 要求）。"""
    return max(8, int(round(int(v) / 8.0)) * 8)


def _resize_keep_aspect(image, target: int):
    """把图缩放到与 target 同面积、保持长宽比，尺寸对齐 8 的倍数。"""
    from PIL import Image

    target = max(8, int(target))
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"图像尺寸非法: {image.size}")
    scale = (target / max(w, h)) ** 0.5
    nw, nh = _r8(w * scale), _r8(h * scale)
    if (nw, nh) == (w, h):
        return image
    return image.resize((nw, nh), Image.LANCZOS)


# ----------------------------------------------------------------------
@node(
    "ControlNetPreprocess",
    category="控制",
    description="把图像预处理成 ControlNet 条件图（骨架/边缘/深度/线稿/柔边/平铺/分割）。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"control": PortSpec(DataType.CONTROL_IMAGE, "控制图")},
    params={
        "kind": ParamSpec("choice", default="openpose", label="预处理类型",
                          choices=_KINDS,
                          help="openpose/canny/depth/lineart/softedge 依赖 "
                               "controlnet-aux；tile = 原图直通。\n"
                               "⚠️ 下游 ControlNetApply 目前只能加载 "
                               "openpose / canny / depth 三种权重；"
                               "lineart / softedge / segmentation 只能用于"
                               "「预处理出图」，接 ControlNetApply 会明确报错"
                               "（不会静默换成别的控制类型）。"),
        "weight": ParamSpec("float", default=1.0, label="控制权重",
                            minimum=0.0, maximum=2.0, step=0.05),
        "start": ParamSpec("float", default=0.0, label="起效步数",
                           minimum=0.0, maximum=1.0, step=0.05),
        "end": ParamSpec("float", default=1.0, label="结束步数",
                         minimum=0.0, maximum=1.0, step=0.05),
        "resolution": ParamSpec("int", default=512, label="分辨率",
                                minimum=64, maximum=4096),
    },
    resources=ResourceRequirement(requires=("cuda",),
                                  estimated_vram_mb=900,
                                  supports_cpu_offload=True),
)
def run_control_preprocess(ctx, node, inputs):
    image = as_image(inputs.get("image"))
    if image is None:
        raise ValueError("缺少输入 image（需要一张图像才能做 ControlNet 预处理）")

    kind = (node.params.get("kind") or "openpose").strip().lower()
    if kind not in _KINDS:
        raise ValueError(f"未知预处理类型: {kind!r}，可用: {list(_KINDS)}")
    weight = float(node.params.get("weight", 1.0))
    start = float(node.params.get("start", 0.0))
    end = float(node.params.get("end", 1.0))
    resolution = int(node.params.get("resolution", 512))
    if end < start:
        logger.info(f"ℹ️ ControlNet end({end}) < start({start})，按无效果处理")

    rgb = image.convert("RGB")
    ctx.status(f"🧩 ControlNet 预处理 [{kind}] {resolution}px ...")
    ctx.check_cancel()

    worked = _resize_keep_aspect(rgb, resolution)
    out = _preprocess(worked, kind)
    ctx.check_cancel()

    if not hasattr(out, "convert"):
        raise RuntimeError(f"预处理 [{kind}] 没有返回图像（得到 {type(out).__name__}）")
    out = out.convert("RGB")
    if out.size != worked.size:
        # 检测器可能自己改了尺寸，统一回目标分辨率
        out = _resize_keep_aspect(out, resolution)

    meta = {"controlnet_type": kind, "controlnet_weight": weight,
            "controlnet_start": start, "controlnet_end": end,
            "controlnet_size": list(out.size)}
    ctx.metadata.update(meta)
    logger.info(f"✅ ControlNet 预处理完成: {kind} {out.size}")

    return {"control": ControlImage(image=out, kind=kind, weight=weight,
                                    start=start, end=end)}


# ----------------------------------------------------------------------
@node(
    "ControlNetApply",
    category="控制",
    description="把控制图装到模型上（按当前架构自动选 SD1.5 / SDXL 权重）。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "control": PortSpec(DataType.CONTROL_IMAGE, "控制图")},
    outputs={"model": PortSpec(DataType.MODEL, "模型"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "weight": ParamSpec("float", default=1.0, label="控制权重",
                            minimum=0.0, maximum=2.0, step=0.05),
    },
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=1500,
                                  supports_cpu_offload=True),
)
def run_control_apply(ctx, node, inputs):
    model = inputs.get("model")
    control = inputs.get("control")
    if model is None:
        raise ValueError("缺少输入 model")
    if control is None:
        raise ValueError("缺少输入 control（请先接 ControlNetPreprocess）")
    if ctx.ai is None:
        raise RuntimeError("没有可用的模型管理器（ctx.ai 为空），无法加载 ControlNet")

    kind = (getattr(control, "kind", "") or "").strip().lower()
    raw_weight = node.params.get("weight", 1.0)
    weight = float(raw_weight if raw_weight is not None
                   else getattr(control, "weight", 1.0))
    if kind in ("", "none"):
        meta = {"controlnet_type": "none", "controlnet_weight": weight}
        ctx.metadata.update(meta)
        return {"model": model, "metadata": meta}

    # ── 1) 装载 ControlNet（ModelManager 自己判断 SD1.5 / SDXL）──
    ctx.status(f"🧩 装载 ControlNet [{kind}] ...")
    ctx.check_cancel()
    try:
        ctx.ai.prepare_controlnet(kind)
    except Exception as e:
        raise RuntimeError(
            f"❌ 装载 ControlNet '{kind}' 失败：{type(e).__name__}: {e}\n"
            "   常见原因：ControlNet 模型尚未下载（可在扩展市场安装："
            f"controlnet_{kind} / controlnet_{kind}_sdxl），"
            "或权重的架构与当前底模不匹配（SD1.5 ⇄ SDXL），"
            "或该类型不被当前 ModelManager 支持（目前支持 openpose / "
            "canny / depth）。"
        ) from e
    ctx.check_cancel()

    # ── 2) 生成管线期望的控制图 ──
    # 注意：**不读也不写** ctx.host 上的任何 UI 属性（pose_image_path /
    # ref_image_path 等）—— 那样工作流就没法离开界面跑。
    control_image = as_image(getattr(control, "image", None))
    getter = getattr(ctx.ai, "get_control_image", None)
    if callable(getter) and control_image is not None:
        try:
            got = getter(control_image, kind)
            if got is not None:
                control_image = got
                logger.info(f"✅ 已由模型管理器重新生成控制图 [{kind}]")
        except Exception as e:
            logger.warning(f"⚠️ get_control_image 失败，沿用预处理结果: {e}")
    ctx.check_cancel()

    # ── 3) 把控制信息挂到模型上（下游采样节点读取）──
    if getattr(model, "meta", None) is None:
        model.meta = {}
    model.meta["control_image"] = control_image
    model.meta["control_kind"] = kind
    model.meta["control_weight"] = weight
    model.meta["control_start"] = float(getattr(control, "start", 0.0) or 0.0)
    model.meta["control_end"] = float(getattr(control, "end", 1.0) or 1.0)

    meta = {"controlnet_type": kind, "controlnet_weight": weight}
    ctx.metadata.update(meta)
    logger.info(f"✅ ControlNet 已应用: {kind} weight={weight}")
    return {"model": model, "metadata": meta}
