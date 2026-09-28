# nodes/image_nodes.py
"""图像节点：ImageResize / ImageCrop / MaskCreate / Inpaint / SaveImage。

节点约定（所有节点模块统一遵守）
--------------------------------
· `run(ctx, node, inputs)` 返回 **dict**，键名必须与 NodeSpec.outputs 一致；
· 需要重依赖（torch/PIL/diffusers）时在函数**内部** import —— 这样 `import nodes`
  不会拖慢界面启动；
· 出错就抛异常，执行器会包成带 node_id 的 NodeError 并定位到本节点；
· 不要直接操作界面控件，只通过 `ctx`（将来才能无头跑）。

本模块的定位
------------
纯 CPU 的几何/遮罩操作（缩放、裁剪、遮罩）与"最后一公里"的落盘
（SaveImage）都放在这里：它们不占显存，但对**出图质量**影响极大 ——
补黑边、越界裁剪、水印/元数据各写各的，都是以前踩过的坑。
"""
from __future__ import annotations

import logging
import os

from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import as_image

logger = logging.getLogger(__name__)

_RESIZE_MODES = ("拉伸", "适应并补边", "裁剪")
_MASK_MODES = ("全图", "矩形", "椭圆")
_SAVE_FORMATS = ("PNG", "JPEG", "WEBP")

# 格式 -> 扩展名（默认 PNG，与文件名约定保持一致）
# 注意 JPEG 用 .jpeg：utils.image_metadata.save_image 是从**扩展名**反推 PIL 的
# format 名（"JPG" 不是合法 format），用 .jpg 会走"降级为普通保存"的分支，
# 元数据就写不进去了。
_FMT_EXT = {"PNG": ".png", "JPEG": ".jpeg", "WEBP": ".webp"}


# ----------------------------------------------------------------------
#  小工具
# ----------------------------------------------------------------------
def _unique_path(path: str) -> str:
    """目标文件已存在时追加 _1 / _2 ... 直到不冲突。

    同一秒内连出多张图（批量 / XY）会撞名，撞名后后一张会**静默覆盖**前一张，
    所以保存前一律走这里。
    """
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    for i in range(1, 10000):
        cand = f"{root}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
    raise RuntimeError(f"同名文件过多，无法为 {path} 找到可用文件名")


def _pad_edge_extend(image, w: int, h: int):
    """等比缩放后居中放置，四周用**边缘像素**向外延伸补齐。

    补黑边会在图四周留下一圈黑框，后续再采样/放大时黑边会渗进画面中心，
    所以这里用边缘延伸：补出来的像素与原图边缘连续，肉眼看不到接缝。
    """
    import numpy as np
    from PIL import Image

    iw, ih = image.size
    scale = min(w / float(iw), h / float(ih))
    nw = max(1, min(w, int(round(iw * scale))))
    nh = max(1, min(h, int(round(ih * scale))))
    small = image.resize((nw, nh), Image.LANCZOS)

    left = (w - nw) // 2
    top = (h - nh) // 2
    arr = np.asarray(small.convert("RGB"))
    arr = np.pad(arr, ((top, h - nh - top), (left, w - nw - left), (0, 0)),
                 mode="edge")
    return Image.fromarray(arr, "RGB")


def _cover_crop(image, w: int, h: int):
    """等比放大到**刚好盖住**目标框，再从中心裁掉多余部分（不拉伸变形）。"""
    from PIL import Image

    iw, ih = image.size
    scale = max(w / float(iw), h / float(ih))
    nw = max(w, int(round(iw * scale)))
    nh = max(h, int(round(ih * scale)))
    big = image.resize((nw, nh), Image.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    return big.crop((left, top, left + w, top + h))


def _clamp_region(x1, y1, x2, y2, w: int, h: int):
    """把 (x1,y1,x2,y2) 夹进图像范围；0 表示"到边缘"。

    越界不再直接丢给 PIL —— PIL 会抛
    `Coordinate 'right' is less than 'left'`，那个报错完全看不出是坐标问题。
    这里夹取后再校验，报错信息带上原始坐标与图像尺寸。
    """
    x2 = w if x2 <= 0 else x2
    y2 = h if y2 <= 0 else y2
    l = max(0, min(int(x1), w))
    t = max(0, min(int(y1), h))
    r = max(0, min(int(x2), w))
    b = max(0, min(int(y2), h))
    if l >= r or t >= b:
        raise ValueError(
            f"区域无效：({x1}, {y1}) - ({x2}, {y2}) 在 {w}x{h} 的图像上"
            f"夹取后为 ({l}, {t}) - ({r}, {b})，宽高必须为正")
    return l, t, r, b


def _tensor_to_pil(model, tensor):
    """兜底：管线返回潜张量/图像张量时自己解码成 PIL。取不到返回 None。"""
    try:
        import torch
    except Exception:
        return None
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
        return None

    t = tensor
    if t.shape[1] not in (3, 4):          # 通道数不像图像 -> 当潜张量解码
        vae = (getattr(model, "vae", None)
               or getattr(getattr(model, "pipe", None), "vae", None))
        if vae is None:
            raise RuntimeError("管线返回潜张量，但模型没有可用的 VAE 解码")
        with torch.inference_mode():
            if hasattr(vae.config, "scaling_factor"):
                t = t / vae.config.scaling_factor
            t = vae.decode(t.to(device=vae.device, dtype=vae.dtype)).sample
    if float(t.min()) < 0:                # 还在 [-1,1] 区间 -> 归一化
        t = (t / 2 + 0.5).clamp(0, 1)

    arr = (t[0].permute(1, 2, 0).float().cpu().numpy() * 255).round()
    from PIL import Image
    arr = arr.astype("uint8")
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], "L")
    return Image.fromarray(arr[:, :, :3], "RGB")


def _find_workflow(ctx):
    """尽力从上下文/宿主上找一个 Workflow 对象；找不到返回 None。"""
    from workflow.schema import Workflow

    for src in (ctx, getattr(ctx, "host", None), getattr(ctx, "ai", None)):
        if src is None:
            continue
        for attr in ("workflow", "current_workflow"):
            wf = getattr(src, attr, None)
            if isinstance(wf, Workflow):
                return wf
    return None


# ----------------------------------------------------------------------
@node(
    "ImageResize",
    category="图像",
    description="调整图像尺寸：拉伸 / 适应并补边（边缘延伸）/ 中心裁剪。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"image": PortSpec(DataType.IMAGE, "图像")},
    params={
        "width": ParamSpec("int", default=512, label="宽",
                           minimum=8, maximum=8192),
        "height": ParamSpec("int", default=768, label="高",
                            minimum=8, maximum=8192),
        "mode": ParamSpec("choice", default="拉伸", label="方式",
                          choices=_RESIZE_MODES,
                          help="拉伸=直接缩放；适应并补边=保持比例并延伸边缘；"
                               "裁剪=保持比例并中心裁剪"),
    },
    resources=ResourceRequirement(estimated_vram_mb=0, requires=()),
)
def run_image_resize(ctx, node, inputs):
    from PIL import Image

    image = as_image(inputs.get("image"))
    if image is None:
        raise ValueError("缺少输入 image")
    image = image.convert("RGB")

    width = int(node.params.get("width", 512))
    height = int(node.params.get("height", 768))
    if width < 1 or height < 1:
        raise ValueError(f"尺寸必须为正：{width}x{height}")
    mode = (node.params.get("mode") or "拉伸").strip()
    if mode not in _RESIZE_MODES:
        raise ValueError(f"未知的缩放方式: {mode}")

    iw, ih = image.size
    if (iw, ih) == (width, height):
        return {"image": image}          # 尺寸已一致，不做无谓的重采样

    ctx.status(f"🖼️ 缩放 {iw}x{ih} → {width}x{height}（{mode}）")
    ctx.check_cancel()

    if mode == "拉伸":
        out = image.resize((width, height), Image.LANCZOS)
    elif mode == "适应并补边":
        out = _pad_edge_extend(image, width, height)
    else:
        out = _cover_crop(image, width, height)

    ctx.check_cancel()
    ctx.metadata.update({"width": width, "height": height,
                         "resize_mode": mode, "source_size": f"{iw}x{ih}"})
    return {"image": out}


# ----------------------------------------------------------------------
@node(
    "ImageCrop",
    category="图像",
    description="按坐标裁剪图像；right/bottom 填 0 表示到边缘，越界会被夹取。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"image": PortSpec(DataType.IMAGE, "图像")},
    params={
        "left": ParamSpec("int", default=0, label="左", minimum=0,
                          maximum=16384),
        "top": ParamSpec("int", default=0, label="上", minimum=0,
                         maximum=16384),
        "right": ParamSpec("int", default=0, label="右",
                           minimum=0, maximum=16384,
                           help="0 表示到图像右边缘"),
        "bottom": ParamSpec("int", default=0, label="下",
                            minimum=0, maximum=16384,
                            help="0 表示到图像下边缘"),
    },
    resources=ResourceRequirement(estimated_vram_mb=0, requires=()),
)
def run_image_crop(ctx, node, inputs):
    image = as_image(inputs.get("image"))
    if image is None:
        raise ValueError("缺少输入 image")
    image = image.convert("RGB")
    w, h = image.size

    left = int(node.params.get("left", 0) or 0)
    top = int(node.params.get("top", 0) or 0)
    right = int(node.params.get("right", 0) or 0)
    bottom = int(node.params.get("bottom", 0) or 0)

    # 先夹取再校验：绝不把越界坐标丢给 PIL（它的报错完全指不到坐标）
    l, t, r, b = _clamp_region(left, top, right, bottom, w, h)

    ctx.status(f"✂️ 裁剪 {w}x{h} → {r - l}x{b - t}")
    out = image.crop((l, t, r, b))
    ctx.check_cancel()

    ctx.metadata.update({"crop_box": [l, t, r, b],
                         "width": out.width, "height": out.height})
    return {"image": out}


# ----------------------------------------------------------------------
@node(
    "MaskCreate",
    category="图像",
    description="生成重绘遮罩（255=重绘区域）：全图 / 矩形 / 椭圆，可羽化。",
    inputs={"image": PortSpec(DataType.IMAGE, optional=True,
                              label="参考图（仅用于取尺寸）")},
    outputs={"mask": PortSpec(DataType.MASK, "遮罩")},
    params={
        "mode": ParamSpec("choice", default="全图", label="形状",
                          choices=_MASK_MODES),
        "x1": ParamSpec("int", default=0, label="左", minimum=0,
                        maximum=16384),
        "y1": ParamSpec("int", default=0, label="上", minimum=0,
                        maximum=16384),
        "x2": ParamSpec("int", default=0, label="右", minimum=0,
                        maximum=16384, help="0 表示到图像右边缘"),
        "y2": ParamSpec("int", default=0, label="下", minimum=0,
                        maximum=16384, help="0 表示到图像下边缘"),
        "feather": ParamSpec("int", default=8, label="羽化", minimum=0,
                             maximum=128,
                             help="高斯模糊半径，0 表示硬边"),
    },
    resources=ResourceRequirement(estimated_vram_mb=0, requires=()),
)
def run_mask_create(ctx, node, inputs):
    from PIL import Image, ImageDraw, ImageFilter

    image = as_image(inputs.get("image"))
    if image is not None:
        w, h = image.size
    else:
        # 没接参考图时用上游（采样节点）写进元数据的尺寸，仍拿不到就报错
        w = int(ctx.metadata.get("width") or 0)
        h = int(ctx.metadata.get("height") or 0)
        if w <= 0 or h <= 0:
            raise ValueError("没有参考图，也没有可用的尺寸（请接入 image）")

    mode = (node.params.get("mode") or "全图").strip()
    if mode not in _MASK_MODES:
        raise ValueError(f"未知的遮罩形状: {mode}")

    if mode == "全图":
        # 整张都要重绘：全白遮罩；羽化对纯色无影响，一并走到下面的统一流程
        x1 = y1 = x2 = y2 = None
    else:
        x1, y1, x2, y2 = _clamp_region(
            int(node.params.get("x1", 0) or 0), int(node.params.get("y1", 0) or 0),
            int(node.params.get("x2", 0) or 0), int(node.params.get("y2", 0) or 0),
            w, h)

    ctx.status(f"🎭 生成遮罩 {mode} {w}x{h}")
    ctx.check_cancel()

    mask = Image.new("L", (w, h), 255 if mode == "全图" else 0)
    if mode != "全图":
        draw = ImageDraw.Draw(mask)
        if mode == "矩形":
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=255)
        else:
            draw.ellipse([x1, y1, x2 - 1, y2 - 1], fill=255)

    feather = int(node.params.get("feather", 8) or 0)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    ctx.check_cancel()

    ctx.metadata.update({"mask_mode": mode, "feather": feather,
                         "mask_size": f"{w}x{h}"})
    return {"mask": mask}


# ----------------------------------------------------------------------
@node(
    "Inpaint",
    category="图像",
    description="局部重绘：按遮罩重画图像（白=重绘区域），一步到位出图。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "image": PortSpec(DataType.IMAGE, "图像"),
            "mask": PortSpec(DataType.MASK, "遮罩"),
            "positive": PortSpec(DataType.TEXT_COND, "正面条件"),
            "negative": PortSpec(DataType.TEXT_COND, "负面条件",
                                 optional=True)},
    outputs={"image": PortSpec(DataType.IMAGE, "图像")},
    params={
        "steps": ParamSpec("int", default=24, label="步数",
                           minimum=1, maximum=150),
        "cfg": ParamSpec("float", default=6.5, label="CFG",
                         minimum=0.0, maximum=30.0, step=0.5),
        "seed": ParamSpec("seed", default=-1, label="种子",
                          help="-1 表示随机"),
        "strength": ParamSpec("float", default=0.45, label="重绘强度",
                              minimum=0.05, maximum=1.0, step=0.05),
    },
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=2600,
                                  supports_cpu_offload=True,
                                  supports_tiled=True,
                                  exclusive_gpu=True, needs_model=True),
)
def run_inpaint(ctx, node, inputs):
    import torch
    from PIL import Image

    # 种子发生器复用采样节点那一套，避免两处实现慢慢跑偏
    from nodes.sampling_nodes import _generator, _resolve_seed

    model = inputs.get("model")
    image = as_image(inputs.get("image"))
    mask = inputs.get("mask")
    pos = inputs.get("positive")
    neg = inputs.get("negative")

    if model is None:
        raise ValueError("缺少输入 model")
    if image is None:
        raise ValueError("缺少输入 image")
    if mask is None:
        raise ValueError("缺少输入 mask（重绘区域，白=重绘）")
    image = image.convert("RGB")
    if not hasattr(mask, "convert"):
        raise ValueError(f"mask 必须是 PIL 图像（mode \"L\"），实际 {type(mask).__name__}")

    adapter = getattr(model, "adapter", None)
    if adapter is None:
        raise RuntimeError(f"模型 {model.describe()} 没有适配器，无法局部重绘")
    if not getattr(adapter.caps, "inpaint", False):
        raise RuntimeError(f"{model.arch_id} 架构不支持局部重绘（inpaint）")
    pipe = adapter.pipe(model, "inpaint")
    if pipe is None:
        raise RuntimeError(f"模型 {model.describe()} 没有可用的 inpaint 管线")

    # 遮罩尺寸必须与图像一致，否则 diffusers 会按错的尺寸处理
    mask = mask.convert("L")
    if mask.size != image.size:
        ctx.status(f"🎭 遮罩尺寸 {mask.size} 与图像 {image.size} 不一致，已自动缩放")
        mask = mask.resize(image.size, Image.NEAREST)

    steps = int(node.params.get("steps", 24))
    cfg = float(node.params.get("cfg", adapter.default_guidance()))
    seed = _resolve_seed(node.params.get("seed", -1))
    strength = float(node.params.get("strength", 0.45))

    kw = adapter.sample_kwargs(pos, steps=steps, cfg=cfg, seed=seed,
                               width=image.width, height=image.height)
    # 负面条件优先用独立输入（与 Sampler 保持一致）
    if neg is not None and getattr(neg, "negative_embeds", None) is not None:
        kw["negative_prompt_embeds"] = neg.negative_embeds
        if getattr(neg, "negative_pooled", None) is not None:
            kw["negative_pooled_prompt_embeds"] = neg.negative_pooled

    kw["image"] = image
    kw["mask_image"] = mask
    kw["strength"] = strength
    kw["generator"] = _generator(ctx.device, seed)
    # 重绘尺寸跟随输入图，避免管线被 width/height 拉成别的比例
    kw.pop("width", None)
    kw.pop("height", None)

    ctx.status(f"🩹 局部重绘 {steps} 步 · 强度 {strength} · seed {seed}")
    ctx.check_cancel()
    with torch.inference_mode():
        res = pipe(**kw)
    ctx.check_cancel()

    imgs = getattr(res, "images", None)
    if imgs is None:
        raise RuntimeError("重绘管线没有返回图像")
    out = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
    if out is None:
        raise RuntimeError("重绘管线返回了空的图像列表")

    pil = as_image(out)
    if pil is None:
        pil = _tensor_to_pil(model, out)
    if pil is None:
        raise RuntimeError(f"无法把管线返回值转成图像（{type(out).__name__}）")

    meta = {"steps": steps, "cfg": cfg, "seed": seed, "strength": strength,
            "workflow_kind": "inpaint", "width": image.width,
            "height": image.height}
    ctx.metadata.update(meta)
    return {"image": pil.convert("RGB")}


# ----------------------------------------------------------------------
@node(
    "SaveImage",
    category="输出",
    description="保存图像到输出目录（写入元数据，水印走统一实现）。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像"),
            "metadata": PortSpec(DataType.METADATA, "元数据", optional=True)},
    outputs={"path": PortSpec(DataType.METADATA, "路径"),
             "metadata": PortSpec(DataType.METADATA, "元数据")},
    params={
        "filename_prefix": ParamSpec("str", default="", label="文件名前缀",
                                     help="留空则用 wf_"),
        "subdir": ParamSpec("str", default="", label="子目录",
                            help="相对输出目录，可填 a/b"),
        "format": ParamSpec("choice", default="PNG", label="格式",
                            choices=_SAVE_FORMATS),
        "watermark": ParamSpec("bool", default=True, label="加水印"),
        "save_workflow": ParamSpec("bool", default=False, label="写入工作流"),
    },
    resources=ResourceRequirement(estimated_vram_mb=0, requires=()),
)
def run_save_image(ctx, node, inputs):
    import datetime

    from utils.image_metadata import save_image
    from utils.paths import OUTPUT_DIR, host_output_dir

    image = as_image(inputs.get("image"))
    if image is None:
        raise ValueError("缺少输入 image")
    image = image.convert("RGB")

    prefix = (node.params.get("filename_prefix") or "").strip() or "wf_"
    subdir = (node.params.get("subdir") or "").strip()
    fmt = (node.params.get("format") or "PNG").strip().upper()
    ext = _FMT_EXT.get(fmt)
    if ext is None:
        raise ValueError(f"不支持的保存格式: {fmt}")
    want_wm = bool(node.params.get("watermark", True))
    save_wf = bool(node.params.get("save_workflow", False))

    # 输出目录与主出图共用一条来源（否则"设置里改了目录，工作流还写旧目录"）
    out_dir = host_output_dir(ctx.host) if ctx.host is not None else OUTPUT_DIR
    if subdir:
        parts = [p for p in subdir.replace("\\", "/").split("/") if p
                 and p not in (".", "..")]
        if parts:
            out_dir = os.path.join(out_dir, *parts)
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _unique_path(os.path.join(out_dir, f"{prefix}{ts}_0{ext}"))

    # ---- 元数据：上游 dict + 本次运行贡献（下划线开头的是运行期内部量）----
    meta: dict = {}
    src = inputs.get("metadata")
    if isinstance(src, dict):
        meta.update(src)
    if isinstance(ctx.metadata, dict):
        meta.update({k: v for k, v in ctx.metadata.items()
                     if not str(k).startswith("_")})
    # 与 utils/app_generation._save_with_meta 一致：缺的通用字段补 unknown，
    # 免得 A1111 参数串里出现空洞
    # 尺寸没人报过时用真实落盘尺寸，至少 Size 字段不空
    if not meta.get("width"):
        meta["width"] = image.width
    if not meta.get("height"):
        meta["height"] = image.height
    for k in ("prompt", "prompt_final", "negative", "negative_final", "model",
              "model_hash", "sampler", "scheduler", "steps", "cfg", "seed",
              "size"):
        meta.setdefault(k, "unknown")
    if meta.get("size") == "unknown":
        try:
            meta["size"] = f"{int(meta['width'])}x{int(meta['height'])}"
        except Exception:
            pass
    meta["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["workflow_node"] = ctx.current_node or node.id
    meta["save_format"] = fmt

    if save_wf:
        wf = _find_workflow(ctx)
        if wf is None:
            logger.info("ℹ️ save_workflow 已开启，但拿不到 Workflow 对象，跳过")
        else:
            try:
                from workflow import serializers
                meta["workflow_name"] = wf.name
                meta["nodes"] = ", ".join(f"{n.id}:{n.type}" for n in wf.nodes)
                meta["workflow"] = serializers.dumps(wf)
            except Exception as e:
                logger.warning(f"⚠️ 工作流信息写入元数据失败（忽略）: {e}")

    ctx.status(f"💾 保存图像 {os.path.basename(path)}")
    ctx.check_cancel()

    # 水印统一由 utils.image_metadata.save_image 施加（它读 host.config），
    # 显式关掉时临时改配置并在 finally 里还原
    cfg = getattr(ctx.host, "config", None)
    if not want_wm and cfg is not None:
        prev = getattr(cfg, "watermark_enabled", False)
        try:
            cfg.watermark_enabled = False
            saved = save_image(ctx.host, image, path, meta)
        finally:
            cfg.watermark_enabled = prev
    else:
        saved = save_image(ctx.host, image, path, meta)

    saved = saved or path
    if not os.path.exists(saved):
        raise RuntimeError(f"图像保存失败（文件不存在）: {saved}")

    ctx.metadata["output_path"] = saved
    return {"path": saved, "metadata": meta}


# ----------------------------------------------------------------------
@node(
    "LoadImage",
    category="图像",
    description="从磁盘读取一张图（图生图 / ControlNet 参考图 / IP-Adapter 参考图的入口）。",
    outputs={"image": PortSpec(DataType.IMAGE, "图像"),
             "mask": PortSpec(DataType.MASK, "遮罩", optional=True),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "path": ParamSpec("str", default="", label="图片路径"),
        "as_mask": ParamSpec("bool", default=False, label="作为遮罩读取",
                             help="开启后按灰度图读成 L 模式（255=重绘区域）"),
        "max_side": ParamSpec("int", default=0, label="最长边限制",
                              minimum=0, maximum=8192,
                              help="0 = 不缩放"),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0),
)
def run_load_image(ctx, node, inputs):
    """把文件读成 IMAGE（可选同时给一份 MASK）。

    为什么需要它：ControlNetPreprocess / IPAdapterApply / Inpaint / Sampler
    的 `image` 都是必填输入，没有加载节点就没法在工作流里表达
    "用这张参考图"。少一个这样的源节点，整条控制链路都搭不出来。
    """
    from utils.image_metadata import enrich_metadata  # noqa: F401  (保持依赖显式)

    path = (node.params.get("path") or "").strip()
    if not path:
        raise ValueError("没有指定图片路径")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到图片: {path}")

    as_mask = bool(node.params.get("as_mask", False))
    try:
        max_side = int(node.params.get("max_side", 0) or 0)
    except Exception:
        max_side = 0

    from PIL import Image
    with Image.open(path) as im:
        img = im.convert("L" if as_mask else "RGB")
        if max_side > 0 and max(img.size) > max_side:
            ratio = max_side / max(img.size)
            new = (max(1, int(img.width * ratio)),
                   max(1, int(img.height * ratio)))
            img = img.resize(new, Image.LANCZOS)
        img.load()

    meta = {"source_image": path, "source_size": f"{img.width}x{img.height}"}
    ctx.metadata.update(meta)
    return {"image": img, "mask": img if as_mask else None, "metadata": meta}
