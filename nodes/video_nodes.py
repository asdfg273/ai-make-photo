# nodes/video_nodes.py
"""视频节点：VideoFrameBatch（把图像 / 图像序列 / 目录汇总成帧批次）。

为什么要一个"归一化"节点
------------------------
上游可能是单张图、ImageBatch、帧列表，甚至是一个装着 PNG 序列的目录。
把它们统一成 `ImageBatch` 之后，"帧数 / fps / 短边尺寸"这些**视频才有的
约束**就在一个地方处理完，下游（插帧、逐帧重绘、合成视频）不必各自判断。

约定（与其它节点模块一致）
--------------------------
· 重依赖（PIL）在函数内部 import；
· 出错就抛异常，执行器会包成带 node_id 的 NodeError；
· 不碰界面控件，只走 ctx。
"""
from __future__ import annotations

import logging
import os

from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import ImageBatch, as_image

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _load_directory(path: str) -> list:
    """按文件名顺序读取目录下的图片（帧序列）。"""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"找不到目录: {path}")
    names = sorted(
        f for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        and os.path.isfile(os.path.join(path, f)))
    if not names:
        raise ValueError(
            f"目录里没有可用图片（支持 {', '.join(_IMAGE_EXTS)}）: {path}")

    from PIL import Image

    frames = []
    for name in names:
        full = os.path.join(path, name)
        try:
            with Image.open(full) as im:
                frames.append(im.convert("RGB").copy())
        except Exception as e:
            logger.warning(f"⚠️ 跳过无法读取的帧 {name}: {e}")
    if not frames:
        raise RuntimeError(f"目录里的图片全部读取失败: {path}")
    return frames


def _normalize_frames(value) -> list:
    """把输入归一化成 PIL 图像列表。"""
    if value is None:
        raise ValueError("缺少输入 images（图像 / ImageBatch / 帧列表 / 目录）")

    # 1) 目录路径（帧序列）
    if isinstance(value, (str, os.PathLike)):
        p = os.fspath(value)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"路径不是目录（本节点只接受图片序列目录）: {p}")
        return _load_directory(p)

    # 2) 单张图
    single = as_image(value)
    if single is not None and not hasattr(value, "images"):
        return [single.convert("RGB")]

    # 3) 类 ImageBatch 的对象 / 列表 / 元组
    if hasattr(value, "images"):
        value = list(value.images)
    elif isinstance(value, (list, tuple)):
        value = list(value)
    else:
        raise TypeError(
            f"不支持的输入类型: {type(value).__name__}"
            "（需要图像 / ImageBatch / 图像列表 / 目录路径）")

    frames = []
    for i, item in enumerate(value):
        img = as_image(item)
        if img is None:
            logger.warning(f"⚠️ 跳过第 {i} 帧（无法转成图像: "
                           f"{type(item).__name__}）")
            continue
        frames.append(img.convert("RGB"))
    if not frames:
        raise ValueError("输入里没有任何可用的图像帧")
    return frames


def _fit_short_side(image, target: int):
    """把短边缩放到**恰好等于** target（长宽比保持，等比缩放）。

    · target <= 0 → 不缩放；
    · 短边本来就等于 target（含 1px 舍入误差）→ 原样返回，不动像素；
    · 短边大于 / 小于 target → 等比缩放到相等（大图缩、小图放大），
      这样同一批帧的尺寸才统一，后续模型才能吃。
    """
    from PIL import Image

    if target <= 0:
        return image
    w, h = image.size
    short = min(w, h)
    if short <= 0:
        return image
    if abs(short - target) <= max(1, int(target * 0.01)):
        # 已经在目标短边附近：再缩一次只会得到同样尺寸（甚至来回抖动）
        return image

    scale = target / float(short)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if (nw, nh) == (w, h):
        return image
    return image.resize((nw, nh), Image.LANCZOS)


@node(
    "VideoFrameBatch",
    category="视频",
    description="把图像 / 图像序列 / 帧目录汇总成帧批次（限帧数、按短边缩放并带上 fps）。",
    inputs={"images": PortSpec(DataType.ANY, "图像或帧序列",
                               optional=True)},
    outputs={"frames": PortSpec(DataType.ANY, "帧批次"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "max_frames": ParamSpec("int", default=48, label="最大帧数",
                                minimum=1, maximum=600),
        "fps": ParamSpec("float", default=8.0, label="帧率",
                         minimum=1.0, maximum=60.0, step=1.0),
        "resize": ParamSpec("int", default=512, label="短边像素",
                            minimum=64, maximum=2048,
                            help="帧的短边统一缩放到该值（等比，已等于该值则不动）"),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0),
)
def run_video_frame_batch(ctx, node, inputs):
    frames = _normalize_frames(inputs.get("images"))

    max_frames = int(node.params.get("max_frames", 48))
    fps = float(node.params.get("fps", 8.0))
    resize = int(node.params.get("resize", 512))

    total = len(frames)
    if max_frames > 0 and total > max_frames:
        logger.info(f"✂️ 帧数 {total} 超过上限 {max_frames}，截断")
        frames = frames[:max_frames]

    ctx.status(f"🎬 整理帧批次 {len(frames)}/{total} 帧 · {fps}fps "
               f"· 短边 {resize}px")
    ctx.check_cancel()

    out = []
    for i, frame in enumerate(frames):
        out.append(_fit_short_side(frame, resize))
        # 大批量帧时不至于长时间不响应取消
        if i % 8 == 0:
            ctx.check_cancel()
    ctx.check_cancel()

    meta = {"frame_count": len(out), "fps": fps,
            "source_frame_count": total,
            "frame_size": list(out[0].size) if out else []}
    ctx.metadata.update(meta)
    logger.info(f"✅ 帧批次就绪: {meta['frame_count']} 帧, "
                f"尺寸 {meta['frame_size']}")

    return {"frames": ImageBatch(images=out, fps=fps), "metadata": meta}
