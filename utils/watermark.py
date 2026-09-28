# utils/watermark.py
"""给生成图加水印（文字 / 图片），位置、透明度、大小可调。

设计取舍
--------
· 文字水印**带描边**：浅色背景上纯白字、深色背景上纯黑字都会糊掉，
  描边（stroke）能让两种背景都读得清，比单纯调透明度实用得多；
· 全部在 RGBA 图层上合成，**不修改原图 mode**，RGB 图进出仍是 RGB；
· 任何异常都退回原图 —— 水印失败绝不能让出图失败；
· 不依赖任何在线字体：按候选列表找系统字体，全都没有就用 PIL 内置位图字体。

配置字段（AppConfig）：
    watermark_enabled / watermark_text / watermark_image
    watermark_position / watermark_opacity / watermark_scale
"""
from __future__ import annotations

import logging
import os

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

POSITIONS = ("右下", "左下", "右上", "左上", "居中", "平铺")

# 常见中英文字体候选（Windows 优先，其次 mac/linux）
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",     # 黑体
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)

_MARGIN_RATIO = 0.02      # 边距 = 短边 * 该比例
_MIN_FONT = 12
_FONT_RATIO = 0.035       # 字号 = 短边 * 该比例 * scale


def find_font_path() -> str | None:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = find_font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception as e:
            logger.debug(f"字体加载失败 {path}: {e}")
    try:
        return ImageFont.load_default(size)      # Pillow ≥ 9.2
    except Exception:
        return ImageFont.load_default()


def _anchor_xy(position: str, box, canvas):
    """按位置算左上角坐标。"""
    m = max(4, int(min(canvas) * _MARGIN_RATIO))
    w, h = box
    W, H = canvas
    if position == "右下":
        return W - w - m, H - h - m
    if position == "左下":
        return m, H - h - m
    if position == "右上":
        return W - w - m, m
    if position == "左上":
        return m, m
    # 居中
    return (W - w) // 2, (H - h) // 2


def _text_layer(size, text, font, opacity, position):
    """生成一张与画面同尺寸的文字图层（RGBA）。"""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    alpha = max(0, min(255, int(255 * opacity)))
    stroke = max(1, int(getattr(font, "size", 16) * 0.08))
    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    except Exception:
        bbox = (0, 0, len(text) * 8, 16)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if position == "平铺":
        step_x = max(1, tw + int(min(size) * 0.08))
        step_y = max(1, th + int(min(size) * 0.08))
        for y in range(0, size[1], step_y):
            for x in range(0, size[0], step_x):
                draw.text((x - bbox[0], y - bbox[1]), text, font=font,
                          fill=(255, 255, 255, alpha),
                          stroke_width=stroke, stroke_fill=(0, 0, 0, alpha))
        return layer

    x, y = _anchor_xy(position, (tw, th), size)
    draw.text((x - bbox[0], y - bbox[1]), text, font=font,
              fill=(255, 255, 255, alpha),
              stroke_width=stroke, stroke_fill=(0, 0, 0, alpha))
    return layer


def _image_layer(size, img_path, opacity, position, scale):
    """把水印图片缩放到合适大小并生成同尺寸图层。"""
    wm = Image.open(img_path).convert("RGBA")
    target_w = max(1, int(size[0] * 0.18 * max(0.05, scale)))
    ratio = target_w / wm.width
    wm = wm.resize((target_w, max(1, int(wm.height * ratio))), Image.LANCZOS)

    if opacity < 1.0:
        a = wm.getchannel("A").point(lambda v: int(v * opacity))
        wm.putalpha(a)

    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if position == "平铺":
        step_x, step_y = wm.width * 2, wm.height * 2
        for y in range(0, size[1], step_y):
            for x in range(0, size[0], step_x):
                layer.alpha_composite(wm, (x, y))
    else:
        x, y = _anchor_xy(position, wm.size, size)
        layer.alpha_composite(wm, (x, y))
    return layer


def apply_watermark(img: Image.Image, cfg) -> Image.Image:
    """按配置给图片加水印。没启用、没内容或出错都原样返回。"""
    try:
        if img is None or not getattr(cfg, "watermark_enabled", False):
            return img

        text = (getattr(cfg, "watermark_text", "") or "").strip()
        img_path = (getattr(cfg, "watermark_image", "") or "").strip()
        position = getattr(cfg, "watermark_position", "右下") or "右下"
        if position not in POSITIONS:
            position = "右下"
        try:
            opacity = float(getattr(cfg, "watermark_opacity", 0.6))
        except Exception:
            opacity = 0.6
        opacity = max(0.02, min(1.0, opacity))
        try:
            scale = float(getattr(cfg, "watermark_scale", 1.0))
        except Exception:
            scale = 1.0
        scale = max(0.2, min(5.0, scale))

        base = img.convert("RGBA")
        layer = None

        if img_path and os.path.exists(img_path):
            try:
                layer = _image_layer(base.size, img_path, opacity, position, scale)
            except Exception as e:
                logger.warning(f"⚠️ 图片水印加载失败，改用文字水印: {e}")
                layer = None

        if layer is None and text:
            font = _load_font(max(_MIN_FONT,
                                  int(min(base.size) * _FONT_RATIO * scale)))
            layer = _text_layer(base.size, text, font, opacity, position)

        if layer is None:
            return img          # 既没图也没字 → 什么都不做

        out = Image.alpha_composite(base, layer)
        return out.convert(img.mode if img.mode != "RGBA" else "RGBA")

    except Exception as e:
        logger.warning(f"⚠️ 水印添加失败（已跳过）: {e}")
        return img


def describe(cfg) -> str:
    """给元数据用的一句话说明。"""
    if not getattr(cfg, "watermark_enabled", False):
        return "off"
    text = (getattr(cfg, "watermark_text", "") or "").strip()
    img = (getattr(cfg, "watermark_image", "") or "").strip()
    kind = "image" if (img and os.path.exists(img)) else ("text" if text else "none")
    return (f"{kind} @{getattr(cfg, 'watermark_position', '右下')} "
            f"opacity={getattr(cfg, 'watermark_opacity', 0.6)} "
            f"scale={getattr(cfg, 'watermark_scale', 1.0)}")
