# utils/tiled_diffusion.py
# ============================================================
#  Tiled Diffusion: 分块生成大图（突破显存/算力限制）
#  原理: 将大图切成多个 tile 分别生成，重叠区域加权融合消接缝
# ============================================================
import torch
import numpy as np
from PIL import Image
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


def _make_blend_mask(tile_w: int, tile_h: int, overlap: int) -> np.ndarray:
    """生成 tile 的羽化权重图(中心=1, 边缘=0)，用于无缝融合

    ⚠️ fade 不能含 0：图像最外一圈只被单块覆盖，若该处权重为 0，
    归一化时 0/1e-6 = 0 会让成品四周出现 1px 黑线。
    这里取开区间（[1:-1]）让最小权重恒 > 0。
    """
    mask = np.ones((tile_h, tile_w), dtype=np.float32)
    if overlap <= 0:
        return mask
    # overlap=1 时 linspace(0,1,3)[1:-1] 只有 1 个点，取 [0.5] 避免空数组
    fade = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
    if fade.size == 0:
        fade = np.array([0.5], dtype=np.float32)
    ov = fade.size
    if ov > tile_h or ov > tile_w:
        # 羽化不能超过 tile 本身
        ov = max(1, min(ov, tile_h // 2, tile_w // 2))
        fade = fade[:ov]
    # 两端各留一点非零底，彻底杜绝 0 权重
    fade = np.maximum(fade, 1e-3)
    mask[:ov, :] *= fade[:, None]
    mask[-ov:, :] *= fade[::-1][:, None]
    mask[:, :ov] *= fade[None, :]
    mask[:, -ov:] *= fade[None, ::-1]
    return mask


def _compute_tiles(W: int, H: int, tile_size: int, overlap: int):
    """计算 tile 坐标列表 [(x, y, w, h), ...]

    tile_size / overlap 会被夹到合法范围：
    以前 tile_size 可大于目标尺寸（UI 允许 tile=1024 而宽高最小 768），
    导致 canvas 切片被裁成 768 而 tile 输出是 1024×1024 →
    "non-broadcastable output operand" 直接崩。
    """
    tile_size = int(min(tile_size, W, H))
    tile_size = max(8, tile_size)
    overlap = int(min(overlap, max(0, tile_size - 8)))
    stride = max(1, tile_size - overlap)

    xs = list(range(0, max(W - tile_size, 0) + 1, stride))
    ys = list(range(0, max(H - tile_size, 0) + 1, stride))
    # 保证最后一块贴到右/下边
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    tiles = []
    for y in ys:
        for x in xs:
            tiles.append((x, y, tile_size, tile_size))
    return tiles


def tiled_img2img(
    pipe,
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    target_width: int = 2048,
    target_height: int = 2048,
    tile_size: int = 768,
    overlap: int = 96,
    strength: float = 0.4,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.0,
    seed: int = -1,
    callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> Image.Image:
    """
    分块 img2img 生成大图(主力函数)。
    
    工作流:
      1. 把 init_image 上采样到 target 尺寸
      2. 切成 N 个 tile
      3. 每个 tile 独立 img2img
      4. 用羽化权重融合回大图
    
    Args:
      pipe: diffusers img2img pipeline
      init_image: 输入图(任意尺寸,会被上采样到 target)
      target_width/height: 最终目标分辨率
      tile_size: 单块大小(建议 512/768)
      overlap: 重叠像素(消接缝, 建议 64-128)
      strength: img2img 强度(0.3-0.6 保留原图,>0.6 大改)
    
    Returns:
      PIL.Image (target_width x target_height)
    """
    device = pipe.device

    # 目标尺寸对齐 8 的倍数（VAE/UNet 要求），并保证不小于一个 tile
    target_width = max(8, (int(target_width) // 8) * 8)
    target_height = max(8, (int(target_height) // 8) * 8)
    # 先夹紧 tile 参数，保证后面每块实际的 (w,h) 与 blend_mask 完全一致
    tile_size = max(8, min(int(tile_size), target_width, target_height))
    overlap = max(0, min(int(overlap), tile_size - 8))

    # 1. 上采样初始图到目标尺寸(Lanczos 高质量)
    if init_image.size != (target_width, target_height):
        init_image = init_image.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )

    # 2. 计算 tile 网格
    tiles = _compute_tiles(target_width, target_height, tile_size, overlap)
    total = len(tiles)
    logger.info(f"🧩 Tiled Diffusion: {target_width}x{target_height}, "
          f"tile={tile_size}, overlap={overlap}, 共 {total} 块")

    # 3. 准备输出画布(累加 + 权重)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.float32)
    weight = np.zeros((target_height, target_width, 1), dtype=np.float32)
    blend_mask = _make_blend_mask(tile_size, tile_size, overlap)[..., None]

    # 4. 逐块生成
    for idx, (x, y, w, h) in enumerate(tiles):
        if cancel_check and cancel_check():
            raise InterruptedError("用户取消")

        # 边界兜底：极端参数下仍保证切片与 mask 尺寸一致
        w = min(w, target_width - x)
        h = min(h, target_height - y)
        if w <= 0 or h <= 0:
            continue
        mask = blend_mask[:h, :w, :]

        logger.info(f"  🧩 [{idx+1}/{total}] 块 ({x},{y}) {w}x{h}")

        # 裁出 tile
        tile_img = init_image.crop((x, y, x + w, y + h))

        # 生成器
        gen = None
        if seed >= 0:
            gen = torch.Generator(device).manual_seed(seed + idx)

        # img2img 单块
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=tile_img,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                width=w,
                height=h,
            ).images[0]
        except TypeError:
            # 老版本不支持 width/height 参数
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=tile_img,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            ).images[0]

        # 融合到画布
        tile_arr = np.array(result, dtype=np.float32)
        if tile_arr.shape[0] != h or tile_arr.shape[1] != w:
            # 管线可能返回尺寸不一致的结果，先对齐再融合
            tile_arr = np.array(
                Image.fromarray(tile_arr.astype(np.uint8)).resize(
                    (w, h), Image.Resampling.LANCZOS),
                dtype=np.float32)
        canvas[y:y+h, x:x+w, :] += tile_arr * mask
        weight[y:y+h, x:x+w, :] += mask

        # 进度回调
        if callback:
            callback(idx + 1, total)

    # 5. 归一化(除以累计权重)
    #    mask 已保证恒 > 0，这里的兜底只为防御异常输入
    uncovered = weight <= 1e-6
    if uncovered.any():
        logger.warning(f"⚠️ 有 {int(uncovered.sum())} 个像素未被任何 tile 覆盖，将置黑")
    weight = np.where(uncovered, 1.0, weight)
    final = (canvas / weight).clip(0, 255).astype(np.uint8)

    return Image.fromarray(final)


def upscale_with_tiled_diffusion(
    pipe,
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    scale: float = 2.0,
    **kwargs
) -> Image.Image:
    """便捷封装: 按倍数放大"""
    W, H = init_image.size
    target_w = int(W * scale)
    target_h = int(H * scale)
    # 对齐 8 的倍数
    target_w = (target_w // 8) * 8
    target_h = (target_h // 8) * 8
    return tiled_img2img(
        pipe, init_image, prompt, negative_prompt,
        target_width=target_w, target_height=target_h,
        **kwargs
    )
