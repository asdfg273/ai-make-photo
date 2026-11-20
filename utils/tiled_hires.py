# utils/tiled_hires.py
"""分块高清（tiled img2img）：把"整图放大再重绘"拆成"放大 → 逐块重绘 → 羽化拼回"。

为什么需要它
------------
整图 img2img 放大 2 倍时，扩散的注意力与激活都按**整张图**分配显存：
1024² → 2048² 在 8GB 卡上基本是必爆（或者被 WDDM 甩到共享内存，
速度掉一个数量级）。分块把显存需求钉在「一块的尺寸」上，与目标大图无关。

和 `nodes/tile_nodes.py` 的分工
--------------------------------
· 本模块是**无 UI、无 Qt、无 workflow 依赖**的实现层，主界面与工作流节点
  共用同一套逐块循环语义（种子递增、逐块尺寸、跳过条件、元数据键）。
· `nodes/tile_nodes.py` 的 `TiledHiresFix` 负责把工作流参数翻译成这里的入参，
  并把 NodeContext 的 status/progress/cancel 包成回调。

关于「每块必须显式给 height/width」
-----------------------------------
Anima 这类 ModularPipeline 会按传入的参数选 block：给了 image= 就走
vae_encoder + denoise（图生图）。但如果**不传 height/width**，它不会用
输入图的尺寸，而是用管线默认的 1024 —— 于是每一块都在跑一张 1024 的整图
（实测：4 块 × 640² 从 15.5 秒变成 114.8 秒，峰值显存 7.1GB 涨到 9.7GB）。
所以这里逐块传 `height=t["h"], width=t["w"]`，并在块返回值尺寸不符时
按块尺寸重采样兜底。
"""
from __future__ import annotations

import inspect
import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils import tiling

logger = logging.getLogger(__name__)

__all__ = [
    "SIZE_MULTIPLE", "BLEND_MODES", "plan_target_size", "resolve_mode",
    "tiled_img2img", "describe", "MIN_EFFECTIVE_STEPS", "boost_steps",
    "TILED_MAX_DENOISE",
]

# 扩散模型对尺寸敏感，统一对齐到 16 的倍数。
# 8 在 SD/SDXL 够用，但 Anima 这类块式管线是 VAE /8 × transformer patch /2
# = 16 —— 给 472（8 的倍数但不是 16 的倍数）会让 patch_embed 的 reshape
# 直接崩（实测 "shape '[1, 17, 1, 1, 32, 2, 29, 2]' is invalid"）。
SIZE_MULTIPLE = 16

# 每块至少要跑这么多**生效**步数。
# strength 会把总步数按比例切片：steps=4、strength=0.35 → 生效 1 步，
# 实测每块会被整块重画（MAD 23.7，一只猫变三只）；同样强度下把总步数
# 抬到 9（生效 3 步）就恢复正常（真机实测数据见 docs/audits 第 14 章）。
MIN_EFFECTIVE_STEPS = 3

# 分块重绘的强度上限。
# 每块只看得到自己那 640px 的内容，强度一大就会"各画各的"：真机实测
# 一只猫在 1024² 里被画成三只，块边界也会显形。0.25 以内是"细化"
# 而不是"重新构图"（0.15 → 偏离底图 MAD 5.7；0.35 → 11.2 且主体重复）。
TILED_MAX_DENOISE = 0.25

# 界面/节点上的融合方式 → tiling.blend_stitch 的 mode
BLEND_MODES = {
    "羽化": "feather",
    "线性": "linear",
    "硬切": "hard",
    "feather": "feather",
    "linear": "linear",
    "hard": "hard",
}

# 高清方式（界面「高清方式」下拉 / config.hires_mode）
MODE_AUTO = "auto"
MODE_LATENT = "latent"     # 整图 img2img（快，吃显存）
MODE_TILED = "tiled"       # 分块 img2img（慢一点，省显存）

# 总量低于此值时 auto 选分块：8GB 卡整图 2× 放大非常容易爆
AUTO_TILED_BELOW_GB = 10.0


def resolve_mode(ui_text: str = "", cfg_mode: str = "auto",
                 total_gb: Optional[float] = None) -> str:
    """决定本次高清走「整图」还是「分块」。

    优先级：界面下拉 > config > 自动（按显存总量）。
    与 `workflow/builder.py` 的同名逻辑保持一致；两处都只看这三样，
    不按模型名判断。
    """
    text = str(ui_text or "")
    if "分块" in text:
        return MODE_TILED
    if "整图" in text:
        return MODE_LATENT
    mode = str(cfg_mode or MODE_AUTO).strip().lower()
    if mode in (MODE_LATENT, MODE_TILED):
        return mode
    if total_gb is None:
        try:
            from workflow.memory_policy import DeviceCapabilities

            total_gb = DeviceCapabilities.detect().total_gb
        except Exception as e:                      # 无 GPU / 依赖缺失
            logger.debug(f"显存总量探测失败（按整图处理）: {e}")
            return MODE_LATENT
    try:
        if float(total_gb) < AUTO_TILED_BELOW_GB:
            return MODE_TILED
    except (TypeError, ValueError):
        pass
    return MODE_LATENT


def plan_target_size(image, scale: float,
                     multiple: int = SIZE_MULTIPLE) -> Tuple[int, int]:
    """按倍率算目标尺寸，并对齐到 `multiple` 的倍数。"""
    scale = max(1.0, float(scale))
    raw_w = max(multiple, int(round(image.width * scale)))
    raw_h = max(multiple, int(round(image.height * scale)))
    return tiling.fit_size_to_multiple(raw_w, raw_h, multiple)


def _resample(name: str):
    from PIL import Image

    key = str(name or "Lanczos").strip().lower()
    table = {
        "lanczos": Image.LANCZOS,
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }
    return table.get(key, Image.LANCZOS)


def _new_seed() -> int:
    return random.randint(0, 2 ** 31 - 1)


def _generator(device: str, seed: int):
    import torch

    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except Exception:                               # CPU-only / 非法 device
        return torch.Generator().manual_seed(int(seed))


def describe(meta: Dict[str, Any]) -> str:
    """把元数据变成一行中文说明（日志与状态栏共用）。"""
    steps = meta.get("tile_steps")
    steps_txt = f" · {steps} 步" if steps else ""
    return (f"{meta.get('hires_size', '?')} · 块 {meta.get('tile_size', '?')}"
            f"/重叠 {meta.get('tile_overlap_effective', meta.get('tile_overlap', '?'))}"
            f" · {meta.get('tile_count', 0)} 块"
            f"{steps_txt} · "
            f"强度 {meta.get('tile_denoise', '?')} · "
            f"{meta.get('tile_method', 'Lanczos')} 放大")


def _align_to(image, size: Tuple[int, int]):
    """把块输出重采样到计划尺寸（管线可能按 8 的倍数裁过，不齐就会拼错位）。"""
    from PIL import Image

    if image.size == tuple(size):
        return image
    return image.resize(tuple(size), Image.LANCZOS)


def _pick_image(res) -> Optional[Any]:
    imgs = getattr(res, "images", None)
    if imgs is None:
        return None
    if not isinstance(imgs, (list, tuple)):
        return imgs
    return imgs[0] if imgs else None


def _accepts_kw(pipe, *names: str) -> bool:
    """管线 `__call__` 是否**显式声明**了这些形参。

    注意：diffusers 的管线全都带 `**kwargs`，所以"接受"不等于"会用" ——
    SD1.5/SDXL 的 img2img 没有 width/height 形参，传进去只会被静默忽略；
    Anima 的 ModularPipeline 走 `**kwargs → PipelineState`，必须给
    width/height，否则按默认 1024 出图。所以这里按**形参**判断。
    """
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return True
    return all(name in params for name in names)


def _looks_modular(pipe) -> bool:
    """是不是 diffusers 的 ModularPipeline（块式，按 state 传参）。"""
    blocks = getattr(pipe, "blocks", None)
    return blocks is not None and hasattr(blocks, "sub_blocks")


def _needs_explicit_size(pipe) -> bool:
    """要不要显式告诉管线出图尺寸。

    - 显式有形参（SD3 img2img、各类 inpaint）：给；
    - ModularPipeline（Anima）/纯 `**kwargs` 包装器：必须给，否则默认 1024；
    - SD1.5/SDXL 的 img2img：有别的显式形参、但没有 width/height ——
      尺寸由输入图决定，给了也只会被静默忽略。
    """
    if _looks_modular(pipe):
        return True
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return True
    explicit = [p for p in params.values()
                if p.kind not in (inspect.Parameter.VAR_KEYWORD,
                                  inspect.Parameter.VAR_POSITIONAL)
                and p.name != "self"]
    if not explicit:
        # 只吃 **kwargs 的管线（包装器 / 块式）—— 只能按"需要尺寸"处理
        return True
    return all(name in params for name in ("width", "height"))


def boost_steps(steps: int, denoise: float,
                minimum: int = MIN_EFFECTIVE_STEPS) -> int:
    """算出让"生效步数 >= minimum"所需的总步数（不改变 strength 本身）。

    为什么要这个：img2img 的有效步数是 `steps * strength` 向下取整。
    steps=4、strength=0.35 只有 1 步生效 —— 这种"一步定生死"的去噪会把
    每块内容整块重画（真机 MAD 23.7，一只猫被画成三只）；把总步数抬到
    `minimum / strength` 后（同样是 strength 0.35，总步数 9），生效 3 步，
    MAD 降到 ~10 并保留原构图。

    `strength` 是用户要的"改动幅度"，不能动；能动的只有总步数。
    """
    try:
        steps = int(steps)
        denoise = float(denoise)
    except (TypeError, ValueError):
        return int(steps or 0)
    if steps <= 0 or denoise <= 0.0:
        return steps
    effective = max(1, int(round(steps * denoise)))
    if effective >= int(minimum):
        return steps
    needed = int(math.ceil(int(minimum) / max(denoise, 1e-3)))
    return max(steps, needed)


def tiled_img2img(
    pipe,
    image,
    *,
    scale: float = 2.0,
    tile_size: int = 640,
    overlap: int = 64,
    denoise: float = 0.35,
    call_kwargs: Optional[Dict[str, Any]] = None,    seed: int = -1,
    method: str = "Lanczos",
    blend: str = "羽化",
    color_correct: bool = True,
    feather: Optional[int] = None,
    max_tiles: int = 0,
    max_denoise: float = TILED_MAX_DENOISE,
    pass_size: Optional[bool] = None,
    device: str = "cuda",
    call: Optional[Callable[..., Any]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    status: Optional[Callable[[str], None]] = None,
    check_cancel: Optional[Callable[[], None]] = None,
    after_tile: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """放大 `image` 并逐块 img2img，最后羽化拼回整图。

    参数
    ----
    pipe        : 已就绪的 img2img 管线（Anima 的 ModularPipeline 直接传同一个对象）
    scale       : 放大倍率（1.0 表示不放大，按原尺寸分块重绘）
    tile_size   : 块边长（会被显存策略与目标尺寸夹到合理范围）
    overlap     : 块间重叠（下界，实际可能更大，见 utils.tiling.plan_tiles）
    denoise     : 重绘强度；<=0 表示不跑扩散，等价于纯放大
    call_kwargs : 每块都带的额外参数（prompt / num_inference_steps / guidance_scale …）
    seed        : 随机种子；-1 表示随机（第 i 块用 base+i，保证可复现）
    blend       : "羽化" / "线性" / "硬切"
    max_tiles   : >0 时块数超限直接报错（防止误设参数跑上一小时）
    max_denoise : 强度上限（默认 TILED_MAX_DENOISE=0.25）；超过会被夹住并告警
    pass_size   : 是否给每块传 width/height。None = 看管线签名自动决定：
                  SD1.5/SDXL 的 img2img `__call__` **没有** width/height 参数
                  （尺寸跟着传入的 image 走），硬传会 TypeError；而 Anima 的
                  ModularPipeline 不传就按默认 1024 出图，必须传
    call        : 自定义调用器 `call(**kwargs) -> result`；默认 `pipe(**kwargs)`
    progress    : `progress(done, total)` 进度回调
    status      : `status(text)` 状态文本回调
    check_cancel: 每块前后调用，抛异常即中止
    after_tile  : `after_tile(index, total)` 每块跑完调用（工作流用它省显存）

    返回 (PIL.Image, meta)；meta 的键与工作流节点保持一致，便于写进图片元数据。
    """
    from PIL import Image                      # noqa: F401  (尺寸对齐要用)

    image = image.convert("RGB")
    call_kwargs = dict(call_kwargs or {})
    # 这些参数与"逐块给 image/尺寸"冲突，一律清掉，避免调用方踩坑
    for key in ("image", "width", "height", "strength", "latents"):
        call_kwargs.pop(key, None)

    denoise = max(0.0, min(1.0, float(denoise)))
    scale = max(1.0, float(scale))

    # 强度上限：每块只看得到自己那部分画面，强度一大就"各画各的"
    requested_denoise = denoise
    if max_denoise is not None and 0.0 < float(max_denoise) < denoise:
        denoise = float(max_denoise)
        # 只说结果，不劝人（"分块强度"滑杆本身就有这个上限，用户能选）
        logger.info(f"🧩 分块强度 {requested_denoise:g} 按上限 {denoise:g} 执行"
                    f"（分块时每块只看到局部，强度太大各块会各画各的）")

    # 生效步数不足会把每块整块重画（一只猫变三只），这里抬总步数、不动强度
    steps_in = int(call_kwargs.get("num_inference_steps", 0) or 0)
    steps_used = steps_in
    if steps_in > 0 and denoise > 0.0:
        steps_used = boost_steps(steps_in, denoise)
        if steps_used != steps_in:
            logger.info(
                f"ℹ️ 分块高清：{steps_in} 步 × 强度 {denoise:g} 只有 "
                f"{max(1, int(round(steps_in * denoise)))} 步生效，"
                f"总步数提到 {steps_used}（强度不变）—— 否则每块会被整块重画")
            call_kwargs["num_inference_steps"] = steps_used
    target_w, target_h = plan_target_size(image, scale)
    status = status or (lambda _t: None)
    check_cancel = check_cancel or (lambda: None)
    runner = call or (lambda **kw: pipe(**kw))

    base_seed = _new_seed() if int(seed) < 0 else int(seed)
    mode = BLEND_MODES.get(str(blend).strip(), "feather")
    blend_label = str(blend or "羽化")

    status(f"🔍 放大 {scale:g}× · {image.width}x{image.height} → "
           f"{target_w}x{target_h} · {method}")
    upscaled = image.resize((target_w, target_h), _resample(method))
    check_cancel()

    meta: Dict[str, Any] = {
        "tile_scale": scale,
        "tile_method": str(method),
        "tile_blend": blend_label,
        "tile_denoise": denoise,
        "tile_denoise_requested": requested_denoise,
        "tile_color_correct": bool(color_correct),
        "tile_seed": base_seed,
        "tile_size": int(tile_size),
        "tile_overlap": int(overlap),
        "tile_steps": steps_used,
        "tile_effective_steps": (max(1, int(round(steps_used * denoise)))
                                 if steps_used else 0),
        "tiled_hires": True,
        "hires_size": f"{target_w}x{target_h}",
    }

    # denoise<=0：语义是"一次扩散都不要跑"，连块都不用切
    if denoise <= 0.0:
        logger.info("ℹ️ 重绘强度为 0：跳过分块与扩散，等价于纯放大")
        meta.update({"tile_count": 0, "tile_total": 0,
                     "tile_note": "denoise=0，未跑扩散"})
        if progress:
            progress(1, 1)
        return upscaled, meta

    tile = tiling.adaptive_tile_size((target_w, target_h), int(tile_size))
    if tile != int(tile_size):
        logger.info(f"ℹ️ 分块尺寸由 {tile_size} 调整为 {tile}（受显存策略/"
                    f"图片尺寸限制，且对齐到 {SIZE_MULTIPLE} 的倍数）")
    overlap_n = tiling.clamp_overlap(int(overlap), tile)
    meta["tile_size"] = tile
    meta["tile_overlap"] = overlap_n

    plan = tiling.plan_tiles(target_w, target_h, tile, overlap_n)
    n = len(plan)
    if max_tiles and n > max_tiles:
        raise ValueError(
            f"预计需要 {n} 块，超过「最大块数」限制 {max_tiles}。"
            f"请增大块尺寸、降低放大倍率，或把最大块数设为 0 取消限制"
            f"（尺寸 {target_w}x{target_h}，块 {tile}，重叠 {overlap_n}）")
    if not tiling.verify_coverage(target_w, target_h, plan):
        raise RuntimeError(f"分块方案未覆盖全部像素（{target_w}x{target_h}，"
                           f"块 {tile}，重叠 {overlap_n}），已中止以避免出残图")

    logger.info(f"🧩 分块高清：{tiling.describe_plan(plan)} · 强度 {denoise} · "
                f"{method} 放大 · {blend_label}融合 · "
                f"色彩对齐{'开' if color_correct else '关'}")
    logger.info(f"🎲 本轮基础种子 {base_seed}"
                f"（{'用户指定' if int(seed) >= 0 else '随机生成'}，"
                f"第 i 块用 {base_seed}+i，可复现）")

    items: List[Tuple[Dict[str, Any], Any]] = []
    if pass_size is None:
        pass_size = _needs_explicit_size(pipe)
    if not pass_size:
        logger.info("ℹ️ 该 img2img 管线不接受 width/height 参数（SD1.5/SDXL 常见），"
                    "每块尺寸由传入的裁剪块决定")
    meta["tile_pass_size"] = bool(pass_size)
    for i, t in enumerate(plan, start=1):
        check_cancel()
        status(f"分块高清 {i}/{n} …（{t['w']}x{t['h']} @ {t['x']},{t['y']}）")
        crop = upscaled.crop((t["x"], t["y"],
                              t["x"] + t["w"], t["y"] + t["h"]))
        call_kw = dict(call_kwargs)
        call_kw["image"] = crop
        # ⚠️ 必须传块尺寸：ModularPipeline 不传就按默认 1024 出图，
        # 等于每块都在跑整图（实测慢 7 倍、峰值显存 +2.6GB）
        if pass_size:
            call_kw["height"] = t["h"]
            call_kw["width"] = t["w"]
        call_kw["strength"] = denoise
        call_kw["generator"] = _generator(device, base_seed + i)
        try:
            res = runner(**call_kw)
        finally:
            if after_tile:
                after_tile(i, n)
        out = _pick_image(res)
        if out is None:
            raise RuntimeError(f"第 {i}/{n} 块 img2img 没有返回图像")
        out = out.convert("RGB")
        if out.size != (t["w"], t["h"]):
            logger.debug(f"块 {i} 返回尺寸 {out.size} != {t['w']}x{t['h']}，"
                         f"已重采样对齐")
            out = _align_to(out, (t["w"], t["h"]))
        items.append((t, out))
        if progress:
            progress(i, n)

    check_cancel()
    status(f"🧵 融合 {len(items)} 块 → {target_w}x{target_h}")
    # 羽化宽度按**实际**重叠算，不按请求值：请求 48 时 plan_tiles 为了均摊
    # 间距往往给出 256 的重叠，如果仍按 48/2=24 羽化，就会留下 230 多像素的
    # "只属于某一块"的区域 —— 平色区能看出明显的色阶台阶（实测 y=384 处）。
    ov_eff = tiling.effective_overlap(plan)
    feath = (max(8, min(int(ov_eff[0]), int(ov_eff[1])) // 2)
             if feather is None else int(feather))
    out_img = tiling.blend_stitch((target_w, target_h), items,
                                  base=upscaled, feather=feath, mode=mode,
                                  color_correct=color_correct).convert("RGB")
    check_cancel()

    meta.update({
        "tile_count": len(items),
        "tile_total": len(items),
        "tile_overlap_effective": tiling.effective_overlap(plan),
        "tile_feather": feath,
        "tile_no_tiling": False,
    })
    logger.info(f"✅ 分块高清完成：{describe(meta)}")
    return out_img, meta
