# nodes/tile_nodes.py
"""分块放大节点：TiledHiresFix（分块高清修复）/ TiledUpscale（分块纯缩放）。

为什么需要"分块"的高清修复
--------------------------
`HiresFix` 是"整图放大后再跑一次低强度图生图"。在 8GB 卡上，2 倍放大到
1024x1024 之后整图重绘的显存峰值会顶到物理上限，再往上（4K）直接没戏。
分块把"重绘"的输入从"整张大图"换成"一张 tile 大小的小图"：显存峰值与
**最终尺寸无关**，只与 `tile_size` 有关，于是 2048 / 4096 都能跑。

    base ──resize(scale)──┐
                          ├─ 切成 N 块（带 overlap 重叠）─ 每块 img2img ─ 融合 ─ image
                          └─（denoise=0 时直接就是放大结果，不跑模型）

节点顺序自由
------------
这是**普通节点**：输入输出都是标准端口，所以工作流里可以随意排

    base → TiledHiresFix → ADetailer      （先补细节，再修脸）
    base → ADetailer     → TiledHiresFix  （先修脸，再整体放大重绘）

执行器只按连线拓扑排序，不做任何"高清必须在修脸之后"的假设。

刻意与 `HiresFix` 保持一致的几件事
---------------------------------
· 目标尺寸、tile 尺寸都对齐到 8（VAE 的 8 倍下采样），免得 diffusers 内部
  悄悄裁像素导致融合尺寸对不上；
· 能力判断走 `adapters.capabilities_for(...).require("img2img")`，
  **不判断模型名**；
· 种子复用 `nodes.sampling_nodes` 的 `_resolve_seed` / `_generator`，
  语义（-1=随机）与采样节点完全一致，不会出现两套随机规则。
"""
from __future__ import annotations

import logging

from workflow import adapters as adapters_mod
from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import as_image
from utils import tiled_hires

logger = logging.getLogger(__name__)

# VAE 的 8 倍下采样：宽高、tile 都对它对齐
_SIZE_MULTIPLE = 8

_RESAMPLE = {
    "LANCZOS": "LANCZOS",
    "BICUBIC": "BICUBIC",
    "NEAREST": "NEAREST",
}

# 界面选项 -> utils.tiling 的 mode
_BLEND_MODE = {"羽化": "feather", "线性": "linear", "硬切": "hard"}

# 显存模式
_VRAM_AUTO = "自动"
_VRAM_SAVER = "省显存"
_VRAM_BALANCED = "平衡"
_VRAM_PERF = "高性能"

# 「平衡」模式下每隔几块清一次缓存
_BALANCED_EVERY = 4

# HiresFix 的 estimated_vram_mb 是 2200（整图重绘）。这里块的尺寸可控，
# 峰值 = 底模常驻 + 一块 tile 的激活值，比整图重绘低一截。
_VRAM_HIRES_MB = 1800


def _resample(name):
    """把方法名转成 PIL 重采样常量；未知名字按 LANCZOS 处理。"""
    from PIL import Image

    key = str(name or "LANCZOS").strip().upper()
    if key not in _RESAMPLE:
        key = "LANCZOS"
    return getattr(Image.Resampling, key, None) or getattr(Image, key)


def _resize(image, size, method: str = "LANCZOS"):
    """按 (宽, 高) 缩放；尺寸没变则原样返回（避免无意义的像素重采样）。

    注意：只在真正需要缩放时才会调用 `resize`，而 `resize` 返回的是**新对象**，
    绝不原地修改调用方传进来的图。
    """
    if image.size == (int(size[0]), int(size[1])):
        return image
    return image.resize((int(size[0]), int(size[1])), _resample(method))


def _align_to(image, size):
    """把管线返回的图按 LANCZOS 对齐到指定尺寸（尺寸一致则原样返回）。"""
    size = (int(size[0]), int(size[1]))
    if image.size == size:
        return image
    return image.resize(size, _resample("LANCZOS"))


def _parse_scale(raw):
    """倍率既可能是 float，也可能是老工作流里存的字符串 "2.0"。"""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(str(raw).strip())


def _apply_negative(adapter, kw: dict, neg) -> None:
    """负面条件优先用独立输入（细节与 Sampler 节点保持一致）。"""
    if neg is None:
        return
    if getattr(neg, "negative_embeds", None) is not None:
        kw["negative_prompt_embeds"] = neg.negative_embeds
        if getattr(neg, "negative_pooled", None) is not None:
            kw["negative_pooled_prompt_embeds"] = neg.negative_pooled


def _free_vram(why: str = "") -> bool:
    """尽力回收显存；CPU-only 或 torch 缺席时静默返回 False。

    每一步都包在 try/except 里：这个函数只应该"更省显存"，
    绝不能因为某个环境没有 CUDA 就让整条工作流挂掉。
    """
    ok = False
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            ok = True
    except Exception as e:
        logger.debug(f"显存回收跳过（{why}）: {e}")
    return ok


def _open_attention_slicing(pipe) -> bool:
    """尝试打开 attention slicing（省显存模式用）。

    ⚠️ 项目实测结论（`utils/vram_manager._set_attention_slicing` 的注释）：
    在 torch 2.x + SDPA 下开 attention slicing 会**更慢、峰值还更高**
    （1024x1024: 开 111.8s/6.26GB，不开 9.53s/2.62GB）。
    所以这里只是"按用户明确要求去尝试"，失败就跳过，并把风险写进日志，
    绝不把它当成默认省显存手段。
    """
    for name in ("enable_attention_slicing", "enable_vae_slicing"):
        fn = getattr(pipe, name, None)
        if not callable(fn):
            continue
        try:
            fn()
            logger.info(f"ℹ️ 省显存模式：已尝试 {name}()（本项目实测此项可能变慢，"
                        f"仅在确实 OOM 时使用）")
            return True
        except Exception as e:
            logger.debug(f"{name} 失败（忽略）: {e}")
    return False


@node(
    "TiledHiresFix",
    category="放大",
    description="分块高清修复：放大 → 切成带重叠的小块 → 每块图生图重绘 → "
                "羽化融合。显存峰值只取决于块尺寸，与最终分辨率无关，"
                "所以能在 8GB 卡上做到 2K/4K。denoise=0 时退化为纯放大。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "image": PortSpec(DataType.IMAGE, "图像"),
            "positive": PortSpec(DataType.TEXT_COND, "正面条件"),
            "negative": PortSpec(DataType.TEXT_COND, "负面条件", optional=True)},
    outputs={"image": PortSpec(DataType.IMAGE, "图像"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "scale": ParamSpec("float", default=2.0, label="放大倍率",
                           minimum=1.0, maximum=4.0, step=0.05,
                           help="先按此倍率放大，再分块重绘补细节"),
        "tile_size": ParamSpec("int", default=640, label="分块尺寸",
                               minimum=256, maximum=1536, step=64,
                               help="每块边长。8GB 显卡建议 512~640；"
                                    "块越大上下文越完整，但显存峰值越高"),
        "overlap": ParamSpec("int", default=48, label="重叠像素",
                             minimum=0, maximum=256, step=8,
                             help="相邻块的重叠宽度，用于消除接缝。"
                                  "8GB 显卡建议 32~64；越大越慢"),
        "denoise": ParamSpec("float", default=0.35, label="重绘强度",
                             minimum=0.0, maximum=1.0, step=0.05,
                             help="0 = 只放大不重绘（不跑模型）；"
                                  "8GB 显卡建议 0.25~0.45，过高会让块内容互相打架"),
        "steps": ParamSpec("int", default=24, label="步数",
                           minimum=1, maximum=150,
                           help="每一块的采样步数（总耗时 ≈ 步数 × 块数，"
                                "块多时建议适当降低）"),
        "cfg": ParamSpec("float", default=7.0, label="CFG",
                         minimum=0.0, maximum=30.0, step=0.5,
                         help="提示词引导强度；分块重绘时建议比文生图略低"),
        "seed": ParamSpec("seed", default=-1, label="种子",
                          help="-1 表示随机（随机值会记进日志与元数据）；"
                               "整节点共用一个基础种子，第 i 块用 种子+i，"
                               "同一参数重复运行结果一致"),
        "upscale_method": ParamSpec("choice", default="Lanczos", label="放大方式",
                                    choices=("Lanczos", "Bicubic", "Nearest"),
                                    help="放大底图用的插值方法，扩散只负责补细节"),
        "blend": ParamSpec("choice", default="羽化", label="边缘融合",
                           choices=("羽化", "线性", "硬切"),
                           help="羽化=平滑过渡（推荐）；线性=整块渐变；"
                                "硬切=不做融合（重叠处后一块覆盖前一块，"
                                "高 denoise 时会看到直线接缝）"),
        "color_correct": ParamSpec("bool", default=True, label="块间色彩对齐",
                                   help="按每块均值/方差对齐到底图，"
                                        "消除块与块之间的色差；会略微增加耗时"),
        "vram_mode": ParamSpec("choice", default="自动", label="显存模式",
                               choices=("自动", "省显存", "平衡", "高性能"),
                               help="自动=跟随当前显存策略；省显存=每块之间清缓存"
                                    "并尝试 attention slicing；平衡=每 N 块清一次；"
                                    "高性能=不清缓存直接连跑"),
        "max_tiles": ParamSpec("int", default=0, label="最大块数",
                               minimum=0, maximum=4096,
                               help="安全阀：预计块数超过它就立刻报错，"
                                    "而不是跑到一半 OOM。0 = 不限制"),
    },
    resources=ResourceRequirement(requires=("cuda",),
                                  estimated_vram_mb=_VRAM_HIRES_MB,
                                  supports_cpu_offload=True,
                                  supports_tiled=True),
)
def run_tiled_hires_fix(ctx, node, inputs):
    """分块高清修复：放大 → 分块 img2img → 融合。"""
    import torch

    # 复用采样节点的种子工具，避免两处随机语义不一致
    from nodes.sampling_nodes import _generator, _resolve_seed
    from utils import tiling

    model = inputs.get("model")
    image = as_image(inputs.get("image"))
    pos = inputs.get("positive")
    neg = inputs.get("negative")
    if model is None:
        raise ValueError("缺少输入 model")
    if image is None:
        raise ValueError("缺少输入 image（分块高清修复需要一张底图）")

    adapter = getattr(model, "adapter", None)
    if adapter is None:
        raise RuntimeError("模型没有适配器，无法确定 img2img 调用方式")
    # 一律按**能力**判断，不按模型名判断
    caps = adapters_mod.capabilities_for(getattr(model, "arch_id", "") or "sd15")
    caps.require("img2img")
    if not getattr(adapter.caps, "img2img", True):
        raise adapters_mod.UnsupportedFeatureError(
            "图生图", getattr(model, "arch_id", "unknown"))

    # ---- 参数 ----
    scale = max(1.0, _parse_scale(node.params.get("scale", 2.0)))
    denoise = float(node.params.get("denoise", 0.35))
    tile_param = int(node.params.get("tile_size", 640))
    overlap = int(node.params.get("overlap", 48))
    steps = int(node.params.get("steps", 24))
    cfg = float(node.params.get("cfg", adapter.default_guidance()))
    seed_param = int(node.params.get("seed", -1) or -1)
    base_seed = _resolve_seed(seed_param)
    method = str(node.params.get("upscale_method", "Lanczos") or "Lanczos")
    blend_label = str(node.params.get("blend", "羽化") or "羽化")
    mode = _BLEND_MODE.get(blend_label.strip(), "feather")
    color_correct = bool(node.params.get("color_correct", True))
    vram_mode = str(node.params.get("vram_mode", _VRAM_AUTO) or _VRAM_AUTO).strip()
    max_tiles = max(0, int(node.params.get("max_tiles", 0) or 0))
    if denoise < 0.0:
        denoise = 0.0
    if denoise > 1.0:
        denoise = 1.0

    # ---- 1) 目标尺寸（向下对齐到 8 的倍数）----
    raw_w = max(8, int(round(image.width * scale)))
    raw_h = max(8, int(round(image.height * scale)))
    target_w, target_h = tiling.fit_size_to_multiple(raw_w, raw_h,
                                                     _SIZE_MULTIPLE)
    if (target_w, target_h) != (raw_w, raw_h):
        # 不静默取整：把"你要 2.0 倍，实际得到 1023x767"讲清楚
        logger.info(f"ℹ️ 目标尺寸对齐到 {_SIZE_MULTIPLE} 的倍数："
                    f"{raw_w}x{raw_h} → {target_w}x{target_h}")

    # 底图：放大只碰副本，调用方的图保持原样
    ctx.status(f"🔍 放大 {scale}× · {image.width}x{image.height} → "
               f"{target_w}x{target_h} · {method}")
    upscaled = _resize(image.convert("RGB"), (target_w, target_h), method)
    ctx.check_cancel()

    # 纯放大（不重绘）：到这里就可以返回了
    # denoise<=0 的语义就是"一次图生图都不要跑"，所以连 img2img 管线都不取。
    if denoise <= 0.0:
        logger.info("ℹ️ 重绘强度为 0：跳过分块与扩散，本节点等价于纯放大")
        meta = {"tile_size": tile_param, "tile_overlap": overlap,
                "tile_denoise": 0.0, "tile_scale": scale, "tile_count": 0,
                "tile_total": 0, "tile_method": method, "tile_blend": blend_label,
                "tile_vram_mode": vram_mode, "tile_color_correct": color_correct,
                "tiled_hires": True, "hires_size": f"{target_w}x{target_h}",
                "tile_note": "denoise=0，未跑扩散"}
        ctx.metadata.update(meta)
        ctx.progress(1, 1)
        return {"image": upscaled, "metadata": meta}

    # ---- 2) 块尺寸：用户设置与显存策略取较小者 ----
    tile = tiling.adaptive_tile_size((target_w, target_h), tile_param)
    if tile != tile_param:
        logger.info(f"ℹ️ 分块尺寸由 {tile_param} 调整为 {tile}（受显存策略/"
                    f"图片尺寸限制，且对齐到 {_SIZE_MULTIPLE} 的倍数）")
    overlap = tiling.clamp_overlap(overlap, tile)

    pipe = adapter.pipe(model, "img2img")
    if pipe is None:
        raise adapters_mod.UnsupportedFeatureError(
            "分块高清修复", model.arch_id, "该架构没有可用的 img2img 管线")

    kw = adapter.sample_kwargs(pos, steps=steps, cfg=cfg, seed=base_seed,
                               width=tile, height=tile)
    _apply_negative(adapter, kw, neg)
    kw["strength"] = denoise
    # 已经用 prompt_embeds 的架构（Compel 路线）才需要丢掉文本形式的 prompt，
    # 否则 Anima/SD3 这类非 Compel 架构的提示词会被一起扔掉 —— 它们只有 prompt。
    if "prompt_embeds" in kw:
        kw.pop("prompt", None)
        kw.pop("negative_prompt", None)
    # 逐块尺寸要不要传，取决于管线认不认：
    #   * SD1.5/SDXL 的 img2img `__call__` 没有 width/height（尺寸跟着 image 走），
    #     硬传会 TypeError；
    #   * Anima 的 ModularPipeline 恰恰相反，不传就按默认 1024 出图，等于每块
    #     都在跑整图（实测单块 28~30s、峰值 9679MB）。
    pass_size = (getattr(model, "arch_id", "") == "anima"
                 or tiled_hires._accepts_kw(pipe, "width", "height"))
    if not pass_size:
        kw.pop("width", None)
        kw.pop("height", None)
    # 强度上限 + 步数补足（与主界面分块高清同一套策略）
    requested_denoise = denoise
    if 0.0 < tiled_hires.TILED_MAX_DENOISE < denoise:
        denoise = tiled_hires.TILED_MAX_DENOISE
        kw["strength"] = denoise
        logger.warning(
            f"⚠️ 分块重绘强度 {requested_denoise:g} 超过上限 {denoise:g}，"
            f"已按 {denoise:g} 执行 —— 分块时每块只看到自己那部分画面，"
            f"强度太大会各画各的（实测 0.35 就出现主体重复与可见块边界）")
    steps_used = tiled_hires.boost_steps(
        int(kw.get("num_inference_steps", steps) or steps), denoise)
    if steps_used != int(kw.get("num_inference_steps", steps) or steps):
        logger.info(f"ℹ️ 分块高清：{steps} 步 × 强度 {denoise:g} 生效步数不足 "
                    f"{tiled_hires.MIN_EFFECTIVE_STEPS}，总步数提到 "
                    f"{steps_used}（强度不变）—— 否则每块会被整块重画")
        kw["num_inference_steps"] = steps_used

    items = []
    plan = []
    no_tile = (target_w <= tile and target_h <= tile)

    if no_tile:
        # ---- 3a) 放大后还没超过一块：整图做一次 img2img，不必分块 ----
        logger.info(f"ℹ️ 放大后 {target_w}x{target_h} 未超过块尺寸 {tile}，"
                    f"按整图一次图生图处理（不分块）")
        ctx.status(f"✨ 整图高清修复 {steps} 步 · {target_w}x{target_h} · "
                   f"强度 {denoise} · seed {base_seed}")
        call = dict(kw)
        call["image"] = upscaled
        if pass_size:
            call["width"] = target_w
            call["height"] = target_h
        call["generator"] = _generator(ctx.device, base_seed)
        ctx.check_cancel()
        with torch.inference_mode():
            res = pipe(**call)
        ctx.check_cancel()
        out = _pick_image(res)
        if out is None:
            raise RuntimeError("img2img 管线没有返回可用图像")
        items.append(({"x": 0, "y": 0, "w": target_w, "h": target_h}, out))
        ctx.progress(1, 1)
    else:
        # ---- 3b) 分块 ----
        plan = tiling.plan_tiles(target_w, target_h, tile, overlap)
        n = len(plan)
        if max_tiles and n > max_tiles:
            raise ValueError(
                f"预计需要 {n} 块，超过「最大块数」限制 {max_tiles}。"
                f"请增大 tile_size、降低放大倍率，或把 max_tiles 设为 0 取消限制"
                f"（尺寸 {target_w}x{target_h}，块 {tile}，重叠 {overlap}）")

        if not tiling.verify_coverage(target_w, target_h, plan):
            raise RuntimeError(f"分块方案未覆盖全部像素（{target_w}x{target_h}，"
                               f"块 {tile}，重叠 {overlap}），已中止以避免出残图")

        logger.info(f"🧩 分块高清修复：{tiling.describe_plan(plan)} · "
                    f"强度 {denoise} · {steps} 步 · "
                    f"{method} 放大 · {blend_label}融合 · "
                    f"色彩对齐{'开' if color_correct else '关'} · 显存{vram_mode}")
        logger.info(f"🎲 本轮基础种子 {base_seed}"
                    f"（{'用户指定' if seed_param >= 0 else '随机生成'}，"
                    f"第 i 块用 {base_seed}+i，可复现）")

        slicing_done = False
        for i, t in enumerate(plan, start=1):
            ctx.check_cancel()
            ctx.status(f"分块高清 {i}/{n} …（{t['w']}x{t['h']} @ {t['x']},{t['y']}）")

            # 省显存：进入前尝试打开 attention slicing（只做一次）
            if vram_mode == _VRAM_SAVER and not slicing_done:
                _open_attention_slicing(pipe)
                slicing_done = True

            crop = upscaled.crop((t["x"], t["y"],
                                  t["x"] + t["w"], t["y"] + t["h"]))
            tile_seed = base_seed + i
            call = dict(kw)
            call["image"] = crop
            if pass_size:
                call["width"] = t["w"]
                call["height"] = t["h"]
            call["generator"] = _generator(ctx.device, tile_seed)

            try:
                with torch.inference_mode():
                    res = pipe(**call)
            finally:
                # 块之间释放激活值：真正的省显存手段（比 attention slicing 有效）
                if vram_mode == _VRAM_SAVER:
                    _free_vram(f"块 {i}/{n} 后回收")
                elif vram_mode == _VRAM_BALANCED and (i % _BALANCED_EVERY == 0):
                    _free_vram(f"平衡模式每 {_BALANCED_EVERY} 块回收")

            out = _pick_image(res)
            if out is None:
                raise RuntimeError(f"第 {i}/{n} 块 img2img 没有返回图像")
            if out.size != (t["w"], t["h"]):
                # 管线可能按 8 的倍数裁过尺寸，先对齐再融合，否则拼图会错位
                logger.debug(f"块 {i} 返回尺寸 {out.size} != {t['w']}x{t['h']}，"
                             f"已重采样对齐")
                out = _align_to(out, (t["w"], t["h"]))
            items.append((t, out))
            ctx.progress(i, n)

        if vram_mode == _VRAM_SAVER:
            _free_vram("分块结束后回收")

    # ---- 4) 融合 ----
    ctx.status(f"🧵 融合 {len(items)} 块 → {target_w}x{target_h}")
    # 羽化宽度按**实际**重叠算（plan_tiles 会给比请求值更大的重叠，
    # 按请求值的一半羽化会在平色区留下明显色阶）+ 整图分支就按图的一半
    if no_tile:
        feath = max(8, min(target_w, target_h) // 2)
    else:
        _ov = tiling.effective_overlap(plan)
        feath = max(8, min(int(_ov[0]), int(_ov[1])) // 2)
    out_img = tiling.blend_stitch((target_w, target_h), items,
                                  base=upscaled, feather=feath,
                                  mode=mode, color_correct=color_correct)
    out_img = out_img.convert("RGB")
    ctx.check_cancel()

    meta = {"tile_size": tile, "tile_overlap": overlap,
            "tile_overlap_effective": tiling.effective_overlap(plan),
            "tile_denoise": denoise,
            "tile_denoise_requested": requested_denoise,
            "tile_steps": int(kw.get("num_inference_steps", steps) or steps),
            "tile_feather": feath,
            "tile_pass_size": bool(pass_size),
            "tile_scale": scale,
            "tile_count": len(items), "tile_total": len(items),
            "tile_method": method, "tile_blend": blend_label,
            "tile_vram_mode": vram_mode, "tile_color_correct": color_correct,
            "tile_seed": base_seed,
            "tiled_hires": True, "tile_no_tiling": bool(no_tile),
            "hires_size": f"{target_w}x{target_h}"}
    ctx.metadata.update(meta)
    logger.info(f"✅ 分块高清完成：{tiling.describe_plan([t for t, _ in items])} · "
                f"强度 {denoise}")
    return {"image": out_img, "metadata": meta}


def _pick_image(res):
    """从管线返回值里取第一张图（转成 PIL RGB）；取不到返回 None。"""
    imgs = getattr(res, "images", None)
    if imgs is None:
        return None
    out = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
    pil = as_image(out)
    if pil is None:
        return None
    return pil.convert("RGB")


# ----------------------------------------------------------------------
@node(
    "TiledUpscale",
    category="放大",
    description="分块纯缩放（不跑扩散、不吃显存）：按块逐步缩放再拼回，"
                "避免一次性分配整张巨图的内存，适合把 4K 图在 8GB 机器上放大。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"image": PortSpec(DataType.IMAGE, "图像"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "scale": ParamSpec("float", default=2.0, label="放大倍率",
                           minimum=1.0, maximum=8.0, step=0.5,
                           help="目标尺寸 = 原尺寸 × 倍率（向下对齐到 8 的倍数）"),
        "upscale_method": ParamSpec("choice", default="Lanczos", label="放大方式",
                                    choices=("Lanczos", "Bicubic", "Nearest"),
                                    help="插值方法；Nearest 只适合像素画"),
        "tile_size": ParamSpec("int", default=640, label="分块尺寸",
                               minimum=256, maximum=1536, step=64,
                               help="每块边长。分块缩放的意义在于："
                                    "峰值内存只与块尺寸有关，而不是整张目标图"),
        "overlap": ParamSpec("int", default=48, label="重叠像素",
                             minimum=0, maximum=256, step=8,
                             help="相邻块的重叠宽度，用于消除接缝"),
        "blend": ParamSpec("choice", default="羽化", label="边缘融合",
                           choices=("羽化", "线性", "硬切"),
                           help="羽化=平滑过渡（推荐，缩放后各块完全一致，"
                                "任何模式结果都相同）"),
        "max_tiles": ParamSpec("int", default=0, label="最大块数",
                               minimum=0, maximum=4096,
                               help="安全阀：预计块数超过它就报错。0 = 不限制"),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0),
)
def run_tiled_upscale(ctx, node, inputs):
    """分块纯缩放：适合超大图 / 低内存机器。

    为什么要这个节点（而不是直接用 `Upscale`）
    ------------------------------------------
    `Upscale` 一次性 `resize` 到目标尺寸：8K 目标图仅 RGB 缓冲就是
    8000x8000x3 ≈ 183MB，PIL 重采样过程中的中间缓冲还要翻几倍，
    8GB 笔记本上很容易把内存打满（进而触发换页，把整机拖住）。
    这里按块处理：**任一时刻只多留一块的中间结果**，不再出现
    "源图 + 目标图 + 中间缓冲"同时驻留的峰值。

    核心技巧：**先整图缩放一次，再按块裁出来拼**，而不是"每块各自缩放"。
    原因有两个：
    · 质量：每块独立缩放时，块边界处的插值取样窗口不跨块，接缝处会有
      半像素级的错位，放大倍率高时看得出来；整图缩放 + 裁剪则天然一致；
    · 省事且不易错：块坐标直接就是目标坐标系里的坐标，不用来回换算。
    真正的内存收益来自"D 阶段不再物化整张目标图的 float 缓冲"：
    `blend_stitch` 只有在需要时才去碰底图（见 utils/tiling.py 里的说明）。
    """
    from utils import tiling

    image = as_image(inputs.get("image"))
    if image is None:
        raise ValueError("缺少输入 image")

    scale = max(1.0, _parse_scale(node.params.get("scale", 2.0)))
    method = str(node.params.get("upscale_method", "Lanczos") or "Lanczos")
    tile_param = int(node.params.get("tile_size", 640))
    overlap = int(node.params.get("overlap", 48))
    blend_label = str(node.params.get("blend", "羽化") or "羽化")
    mode = _BLEND_MODE.get(blend_label.strip(), "feather")
    max_tiles = max(0, int(node.params.get("max_tiles", 0) or 0))

    src = image.convert("RGB")
    raw_w = max(8, int(round(src.width * scale)))
    raw_h = max(8, int(round(src.height * scale)))
    target_w, target_h = tiling.fit_size_to_multiple(raw_w, raw_h,
                                                     _SIZE_MULTIPLE)

    ctx.status(f"🔍 分块放大 {scale}× · {src.width}x{src.height} → "
               f"{target_w}x{target_h} · {method}")

    # 一次性缩放到目标尺寸（质量与整图缩放完全一致）
    big = _resize(src, (target_w, target_h), method)
    ctx.check_cancel()

    tile = tiling.adaptive_tile_size((target_w, target_h), tile_param)
    # 小图 / 小倍率：分块没有意义，直接返回
    if target_w <= tile and target_h <= tile:
        meta = {"tile_size": tile, "tile_overlap": int(overlap),
                "tile_scale": scale, "tile_count": 1, "tile_total": 1,
                "tile_method": method, "tile_blend": blend_label,
                "tile_no_tiling": True, "tile_upscale_only": True,
                "hires_size": f"{target_w}x{target_h}"}
        ctx.metadata.update(meta)
        ctx.progress(1, 1)
        return {"image": big.convert("RGB"), "metadata": meta}

    overlap = tiling.clamp_overlap(overlap, tile)
    plan = tiling.plan_tiles(target_w, target_h, tile, overlap)
    n = len(plan)
    if max_tiles and n > max_tiles:
        raise ValueError(f"预计需要 {n} 块，超过「最大块数」限制 {max_tiles}"
                         f"（目标 {target_w}x{target_h}，块 {tile}）")
    if not tiling.verify_coverage(target_w, target_h, plan):
        raise RuntimeError("分块方案未覆盖全部像素，已中止")

    logger.info(f"🧩 分块缩放：{tiling.describe_plan(plan)} · {method} · "
                f"{blend_label}融合（不跑扩散）")

    items = []
    for i, t in enumerate(plan, start=1):
        ctx.check_cancel()
        ctx.status(f"分块缩放 {i}/{n} …")
        crop = big.crop((t["x"], t["y"],
                         t["x"] + t["w"], t["y"] + t["h"]))
        items.append((t, crop))
        ctx.progress(i, n)
    ctx.check_cancel()

    # base=big：各块本来就来自同一张图，色彩对齐对结果没有影响（保持参数一致）
    out_img = tiling.blend_stitch((target_w, target_h), items, base=big,
                                  feather=max(8, overlap // 2), mode=mode)
    out_img = out_img.convert("RGB")

    meta = {"tile_size": tile, "tile_overlap": overlap,
            "tile_overlap_effective": tiling.effective_overlap(plan), "tile_scale": scale,
            "tile_count": n, "tile_total": n, "tile_method": method,
            "tile_blend": blend_label, "tile_upscale_only": True,
            "hires_size": f"{target_w}x{target_h}"}
    ctx.metadata.update(meta)
    return {"image": out_img, "metadata": meta}
