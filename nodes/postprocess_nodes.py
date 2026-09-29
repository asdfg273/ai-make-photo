# nodes/postprocess_nodes.py
"""后处理节点：ImageNoise / ImageBlur / ImageSharpen / ColorAdjust / ImagePosterize。

本模块的定位
------------
这是"出图之后的最后一公里"：噪点、模糊、锐化、色彩、色调分离。它们都是
**纯 CPU 的 PIL 像素级操作**，不加载任何模型、不碰 torch/diffusers、
不吃显存 —— 所以统一声明 `ResourceRequirement(requires=())`、
`estimated_vram_mb=0`、`supports_tiled=False`。执行器的显存策略看到这些声明
后会直接把它们排到 CPU 阶段，不会为了加个噪点去卸载/重载底模。

节点约定（所有节点模块统一遵守）
--------------------------------
· `run(ctx, node, inputs)` 返回 **dict**，键名必须与 NodeSpec.outputs 一致；
· 需要重依赖时在函数**内部** import（PIL/numpy 是本项目的硬依赖，
  与 `nodes/image_nodes.py` 保持一致：模块顶层不 import，函数内 import，
  这样 `import nodes` 不会拖慢界面启动）；
· 出错就抛异常（本模块的错误信息一律用中文，能直接告诉用户哪里填错了），
  执行器会包成带 node_id 的 NodeError 并定位到本节点；
· 只通过 `ctx` 访问运行期信息，不直接操作界面控件。

一条硬规则：**绝不原地修改调用方传入的图像**
--------------------------------------------
上游节点返回的 PIL 图像可能被缓存复用（执行器的 NodeCache）或被多个下游
同时消费，`ImageDraw`/`putdata`/numpy 视图回写这类"原地改"会污染缓存、
让"只重跑此节点"的结果对不上。所以本模块每个节点都先 `copy()` 再改，
保证输入图在调用前后逐字节一致。
"""
from __future__ import annotations

import logging
import random

from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import as_image

logger = logging.getLogger(__name__)

# 模糊类型（"径向"是缩放模糊，见 _radial_blur 的说明）
_BLUR_KINDS = ("高斯", "方框", "径向")
# 噪点类型
_NOISE_KINDS = ("高斯", "均匀", "盐椒")
# 色调分离的三种模式
_POSTERIZE_MODES = ("色调分离", "阈值", "反相")

# 盐椒噪点里"盐"（白点）的占比，剩下的是"椒"（黑点）
_SALT_RATIO = 0.5

# 径向（缩放）模糊：低配实现的强度常数与预览边长上限。
# 完整版会按半径做几十层叠加，这里刻意做成"层数 / 缩放步进由半径推导、
# 且先在最长边不超过该值的缩略图上算"，保证 100 半径也是毫秒级而不是几秒。
_RADIAL_STEPS = 16          # 叠加层数（层数越多越平滑，越慢）
_RADIAL_ZOOM_AT_MAX = 0.25  # 半径 100 时最外层放大到 1.25 倍
_RADIAL_MAX_SIDE = 256      # 计算径向模糊时先缩到这个边长以内


# ----------------------------------------------------------------------
#  小工具
# ----------------------------------------------------------------------
def _require_pil(value, name: str = "image"):
    """把输入转成 PIL 图像，转不了就抛**说得清原因**的中文错误。

    注意必须接收**原始输入**（而不是已经过 as_image 的结果）：as_image 对
    "不是图像"的值一律返回 None，只判 None 会把"上游接了个数字/字符串"
    这类错误吞成"缺少输入"，用户完全看不出问题在哪。所以这里先 as_image，
    失败时再回头看原始值是什么类型，据此分别报错。
    """
    img = as_image(value)
    if img is not None:
        return img
    if value is None:
        raise ValueError(f"缺少输入 {name}")
    raise TypeError(
        f"输入不是 PIL 图像（{name} 实际类型 {type(value).__name__}），"
        f"请接入 IMAGE 端口的图像")


def _sharpenable(image):
    """色相/锐化类操作前统一处理模式问题。

    调色板图（P）、单通道图（L）、带透明通道的图（RGBA）在
    ImageEnhance.Sharpness 上有两条坑：P 模式强转 RGB 会变成乱色，
    RGBA 模式会直接抛 "image has wrong mode"。所以：
      · P   -> 先 convert("RGB") 把调色板展开（这是唯一正确的展开时机）；
      · RGBA/LA -> 拆出 alpha，只对 RGB 部分做增强，最后再贴回去；
      · 其余（RGB/L/CMYK...）-> 交给 PIL 原样处理，不做无谓转换。
    """
    if image.mode == "P":
        image = image.convert("RGB")
    if image.mode in ("RGBA", "LA"):
        base = image.convert("RGBA")
        alpha = base.getchannel("A")
        return base.convert("RGB"), alpha
    return image, None


def _merge_alpha(image, alpha, size):
    """把增强后的 RGB 与原来的 alpha 合回 RGBA；alpha 为 None 时原样返回。"""
    if alpha is None:
        return image
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    if rgb.size != size:
        rgb = rgb.resize(size)
    if alpha.size != size:
        alpha = alpha.resize(size)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def _append_alpha(rgb, alpha):
    """给模糊后的 RGB 补一份 alpha（模糊要连 alpha 一起模糊才对，见调用处）。"""
    base = rgb if rgb.mode == "RGB" else rgb.convert("RGB")
    out = base.convert("RGBA")
    out.putalpha(alpha)
    return out


def _resolve_seed(value) -> int:
    """把 seed 参数规范成非负整数：-1（或非法值）表示随机。

    与 `Sampler` / `Inpaint` 的种子语义保持一致（-1 = 随机），但这里
    不依赖 sampling_nodes —— 后处理节点不该因为"采样模块 import 了 torch"
    而把 torch 拖进来。
    """
    try:
        seed = int(value)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        seed = random.randrange(0, 2 ** 31)
    return seed


# ----------------------------------------------------------------------
#  ImageNoise
# ----------------------------------------------------------------------
def _gaussian_noise(arr, rng, amount, monochrome, h, w):
    """高斯噪点：均值 0、标准差 amount*255 的正态分布。"""
    shape = (h, w, 1) if monochrome else (h, w, arr.shape[2])
    noise = rng.standard_normal(shape) * (amount * 255.0)
    return arr + noise


def _uniform_noise(arr, rng, amount, monochrome, h, w):
    """均匀噪点：在 ±amount*255 之间均匀取值。"""
    shape = (h, w, 1) if monochrome else (h, w, arr.shape[2])
    noise = (rng.random(shape) * 2.0 - 1.0) * (amount * 255.0)
    return arr + noise


def _salt_pepper(arr, rng, amount, monochrome, h, w):
    """盐椒噪点：按 amount 比例随机把像素打成纯黑/纯白。

    单色模式下"盐/椒"两个掩码对三个通道是同一张，所以打出来仍是灰阶
    （黑点/白点/中性灰），这正是单色噪点该有的样子。
    """
    shape = (h, w, 1) if monochrome else (h, w, arr.shape[2])
    pick = rng.random(shape)
    salt = (pick >= 1.0 - amount * _SALT_RATIO) & (pick < 1.0)
    pepper = pick < amount * _SALT_RATIO
    out = arr.copy()
    out[salt] = 255.0
    out[pepper] = 0.0
    return out


def _apply_noise_numpy(image, kind: str, amount: float, seed: int,
                       monochrome: bool):
    """用 numpy 施加噪点（快路径）：返回新图像，不改动入参。"""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    src = image
    if src.mode == "P":
        # 调色板图不能直接走 fromarray：numpy 里它是"索引值"而不是颜色，
        # 加噪后索引会串到别的调色板项，出来一片乱色。先展开成真彩。
        src = src.convert("RGB")
    arr = np.asarray(src, dtype=np.float32)
    # 单通道图（L / P 展开后只有 2 维）需要补一维才能按 (h, w, c) 处理
    squeezed = False
    if arr.ndim == 2:
        arr = arr[:, :, None]
        squeezed = True
    h, w = arr.shape[0], arr.shape[1]

    fn = {"高斯": _gaussian_noise, "均匀": _uniform_noise,
          "盐椒": _salt_pepper}[kind]
    out = fn(arr, rng, amount, monochrome, h, w)
    out = np.clip(out, 0.0, 255.0).round().astype(np.uint8)

    if squeezed:
        # 单通道：numpy 侧丢掉多余通道即可，"单色/彩色"对灰度图无差别
        return Image.fromarray(out[:, :, 0], src.mode)
    if out.shape[2] == 1:
        out = np.repeat(out, arr.shape[2], axis=2)
    return Image.fromarray(out, src.mode)


def _apply_noise_pil(image, kind: str, amount: float, seed: int,
                     monochrome: bool):
    """纯 PIL 的降级实现（numpy 不可用时使用）。

    用 `Image.effect_noise` 生成均值 128 的均匀噪声当随机源（它是确定性的，
    与 seed 无关，所以这条路径**不可复现** —— 只有 numpy 在时才有种子语义，
    这里必须把话说清楚）。三种噪点的差异：
      · 高斯 / 均匀 -> 都退化成均匀噪声，只是幅度不同（如实降级，不假装）；
      · 盐椒       -> 用同一份噪声做阈值切分，打纯黑/纯白点。
    """
    from PIL import Image

    logger.warning("⚠️ numpy 不可用，ImageNoise 降级为纯 PIL 实现："
                   "噪点分布被近似，且 seed 不再生效（无法复现）")
    src = image.convert("RGB")
    w, h = src.size
    noise = Image.effect_noise((w, h), 128.0).convert("L")
    amount = max(0.0, min(1.0, amount))

    if kind == "盐椒":
        hi = int(255 - amount * _SALT_RATIO * 255)     # 高于它 -> 白点
        lo = int(amount * _SALT_RATIO * 255)           # 低于它 -> 黑点
        mask = noise.point(lambda v: 255 if (v >= hi or v <= lo) else 0)
        return Image.composite(Image.new("RGB", (w, h), (255, 255, 255)),
                               src, mask)

    # 高斯与均匀都退化成"加均匀噪声"：effect_noise 的均值是 128，
    # 减掉 128 得到零均值噪声后按 amount 缩放叠加到原图上。
    rgb_noise = noise.convert("RGB")
    if monochrome:
        rgb_noise = Image.merge("RGB", (noise, noise, noise))
    return _add_uniform_noise(src, rgb_noise, amount)


def _add_uniform_noise(src, noise, amount: float):
    """把零均值噪声按 amount 叠加到原图（纯 PIL 路径专用）。

    `effect_noise` 的均值是 128、范围大致 ±128，所以用 offset=-128*amount
    抵消直流分量，再用 scale=amount 控制幅度。
    """
    from PIL import ImageChops

    return ImageChops.add(src, noise, scale=amount, offset=-128 * amount)


# ----------------------------------------------------------------------
@node(
    "ImageNoise",
    category="后处理",
    description="给图片加噪：高斯 / 均匀 / 盐椒三种噪点，可用种子复现、"
                "可只加单色噪点（不引入彩色杂点）。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"image": PortSpec(DataType.IMAGE, "图像")},
    params={
        "amount": ParamSpec("float", default=0.05, label="噪点强度",
                            minimum=0.0, maximum=1.0, step=0.01,
                            help="噪点幅度占满量程的比例：0=不加噪，"
                                 "0.05 约 ±13 级；盐椒模式下同时是噪点比例"),
        "type": ParamSpec("choice", default="高斯", label="类型",
                          choices=_NOISE_KINDS,
                          help="高斯=正态分布噪点（最自然）；均匀=等概率噪点；"
                               "盐椒=随机黑白点"),
        "seed": ParamSpec("seed", default=-1, label="种子",
                          help="-1 = 随机；固定种子可复现同一份噪点图"),
        "monochrome": ParamSpec("bool", default=False, label="单色噪点",
                                help="开启后三个通道用同一份噪点（灰阶噪点，"
                                     "不会出现彩色杂点）"),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0,
                                  supports_tiled=False),
)
def run_image_noise(ctx, node, inputs):
    image = _require_pil(inputs.get("image"))

    amount = float(node.params.get("amount", 0.05) or 0.0)
    amount = max(0.0, min(1.0, amount))
    kind = str(node.params.get("type", "高斯") or "高斯").strip()
    if kind not in _NOISE_KINDS:
        raise ValueError(f"未知的噪点类型: {kind}")
    monochrome = bool(node.params.get("monochrome", False))
    if amount <= 0.0:
        # 强度为 0 就是恒等变换：直接给一份拷贝，保证"不原地改"且逐字节相同
        ctx.status("🔊 噪点强度为 0，原样返回")
        ctx.metadata.update({"noise_type": kind, "noise_amount": 0.0,
                             "noise_seed": None})
        return {"image": image.copy()}

    seed = _resolve_seed(node.params.get("seed", -1))
    ctx.status(f"🔊 加噪 {kind} · 强度 {amount:g} · seed {seed}"
               + ("（单色）" if monochrome else ""))
    ctx.check_cancel()

    try:
        out = _apply_noise_numpy(image, kind, amount, seed, monochrome)
    except ImportError as e:                    # numpy 缺失 -> 纯 PIL 兜底
        logger.warning(f"⚠️ numpy 不可用（{e}），ImageNoise 走 PIL 降级路径")
        out = _apply_noise_pil(image, kind, amount, seed, monochrome)

    ctx.check_cancel()
    ctx.metadata.update({"noise_type": kind, "noise_amount": amount,
                         "noise_seed": seed, "noise_monochrome": monochrome})
    return {"image": out}


# ----------------------------------------------------------------------
#  ImageBlur
# ----------------------------------------------------------------------
def _radial_blur(image, radius: float):
    """径向（缩放/变焦）模糊：中心清晰、四周向外拉丝。

    实现是**多帧缩放叠加**（低配版缩放模糊）：把图按 1 → 1+zoom 逐层放大，
    每层按"与中心帧的重合度"加权（所以中心几乎不糊、边缘糊得最狠），
    加权平均后还原尺寸。

    三点如实说明：
    · 是**缩放模糊的近似**，没有真正的角度卷积、没有螺旋效果，
      观感与 PS"径向模糊·缩放"接近但不完全相同；
    · zoom = 半径/400（上限 0.25），层数固定 16 层；
    · 为控制耗时，先在最长边不超过 256 的缩略图上计算再插值回原尺寸 ——
      这是**画质换速度**的折中，半径 100 也是毫秒级而不是几秒。
    """
    from PIL import Image

    if radius <= 0:
        return image.copy()

    zoom = min(_RADIAL_ZOOM_AT_MAX, radius / 400.0)
    if zoom <= 0:
        return image.copy()

    src = image
    work = src
    if max(work.size) > _RADIAL_MAX_SIDE:
        k = _RADIAL_MAX_SIDE / float(max(work.size))
        work = work.resize((max(1, int(round(work.width * k))),
                            max(1, int(round(work.height * k)))),
                           Image.BILINEAR)
    base = work if work.mode == "RGB" else work.convert("RGB")
    w, h = base.size

    def _weights():
        out = []
        for i in range(_RADIAL_STEPS):
            t = i / float(_RADIAL_STEPS - 1)      # 0 = 中心，1 = 最外
            same = 1.0 - zoom * t                 # 与中心帧的重合度 ≈ 清晰度
            out.append((t, same * same))          # 平方让中心权重更突出
        return out

    layers = _weights()
    total = sum(wt for _, wt in layers) or 1.0

    try:
        import numpy as np

        acc = np.zeros((h, w, 3), dtype=np.float32)
        for t, weight in layers:
            acc += np.asarray(_zoom_layer(base, zoom * t),
                              dtype=np.float32) * (weight / total)
        out = Image.fromarray(
            np.clip(acc, 0.0, 255.0).round().astype("uint8"), "RGB")
    except ImportError:
        # 没有 numpy 时用 alpha 加权合成：把每层当半透明覆盖层从底色叠上去
        acc = Image.new("RGBA", (w, h), (0, 0, 0, 255))
        for t, weight in layers:
            rgba = _zoom_layer(base, zoom * t).convert("RGBA")
            alpha = rgba.getchannel("A").point(
                lambda v, a=weight / total: int(v * a))
            rgba.putalpha(alpha)
            acc = Image.alpha_composite(acc, rgba)
        out = acc.convert("RGB")

    if out.size != src.size:
        out = out.resize(src.size, Image.BICUBIC)
    return out


def _zoom_layer(base, factor: float):
    """把 `base` 放大 1+factor 倍后按**中心**裁回原尺寸（缩放模糊的一层）。

    必须从中心裁：径向模糊的定义就是所有像素沿"从画面中心向外的方向"
    被拉伸，从左上角裁会变成平移模糊，方向完全不对。
    """
    from PIL import Image

    w, h = base.size
    big = base.resize((max(1, int(round(w * (1.0 + factor)))),
                       max(1, int(round(h * (1.0 + factor))))),
                      Image.BICUBIC)
    left = (big.width - w) // 2
    top = (big.height - h) // 2
    return big.crop((left, top, left + w, top + h))


@node(
    "ImageBlur",
    category="后处理",
    description="模糊/柔化：高斯模糊、方框模糊，以及径向（缩放）模糊。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"image": PortSpec(DataType.IMAGE, "图像")},
    params={
        "radius": ParamSpec("float", default=2.0, label="半径",
                            minimum=0.0, maximum=100.0, step=0.5,
                            help="模糊半径（像素）：0 = 不模糊；"
                                 "径向模式下半径还决定四周拉丝的强度"),
        "kind": ParamSpec("choice", default="高斯", label="类型",
                          choices=_BLUR_KINDS,
                          help="高斯=ImageFilter.GaussianBlur（最柔和）；"
                               "方框=ImageFilter.BoxBlur（更快、偏方）；"
                               "径向=多帧缩放叠加的缩放模糊（中心清晰、"
                               "四周向外拉丝，是低配近似实现）"),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0,
                                  supports_tiled=False),
)
def run_image_blur(ctx, node, inputs):
    from PIL import ImageFilter

    image = _require_pil(inputs.get("image"))

    radius = float(node.params.get("radius", 2.0) or 0.0)
    radius = max(0.0, min(100.0, radius))
    kind = str(node.params.get("kind", "高斯") or "高斯").strip()
    if kind not in _BLUR_KINDS:
        raise ValueError(f"未知的模糊类型: {kind}")

    if radius <= 0.0:
        ctx.status("🌫️ 半径为 0，原样返回")
        ctx.metadata.update({"blur_kind": kind, "blur_radius": 0.0})
        return {"image": image.copy()}

    ctx.status(f"🌫️ 模糊 {kind} · 半径 {radius:g}")
    ctx.check_cancel()

    # 带透明通道的图要连 alpha 一起模糊，否则边缘会出现半透明脏边
    alpha = None
    work = image
    if image.mode in ("RGBA", "LA"):
        work = image.convert("RGBA")
        alpha = work.getchannel("A")
        work = work.convert("RGB")

    if kind == "高斯":
        out = work.filter(ImageFilter.GaussianBlur(radius=radius))
    elif kind == "方框":
        out = work.filter(ImageFilter.BoxBlur(radius=radius))
    else:
        out = _radial_blur(work, radius)

    if alpha is not None:
        blurred_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=radius))
        out = _append_alpha(out, blurred_alpha)

    ctx.check_cancel()
    ctx.metadata.update({"blur_kind": kind, "blur_radius": radius})
    return {"image": out}


# ----------------------------------------------------------------------
#  ImageSharpen
# ----------------------------------------------------------------------
@node(
    "ImageSharpen",
    category="后处理",
    description="锐化：USM（UnsharpMask）增强边缘细节，可分别控制半径、"
                "百分比与阈值。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"image": PortSpec(DataType.IMAGE, "图像")},
    params={
        "amount": ParamSpec("float", default=1.0, label="强度",
                            minimum=0.0, maximum=10.0, step=0.1,
                            help="整体强度倍率，乘以锐化百分比：1.0 = 用原始百分比，"
                                 "0 = 不锐化"),
        "radius": ParamSpec("float", default=1.0, label="半径",
                            minimum=0.1, maximum=10.0, step=0.1,
                            help="参与锐化的边缘宽度（像素），越小越锐利越容易出白边"),
        "percent": ParamSpec("int", default=150, label="锐化百分比",
                             minimum=0, maximum=500,
                             help="边缘对比度增强幅度（USM percent），"
                                  "150 是常用值，超过 300 容易过冲"),
        "threshold": ParamSpec("int", default=3, label="阈值",
                               minimum=0, maximum=255,
                               help="相邻像素差小于该值就不锐化，"
                                    "用它避免把噪点和渐变也一起锐化出来"),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0,
                                  supports_tiled=False),
)
def run_image_sharpen(ctx, node, inputs):
    from PIL import ImageFilter

    image = _require_pil(inputs.get("image"))

    amount = float(node.params.get("amount", 1.0) or 0.0)
    amount = max(0.0, min(10.0, amount))
    radius = float(node.params.get("radius", 1.0) or 1.0)
    radius = max(0.1, min(10.0, radius))
    percent = int(node.params.get("percent", 150) or 0)
    percent = max(0, min(500, percent))
    threshold = int(node.params.get("threshold", 3) or 0)
    threshold = max(0, min(255, threshold))

    # amount 单独为 0（或 percent 为 0）等价于不锐化，避免无谓的全图卷积
    effective = int(round(percent * amount))
    if effective <= 0:
        ctx.status("🔪 锐化强度为 0，原样返回")
        ctx.metadata.update({"sharpen_amount": amount, "sharpen_percent": 0,
                             "sharpen_radius": radius,
                             "sharpen_threshold": threshold})
        return {"image": image.copy()}

    ctx.status(f"🔪 锐化 半径 {radius:g} · 百分比 {effective} · 阈值 {threshold}")
    ctx.check_cancel()

    work, alpha = _sharpenable(image)
    out = work.filter(ImageFilter.UnsharpMask(
        radius=radius, percent=effective, threshold=threshold))
    out = _merge_alpha(out, alpha, image.size)

    ctx.check_cancel()
    ctx.metadata.update({"sharpen_amount": amount,
                         "sharpen_percent": effective,
                         "sharpen_radius": radius,
                         "sharpen_threshold": threshold})
    return {"image": out}


# ----------------------------------------------------------------------
#  ColorAdjust
# ----------------------------------------------------------------------
@node(
    "ColorAdjust",
    category="后处理",
    description="色彩调整：亮度 / 对比度 / 饱和度 / 锐度四项一次性调节"
                "（等价于 PS 的基础调整层）。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"image": PortSpec(DataType.IMAGE, "图像")},
    params={
        "brightness": ParamSpec("float", default=1.0, label="亮度",
                                minimum=0.0, maximum=3.0, step=0.05,
                                help="1.0 = 不变；<1 变暗，>1 变亮，0 = 全黑"),
        "contrast": ParamSpec("float", default=1.0, label="对比度",
                              minimum=0.0, maximum=3.0, step=0.05,
                              help="1.0 = 不变；<1 更灰，>1 更硬，0 = 中灰"),
        "saturation": ParamSpec("float", default=1.0, label="饱和度",
                                minimum=0.0, maximum=3.0, step=0.05,
                                help="1.0 = 不变；0 = 完全去色（灰度），"
                                     ">1 更艳（可能溢出）"),
        "sharpness": ParamSpec("float", default=1.0, label="锐度",
                               minimum=0.0, maximum=3.0, step=0.05,
                               help="1.0 = 不变；0 = 完全模糊，>1 更锐"),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0,
                                  supports_tiled=False),
)
def run_color_adjust(ctx, node, inputs):
    from PIL import ImageEnhance

    image = _require_pil(inputs.get("image"))

    def _p(name: str, default: float) -> float:
        try:
            v = float(node.params.get(name, default))
        except (TypeError, ValueError):
            v = default
        return max(0.0, min(3.0, v))

    brightness = _p("brightness", 1.0)
    contrast = _p("contrast", 1.0)
    saturation = _p("saturation", 1.0)
    sharpness = _p("sharpness", 1.0)

    # 模式守卫：P 模式先展开成 RGB，RGBA/LA 拆出 alpha 单独保留
    work, alpha = _sharpenable(image)
    if work.mode not in ("RGB", "L"):
        # 其余冷门模式（CMYK/YCbCr/I;16 等）统一转 RGB：色彩增强在
        # 这些模式上行为不一致，转成 RGB 后结果才是可预期的
        work = work.convert("RGB")

    ctx.status(f"🎨 调色 亮度 {brightness:g} · 对比度 {contrast:g} · "
               f"饱和度 {saturation:g} · 锐度 {sharpness:g}")
    ctx.check_cancel()

    # 顺序：亮度 -> 对比度 -> 饱和度 -> 锐度。
    # 这个顺序很重要（先定曝光再定反差，最后上饱和/锐化），
    # 反过来会先把噪点和色偏一起放大。
    out = work
    steps = (("brightness", brightness,
              lambda im, v: ImageEnhance.Brightness(im).enhance(v)),
             ("contrast", contrast,
              lambda im, v: ImageEnhance.Contrast(im).enhance(v)),
             ("saturation", saturation,
              lambda im, v: ImageEnhance.Color(im).enhance(v)),
             ("sharpness", sharpness,
              lambda im, v: ImageEnhance.Sharpness(im).enhance(v)))
    for name, value, apply in steps:
        if value == 1.0:
            continue                     # 1.0 是恒等，跳过全图卷积
        if name == "saturation" and out.mode == "L":
            continue                     # 灰度图没有饱和度可调，如实跳过
        out = apply(out, value)
        ctx.check_cancel()

    out = _merge_alpha(out, alpha, image.size)
    if out is image:
        # 四个参数全是 1.0（或只剩被跳过的项）时上面一步都没做，out 就是
        # 输入图本身。这里必须补一份拷贝：下游可能原地改它，而上游图可能
        # 正躺在执行器缓存里，直接被改会污染缓存。
        out = image.copy()
    ctx.metadata.update({"brightness": brightness, "contrast": contrast,
                         "saturation": saturation, "sharpness": sharpness})
    return {"image": out}


# ----------------------------------------------------------------------
#  ImagePosterize
# ----------------------------------------------------------------------
@node(
    "ImagePosterize",
    category="后处理",
    description="色调分离 / 色阶压缩：减少每个通道的色阶数，或做纯黑白阈值、"
                "整图反相。",
    inputs={"image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"image": PortSpec(DataType.IMAGE, "图像")},
    params={
        "bits": ParamSpec("int", default=4, label="保留位数",
                          minimum=1, maximum=8,
                          help="每个通道保留的二进制位数：1 = 只剩黑白两级，"
                               "8 = 不损失（只对「色调分离」模式生效）"),
        "mode": ParamSpec("choice", default="色调分离", label="模式",
                          choices=_POSTERIZE_MODES,
                          help="色调分离=按位数压缩色阶；"
                               "阈值=灰度后以 128 为界转纯黑白；"
                               "反相=颜色取反（负片）"),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0,
                                  supports_tiled=False),
)
def run_image_posterize(ctx, node, inputs):
    from PIL import ImageOps

    image = _require_pil(inputs.get("image"))

    bits = int(node.params.get("bits", 4) or 4)
    bits = max(1, min(8, bits))
    mode = str(node.params.get("mode", "色调分离") or "色调分离").strip()
    if mode not in _POSTERIZE_MODES:
        raise ValueError(f"未知的色调分离模式: {mode}")

    ctx.status(f"🎚️ 色调分离 {mode}" + (f" · 保留 {bits} 位"
                                        if mode == "色调分离" else ""))
    ctx.check_cancel()

    if mode == "色调分离":
        # posterize 只在 L/RGB 上语义明确，其余模式先转 RGB
        work = image if image.mode in ("L", "RGB") else image.convert("RGB")
        out = ImageOps.posterize(work, bits)
    elif mode == "阈值":
        # 灰度化后按 128 为界二值化，再转回 RGB（工作流里 IMAGE 统一按 RGB 用）
        out = image.convert("L").point(
            lambda v: 255 if v > 128 else 0).convert("RGB")
    else:
        out = ImageOps.invert(image.convert("RGB"))

    ctx.check_cancel()
    ctx.metadata.update({"posterize_mode": mode, "posterize_bits": bits})
    return {"image": out}
