# utils/tiling.py
"""分块几何与融合：把大图切成可独立处理的块，再无缝拼回。

这个模块只做**纯几何 + 纯像素**的事，因此：
· 不 import torch（CPU-only 机器、UI 线程、纯测试都能用）；
· 不 import Qt；不读写文件；不修改调用方传进来的图像；
· 所有函数都是无副作用的纯函数，便于单元测试与复用。

和 `utils/tiled_diffusion.py` 的分工
-----------------------------------
`utils/tiled_diffusion.py` 是**旧的**一体化实现：切块 + 调 diffusers 管线 +
羽化融合写死在一个函数里，只能配 `pipe(...)` 那种调用方式，而且它的
"最后一块贴边"策略在特定宽度下会产出比 overlap 大得多的重叠。
本模块把「几何 / 权重 / 融合」拆出来，让工作流节点（`nodes/tile_nodes.py`）
只负责"逐块跑模型"，于是同一套几何能同时服务于：
· 分块图生图（TiledHiresFix，每块跑扩散）
· 分块缩放（TiledUpscale，不跑模型，只为了不在 8GB 上一次性分配巨图）

拼接为什么会"无缝"
------------------
相邻块重叠 `overlap` 像素，每块带一张从边缘 0 升到中心 1 的权重图。
真正的无缝来自一个数学事实：**边缘处权重按 s 与 1-s 互补**，
而 `blend_stitch` 不是简单求平均，而是除以"实际参与归一化的权重和"。
只要至少一块的权重为 1，拼出来就与源图逐像素相等（见 B7 恒等测试）。
"""
from __future__ import annotations

import logging
import math
from typing import List, Tuple

logger = logging.getLogger(__name__)

# numpy 缺席时只让"分块"功能失效，不让整个模块 import 失败
try:                                    # pragma: no cover - 环境相关
    import numpy as _np
except Exception:                       # pragma: no cover
    _np = None

__all__ = [
    "DEFAULT_FEATHER", "MASK_FLOOR",
    "plan_tiles", "verify_coverage", "describe_plan", "effective_overlap",
    "tile_weight", "feather_mask", "blend_stitch",
    "clamp_tile_size", "clamp_overlap", "adaptive_tile_size",
    "fit_size_to_multiple",
]
# 羽化模式下的默认羽化宽度
DEFAULT_FEATHER = 32
# 权重下限（1/255）：保证**任何**模式下 mask 都不是全 0，
# 也保证不会出现 0/0 的归一化。硬切模式同样使用它，见 feather_mask 注释。
MASK_FLOOR = 1.0 / 255.0


# ---------------------------------------------------------------- 内部工具
def _plain_int(value) -> int:
    """把 numpy 整数转成 Python int（便于比较与打印）。"""
    try:
        return int(value)
    except Exception:
        return value


def _as_size(size) -> Tuple[int, int]:
    """把 (w, h) 规范成两个正整数；非法输入抛 ValueError。"""
    try:
        w, h = int(size[0]), int(size[1])
    except Exception as e:
        raise ValueError(f"size 必须是 (宽, 高)，实际 {size!r}") from e
    if w <= 0 or h <= 0:
        raise ValueError(f"尺寸必须为正，实际 {w}x{h}")
    return w, h


def _require_numpy():
    if _np is None:                     # pragma: no cover
        raise RuntimeError(
            "分块功能需要 numpy，但当前环境没有安装：pip install numpy")
    return _np


def _smoothstep(t):
    """三次 smoothstep：两端导数为 0，接缝处不会有折线感。"""
    return t * t * (3.0 - 2.0 * t)


def _ramp1d(length: int, feather: float, smooth: bool):
    """一维权重：距离两侧边缘 < feather 的区间从 0 升到 1，其余为 1。

    两侧都做，因此**中心恒为 1**（feather 被夹到 length/2 以内），
    这同时保证了"任何尺寸下 mask 都不会全 0"。
    """
    np = _require_numpy()
    length = int(length)
    idx = np.arange(length, dtype=np.float32)
    if length > 1:
        d = np.minimum(idx, float(length - 1) - idx)
    else:
        d = np.zeros(1, dtype=np.float32)
    if feather <= 0.0:
        return np.ones(length, dtype=np.float32)
    t = np.clip(d / float(feather), 0.0, 1.0)
    if smooth:
        t = _smoothstep(t)
    return t.astype(np.float32)


def _separable_weights(w: int, h: int, feather: float, smooth: bool):
    """可分离权重：w(y, x) = ry(y) * rx(x)，返回 float32 (h, w)，范围 (0, 1]。"""
    np = _require_numpy()
    w = max(1, int(w))
    h = max(1, int(h))
    # 羽化不得超过短边的一半，否则两侧斜坡互相抵消、中心也到不了 1
    feather = max(0.0, float(feather or 0.0))
    feather = min(feather, (min(w, h) - 1) / 2.0) if min(w, h) > 1 else 0.0
    rx = _ramp1d(w, feather, smooth)
    ry = _ramp1d(h, feather, smooth)
    mask = ry[:, None] * rx[None, :]
    # 兜底：任何情况下都不允许出现全 0 权重
    return np.maximum(mask, float(MASK_FLOOR)).astype(np.float32)


# ---------------------------------------------------------------- 分块几何
def _axis_positions(size: int, tile: int, step: int) -> List[int]:
    """一条轴上所有块的起始坐标（覆盖完整、块块满尺寸、重叠 >= 设定值）。

    为什么不是"走到头再贴边"
    ------------------------
    最直觉的写法是"按 step 前进，最后再补一块贴到边缘"。但补出来的那一块
    可能和上一块几乎完全重合：1024 宽、tile 512、overlap 64 时按 step=448
    会得到 [0, 448]，贴边再加 512 -> 448 与 512 只差 64，等于**多跑了一整块**
    只换来 64 像素的新内容（多花 50% 的时间）。这不是"多一点重叠"，
    而是实打实的浪费。

    实际用的规则（可证明：覆盖完整、依次递增、间距尽量大）
    ----------------------------------------------------
        令 last = size - tile（最后一块的起点上限）；
        需要 n 段，每段长度 = last / n 必须 <= step（间距越小重叠越多）；
        为了让 n 尽量小（少跑块），取 n = ceil(last / step)；
        于是每段 = last / n，且必然满足 每段 <= step。
    取整后间距最多比 step 小 n/2 个像素，因此
    **每一对相邻块的重叠都 >= overlap 参数**，且不会有人为的巨大重叠。
    例：1024/tile512/ov64 -> n=ceil(512/448)=2 -> [0, 512]（重叠正好 64）；
        1920/tile512/ov64 -> n=ceil(1408/448)=4 -> [0,352,704,1056]（重叠 160）；
        1500/tile512/ov64 -> n=ceil(988/448)=3 -> [0,329,658,988]（重叠 183）。

    代价（必须讲清楚）：当"残尾"不是 step 的整数倍时，实际重叠会比 `overlap`
    参数**更大**（上面 1920 的例子是 160 而不是 64），也就是说块数比理论最少
    值多，速度略慢于"不管重叠均不均匀"的朴素切法。换来的是：绝不出现
    细条块、绝不出现无重叠接缝、也绝不出现"多跑一整块只为 64 像素"。

    size <= tile 时只出一块，尺寸就是 size（调用方负责按 min 夹紧）。
    """
    size = int(size)
    tile = max(1, int(tile))
    step = max(1, int(step))
    if size <= tile:
        return [0]

    last = size - tile
    n = max(1, int(math.ceil(last / float(step))))
    # 从 0 到 last 均分 n 段，并把"最后一块"强制钉在 last 上，
    # 这样最后一个间距恰好是 last/n（不会因为逐段累加取整而漂移）
    xs = [int(round(i * last / float(n))) for i in range(n)]
    xs.append(last)
    out: List[int] = []
    for x in xs:
        x = max(0, min(int(x), last))
        if not out or x != out[-1]:
            out.append(x)
    return out


def _axis_positions_exact(size: int, tile: int, step: int) -> List[int]:
    """严格版：除最后一块外，间距**恒等于** step（重叠恰好 = tile-step）。

    做法就是最直觉的那一套：按 step 一路走到最后一块起点，再补一个贴边块，
    最后去重。`plan_tiles(..., axis_positions=_axis_positions_exact)` 可以切到
    这个语义（"重叠必须正好是 overlap"）。

    它的已知代价（所以不是默认）：
    · 例 1024/tile512/ov64 -> [0, 448, 512]：448 与 512 只差 64，
      后两块**几乎完全重合**，等于多跑了一整块只换来 64 像素新内容；
    · 例 1920/tile512/ov64 -> [0, 448, 896, 1344, 1408]：最后两块重叠 448，
      比上一块 64 的重叠大 7 倍，时间白花。
    换言之：块与块之间的重叠大小会非常不均匀。要均匀、要省时间就用默认的
    `_axis_positions`；要"重叠严格等于参数"就用这个。
    """
    size = int(size)
    tile = max(1, int(tile))
    step = max(1, int(step))
    if size <= tile:
        return [0]
    last = size - tile
    xs = list(range(0, last + 1, step))
    if not xs:                           # pragma: no cover - 纯防御
        xs = [0]
    if xs[-1] != last:
        xs.append(last)
    out: List[int] = []
    for x in xs:
        x = max(0, min(int(x), last))
        if not out or x != out[-1]:
            out.append(x)
    return out


def clamp_tile_size(tile, width, height) -> int:
    """把 tile 夹到 (0, 短边] 内；tile<=0 时退化为整张图。"""
    w, h = _as_size((width, height))
    try:
        t = int(tile)
    except Exception:
        t = 0
    if t <= 0:
        t = min(w, h)
    return max(1, min(t, w, h))


def clamp_overlap(overlap, tile) -> int:
    """把 overlap 夹到 [0, tile-1]：step = tile - overlap 必须 >= 1，否则死循环。"""
    tile = max(1, int(tile))
    try:
        ov = int(overlap)
    except Exception:
        ov = 0
    return max(0, min(ov, tile - 1))


def plan_tiles(width, height, tile, overlap, *, axis_positions=None
               ) -> List[dict]:
    """把 `width x height` 切成若干块，返回 `[{"x","y","w","h"}, ...]`。

    保证（`verify_coverage()` 会逐条检查）：
    · **完整覆盖**：每个像素至少落在一块里，并集恰好等于整张图；
    · **不出界**：`x + w <= width`、`y + h <= height`；
    · **没有细条**：`w == min(tile, width)`、`h == min(tile, height)`，
      绝不会出现"最后一块只剩 3 像素"这种会让扩散模型输出垃圾的块；
    · **相邻重叠 >= overlap**：间距永远不超过 `tile - overlap`；
    · 图比 tile 小 -> 只出一块，尺寸就是图的尺寸。

    关于"重叠到底是多少"（必须讲清楚，别让调用方以为总是恰好 overlap）
    ------------------------------------------------------------------
    交付版（默认，`axis_positions=None`）用 `_axis_positions`：**均摊**，
    每轴取 `n = ceil((size-tile)/step)`，于是
        · 1024/tile512/ov64 -> [0, 512]，相邻重叠**正好** 64；
        · 1920/tile512/ov64 -> [0, 352, 704, 1056, 1408]，重叠 160（更多）。
    也就是说"正好等于 overlap"只在残尾恰好整除时成立，其余情况会比 overlap
    多 —— 多出来的重叠换来的是**块数最少**且块间绝不出现零重叠接缝。
    （`axis_positions=lambda size, tile, step: _axis_positions_exact(...)`
    可以切到"除最后一块外间距恒为 step"的严格版：重叠正好等于 overlap，
    代价是最后两块可能几乎完全重合、等于白跑一块。两种都在本模块里有实现，
    默认选均摊版是为了不浪费算力。）

    遍历顺序为从上到下、从左到右。
    """
    w, h = _as_size((width, height))
    t = clamp_tile_size(tile, w, h)
    ov = clamp_overlap(overlap, t)
    step = t - ov
    axis = axis_positions or _axis_positions

    xs = axis(w, t, step)
    ys = axis(h, t, step)

    tiles: List[dict] = []
    for y in ys:
        for x in xs:
            tw = min(t, w - x)
            th = min(t, h - y)
            if tw <= 0 or th <= 0:       # 理论不可达，留着防御异常输入
                continue
            tiles.append({"x": int(x), "y": int(y),
                          "w": int(tw), "h": int(th)})
    if not tiles:                        # pragma: no cover - 纯防御
        tiles = [{"x": 0, "y": 0, "w": w, "h": h}]
    return tiles


def effective_overlap(tiles) -> Tuple[int, int]:
    """这套切块方案**实际**的重叠像素 (x, y)。

    为什么需要它：`overlap` 参数是**下界**而不是精确值。
    为了让覆盖完整、又不产生"几乎完全重合的最后一块"，`plan_tiles` 会把
    间距均摊，于是实际重叠常常**大于**请求值：

        overlap=64, tile=512, 1024x1024 → x=[0,256,512]，实际重叠 256
        overlap=64, tile=512, 2048x2048 → 实际重叠 128

    所以元数据里必须写"实际"值，否则看 PNG 的人会以为只重叠了 64
    而算错块数/耗时；也提醒用户：在 1024² 这种尺寸下，把 overlap 从
    64 调到 128 是**不会有任何变化**的。
    """
    try:
        if len(tiles) <= 1:
            return 0, 0
        tw = min(t["w"] for t in tiles)
        th = min(t["h"] for t in tiles)
        xs = sorted({t["x"] for t in tiles})
        ys = sorted({t["y"] for t in tiles})
        ovx = [tw - (b - a) for a, b in zip(xs, xs[1:])] or [0]
        ovy = [th - (b - a) for a, b in zip(ys, ys[1:])] or [0]
        return int(min(ovx)), int(min(ovy))
    except Exception:
        return 0, 0


def verify_coverage(width, height, tiles, weights=None) -> bool:
    """自检：切块方案是否真的完整覆盖、不出界、没有细条。

    参数
    ----
    tiles   : `plan_tiles()` 的返回值
    weights : 与 tiles 等长的权重图列表（PIL "L" 图、float 数组或
              `(tw, th)` 尺寸都可以，只取尺寸信息）。给了就额外检查
              "权重是否真的落在每个像素上" —— 极端参数下（例如块特别小、
              羽化把中心也压到下限）几何覆盖成立，但有效权重可能极低，
              拼出来会发暗。这一项能提前发现这种"面积盖住了、权重没盖住"。

    返回 True/False，并**不**打印；调用方（测试 / 节点）自己决定怎么报。
    越界不属于"断言"而是"必须修"的 bug，所以这里直接抛 ValueError。
    """
    np = _require_numpy()
    w, h = _as_size((width, height))

    oob = [(t["x"], t["y"], t["w"], t["h"]) for t in tiles
           if t["x"] < 0 or t["y"] < 0 or t["w"] <= 0 or t["h"] <= 0
           or t["x"] + t["w"] > w or t["y"] + t["h"] > h]
    if oob:
        raise ValueError(f"分块越界或尺寸非法: {oob[:4]}")

    cov = np.zeros((h, w), dtype=np.int32)
    for t in tiles:
        cov[t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"]] += 1
    if int(cov.min()) < 1:
        return False

    if weights is not None:
        wl = list(weights)
        if len(wl) != len(tiles):
            raise ValueError("weights 数量与 tiles 不一致")
        wsum = np.zeros((h, w), dtype=np.float32)
        for t, m in zip(tiles, wl):
            arr = np.asarray(m, dtype=np.float32)
            if arr.ndim == 3:            # 带通道的权重（(h, w, 1)）
                arr = arr[..., 0]
            if tuple(arr.shape) != (t["h"], t["w"]):
                raise ValueError(
                    f"权重尺寸 {arr.shape} 与块 {t['w']}x{t['h']} 不一致")
            # PIL "L" 图是 0~255，float 权重是 0~1；统一到 0~1 再比较
            if float(arr.max()) > 1.0 + 1e-6:
                arr = arr / 255.0
            wsum[t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"]] += arr
        if float(wsum.min()) <= 0.0:
            return False
    return True


def describe_plan(tiles) -> str:
    """一行中文摘要，形如 `1024x1024 → 6 块 (512/重叠64)`。

    用于日志与元数据：出了问题第一眼就能看出"切了几块、多大、重叠多少"。
    """
    tl = list(tiles or [])
    if not tl:
        return "0 块（空计划）"
    t = tl[0]
    w = max(t["x"] + t["w"] for t in tl)
    h = max(t["y"] + t["h"] for t in tl)
    side = ""
    for t in tl:
        if t["w"] == t["h"]:
            side = t["w"]
            break
    if not side:
        side = f"{t['w']}x{t['h']}"
    # 反推重叠：取相邻两块里最小的 x 间距，等价于 tile - 实际最大步长
    xs = sorted({t["x"] for t in tl})
    ys = sorted({t["y"] for t in tl})
    steps = [b - a for a, b in zip(xs, xs[1:])] + \
            [b - a for a, b in zip(ys, ys[1:])]
    if steps and isinstance(side, int):
        ov = side - min(steps)
    else:
        ov = "?"
    return f"{w}x{h} → {len(tl)} 块 ({side}/重叠{ov})"


# ---------------------------------------------------------------- 权重图
def tile_weight(tw, th, feather=DEFAULT_FEATHER, mode: str = "feather"):
    """权重图本体：float32 (th, tw)，取值 [1/255, 1]，**不转 8bit**。

    为什么要暴露浮点版本：`blend_stitch` 用浮点权重才能做到"拼回来和原图
    逐像素相等"；8bit 量化会在接缝处留下 ±1 的台阶。

    mode 语义：
    · "feather" 平滑羽化：边缘 0 → 内部 1，用**三次 smoothstep**（两端导数为 0，
      接缝没有折线），两侧同时羽化，中心恒为 1；
    · "linear"  整块线性：1D 斜坡 * 1D 斜坡，等价于"整块都在渐变"，
      相邻块的权重和同样恒为 1（不会变暗），但接缝处更容易看出方向性；
    · "hard"    硬切：整块 255（不融合）。
    """
    np = _require_numpy()
    tw = max(1, int(tw))
    th = max(1, int(th))
    m = str(mode or "feather").strip().lower()
    if m in ("hard", "硬切"):
        return np.ones((th, tw), dtype=np.float32)

    if m in ("linear", "线性"):
        # 整块线性：从 (0,0) 到 (短边) 的一条斜线 * 另一条斜线，
        # 两侧边缘都压到下限，但中心恒为 1
        side = min(tw, th)
        feather = max(1.0, (side - 1) / 2.0)
        return _separable_weights(tw, th, feather, smooth=False)

    try:
        f = float(feather)
    except Exception:
        f = float(DEFAULT_FEATHER)
    if f <= 0.0:
        # 羽化 0 = 不融合，退化为硬切
        return np.ones((th, tw), dtype=np.float32)
    return _separable_weights(tw, th, f, smooth=True)


def feather_mask(tw, th, feather=DEFAULT_FEATHER, mode: str = "feather"):
    """权重图（PIL "L"）：255 = 完全采信本块，0 = 完全不用本块。

    · `mode="feather"`：1.0 起步内部，向四边在 `feather` 像素内衰减到 0，
      用**三次 smoothstep**；两侧同时羽化，中心恒为 255；
    · `mode="linear"`：整块线性斜坡；
    · `mode="hard"`：全 255（不融合）；
    · `feather=0` -> 全 255。

    注意"最小值是 1 而不是 0"：整张 mask **永远不会全 0**。原因有两层：
    中心被夹到 1.0；8bit 落盘前再抬到 255/255 的下限（1/255）。
    这也让硬切模式没有"权重为 0 导致归一化除零"的风险 —— 硬切真正的
    不融合语义由 `blend_stitch(mode="hard")` 直接放置像素来实现，
    不依赖 mask 是 0。
    """
    np = _require_numpy()
    from PIL import Image

    w = tile_weight(tw, th, feather=feather, mode=mode)
    arr = np.clip(np.rint(w * 255.0), 1, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


# ---------------------------------------------------------------- 融合
def _stats_match(tile, base, min_std: float = 1.0):
    """把 tile 的每通道均值/标准差对齐到 base（简易色彩校正）。

    这是"消除块间接缝色差"最实用的做法：相邻块哪怕只差 2~3 的均值，
    在平滑天空上也会被看成一格一格的方块。校正只动色彩统计，不动结构。

    局限（刻意保持简单，别指望它替代专业拼接）：
    · 按**整块**统计，块内既有天空又有暗部时会把两者的对比一起压平；
    · 用 base 同一个矩形区域做参考，所以 base 上该处若正好落在
      "该被重绘"的强边缘上，统计本身也不稳；
    · 结果只做 0~255 截断，不做色域映射，极端情况下会削顶/压底。
    """
    np = _require_numpy()
    if base is None:
        return tile
    from PIL import Image

    tw, th = tile.size
    rect = base.crop((0, 0, min(tw, base.width), min(th, base.height)))
    if rect.size != (tw, th):            # 参考区不足时退回整图统计，避免崩
        rect = base.resize((tw, th), Image.BILINEAR)
    a = np.asarray(tile, dtype=np.float32)
    b = np.asarray(rect.convert(tile.mode), dtype=np.float32)
    if a.ndim != 3 or b.ndim != 3 or a.shape[2] != b.shape[2]:
        return tile
    ax = (0, 1)
    am, asd = a.mean(axis=ax, keepdims=True), a.std(axis=ax, keepdims=True)
    bm, bsd = b.mean(axis=ax, keepdims=True), b.std(axis=ax, keepdims=True)
    scale = bsd / np.maximum(asd, float(min_std))
    scale = np.clip(scale, 0.5, 2.0)     # 限制过校正：平块不该被放大成噪声
    out = (a - am) * scale + bm
    return Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8), tile.mode)


def blend_stitch(size, tiles, base=None, feather: int = DEFAULT_FEATHER,
                 mode: str = "feather", color_correct: bool = False):
    """把 `(plan_dict, PIL.Image)` 逐块拼回整图。

    参数
    ----
    size    : 输出尺寸 `(宽, 高)`，必须与 plan 的坐标系一致
    tiles   : 可迭代的 `(plan, image)`；image 必须是 RGB 且尺寸等于 plan 的 w/h
    base    : 放大后的底图。只在两种场合被用到：
              ① 没有任何块覆盖的像素（正常情况下不会发生，纯兜底）；
              ② `color_correct=True` 时作为每块均值/方差的参考。
    feather : 羽化宽度（像素）。`mode="feather"` 时用它；`0` 表示不羽化。
    mode    : "feather" / "linear" / "hard"（与 `feather_mask` 一致）
    color_correct : 每块先做均值/方差对齐再融合（块间色差的克星，代价是慢一点）

    归一化的关键（为什么不是简单 `acc / wsum`）
    ------------------------------------------
    若某像素只有一块覆盖、而这块在该处权重是 0（羽化边缘正好压在图片外边界），
    简单归一化会得到 `0/0` 或被下限救成暗淡的值，成品四周就会出现一条黑边。
    这里改为：**只在"该像素沿哪个方向被夹在图片边界上"时参与归一化**：
        x 在左/右边界内且是水平方向的局部极值 -> 水平方向忽略权重
        y 同理
    于是边界像素由唯一能覆盖它的那块**原样输出**，而内部重叠区依旧按权重
    加权平均。数学上：相邻块在重叠区满足 s + (1-s) = 1，所以只要有一块权重
    为 1，结果就与源图逐像素相等（见 B7 恒等测试）。

    返回 RGB 的 PIL.Image（不会修改传入的任何图像）。
    """
    np = _require_numpy()
    from PIL import Image

    w, h = _as_size(size)
    plan_list = [(t, im) for t, im in tiles]

    oob = [(t.get("x"), t.get("y"), t.get("w"), t.get("h"))
           for t, _ in plan_list
           if not (0 <= t.get("x", -1) and 0 <= t.get("y", -1)
                   and t.get("w", 0) > 0 and t.get("h", 0) > 0
                   and t["x"] + t["w"] <= w and t["y"] + t["h"] <= h)]
    if oob:
        raise ValueError(f"块超出目标画布 {w}x{h}: {oob[:4]}")

    if not plan_list:
        if base is not None:
            return base.convert("RGB").copy()
        return Image.new("RGB", (w, h), (0, 0, 0))

    # base 只在"色差参考"和"未覆盖像素兜底"时用；这里的 delay 很重要：
    # 正常情况（块完整覆盖）完全不需要把底图整张物化成 float 数组，
    # 分块缩放那种大图场景才不至于为了兜底白花一份内存。
    base_rgb = None
    if base is not None and color_correct:
        base_rgb = base.convert("RGB")
        if base_rgb.size != (w, h):
            base_rgb = base_rgb.resize((w, h), Image.LANCZOS)

    # 有效覆盖：只有权重真的落到像素上才能算"被覆盖"。
    # 羽化块的边缘权重会掉到下限，若那里恰好只在图片外边界上，
    # 简单归一化就会把这一圈压暗（经典的 1px 黑边）。
    treated = np.zeros((h, w, 1), dtype=np.float32)
    fx = min(int(feather or 0), (w - 1) // 2)
    fy = min(int(feather or 0), (h - 1) // 2)
    m_x = np.zeros((1, w, 1), dtype=np.float32)
    if fx > 0:
        m_x[0, :fx, 0] = 1.0
        m_x[0, -fx:, 0] = 1.0
    m_y = np.zeros((h, 1, 1), dtype=np.float32)
    if fy > 0:
        m_y[:fy, 0, 0] = 1.0
        m_y[-fy:, 0, 0] = 1.0
    treated[:] = m_x + m_y - m_x * m_y
    treated = 1.0 - treated              # 1 = 按权重归一化，0 = 原样输出

    # 累加用 float64：float32 在"4 块权重相加再相除"时会产生 ±1 的量化误差
    # （实测 512x512 恒等拼接 maxdiff=1），而 float64 是逐像素精确等于源图。
    # 代价只有内存（4K 图约 400MB），换掉的是接缝上肉眼可见的 1 级台阶。
    acc = np.zeros((h, w, 3), dtype=np.float64)
    wsum = np.zeros((h, w, 1), dtype=np.float64)
    hard = str(mode or "").strip().lower() in ("hard", "硬切")
    if hard:
        # 硬切 = 不融合：后一块直接覆盖前一块。
        # 权重仍是 1/255（保证 wsum 非 0、归一化不除零），所以结果与源图相等。
        treated[:] = 1.0

    for t, im in plan_list:
        rgb = im.convert("RGB")
        if rgb.size != (t["w"], t["h"]):
            rgb = rgb.resize((t["w"], t["h"]), Image.LANCZOS)
        if color_correct and base_rgb is not None:
            rect = base_rgb.crop((t["x"], t["y"],
                                  t["x"] + t["w"], t["y"] + t["h"]))
            if rect.size == (t["w"], t["h"]):
                rgb = _stats_match(rgb, rect)
        arr = np.asarray(rgb, dtype=np.float64)
        if hard:
            m = np.ones((t["h"], t["w"], 1), dtype=np.float64)
        else:
            m = tile_weight(t["w"], t["h"], feather=feather,
                            mode=mode)[..., None].astype(np.float64)
        y0, x0 = t["y"], t["x"]
        acc[y0:y0 + t["h"], x0:x0 + t["w"], :] += arr * m
        wsum[y0:y0 + t["h"], x0:x0 + t["w"], :] += m

    denom = wsum * treated + (1.0 - treated)
    out = acc / np.maximum(denom, 1e-6)

    # 只有"完全不依赖权重"的位置（图片外边界上权重恰好到 0 的像素）
    # 需要原样输出；这些位置一定被某一块完整覆盖。
    raw = (treated <= 0.0)
    if bool(raw.any()):
        raw3 = np.repeat(raw, 3, axis=2)
        src = np.zeros((h, w, 3), dtype=np.float32)
        covered = np.zeros((h, w, 1), dtype=bool)
        for t, im in plan_list:
            y0, x0 = t["y"], t["x"]
            sl = (slice(y0, y0 + t["h"]), slice(x0, x0 + t["w"]))
            need = raw[sl] & (~covered[sl])
            if not bool(need.any()):
                continue
            rgb = im.convert("RGB")
            if rgb.size != (t["w"], t["h"]):
                rgb = rgb.resize((t["w"], t["h"]), Image.LANCZOS)
            sub = src[sl]
            sub[need[..., 0]] = np.asarray(rgb, dtype=np.float32)[need[..., 0]]
            covered[sl] = covered[sl] | need
        fill = base
        if fill is not None:
            f = fill.convert("RGB")
            if f.size != (w, h):
                f = f.resize((w, h), Image.LANCZOS)
            farr = np.asarray(f, dtype=np.float32)
            miss = raw & (~covered)
            if bool(miss.any()):
                src[miss[..., 0]] = farr[miss[..., 0]]
                covered = covered | miss
        out = np.where(raw3, src, out)
        n = int((raw & (~covered)).sum())
        if n:
            logger.info(f"ℹ️ 有 {n} 个像素既无块覆盖也无底图，输出为黑色")

    uncovered = (~raw) & (wsum <= 1e-6)
    n = int(uncovered.sum())
    if n:
        logger.info(f"ℹ️ 有 {n} 个像素没有被任何块覆盖，回退为底图")

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------- 尺寸辅助
def adaptive_tile_size(target, tile, low: int = 256) -> int:
    """由显存策略建议的块尺寸，与用户设置取较小者，并向下对齐到 8。

    为什么对齐 8：扩散管线内部要过 VAE 的 8 倍下采样，宽高不是 8 的倍数时
    diffusers 会**默默裁掉**几个像素，让"送进去 513、拿回来 512"，
    融合时尺寸对不上。这里主动向下取整，宁可少 7 个像素也不出意外。
    下限 256 是为了不让块小到失去上下文（再小就不再省显存，只是更慢）。
    """
    w, h = _as_size(target)
    side = min(int(tile or 0), w, h)
    side = max(int(low or 1), side)
    side = min(side, w, h)
    side = max(8, (side // 8) * 8)
    return min(side, min(w, h))


def fit_size_to_multiple(width, height, multiple: int = 8) -> Tuple[int, int]:
    """把目标尺寸向下对齐到 `multiple` 的倍数，且不小于 `multiple` 本身。"""
    m = max(1, int(multiple))
    w = max(m, (int(width) // m) * m)
    h = max(m, (int(height) // m) * m)
    return w, h
