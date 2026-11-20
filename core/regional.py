# core/regional.py
"""区域提示词与局部重绘引擎（**无界面依赖**，可被工作流/界面/测试直接调用）。

本模块只做一件事：给一张底图，按"区域"逐个做**蒙版局部重绘**，再按羽化
蒙版合成回底图。区域可以带自己的提示词、局部负面提示词、权重（=重绘强度）、
以及可选的局部 ControlNet / 局部 ADetailer。

实现的路线：顺序蒙版局部重绘（区域精修）
==========================================
    base(全局提示词)  ──►  区域1 蒙版重绘  ──►  区域2 蒙版重绘  ──► ...  ──►  合成
                          强度=区域权重        强度=区域权重

**每个区域跑一次独立的 inpaint 采样**，提示词/负面词/权重都只作用于那一次
采样，结果按该区域的羽化蒙版贴回当前图。为什么选这条路线：

1. **复用既有管线**：直接走 `adapter.pipe(model, "inpaint")`，和 `Inpaint`
   节点是同一条代码路径。没有新造采样循环，也就没有新的显存/调度坑。
2. **行为可预测**：一个区域就是一次 inpaint，"权重 = 重绘强度"是用户立刻
   能理解、能预期结果的语义（0.3 轻微修、0.8 大改）。
3. **8GB 显存友好**：区域**逐个**处理，任何时刻只有一次采样在跑，峰值显存
   与区域数量基本无关（实测见测试输出的 max_memory_allocated）。
4. **六项能力齐全**：矩形区域、蒙版区域、区域权重、局部负面提示词、
   局部 ControlNet、局部 ADetailer 都能表达。

代价（必须讲清楚，不能藏）
--------------------------
因为区域是在**底图出来之后**才精修的，区域提示词**不参与**底图早中期的
去噪步，所以全局协调性（不同区域的光照/风格统一程度）会略低于
"每一步都做注意力耦合"的方法（A1111 Regional Prompter / Latent Couple /
ComfyUI 的 attention couple）。本实现**故意不做**逐步注意力耦合：
那需要 hook UNet/Transformer 的 cross-attention 并按蒙版改写注意力掩码，
对不同架构（SD1.5 / SDXL / SD3 / Flux）各写一套，8GB 上还容易把显存吃爆。
**逐步耦合属于后续工作**，接口上已经有位置（`Region` 与 `apply_regions`
的签名不需要改动，只要替换采样内核即可）。

局部 ControlNet / 局部 ADetailer 的现状（诚实说明）
---------------------------------------------------
· 局部 **ControlNet**：`core.model_manager.prepare_controlnet()` 构建的是
  **txt2img 的 ControlNet 管线**（没有 mask_image 入参），所以区域内带 CN 时
  走的是"CN 引导的文生图 + 用区域蒙版贴回"，而不是"CN 引导的 inpaint"。
  取不到 CN 管线（未配置/未下载权重）时**只记 warning 并继续**，不会让整
  区域失败 —— 局部 CN 是增强项，不是必需项。
· 局部 **ADetailer**：ADetailer 的检测器会扫**整张图**，没有"只看这块蒙版"
  的入参，所以这里先用蒙版框出该区域的包围盒（带 15% 留白）裁出小图，
  在**小图上**跑 ADetailer，再按羽化蒙版贴回。这样"局部"是真的局部；
  代价是检测只在区域框内进行（区域外的人脸不会被本区域修到，符合预期）。
· 两者拿不到时都只 warning，不进 `region_errors` —— 只有"这一次重绘本身
  失败"才是错误。

约定
----
· 蒙版：白(255)=要重绘的区域；`region_masks()` 返回 PIL "L" 图。
· 合成：`out = base * (1 - m) + redrawn * m`，m 是羽化后的浮点蒙版。
· **绝不修改调用方传入的 PIL 图**（全部走 copy）。
· torch / diffusers 一律在函数内部 import —— 本模块 import 时零重依赖，
  界面和纯逻辑测试都能安全导入。
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "Region", "RegionsValue",
    "regions_from_json", "regions_to_json", "region_masks",
    "describe", "validate_regions", "plan_regions",
    "apply_regions", "Cancelled",
]

# 合成与统计用的数据类型（numpy/PIL 都是本项目的硬依赖，缺失时函数内部会
# 给出明确报错，而不是让模块 import 直接失败）
_EPS = 1e-6

# 默认基准分辨率：RegionalPrompt 节点在没有底图时用它做几何校验
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 768

# 无效值判定：矩形的宽或高小于该值就当作"空矩形"
_MIN_RECT_PX = 1e-6


# ======================================================================
#  取消信号（与 workflow.schema.Cancelled 是同一个类，避免"两种取消"）
# ======================================================================
def _cancelled_class():
    """延迟取 Cancelled 类。

    优先级用 `workflow.schema.Cancelled`（执行器认它），拿不到就退化成模块内
    自定义异常 —— core 不应该在 import 期强依赖 workflow 包。
    """
    try:
        from workflow.schema import Cancelled as _C
        return _C
    except Exception:                                   # pragma: no cover
        return _FallbackCancelled


class _FallbackCancelled(Exception):
    """兜底取消异常（只在 workflow 包不可用时出现）。"""


Cancelled = _cancelled_class()


def _raise_cancelled(msg: str = "用户已取消"):
    raise Cancelled(msg)


# ======================================================================
#  区域定义
# ======================================================================
def _f(value, default: float) -> float:
    """把任意值转 float；失败给默认值（容错：绝不抛）。"""
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def _i(value, default: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _s(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _b(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "是")
    try:
        return bool(value)
    except Exception:
        return default


def _d(value) -> Optional[dict]:
    """把值规范成 dict 或 None（容错：绝不抛）。"""
    if isinstance(value, dict):
        return dict(value)
    return None


def _rect_tuple(value, default=(0.0, 0.0, 0.3, 0.3)) -> tuple:
    """把任意值规范成 4 元 float 元组（容错：绝不抛）。"""
    try:
        seq = list(value)
    except Exception:
        return tuple(default)
    if len(seq) < 4:
        return tuple(default)
    try:
        return tuple(float(v) for v in seq[:4])
    except Exception:
        return tuple(default)


@dataclass
class Region:
    """一个区域的定义（纯数据，可 JSON 序列化，UI 与引擎共用）。

    坐标一律**归一化**（0~1，相对整图），这样同一份区域定义能用在任意输出
    尺寸上 —— 这一点很重要：用户可能在 UI 里按 512x768 画框，最后生成
    1024x1536，归一化坐标不需要任何转换。
    """

    name: str = ""              # 显示名，如 "脸部"
    prompt: str = ""
    negative: str = ""          # 局部负面提示词（空=只用全局负面）
    weight: float = 0.5         # 区域权重 → 该区域 inpaint 的 strength（0~1）
    shape: str = "rect"         # "rect" | "mask"
    rect: tuple = (0.0, 0.0, 0.3, 0.3)   # **归一化** (x0,y0,x1,y1)，0~1，相对整图
    mask_path: str = ""         # shape=="mask" 时的黑白图路径（白=生效）
    color: str = "#f38ba8"      # UI 画框用，引擎忽略
    enabled: bool = True
    # 局部附加能力（都可选）
    controlnet: Optional[dict] = None   # {"kind":"openpose","weight":0.8,"image_path":""}
    adetailer: Optional[dict] = None    # {"target":"二次元脸部","strength":0.35}
    # 局部 IP-Adapter（多参考图）：该区域**额外**参与的参考图。
    #   {"refs": [{"role":"style","path":"...","weight":0.6,
    #              "mask_path":""}, ...], "base": ""}
    # 语义与 IP-Adapter 节点一致：本区域采样时把这几张参考图一起送进去，
    # 并且**只用本区域的蒙版**限制它们（参考图会先按蒙版抠图）。
    # 为空 → 本区域不额外引入参考图（但若全局已经装了 IPA，仍要占位，
    # 详见 `_region_sampler` 里的说明）。
    ip_adapter: Optional[dict] = None
    feather: int = 16           # 该区域边缘羽化像素

    # ------------------------------------------------------------------
    def __post_init__(self):
        """字段规范化（容错）：手工构造 Region 时也不该因为类型怪异而炸。

        dataclass 的 `rect` 默认值是元组，但调用方（界面拖动、JSON）经常塞
        list 或长度不足的序列进来，这里一次性统一成 4 元 float 元组。
        """
        self.rect = _rect_tuple(self.rect, (0.0, 0.0, 0.3, 0.3))
        self.weight = _f(self.weight, 0.5)
        self.feather = _i(self.feather, 16)
        shape = (_s(self.shape, "rect") or "rect").strip().lower()
        if shape in ("矩形", "rectangle", "box"):
            shape = "rect"
        elif shape in ("蒙版", "遮罩", "mask_image"):
            shape = "mask"
        self.shape = shape if shape in ("rect", "mask") else "rect"
        self.controlnet = _d(self.controlnet)
        self.adetailer = _d(self.adetailer)
        self.ip_adapter = _d(self.ip_adapter)

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的 dict（字段齐全，UI/引擎共用一份）。"""
        return {
            "name": self.name,
            "prompt": self.prompt,
            "negative": self.negative,
            "weight": self.weight,
            "shape": self.shape,
            "rect": list(self.rect),
            "mask_path": self.mask_path,
            "color": self.color,
            "enabled": self.enabled,
            "controlnet": dict(self.controlnet) if self.controlnet else None,
            "adetailer": dict(self.adetailer) if self.adetailer else None,
            "ip_adapter": dict(self.ip_adapter) if self.ip_adapter else None,
            "feather": self.feather,
        }

    @staticmethod
    def from_dict(d) -> "Region":
        """从 dict 构造；**容错**：缺字段用默认值，任何垃圾输入都不抛异常。

        为什么这么强调容错：区域定义会从 JSON、UI 剪贴板、旧版本工作流文件
        三个地方进来。任何一种格式变化都不该让整张工作流跑不起来 —— 用户可以
        在界面上看到"这个区域被跳过了"，但不该看到一个 TypeError。
        """
        if not isinstance(d, dict):
            return Region()
        r = Region()
        r.name = _s(d.get("name"), "")
        r.prompt = _s(d.get("prompt"), "")
        r.negative = _s(d.get("negative"), "")
        r.weight = _f(d.get("weight"), 0.5)
        r.shape = (_s(d.get("shape"), "rect") or "rect").strip().lower()
        if r.shape in ("矩形", "rectangle", "box"):
            r.shape = "rect"
        elif r.shape in ("蒙版", "遮罩", "mask_image"):
            r.shape = "mask"
        r.rect = _rect_tuple(d.get("rect"), (0.0, 0.0, 0.3, 0.3))
        r.mask_path = _s(d.get("mask_path"), "")
        r.color = _s(d.get("color"), "#f38ba8") or "#f38ba8"
        r.enabled = _b(d.get("enabled"), True)
        r.controlnet = _d(d.get("controlnet"))
        r.adetailer = _d(d.get("adetailer"))
        r.ip_adapter = _d(d.get("ip_adapter"))
        r.feather = _i(d.get("feather"), 16)
        return r

    # ------------------------------------------------------------------
    def display_name(self) -> str:
        """带序号兜底的显示名（空名字不显示成空白）。"""
        return self.name.strip() or "未命名区域"

    def is_mask_shape(self) -> bool:
        return self.shape == "mask"

    def rect_px(self, width: int, height: int,
                clamp: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """归一化矩形 → 像素矩形 (l, t, r, b)；空/无效返回 None。

        `clamp=True` 时把越界坐标夹进图像范围（用户把框拖出画布是常事，
        不该整区域失效）。宽或高为 0 视为空矩形，返回 None。
        """
        x0, y0, x1, y1 = self.rect
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        l = int(round(x0 * width))
        t = int(round(y0 * height))
        r = int(round(x1 * width))
        b = int(round(y1 * height))
        if clamp:
            l = max(0, min(l, width))
            t = max(0, min(t, height))
            r = max(0, min(r, width))
            b = max(0, min(b, height))
        if r - l <= _MIN_RECT_PX or b - t <= _MIN_RECT_PX:
            return None
        return l, t, r, b

    def merged_controlnet(self, defaults: Optional[dict] = None) -> Optional[dict]:
        """合并默认 CN 参数与区域覆盖；kind 为空视为"没有局部 CN"。"""
        base = dict(defaults or {})
        base.update(self.controlnet or {})
        kind = _s(base.get("kind"), "").strip().lower()
        if not kind or kind in ("none", "无", "关闭"):
            return None
        base["kind"] = kind
        return base

    def merged_adetailer(self) -> Optional[dict]:
        """局部 ADetailer 配置；target 为空视为"不修"。"""
        d = dict(self.adetailer or {})
        target = _s(d.get("target"), "").strip()
        if not target or target in ("none", "无", "关闭"):
            return None
        d["target"] = target
        return d

    def merged_ip_adapter(self) -> Optional[dict]:
        """局部 IP-Adapter 配置；没有任何参考图视为"本区域不额外引入"。

        接受两种写法（都容错）：
          · {"refs": [ {"role":..,"path":..,"weight":..}, ... ]}
          · {"role":.., "path":.., "weight":..}     ← 单条参考图
        """
        d = dict(self.ip_adapter or {})
        refs = d.get("refs")
        if refs is None:
            # 单条写法：本 dict 自己就是一条参考图（去掉 refs/base 这类键）
            single = {k: v for k, v in d.items() if k not in ("refs", "base")}
            if any(single.get(k) for k in ("path", "image_path", "file", "role")):
                refs = [single]
        if isinstance(refs, dict):
            refs = [refs]
        if not isinstance(refs, (list, tuple)) or not refs:
            return None
        d["refs"] = list(refs)
        return d


class RegionsValue(list):
    """`REGIONS` 端口上流动的值：就是个 list[Region]。

    单独定义一个子类只为让"这串东西是区域列表"在下游一眼可辨，
    行为与 list 完全一致（`[r for r in val]`、`len(val)` 都照常）。
    """

    def to_json(self) -> str:
        return regions_to_json(self)


# ======================================================================
#  JSON 读写（全部容错）
# ======================================================================
def regions_from_json(text) -> List[Region]:
    """从 JSON 文本解析区域列表；**非法输入返回 []**（绝不抛）。

    接受的形态：
      · `[ {...}, {...} ]`              —— 标准形态
      · `{"regions": [ ... ]}`         —— 带外层包装（编辑器导出常见）
      · 单个 `{...}`                    —— 懒人写法，当成一个区域
    """
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        raw = list(text)
    elif isinstance(text, dict):
        raw = text.get("regions", text)
        if not isinstance(raw, (list, tuple)):
            raw = [text]
    else:
        s = str(text).strip()
        if not s:
            return []
        try:
            data = json.loads(s)
        except Exception:
            logger.info("ℹ️ 区域 JSON 解析失败（按空列表处理）")
            return []
        if isinstance(data, dict) and "regions" not in data:
            # 单个区域对象（懒人写法）；空对象 {} 视为"没有区域"
            return [Region.from_dict(data)] if data else []
        return regions_from_json(data)

    out: List[Region] = []
    for item in raw:
        if isinstance(item, Region):
            out.append(item)
        else:
            out.append(Region.from_dict(item))
    return out


def regions_to_json(regions) -> str:
    """区域列表 → JSON 文本（ensure_ascii=False，中文名保持可读）。"""
    items = []
    for r in regions or []:
        if isinstance(r, Region):
            items.append(r.to_dict())
        elif isinstance(r, dict):
            items.append(Region.from_dict(r).to_dict())
    try:
        return json.dumps(items, ensure_ascii=False, indent=2)
    except Exception:                                   # pragma: no cover
        return "[]"


def coerce_regions(value) -> List[Region]:
    """把端口上的任意值规范成 list[Region]（容错，供节点调用）。"""
    if value is None:
        return []
    if isinstance(value, str):
        return regions_from_json(value)
    if isinstance(value, Region):
        return [value]
    if isinstance(value, dict):
        if "regions" in value and isinstance(value["regions"], (list, tuple)):
            return coerce_regions(value["regions"])
        return [Region.from_dict(value)]
    try:
        return [r if isinstance(r, Region) else Region.from_dict(r)
                for r in value]
    except Exception:
        return []


# ======================================================================
#  蒙版生成
# ======================================================================
def _ramp(length: int, feather: float):
    """一维羽化斜坡：两端 feather 像素内 0→1，中间恒为 1。

    用三次 smoothstep（两端导数为 0），贴回时不会在边界留下折线感；
    斜坡宽度夹到 length/2 以内，保证**中心永远是 1**（不会整块被压暗）。
    """
    import numpy as np

    length = max(1, int(length))
    idx = np.arange(length, dtype=np.float32)
    if length > 1:
        d = np.minimum(idx, float(length - 1) - idx)
    else:
        d = np.zeros(1, dtype=np.float32)
    f = float(feather or 0.0)
    if f <= 0.0:
        return np.ones(length, dtype=np.float32)
    f = min(f, (length - 1) / 2.0) if length > 1 else 0.0
    if f <= 0.0:
        return np.ones(length, dtype=np.float32)
    t = np.clip(d / f, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _alpha_rect(width: int, height: int, box, feather: int):
    """矩形蒙版的浮点 alpha（0~1），(h, w)。

    羽化**只向框外扩散、不啃框内**：先生成外扩 feather 像素的矩形，
    再用可分离斜坡衰减，于是 `[l+feather, r-feather]` 区间恒为 1.0，
    框外 feather 像素内平滑降到 0。这与 `MaskCreate` 用高斯模糊的观感一致，
    但不会像高斯那样把框中心也压暗（小区域的常见坑）。
    """
    import numpy as np

    l, t, r, b = box
    m = np.zeros((height, width), dtype=np.float32)
    if r <= l or b <= t:
        return m
    f = max(0, int(feather or 0))
    lx, ty = max(0, l - f), max(0, t - f)
    rx, by = min(width, r + f), min(height, b + f)
    if rx <= lx or by <= ty:
        return m
    w = rx - lx
    h = by - ty
    # 可分离斜坡：row * col，矩形内侧 >= feather 的区间恒为 1.0
    alpha = _ramp(w, f)[None, :] * _ramp(h, f)[:, None]
    # 内区显式压到 1.0：极窄的矩形（宽或高小于 2*feather）也不会被斜坡削掉中心
    alpha[max(0, t - ty):max(1, b - ty),
          max(0, l - lx):max(1, r - lx)] = 1.0
    m[ty:by, lx:rx] = alpha
    return m


def _load_mask_alpha(path: str, width: int, height: int):
    """读蒙版文件 → 与目标同尺寸的浮点 alpha；失败返回 None（不抛）。"""
    import numpy as np
    from PIL import Image

    try:
        if not path or not os.path.exists(path):
            return None
        with Image.open(path) as im:
            g = im.convert("L")
            if g.size != (width, height):
                g = g.resize((width, height), Image.LANCZOS)
            return np.asarray(g, dtype=np.float32) / 255.0
    except Exception as e:
        logger.warning(f"⚠️ 读取区域蒙版失败（跳过该区域）: {path} ({e})")
        return None


def _alpha_to_pil(alpha) -> Any:
    """浮点 alpha(0~1) → PIL "L"（255 = 生效）。

    注意：这里**不做** `clip(1,255)` —— 区域蒙版本该可以是全 0（判定为
    "空蒙版"并被跳过），与分块融合的 mask 语义（不允许全 0）不同。
    """
    import numpy as np
    from PIL import Image

    arr = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    arr = np.rint(arr * 255.0).astype("uint8")
    return Image.fromarray(arr, "L")


def region_masks(regions, width: int, height: int,
                 feather_override: Optional[int] = None
                 ) -> List[Tuple[Region, Any]]:
    """把区域定义光栅化成蒙版，返回 `[(Region, PIL "L" 蒙版), ...]`。

    · `shape=="rect"`：按归一化坐标 × 宽高光栅化，越界坐标自动 clamp；
    · `shape=="mask"`：读图 → 缩放到 (width, height) → 灰度；
    · 两者都做 `feather` 羽化（区域自己的羽化像素数，默认 16）；
    · 空矩形（宽或高为 0，或 clamp 后没有面积）、缺文件、读图失败的蒙版
      体积为 0 —— 由 `apply_regions` 判定为"空蒙版"并跳过。

    `feather_override` 非 None 时**覆盖**所有区域的羽化值（只影响本次调用，
    不改 Region 对象本身）。
    """
    width = max(1, int(width))
    height = max(1, int(height))
    import numpy as np

    out: List[Tuple[Region, Any]] = []
    for r in regions or []:
        if not isinstance(r, Region):
            r = Region.from_dict(r)
        f = r.feather if feather_override is None else int(feather_override)
        alpha = None
        if r.is_mask_shape():
            alpha = _load_mask_alpha(r.mask_path, width, height)
        else:
            box = r.rect_px(width, height, clamp=True)
            if box is not None:
                alpha = _alpha_rect(width, height, box, f)
        if alpha is None:
            alpha = np.zeros((height, width), dtype=np.float32)
        out.append((r, _alpha_to_pil(alpha)))
    return out


# ======================================================================
#  计划 / 描述 / 校验
# ======================================================================
def _parse_name_filter(only_region) -> List[str]:
    """把 only_region 参数解析成名字列表（支持逗号/顿号/分号分隔）。"""
    if not only_region:
        return []
    if isinstance(only_region, (list, tuple, set)):
        raw = [str(x) for x in only_region]
    else:
        s = str(only_region)
        for ch in ("，", "、", ";", "；", "|"):
            s = s.replace(ch, ",")
        raw = s.split(",")
    return [x.strip() for x in raw if x.strip()]


def plan_regions(regions, *, width: int, height: int,
                 only_region="", max_regions: int = 0,
                 strict: bool = True) -> dict:
    """算出"这次到底会跑哪些区域"，不碰 GPU。

    返回 `{"run": [(Region, box_or_None)], "skipped": [(Region, 原因)]}`。
    界面可以先用它做预览/预检，节点用它决定循环范围 —— 两边同一套判定，
    不会出现"界面说要跑 3 个、实际只跑 2 个"。
    """
    width = max(1, int(width))
    height = max(1, int(height))
    wanted = _parse_name_filter(only_region)
    limit = int(max_regions or 0)

    run: List[Tuple[Region, Optional[Tuple[int, int, int, int]]]] = []
    skipped: List[Tuple[Region, str]] = []

    for r in regions or []:
        if not isinstance(r, Region):
            r = Region.from_dict(r)
        label = r.display_name()

        if not r.enabled:
            skipped.append((r, f"区域「{label}」已禁用"))
            continue
        if wanted and label not in wanted and (r.name.strip() or label) not in wanted:
            skipped.append((r, f"区域「{label}」不在 only_region 选择内"))
            continue

        box = None
        if r.is_mask_shape():
            if not r.mask_path or not os.path.exists(r.mask_path):
                skipped.append((r, f"区域「{label}」的蒙版文件不存在"))
                continue
        else:
            box = r.rect_px(width, height, clamp=True)
            if box is None:
                skipped.append((r, f"区域「{label}」的矩形为空"))
                continue

        if limit and len(run) >= limit:
            skipped.append((r, f"区域「{label}」超出 max_regions={limit} 上限"))
            continue

        if strict:
            # 只有"物理上跑不了"的问题才拦下：权重越界/为 0 照样跑（强度按 0
            # 处理就是不改图，用户能在元数据里看到），避免"界面说要跑、
            # 引擎悄悄跳过"这种对不上的行为。
            probs = [p for p in _problems(r, width, height, box=box)
                     if p[0] in ("mask_missing", "empty_rect")]
            if probs:
                skipped.append((r, probs[0][2]))
                continue

        run.append((r, box))
    return {"run": run, "skipped": skipped}


def _problems(r: Region, width: int, height: int,
              box=None) -> List[Tuple[str, str, str]]:
    """内部：返回 `[(code, level, 中文说明), ...]`；level ∈ {"error","warn"}。"""
    out: List[Tuple[str, str, str]] = []
    label = r.display_name()

    # 权重必须落在 [0, 1]（它就是 inpaint 的 strength）
    if r.weight < 0.0 or r.weight > 1.0:
        out.append(("weight", "error",
                    f"区域「{label}」的权重 {r.weight:g} 越界（必须在 0~1 之间）"))
    elif r.weight <= 0.0:
        out.append(("weight", "warn",
                    f"区域「{label}」的权重为 0，重绘强度为 0 等于不生效"))

    if not (r.prompt or "").strip():
        out.append(("prompt", "warn",
                    f"区域「{label}」没有写提示词，将按空提示词重绘"))

    if r.is_mask_shape():
        if not r.mask_path:
            out.append(("mask_missing", "error",
                        f"区域「{label}」是蒙版区域但没有填蒙版文件"))
        elif not os.path.exists(r.mask_path):
            out.append(("mask_missing", "error",
                        f"区域「{label}」的蒙版文件不存在: {r.mask_path}"))
    else:
        if box is None:
            box = r.rect_px(width, height, clamp=True)
        if box is None:
            out.append(("empty_rect", "error",
                        f"区域「{label}」的矩形为空（宽或高为 0，"
                        f"或在 {width}x{height} 上被夹取后没有面积）"))
        else:
            l, t, rr, bb = box
            if (rr - l) * (bb - t) < 64:
                out.append(("too_small", "warn",
                            f"区域「{label}」只有 {rr - l}x{bb - t} 像素，"
                            f"太小了，重绘结果可能不可用"))

    if r.feather < 0:
        out.append(("feather", "warn",
                    f"区域「{label}」的羽化值 {r.feather} 为负，按 0 处理"))

    cn = r.merged_controlnet()
    if cn is not None:
        ip = _s(cn.get("image_path"), "").strip()
        if ip and not os.path.exists(ip):
            out.append(("cn_image", "warn",
                        f"区域「{label}」的 ControlNet 参考图不存在: {ip}"
                        f"（将改用底图）"))
    ad = r.merged_adetailer()
    if ad is not None and not os.path.exists(_adetailer_hint_path(ad)):
        # 只提示，不当错误：检测模型可能放在别的目录，运行时还会再找一次
        out.append(("adetailer", "warn",
                    f"区域「{label}」的 ADetailer 目标「{ad['target']}」"
                    f"本地可能没有检测模型，运行时取不到会跳过"))
    return out


def _adetailer_hint_path(ad: dict) -> str:
    """尽力猜一个检测模型路径（只用于提示，猜不到返回不可能存在的路径）。"""
    try:
        from utils.image_processor import (ADETAILER_MODELS)
    except Exception:
        return "\x00"
    cfg = ADETAILER_MODELS.get(ad.get("target"))
    if not cfg:
        return "\x00"
    try:
        from utils.config import AppConfig  # noqa: F401  (保持依赖显式)
    except Exception:
        pass
    try:
        import utils.image_processor as ip
        return ip._resolve_local_adetailer_model(cfg) or ip._get_adetailer_model_path(cfg)
    except Exception:
        return "\x00"


def validate_regions(regions, width: int, height: int) -> List[str]:
    """校验区域定义，返回**中文问题列表**（空列表 = 没问题）。

    只报告、不修改、不抛异常：界面可以直接把这份列表贴到提示条上。
    """
    width = max(1, int(width or DEFAULT_WIDTH))
    height = max(1, int(height or DEFAULT_HEIGHT))
    msgs: List[str] = []
    items = list(regions or [])
    if not items:
        return ["没有定义任何区域（regions 为空）"]
    for r in items:
        if not isinstance(r, Region):
            r = Region.from_dict(r)
        for _code, _level, msg in _problems(r, width, height):
            msgs.append(msg)
    if not any(_b(getattr(r, "enabled", True), True) for r in items):
        msgs.append("所有区域都被禁用了，这次不会做任何重绘")
    return msgs


def describe(regions) -> str:
    """一行中文摘要，如 `3 个区域：脸部(0.55) 衣服(0.45) 背景(0.35)`。"""
    items = []
    for r in regions or []:
        items.append(r if isinstance(r, Region) else Region.from_dict(r))
    if not items:
        return "0 个区域"
    parts = []
    for r in items:
        tag = r.display_name()
        if r.is_mask_shape():
            tag += "(蒙版)"
        if not r.enabled:
            tag += "[禁用]"
        parts.append(f"{tag}({r.weight:.2f})")
    return f"{len(items)} 个区域：" + " ".join(parts)


# ======================================================================
#  引擎主体
# ======================================================================
def _resolve_seed(seed) -> int:
    """复用采样节点的种子语义（-1/None → 随机）。"""
    try:
        from nodes.sampling_nodes import _resolve_seed as _rs
        return int(_rs(seed))
    except Exception:
        import random
        try:
            v = int(seed)
        except Exception:
            v = -1
        return v if v >= 0 else random.randint(1, 2_147_483_647)


def _generator(device: str, seed: int):
    """复用采样节点的 generator 构造（设备不一致时自动退 CPU）。"""
    try:
        from nodes.sampling_nodes import _generator as _g
        return _g(device, seed)
    except Exception:
        import torch
        try:
            return torch.Generator(device=device).manual_seed(int(seed))
        except Exception:
            return torch.Generator().manual_seed(int(seed))


class _TextCond:
    """把裸字符串包装成"能被 `adapter.sample_kwargs` 认出文本"的对象。

    为什么需要它：非 Compel 架构（Anima/Cosmos）的 `sample_kwargs` 是
    `getattr(conditioning, "text", "")` 取文本的 —— 直接传 `str` 会拿到空
    提示词（功能静默失效）。这里给一个最小适配器，既让字符串被正确识别，
    也让 `ready` 为 False，从而走"原始提示词"分支（由管线自己编码）。
    """

    __slots__ = ("text", "negative_text", "embeds")

    def __init__(self, text: str, negative_text: str = ""):
        self.text = text or ""
        self.negative_text = negative_text or ""
        self.embeds = None

    @property
    def ready(self) -> bool:
        return False


def _encode_prompt(ai, text: str, negative: str) -> Any:
    """尽力把区域提示词编码成 Conditioning；失败退回裸文本。

    复用 `nodes.text_nodes._encode`（与 TextEncode 节点**同一条**编码路径），
    这样区域提示词的权重语法（compel `(word:1.2)`）、CLIP Skip、缓存都和
    主流程一致，不会出现"区域里的权重语法不生效"。
    """
    text = text or ""
    negative = negative or ""
    if ai is not None:
        try:
            from nodes.text_nodes import _encode
            cond = _encode(_AiProxy(ai), text, negative)
            if cond is not None and getattr(cond, "embeds", None) is not None:
                if not cond.text:
                    cond.text = text
                if not getattr(cond, "negative_text", ""):
                    cond.negative_text = negative
                return cond
        except Exception as e:
            logger.info(f"ℹ️ 区域提示词编码不可用，改用原始提示词: {e}")
    return _TextCond(text, negative)


class _AiProxy:
    """只带 `ai` 属性的最小上下文，喂给 `nodes.text_nodes._encode`。"""

    __slots__ = ("ai",)

    def __init__(self, ai):
        self.ai = ai


def _mask_bbox(mask) -> Optional[Tuple[int, int, int, int]]:
    """蒙版里所有 >127 像素的包围盒；全黑返回 None。"""
    try:
        import numpy as np
        arr = np.asarray(mask.convert("L"))
        ys, xs = np.nonzero(arr > 127)
        if xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    except Exception:
        return None


def _mask_has_area(mask) -> bool:
    return _mask_bbox(mask) is not None


def _composite(base, redrawn, mask):
    """`out = base*(1-m) + redrawn*m`（m 为羽化蒙版的浮点值）。

    全部在本地副本上做，**不修改传入的图**。
    """
    import numpy as np

    b = base.convert("RGB")
    d = redrawn.convert("RGB")
    if d.size != b.size:
        d = d.resize(b.size, _pil_resample_lanczos())
    m = mask.convert("L")
    if m.size != b.size:
        m = m.resize(b.size, _pil_resample_bilinear())

    ba = np.asarray(b, dtype=np.float32)
    da = np.asarray(d, dtype=np.float32)
    ma = np.asarray(m, dtype=np.float32)[..., None] / 255.0
    out = ba * (1.0 - ma) + da * ma
    from PIL import Image
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype("uint8"), "RGB")


def _pil_resample_lanczos():
    from PIL import Image
    return getattr(Image, "Resampling", Image).LANCZOS


def _pil_resample_bilinear():
    from PIL import Image
    return getattr(Image, "Resampling", Image).BILINEAR


def _crop_region(image, mask, pad_ratio: float = 0.15, min_side: int = 128):
    """按蒙版包围盒 + 留白裁出小图与该蒙版的局部图。

    返回 `(crop_image, crop_mask, box)`；找不到有效包围盒返回
    `(None, None, None)`。
    """

    bbox = _mask_bbox(mask)
    if bbox is None:
        return None, None, None
    l, t, r, b = bbox
    w, h = image.size
    pad = int(round(max(r - l, b - t) * max(0.0, float(pad_ratio or 0.0))))
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(w, r + pad)
    b = min(h, b + pad)
    if r - l <= 0 or b - t <= 0:
        return None, None, None
    # 太小的块送进检测/扩散没有意义，向下对齐到 8 的倍数
    box = (l, t, r, b)
    return (image.convert("RGB").crop(box),
            mask.convert("L").crop(box),
            box)


def _region_adetailer(cur, model, adapter, r: Region, mask,
                      prompt: str, negative: str, *, cancel=None,
                      status=None, log=None) -> Optional[Any]:
    """在**区域包围盒内**跑一次 ADetailer，返回贴回后的整图；失败返回 None。

    为什么要裁剪：ADetailer 的检测器只接受整图，没有"只看蒙版"的入参。
    先把区域框裁出来、在小图上检测重绘、再按蒙版贴回，才是真正的"局部
    ADetailer"（否则它会修到区域外的脸，用户会觉得"我没让它改那里"）。

    `prompt` 只用于日志：ADetailer 自己带提示词模板（与 ADetailer 节点一致），
    这样"局部 ADetailer"和"全局 ADetailer"对同一张脸的结果是一致的。
    """
    ad = r.merged_adetailer()
    if ad is None:
        return None
    try:
        from utils.image_processor import process_adetailer
    except Exception as e:
        if log:
            log(f"⚠️ 区域「{r.display_name()}」的 ADetailer 不可用（跳过）: {e}")
        return None

    inpaint_pipe = adapter.pipe(model, "inpaint")
    i2i_pipe = adapter.pipe(model, "img2img")
    if inpaint_pipe is None and i2i_pipe is None:
        if log:
            log(f"⚠️ 区域「{r.display_name()}」没有 inpaint / img2img 管线，"
                f"ADetailer 跳过")
        return None

    crop, crop_mask, box = _crop_region(cur, mask)
    if crop is None:
        if log:
            log(f"⚠️ 区域「{r.display_name()}」的蒙版为空，ADetailer 跳过")
        return None
    if cancel and cancel():
        _raise_cancelled()

    target = _s(ad.get("target"), "二次元脸部")
    strength = _f(ad.get("strength"), 0.35)
    detail_size = _i(ad.get("detail_size"), 512)
    padding = _f(ad.get("padding"), 0.15)
    mask_blur = _i(ad.get("mask_blur"), 8)

    if status:
        status(f"🩹 区域「{r.display_name()}」ADetailer 精修 [{target}]")
    # 路径裁剪：把提示词留给 ADetailer 自己的模板（与 ADetailer 节点一致）
    neg = (negative or "").strip()
    try:
        repaired = process_adetailer(
            crop, inpaint_pipe, "", neg,
            strength=strength, target=target,
            padding_ratio=padding, mask_blur=mask_blur,
            detail_size=detail_size,
            img2img_pipe=i2i_pipe)
    except Exception as e:
        if log:
            log(f"⚠️ 区域「{r.display_name()}」ADetailer 失败（跳过）: "
                f"{type(e).__name__}: {e}")
        return None
    if repaired is None:
        if log:
            log(f"⚠️ 区域「{r.display_name()}」ADetailer 返回空结果（跳过）")
        return None

    from workflow.values import as_image
    small = as_image(repaired)
    if small is None:
        if log:
            log(f"⚠️ 区域「{r.display_name()}」ADetailer 结果无法识别（跳过）")
        return None

    # 贴回：把结果放大回裁剪框尺寸，用**该区域的羽化蒙版**贴回整图
    l, t, rr, bb = box
    small = small.convert("RGB").resize((rr - l, bb - t),
                                        _pil_resample_lanczos())
    patch_mask = mask.convert("L").crop(box)
    base = cur.convert("RGB")
    tmp = base.copy()
    tmp.paste(small, (l, t))
    return _composite(base, tmp, _paste_mask_full(base.size, patch_mask,
                                                  (l, t)))


def _paste_mask_full(size, patch_mask, offset):
    """把局部蒙版放回整图坐标系（其余为 0）。"""
    from PIL import Image
    full = Image.new("L", size, 0)
    full.paste(patch_mask.convert("L"), offset)
    return full


def _region_ip_adapter_refs(r: Region, mask):
    """本区域的参考图列表（没有配就返回空表）。

    配置形状（两种都认，见 `Region.merged_ip_adapter`）::

        "ip_adapter": {"refs": [{"role":"style","path":"...","weight":0.6}]}
        "ip_adapter": {"role":"identity","path":"...","weight":0.8}

    每条参考图都会带上**本区域蒙版**（`mask`）作为它的区域掩码 —— 这就是
    "这张参考图只影响这块区域"的实现方式（参考图抠图 + 注意力掩码）。
    """
    cfg = r.merged_ip_adapter()
    if not cfg:
        return []
    from core import ipa_multi as ipa_mod
    refs = ipa_mod.coerce_references(cfg.get("refs") or [])
    out = []
    for ref in refs:
        if ref.mask is None and not ref.mask_path and mask is not None:
            ref.mask = mask
        out.append(ref)
    return out


def _apply_region_ip_adapter(ai, kw: dict, r: Region, mask, cur, *,
                             status=None, log=None) -> dict:
    """把该区域的 IP-Adapter 参考图塞进采样参数；返回 `{warning, refs, scales}`。

    设计取舍（诚实说明）
    --------------------
    · 区域重绘本身是**顺序蒙版重绘**（一个区域一次采样），所以这里天然是
      "每个区域一套参考图"，而不是"所有参考图合成一次"。
    · 参考图数量会被补齐/截断到当前 UNet 上**已安装的适配器数量**
      （`ai.ipa_adapter_count()`），因为 diffusers 会严格校验长度。
      补齐用 1x1 黑图 + 强度 0 → 该适配器被跳过，不污染画面。
    · 采样完把该管线的 scale 恢复成全局值，避免"下一个区域偷偷继承上一个
      区域的强度"。
    · 任何失败都只 **warning**：局部参考图是增强项，取不到就退回普通蒙版
      重绘，绝不让整区域失败（与局部 ControlNet 的策略一致）。
    """
    info = {"warning": "", "refs": 0, "scales": []}
    count = 0
    try:
        count = int(ai.ipa_adapter_count()) if ai is not None else 0
    except Exception:
        count = 0
    if count <= 0:
        return info
    if ai is None or not getattr(ai, "ip_adapter_loaded", False):
        return info

    refs = _region_ip_adapter_refs(r, mask)
    try:
        builder = getattr(ai, "build_ip_adapter_kwargs", None)
        if not callable(builder):
            return info
        extra, meta = builder(refs if refs else None,
                              width=cur.width, height=cur.height,
                              has_masks=bool(refs))
    except Exception as e:
        info["warning"] = (f"区域「{r.display_name()}」的局部参考图准备失败"
                           f"（已退回普通重绘）: {type(e).__name__}: {e}")
        return info

    if not extra:
        # 没配参考图但全局装了 IPA：必须给占位图，否则整个采样会抛
        # TypeError: 'NoneType' object is not iterable（见 sampling_nodes 的说明）
        try:
            from PIL import Image as _PIL
            extra = {"ip_adapter_image":
                     [_PIL.new("RGB", (1, 1), (0, 0, 0)) for _ in range(count)]}
            meta = {"scales": [0.0] * count, "notes": ["占位（本区域无参考图）"]}
        except Exception:
            return info

    kw.update(extra)
    scales = list(meta.get("scales") or [])
    info["refs"] = len(refs)
    info["scales"] = scales

    # 只改**本区域采样用的那条管线**的强度，并且**记下原值**：
    # 下一个区域若没有自己的参考图，必须回到全局强度，不能偷偷继承。
    # ⚠️ 恢复动作在这里立刻做也可以（scale 是在采样前写进 attn processor 的），
    # 但为了不依赖 diffusers 内部读取时机，仍由调用方在采样后调用
    # `restore_region_ip_adapter(kw)` 恢复 —— 两条路都留着。
    restore = []
    try:
        targets = [kw.get("_ipa_pipe") or getattr(ai, "inpaint_pipe", None)]
        setter = getattr(ai, "_set_multi_scale", None)
        if callable(setter) and scales:
            # 读不到管线自己的 scale 时，退回管理器记着的全局强度 ——
            # 总之必须有一个"恢复目标"，否则后一个区域会继承前一个的强度。
            fallback = list(getattr(ai, "_ipa_scales", None) or [])
            for p in targets:
                if p is None:
                    continue
                before = read_ipa_scales(p) or fallback or None
                restore.append((p, before))
            setter(scales, pipes=targets)
    except Exception as e:
        info["warning"] = (f"区域「{r.display_name()}」设置参考图强度失败"
                           f"（沿用上一步）: {e}")

    kw["_ipa_restore"] = restore
    for note in (meta.get("notes") or []):
        if log:
            log(f"ℹ️ 区域「{r.display_name()}」参考图: {note}")
    if refs and status:
        status(f"🎭 区域「{r.display_name()}」参考图 {len(refs)} 条"
               f"（只作用于本区域）")
    return info


def read_ipa_scales(pipe):
    """读出某条管线当前的 per-adapter scale（读不到返回 None）。"""
    if pipe is None:
        return None
    unet = getattr(pipe, "unet", None)
    procs = getattr(unet, "attn_processors", None)
    if not procs:
        return None
    for name, proc in procs.items():
        if "attn2" not in name:
            continue
        scale = getattr(proc, "scale", None)
        if scale is None:
            continue
        if isinstance(scale, (list, tuple)):
            return [float(s) for s in scale]
        try:
            return [float(scale)]
        except Exception:
            continue
    return None


def restore_region_ip_adapter(kw: dict, ai=None) -> int:
    """区域采样结束后把 IPA 强度恢复成采样前的值，返回恢复的管线数。

    为什么必须恢复：scale 是写在注意力处理器上的**全局状态**，
    区域之间共用同一条 inpaint 管线。不恢复的话，第一个区域的参考图强度
    会被后面所有区域继承（"我只给区域 A 配了参考图，结果 B、C 也变了"）。
    """
    restore = kw.pop("_ipa_restore", None) or []
    n = 0
    for pipe, before in restore:
        if pipe is None:
            continue
        try:
            setter = getattr(pipe, "set_ip_adapter_scale", None)
            if not callable(setter):
                continue
            if before is None:
                continue
            payload = before[0] if len(before) == 1 else before
            setter(payload)
            pipe._ai_ipa_scale = max(before) if before else 0.0
            n += 1
        except Exception as e:
            logger.warning(f"⚠️ 恢复区域 IPA 强度失败: {e}")
    return n


class _DeviceHolder:
    """保存本次运行的 device，供 `_generator(device, seed)` 使用。

    为什么用模块级小对象而不是把 device 一路透传：`_region_sampler` 是内部
    函数，签名已经很长了；device 在一次 `apply_regions` 调用内是常量，
    放在这里既省参数又不会跨调用串味（每次 apply_regions 开头覆盖它）。
    """

    device: str = "cuda"


_ctx_device_holder = _DeviceHolder()


def _region_sampler(model, adapter, ai, r: Region, cur, mask, *, steps: int,
                    cfg: float, seed: int, global_negative: str,
                    status=None, log=None):
    """跑一个区域的采样，返回该区域的"重绘结果"（整图尺寸，仅在蒙版内有意义）。

    返回 `(redrawn_image, warning_or_None)`；无法完成时抛异常（由调用方记进
    region_errors）。
    """
    import torch

    neg_text = (r.negative or "").strip() or (global_negative or "")
    pos = _encode_prompt(ai, r.prompt or "", neg_text)

    cn = r.merged_controlnet()
    cn_kwargs: dict = {}
    warning: Optional[str] = None

    if cn is not None:
        prepared, cn_kwargs, cn_warn = _prepare_region_controlnet(
            ai, adapter, model, r, cur, cn, status=status, log=log)
        if cn_warn:
            warning = cn_warn

    kw = adapter.sample_kwargs(pos, steps=steps, cfg=cfg, seed=seed,
                               width=cur.width, height=cur.height)

    # 负面条件：优先用"编码好的局部负面词"，否则用全局负面的原文。
    # ⚠️ 二者**只能给一个**：diffusers 会直接报
    # `Cannot forward both negative_prompt and negative_prompt_embeds`。
    # 所以给 embeds 时必须把 sample_kwargs 里可能已经塞进来的 negative_prompt
    # 清掉（非 Compel 架构会塞原创文本，而那条路上本来也没有 embeds）。
    if neg_text:
        neg_emb = getattr(pos, "negative_embeds", None)
        if neg_emb is not None and getattr(adapter.caps, "negative", True):
            kw.pop("negative_prompt", None)
            kw["negative_prompt_embeds"] = neg_emb
            np_ = getattr(pos, "negative_pooled", None)
            if np_ is not None:
                kw["negative_pooled_prompt_embeds"] = np_
        else:
            kw.pop("negative_prompt_embeds", None)
            kw.pop("negative_pooled_prompt_embeds", None)
            kw["negative_prompt"] = neg_text
    else:
        # 没有局部也没有全局负面词：保持"原创文本"形态（别留一个 None 让管线报错）
        kw.pop("negative_prompt_embeds", None)
        kw.pop("negative_pooled_prompt_embeds", None)
        kw["negative_prompt"] = ""

    if cn_kwargs:
        pipe = cn_kwargs.pop("_pipe")
        kw.update(cn_kwargs)
        if status:
            status(f"🧩 区域「{r.display_name()}」ControlNet "
                   f"[{cn['kind']}] 引导生成")
    else:
        pipe = adapter.pipe(model, "inpaint")
        if pipe is None:
            raise RuntimeError(
                f"模型 {getattr(model, 'describe', lambda: model)()} "
                f"没有可用的 inpaint 管线")
        kw["image"] = cur
        kw["mask_image"] = mask
        kw["strength"] = max(0.0, min(1.0, float(r.weight)))

    # ── 局部 IP-Adapter（多参考图按区域生效）─────────────────────────
    # 怎么做到"只影响该区域"，两步缺一不可：
    #   1) 参考图侧：按本区域蒙版把参考图抠出来（区域外内容不进 CLIP）；
    #   2) 注意力侧：把本区域蒙版当 ip_adapter_masks 传进去，
    #      限制该参考图只在蒙版范围内参与注意力（=真正的"局部"）。
    # ⚠️ 只要 UNet 上装了 IPA，每次调用都必须给 ip_adapter_image，
    #    否则 diffusers 会在 UNet 里抛一个与 IPA 无关的 TypeError，
    #    所以没有配参考图的区域也要送占位图（强度 0）。
    ipa_info = _apply_region_ip_adapter(ai, kw, r, mask, cur,
                                       status=status, log=log)
    warning = warning or ipa_info.get("warning") or None

    # 尺寸交给输入图决定（与 Inpaint 节点一致，避免被拉成别的比例）
    kw.pop("width", None)
    kw.pop("height", None)
    kw["generator"] = _generator(getattr(_ctx_device_holder, "device", "cuda"),
                                 seed)
    kw = _sanitize_call_kwargs(pipe, kw)

    with torch.inference_mode():
        try:
            res = pipe(**kw)
        finally:
            # 采样一结束就把 IPA 强度还回去：不还的话，本区域的参考图强度
            # 会被**后面所有区域**继承（"我只给 A 配了参考图，B/C 也变了"）。
            if kw.get("_ipa_restore"):
                restore_region_ip_adapter(kw, ai)

    imgs = getattr(res, "images", None)
    if imgs is None:
        raise RuntimeError("区域重绘管线没有返回图像")
    out = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
    if out is None:
        raise RuntimeError("区域重绘管线返回了空的图像列表")

    from workflow.values import as_image
    pil = as_image(out)
    if pil is None:
        pil = _tensor_to_pil(model, out)
    if pil is None:
        raise RuntimeError(
            f"无法把区域重绘结果转成图像（{type(out).__name__}）")
    return pil.convert("RGB"), warning


def _prepare_region_controlnet(ai, adapter, model, r: Region, cur, cn: dict,
                               *, status=None, log=None):
    """准备一个区域的局部 ControlNet。

    返回 `(是否成功, 调用参数, warning_or_None)`。**任何失败都只 warning**：
    局部 CN 是增强项，取不到就退回普通蒙版重绘，绝不让整区域失败。
    """
    warn: Optional[str] = None
    if not getattr(adapter.caps, "controlnet", True):
        return False, {}, (f"区域「{r.display_name()}」的局部 ControlNet 未生效："
                           f"当前架构不支持 ControlNet")
    prepare = getattr(ai, "prepare_controlnet", None)
    if ai is None or not callable(prepare):
        return False, {}, (f"区域「{r.display_name()}」的局部 ControlNet 未生效："
                           f"没有可用的模型管理器（ctx.ai 为空）")
    kind = _s(cn.get("kind"), "openpose")
    try:
        if status:
            status(f"🧩 区域「{r.display_name()}」准备 ControlNet [{kind}]")
        prepare(kind)
    except Exception as e:
        return False, {}, (f"区域「{r.display_name()}」的局部 ControlNet 未生效："
                           f"装载 {kind} 失败（{type(e).__name__}: {e}）")

    pipe = getattr(ai, "controlnet_pipe", None)
    if pipe is None:
        return False, {}, (f"区域「{r.display_name()}」的局部 ControlNet 未生效："
                           f"模型管理器没有 controlnet_pipe")

    # 控制图：优先用区域自带的参考图，其次用当前图（tile/canny 这类从图推图）
    img = None
    path = _s(cn.get("image_path"), "").strip()
    if path:
        try:
            from PIL import Image
            with Image.open(path) as im:
                img = im.convert("RGB")
                img.load()
        except Exception as e:
            warn = (f"区域「{r.display_name()}」的 ControlNet 参考图读取失败，"
                    f"已改用当前图（{e}）")
    if img is None:
        img = cur.convert("RGB")
    if img.size != cur.size:
        img = img.resize(cur.size, _pil_resample_lanczos())

    # 走管理器的控制图生成（canny/depth/openpose 的特征提取在那边有缓存）
    getter = getattr(ai, "get_control_image", None)
    if callable(getter):
        try:
            got = getter(img, kind)
            if got is not None:
                from workflow.values import as_image
                got = as_image(got)
                if got is not None:
                    img = got.convert("RGB")
                    if img.size != cur.size:
                        img = img.resize(cur.size, _pil_resample_lanczos())
        except Exception as e:
            logger.info(f"ℹ️ get_control_image 失败，沿用原图作控制图: {e}")

    weight = _f(cn.get("weight"), 0.8)
    start = _f(cn.get("start"), 0.0)
    end = _f(cn.get("end"), 1.0)
    call_kwargs = {
        "_pipe": pipe,
        "image": img,
        "controlnet_conditioning_scale": float(weight),
    }
    if end < start:
        start, end = 0.0, 1.0
    call_kwargs["control_guidance_start"] = float(start)
    call_kwargs["control_guidance_end"] = float(end)
    return True, call_kwargs, warn


def _sanitize_call_kwargs(pipe, kw: dict) -> dict:
    """按管线的 `__call__` 签名剔除不认识的键（不同 diffusers 版本差异大）。

    为什么要做：多传一个未知 kwargs 会让管线抛 TypeError，而"某个版本多支持
    一个参数"这种事完全不该让用户的重绘失败。签名取不到时原样返回。
    """
    try:
        import inspect
        sig = inspect.signature(pipe.__call__)
    except Exception:
        return kw
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kw
    unknown = [k for k in kw if k not in params]
    if not unknown:
        return kw
    out = {k: v for k, v in kw.items() if k in params}
    logger.info(f"ℹ️ 已按管线签名忽略不支持的参数: {', '.join(sorted(unknown))}")
    return out


def _tensor_to_pil(model, tensor):
    """兜底：管线返回张量时自己解码成 PIL（与 image_nodes 的策略一致）。"""
    try:
        import torch
    except Exception:
        return None
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
        return None
    t = tensor
    if t.shape[1] not in (3, 4):
        vae = (getattr(model, "vae", None)
               or getattr(getattr(model, "pipe", None), "vae", None))
        if vae is None:
            return None
        with torch.inference_mode():
            if hasattr(vae.config, "scaling_factor"):
                t = t / vae.config.scaling_factor
            t = vae.decode(t.to(device=vae.device, dtype=vae.dtype)).sample
    if float(t.min()) < 0:
        t = (t / 2 + 0.5).clamp(0, 1)
    from PIL import Image
    arr = (t[0].permute(1, 2, 0).float().cpu().numpy() * 255).round()
    arr = arr.astype("uint8")
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], "L")
    return Image.fromarray(arr[:, :, :3], "RGB")


def apply_regions(ai, model, base_image, regions, *,
                  global_negative: str = "", steps: int = 24, cfg: float = 7.0,
                  seed: int = -1,
                  progress: Optional[Callable[[int, int], None]] = None,
                  cancel: Optional[Callable[[], bool]] = None,
                  status: Optional[Callable[[str], None]] = None,
                  log: Optional[Callable[[str], None]] = None,
                  device: str = "",
                  only_region="", max_regions: int = 0,
                  feather: Optional[int] = None,
                  encode_prompt: bool = True) -> Tuple[Any, dict]:
    """在 base_image 上按区域逐个做蒙版重绘，返回 `(结果图, 元数据)`。

    对每个 enabled 且蒙版非空的区域：
      1. `negative = region.negative or global_negative`
      2. 若 `region.controlnet` 给了且 kind 可用 → 为该区域准备 ControlNet
         （复用 `ctx.ai.prepare_controlnet`；拿不到就 warning 跳过局部 CN）
      3. 用 inpaint 管线 + 该区域蒙版 + `region.prompt` 跑一次，
         `strength = region.weight`
      4. 若 `region.adetailer` 给了 → 在该区域的蒙版范围内跑 ADetailer
      5. 把结果按羽化蒙版合成回当前图

    · 每个区域之间调用 `cancel()`、`progress(i, n)`、`status()`；
    · **单个区域失败不会让整件事失败**：记进 `metadata["region_errors"]`
      并继续下一个；
    · seed：`base_seed + 区域序号`（`seed=-1` 时随机一次并记录在元数据里）。

    参数
    ----
    progress : fn(i, n)         第 i 个区域（1 基）共 n 个
    cancel   : fn() -> bool     返回 True 时抛 Cancelled
    status   : fn(msg)          中文状态
    log      : fn(msg)          日志/告警（warning 也会写进返回值）

    返回的元数据至少含：regions(数量) / 每个区域的 name,weight,shape,applied,
    warning / region_errors / seed / seconds。
    """
    t0 = time.time()
    from workflow.values import as_image
    import workflow.adapters as adapters_mod

    if model is None:
        raise ValueError("缺少 model（区域重绘需要 inpaint 管线）")
    base = as_image(base_image)
    if base is None:
        raise ValueError("缺少 base_image（区域重绘需要一张底图）")
    base = base.convert("RGB").copy()

    if device:
        _ctx_device_holder.device = device

    items = coerce_regions(regions)
    width, height = base.width, base.height

    adapter = getattr(model, "adapter", None)
    if adapter is None:
        adapter = adapters_mod.adapter_for(model)
    caps = adapters_mod.capabilities_of(model)
    # 能力门禁：架构没有 inpaint 就直接抛，不带病往下跑
    caps.require("inpaint")
    if adapter.pipe(model, "inpaint") is None:
        raise adapters_mod.UnsupportedFeatureError(
            "局部重绘", getattr(model, "arch_id", "") or "",
            "该架构没有可用的 inpaint 管线")

    plan = plan_regions(items, width=width, height=height,
                        only_region=only_region, max_regions=max_regions)
    todo = plan["run"]
    n = len(todo)

    base_seed = _resolve_seed(seed)
    if isinstance(seed, (int, float)) and int(seed) < 0:
        if log:
            log(f"🎲 区域重绘随机种子: {base_seed}")

    # 区域定义里的羽化可以被 apply_regions(feather=) 统一下压（UI 的全局开关）。
    # ⚠️ 只写进"本次运行的旁表"，**不修改调用方传进来的 Region 对象** ——
    # 那些对象往往就是界面编辑器持有的实例，改一个字段会让界面上显示的值
    # 与用户设置的不一致（而且用户下次保存工作流就把覆盖值存进去了）。
    feather_override = None if feather is None else max(0, int(feather))
    masks = dict()
    for r, _box in todo:
        masks[id(r)] = None
    _pairs = region_masks([r for r, _b in todo], width, height,
                          feather_override=feather_override)
    for (r, m) in _pairs:
        masks[id(r)] = m

    if status:
        if n == 0:
            status("🤷 没有可执行的区域（全部禁用/空蒙版/被过滤）")
        else:
            status(f"🎨 区域精修开始：{n} 个区域 · {describe([r for r, _b in todo])}")

    cur = base
    applied = 0
    details: List[dict] = []
    errors: List[str] = []
    warnings: List[str] = []

    # 被跳过的区域也要如实写进元数据（用户才知道"为什么这个区域没跑"）
    for r, why in plan["skipped"]:
        if log:
            log(f"ℹ️ {why}")
        warnings.append(why)

    for i, (r, _box) in enumerate(todo, start=1):
        if cancel and cancel():
            _raise_cancelled()
        if progress:
            try:
                progress(i, n)
            except Exception:
                pass

        label = r.display_name()
        mask = masks.get(id(r))
        info = {"name": r.name, "label": label, "weight": float(r.weight),
                "shape": r.shape, "applied": False, "warning": "",
                "seed": int(base_seed) + i - 1, "feather": int(r.feather or 0),
                "seconds": 0.0}
        t_reg = time.time()

        if mask is None or not _mask_has_area(mask):
            info["warning"] = "蒙版为空，跳过"
            warnings.append(f"区域「{label}」的蒙版为空，已跳过")
            if log:
                log(f"ℹ️ 区域「{label}」的蒙版为空，已跳过")
            info["seconds"] = round(time.time() - t_reg, 3)
            details.append(info)
            continue

        # 未开启编码（或没有 ai）时，区域提示词按原文送 —— 与 TextEncode 一致
        if not encode_prompt:
            ai = None
        region_seed = int(base_seed) + i - 1

        if status:
            status(f"🩹 区域 {i}/{n} 「{label}」"
                   f" 强度 {r.weight:.2f} · seed {region_seed}")
        if log:
            log(f"🩹 区域 {i}/{n} 「{label}」shape={r.shape} "
                f"weight={r.weight:.2f} prompt={(r.prompt or '')[:60]!r}")

        try:
            redrawn, warn = _region_sampler(
                model, adapter, ai if encode_prompt else None, r, cur, mask,
                steps=int(steps), cfg=float(cfg), seed=region_seed,
                global_negative=global_negative, status=status, log=log)
            if warn:
                info["warning"] = warn
                warnings.append(warn)
                if log:
                    log(f"⚠️ {warn}")

            # 局部 ADetailer（拿不到只 warning，跳过）
            #
            # ⚠️ 必须传 `redrawn`（本区域刚重绘出来的图），不能传 `cur`
            # （重绘**之前**的图）。ADetailer 会裁剪该区域 → 处理 → 按蒙版贴回，
            # 如果裁的是重绘前的图，贴回来就把区域提示词的重绘结果**整个盖掉**：
            # 实测右半区域「a big blue banana」weight 0.7 时差异从 43.6 掉到 1.1，
            # 看起来就像"开了局部修脸，区域提示词就失效了"。
            if r.merged_adetailer() is not None:
                try:
                    patched = _region_adetailer(
                        redrawn, model, adapter, r, mask,
                        prompt=r.prompt or "", negative=r.negative or "",
                        cancel=cancel, status=status, log=log)
                    if patched is not None:
                        redrawn = patched
                except Cancelled:
                    raise
                except Exception as e:
                    msg = (f"区域「{label}」的 ADetailer 失败（已跳过）: "
                           f"{type(e).__name__}: {e}")
                    info["warning"] = (info["warning"] + "；" + msg).strip("；")
                    warnings.append(msg)
                    if log:
                        log(f"⚠️ {msg}")

            if cancel and cancel():
                _raise_cancelled()

            # 合成回当前图（当前图是上一轮区域的结果 → 区域之间自然叠加）
            cur = _composite(cur, redrawn, mask)
            info["applied"] = True
            applied += 1
        except Cancelled:
            raise
        except Exception as e:
            msg = f"区域「{label}」重绘失败: {type(e).__name__}: {e}"
            errors.append(msg)
            info["warning"] = msg
            if log:
                log(f"❌ {msg}")
            logger.warning(f"⚠️ {msg}", exc_info=True)
        finally:
            info["seconds"] = round(time.time() - t_reg, 3)
            details.append(info)

    if cancel and cancel():
        _raise_cancelled()
    if progress:
        try:
            progress(n, n)
        except Exception:
            pass

    seconds = round(time.time() - t0, 3)
    meta = {
        "regions": len(items),
        "regions_planned": n,
        "regions_applied": applied,
        "regions_skipped": len(plan["skipped"]),
        "region_details": details,
        "region_errors": errors,
        "region_warnings": warnings,
        # 实际参与本次运行的区域对象（界面可以用它画"哪几个生效了"）
        "regions_run": [r for r, _b in todo],
        "seed": int(base_seed),
        "steps": int(steps),
        "cfg": float(cfg),
        "method": "sequential_masked_regional_redraw",
        "seconds": seconds,
        "width": cur.width,
        "height": cur.height,
        "device": _ctx_device_holder.device,
    }
    if status:
        if errors:
            status(f"⚠️ 区域精修完成：{applied}/{n} 个区域生效，"
                   f"{len(errors)} 个失败 · {seconds}s")
        else:
            status(f"✅ 区域精修完成：{applied}/{n} 个区域生效 · {seconds}s")
    if log:
        log(f"✅ 区域精修完成 {describe(items)} → 生效 {applied}/{n} · {seconds}s")

    # 结果图也复制一份：调用方拿到的图与 base/中间态无别名关系
    return cur.convert("RGB").copy(), meta
