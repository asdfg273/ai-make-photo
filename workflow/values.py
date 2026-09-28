# workflow/values.py
"""端口上流动的**具体值类型**。

约定（节点作者必须遵守）：
    MODEL          -> ModelRef
    VAE            -> VaeRef
    LORA           -> LoraRef
    TEXT_COND      -> Conditioning
    LATENT         -> Latent
    IMAGE          -> PIL.Image.Image (RGB)
    MASK           -> PIL.Image.Image ("L")，255=重绘区域
    CONTROL_IMAGE  -> ControlImage
    SCHEDULER      -> 任意 scheduler 对象（由适配器产出）
    NUMBER         -> int / float
    BOOLEAN        -> bool
    METADATA       -> dict

这些类型都是**普通数据容器**，不持有 GPU 状态（除了 tensor 字段本身），
这样缓存、序列化和"只重跑某个节点"都好做。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelRef:
    """一个已就绪的底模。`pipe` 由模型族适配器负责装配。"""

    name: str = ""
    path: str = ""
    arch_id: str = "sd15"
    pipe: Any = None                  # txt2img pipeline
    img2img_pipe: Any = None
    inpaint_pipe: Any = None
    adapter: Any = None               # workflow.adapters.ModelAdapter
    vae: Any = None
    scheduler: Any = None
    is_sdxl: bool = False
    clip_skip: int = 0
    dtype: Any = None
    loaded: bool = False
    meta: dict = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.name} [{self.arch_id}]"


@dataclass
class VaeRef:
    name: str = ""
    path: str = ""
    vae: Any = None

    def describe(self) -> str:
        return self.name or "vae"


@dataclass
class LoraRef:
    name: str = ""
    path: str = ""
    weight: float = 1.0
    applied: bool = False

    def describe(self) -> str:
        return f"{self.name}@{self.weight}"


@dataclass
class Conditioning:
    """文本条件。

    `embeds` 是**已经编码好**的张量（compel 或管线自带编码器产出）；
    保留 `text` 是为了元数据与调试时能看清到底送了什么。
    """

    embeds: Any = None
    pooled: Any = None
    negative_embeds: Any = None
    negative_pooled: Any = None
    text: str = ""
    negative_text: str = ""

    @property
    def ready(self) -> bool:
        return self.embeds is not None


@dataclass
class Latent:
    """潜空间张量。"""

    tensor: Any = None
    is_noise: bool = False

    def shape(self):
        try:
            return tuple(self.tensor.shape)
        except Exception:
            return ()


@dataclass
class ControlImage:
    """ControlNet 的条件图（已经预处理过的）。"""

    image: Any = None                 # PIL.Image
    kind: str = ""                    # openpose / canny / depth ...
    weight: float = 1.0
    start: float = 0.0
    end: float = 1.0

    def describe(self) -> str:
        return f"{self.kind or 'control'}@{self.weight}"


@dataclass
class ImageBatch:
    """多张图（视频帧 / 批量变体）。"""

    images: list = field(default_factory=list)
    fps: float = 8.0

    def __len__(self):
        return len(self.images)


def as_image(value) -> Optional[Any]:
    """尽力把值转成 PIL.Image；失败返回 None（不抛异常）。"""
    try:
        from PIL import Image
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, ImageBatch):
        return value.images[0] if value.images else None
    if hasattr(value, "convert"):          # 鸭子类型
        try:
            return value.convert("RGB")
        except Exception:
            return None
    return None
