# core/identity/base.py
"""身份后端抽象 —— 全部本地，不引用任何第三方人脸识别模型。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentitySpec:
    """一次生成里"用哪个身份后端"的完整描述。"""
    provider: str = "passthrough"
    strength: float = 1.0
    # 仅 ID-embedding 类后端有意义；passthrough 忽略
    face_weight: float = 0.0
    extra: dict = field(default_factory=dict)


@runtime_checkable
class IdentityProvider(Protocol):
    name: str
    display_name: str
    requires_face_model: bool          # ← 合规矩阵直接读这个字段
    extra_vram_gb: float
    cacheable: bool

    def load(self, device: str = "cuda") -> None: ...
    def is_loaded(self) -> bool: ...
    def prepare(self, image: Image.Image, spec: IdentitySpec) -> Any: ...
    def unload(self) -> None: ...


class IdentityProviderBase:
    """默认实现，子类通常只需覆写 prepare。"""

    name = "base"
    display_name = "基类"
    requires_face_model = False
    extra_vram_gb = 0.0
    cacheable = False
    # 需要宿主腾显存时置 True（resolver 会在 load 前调用 on_before_load）
    needs_vram_headroom = False

    def load(self, device: str = "cuda") -> None:
        return None

    def is_loaded(self) -> bool:
        return True

    def prepare(self, image: Image.Image, spec: IdentitySpec) -> Any:
        raise NotImplementedError

    def unload(self) -> None:
        return None

    def describe(self) -> str:
        tag = "需第三方人脸权重" if self.requires_face_model else "无外部人脸权重"
        vram = f"+{self.extra_vram_gb:.1f}GB" if self.extra_vram_gb else "+0GB"
        return f"{self.display_name}（{tag}, {vram}）"