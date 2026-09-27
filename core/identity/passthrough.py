# core/identity/passthrough.py
"""provider #1：原图直传 —— 复刻现有行为，输出逐像素不变。"""
from __future__ import annotations

from typing import Any

from PIL import Image

from core.identity.base import IdentityProviderBase, IdentitySpec


class PassthroughProvider(IdentityProviderBase):
    name = "passthrough"
    display_name = "原图直传（现有 CLIP-IPAdapter 行为）"
    requires_face_model = False
    extra_vram_gb = 0.0
    cacheable = False           # 没有可缓存的东西

    def prepare(self, image: Image.Image, spec: IdentitySpec) -> Any:
        # 原样返回：ip_adapter_image= 收到的东西和改造前完全一致
        return image