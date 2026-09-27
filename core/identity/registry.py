# core/identity/registry.py
"""provider 注册表 + 全局 resolver。模块级单例，任何宿主都能直接用。"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Callable, Optional

from PIL import Image

from core.identity.base import IdentityProviderBase, IdentitySpec
from core.identity.passthrough import PassthroughProvider

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, IdentityProviderBase] = {}


def register(provider: IdentityProviderBase) -> None:
    if provider.name in _REGISTRY:
        logger.warning(f"⚠️ 身份后端 {provider.name} 被重复注册，后者覆盖前者")
    _REGISTRY[provider.name] = provider
    logger.debug(f"🧬 注册身份后端: {provider.name}")


def get(name: str) -> IdentityProviderBase:
    if name not in _REGISTRY:
        logger.warning(f"⚠️ 未知身份后端 {name}，回退 passthrough")
        return _REGISTRY["passthrough"]
    return _REGISTRY[name]


def available() -> list[IdentityProviderBase]:
    return list(_REGISTRY.values())


class IdentityResolver:
    """把"参考图"翻译成 ip_adapter_image= 能吃的值，并管缓存与显存。"""

    def __init__(self, default: str = "passthrough", cache_size: int = 16):
        self._spec = IdentitySpec(provider=default)
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._cache_size = cache_size
        self._active: Optional[IdentityProviderBase] = None
        self._device = "cuda"
        # 宿主注入：需要腾显存时调这个
        self.on_before_load: Optional[Callable[[], None]] = None

    # ---------- 配置 ----------
    @property
    def spec(self) -> IdentitySpec:
        return self._spec

    def current(self) -> IdentityProviderBase:
        return get(self._spec.provider)

    def set_provider(self, name: str) -> bool:
        if name == self._spec.provider:
            return True
        old = self.current()
        new = get(name)
        if new.name != name:
            return False                       # 名字不存在
        logger.info(f"🧬 身份后端切换: {old.name} → {new.name}")
        self._unload_active()
        self._cache.clear()
        self._spec = IdentitySpec(
            provider=name,
            strength=self._spec.strength,
            face_weight=self._spec.face_weight,
        )
        return True

    def set_strength(self, strength: float) -> None:
        self._spec = IdentitySpec(
            provider=self._spec.provider,
            strength=strength,
            face_weight=self._spec.face_weight,
        )

    # ---------- 主入口 ----------
    def resolve(self, image: Optional[Image.Image]) -> Any:
        if image is None:
            return None
        prov = self.current()

        if not prov.is_loaded():
            self._load(prov)

        if not prov.cacheable:
            return prov.prepare(image, self._spec)

        key = self._cache_key(prov, image)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit

        result = prov.prepare(image, self._spec)
        self._cache[key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    # ---------- 内部 ----------
    def _cache_key(self, prov: IdentityProviderBase, image: Image.Image) -> str:
        # 按内容哈希，不按路径 —— 同一路径换图必须失效
        h = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
        return f"{prov.name}|{self._spec.strength}|{self._spec.face_weight}|{h}"

    def _load(self, prov: IdentityProviderBase) -> None:
        if prov.needs_vram_headroom and self.on_before_load:
            try:
                self.on_before_load()
            except Exception as e:
                logger.warning(f"⚠️ 腾显存回调失败: {e}")
        logger.info(f"🧬 加载身份后端: {prov.describe()}")
        prov.load(self._device)
        self._active = prov

    def _unload_active(self) -> None:
        if self._active is None:
            return
        try:
            self._active.unload()
            logger.info(f"🧬 已卸载身份后端: {self._active.name}")
        except Exception as e:
            logger.warning(f"⚠️ 卸载身份后端失败: {e}")
        self._active = None

    def release(self) -> None:
        self._unload_active()
        self._cache.clear()


# ---------- 模块级单例 ----------
_RESOLVER = IdentityResolver()


def register_defaults() -> None:
    register(PassthroughProvider())


def ipa_input(image: Optional[Image.Image]) -> Any:
    """所有 ip_adapter_image= 调用点统一走这里。"""
    return _RESOLVER.resolve(image)


def resolver() -> IdentityResolver:
    return _RESOLVER


def set_provider(name: str) -> bool:
    return _RESOLVER.set_provider(name)


def current_spec() -> IdentitySpec:
    return _RESOLVER.spec


def compliance_matrix() -> list[dict]:
    """给 README 用的许可矩阵 —— 每个后端的第三方权重依赖。"""
    return [
        {
            "backend": p.name,
            "label": p.display_name,
            "third_party_face_model": p.requires_face_model,
            "extra_vram_gb": p.extra_vram_gb,
        }
        for p in available()
    ]


register_defaults()