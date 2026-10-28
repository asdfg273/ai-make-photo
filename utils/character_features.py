"""角色提示词视觉特征的持久缓存读写与预提取。"""
from __future__ import annotations

import logging

from utils.character_references import (
    cached_feature, normalize_references, store_feature,
)

logger = logging.getLogger(__name__)


def feature_for_generation(host, ref_img, allow_extract: bool = True) -> str:
    """优先读取角色 JSON 缓存，未命中时按设置调用宿主 AI 并回填。"""
    if not ref_img or not isinstance(ref_img, str):
        extractor = getattr(host, "_gt_extract_character_features", None)
        return extractor(ref_img) if callable(extractor) else ""
    char = getattr(host, "_active_character", None)
    if char is not None:
        cached = cached_feature(char, ref_img)
        if cached:
            logger.info("复用角色卡持久特征缓存")
            return cached
    extractor = getattr(host, "_gt_extract_character_features", None)
    if not allow_extract or not callable(extractor):
        return ""
    text = extractor(ref_img) or ""
    if text and char is not None:
        store_feature(char, ref_img, text)
        library = getattr(host, "_character_library", None)
        saver = getattr(library, "save_character", None)
        if callable(saver):
            try:
                saver(char)
            except Exception as exc:
                logger.warning("角色特征缓存回填失败: %s", exc)
    return text


def preextract_character_features(host, char) -> None:
    """保存角色前预提取所有参考图；单张失败不影响角色保存。"""
    extractor = getattr(host, "_gt_extract_character_features", None)
    if not callable(extractor):
        return
    previous = getattr(host, "_active_character", None)
    host._active_character = char
    try:
        for ref in normalize_references(char):
            if cached_feature(char, ref["path"]):
                continue
            try:
                feature_for_generation(host, ref["path"])
            except Exception as exc:
                logger.warning("预提取角色特征失败 %s: %s", ref["path"], exc)
    finally:
        host._active_character = previous
