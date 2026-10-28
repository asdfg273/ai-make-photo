"""角色多参考图、姿势选择和可序列化提示词特征缓存。"""
from __future__ import annotations

import os
from pathlib import Path

_POSES = ("front", "side", "full", "extra")
_POSE_ALIASES = {
    "正面": "front", "front": "front", "frontal": "front",
    "侧面": "side", "side": "side", "profile": "side",
    "全身": "full", "full": "full", "fullbody": "full", "full-body": "full",
    "备用": "extra", "extra": "extra",
}


def normalize_references(character) -> list[dict]:
    """兼容旧单图字段，保留最多四张存在且路径唯一的参考图。"""
    raw = list(getattr(character, "reference_images", None) or [])
    legacy = getattr(character, "ref_image_path", "") or ""
    if legacy and not any(str(item.get("path", "")) == legacy
                          for item in raw if isinstance(item, dict)):
        raw.insert(0, {"path": legacy, "pose": "front"})
    refs: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "") or "").strip()
        if not path or not Path(path).is_file():
            continue
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        pose = _POSE_ALIASES.get(str(item.get("pose", "front")).strip().lower(), "front")
        refs.append({"path": path, "pose": pose})
        if len(refs) == 4:
            break
    if refs and refs[0]["pose"] == "extra":
        refs[0]["pose"] = "front"
    character.reference_images = refs
    character.ref_image_path = refs[0]["path"] if refs else legacy
    return refs


def select_reference(character, pose_prompt: str = "", index: int = 0) -> dict:
    """按场景姿势选图；没有明确姿势时按图片序号轮转。"""
    refs = normalize_references(character)
    if not refs:
        return {}
    text = (pose_prompt or "").lower()
    pose = ""
    if any(word in text for word in ("side", "profile", "侧面")):
        pose = "side"
    elif any(word in text for word in ("full body", "full-body", "full_body", "全身")):
        pose = "full"
    elif any(word in text for word in ("front", "正面")):
        pose = "front"
    if pose:
        match = next((ref for ref in refs if ref["pose"] == pose), None)
        if match:
            return dict(match)
    return dict(refs[index % len(refs)])


def image_signature(path: str) -> dict:
    """文件内容版本签名；路径不变但内容改变也会失效。"""
    absolute = os.path.abspath(path)
    try:
        stat = os.stat(absolute)
        return {"path": absolute, "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    except OSError:
        return {"path": absolute, "mtime_ns": 0, "size": 0}


def cached_feature(character, path: str) -> str:
    """读取与当前文件签名匹配的提示词视觉特征。"""
    cache = getattr(character, "extracted_features", None)
    if not isinstance(cache, dict) or cache.get("version") != 1:
        return ""
    signature = image_signature(path)
    for item in (cache.get("refs") or {}).values():
        if not isinstance(item, dict):
            continue
        if all(item.get(key) == signature[key]
               for key in ("path", "mtime_ns", "size")):
            return str(item.get("text", "") or "")
    return ""


def store_feature(character, path: str, text: str) -> None:
    """写入单张参考图缓存，不影响其它姿势的有效缓存。"""
    if not text:
        return
    cache = getattr(character, "extracted_features", None)
    if not isinstance(cache, dict) or cache.get("version") != 1:
        cache = {"version": 1, "refs": {}}
    refs = normalize_references(character)
    signature = image_signature(path)
    pose = next((ref["pose"] for ref in refs
                 if os.path.normcase(ref["path"]) == os.path.normcase(signature["path"])), "extra")
    key = pose
    suffix = 2
    while key in cache["refs"] and cache["refs"][key].get("path") != signature["path"]:
        key = f"{pose}_{suffix}"
        suffix += 1
    cache["refs"][key] = {**signature, "text": str(text)}
    character.extracted_features = cache
