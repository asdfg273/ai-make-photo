"""模型同目录示例图发现；只检查路径，不读取图像内容。"""
from __future__ import annotations

import os

_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def find_model_preview(model_path: str) -> str | None:
    if not model_path:
        return None
    stem = os.path.splitext(model_path)[0]
    for suffix in (".preview", ""):
        for ext in _EXTENSIONS:
            candidate = stem + suffix + ext
            if os.path.isfile(candidate):
                return candidate
    return None
