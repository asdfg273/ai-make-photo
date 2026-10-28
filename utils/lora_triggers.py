"""LoRA 展示名、旁文件触发词解析及提示词去重合并。"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def split_display(text: str) -> tuple[str, str]:
    value = (text or "").strip()
    if not value or value in ("无", "None"):
        return "", ""
    if value.endswith("]") and "[" in value:
        name, trigger = value.rsplit("[", 1)
        return name.strip(), trigger[:-1].strip()
    return value, ""


def _sidecar_candidates(name: str, lora_path: str = ""):
    if lora_path:
        yield Path(lora_path).with_suffix(".txt")
    base = Path(name).stem
    for sub in ("sdxl", "sd1.5", "sd15", ""):
        yield Path("loras") / sub / f"{base}.txt" if sub else Path("loras") / f"{base}.txt"


def trigger_for_display(text: str, lora_path: str = "") -> str:
    """内联 name[trigger] 优先，其次读取既有 .txt 触发词。"""
    name, inline = split_display(text)
    if not name:
        return ""
    if inline:
        return inline
    for path in _sidecar_candidates(name, lora_path):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception as exc:
            logger.warning("读取 LoRA 触发词失败 %s: %s", path, exc)
    return ""


def merge_triggers(prompt: str, displays, lora_paths=None) -> str:
    """把所选 LoRA 的触发词前插一次，不改变 LoRA 文件或重复词组。"""
    paths = list(lora_paths or [])
    triggers = []
    for index, display in enumerate(displays or []):
        path = paths[index] if index < len(paths) else ""
        trigger = trigger_for_display(str(display or ""), path)
        if trigger and trigger not in triggers:
            triggers.append(trigger)
    current = (prompt or "").strip()
    missing = [trigger for trigger in triggers
               if trigger.casefold() not in current.casefold()]
    if not missing:
        return current
    prefix = ", ".join(missing)
    return f"{prefix}, {current}" if current else prefix
