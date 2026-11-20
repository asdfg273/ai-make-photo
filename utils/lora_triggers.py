"""LoRA 展示名、旁文件触发词解析及提示词去重合并。"""
from __future__ import annotations

import logging
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


def _lora_roots():
    """LoRA 搜索根目录（绝对路径）。

    ⚠️ 这里曾经直接用相对路径 `Path("loras")`。触发词旁文件（`xxx.txt`）
    是跟着 LoRA 放在 `<项目>/loras/` 下的，而相对路径只在"当前工作目录
    正好是项目根"时才命中：换 CWD（测试、快捷方式启动、从别的目录调用
    本模块）就**静默读不到触发词**，LoRA 加载了但触发词没进提示词。
    """
    roots = []
    try:
        from utils import paths  # 延迟导入：避免 utils 包循环导入
        lora_dir = getattr(paths, "LORA_DIR", None)
        if lora_dir:
            roots.append(Path(lora_dir))
    except Exception as exc:
        logger.debug("读取 LORA_DIR 失败，回退相对路径: %s", exc)
    roots.append(Path("loras"))
    return roots


def _sidecar_candidates(name: str, lora_path: str = ""):
    if lora_path:
        yield Path(lora_path).with_suffix(".txt")
    base = Path(name).stem
    for root in _lora_roots():
        for sub in ("sdxl", "sd1.5", "sd15", ""):
            yield (root / sub / f"{base}.txt") if sub else (root / f"{base}.txt")


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
