# -*- coding: utf-8 -*-
"""运行环境自检：解释器 / 依赖版本是否满足本程序的功能需求。

为什么要有这个
--------------
本项目同时存在两个解释器：项目自带的 `venv`（diffusers 0.40）和系统
Python（diffusers 0.32）。用错一个不会立刻报错，而是**功能静默降级**：

* `diffusers.hooks.apply_group_offloading` 不存在 → 8GB 卡跑 MOSS-Image
  的"分块卸载"路径直接抛错；
* `diffusers.ModularPipeline` 不存在 → Anima 架构加载不了；
* `diffusers.PipelineQuantizationConfig` 不存在 → DiT 量化一档都用不了。

用户看到的现象往往是"某个模型用不了"或"显存调度怪怪的"，很难联想到
是解释器搞错了。所以在启动时把这件事说清楚。
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# (属性/模块, 缺失时不能用什么)
_REQUIREMENTS = (
    ("diffusers.ModularPipeline", "Anima 架构（ModularPipeline）"),
    ("diffusers.hooks.apply_group_offloading", "DiT 分块卸载（8GB 卡跑 MOSS-Image / Flux）"),
    ("diffusers.PipelineQuantizationConfig", "DiT 量化（8bit / FP8 / NVFP4）"),
)


def _has_dotted(dotted: str) -> bool:
    try:
        mod_name, _, attr = dotted.rpartition(".")
        if not mod_name:
            return False
        mod = __import__(mod_name, fromlist=[attr or "*"])
        return hasattr(mod, attr) if attr else True
    except Exception:
        return False


def check_runtime() -> dict:
    """返回 {interpreter, venv, diffusers, torch, missing: [(feature, what)], ok}。"""
    rep = {
        "interpreter": sys.executable,
        "venv": sys.prefix != sys.base_prefix,
        "diffusers": "",
        "torch": "",
        "missing": [],
        "ok": True,
    }
    try:
        import diffusers
        rep["diffusers"] = getattr(diffusers, "__version__", "?")
    except Exception as e:
        rep["diffusers"] = f"导入失败: {e}"
        rep["ok"] = False
    try:
        import torch
        rep["torch"] = getattr(torch, "__version__", "?")
    except Exception:
        rep["torch"] = "?"

    for dotted, what in _REQUIREMENTS:
        if not _has_dotted(dotted):
            rep["missing"].append((dotted, what))
    if rep["missing"]:
        rep["ok"] = False
    return rep


def summary_text(rep: dict) -> str:
    """给界面/日志用的一段中文说明（无问题时返回空串）。"""
    if rep.get("ok"):
        return ""
    lines = ["⚠️ 当前运行环境的依赖版本不够，部分功能会用不了："]
    for dotted, what in rep.get("missing", []):
        lines.append(f"  · 缺 {dotted} → {what}")
    lines.append("")
    lines.append(f"  解释器: {rep.get('interpreter')}")
    lines.append(f"  diffusers {rep.get('diffusers')} · torch {rep.get('torch')}")
    if not rep.get("venv"):
        lines.append("")
        lines.append("  你现在**不是**用项目自带的 venv 启动的。请改用：")
        lines.append("    venv\\Scripts\\python.exe main.py")
        lines.append("  （或把这个解释器里的 diffusers/transformers 升到与 "
                     "requirements.txt 一致）")
    return "\n".join(lines)


def log_report(rep: dict | None = None) -> dict:
    rep = rep or check_runtime()
    if rep.get("ok"):
        logger.info("运行环境自检通过：diffusers %s · torch %s",
                    rep.get("diffusers"), rep.get("torch"))
    else:
        for line in summary_text(rep).splitlines():
            if line.strip():
                logger.warning(line.strip())
    return rep


def interpreter_hint() -> str:
    """给"怎么启动"用的绝对路径提示。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "venv", "Scripts", "python.exe")
