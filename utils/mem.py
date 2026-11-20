# -*- coding: utf-8 -*-
"""进程内存工具：把「已不需要的页」真正还给系统。

背景（实测）：MOSS-Image 的 25.4GB 权重是 safetensors 内存映射，量化/推理时会
把整份文件页读进进程工作集。即使 Python 侧引用已经全删、`gc.collect()` +
`torch.cuda.empty_cache()` 都跑过，Windows 仍把这些**文件页**留在工作集里，
任务管理器上就表现为「换了模型内存还是 21GB」。必须显式
`EmptyWorkingSet` 才会把它们退回 standby 列表。
"""
from __future__ import annotations

import ctypes
import gc
import logging
import sys

logger = logging.getLogger(__name__)


def trim_working_set() -> bool:
    """把当前进程的工作集尽量还给系统（Windows）；其他平台是 no-op。

    返回 True 表示调用成功。失败只记 debug —— 这是"锦上添花"的操作，
    不能因为它失败而中断换模型/卸载流程。
    """
    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        handle = kernel32.GetCurrentProcess()
        ok = bool(psapi.EmptyWorkingSet(handle))
        return ok
    except Exception as exc:  # noqa: BLE001 - 尽力而为
        logger.debug("EmptyWorkingSet 失败（忽略）: %s", exc)
        return False


def rss_mb() -> float:
    """当前进程 RSS（MB）；取不到返回 -1。"""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:  # noqa: BLE001
        return -1.0


def vram_mb() -> tuple:
    """(已分配, 已保留, 空闲) 显存，单位 MB；没有 CUDA 返回 (-1, -1, -1)。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return (-1.0, -1.0, -1.0)
        free_b, _total = torch.cuda.mem_get_info()
        return (torch.cuda.memory_allocated() / 1024 ** 2,
                torch.cuda.memory_reserved() / 1024 ** 2,
                free_b / 1024 ** 2)
    except Exception:  # noqa: BLE001
        return (-1.0, -1.0, -1.0)


def release_memory(*, trim: bool = True, empty_cache: bool = True) -> dict:
    """GC + 显存缓存 + 工作集三连，返回前后对比（给日志/界面用）。"""
    before_rss = rss_mb()
    before_vram = vram_mb()
    gc.collect()
    if empty_cache:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:  # noqa: BLE001
            logger.debug("empty_cache 失败（忽略）: %s", exc)
    if trim:
        trim_working_set()
    return {
        "rss_before": before_rss,
        "rss_after": rss_mb(),
        "vram_before": before_vram,
        "vram_after": vram_mb(),
    }


def describe(delta: dict) -> str:
    """把 `release_memory()` 的结果写成一行中文。"""
    if not delta:
        return ""
    r0 = delta.get("rss_before", -1)
    r1 = delta.get("rss_after", -1)
    v0 = delta.get("vram_before") or (-1, -1, -1)
    v1 = delta.get("vram_after") or (-1, -1, -1)
    parts = []
    if r1 >= 0:
        freed = r0 - r1
        if freed >= 1:
            parts.append(f"进程内存 {r0:.0f} → {r1:.0f} MB（释放 {freed:.0f} MB）")
        else:
            parts.append(f"进程内存 {r0:.0f} MB（已是最低）")
    if v1[2] >= 0:
        parts.append(f"显存空闲 {v0[2]:.0f} → {v1[2]:.0f} MB")
    return "；".join(parts)
