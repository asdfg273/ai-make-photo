# ui/workflow_editor/param_sources.py
"""节点参数的「本地文件选项源」—— 把填文件名变成选文件名。

为什么需要它
------------
`LoadCheckpoint` / `LoadLoRA` / `LoadVAE` / `LoadImage` 的参数都是
「一个字符串 + 用户自己记得住文件名」。对老用户是习惯，对新用户就是
"我明明有模型，却不知道该往框里敲什么"。这里把项目里**已经存在的**
扫描逻辑（`utils.model_scanner` / `utils.paths`）包成几个纯数据函数，
编辑器按 `ParamSpec.source` 取列表，渲染成下拉框。

设计约束
--------
· **绝不 import Qt**：本模块要能在无界面（headless）环境下被导入/测试，
  所以只有标准库 + 函数内部的重量级 import；
· **绝不抛异常**：任何一个源扫不到东西都只返回 `[]`（debug 日志），
  界面顶多是"下拉框里没选项"，不能因为磁盘/权限问题把编辑器打崩；
· **带 5 秒 TTL 缓存**：展开节点、反复点同一个下拉框不该反复扫盘，
  但用户刚放进来的模型也要"很快就出现"，所以缓存很短且有 `invalidate()`；
· **纯数据**：不往列表里塞「（无）」这类占位项（那是界面的事），
  这样同一份列表能被下拉框、搜索框、校验逻辑共用。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

logger = logging.getLogger(__name__)

# 缓存有效期（秒）。取 5 秒：比"每次展开都扫盘"省得多，
# 又比"缓存到进程结束"更符合"我刚丢了个模型进去"的直觉。
_CACHE_TTL = 5.0

# {来源名: (写入时刻 monotonic, 结果列表)}
_CACHE: dict = {}

# 最近图片最多返回多少条（下拉框太长反而不好选）
_MAX_IMAGES = 40

# LoRA / VAE 认识的文件后缀（与 core.model_manager.get_available_loras 一致）
_LORA_EXTS = (".safetensors", ".ckpt", ".pt")


# ----------------------------------------------------------------------
#  内部工具
# ----------------------------------------------------------------------
def _cached(key: str, scan: Callable[[], list]) -> list:
    """带 TTL 的取数：命中就返回副本，未命中/过期则重扫。

    任何异常都在这里被吃掉并记 debug —— 界面只需要一个列表。
    返回的是**副本**，调用方随便改不会污染缓存。
    """
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit is not None and (now - hit[0]) < _CACHE_TTL:
        return list(hit[1])

    try:
        values = [str(v) for v in scan()]
    except Exception:
        logger.debug(f"扫描本地文件失败（按空列表处理）: {key}", exc_info=True)
        values = []

    _CACHE[key] = (now, values)
    return list(values)


def _walk_files(root: str, exts: tuple) -> list:
    """递归收集 root 下指定后缀的文件名（basename，去重后排序）。

    目录不存在、没权限等等一律当"没有文件"，不向外抛。
    """
    names = set()
    if not root or not os.path.isdir(root):
        return []
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(exts):
                names.add(f)
    return sorted(names)


# ----------------------------------------------------------------------
#  各个来源
# ----------------------------------------------------------------------
def checkpoints() -> list[str]:
    """底模文件名（如 "meinamix_v12Final.safetensors"）。

    复用 `utils.model_scanner`：它按 `MODEL_TYPES`（由架构注册表派生）
    逐个子目录扫描，这里只把结果里的 `name` 收集起来。
    """
    return _cached("checkpoints", _scan_checkpoints)


def _scan_checkpoints() -> list:
    from utils.model_scanner import MODEL_TYPES, scan_models

    names = set()
    for subdir in MODEL_TYPES:
        try:
            for item in scan_models(subdir):
                name = str(item.get("name") or "").strip()
                if name:
                    names.add(name)
        except Exception:
            # 单个子目录坏了不该拖掉其它类型的模型
            logger.debug(f"扫描模型子目录失败: {subdir}", exc_info=True)
    return sorted(names)


def loras() -> list[str]:
    """LoRA 文件名（递归扫描 loras/ 的所有子目录，只留 basename）。

    跨子目录去重：`sd1.5/`、`sdxl/` 下同名文件在列表里只出现一次
    （执行侧 `apply_multiple_loras` 自己会按架构找子目录）。
    """
    return _cached("loras", _scan_loras)


def _scan_loras() -> list:
    from utils import paths

    return _walk_files(paths.LORA_DIR, _LORA_EXTS)


def vae() -> list[str]:
    """VAE 文件名（models/ 与 weights/ 下名字里带 "vae" 的模型文件）。

    判定沿用了 `run_load_vae` 的搜索范围（MODEL_DIR + WEIGHTS_DIR），
    但没有单独的 VAE 目录清单，只能按文件名里的 "vae" 认。
    """
    return _cached("vae", _scan_vae)


def _scan_vae() -> list:
    from utils import paths

    names = set()
    for root in (paths.MODEL_DIR, paths.WEIGHTS_DIR):
        for name in _walk_files(root, _LORA_EXTS):
            if "vae" in name.lower():
                names.add(name)
    return sorted(names)


def images() -> list[str]:
    """最近图片（参考图 / 姿势图常用），绝对路径，最新在前，最多 40 张。

    只看输出目录（photo/）**本层**的文件：那是用户刚生成、最可能拿来
    当参考图的东西；递归下去会把成百上千张旧图翻出来，反而难选。
    """
    return _cached("images", _scan_images)


def _scan_images() -> list:
    from utils import paths

    root = paths.OUTPUT_DIR
    if not os.path.isdir(root):
        return []

    entries = []
    for f in os.listdir(root):
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        full = os.path.join(root, f)
        try:
            if not os.path.isfile(full):
                continue
            entries.append((os.path.getmtime(full), full))
        except OSError:
            continue

    entries.sort(key=lambda pair: pair[0], reverse=True)
    return [full for _mtime, full in entries[:_MAX_IMAGES]]


# ----------------------------------------------------------------------
#  对外分发
# ----------------------------------------------------------------------
def providers() -> dict[str, Callable[[], list]]:
    """{source 名: 取数函数}。`ParamSpec.source` 的取值就是这里的键。"""
    return {
        "checkpoints": checkpoints,
        "loras": loras,
        "vae": vae,
        "images": images,
    }


def options_for(source: str) -> list[str]:
    """按 `ParamSpec.source` 取选项列表；未知来源 / 空来源返回 `[]`。"""
    fn = providers().get((source or "").strip())
    if fn is None:
        return []
    return fn()


def describe(source: str) -> str:
    """给界面用的中文短标签（如下拉框前面的"本地 LoRA"）。"""
    return {
        "checkpoints": "本地底模",
        "loras": "本地 LoRA",
        "vae": "本地 VAE",
        "images": "最近图片",
    }.get((source or "").strip(), "")


def invalidate() -> None:
    """清空缓存：用户点了"刷新"，或外部刚写完文件时调用。"""
    _CACHE.clear()
