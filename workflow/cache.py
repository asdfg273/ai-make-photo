# workflow/cache.py
"""节点级缓存：让"只重跑某个节点"和"改一个参数不全量重算"成为可能。

缓存键 = (节点类型, 解析后的输入, 参数)。上游没变、参数没变 → 命中。
被取消或失败的节点不写缓存。

刻意不做的事
------------
· 不缓存大张量到磁盘（会把磁盘打爆）；只在内存里留 LRU。
· 不跨进程共享 —— 需要落盘时请用 serializers 存工作流本身。
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _stable(obj: Any, depth: int = 0) -> Any:
    """把值转成可哈希/可 JSON 的稳定表示。遇到不可序列化对象退化为 id()。"""
    if depth > 6:
        return "<deep>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_stable(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _stable(v, depth + 1) for k, v in sorted(
            obj.items(), key=lambda kv: str(kv[0]))}
    # PIL 图：尺寸 + 模式 + **缩略图**摘要
    # ⚠️ 不要对整图 tobytes() 做 md5：2048x2048 实测约 20ms、4096 约 4 倍，
    # 而缓存键每次执行都要算一遍。缩略图足以区分"是不是同一张图"，
    # 代价与图像大小基本无关。
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            thumb = obj.copy()
            thumb.thumbnail((64, 64), Image.Resampling.NEAREST)
            digest = hashlib.md5(thumb.tobytes()).hexdigest()[:12]
            return (f"img:{obj.mode}:{obj.size}:"
                    f"{hashlib.md5(obj.tobytes()[:4096]).hexdigest()[:8]}:{digest}")
    except Exception:
        pass
    # 张量：形状+设备+少量采样
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return (f"t:{tuple(obj.shape)}:{obj.dtype}:{obj.device}:"
                    f"{float(obj.flatten()[:8].float().sum()):.6f}")
    except Exception:
        pass
    return f"<{type(obj).__name__}:{id(obj)}>"


def make_key(node_type: str, inputs: Dict[str, Any],
             params: Dict[str, Any]) -> str:
    payload = json.dumps({"t": node_type, "i": _stable(inputs),
                          "p": _stable(params)},
                         sort_keys=True, ensure_ascii=False,
                         default=lambda o: f"<{type(o).__name__}>")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class NodeCache:
    """内存 LRU 缓存。`invalidate_subtree` 用于"改了上游就丢掉下游"。"""

    def __init__(self, max_entries: int = 24, enabled: bool = True):
        self.max_entries = max(1, int(max_entries))
        self.enabled = bool(enabled)
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._node_key: Dict[str, str] = {}     # node_id -> cache key
        self.hits = 0
        self.misses = 0

    # ---- 基本操作 ----
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def bind(self, node_id: str, key: str) -> None:
        self._node_key[node_id] = key

    def key_for(self, node_id: str) -> Optional[str]:
        return self._node_key.get(node_id)

    # ---- 失效 ----
    def invalidate(self, node_id: str) -> None:
        key = self._node_key.pop(node_id, None)
        if key:
            self._store.pop(key, None)

    def invalidate_many(self, node_ids) -> int:
        n = 0
        for nid in node_ids:
            if nid in self._node_key:
                self.invalidate(nid)
                n += 1
        return n

    def clear(self) -> None:
        self._store.clear()
        self._node_key.clear()
        self.hits = self.misses = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._store), "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0}

    def describe(self) -> str:
        st = self.stats()
        return (f"缓存 {st['entries']}/{self.max_entries}，"
                f"命中率 {st['hit_rate'] * 100:.0f}%")
