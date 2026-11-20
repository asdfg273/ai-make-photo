# -*- coding: utf-8 -*-
"""工作流编辑器的撤销/重做栈。

设计取舍（为什么不是命令对象）：
    画布是"整图可序列化"的（`GraphScene.to_workflow()` 就能拿到全部状态），
    所以最省事也最不容易出错的做法是**快照式**撤销：每次改动前存一份快照。
    代价是每步都存一份 JSON（几 KB），换来的好处是任何操作都能撤销，
    包括拖动节点位置、批量粘贴、载入模板 —— 不需要给每种操作写一对 do/undo。
"""

from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)


class HistoryStack:
    """快照式撤销栈。"""

    def __init__(self, limit: int = 80):
        self.limit = max(2, int(limit))
        self._undo: list[tuple[str, dict]] = []
        self._redo: list[tuple[str, dict]] = []

    # ---------------------------------------------------------------- 状态
    def clear(self):
        self._undo.clear()
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_label(self) -> str:
        return self._undo[-1][0] if self._undo else ""

    def redo_label(self) -> str:
        return self._redo[-1][0] if self._redo else ""

    def depth(self) -> int:
        return len(self._undo)

    # ---------------------------------------------------------------- 读写
    def push(self, label: str, state: dict):
        """记录"改动前"的状态。调用方在真正修改之前调用。"""
        if state is None:
            return
        self._undo.append((str(label or "编辑"), copy.deepcopy(state)))
        if len(self._undo) > self.limit:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current: dict):
        """返回 (label, 快照)；没有可撤销时返回 None。"""
        if not self._undo:
            return None
        label, state = self._undo.pop()
        if current is not None:
            self._redo.append((label, copy.deepcopy(current)))
        return label, state

    def redo(self, current: dict):
        if not self._redo:
            return None
        label, state = self._redo.pop()
        if current is not None:
            self._undo.append((label, copy.deepcopy(current)))
        return label, state
