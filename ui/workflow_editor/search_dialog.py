# -*- coding: utf-8 -*-
"""节点搜索弹窗（参考 ComfyUI 的双击搜索添加）。

三种用法：
1. 画布空白处双击 → 列出全部节点；
2. Ctrl+F → 同上；
3. 从端口拖到空白处松手 → 只列出"输入口能接住这个类型"的节点，
   选中后自动建节点并连线（`type_hint` 就是端口类型）。
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QVBoxLayout)

from ui.i18n import ui_i18n
from workflow import registry
from workflow.schema import DataType, types_compatible

logger = logging.getLogger(__name__)


def _spec_accepts(spec, dtype: DataType | None) -> bool:
    """该节点的任一输入口能否接住 dtype。"""
    if dtype is None or dtype == DataType.ANY:
        return True
    for port in (spec.inputs or {}).values():
        if types_compatible(dtype, port.type):
            return True
    return False


class NodeSearchDialog(QDialog):
    """输入关键字过滤 → 回车/双击确认。"""

    def __init__(self, parent=None, type_hint: str = "", title: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title or ui_i18n.tr("添加节点"))
        self.resize(420, 460)
        self._dtype = None
        if type_hint:
            try:
                self._dtype = DataType(str(type_hint).upper())
            except ValueError:
                self._dtype = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        if self._dtype is not None:
            lay.addWidget(QLabel(ui_i18n.trf(
                "只显示能接住 {type} 的节点", type=self._dtype.value)))

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(ui_i18n.tr("搜索节点（类型 / 分类 / 说明）…"))
        self.edit.textChanged.connect(self._apply_filter)
        self.edit.returnPressed.connect(self._accept_current)
        lay.addWidget(self.edit)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda *_: self._accept_current())
        lay.addWidget(self.list, 1)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept_current)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        registry.load_builtin_nodes()
        self._rows: list[tuple[str, str]] = []      # (node_type, 显示文本)
        for spec in registry.all_specs().values():
            if not _spec_accepts(spec, self._dtype):
                continue
            label = spec.type
            cat = ui_i18n.tr(spec.category or "")
            if cat:
                label = f"{spec.type}    · {cat}"
            self._rows.append((spec.type, label))
        self._rows.sort(key=lambda r: r[0].lower())
        self._populate(self._rows)
        self.edit.setFocus(Qt.FocusReason.OtherFocusReason)

    # ------------------------------------------------------------------
    def _populate(self, rows):
        self.list.clear()
        for ntype, label in rows:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, ntype)
            self.list.addItem(item)
        if rows:
            self.list.setCurrentRow(0)

    def _apply_filter(self, text: str):
        q = (text or "").strip().lower()
        if not q:
            self._populate(self._rows)
            return
        hits = []
        for ntype, label in self._rows:
            spec = registry.get(ntype)
            hay = " ".join(x for x in (
                ntype, label, spec.category if spec else "",
                getattr(spec, "description", "") or "") if x).lower()
            if q in hay:
                hits.append((ntype, label))
        self._populate(hits)

    def _accept_current(self):
        if self.selected_type():
            self.accept()
        else:
            self.reject()

    def selected_type(self) -> str:
        item = self.list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    # ------------------------------------------------------------------
    @staticmethod
    def pick(parent=None, type_hint: str = "", title: str = "") -> str:
        """弹出并返回选中的节点类型（取消返回空串）。"""
        dlg = NodeSearchDialog(parent, type_hint=type_hint, title=title)
        try:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return ""
            return dlg.selected_type()
        finally:
            dlg.deleteLater()
