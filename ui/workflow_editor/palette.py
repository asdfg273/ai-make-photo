# ui/workflow_editor/palette.py
"""节点面板：按分类列出**已注册**的节点，支持搜索、双击/拖拽添加。

节点清单完全来自 `workflow.registry.catalog()` —— 加一个节点文件，
这里自动出现，不用改界面。
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (QFrame, QLabel, QLineEdit, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout)

from ui.i18n import ui_i18n
from workflow import registry

logger = logging.getLogger(__name__)

# 分类展示顺序（没列到的分类排在后面）
CATEGORY_ORDER = ["模型", "文本", "采样", "潜空间", "图像", "控制", "适配",
                  "修复", "放大", "视频", "输出", "通用", "测试", "高级"]

# 能执行任意代码的节点：单独标红，避免误拖到正式工作流里
RISKY_NODE_TYPES = {"PythonScript"}


class NodePalette(QFrame):
    """左侧节点库。双击某项 → 请求添加节点。"""

    addRequested = pyqtSignal(str)          # node_type

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        title = QLabel("节点库")
        title.setStyleSheet("font-weight:bold; color:#cdd6f4;")
        lay.addWidget(title)

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("搜索节点…")
        self.edit_search.textChanged.connect(self._apply_filter)
        lay.addWidget(self.edit_search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(1)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setToolTip("双击添加到画布；也可以从这里拖到画布")
        self.tree.setDragEnabled(True)
        lay.addWidget(self.tree, 1)

        self.hint = QLabel("提示：拖端口连线，选中后 Del 删除")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#7f849c; font-size:11px;")
        lay.addWidget(self.hint)

        ui_i18n.language_changed.connect(self.retranslate_dynamic)
        self.reload()
        ui_i18n.translate_widget_tree(self)

    # ------------------------------------------------------------------
    def reload(self):
        registry.load_builtin_nodes()
        self.tree.clear()
        specs = registry.all_specs()
        by_cat: dict[str, list] = {}
        for s in specs.values():
            by_cat.setdefault(s.category, []).append(s)

        cats = [c for c in CATEGORY_ORDER if c in by_cat] + \
               [c for c in sorted(by_cat) if c not in CATEGORY_ORDER]

        for cat in cats:
            top = QTreeWidgetItem([ui_i18n.trf(
                "{category}（{count}）", category=ui_i18n.tr(cat), count=len(by_cat[cat]))])
            top.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.tree.addTopLevelItem(top)
            for s in sorted(by_cat[cat], key=lambda x: x.type):
                child = QTreeWidgetItem([s.type])
                child.setData(0, Qt.ItemDataRole.UserRole, s.type)
                tip = ui_i18n.tr(s.description or "")
                res = s.resources
                if res.estimated_vram_mb:
                    tip += "\n\n" + ui_i18n.trf(
                        "显存峰值 ≈ {value} MB", value=res.estimated_vram_mb)
                if res.requires:
                    tip += "\n" + ui_i18n.trf(
                        "依赖: {requirements}", requirements=", ".join(res.requires))
                if s.type in RISKY_NODE_TYPES:
                    child.setForeground(0, QBrush(QColor("#f38ba8")))
                    tip = (ui_i18n.tr("⚠️ 会执行你自己写的 Python 代码")
                           + "（" + ui_i18n.tr("默认关闭，需在设置里打开") + "）\n\n"
                           + tip)
                if tip:
                    child.setToolTip(0, tip.strip())
                top.addChild(child)
            top.setExpanded(True)
        template = ("共 1 种节点 · 双击添加 · Del 删除" if len(specs) == 1
                    else "共 {count} 种节点 · 双击添加 · Del 删除")
        self.hint.setText(ui_i18n.trf(template, count=len(specs)))
        self._apply_filter(self.edit_search.text())

    def retranslate_dynamic(self, *_):
        """重建分类与提示，保留搜索框中的用户输入。"""
        self.reload()

    def _apply_filter(self, text: str):
        text = (text or "").strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            visible = 0
            for j in range(top.childCount()):
                ch = top.child(j)
                hit = (not text) or (text in ch.text(0).lower())
                ch.setHidden(not hit)
                visible += 1 if hit else 0
            top.setHidden(visible == 0)
            if text and visible:
                top.setExpanded(True)

    def _on_double_click(self, item: QTreeWidgetItem, _col: int):
        ntype = item.data(0, Qt.ItemDataRole.UserRole)
        if ntype:
            self.addRequested.emit(str(ntype))
