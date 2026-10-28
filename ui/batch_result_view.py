"""批量出图结果视图：角色/场景/序号归档、预览与精修选择。"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSplitter, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ui.i18n import ui_i18n


class BatchResultView(QWidget):
    send_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._preview_path = ""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel(ui_i18n.tr("角色 / 场景 / 序号"))
        self.tree.setMinimumWidth(280)
        self.tree.currentItemChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.tree)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.preview = QLabel(ui_i18n.tr("选择一张图片查看预览"))
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 280)
        self.preview.setProperty("role", "hint")
        right_layout.addWidget(self.preview, 1)

        self.lbl_path = QLabel("")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setProperty("role", "hint")
        right_layout.addWidget(self.lbl_path)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.btn_send = QPushButton(ui_i18n.tr("发送到图生图精修"))
        self.btn_send.setEnabled(False)
        self.btn_send.clicked.connect(self._on_send)
        actions.addWidget(self.btn_send)
        right_layout.addLayout(actions)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

    def set_manifest(self, session: dict):
        """重建归档树；输入为空时保留一个清晰的空状态。"""
        self.tree.clear()
        self._preview_path = ""
        self.preview.clear()
        self.btn_send.setEnabled(False)
        jobs = session.get("jobs", []) if isinstance(session, dict) else []
        if not jobs:
            self.preview.setText(ui_i18n.tr("暂无批量结果"))
            self.lbl_path.clear()
            return

        character = QTreeWidgetItem([
            session.get("character_name") or ui_i18n.tr("未命名角色")])
        self.tree.addTopLevelItem(character)
        scene_items = {}
        first_result = None
        status_labels = {
            "done": ui_i18n.tr("完成"), "failed": ui_i18n.tr("失败"),
            "pending": ui_i18n.tr("待生成"), "running": ui_i18n.tr("生成中"),
        }
        for job in jobs:
            scene_uuid = job.get("scene_uuid", "")
            if scene_uuid not in scene_items:
                scene_item = QTreeWidgetItem([
                    job.get("scene_name") or ui_i18n.tr("(未命名)")])
                character.addChild(scene_item)
                scene_items[scene_uuid] = scene_item
            paths = job.get("output_paths", []) or []
            path = str(paths[0]) if paths else ""
            status = job.get("status", "pending")
            leaf = QTreeWidgetItem([ui_i18n.trf(
                "第 {index} 张 — {status}", index=int(job.get("index", 0)) + 1,
                status=status_labels.get(status, status))])
            leaf.setData(0, Qt.ItemDataRole.UserRole, path)
            leaf.setData(0, Qt.ItemDataRole.UserRole + 1, status)
            scene_items[scene_uuid].addChild(leaf)
            if first_result is None and status == "done" and Path(path).is_file():
                first_result = leaf
        self.tree.expandAll()
        if first_result is not None:
            self.tree.setCurrentItem(first_result)
        else:
            self.preview.setText(ui_i18n.tr("暂无可预览的已完成图片"))
            self.lbl_path.clear()

    def _selected_path(self) -> str:
        item = self.tree.currentItem()
        if item is None:
            return ""
        path = item.data(0, Qt.ItemDataRole.UserRole)
        return str(path or "")

    def _on_selection_changed(self, current, _previous=None):
        self.btn_send.setEnabled(False)
        self._preview_path = ""
        if current is None:
            return
        status = current.data(0, Qt.ItemDataRole.UserRole + 1)
        path = self._selected_path()
        if status is None:
            return
        if status != "done" or not path or not Path(path).is_file():
            self.preview.clear()
            self.preview.setText(ui_i18n.tr("图片尚未生成或文件不存在"))
            self.lbl_path.setText(path)
            return
        self._preview_path = path
        self.lbl_path.setText(path)
        self._update_preview()
        self.btn_send.setEnabled(True)

    def _update_preview(self):
        if not self._preview_path:
            return
        pixmap = QPixmap(self._preview_path)
        if pixmap.isNull():
            self.preview.setText(ui_i18n.tr("无法读取图片预览"))
            self.btn_send.setEnabled(False)
            return
        width = max(160, self.preview.width() - 12)
        height = max(160, self.preview.height() - 12)
        self.preview.setPixmap(pixmap.scaled(
            width, height, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()

    def _on_send(self):
        path = self._selected_path()
        if path and Path(path).is_file() and self.btn_send.isEnabled():
            self.send_requested.emit(path)
