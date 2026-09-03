# ui/components/filmstrip.py — 底部胶片条：最近媒体一览，点击跳画廊
import os
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QMenu
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QIcon

from ui.gallery_panel import GalleryPanel


class FilmStrip(QListWidget):
    """横向缩略图条：展示最近生成的图片/动画，点击发出 media_clicked(path)。"""

    media_clicked = pyqtSignal(str)
    reuse_requested = pyqtSignal(str)   # 右键「套用参数」

    def __init__(self, parent=None, gallery=None):
        super().__init__(parent)
        self._gallery = gallery          # 右键菜单动作委托给统一画廊
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setIconSize(QSize(96, 96))
        self.setFixedHeight(116)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self.itemClicked.connect(self._on_clicked)
        self.itemDoubleClicked.connect(self._on_clicked)

    def refresh(self, paths):
        """重建胶片条内容（paths: 最新在前）。"""
        self.clear()
        for path in paths:
            self._add(path)

    def _add(self, path: str):
        item = QListWidgetItem()
        if GalleryPanel.media_kind(path) == "video":
            icon = GalleryPanel.video_frame_icon(path, 96) or \
                GalleryPanel._video_placeholder_icon()
            item.setIcon(icon)
        else:
            pix = QPixmap(path)
            if not pix.isNull():
                item.setIcon(QIcon(pix.scaled(
                    96, 96,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)))
        item.setToolTip(f"{os.path.basename(path)}\n单击: 选中并回填参数\n右键: 更多操作")
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.addItem(item)

    def _on_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.media_clicked.emit(path)

    # ---------- 右键菜单（与画廊一致的收藏/文件夹/移除/删除）----------
    def _show_menu(self, pos):
        item = self.itemAt(pos)
        if item is None or self._gallery is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        g = self._gallery
        menu = QMenu(self)
        is_video = GalleryPanel.media_kind(path) == "video"
        act_reuse = None
        if not is_video:
            act_reuse = menu.addAction("🔁 套用参数到生成区")
            menu.addSeparator()
        is_fav = os.path.abspath(path) in g._favs
        act_fav = menu.addAction("💔 取消收藏" if is_fav else "⭐ 加入收藏")
        act_folder = menu.addAction("📁 打开所在文件夹")
        menu.addSeparator()
        act_remove = menu.addAction("🗑 从画廊移除")
        act_del = menu.addAction("❌ 删除文件")

        chosen = menu.exec(self.viewport().mapToGlobal(pos))
        if act_reuse is not None and chosen == act_reuse:
            self.reuse_requested.emit(path)
        elif chosen == act_fav:
            g._toggle_fav([path])
        elif chosen == act_folder:
            g._open_folder(path)
        elif chosen == act_remove:
            g._remove_from_view([path])
        elif chosen == act_del:
            g._batch_delete_files([path])
