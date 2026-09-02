# ui/components/filmstrip.py — 底部胶片条：最近媒体一览，点击跳画廊
import os
from PyQt6.QtWidgets import QListWidget, QListWidgetItem
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QIcon

from ui.gallery_panel import GalleryPanel


class FilmStrip(QListWidget):
    """横向缩略图条：展示最近生成的图片/动画，点击发出 media_clicked(path)。"""

    media_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setIconSize(QSize(96, 96))
        self.setFixedHeight(116)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
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
            item.setIcon(GalleryPanel._video_placeholder_icon())
        else:
            pix = QPixmap(path)
            if not pix.isNull():
                item.setIcon(QIcon(pix.scaled(
                    96, 96,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)))
        item.setToolTip(os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.addItem(item)

    def _on_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.media_clicked.emit(path)
