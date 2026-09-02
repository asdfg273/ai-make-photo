# ui/pages/gallery_page.py
# ============================================================
#  画廊页 — 统一图片/动画画廊
#  顶部工具条：媒体三态切换（全部/图片/动画），搜索/收藏/废弃
#  过滤保留在 GalleryPanel 自带的搜索行里（不重造）。
#  host.gallery 全局单例在此创建并挂入本页工作区。
# ============================================================
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QButtonGroup)
from PyQt6.QtCore import Qt

from ui.pages.base import PageBase
from ui.gallery_panel import GalleryPanel

logger = logging.getLogger(__name__)


class GalleryPage(PageBase):
    page_id, title, icon = "gallery", "画廊", "🖼️"

    def build(self, host):
        self._host = host
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── 顶部工具条：媒体类型切换 ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        lbl = QLabel("媒体类型:")
        lbl.setProperty("role", "hint")
        toolbar.addWidget(lbl)

        self._media_group = QButtonGroup(w)
        self._media_group.setExclusive(True)
        btn_all = QPushButton("🗂️ 全部")
        btn_img = QPushButton("🖼️ 图片")
        btn_vid = QPushButton("🎬 动画")
        btn_all.setObjectName("btnMediaAll")
        btn_img.setObjectName("btnMediaImage")
        btn_vid.setObjectName("btnMediaVideo")
        for key, b in (("all", btn_all), ("image", btn_img), ("video", btn_vid)):
            b.setCheckable(True)
            b.setProperty("mediaMode", key)
            self._media_group.addButton(b)
            toolbar.addWidget(b)
        btn_all.setChecked(True)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # ── 画廊单例（全局控件，契约名 host.gallery）──
        host.gallery = GalleryPanel()
        if hasattr(host, "_on_gallery_picked"):
            host.gallery.image_selected.connect(host._on_gallery_picked)
        if hasattr(host, "apply_meta_params"):
            host.gallery.apply_params_signal.connect(host.apply_meta_params)

        # ── 内嵌元数据侧栏（替代 📋 浮窗；选中即显示，点"套用"回填参数）──
        from PyQt6.QtWidgets import QSplitter
        meta = host.gallery.meta_panel
        meta.setWindowFlags(Qt.WindowType.Widget)   # 浮窗 → 内嵌子控件
        meta.setVisible(True)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(host.gallery)
        splitter.addWidget(meta)
        splitter.setSizes([1000, 340])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setChildrenCollapsible(True)       # 侧栏可拖没
        layout.addWidget(splitter, 1)
        self._meta_panel = meta

        # 工具条 → 画廊过滤
        self._media_group.buttonClicked.connect(
            lambda b: host.gallery.set_media_filter(b.property("mediaMode")))

        self._workspace = w

    def workspace(self) -> QWidget:
        return self._workspace

    def params_widget(self):
        return None  # 元数据沿用 GalleryPanel 的 📋 浮窗（偏离 spec 内嵌面板，保留现有行为）
