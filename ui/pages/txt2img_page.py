# ui/pages/txt2img_page.py
# ============================================================
#  文生图页 — 中央工作区：预览画布 + 操作按钮 + 画廊(暂) + 日志
#  从 ui_builder._build_right_panel（1509-1606 行）迁入，属性名不变
# ============================================================
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTextEdit, QLabel, QSplitter, QSizePolicy)
from PyQt6.QtCore import Qt

from ui.pages.base import PageBase
from ui.widgets import GpuCanvas
from ui.gallery_panel import GalleryPanel
from ui.core_panel import _wire

logger = logging.getLogger(__name__)


class Txt2ImgPage(PageBase):
    page_id, title, icon = "txt2img", "文生图", "🎨"

    def build(self, host):
        self._host = host
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── 上半区: 预览画布 ──
        host.lbl_preview = GpuCanvas()
        host.lbl_preview.setText("等待生成...")
        host.lbl_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        preview_wrap = QWidget()
        preview_layout = QVBoxLayout(preview_wrap)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)
        preview_layout.addWidget(host.lbl_preview, 1)

        # ── 4 个操作按钮 ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        host.btn_open_editor = QPushButton("🖌️ 编辑")
        host.btn_save_as = QPushButton("💾 另存为")
        host.btn_send_img2img = QPushButton("🔄 转图生图")
        host.btn_send_inpaint = QPushButton("🎭 转重绘")
        for b in (host.btn_open_editor, host.btn_save_as,
                  host.btn_send_img2img, host.btn_send_inpaint):
            btn_row.addWidget(b)
        preview_layout.addLayout(btn_row)
        _wire(host, host.btn_open_editor.clicked, "open_gallery_to_edit")
        _wire(host, host.btn_save_as.clicked, "save_current_image_as")
        _wire(host, host.btn_send_img2img.clicked, "send_preview_to_img2img")
        _wire(host, host.btn_send_inpaint.clicked, "send_preview_to_inpaint")

        # ── 下半区: 画廊（Task 11 迁往独立画廊页，此处先保留）──
        gallery_wrap = QWidget()
        gallery_layout = QVBoxLayout(gallery_wrap)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        gallery_layout.setSpacing(2)

        lbl_gallery_title = QLabel("🖼️ 历史画廊 (双击大图 · 右键菜单)")
        lbl_gallery_title.setProperty("role", "title")
        gallery_layout.addWidget(lbl_gallery_title)

        host.gallery = GalleryPanel()
        host.gallery.setMinimumHeight(180)
        if hasattr(host, "_on_gallery_picked"):
            host.gallery.image_selected.connect(host._on_gallery_picked)
        if hasattr(host, 'apply_meta_params'):
            host.gallery.apply_params_signal.connect(host.apply_meta_params)
        gallery_layout.addWidget(host.gallery, 1)

        # ── 上下分割，可拖动 ──
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(preview_wrap)
        right_splitter.addWidget(gallery_wrap)
        right_splitter.setSizes([500, 400])
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setHandleWidth(4)
        layout.addWidget(right_splitter, 1)
        host.right_splitter = right_splitter

        # ── 日志 ──
        lbl_log = QLabel("📋 生成日志:")
        lbl_log.setProperty("role", "hint")
        layout.addWidget(lbl_log)
        host.txt_log_image = QTextEdit()
        host.txt_log_image.setReadOnly(True)
        host.txt_log_image.setMaximumHeight(140)
        layout.addWidget(host.txt_log_image, 1)

        self._workspace = w

    def workspace(self) -> QWidget:
        return self._workspace

    def params_widget(self):
        return None  # 文生图无专属参数，核心区已覆盖
