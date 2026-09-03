# ui/pages/txt2img_page.py
# ============================================================
#  文生图页 — 中央工作区：预览画布 + 操作按钮 + 日志
#  从 ui_builder._build_right_panel（1509-1606 行）迁入，属性名不变
#  画廊已迁往独立画廊页（ui/pages/gallery_page.py）
# ============================================================
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTextEdit, QLabel, QSizePolicy)

from ui.pages.base import PageBase
from ui.widgets import GpuCanvas
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

        # ── 上半区: 预览画布（普通 / 修前修后对比 二合一）──
        host.lbl_preview = GpuCanvas()
        host.lbl_preview.setText("等待生成...")
        host.lbl_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        from ui.widgets import CompareCanvas
        from PyQt6.QtWidgets import QStackedWidget
        host.compare_canvas = CompareCanvas()
        host.preview_stack = QStackedWidget()
        host.preview_stack.addWidget(host.lbl_preview)      # 0 普通
        host.preview_stack.addWidget(host.compare_canvas)   # 1 对比
        host.preview_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        preview_wrap = QWidget()
        preview_layout = QVBoxLayout(preview_wrap)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)
        preview_layout.addWidget(host.preview_stack, 1)

        # ── 5 个操作按钮 ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        host.btn_open_editor = QPushButton("🖌️ 编辑")
        host.btn_save_as = QPushButton("💾 另存为")
        host.btn_send_img2img = QPushButton("🔄 转图生图")
        host.btn_send_inpaint = QPushButton("🎭 转重绘")
        host.btn_compare = QPushButton("⚖️ 对比")
        host.btn_compare.setCheckable(True)
        host.btn_compare.setEnabled(False)   # 有修前快照后启用
        host.btn_compare.setToolTip("修前 / 修后 对比滑条（需开启 Hires.fix 或 ADetailer）")
        for b in (host.btn_open_editor, host.btn_save_as,
                  host.btn_send_img2img, host.btn_send_inpaint,
                  host.btn_compare):
            btn_row.addWidget(b)
        preview_layout.addLayout(btn_row)
        _wire(host, host.btn_open_editor.clicked, "open_gallery_to_edit")
        _wire(host, host.btn_save_as.clicked, "save_current_image_as")
        _wire(host, host.btn_send_img2img.clicked, "send_preview_to_img2img")
        _wire(host, host.btn_send_inpaint.clicked, "send_preview_to_inpaint")
        _wire(host, host.btn_compare.toggled, "_on_compare_toggled")

        # ── 预览区直接铺满（画廊已迁往独立画廊页）──
        layout.addWidget(preview_wrap, 1)

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
