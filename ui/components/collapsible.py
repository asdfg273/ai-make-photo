# ui/components/collapsible.py
# 可折叠分组：标题行(点击展开/收起) + 内容区。LoRA/ControlNet/高级/X-Y 用
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QToolButton
from PyQt6.QtCore import Qt


class CollapsibleSection(QWidget):
    def __init__(self, title: str, collapsed: bool = True, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self._btn = QToolButton()
        self._btn.setText(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(not collapsed)
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._btn.setArrowType(Qt.ArrowType.RightArrow if collapsed
                               else Qt.ArrowType.DownArrow)
        self._btn.setStyleSheet(
            "QToolButton { background:#22303d; border:none; border-radius:6px;"
            " padding:8px; font-weight:bold; text-align:left; }")
        self._btn.toggled.connect(self._on_toggled)
        root.addWidget(self._btn)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 4, 8, 8)
        self.content_layout.setSpacing(6)
        self.content.setVisible(not collapsed)
        root.addWidget(self.content)

    def _on_toggled(self, checked: bool):
        self.content.setVisible(checked)
        self._btn.setArrowType(Qt.ArrowType.DownArrow if checked
                               else Qt.ArrowType.RightArrow)

    def set_collapsed(self, collapsed: bool):
        self._btn.setChecked(not collapsed)

    def is_collapsed(self) -> bool:
        return not self._btn.isChecked()
