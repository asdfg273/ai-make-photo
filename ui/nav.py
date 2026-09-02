# ui/nav.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QToolButton, QButtonGroup
from PyQt6.QtCore import pyqtSignal, Qt


class NavRail(QWidget):
    """左侧导航栏：读页面注册表自动生成按钮，单选。"""
    page_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(64)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 8, 6, 8)
        self._layout.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QToolButton] = {}
        self._layout.addStretch(1)

    def set_pages(self, pages: list) -> None:
        for cls in pages:
            btn = QToolButton()
            btn.setObjectName("navBtn")          # theme.py 的 QSS 钩子
            btn.setText(f"{cls.icon}\n{cls.title}")
            btn.setCheckable(True)
            btn.setToolTip(cls.title)
            btn.clicked.connect(
                lambda _=False, pid=cls.page_id: self.page_selected.emit(pid))
            self._buttons[cls.page_id] = btn
            self._group.addButton(btn)
            self._layout.insertWidget(self._layout.count() - 1, btn)

    def select(self, page_id: str) -> None:
        btn = self._buttons.get(page_id)
        if btn is not None:
            btn.setChecked(True)
            self.page_selected.emit(page_id)
