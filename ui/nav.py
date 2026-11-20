# ui/nav.py
# 左侧导航栏：读页面注册表自动生成按钮，单选。
# 纯文字导航（无图标），选中态由 theme.py 的 QSS 钩子 #navBtn 渲染。
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QToolButton, QButtonGroup,
                             QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal


class NavRail(QWidget):
    """左侧导航栏：读页面注册表自动生成按钮，单选。"""
    page_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(136)
        self.setObjectName("navRail")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 12, 8, 12)
        self._layout.setSpacing(3)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QToolButton] = {}
        self._layout.addStretch(1)

    def set_pages(self, pages: list) -> None:
        from ui.icons import get_icon
        for cls in pages:
            btn = QToolButton()
            btn.setObjectName("navBtn")          # theme.py 的 QSS 钩子
            btn.setText(cls.title)
            btn.setCheckable(True)
            btn.setToolTip(cls.title)
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            # 图标接口：注册了 page_id（或页面自带的 icon key）就显示图标，
            # 未注册保持纯文字，行为与之前完全一致
            icon = get_icon(cls.page_id) or get_icon(getattr(cls, "icon", ""))
            if icon is not None:
                btn.setIcon(icon)
                btn.setToolButtonStyle(
                    Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            else:
                btn.setToolButtonStyle(
                    Qt.ToolButtonStyle.ToolButtonTextOnly)
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
