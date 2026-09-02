# ui/pages/base.py
from PyQt6.QtWidgets import QWidget


class PageBase(QWidget):
    """页面基类。子类设 page_id/title/icon，实现 build()。
    扩展新功能 = 新建 page 文件 + ui/pages/__init__.py 的 PAGES 加一行。"""
    page_id: str = ""
    title: str = ""
    icon: str = ""

    def build(self, host) -> None:
        """构建页面控件；页面专属控件按契约名挂到 host。host 为主窗口。"""
        raise NotImplementedError

    def workspace(self) -> QWidget:
        """中央工作区内容。默认空。"""
        return QWidget()

    def params_widget(self) -> QWidget | None:
        """右侧参数面板的页面专属区。None = 本页无专属参数。"""
        return None
