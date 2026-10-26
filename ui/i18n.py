# ui/i18n.py
# ============================================================
#  轻量 UI 多语言：集中词典 + 控件树就地重翻译
#  - 不重建控件，因此不会丢失输入、模型选择或生成状态
#  - 只翻译已知静态文本；用户输入、文件名、日志和元数据保持原样
# ============================================================
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QEvent, QObject, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractButton, QApplication, QComboBox, QDoubleSpinBox, QGroupBox,
    QLabel, QLineEdit, QMenu, QPlainTextEdit, QSpinBox,
    QTabWidget, QTableWidget, QTextEdit, QTreeWidget, QWidget,
)

from ui.i18n_catalog import TRANSLATIONS

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "zh": "中文",
    "ja": "日本語",
    "en": "English",
}
LANGUAGE_ORDER = ("zh", "ja", "en")


class UiLanguageManager(QObject):
    """管理界面语言，并将静态文本就地翻译为中文/日文/英文。"""

    language_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._language = "zh"
        self._app: Optional[QApplication] = None
        self._reverse = {"en": {}, "ja": {}}
        for source, pair in TRANSLATIONS.items():
            for lang, translated in zip(("en", "ja"), pair):
                # 同语言内的重复译文保留先出现者，避免反向还原不稳定。
                self._reverse[lang].setdefault(translated, source)

    @property
    def language(self) -> str:
        return self._language

    def normalize(self, language: str) -> str:
        language = str(language or "zh").strip().lower().replace("_", "-")
        language = language.split("-", 1)[0]
        return language if language in SUPPORTED_LANGUAGES else "zh"

    def set_language(self, language: str) -> bool:
        language = self.normalize(language)
        if language == self._language:
            return False
        self._language = language
        logger.info(f"界面语言切换为: {SUPPORTED_LANGUAGES[language]}")
        self.language_changed.emit(language)
        self.translate_all_windows()
        return True

    def install(self, app: QApplication) -> None:
        """监听新窗口 Show 事件，使后打开的对话框也使用当前语言。"""
        if app is None or self._app is app:
            return
        if self._app is not None:
            self._app.removeEventFilter(self)
        self._app = app
        app.installEventFilter(self)

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.Show
                and isinstance(obj, QWidget) and obj.isWindow()):
            self.translate_widget_tree(obj)
        return False

    def _source(self, text: str) -> str:
        """从当前显示文字找回词典原文，支持 zh → en → ja 直接往返。"""
        if not text:
            return text
        if text in TRANSLATIONS:
            return text
        for lang in ("en", "ja"):
            source = self._reverse[lang].get(text)
            if source is not None:
                return source
        return text

    def tr(self, text: str, language: str | None = None) -> str:
        if not text:
            return text
        source = self._source(text)
        lang = self.normalize(language or self._language)
        if lang == "zh":
            return source
        pair = TRANSLATIONS.get(source)
        if not pair:
            return source
        translated = pair[0] if lang == "en" else pair[1]
        return translated or source

    def translate_all_windows(self) -> None:
        app = self._app or QApplication.instance()
        if app is None:
            return
        for widget in app.allWidgets():
            if widget.isWindow():
                self.translate_widget_tree(widget)

    def _translated(self, text: str) -> str:
        # 保留原文前后的空白，避免布局或快捷键标记被意外改动。
        if not text:
            return text
        leading = text[:len(text) - len(text.lstrip())]
        trailing = text[len(text.rstrip()):]
        core = text.strip()
        return leading + self.tr(core) + trailing

    def translate_widget_tree(self, root: QWidget) -> None:
        """就地翻译 root 及其子控件；未知文本和用户正文不做任何改动。"""
        if root is None:
            return
        widgets = [root] + root.findChildren(QWidget)
        for widget in widgets:
            try:
                self._translate_widget(widget)
            except (RuntimeError, AttributeError):
                # Qt 对象可能在窗口销毁期间进入列表；跳过即可。
                continue
        for action in root.findChildren(QAction):
            try:
                self._translate_action(action)
            except (RuntimeError, AttributeError):
                continue

    def _translate_action(self, action: QAction) -> None:
        action.setText(self._translated(action.text()))
        if action.toolTip():
            action.setToolTip(self._translated(action.toolTip()))
        if action.statusTip():
            action.setStatusTip(self._translated(action.statusTip()))

    def _translate_widget(self, widget: QWidget) -> None:
        if widget.isWindow() and widget.windowTitle():
            widget.setWindowTitle(self._translated(widget.windowTitle()))
        if widget.toolTip():
            widget.setToolTip(self._translated(widget.toolTip()))
        if widget.statusTip():
            widget.setStatusTip(self._translated(widget.statusTip()))

        if isinstance(widget, QLabel):
            widget.setText(self._translated(widget.text()))
        if isinstance(widget, QAbstractButton):
            widget.setText(self._translated(widget.text()))
        if isinstance(widget, QGroupBox):
            widget.setTitle(self._translated(widget.title()))
        if isinstance(widget, QMenu):
            widget.setTitle(self._translated(widget.title()))

        if isinstance(widget, QLineEdit):
            widget.setPlaceholderText(self._translated(widget.placeholderText()))
        if isinstance(widget, (QTextEdit, QPlainTextEdit)):
            widget.setPlaceholderText(self._translated(widget.placeholderText()))

        if (isinstance(widget, QComboBox)
                and widget.property("i18nItems") is True):
            # 很多生成逻辑读取 currentText()。只有明确标记为纯展示用途的
            # 下拉框才翻译选项，避免界面语言改变业务参数值。
            blocked = widget.blockSignals(True)
            index = widget.currentIndex()
            try:
                for i in range(widget.count()):
                    old = widget.itemText(i)
                    new = self._translated(old)
                    if new != old:
                        widget.setItemText(i, new)
                widget.setCurrentIndex(index)
            finally:
                widget.blockSignals(blocked)

        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setSuffix(self._translated(widget.suffix()))
            widget.setPrefix(self._translated(widget.prefix()))
            if isinstance(widget, QSpinBox):
                widget.setSpecialValueText(
                    self._translated(widget.specialValueText()))

        if isinstance(widget, QTabWidget):
            for i in range(widget.count()):
                widget.setTabText(i, self._translated(widget.tabText(i)))
                if widget.tabToolTip(i):
                    widget.setTabToolTip(
                        i, self._translated(widget.tabToolTip(i)))

        # QListWidget 常用于文件名/角色名等用户数据，不做条目正文翻译。
        if isinstance(widget, QTableWidget):
            for i in range(widget.columnCount()):
                item = widget.horizontalHeaderItem(i)
                if item is not None:
                    item.setText(self._translated(item.text()))
            for i in range(widget.rowCount()):
                item = widget.verticalHeaderItem(i)
                if item is not None:
                    item.setText(self._translated(item.text()))

        if isinstance(widget, QTreeWidget):
            item = widget.headerItem()
            for i in range(widget.columnCount()):
                item.setText(i, self._translated(item.text(i)))


ui_i18n = UiLanguageManager()


def tr(text: str) -> str:
    return ui_i18n.tr(text)
