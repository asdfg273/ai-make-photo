# ui/i18n.py
# ============================================================
#  轻量 UI 多语言：集中词典 + 控件树就地重翻译
#  - 不重建控件，因此不会丢失输入、模型选择或生成状态
#  - 只翻译已知静态文本；用户输入、文件名、日志和元数据保持原样
# ============================================================
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QEvent, QLibraryInfo, QObject, QTranslator, pyqtSignal
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

# PyQt6 自带的 qtbase 翻译仍保留这两条英文；集中补齐，支持三语言往返。
# 格式：Qt 英文原文 -> (中文, 日本語)，快捷键标记予以保留。
QT_STANDARD_TRANSLATIONS = {
    "&Look in:": ("查找范围(&L)：", "参照位置(&L):"),
    "Files of &type:": ("文件类型(&T)：", "ファイルの種類(&T):"),
}


class UiLanguageManager(QObject):
    """管理界面语言，并将静态文本就地翻译为中文/日文/英文。"""

    language_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._language = "zh"
        self._app: Optional[QApplication] = None
        self._qt_translator: Optional[QTranslator] = None
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
        self._sync_qt_translator()
        logger.info(f"界面语言切换为: {SUPPORTED_LANGUAGES[language]}")
        self.language_changed.emit(language)
        self.translate_all_windows()
        return True

    def install(self, app: QApplication) -> None:
        """监听新窗口 Show 事件，并同步 Qt 自带控件的界面翻译。"""
        if app is None or self._app is app:
            return
        if self._app is not None:
            self._app.removeEventFilter(self)
        self._app = app
        app.installEventFilter(self)
        self._sync_qt_translator()

    def _sync_qt_translator(self) -> None:
        """让 Qt 标准编辑菜单等系统文案与应用语言保持一致。"""
        app = self._app or QApplication.instance()
        if app is None:
            return
        if self._qt_translator is not None:
            app.removeTranslator(self._qt_translator)
            self._qt_translator.deleteLater()
            self._qt_translator = None
        if self._language == "en":
            return

        translator = QTranslator(self)
        locale = "zh_CN" if self._language == "zh" else "ja"
        directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        if not translator.load(f"qtbase_{locale}", directory):
            logger.warning("无法加载 Qt %s 界面翻译", SUPPORTED_LANGUAGES[self._language])
            translator.deleteLater()
            return
        app.installTranslator(translator)
        # 必须持有引用；局部变量被回收后，Qt 菜单会退回默认英文。
        self._qt_translator = translator

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.LanguageChange
                and isinstance(obj, QLabel)):
            core = obj.text().strip()
            if any(core == source or core in pair
                   for source, pair in QT_STANDARD_TRANSLATIONS.items()):
                # 消耗这两个漏译标签的事件，防止 QFileDialog 随后把它们
                # 重新写成英文；其他标签仍由 Qt 原生翻译机制处理。
                obj.setText(self._translated(obj.text()))
                return True
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

    def trf(self, text: str, /, **values) -> str:
        """翻译带命名占位符的动态文案，再填回文件名、数量和错误正文。"""
        return self.tr(text).format(**values)

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
        core = self.tr(text.strip())
        for source, pair in QT_STANDARD_TRANSLATIONS.items():
            if core == source or core in pair:
                core = source if self._language == "en" else (
                    pair[0] if self._language == "zh" else pair[1])
                break
        return leading + core + trailing

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
