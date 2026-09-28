# ui/workflow_editor/__init__.py
"""工作流可视化编辑器（基础版）。

对外只需要 `WorkflowEditorWindow`：主窗口把它当成普通对话框打开即可。

    from ui.workflow_editor import open_editor
    open_editor(host=self)          # host = 主窗口（提供 ai / config）
"""
from __future__ import annotations

__all__ = ["WorkflowEditorWindow", "open_editor"]


def __getattr__(name):
    """惰性导入：避免只 import 包就把整套画布/Qt 图形项建起来。"""
    if name == "WorkflowEditorWindow":
        from ui.workflow_editor.window import WorkflowEditorWindow
        return WorkflowEditorWindow
    raise AttributeError(name)


def open_editor(host=None, parent=None) -> "object":
    """打开（或复用）编辑器窗口。返回窗口对象。

    用 `WA_DeleteOnClose` + 在宿主上挂引用，避免"打开多次留下多份实例"
    或者"窗口被 GC 掉导致闪退"。
    """
    from PyQt6.QtCore import Qt
    from ui.workflow_editor.window import WorkflowEditorWindow

    existing = getattr(host, "_workflow_editor", None) if host is not None else None
    if existing is not None:
        try:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing
        except RuntimeError:
            pass          # 底层 C++ 对象已销毁，下面重建

    win = WorkflowEditorWindow(host=host, parent=parent)
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    if host is not None:
        host._workflow_editor = win
    win.show()
    return win
