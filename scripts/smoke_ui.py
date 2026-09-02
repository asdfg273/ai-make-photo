# scripts/smoke_ui.py — v6 外壳冒烟：无模型加载起壳，3 秒后自动退出
# 用法: venv/Scripts/python.exe scripts/smoke_ui.py [--hold]（--hold 不自动退出，人工走查）
import os, sys
os.environ.setdefault("AI_STUDIO_UI", "v2")
os.environ.setdefault("QT_QPA_PLATFORM", "windows" if "--hold" in sys.argv else "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QTimer

from ui.theme import apply_theme
from ui.shell import ShellMixin


class SmokeApp(QMainWindow, ShellMixin):
    pass


def main():
    app = QApplication(sys.argv)
    theme = apply_theme(app)
    win = SmokeApp()
    win.setup_ui()
    win.resize(1440, 900)
    win.show()
    print(f"[smoke] theme={theme}, pages={list(win._pages.keys())}")
    print(f"[smoke] gallery={'OK' if getattr(win, 'gallery', None) is not None else 'MISSING'}, "
          f"filmstrip={'OK' if getattr(win, 'filmstrip', None) is not None else 'MISSING'}")
    if "--hold" in sys.argv:
        print("[smoke] --hold 模式：窗口保持，关闭窗口退出")
        sys.exit(app.exec())
    QTimer.singleShot(3000, app.quit)
    app.exec()
    print("[smoke] 3 秒无崩溃，冒烟通过")


if __name__ == "__main__":
    main()
