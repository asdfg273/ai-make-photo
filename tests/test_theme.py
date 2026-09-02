# tests/test_theme.py — 主题模块契约测试（无头）
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication([])
    from ui.theme import apply_theme, PALETTE, APP_QSS
    used = apply_theme(app)
    assert used in ("qdarkstyle", "fusion-fallback"), used
    assert isinstance(PALETTE, dict) and "accent" in PALETTE
    assert "QPushButton" in APP_QSS
    ss = app.styleSheet()
    assert len(ss) > 100, "样式表未生效"
    print(f"PASS test_theme (theme={used})")

if __name__ == "__main__":
    main()
