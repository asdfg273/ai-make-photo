# tests/test_nav.py
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication([])
    from ui.nav import NavRail
    from ui.pages.base import PageBase

    class DummyPage(PageBase):
        page_id, title, icon = "dummy", "测试", "🧪"
        def build(self, host): pass

    rail = NavRail()
    got = []
    rail.page_selected.connect(got.append)
    rail.set_pages([DummyPage])
    rail.select("dummy")
    assert got == ["dummy"]
    assert rail._buttons["dummy"].isChecked()
    print("PASS test_nav")

if __name__ == "__main__":
    main()
