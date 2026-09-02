# tests/test_collapsible.py
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QApplication, QLabel

def main():
    app = QApplication([])
    from ui.components.collapsible import CollapsibleSection
    sec = CollapsibleSection("LoRA", collapsed=True)
    assert sec.is_collapsed() is True
    assert sec.content.isHidden()
    sec.content_layout.addWidget(QLabel("x"))
    sec.set_collapsed(False)
    assert not sec.content.isHidden()
    print("PASS test_collapsible")

if __name__ == "__main__":
    main()
