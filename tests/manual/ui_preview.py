# ui_preview.py — 离屏渲染主界面截图（重构视觉验证用手动脚本）
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from ui.theme import apply_theme
apply_theme(_APP)

import main
win = main.AIDesktopApp()
win.resize(1600, 950)
deadline = time.time() + 3
while time.time() < deadline:
    _APP.processEvents()
    time.sleep(0.05)

out = str(ROOT / "tests" / "artifacts" / "ui_preview" / "ui_preview.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
win.grab().save(out)
print("saved:", out)

# 再截动画页/画廊页/角色库页
for pid in ("video", "gallery", "characters"):
    win.nav.select(pid)
    _APP.processEvents()
    time.sleep(0.3)
    _APP.processEvents()
    p2 = out.replace(".png", f"_{pid}.png")
    win.grab().save(p2)
    print("saved:", p2)

os._exit(0)
