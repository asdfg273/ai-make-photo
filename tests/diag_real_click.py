# tests/diag_real_click.py — 真实窗口诊断：按钮渲染 + 胶片条回填（无头）
import os, sys, time, glob
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
_APP = QApplication.instance() or QApplication([])

from ui.theme import apply_theme
apply_theme(_APP)

import main
win = main.AIDesktopApp()
win.resize(1500, 950)
win.show()
deadline = time.time() + 3
while time.time() < deadline:
    _APP.processEvents(); time.sleep(0.05)

# ── 1. 回译对比按钮 ──
btn = getattr(win, "btn_trans_compare", None)
print("① btn_trans_compare 存在:", btn is not None)
if btn is not None:
    print("   visible:", btn.isVisible(), "size:", btn.size().width(), "x", btn.size().height(),
          "text:", btn.text())

# ── 2. 真实 PNG 直接回填 ──
png = sorted(glob.glob(os.path.join("photo", "*.png")), key=os.path.getmtime)[-1]
print("② 测试图:", os.path.basename(png))
win.txt_prompt.setPlainText("BEFORE")
win.spin_steps.setValue(5)
win.reuse_params_from_path(png)
print("   prompt 头:", win.txt_prompt.toPlainText()[:40])
print("   steps:", win.spin_steps.value(), "| seed:", win.spin_seed.value(),
      "| res:", win.combo_res.currentText(), "| sampler:", win.combo_sampler.currentText(),
      "| model:", win.combo_model.currentText()[:40])
try:
    cfg = win.scale_cfg.value() if hasattr(win.scale_cfg, "value") else win.scale_cfg.get_value()
except Exception as e:
    cfg = f"读取失败 {e}"
print("   cfg:", cfg)

# ── 3. 胶片条信号路径回填 ──
win.txt_prompt.setPlainText("BEFORE2")
win.spin_steps.setValue(7)
win.filmstrip.media_clicked.emit(png)
_APP.processEvents()
print("③ 胶片条 emit 后 prompt 头:", win.txt_prompt.toPlainText()[:40],
      "| steps:", win.spin_steps.value())

# ── 4. 截图：整体 + 核心区 ──
os.makedirs("tests/shots", exist_ok=True)
win.nav.select("txt2img")
_APP.processEvents()
win.grab().save("tests/shots/diag_full.png")
print("④ 截图已存 tests/shots/diag_full.png")

sys.stdout.flush()
os._exit(0)
