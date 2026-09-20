# tests/test_inpaint_flow.py
# ============================================================
#  编辑器「发送到局部重绘」+ 图生图参考图缩略图 回归测试
#  用法: venv\Scripts\python tests\test_inpaint_flow.py
# ============================================================
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QLabel, QDialog
from PIL import Image

from photo_turn.pro_editor_qt import ProImageEditor
from utils.app_events import EventMixin

SRC = os.path.join("tests", "_inpaint_src.png")


def _make_src():
    Image.new("RGB", (320, 240), (100, 120, 140)).save(SRC)


def _paint_mask(ed):
    ed.toggle_mask_brush()
    ed.on_mouse_press(50, 50)
    ed.on_mouse_drag(120, 100)
    ed.on_mouse_release(120, 100)


def test_editor_send_to_inpaint(app):
    _make_src()

    # 1) 空遮罩 → 不触发回调,不关闭,状态栏警告
    called = []
    ed = ProImageEditor(None, SRC, callback_on_inpaint=lambda i, m: called.append(1))
    ed.send_to_inpaint()
    assert not called, "空遮罩不应触发回调"
    assert "遮罩为空" in ed.lbl_status.text()
    assert ed.result() != QDialog.DialogCode.Accepted
    print("1) 空遮罩拦截 OK")

    # 2) 画了遮罩 → 回调收到 (图, 遮罩),编辑器关闭
    _paint_mask(ed)
    assert ed.mask_img.getextrema()[1] > 0
    ed.send_to_inpaint()
    assert len(called) == 1, "应触发一次 inpaint 回调"
    assert ed.result() == QDialog.DialogCode.Accepted, "发送后应关闭编辑器"
    assert ed._dirty is False
    print("2) 遮罩发送 + 关闭 OK")

    # 3) 无专用回调 → 回退到 callback_on_save
    saved = []
    ed2 = ProImageEditor(None, SRC, callback_on_save=lambda i, m: saved.append(1))
    _paint_mask(ed2)
    ed2.send_to_inpaint()
    assert len(saved) == 1, "应回退到 callback_on_save"
    print("3) 回退 callback_on_save OK")


def test_ref_thumb(app):
    class Host(EventMixin):
        pass

    h = Host()
    h.lbl_img_path = QLabel()
    h.lbl_ref_thumb = QLabel()
    h.ref_image_path = None
    h.mask_image_path = None

    # 4) 仅参考图 → 有缩略图
    h.set_reference_image(SRC)
    pm = h.lbl_ref_thumb.pixmap()
    assert pm is not None and not pm.isNull(), "参考图缩略图应显示"
    assert "已加载底图" in h.lbl_img_path.text()
    print("4) 参考图缩略图 OK")

    # 5) 带遮罩 → 缩略图仍显示(红叠加),状态为局部重绘
    mask = Image.new("L", (320, 240), 0)
    for x in range(50, 120):
        for y in range(50, 100):
            mask.putpixel((x, y), 255)
    mask_path = os.path.join("tests", "_inpaint_mask.png")
    mask.save(mask_path)
    h.set_reference_image(SRC, mask_path)
    pm2 = h.lbl_ref_thumb.pixmap()
    assert pm2 is not None and not pm2.isNull()
    assert "局部重绘" in h.lbl_img_path.text()
    print("5) 遮罩叠加缩略图 OK")

    # 6) 清除 → 回到占位文本
    h.clear_reference()
    assert h.lbl_ref_thumb.text() == "无参考图"
    pm3 = h.lbl_ref_thumb.pixmap()
    assert pm3 is None or pm3.isNull()
    assert h.lbl_img_path.text() == "未选择参考图"
    print("6) 清除复位 OK")


def main():
    app = QApplication([])
    test_editor_send_to_inpaint(app)
    test_ref_thumb(app)
    for f in (SRC, os.path.join("tests", "_inpaint_mask.png")):
        if os.path.exists(f):
            os.remove(f)
    print("INPAINT_FLOW_OK")


if __name__ == "__main__":
    main()
