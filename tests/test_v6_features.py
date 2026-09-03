# tests/test_v6_features.py — v6 新功能测试（无头，纯脚本）
import os, sys, time, tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])
_WINDOWS = []


def _make_png(path, color=(200, 80, 40)):
    from PIL import Image
    Image.new("RGB", (64, 48), color).save(path, "PNG")


def _mini_app():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    _WINDOWS.append(win)
    win.setup_ui()
    return win


def test_compare_canvas():
    """对比画布：设置前后图 + 拖拽分割线 + 绘制不崩。"""
    from ui.widgets import CompareCanvas
    from PyQt6.QtGui import QPixmap
    tmp = tempfile.mkdtemp()
    a = os.path.join(tmp, "a.png"); _make_png(a, (255, 0, 0))
    b = os.path.join(tmp, "b.png"); _make_png(b, (0, 0, 255))
    cc = CompareCanvas()
    _WINDOWS.append(cc)
    cc.resize(400, 300)
    assert not cc.has_images()
    cc.set_images(QPixmap(a), QPixmap(b))
    assert cc.has_images()
    # 模拟拖拽
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    cc._set_ratio_from_x(100)
    assert abs(cc._ratio - 100 / 400) < 1e-6
    cc.grab()   # 触发 paintEvent，不崩即过
    print("PASS test_compare_canvas")


def test_compare_flow():
    """prefix_signal → 对比按钮启用；toggle 切换 preview_stack。"""
    win = _mini_app()
    tmp = tempfile.mkdtemp()
    pre = os.path.join(tmp, "pre.png"); _make_png(pre, (255, 0, 0))
    res = os.path.join(tmp, "res.png"); _make_png(res, (0, 0, 255))
    assert not win.btn_compare.isEnabled()
    win._on_prefix_image(pre)
    assert win.btn_compare.isEnabled()
    win.last_generated_path = res
    win._on_compare_toggled(True)
    assert win.preview_stack.currentIndex() == 1
    assert win.compare_canvas.has_images()
    win._on_compare_toggled(False)
    assert win.preview_stack.currentIndex() == 0
    print("PASS test_compare_flow")


def test_compare_toggle_without_snapshot():
    """无修前快照时 toggle 自动弹回并提示。"""
    win = _mini_app()
    win._prefix_image_path = None
    win._on_compare_toggled(True)
    assert win.preview_stack.currentIndex() == 0
    assert not win.btn_compare.isChecked()
    print("PASS test_compare_toggle_without_snapshot")


def test_resource_monitor():
    """状态栏资源监控标签存在且有内容。"""
    win = _mini_app()
    win._update_resource_label()
    txt = win.lbl_resource.text()
    assert "内存" in txt, f"资源标签异常: {txt}"
    assert win._resource_timer.isActive()
    win._resource_timer.stop()
    print("PASS test_resource_monitor")


def test_hover_preview():
    """悬浮预览：show 显示浮层 / hide 隐藏；视频不触发。"""
    from ui.gallery_panel import GalleryPanel
    g = GalleryPanel()
    _WINDOWS.append(g)
    tmp = tempfile.mkdtemp()
    png = os.path.join(tmp, "h.png"); _make_png(png)
    g._hover_path = png
    g._show_hover_preview()
    assert g._hover_label is not None
    assert not g._hover_label.pixmap().isNull()
    g._hide_hover_preview()
    assert g._hover_label.isHidden()
    print("PASS test_hover_preview")


def test_filmstrip_reuse_params():
    """胶片条点击图片 → 复用参数被调用。"""
    win = _mini_app()
    tmp = tempfile.mkdtemp()
    png = os.path.join(tmp, "f.png"); _make_png(png)
    calls = []
    win.reuse_params_from_path = lambda p: calls.append(p)
    win._on_filmstrip_clicked(png)
    assert calls == [png], f"复用参数未触发: {calls}"
    # 视频不触发
    win._on_filmstrip_clicked("x.mp4")
    assert calls == [png]
    print("PASS test_filmstrip_reuse_params")


if __name__ == "__main__":
    for fn in (test_compare_canvas, test_compare_flow,
               test_compare_toggle_without_snapshot, test_resource_monitor,
               test_hover_preview, test_filmstrip_reuse_params):
        fn()
    for w in _WINDOWS:
        w.close()
    _APP.processEvents()
    sys.stdout.flush()
    print("\n✅ 全部 v6 新功能测试通过")
    os._exit(0)
