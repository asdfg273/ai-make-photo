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


def _make_png_with_params(path):
    """带 A1111 格式 parameters 元数据的 PNG。"""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    meta = PngInfo()
    meta.add_text("parameters",
                  "1girl, black hair, school uniform\n"
                  "Negative prompt: lowres, bad anatomy\n"
                  "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7.5, "
                  "Seed: 12345, Size: 832x1216, Model: testModel_v1.safetensors")
    Image.new("RGB", (64, 48), (10, 20, 30)).save(path, "PNG", pnginfo=meta)


def test_filmstrip_full_backfill():
    """胶片条复用 = 全量回填 prompt/负面/steps/cfg/seed/采样器/分辨率/模型。"""
    import types
    import main as _m
    from ui.param_snapshot import ParamSnapshotMixin

    win = _mini_app()
    # 嫁接 main 的真实方法与 ParamSnapshotMixin 的安全设值
    win.reuse_params_from_path = types.MethodType(
        _m.AIDesktopApp.reuse_params_from_path, win)
    win._on_apply_gallery_params = types.MethodType(
        _m.AIDesktopApp._on_apply_gallery_params, win)
    for n in ("_safe_set_int", "_safe_set_float", "_safe_set_combo"):
        setattr(win, n, types.MethodType(getattr(ParamSnapshotMixin, n), win))
    win._set_status = lambda *a, **k: None
    win.combo_model.addItem("testModel_v1.safetensors")

    tmp = tempfile.mkdtemp()
    png = os.path.join(tmp, "p.png"); _make_png_with_params(png)
    win.reuse_params_from_path(png)

    assert win.txt_prompt.toPlainText() == "1girl, black hair, school uniform"
    assert win.txt_neg.toPlainText() == "lowres, bad anatomy"
    assert win.spin_steps.value() == 30
    assert win.spin_seed.value() == 12345
    assert win.combo_res.currentText() == "832x1216", win.combo_res.currentText()
    assert win.combo_sampler.currentText() == "DPM++ 2M Karras"
    assert "testModel_v1" in win.combo_model.currentText()
    cfg = win.scale_cfg.value() if hasattr(win.scale_cfg, "value") \
        else win.scale_cfg.get_value()
    assert abs(float(cfg) - 7.5) < 1e-6, f"cfg={cfg}"

    # 无参数图 → 状态提示，不炸
    plain = os.path.join(tmp, "plain.png"); _make_png(plain)
    win.reuse_params_from_path(plain)
    print("PASS test_filmstrip_full_backfill")


def test_trans_compare():
    """翻译回译对比：按钮存在；空提示词早退；stub 翻译三轮内容正确。"""
    import time as _t
    from PyQt6.QtWidgets import QTextEdit

    win = _mini_app()
    assert hasattr(win, "btn_trans_compare"), "缺少回译对比按钮"

    # 空提示词 → 直接返回，不起线程
    win.txt_prompt.setPlainText("")
    win._on_trans_compare()

    class _FakeTr:
        qwen_enhancer = object()   # 假装 Qwen 就绪

        def translate(self, text, mode="auto", target_lang="en"):
            return "cat, girl" if target_lang == "en" else "猫, 女孩"

    win.translator = _FakeTr()
    win.txt_prompt.setPlainText("一个女孩和猫")
    win._on_trans_compare()

    deadline = _t.time() + 5
    while not win.btn_trans_compare.isEnabled() and _t.time() < deadline:
        _APP.processEvents()
        _t.sleep(0.05)
    assert win.btn_trans_compare.isEnabled(), "回译线程未收尾"

    dlg = getattr(win, "_trans_compare_dlg", None)
    assert dlg is not None, "对比弹窗未创建"
    texts = [te.toPlainText() for te in dlg.findChildren(QTextEdit)]
    assert texts[0] == "一个女孩和猫", texts
    assert texts[1] == "cat, girl", texts
    assert texts[2] == "猫, 女孩", texts
    print("PASS test_trans_compare")


def test_img2img_compare_snapshot():
    """图生图对比：生成管线含参考图快照逻辑（静态检查 + 信号流）。"""
    import inspect
    from utils.app_generation import GenerationMixin
    src = inspect.getsource(GenerationMixin)
    assert "ref_image_path" in src and "prefix_signal.emit(_ref)" in src, \
        "缺少图生图参考图快照逻辑"
    # 信号流：参考图路径 → _on_prefix_image → 对比可用
    win = _mini_app()
    tmp = tempfile.mkdtemp()
    ref = os.path.join(tmp, "ref.png"); _make_png(ref, (0, 255, 0))
    res = os.path.join(tmp, "res.png"); _make_png(res, (0, 0, 255))
    win._on_prefix_image(ref)
    win.last_generated_path = res
    win._on_compare_toggled(True)
    assert win.preview_stack.currentIndex() == 1
    assert win.compare_canvas.has_images()
    print("PASS test_img2img_compare_snapshot")


if __name__ == "__main__":
    for fn in (test_compare_canvas, test_compare_flow,
               test_compare_toggle_without_snapshot, test_resource_monitor,
               test_hover_preview, test_filmstrip_reuse_params,
               test_filmstrip_full_backfill, test_trans_compare,
               test_img2img_compare_snapshot):
        fn()
    for w in _WINDOWS:
        w.close()
    _APP.processEvents()
    sys.stdout.flush()
    print("\n✅ 全部 v6 新功能测试通过")
    os._exit(0)
