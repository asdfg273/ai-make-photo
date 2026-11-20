# tests/test_edit_button.py — 主界面「🖌️ 编辑」按钮：编辑预览区当前这张图
#
# 为什么要有这个文件
# ------------------
# 需求（用户）: 「主界面的编辑按钮目前是打开输出文件夹然后选择图片进行编辑，
#                修改为打开现在正在预览的图片进行编辑」
#
# 改动前: btn_open_editor → open_gallery_to_edit → QFileDialog.getOpenFileName
#         （先弹文件管理器让你自己找图）
# 改动后: btn_open_editor → open_editor → 直接拿预览区那张（current_generated_path）
#
# 这里守三件事:
#   1. 有预览图时**绝不弹文件对话框**，且交给 ProImageEditor 的就是预览区那张；
#   2. 预览图路径失效时退回 last_generated_path，而不是直接罢工；
#   3. 一张图都没有时才问一句「要不要从磁盘选」，选「否」什么都不打开。
# 另加一条: btn_edit 别名必须指向同一个按钮 —— 生成中要禁用它的那 5 处
# `hasattr(self, "btn_edit")` 以前全部落空（别名从未赋值）。
#
# 用法：venv\Scripts\python.exe tests\test_edit_button.py
import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码 + 配置隔离）

_bootstrap.isolate_config_globally()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WIN = None


def _make_png(name: str) -> str:
    """在项目内写一张真 PNG（不用 tempfile: 本机 ACL 会锁死临时目录）。"""
    from PIL import Image

    out_dir = os.path.join(_ROOT, "tests", "artifacts", "tmp")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    Image.new("RGB", (8, 8), (12, 34, 56)).save(path)
    return path


def _win():
    """整轮测试共用一个真实 AIDesktopApp（构造很贵）。"""
    global _WIN
    if _WIN is None:
        import main

        _WIN = main.AIDesktopApp()
        _APP.processEvents()
    return _WIN


class _FakeEditor:
    """替身 ProImageEditor：只记录被要求编辑哪张图。"""

    opened = []

    def __init__(self, parent, path, **kwargs):
        _FakeEditor.opened.append(
            {"path": str(path), "kwargs": sorted(kwargs)})
        self._parent = parent

    def exec(self):
        return 0

    def run_adetailer(self):  # 「发送到修脸」会 QTimer.singleShot 调它
        pass


def _patch_editor(monkeypatch_store):
    import utils.app_events as app_events

    monkeypatch_store["orig_editor"] = app_events.ProImageEditor
    app_events.ProImageEditor = _FakeEditor


def _unpatch_editor(monkeypatch_store):
    import utils.app_events as app_events

    app_events.ProImageEditor = monkeypatch_store["orig_editor"]


def _block_file_dialog(monkeypatch_store, result=("", "")):
    """把 QFileDialog.getOpenFileName 换掉：记录调用次数，返回给定结果。"""
    import utils.app_events as app_events

    monkeypatch_store["dialog_calls"] = []

    def fake(*args, **kwargs):
        monkeypatch_store["dialog_calls"].append(args)
        return result

    monkeypatch_store["orig_dialog"] = app_events.QFileDialog.getOpenFileName
    app_events.QFileDialog.getOpenFileName = staticmethod(fake)


def _unblock_file_dialog(monkeypatch_store):
    import utils.app_events as app_events

    app_events.QFileDialog.getOpenFileName = monkeypatch_store["orig_dialog"]


# ----------------------------------------------------------------------
#  1) 有预览图 → 直接编辑它，不弹文件对话框
# ----------------------------------------------------------------------
def test_button_edits_preview_image_without_file_dialog():
    win = _win()
    store = {}
    _patch_editor(store)
    _block_file_dialog(store)
    try:
        path = _make_png("edit_btn_preview.png")
        win.current_generated_path = path
        win.last_generated_path = path
        _FakeEditor.opened.clear()

        win.btn_open_editor.click()
        _APP.processEvents()

        assert not store["dialog_calls"], \
            "点了「编辑」还是弹了文件对话框 —— 应该直接编辑预览区那张图"
        assert _FakeEditor.opened, "「编辑」没有打开修图器"
        got = _FakeEditor.opened[-1]["path"]
        assert os.path.normcase(got) == os.path.normcase(os.path.abspath(path)), \
            f"编辑的不是预览区那张图: 期望 {path} 实际 {got}"
    finally:
        _unblock_file_dialog(store)
        _unpatch_editor(store)


# ----------------------------------------------------------------------
#  2) current_generated_path 失效 → 退回 last_generated_path
# ----------------------------------------------------------------------
def test_falls_back_to_last_generated_path():
    win = _win()
    store = {}
    _patch_editor(store)
    _block_file_dialog(store)
    try:
        path = _make_png("edit_btn_last.png")
        win.current_generated_path = os.path.join(
            _ROOT, "tests", "artifacts", "tmp", "does_not_exist.png")
        win.last_generated_path = path
        _FakeEditor.opened.clear()

        win.open_editor()
        _APP.processEvents()

        assert not store["dialog_calls"], "有可用预览图时不该弹文件对话框"
        assert _FakeEditor.opened, "回退到 last_generated_path 失败：没打开修图器"
        got = _FakeEditor.opened[-1]["path"]
        assert os.path.normcase(got) == os.path.normcase(os.path.abspath(path)), \
            f"回退后编辑的图不对: 期望 {path} 实际 {got}"
    finally:
        _unblock_file_dialog(store)
        _unpatch_editor(store)


# ----------------------------------------------------------------------
#  3) 一张图都没有 → 先问一句；选「否」什么都不打开
# ----------------------------------------------------------------------
def test_no_preview_asks_first_and_respects_no():
    win = _win()
    import utils.app_events as app_events

    store = {}
    _patch_editor(store)
    _block_file_dialog(store)
    asked = []
    orig_question = app_events.QMessageBox.question

    def fake_question(parent, title, text, *args, **kwargs):
        asked.append((title, text))
        return app_events.QMessageBox.StandardButton.No

    app_events.QMessageBox.question = staticmethod(fake_question)
    try:
        win.current_generated_path = None
        win.last_generated_path = None
        _FakeEditor.opened.clear()

        win.open_editor()
        _APP.processEvents()

        assert asked, "预览区没图时应该问一句，而不是静默什么都不做"
        assert not _FakeEditor.opened, "选了「否」却还是打开了修图器"
        assert not store["dialog_calls"], "选了「否」却还是弹了文件对话框"
    finally:
        app_events.QMessageBox.question = orig_question
        _unblock_file_dialog(store)
        _unpatch_editor(store)


# ----------------------------------------------------------------------
#  4) 选「是」→ 走磁盘选图，选中后还是交给同一个编辑入口
# ----------------------------------------------------------------------
def test_no_preview_yes_opens_picker_and_edits_picked_file():
    win = _win()
    import utils.app_events as app_events

    picked = _make_png("edit_btn_picked.png")
    store = {}
    _patch_editor(store)
    _block_file_dialog(store, result=(picked, "Image files (*.png)"))
    orig_question = app_events.QMessageBox.question
    app_events.QMessageBox.question = staticmethod(
        lambda *a, **k: app_events.QMessageBox.StandardButton.Yes)
    try:
        win.current_generated_path = None
        win.last_generated_path = None
        _FakeEditor.opened.clear()

        win.open_editor()
        _APP.processEvents()

        assert store["dialog_calls"], "选了「是」却没打开文件对话框"
        assert _FakeEditor.opened, "选完文件后没打开修图器"
        got = _FakeEditor.opened[-1]["path"]
        assert os.path.normcase(got) == os.path.normcase(os.path.abspath(picked)), \
            f"编辑的不是刚选中的图: 期望 {picked} 实际 {got}"
    finally:
        app_events.QMessageBox.question = orig_question
        _unblock_file_dialog(store)
        _unpatch_editor(store)


# ----------------------------------------------------------------------
#  5) btn_edit 别名 == btn_open_editor（生成中禁用那 5 处才生效）
# ----------------------------------------------------------------------
def test_btn_edit_alias_points_at_editor_button():
    win = _win()
    assert getattr(win, "btn_edit", None) is win.btn_open_editor, \
        "btn_edit 别名没指向「编辑」按钮 —— 生成中禁用/恢复会全部落空"
    win.btn_edit.setEnabled(False)
    assert not win.btn_open_editor.isEnabled()
    win.btn_edit.setEnabled(True)
    assert win.btn_open_editor.isEnabled()


# ----------------------------------------------------------------------
#  6) v6 页面里接的是 open_editor，不是老的"弹文件夹选图"
# ----------------------------------------------------------------------
def test_txt2img_page_wires_open_editor():
    path = os.path.join(_ROOT, "ui", "pages", "txt2img_page.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    assert '"open_editor"' in source, "txt2img_page 没把「编辑」接到 open_editor"
    assert '"open_gallery_to_edit"' not in source, \
        "txt2img_page 还接着老的 open_gallery_to_edit（会先弹文件夹选图）"

    builder_path = os.path.join(_ROOT, "ui", "ui_builder.py")
    with open(builder_path, encoding="utf-8") as handle:
        builder = handle.read()
    assert "btn_open_editor.clicked.connect(self.open_editor)" in builder, \
        "ui_builder 里「编辑」没接到 open_editor"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            failed.append(fn.__name__)
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("=" * 60)
    print(f"总计 {len(fns)} 项：PASS {len(fns) - len(failed)} / FAIL {len(failed)}")
    for f in failed:
        print(f"  失败: {f}")
    sys.stdout.flush()
    if _WIN is not None:
        try:
            _WIN.ai = None  # 别在退出时卸载模型
            _WIN.close()
        except Exception:
            pass
    os._exit(1 if failed else 0)  # 跳过后台 torch 线程与 Qt teardown
