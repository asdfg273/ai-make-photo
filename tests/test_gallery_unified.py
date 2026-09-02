# tests/test_gallery_unified.py — 统一画廊（图片/动画）+ 防抖测试（无头，纯脚本）
import os, sys, time, tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])
_WINDOWS = []


def _make_png(path: str):
    from PIL import Image
    img = Image.new("RGB", (32, 32), (200, 80, 40))
    img.save(path, "PNG")


def _drain(ms: int = 400):
    """推进事件循环，等防抖定时器触发。"""
    deadline = time.time() + ms / 1000.0
    while time.time() < deadline:
        _APP.processEvents()
        time.sleep(0.01)


def _fresh_gallery():
    from ui.gallery_panel import GalleryPanel
    g = GalleryPanel()
    _WINDOWS.append(g)
    return g


def test_media_filter_switch():
    """set_media_filter 三态切换 + 视频条目进入画廊。"""
    g = _fresh_gallery()
    tmp = tempfile.mkdtemp()
    png = os.path.join(tmp, "a.png"); _make_png(png)
    mp4 = os.path.join(tmp, "b.mp4"); open(mp4, "wb").write(b"\x00" * 64)

    assert hasattr(g, "set_media_filter"), "缺少 set_media_filter"
    assert hasattr(g, "add_media"), "缺少 add_media"

    g.add_media(png)
    g.add_media(mp4)
    _drain()

    g.set_media_filter("all")
    assert g.list_widget.count() == 2, f"all 应为 2，实际 {g.list_widget.count()}"
    g.set_media_filter("image")
    assert g.list_widget.count() == 1, "image 模式应只剩图片"
    g.set_media_filter("video")
    assert g.list_widget.count() == 1, "video 模式应只剩动画"
    from PyQt6.QtCore import Qt
    p = g.list_widget.item(0).data(Qt.ItemDataRole.UserRole)
    assert p.endswith(".mp4"), f"video 模式留下的是 {p}"
    print("PASS test_media_filter_switch")


def test_add_image_compat():
    """add_image 兼容包装：不存在路径静默 return，不抛异常。"""
    g = _fresh_gallery()
    g.add_image("C:/__definitely_not_exists__/x.png")   # 旧行为：直接 return
    assert len(g._all_items) == 0
    tmp = tempfile.mkdtemp()
    png = os.path.join(tmp, "c.png"); _make_png(png)
    g.add_image(png)
    assert len(g._all_items) == 1
    print("PASS test_add_image_compat")


def test_add_debounce():
    """连续批量 add_media 不触发 N 次重排：200ms 防抖合并。"""
    g = _fresh_gallery()
    tmp = tempfile.mkdtemp()
    paths = []
    for i in range(20):
        p = os.path.join(tmp, f"d{i}.png"); _make_png(p); paths.append(p)

    calls = {"n": 0}
    orig = g._apply_filter
    def counting():
        calls["n"] += 1
        orig()
    g._apply_filter = counting

    for p in paths:
        g.add_media(p)          # 20 次连发
    immediate = calls["n"]
    assert immediate <= 1, f"批量添加期间立即刷新 {immediate} 次，防洪失败"
    _drain(400)
    assert calls["n"] <= 2, f"防抖后总刷新 {calls['n']} 次，应 ≤2"
    assert g.list_widget.count() == 20
    print("PASS test_add_debounce")


def test_reload_scans_videos_subdir():
    """reload_from_dir 同时扫描 directory/videos 下的视频文件。"""
    g = _fresh_gallery()
    tmp = tempfile.mkdtemp()
    _make_png(os.path.join(tmp, "e.png"))
    vdir = os.path.join(tmp, "videos"); os.makedirs(vdir)
    open(os.path.join(vdir, "f.mp4"), "wb").write(b"\x00" * 64)

    g.reload_from_dir(tmp)
    exts = {os.path.splitext(p)[1].lower() for p, _, _ in g._all_items}
    assert ".png" in exts and ".mp4" in exts, f"扫描结果缺少类型: {exts}"
    print("PASS test_reload_scans_videos_subdir")


def test_video_placeholder_thumb():
    """视频缩略图走占位不崩，且带标记。"""
    g = _fresh_gallery()
    tmp = tempfile.mkdtemp()
    mp4 = os.path.join(tmp, "g.mp4"); open(mp4, "wb").write(b"\x00" * 64)
    g.add_media(mp4)
    _drain()
    assert g.list_widget.count() == 1
    item = g.list_widget.item(0)
    assert not item.icon().isNull(), "视频应有占位图标"
    print("PASS test_video_placeholder_thumb")


def test_items_changed_signal():
    """_apply_filter 末尾发出 items_changed，供胶片条联动。"""
    g = _fresh_gallery()
    hits = []
    assert hasattr(g, "items_changed"), "缺少 items_changed 信号"
    g.items_changed.connect(lambda: hits.append(1))
    tmp = tempfile.mkdtemp()
    png = os.path.join(tmp, "h.png"); _make_png(png)
    g.add_media(png)
    _drain()
    assert hits, "items_changed 未发出"
    print("PASS test_items_changed_signal")


def test_video_selected_signal():
    """双击视频发 video_selected 而不是弹图片查看器。"""
    g = _fresh_gallery()
    assert hasattr(g, "video_selected"), "缺少 video_selected 信号"
    hits = []
    g.video_selected.connect(hits.append)
    tmp = tempfile.mkdtemp()
    mp4 = os.path.join(tmp, "i.mp4"); open(mp4, "wb").write(b"\x00" * 64)
    g.add_media(mp4)
    _drain()
    item = g.list_widget.item(0)
    g._on_double_clicked(item)
    assert hits and hits[0].endswith(".mp4"), f"video_selected 未发出: {hits}"
    print("PASS test_video_selected_signal")


def test_filmstrip():
    """胶片条：refresh 填充、点击发 media_clicked。"""
    from ui.components.filmstrip import FilmStrip
    fs = FilmStrip()
    _WINDOWS.append(fs)
    tmp = tempfile.mkdtemp()
    png = os.path.join(tmp, "j.png"); _make_png(png)
    mp4 = os.path.join(tmp, "k.mp4"); open(mp4, "wb").write(b"\x00" * 64)
    fs.refresh([png, mp4])
    assert fs.count() == 2
    hits = []
    fs.media_clicked.connect(hits.append)
    fs._on_clicked(fs.item(0))
    assert hits == [png]
    # 视频占位不崩
    assert not fs.item(1).icon().isNull()
    print("PASS test_filmstrip")


def test_gallery_page():
    """画廊页：注册进 PAGES，工具条切换媒体过滤，重挂 host.gallery 单实例。"""
    from ui.pages import PAGES
    ids = [p.page_id for p in PAGES]
    assert "gallery" in ids, f"GalleryPage 未注册: {ids}"

    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    _WINDOWS.append(win)
    win.setup_ui()
    gpage = win._pages["gallery"]
    ws = gpage.workspace()
    assert ws is not None
    # 画廊单实例重挂进画廊页工作区
    from PyQt6.QtWidgets import QPushButton
    assert win.gallery.parent() is not None
    assert ws.isAncestorOf(win.gallery), "画廊未重挂进画廊页工作区"
    # 工具条三态按钮存在且可切换
    b_all = ws.findChild(QPushButton, "btnMediaAll")
    b_img = ws.findChild(QPushButton, "btnMediaImage")
    b_vid = ws.findChild(QPushButton, "btnMediaVideo")
    assert all((b_all, b_img, b_vid)), "媒体切换按钮缺失"
    b_img.click()
    assert win.gallery._media_filter == "image"
    b_vid.click()
    assert win.gallery._media_filter == "video"
    # 胶片条已就位并联动
    assert win.filmstrip is not None
    print("PASS test_gallery_page")


if __name__ == "__main__":
    for fn in (test_media_filter_switch, test_add_image_compat, test_add_debounce,
               test_reload_scans_videos_subdir, test_video_placeholder_thumb,
               test_items_changed_signal, test_video_selected_signal,
               test_filmstrip, test_gallery_page):
        fn()
    for w in _WINDOWS:
        w.close()
    _APP.processEvents()
    sys.stdout.flush()
    print("\n✅ 全部统一画廊测试通过")
    os._exit(0)
