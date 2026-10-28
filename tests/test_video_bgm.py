"""视频背景配乐控件与参数收集测试；不触发视频生成。"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QWidget

from ui.pages.video_page import VideoPage
from ui.video_panel_mixin import VideoPanelMixin

_APP = QApplication.instance() or QApplication([])


class _Host(QWidget, VideoPanelMixin):
    pass


def _build_host():
    host = _Host()
    page = VideoPage()
    page.build(host)
    return host, page


def _close_host(host, page):
    host.video_player.stop()
    page.params_widget().close()
    page.workspace().close()
    host.close()
    page.params_widget().deleteLater()
    page.workspace().deleteLater()
    host.deleteLater()


def test_video_bgm_controls_and_voice_params():
    host, page = _build_host()
    try:
        assert host.txt_bg_music_path.isReadOnly()
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "music.mp3"
            audio.write_bytes(b"test")
            host.chk_bg_music.setChecked(True)
            host.txt_bg_music_path.setText(str(audio))
            assert host._collect_bg_music({"engine": "chattts"}) == {
                "engine": "chattts", "bg_music": str(audio)}
            host._clear_bg_music()
            assert host._collect_bg_music({"engine": "chattts"}) == {
                "engine": "chattts"}
            host.txt_bg_music_path.setText(str(audio))
            host.chk_bg_music.setChecked(False)
            assert host._collect_bg_music({"engine": "chattts"}) == {
                "engine": "chattts"}
        assert host._collect_bg_music(None) is None
    finally:
        _close_host(host, page)


def test_video_bgm_file_picker_and_clear():
    host, page = _build_host()
    try:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "music.wav"
            audio.write_bytes(b"test")
            with patch(
                    "ui.video_panel_mixin.QFileDialog.getOpenFileName",
                    return_value=(str(audio), "音频文件")):
                host._pick_bg_music()
            assert host.txt_bg_music_path.text() == str(audio)
            assert host.chk_bg_music.isChecked()
            host._clear_bg_music()
            assert host.txt_bg_music_path.text() == ""
    finally:
        _close_host(host, page)


def test_video_bgm_translations_exist():
    from ui.i18n_catalog import TRANSLATIONS

    keys = {
        "添加背景配乐", "配乐文件:", "未选择配乐", "选择背景配乐",
        "音频文件 (*.mp3 *.wav *.flac *.m4a *.aac *.ogg)",
    }
    missing = sorted(key for key in keys if key not in TRANSLATIONS)
    assert not missing, f"缺少配乐英日翻译: {missing}"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"VIDEO_BGM_OK: {len(tests)} tests")
