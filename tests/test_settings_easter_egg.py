# -*- coding: utf-8 -*-
"""设置页那个「点这里」按钮的回归（ui/settings_dialog.py）。

为什么值得测
------------
一个"点一下就打开网页"的按钮，写错的方式有一堆：连错槽、URL 打错字母、
`openUrl` 返回 False 时没人处理（用户点了没反应还以为是程序卡了）。
所以这里钉三件事：按钮存在且文案对、点下去真的把 B 站链接交给系统浏览器、
浏览器打不开时不会静默失败而是把链接摆出来。

跑法: venv\\Scripts\\python.exe tests\\test_settings_easter_egg.py
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bootstrap  # noqa: E402

_bootstrap.isolate_config_globally()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = []


def check(name):
    def deco(fn):
        try:
            info = fn()
            RESULT.append((name, True, info))
            print(f"[PASS] {name}" + (f" | {info}" if info else ""))
        except Exception as exc:  # noqa: BLE001 - 测试脚本自己兜底
            RESULT.append((name, False, str(exc)))
            print(f"[FAIL] {name} | {exc}")
            traceback.print_exc()
        return fn
    return deco


def _make_dialog():
    from PyQt6.QtWidgets import QApplication, QWidget
    from core.config_manager import AppConfig
    from ui.settings_dialog import SettingsDialog

    app = QApplication.instance() or QApplication([])
    host = QWidget()
    host.config = AppConfig.load()
    return app, host, SettingsDialog(host)


@check("「点这里」按钮存在，文案与提示都对")
def test_button_exists():
    from ui.settings_dialog import EASTER_EGG_URL

    app, host, dlg = _make_dialog()
    assert hasattr(dlg, "btn_click_here"), "没有 btn_click_here"
    btn = dlg.btn_click_here
    assert btn.text() == "点这里", btn.text()
    assert btn.toolTip(), "没有 tooltip"
    assert "bilibili.com" in EASTER_EGG_URL, EASTER_EGG_URL
    # 应该在「常规」这一页里
    tabs = [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())]
    assert "常规" in tabs, tabs
    general = dlg.tabs.widget(tabs.index("常规"))
    found = False
    for w in general.findChildren(type(btn)):
        if w.text() == "点这里":
            found = True
            break
    assert found, "按钮不在「常规」页"
    dlg.deleteLater()
    host.deleteLater()
    app.processEvents()
    return EASTER_EGG_URL


@check("点下去会把 B 站链接交给系统浏览器")
def test_click_opens_url():
    from urllib.parse import urlparse
    from PyQt6.QtGui import QDesktopServices
    from ui.settings_dialog import EASTER_EGG_URL

    app, host, dlg = _make_dialog()
    opened = []
    original = QDesktopServices.openUrl

    def fake_open(url):
        opened.append(url.toString())
        return True

    QDesktopServices.openUrl = staticmethod(fake_open)
    try:
        dlg.btn_click_here.click()
    finally:
        QDesktopServices.openUrl = original
    assert len(opened) == 1, opened
    assert opened[0] == EASTER_EGG_URL, opened
    parsed = urlparse(opened[0])
    assert parsed.netloc.endswith("bilibili.com"), parsed
    assert parsed.path.startswith("/video/"), parsed
    dlg.deleteLater()
    host.deleteLater()
    app.processEvents()
    return f"打开 {opened[0]}"


@check("浏览器打不开时把链接摆出来，而不是静默失败")
def test_fallback_message():
    from PyQt6.QtGui import QDesktopServices
    from PyQt6.QtWidgets import QMessageBox
    from ui.settings_dialog import EASTER_EGG_URL

    app, host, dlg = _make_dialog()
    original_open = QDesktopServices.openUrl
    shown = []
    original_exec = QMessageBox.exec

    def fake_exec(self, *a, **kw):
        shown.append((self.windowTitle(), self.text(), self.informativeText()))
        return 0

    QDesktopServices.openUrl = staticmethod(lambda url: False)
    QMessageBox.exec = fake_exec
    try:
        dlg.btn_click_here.click()
    finally:
        QDesktopServices.openUrl = original_open
        QMessageBox.exec = original_exec
    assert len(shown) == 1, shown
    title, text, info = shown[0]
    assert title, "弹窗没有标题"
    assert EASTER_EGG_URL in info or EASTER_EGG_URL in text, shown[0]
    dlg.deleteLater()
    host.deleteLater()
    app.processEvents()
    return f"标题「{title}」里带着链接"


@check("点击不会改动配置（彩蛋不该有副作用）")
def test_no_config_side_effect():
    from PyQt6.QtGui import QDesktopServices
    import json

    app, host, dlg = _make_dialog()
    before = json.dumps(host.config.to_dict() if hasattr(host.config, "to_dict") else {},
                        ensure_ascii=False, sort_keys=True, default=str)
    original = QDesktopServices.openUrl
    QDesktopServices.openUrl = staticmethod(lambda url: True)
    try:
        dlg.btn_click_here.click()
        dlg.btn_click_here.click()
    finally:
        QDesktopServices.openUrl = original
    after = json.dumps(host.config.to_dict() if hasattr(host.config, "to_dict") else {},
                       ensure_ascii=False, sort_keys=True, default=str)
    assert before == after, "配置被改了"
    dlg.deleteLater()
    host.deleteLater()
    app.processEvents()
    return "配置未变"


def main():
    print("=" * 60)
    totals = len(RESULT)
    passed = sum(1 for _, ok, _ in RESULT if ok)
    print(f"总计 {totals} 项：PASS {passed} / FAIL {totals - passed}")
    print("=" * 60)
    return 0 if passed == totals else 1


if __name__ == "__main__":
    sys.exit(main())
