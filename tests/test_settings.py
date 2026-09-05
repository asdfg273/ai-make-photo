# tests/test_settings.py — 设置界面：改键 + 默认值（无头，纯脚本）
import os, sys, tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])
_WINDOWS = []


def _mini_app():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    _WINDOWS.append(win)
    win.setup_ui()
    return win


def _fresh_config():
    """隔离的 AppConfig：save 写到临时文件，不碰真实 app_config.json。"""
    from core.config_manager import AppConfig
    cfg = AppConfig()
    cfg.config_file = os.path.join(tempfile.mkdtemp(), "cfg.json")
    return cfg


def test_shortcut_map_merge():
    """快捷键映射：配置缺项回退默认，已存项覆盖默认。"""
    win = _mini_app()
    win.config = _fresh_config()
    win.config.shortcuts = {"generate": "Ctrl+G"}
    m = win._shortcut_map()
    assert m["generate"] == "Ctrl+G"
    assert m["interrupt"] == "Escape"           # 未改的保持默认
    assert m["page_gallery"] == "Ctrl+4"
    assert m["extension_market"] == "Ctrl+E"
    print("PASS test_shortcut_map_merge")


def test_rebuild_shortcuts_applies_config():
    """重建快捷键：改键后新键位生效，清空键位 = 禁用。"""
    from PyQt6.QtGui import QKeySequence
    win = _mini_app()
    win.config = _fresh_config()
    win.config.shortcuts = {"generate": "Ctrl+G", "interrupt": ""}
    win._rebuild_shortcuts()
    keys = [sc.key() for sc in win._shortcuts]
    assert QKeySequence("Ctrl+G") in keys, keys
    assert QKeySequence("Escape") not in keys, "清空的中断键不应再注册"
    assert QKeySequence("Ctrl+Return") not in keys, "旧的生成键应被替换"
    # 再改回来验证可重复重建
    win.config.shortcuts = {}
    win._rebuild_shortcuts()
    keys2 = [sc.key() for sc in win._shortcuts]
    assert QKeySequence("Ctrl+Return") in keys2
    assert QKeySequence("Escape") in keys2
    print("PASS test_rebuild_shortcuts_applies_config")


def test_settings_dialog_roundtrip():
    """设置对话框：载入当前值 → 修改 → accept 写回 config（且写入隔离文件）。"""
    from ui.settings_dialog import SettingsDialog
    from PyQt6.QtGui import QKeySequence

    win = _mini_app()
    win.config = _fresh_config()
    win.config.default_steps = 42
    win.config.shortcuts = {"generate": "Ctrl+G"}

    dlg = SettingsDialog(win)
    _WINDOWS.append(dlg)
    # 载入检查
    assert dlg.spin_steps.value() == 42
    assert dlg.seq_edits["generate"].keySequence().toString() == "Ctrl+G"

    # 修改：键位 + 默认值
    dlg.seq_edits["generate"].setKeySequence(QKeySequence("Ctrl+Shift+G"))
    dlg.spin_steps.setValue(25)
    dlg.combo_res.setCurrentText("832x1216")
    dlg.combo_trans.setCurrentIndex(1)
    dlg.chk_hires.setChecked(True)
    dlg.accept()

    cfg = win.config
    assert cfg.shortcuts["generate"] == "Ctrl+Shift+G"
    # QKeySequenceEdit 规范化存储（Escape→Esc），按语义等价比较
    assert QKeySequence(cfg.shortcuts["interrupt"]) == QKeySequence("Escape")
    assert cfg.default_steps == 25
    assert (cfg.default_width, cfg.default_height) == (832, 1216)
    assert cfg.default_trans_mode == 1
    assert cfg.use_hires is True
    # 确实写入了隔离文件，且能被 load 读回
    assert os.path.exists(cfg.config_file)
    from core.config_manager import AppConfig
    AppConfig.config_file = cfg.config_file      # 指向隔离文件再读
    try:
        back = AppConfig.load()
        assert back.default_steps == 25
        assert back.shortcuts["generate"] == "Ctrl+Shift+G"
    finally:
        from utils.paths import CONFIG_FILE
        AppConfig.config_file = CONFIG_FILE      # 还原，防污染其他测试
    print("PASS test_settings_dialog_roundtrip")


def test_menu_settings_entry():
    """菜单栏有 设置 菜单且 偏好设置 动作已接线。"""
    win = _mini_app()
    menus = [a.text() for a in win.menuBar().actions()]
    assert any("设置" in t for t in menus), menus
    setting_menu = next(a.menu() for a in win.menuBar().actions()
                        if "设置" in a.text())
    acts = [a.text() for a in setting_menu.actions()]
    assert any("偏好设置" in t for t in acts), acts
    print("PASS test_menu_settings_entry")


if __name__ == "__main__":
    for fn in (test_shortcut_map_merge, test_rebuild_shortcuts_applies_config,
               test_settings_dialog_roundtrip, test_menu_settings_entry):
        fn()
    for w in _WINDOWS:
        w.close()
    _APP.processEvents()
    sys.stdout.flush()
    print("\n✅ 全部设置界面测试通过")
    os._exit(0)
