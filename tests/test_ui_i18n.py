# tests/test_ui_i18n.py
# UI 多语言离屏测试：配置、词典、控件树重翻译与动态文本保护
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QApplication, QComboBox, QGroupBox, QLabel, QLineEdit,
    QMainWindow, QPushButton, QTabWidget, QTextEdit, QWidget,
)

_APP = QApplication.instance() or QApplication([])


def test_config_language_roundtrip():
    from core.config_manager import AppConfig

    old_path = AppConfig.config_file
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        cfg = AppConfig()
        assert cfg.ui_language == "zh"
        cfg.config_file = path
        cfg.ui_language = "ja"
        cfg.save()

        AppConfig.config_file = path
        loaded = AppConfig.load()
        assert loaded.ui_language == "ja"
    finally:
        AppConfig.config_file = old_path
        try:
            os.unlink(path)
        except OSError:
            pass


def test_translate_and_validate_language():
    from ui.i18n import ui_i18n

    old = ui_i18n.language
    try:
        ui_i18n.set_language("en")
        assert ui_i18n.tr("文生图") == "Text to Image"
        ui_i18n.set_language("ja")
        assert ui_i18n.tr("画廊") == "ギャラリー"
        ui_i18n.set_language("not-a-language")
        assert ui_i18n.language == "zh"
        assert ui_i18n.tr("画廊") == "画廊"
    finally:
        ui_i18n.set_language(old)


def test_widget_tree_roundtrip_and_dynamic_text():
    from ui.i18n import ui_i18n

    old = ui_i18n.language
    root = QMainWindow()
    tabs = QTabWidget()
    page = QWidget()
    tabs.addTab(page, "默认参数")
    root.setCentralWidget(tabs)

    group = QGroupBox("模型与设备", page)
    button = QPushButton("开始生成", group)
    status = QLabel("就绪", group)
    combo = QComboBox(group)
    combo.setProperty("i18nItems", True)  # 仅展示用下拉框允许翻译选项
    combo.addItems(["无", "纯词典"])
    business_combo = QComboBox(group)
    business_combo.addItems(["无", "纯词典"])
    edit = QLineEdit(group)
    edit.setPlaceholderText("在此输入正向提示词...")
    edit.setText("用户自己的提示词")
    prompt = QTextEdit(group)
    prompt.setPlainText("不要翻译正文")
    menu = root.menuBar().addMenu("文件")
    menu.addAction("退出")

    try:
        ui_i18n.set_language("en")
        ui_i18n.translate_widget_tree(root)
        assert button.text() == "Generate"
        assert group.title() == "Model & Device"
        assert status.text() == "Ready"
        assert combo.itemText(0) == "None"
        assert combo.itemText(1) == "Dictionary only"
        assert business_combo.itemText(0) == "无"
        assert business_combo.itemText(1) == "纯词典"
        assert edit.placeholderText() == "Enter a positive prompt..."
        assert edit.text() == "用户自己的提示词"
        assert prompt.toPlainText() == "不要翻译正文"
        assert tabs.tabText(0) == "Defaults"
        assert root.menuBar().actions()[0].text() == "File"
        assert menu.actions()[0].text() == "Quit"

        # 业务代码随后写入的新中文状态不能被旧的“就绪”原文覆盖。
        status.setText("正在生成")
        ui_i18n.translate_widget_tree(root)
        assert status.text() == "Generating..."

        ui_i18n.set_language("ja")
        ui_i18n.translate_widget_tree(root)
        assert button.text() == "生成開始"
        assert group.title() == "モデルとデバイス"
        assert status.text() == "生成中"
        assert combo.itemText(0) == "なし"

        ui_i18n.set_language("zh")
        ui_i18n.translate_widget_tree(root)
        assert button.text() == "开始生成"
        assert group.title() == "模型与设备"
        assert status.text() == "正在生成"
        assert combo.itemText(1) == "纯词典"
        assert edit.text() == "用户自己的提示词"
    finally:
        ui_i18n.set_language(old)
        root.close()
        root.deleteLater()


def test_settings_language_control():
    from PyQt6.QtWidgets import QMainWindow
    from core.config_manager import AppConfig
    from ui.i18n import ui_i18n
    from ui.settings_dialog import SettingsDialog

    old = ui_i18n.language
    host = QMainWindow()
    host.config = AppConfig()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    host.config.config_file = path
    dlg = SettingsDialog(host)
    try:
        assert dlg.combo_ui_language.count() == 3
        idx = dlg.combo_ui_language.findData("en")
        assert idx >= 0
        dlg.combo_ui_language.setCurrentIndex(idx)
        dlg.accept()
        assert host.config.ui_language == "en"
        assert ui_i18n.language == "en"
    finally:
        ui_i18n.set_language(old)
        dlg.deleteLater()
        host.close()
        host.deleteLater()
        try:
            os.unlink(path)
        except OSError:
            pass


def test_shell_language_and_layout():
    from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QSizePolicy
    from core.config_manager import AppConfig
    from ui.i18n import ui_i18n
    from ui.shell import ShellMixin
    from ui.theme import PALETTE

    class MiniApp(QMainWindow, ShellMixin):
        pass

    old = ui_i18n.language
    win = MiniApp()
    win.config = AppConfig()
    win.config.ui_language = "en"
    try:
        win.setup_ui()
        assert win.btn_generate.text() == "Generate"
        assert win.nav._buttons["gallery"].text() == "Gallery"
        assert win.combo_trans_mode.itemText(0).strip() == "Dictionary only"
        # 业务值下拉框保持原文，不被界面语言改变。
        assert win.combo_adetailer_model.itemText(0) == "真人脸"
        assert win.combo_hires_mode.itemText(0) == "自动"

        # 更紧凑的工作台布局：宽导航容纳日/英，参数栏把空间让给画布。
        assert 124 <= win.nav.width() <= 144
        assert 380 <= win.params_panel.width() <= 408
        margins = win.centralWidget().layout().contentsMargins()
        assert margins.left() >= 12 and margins.top() >= 10
        assert PALETTE["radius"] <= 5
        assert (win.nav._buttons["gallery"].sizePolicy().horizontalPolicy()
                == QSizePolicy.Policy.Expanding)

        # AI 工具和模型档位分两行，避免英/日长文本互相挤压。
        layouts = win.findChildren(QHBoxLayout)
        enhance_row = next(x for x in layouts if x.indexOf(win.btn_enhance_prompt) >= 0)
        model_row = next(x for x in layouts if x.indexOf(win.combo_ai_model) >= 0)
        assert enhance_row is not model_row

        # 离屏 Qt 可能没有任何字体，sizeHint() 不代表真实桌面排版；
        # 此处只验证结构和尺寸策略，实际截断由 Windows 桌面视觉验收覆盖。

        # 动态状态栏也要随语言重绘，且后续数值刷新不能退回中文。
        win._update_resource_label()
        win._refresh_extension_count()
        assert win.lbl_resource.text().startswith("Memory ")
        assert win.lbl_ext_count.text().startswith("Extensions:")
        resource_values = win.lbl_resource.text().removeprefix("Memory ")
        extension_values = win.lbl_ext_count.text().removeprefix("Extensions:")

        ui_i18n.set_language("ja")
        assert win.nav._buttons["gallery"].text() == "ギャラリー"
        assert win.btn_generate.text() == "生成開始"
        assert win.lbl_resource.text() == f"メモリ {resource_values}"
        assert win.lbl_ext_count.text() == f"拡張:{extension_values}"
        win._update_resource_label()
        win._refresh_extension_count()
        assert win.lbl_resource.text().startswith("メモリ ")
        assert win.lbl_ext_count.text().startswith("拡張:")

        ui_i18n.set_language("zh")
        assert win.btn_generate.text() == "开始生成"
        assert win.lbl_resource.text().startswith("内存 ")
        assert win.lbl_ext_count.text().startswith("扩展:")
    finally:
        ui_i18n.set_language(old)
        timer = getattr(win, "_resource_timer", None)
        if timer is not None:
            timer.stop()
        player = getattr(win, "video_player", None)
        if player is not None:
            player.stop()
            player.deleteLater()
        win.close()
        win.deleteLater()


def test_gallery_count_language_roundtrip():
    from ui.gallery_panel import GalleryPanel
    from ui.i18n import ui_i18n

    old = ui_i18n.language
    gallery = GalleryPanel()
    gallery._all_items = [
        ("missing-visible.png", "", False),
        ("missing-hidden.png", "", True),
    ]
    try:
        ui_i18n.set_language("en")
        gallery._apply_filter()
        assert gallery.lbl_count.text() == "1/2 items (1 hidden)"
        ui_i18n.set_language("ja")
        assert gallery.lbl_count.text() == "1/2 枚（1 件非表示）"
        gallery._apply_filter()
        assert gallery.lbl_count.text() == "1/2 枚（1 件非表示）"
        ui_i18n.set_language("zh")
        assert gallery.lbl_count.text() == "1/2 张（隐藏 1）"
    finally:
        ui_i18n.set_language(old)
        gallery.close()
        gallery.deleteLater()


def test_character_count_language_roundtrip():
    from PyQt6.QtWidgets import QWidget
    from ui.i18n import ui_i18n
    from ui.pages.characters_page import CharactersPage
    from utils.character_library import Character, CharacterLibrary

    old = ui_i18n.language
    with tempfile.TemporaryDirectory() as directory:
        page = CharactersPage()
        page.library = CharacterLibrary(directory)
        page.library.save_character(Character(name="测试角色"))
        host = QWidget()
        try:
            page.build(host)
            ui_i18n.set_language("en")
            page.refresh()
            assert page.lbl_count.text() == "1 character"
            ui_i18n.set_language("ja")
            assert page.lbl_count.text() == "1 キャラクター"
            page.refresh()
            assert page.lbl_count.text() == "1 キャラクター"
            ui_i18n.set_language("zh")
            assert page.lbl_count.text() == "1 个角色"
        finally:
            ui_i18n.set_language(old)
            page.dispose()
            host.close()
            host.deleteLater()


def test_extension_market_language_roundtrip():
    from ui.extension_market import ExtensionMarketDialog
    from ui.i18n import ui_i18n
    from utils.extension_manager import get_status_summary

    old = ui_i18n.language
    dlg = ExtensionMarketDialog()
    try:
        summary = get_status_summary()
        ui_i18n.set_language("en")
        dlg._update_summary()
        assert dlg.lbl_summary.text() == (
            f"Installed: {summary['installed']} / {summary['total']}")
        first_card = next(iter(dlg.cards.values()))
        assert first_card.lbl_status.text() in {"Installed", "Not installed", "Missing (required)"}
        assert first_card.btn_action.text() in {"Uninstall", "Download"}
        assert first_card.lbl_name.text() == "<b>Qwen2-VL-2B Prompt Enhancement</b>"
        assert first_card.lbl_desc.text() == (
            "AI prompt rewriting, image understanding, and Chinese-Japanese translation")
        import re
        for card in dlg.cards.values():
            assert not re.search(r"[\u4e00-\u9fff]", card.lbl_name.text())
            assert not re.search(r"[\u4e00-\u9fff]", card.lbl_desc.text())
        ui_i18n.set_language("ja")
        assert dlg.lbl_summary.text() == (
            f"インストール済み: {summary['installed']} / {summary['total']}")
        assert first_card.lbl_name.text() == "<b>Qwen2-VL-2B プロンプト強化</b>"
        assert first_card.lbl_desc.text() == "AI プロンプトリライト・画像認識・中国語と日本語の翻訳"
        ui_i18n.set_language("zh")
        assert dlg.lbl_summary.text() == (
            f"已安装: {summary['installed']} / {summary['total']}")
        assert first_card.lbl_name.text() == "<b>Qwen2-VL-2B 提示词增强</b>"
    finally:
        ui_i18n.set_language(old)
        dlg.close()
        dlg.deleteLater()


def test_video_status_language_roundtrip():
    from PyQt6.QtWidgets import QLabel, QPushButton, QWidget
    from ui.i18n import ui_i18n
    from ui.video_panel_mixin import VideoPanelMixin

    class VideoHost(QWidget, VideoPanelMixin):
        def __init__(self):
            super().__init__()
            self.btn_gen_video = QPushButton(self)
            self.lbl_video_status = QLabel(self)
            self.is_generating = False
            self.status_messages = []

        def _set_status(self, text, color):
            self.status_messages.append(text)

        def set_progress(self, value):
            pass

    old = ui_i18n.language
    host = VideoHost()
    try:
        ui_i18n.set_language("en")
        host._on_video_generated("result.mp4", 42)
        assert host.btn_gen_video.text() == "Generate Video"
        assert host.lbl_video_status.text() == "Completed (seed=42)"
        ui_i18n.set_language("ja")
        host._retranslate_video_dynamic()
        assert host.btn_gen_video.text() == "動画を生成"
        assert host.lbl_video_status.text() == "完了 (seed=42)"
        host._on_video_gen_error(1, "failed")
        assert host.lbl_video_status.text() == "生成に失敗しました"
        ui_i18n.set_language("zh")
        host._retranslate_video_dynamic()
        assert host.btn_gen_video.text() == "生成视频"
        assert host.lbl_video_status.text() == "生成失败"
    finally:
        ui_i18n.set_language(old)
        host.close()
        host.deleteLater()


if __name__ == "__main__":
    test_config_language_roundtrip()
    test_translate_and_validate_language()
    test_widget_tree_roundtrip_and_dynamic_text()
    test_settings_language_control()
    test_shell_language_and_layout()
    test_gallery_count_language_roundtrip()
    test_character_count_language_roundtrip()
    test_extension_market_language_roundtrip()
    test_video_status_language_roundtrip()
    print("UI_I18N_OK")
    sys.stdout.flush()
    # 与现有 Qt 离屏测试一致，跳过 Multimedia/Qt 的进程退出清理。
    os._exit(0)
