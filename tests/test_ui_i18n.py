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


def test_gallery_favorite_dialog_language():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QMessageBox
    from ui.gallery_panel import GalleryPanel
    from ui.i18n import ui_i18n

    old = ui_i18n.language
    gallery = GalleryPanel()
    gallery._favs = set()
    gallery._save_favs = lambda: None
    gallery._apply_filter = lambda: None
    captured = []
    try:
        with patch.object(QMessageBox, "information", side_effect=(
                lambda _parent, title, text, *args: captured.append((title, text)))):
            ui_i18n.set_language("en")
            gallery._toggle_fav(["example.png"])
            assert captured[-1] == (
                "Favorites", "Added 1 | Removed 0\n1 favorite in total")
            ui_i18n.set_language("ja")
            gallery._toggle_fav(["example.png"])
            assert captured[-1] == (
                "お気に入り", "追加 0 枚 | 解除 1 枚\nお気に入り合計 0 枚")
    finally:
        ui_i18n.set_language(old)
        gallery.close()
        gallery.deleteLater()


def test_extension_uninstall_dialog_language():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QMessageBox
    from ui.extension_market import ExtensionCard
    from ui.i18n import ui_i18n

    old = ui_i18n.language
    card = ExtensionCard("test-required", {
        "name": "ControlNet 姿势 (OpenPose)",
        "desc": "通过骨骼图控制人物姿势",
        "size_mb": 1,
        "required": True,
    }, True)
    captured = []
    try:
        with patch.object(QMessageBox, "warning", side_effect=(
                lambda _parent, title, text, *args: (
                    captured.append((title, text)), QMessageBox.StandardButton.No)[1])):
            ui_i18n.set_language("en")
            card._on_click()
            assert captured[-1][0] == "Confirm Uninstall"
            assert "ControlNet Pose (OpenPose)" in captured[-1][1]
            assert "required extension" in captured[-1][1]
    finally:
        ui_i18n.set_language(old)
        card.close()
        card.deleteLater()


def test_preset_empty_export_dialog_language():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QMessageBox, QWidget
    from ui.i18n import ui_i18n
    from ui.preset_manager import PresetManagerMixin

    class PresetHost(QWidget, PresetManagerMixin):
        pass

    old = ui_i18n.language
    host = PresetHost()
    captured = []
    try:
        with patch("core.presets.load_user_presets", return_value={}), patch.object(
                QMessageBox, "information", side_effect=(
                    lambda _parent, title, text, *args: captured.append((title, text)))):
            ui_i18n.set_language("en")
            host.export_user_presets()
            assert captured[-1] == ("Export", "There are no user presets to export.")
    finally:
        ui_i18n.set_language(old)
        host.close()
        host.deleteLater()


def test_shell_busy_dialog_language():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QMessageBox, QWidget
    from ui.i18n import ui_i18n
    from ui.shell import ShellMixin

    class ShellHost(QWidget, ShellMixin):
        pass

    old = ui_i18n.language
    host = ShellHost()
    host.is_generating = True
    captured = []
    try:
        with patch.object(QMessageBox, "warning", side_effect=(
                lambda _parent, title, text, *args: captured.append((title, text)))):
            ui_i18n.set_language("en")
            host.on_unload_models()
            assert captured[-1] == (
                "Notice", "Stop the current generation task first.")
    finally:
        ui_i18n.set_language(old)
        host.close()
        host.deleteLater()


def test_video_validation_status_language():
    from PyQt6.QtWidgets import QWidget
    from ui.i18n import ui_i18n
    from ui.video_panel_mixin import VideoPanelMixin

    class VideoHost(QWidget, VideoPanelMixin):
        def __init__(self):
            super().__init__()
            self.is_generating = True
            self.status_messages = []

        def _set_status(self, text, color):
            self.status_messages.append(text)

    old = ui_i18n.language
    host = VideoHost()
    try:
        ui_i18n.set_language("en")
        host.on_generate_video()
        assert host.status_messages[-1] == "Generation is already in progress. Please wait."
    finally:
        ui_i18n.set_language(old)
        host.close()
        host.deleteLater()


def test_scene_batch_dialog_language():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QMessageBox
    from ui.i18n import ui_i18n
    from ui.scene_batch_dialog import SceneBatchDialog

    old = ui_i18n.language
    dialog = SceneBatchDialog()
    captured = []
    try:
        with patch.object(QMessageBox, "information", side_effect=(
                lambda _parent, title, text, *args: captured.append((title, text)))):
            ui_i18n.set_language("en")
            dialog._on_run()
            assert captured[-1] == ("Notice", "Select at least one scene.")
            ui_i18n.set_language("ja")
            dialog._save_current()
            assert captured[-1] == ("通知", "左側のリストからシーンを選択してください。")
    finally:
        ui_i18n.set_language(old)
        dialog.close()
        dialog.deleteLater()


def test_workflow_editor_dialog_language():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QMessageBox
    from ui.i18n import ui_i18n
    from ui.workflow_editor.window import WorkflowEditorWindow

    old = ui_i18n.language
    window = WorkflowEditorWindow()
    captured = []
    try:
        with patch.object(QMessageBox, "information", side_effect=(
                lambda _parent, title, text, *args: captured.append((title, text)))):
            ui_i18n.set_language("en")
            window.on_build_from_ui()
            assert captured[-1] == (
                "No Host",
                "The editor is not attached to the main window and cannot build from the interface.")
        window.scene.node_items["n1"] = object()
        with patch.object(QMessageBox, "question", side_effect=(
                lambda _parent, title, text, *args: (
                    captured.append((title, text)), QMessageBox.StandardButton.No)[1])):
            ui_i18n.set_language("ja")
            assert window._confirm_discard() is False
            assert captured[-1] == ("確認", "現在のワークフローは未保存です。破棄しますか？")
    finally:
        ui_i18n.set_language(old)
        window.close()
        window.deleteLater()


def test_workflow_resource_dialog_language():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QMessageBox
    from ui.i18n import ui_i18n
    from ui.workflow_editor.resource_panel import ResourceDialog
    from workflow.schema import Workflow

    old = ui_i18n.language
    dialog = ResourceDialog(Workflow(name="test"))
    captured = []
    try:
        with patch.object(QMessageBox, "information", side_effect=(
                lambda _parent, title, text, *args: captured.append((title, text)))):
            ui_i18n.set_language("en")
            dialog.on_download()
            assert captured[-1] == ("Nothing Selected", "Select an item in the list first.")
            dialog._thread = type("RunningThread", (), {"isRunning": lambda self: True})()
            ui_i18n.set_language("ja")
            dialog._start_download("example")
            assert captured[-1] == ("ダウンロード中", "前のダウンロードがまだ終了していません。")
    finally:
        dialog._thread = None
        ui_i18n.set_language(old)
        dialog.close()
        dialog.deleteLater()


def test_workflow_failure_dialog_language():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QMessageBox
    from ui.i18n import ui_i18n
    from ui.workflow_editor.window import WorkflowEditorWindow

    old = ui_i18n.language
    window = WorkflowEditorWindow()
    captured = []
    try:
        with patch.object(QMessageBox, "critical", side_effect=(
                lambda _parent, title, text, *args: captured.append((title, text)))):
            ui_i18n.set_language("en")
            window._on_run_done({
                "ok": False, "failed_node": "n1", "error": "boom",
                "states": {}, "outputs": {}, "cached": [], "seconds": 0.0,
            })
            assert captured[-1] == ("Execution Failed", "Node [n1] failed:\n\nboom")
    finally:
        ui_i18n.set_language(old)
        window.close()
        window.deleteLater()


def test_param_panel_language():
    from unittest.mock import patch
    from ui.i18n import ui_i18n
    from ui.workflow_editor.param_panel import ParamPanel
    from workflow.schema import (
        DataType, Node, NodeSpec, ParamSpec, PortSpec, ResourceRequirement, Workflow,
    )

    spec = NodeSpec(
        type="I18nTest",
        description="加载底模（.safetensors/.ckpt）。可填文件名或绝对路径。",
        inputs={"image": PortSpec(DataType.IMAGE)},
        params={"seed": ParamSpec(
            kind="seed", default=-1, label="Seed", minimum=-1, maximum=100)},
        resources=ResourceRequirement(
            requires=("cuda", "model"), estimated_vram_mb=1024,
            supports_cpu_offload=True, supports_tiled=False, exclusive_gpu=True),
    )
    wf = Workflow(nodes=[Node(id="n1", type="I18nTest")])
    old = ui_i18n.language
    panel = None
    try:
        with patch("ui.workflow_editor.param_panel.registry.exists", return_value=True), \
                patch("ui.workflow_editor.param_panel.registry.get", return_value=spec):
            ui_i18n.set_language("en")
            panel = ParamPanel()
            assert panel.findChildren(QLabel)[0].text() == "Select a node to view its parameters"
            panel.show_node(wf, "n1")
            labels = [label.text() for label in panel.findChildren(QLabel)]
            assert "Inputs" in labels
            assert "Parameters" in labels
            assert "Resource Requirements" in labels
            assert ("Load a base model (.safetensors/.ckpt). "
                    "Enter a filename or an absolute path.") in labels
            assert any("Peak VRAM ≈ 1024 MB" in text for text in labels)
            assert "Range [-1, 100]" in panel._widgets["seed"].toolTip()
            assert panel._widgets["seed"].specialValueText() == "Random (-1)"

            ui_i18n.set_language("ja")
            labels = [label.text() for label in panel.findChildren(QLabel)]
            assert "入力" in labels
            assert "パラメーター" in labels
            assert "リソース要件" in labels
            assert ("基本モデル（.safetensors/.ckpt）を読み込みます。"
                    "ファイル名または絶対パスを指定できます。") in labels
            assert any("VRAM ピーク ≈ 1024 MB" in text for text in labels)
            assert panel._widgets["seed"].specialValueText() == "ランダム (-1)"
    finally:
        ui_i18n.set_language(old)
        if panel is not None:
            panel.close()
            panel.deleteLater()


def test_region_editor_language():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QFileDialog
    from ui.i18n import ui_i18n
    from ui.region_editor import COL_SHAPE, RegionEditorDialog

    old = ui_i18n.language
    dialog = None
    try:
        ui_i18n.set_language("en")
        dialog = RegionEditorDialog([{
            "name": "Face", "weight": 0.5, "shape": "rect", "enabled": True,
        }], width=640, height=480)
        assert dialog.windowTitle() == "Regional Prompts"
        assert dialog.table.horizontalHeaderItem(COL_SHAPE).text() == "Shape"
        assert dialog.table.item(0, COL_SHAPE).text() == "Rectangle"
        assert dialog.lbl_summary.text() == "1 region enabled: Face(0.50)"
        assert dialog.lbl_hint.text() == "Canvas preserves the output aspect ratio (current 640×480)"

        with patch.object(QFileDialog, "getOpenFileName", return_value=("", "")) as picker:
            dialog._add_mask_region()
            assert picker.call_args.args[1] == "Select Mask Image (white = active region)"

        ui_i18n.set_language("ja")
        assert dialog.windowTitle() == "領域プロンプト"
        assert dialog.table.item(0, COL_SHAPE).text() == "長方形"
        assert dialog.lbl_summary.text() == "1 個の領域を有効化: Face(0.50)"
        assert dialog.lbl_hint.text() == "キャンバスは出力解像度のアスペクト比で表示（現在 640×480）"
    finally:
        ui_i18n.set_language(old)
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()


def test_workflow_report_language():
    from unittest.mock import patch
    from PyQt6.QtWidgets import QMessageBox
    from ui.i18n import ui_i18n
    from ui.workflow_editor.window import WorkflowEditorWindow
    from workflow.preflight import ERROR, PreflightIssue, PreflightResult

    old = ui_i18n.language
    window = WorkflowEditorWindow()
    captured = []
    result = PreflightResult(
        ok=False,
        errors=[PreflightIssue(ERROR, "test", "boom") for _ in range(2)],
        estimated_vram_mb=2048,
        estimated_time=12,
        required_models=[{"present": False}],
        policy="auto",
    )
    try:
        with patch.object(QMessageBox, "information", side_effect=(
                lambda _parent, title, text, *args: captured.append((title, text)))):
            ui_i18n.set_language("en")
            window.on_resources()
            assert captured[-1] == (
                "Resource Summary", "This workflow does not require additional resources.")
        with patch("workflow.preflight.run_preflight", return_value=result), \
                patch.object(QMessageBox, "warning", side_effect=(
                    lambda _parent, title, text, *args: captured.append((title, text)))):
            window.on_preflight()
            text = captured[-1][1]
            assert "Preflight failed (2 errors)" in text
            assert "Estimated VRAM 2048 MB" in text
            assert "Estimated time ≈ 12s" in text
            assert "Resources: 1 required, 1 missing" in text

            ui_i18n.set_language("ja")
            window.on_preflight()
            text = captured[-1][1]
            assert "プレフライト失敗（2 件のエラー）" in text
            assert "推定 VRAM 2048 MB" in text
            assert "推定所要時間 約 12s" in text
            assert "リソース: 必要 1 件、不足 1 件" in text
    finally:
        ui_i18n.set_language(old)
        window.close()
        window.deleteLater()


def test_node_palette_language():
    from unittest.mock import patch
    from ui.i18n import ui_i18n
    from ui.workflow_editor.palette import NodePalette
    from workflow.schema import NodeSpec, ResourceRequirement

    spec = NodeSpec(
        type="I18nNode", category="模型",
        description="加载底模（.safetensors/.ckpt）。可填文件名或绝对路径。",
        resources=ResourceRequirement(estimated_vram_mb=2048, requires=("cuda",)),
    )
    old = ui_i18n.language
    palette = None
    try:
        with patch("ui.workflow_editor.palette.registry.load_builtin_nodes"), \
                patch("ui.workflow_editor.palette.registry.all_specs", return_value={"I18nNode": spec}):
            ui_i18n.set_language("en")
            palette = NodePalette()
            assert palette.findChildren(QLabel)[0].text() == "Node Library"
            assert palette.tree.topLevelItem(0).text(0) == "Models (1)"
            assert palette.hint.text() == "1 node type · Double-click to add · Del to delete"
            tooltip = palette.tree.topLevelItem(0).child(0).toolTip(0)
            assert "Load a base model (.safetensors/.ckpt)." in tooltip
            assert "Peak VRAM ≈ 2048 MB" in tooltip

            ui_i18n.set_language("ja")
            assert palette.tree.topLevelItem(0).text(0) == "モデル（1）"
            assert palette.hint.text() == "1 種類のノード · ダブルクリックで追加 · Del で削除"
            assert "基本モデル（.safetensors/.ckpt）を読み込みます。" in (
                palette.tree.topLevelItem(0).child(0).toolTip(0))
    finally:
        ui_i18n.set_language(old)
        if palette is not None:
            palette.close()
            palette.deleteLater()


def test_workflow_canvas_language():
    from unittest.mock import patch
    from ui.i18n import ui_i18n
    from ui.workflow_editor.canvas import GraphScene
    from workflow.schema import DataType, NodeSpec, ParamSpec, PortSpec

    spec = NodeSpec(
        type="I18nTest",
        description="加载底模（.safetensors/.ckpt）。可填文件名或绝对路径。",
        inputs={"image": PortSpec(DataType.IMAGE)},
        outputs={"out": PortSpec(DataType.IMAGE)},
        params={
            "seed": ParamSpec(kind="seed", default=-1),
            "image": ParamSpec(kind="str", default="", source="images"),
        },
    )
    old = ui_i18n.language
    scene = None
    messages = []
    try:
        ui_i18n.set_language("en")
        scene = GraphScene()
        scene.statusMessage.connect(messages.append)
        with patch("ui.workflow_editor.canvas.registry.exists", return_value=False):
            assert scene.add_node("Missing") is None
            assert messages[-1] == "Unregistered node type: Missing"

        with patch("ui.workflow_editor.canvas.registry.exists", return_value=True), \
                patch("ui.workflow_editor.canvas.registry.get", return_value=spec):
            node_id = scene.add_node("I18nTest")
            assert messages[-1] == "Added I18nTest (i18nTest_1)"
            item = scene.node_items[node_id]
            assert item.param_widgets["seed"].specialValueText() == "Random"
            assert item.preview_note == "No reference image selected"
            assert "Load a base model (.safetensors/.ckpt)." in item.toolTip()
            assert scene.connect_ports(item.ports_out["out"], item.ports_in["image"]) is False
            assert messages[-1] == "Cannot connect to the same node"

            ui_i18n.set_language("ja")
            assert item.param_widgets["seed"].specialValueText() == "ランダム"
            assert item.preview_note == "参照画像が選択されていません"
            assert "基本モデル（.safetensors/.ckpt）を読み込みます。" in item.toolTip()
    finally:
        ui_i18n.set_language(old)
        if scene is not None:
            scene.clear_items()
            scene.deleteLater()


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
    test_gallery_favorite_dialog_language()
    test_extension_uninstall_dialog_language()
    test_preset_empty_export_dialog_language()
    test_shell_busy_dialog_language()
    test_video_validation_status_language()
    test_scene_batch_dialog_language()
    test_workflow_editor_dialog_language()
    test_workflow_resource_dialog_language()
    test_workflow_failure_dialog_language()
    test_param_panel_language()
    test_region_editor_language()
    test_workflow_report_language()
    test_node_palette_language()
    test_workflow_canvas_language()
    print("UI_I18N_OK")
    sys.stdout.flush()
    # 与现有 Qt 离屏测试一致，跳过 Multimedia/Qt 的进程退出清理。
    os._exit(0)
