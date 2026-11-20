# tests/test_photo_turn_ui.py
"""修图窗口视觉契约：专业文案、无表情符号、复用主界面主题。"""
import ast
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from PyQt6.QtWidgets import QApplication, QPushButton

from photo_turn.pro_editor_qt import ProImageEditor
from ui.theme import PALETTE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "tests", "_photo_turn_ui_src.png")


def _decorative_chars(text):
    ranges = (
        (0x2190, 0x21FF),  # 装饰箭头
        (0x2300, 0x23FF),  # 技术符号
        (0x2600, 0x27BF),  # 杂项符号与表情
        (0x2B00, 0x2BFF),
        (0x1F000, 0x1FAFF),
        (0xFE00, 0xFE0F),
    )
    return sorted({c for c in text if any(lo <= ord(c) <= hi for lo, hi in ranges)})


def test_no_decorative_ui_literals():
    offenders = []
    source_dir = os.path.join(ROOT, "photo_turn")
    for name in sorted(os.listdir(source_dir)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(source_dir, name)
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                chars = _decorative_chars(node.value)
                if chars:
                    offenders.append(f"{name}:{node.lineno} {chars!r} {node.value!r}")
    assert not offenders, "修图模块仍含装饰性字符：\n" + "\n".join(offenders[:30])


def test_editor_visual_contract():
    Image.new("RGB", (320, 240), (100, 120, 140)).save(SRC)
    editor = ProImageEditor(None, SRC)
    try:
        assert editor.windowTitle() == "图像编辑器"
        assert editor.lbl_status.text() == "就绪"
        assert PALETTE["bg"] in editor.styleSheet()
        assert PALETTE["accent"] in editor.styleSheet()

        buttons = {button.text(): button for button in editor.findChildren(QPushButton)}
        expected = {
            "画笔", "橡皮", "遮罩", "裁剪", "文字", "对比原图",
            "构图网格", "全屏", "适配窗口", "另存为…", "保存并返回",
            "发送到局部重绘", "人脸修复",
        }
        assert expected <= buttons.keys(), f"缺少按钮：{sorted(expected - buttons.keys())}"
        assert buttons["保存并返回"].property("kind") == "primary"
        assert "Ctrl+S" in buttons["保存并返回"].toolTip()
        assert buttons["人脸修复"].toolTip() == "检测并修复画面中的人脸"

        visible_text = [editor.windowTitle(), editor.toolTip()]
        for widget in editor.findChildren(type(editor.lbl_status)):
            visible_text.extend((widget.text(), widget.toolTip()))
        for button in buttons.values():
            visible_text.extend((button.text(), button.toolTip()))
        assert not _decorative_chars("\n".join(visible_text))
    finally:
        editor.close()
        if os.path.exists(SRC):
            os.remove(SRC)


def main():
    app = QApplication([])
    failures = []
    for test in (test_no_decorative_ui_literals, test_editor_visual_contract):
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures.append((test.__name__, str(exc)))
            print(f"FAIL {test.__name__}: {exc}")
    app.quit()
    if failures:
        raise SystemExit(1)
    print("PHOTO_TURN_UI_OK")


if __name__ == "__main__":
    main()
