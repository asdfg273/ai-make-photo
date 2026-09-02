# tests/test_ui_contract.py — UI 契约测试（无头，纯脚本）
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton


def test_contract_lists():
    from ui.contracts import (GLOBAL_WIDGETS, PAGE_WIDGETS, METHOD_CONTRACT,
                              ALIASES, LIST_ALIASES, CRITICAL)
    assert "txt_prompt" in GLOBAL_WIDGETS
    assert "combo_model" in GLOBAL_WIDGETS
    assert "btn_generate" in GLOBAL_WIDGETS
    assert set(PAGE_WIDGETS) == {"txt2img", "img2img", "video", "gallery"}
    assert "play_video" in METHOD_CONTRACT
    assert ALIASES["btn_gen"] == "btn_generate"
    assert "btn_generate" in CRITICAL


def test_check_and_degrade():
    from ui.contracts import install_aliases, check_contract, apply_degradation
    app = QApplication.instance() or QApplication([])

    class FakeHost:  # 最小假宿主：只有部分控件
        pass

    host = FakeHost()
    host.txt_prompt = QLineEdit()
    host.btn_generate = QPushButton()
    crit, minor = check_contract(host)
    assert "combo_model" in crit            # 关键缺失被识别
    apply_degradation(host, crit)
    assert not host.btn_generate.isEnabled()  # 生成按钮被置灰
    # 别名安装
    install_aliases(host)
    assert host.btn_gen is host.btn_generate
    print("PASS test_contract_lists / test_check_and_degrade")


if __name__ == "__main__":
    test_contract_lists()
    test_check_and_degrade()
