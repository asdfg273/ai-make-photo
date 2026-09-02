# tests/test_ui_contract.py — UI 契约测试（无头，纯脚本）
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton

# 进程级唯一 QApplication，驻留全局防止被 GC（GC 后 Qt 状态损坏会崩）
_APP = QApplication.instance() or QApplication([])


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


def test_shell_skeleton():
    """外壳骨架：无模型加载，最小宿主验证 setup_ui 可跑通。"""
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    win.setup_ui()
    assert win.nav is not None and win.center_stack is not None
    assert callable(win.append_log) and callable(win.set_status)
    assert callable(win.set_progress) and callable(win.play_video)
    win.close()
    print("PASS test_shell_skeleton")


def test_core_widgets():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow
    from ui.contracts import check_contract, GLOBAL_WIDGETS

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    win.setup_ui()
    crit, minor = check_contract(win)
    for name in ("txt_prompt", "txt_neg", "combo_model", "combo_sampler",
                 "spin_steps", "spin_width", "spin_height", "btn_generate",
                 "btn_interrupt", "progress_gen"):
        assert name not in crit, f"关键控件缺失: {name}"
        assert getattr(win, name) is not None
    # 别名指向同一实例
    assert win.btn_gen is win.btn_generate
    # preview_canvas 别名在 Task 7（预览画布迁移）后断言
    win.close()
    print("PASS test_core_widgets")


if __name__ == "__main__":
    test_contract_lists()
    test_check_and_degrade()
    test_shell_skeleton()
    test_core_widgets()
    # Qt offscreen 退出时销毁控件可能段错误，直接 os._exit 跳过 teardown
    sys.stdout.flush()
    os._exit(0)
