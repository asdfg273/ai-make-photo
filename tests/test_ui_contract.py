# tests/test_ui_contract.py — UI 契约测试（无头，纯脚本）
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton

# 进程级唯一 QApplication，驻留全局防止被 GC（GC 后 Qt 状态损坏会崩）
_APP = QApplication.instance() or QApplication([])
# 测试窗口驻留列表：防止提前 GC，退出前统一清理（QMediaPlayer 线程需显式停）
_WINDOWS = []


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
    _WINDOWS.append(win)
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
    _WINDOWS.append(win)
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


def test_txt2img_page():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    _WINDOWS.append(win)
    win.setup_ui()
    assert "txt2img" in win._pages
    assert win.lbl_preview is not None
    assert win.txt_log_image is not None
    assert win.gallery is not None
    assert win.preview_canvas is win.lbl_preview   # 别名指向同一实例
    win.append_log("hello")            # 方法契约：写入 txt_log_image
    assert "hello" in win.txt_log_image.toPlainText()
    win.nav.select("txt2img")
    assert win.center_stack.currentWidget() is win._pages["txt2img"].workspace()
    win.close()
    print("PASS test_txt2img_page")


def test_img2img_page():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    _WINDOWS.append(win)
    win.setup_ui()
    assert "img2img" in win._pages
    assert win.btn_load_img is not None
    assert win.scale_strength is not None
    assert win.scale_str is win.scale_strength   # 别名指向真控件
    win.nav.select("img2img")
    assert win.params_stack.currentWidget() is win._pages["img2img"].params_widget()
    win.close()
    print("PASS test_img2img_page")


def test_shared_groups():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow
    from ui.contracts import check_contract

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    _WINDOWS.append(win)
    win.setup_ui()
    crit, minor = check_contract(win)
    assert crit == [], f"关键契约缺失: {crit}"
    for name in ("combo_lora_0", "scale_lora_0", "combo_cn_type",
                 "chk_enable_hires", "chk_enable_xy", "entry_x_vals"):
        assert getattr(win, name) is not None, name
    assert len(win.combo_loras) == 3        # 列表别名
    win.close()
    print("PASS test_shared_groups")


def test_video_page():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    _WINDOWS.append(win)
    win.setup_ui()
    assert "video" in win._pages
    for name in ("btn_gen_video", "txt_video_prompt", "combo_video_mode",
                 "video_player", "txt_log_video"):
        assert getattr(win, name) is not None, name
    win.nav.select("video")
    assert win.params_stack.currentWidget() is win._pages["video"].params_widget()
    assert not win.params_scroll.isHidden()
    win.close()
    print("PASS test_video_page")


if __name__ == "__main__":
    test_contract_lists()
    test_check_and_degrade()
    test_shell_skeleton()
    test_core_widgets()
    test_txt2img_page()
    test_img2img_page()
    test_shared_groups()
    test_video_page()
    # Qt offscreen 退出时销毁控件可能段错误，直接 os._exit 跳过 teardown;
    # 但 QMediaPlayer 线程需先显式停掉，否则进程悬挂
    for w in _WINDOWS:
        p = getattr(w, "video_player", None)   # 无 parent，不在对象树里
        if p is not None:
            p.stop()
            p.deleteLater()
    _APP.processEvents()
    sys.stdout.flush()
    os._exit(0)
