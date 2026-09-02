# tests/test_real_startup.py — 真实 AIDesktopApp 启动路径验证（无头）
# 覆盖: setup_ui + apply_config_to_ui + 预设/工具提示 + 画廊信号 + 信号桥
# 模型加载在后台 daemon 线程，本测试不等它，验证完构造路径即退出
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

if __name__ == "__main__":
    from ui.theme import apply_theme
    theme = apply_theme(_APP)

    import main
    assert main._UIMixin.__name__ == "ShellMixin", f"默认 UI 应为 ShellMixin，实际 {main._UIMixin.__name__}"

    win = main.AIDesktopApp()
    # 推进事件循环，让 QTimer.singleShot 等延迟调用跑一遍
    deadline = time.time() + 3
    while time.time() < deadline:
        _APP.processEvents()
        time.sleep(0.05)

    # 关键面核查
    assert win.nav is not None and set(win._pages) == {"txt2img", "img2img", "video", "gallery"}
    assert win.gallery is not None and win.filmstrip is not None
    assert win.btn_generate is not None and not win.btn_generate.isEnabled()  # 引擎预热中
    assert callable(win.append_log) and callable(win.play_video)
    # 配置还原跑过（控件存在即未炸）
    assert win.txt_prompt is not None and win.combo_model is not None
    # 页面切换不崩
    for pid in ("img2img", "video", "gallery", "txt2img"):
        win.nav.select(pid)
        _APP.processEvents()
    _APP.processEvents()
    sys.stdout.flush()
    print(f"✅ 真实启动路径验证通过 (theme={theme}, 默认UI=ShellMixin)")
    os._exit(0)   # 跳过后台 torch 线程与 Qt teardown
