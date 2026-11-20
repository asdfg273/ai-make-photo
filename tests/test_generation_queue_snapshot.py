# tests/test_generation_queue_snapshot.py — 队列快照 / 步间缓存 / LoRA 触发词（无头，不跑 GPU）
#
# 为什么要有这个文件
# ------------------
# 「生成队列」是新功能：点一次「加入队列」，把当前界面的**全部参数**存成快照，
# 轮到这个任务时再还原回界面继续生成。它自己的单测（test_inference_productivity）
# 只测了队列的 FIFO / 暂停 / 取消，**没有一个用例真的把参数存进去再还原出来**，
# 于是下面这些静默失效全都没被发现：
#
#   1. FloatSlider 不是 QSlider。`_restore_generation_snapshot` 里只有
#      `hasattr(widget, "set_value")` 分支，而 FloatSlider 只有
#      `float_value()/setValue(float)` → CFG、重绘强度、CN 强度、
#      ADetailer 强度、手部融合、高清重绘幅度**全部还原不了**，
#      排队的第二张及以后都沿用"界面上当下那一组滑杆值"，还不报错。
#   2. `_SNAPSHOT_CONTROLS` 里写的是 `scale_ad_hand_strength`，真实控件叫
#      `scale_ad_hand` → 手部强度从来不进快照。
#   3. 区域提示词 / 多参考图存在 config 里（不是宿主属性），早期快照完全没带，
#      中途改了区域，前面那张图会被"后来的区域设置"重跑。
#   4. `_StepCache.restore()` 曾经 `del model.__dict__["forward"]`：只要别人
#      （video_gen）先给同一个 unet 打过 forward 补丁，`del` 会把**别人的补丁
#      一起删掉**，退回类上的原始 forward。
#   5. `_sidecar_candidates()` 用相对路径 `Path("loras")` 找触发词旁文件，
#      换工作目录就静默读不到（LoRA 加载了但触发词没进提示词）。
#
# 用法：venv\Scripts\python.exe tests\test_generation_queue_snapshot.py
import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码，放最前）

_bootstrap.isolate_config_globally()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QSpinBox, QTextEdit, QWidget,
)

from ui.widgets import FloatSlider  # noqa: E402
from utils.generation_queue_controller import (  # noqa: E402
    GenerationQueueMixin, _SNAPSHOT_CONTROLS,
)

_APP = QApplication.instance() or QApplication([])


def _make_host():
    """只挂需要的那几个控件，避免起整个 AIDesktopApp。"""

    class Host(QWidget, GenerationQueueMixin):
        def __init__(self):
            super().__init__()
            self.scale_cfg = FloatSlider(1.0, 30.0, 0.1)
            self.scale_str = FloatSlider(0.0, 1.0, 0.01)
            self.scale_hires_denoise = FloatSlider(0.0, 1.0, 0.01)
            self.spin_steps = QSpinBox()
            self.spin_steps.setRange(1, 150)
            self.txt_prompt = QTextEdit()
            self.txt_neg = QTextEdit()
            self.chk_use_ipa_refs = QCheckBox()
            self.combo_res = QComboBox()
            for text in ("512x512", "832x1216"):
                self.combo_res.addItem(text)
            self.config = _Config()

    return Host()


class _Config:
    """config 只需要能读写这几个字段。"""

    def __init__(self):
        self.use_regions = False
        self.regions = []
        self.use_ipa_refs = False
        self.ipa_refs = []


# ----------------------------------------------------------------------
#  1) FloatSlider 必须能还原（这条就是"排队图参数串味"的根因）
# ----------------------------------------------------------------------
def test_float_sliders_are_restored():
    host = _make_host()
    host.scale_cfg.setValue(9.5)
    host.scale_str.setValue(0.73)
    host.scale_hires_denoise.setValue(0.41)
    host.spin_steps.setValue(28)
    host.txt_prompt.setPlainText("a cat")
    host.combo_res.setCurrentIndex(1)
    snapshot = host._capture_generation_snapshot()

    # 模拟"用户接着又改了一轮参数"
    host.scale_cfg.setValue(2.5)
    host.scale_str.setValue(0.0)
    host.scale_hires_denoise.setValue(0.99)
    host.spin_steps.setValue(6)
    host.txt_prompt.setPlainText("something else")
    host.combo_res.setCurrentIndex(0)

    host._restore_generation_snapshot(snapshot)

    assert round(host.scale_cfg.float_value(), 3) == 9.5, \
        f"CFG 没还原: {host.scale_cfg.float_value()}"
    assert round(host.scale_str.float_value(), 3) == 0.73, \
        f"重绘强度没还原: {host.scale_str.float_value()}"
    assert round(host.scale_hires_denoise.float_value(), 3) == 0.41, \
        f"高清重绘幅度没还原: {host.scale_hires_denoise.float_value()}"
    assert host.spin_steps.value() == 28, f"步数没还原: {host.spin_steps.value()}"
    assert host.txt_prompt.toPlainText() == "a cat"
    assert host.combo_res.currentText() == "832x1216"


# ----------------------------------------------------------------------
#  2) 快照控件名必须真实存在
# ----------------------------------------------------------------------
def test_snapshot_control_names_are_real():
    """写入 _SNAPSHOT_CONTROLS 的名字必须是真的控件，否则静默漏采。"""
    assert "scale_ad_hand" in _SNAPSHOT_CONTROLS, _SNAPSHOT_CONTROLS
    assert "scale_ad_hand_strength" not in _SNAPSHOT_CONTROLS, \
        "scale_ad_hand_strength 不是真实控件名（真名是 scale_ad_hand）"
    host = _make_host()
    snapshot = host._capture_generation_snapshot()
    # 挂了的控件必须都进了 values
    for name in ("scale_cfg", "scale_str", "scale_hires_denoise",
                 "spin_steps", "txt_prompt", "combo_res"):
        assert name in snapshot["values"], f"{name} 没进快照: {sorted(snapshot['values'])}"


# ----------------------------------------------------------------------
#  3) 区域提示词 / 多参考图（存在 config 里）必须进快照
# ----------------------------------------------------------------------
def test_config_backed_features_are_restored():
    host = _make_host()
    refreshed = []

    def _r():
        refreshed.append("regions")

    def _i():
        refreshed.append("ipa")

    host._refresh_regions_summary = _r
    host._refresh_ipa_refs_summary = _i
    host.config.use_regions = True
    host.config.regions = [{"name": "A", "prompt": "red dress", "enabled": True}]
    host.config.use_ipa_refs = True
    host.config.ipa_refs = [{"path": "a.png", "role": "identity", "enabled": True}]
    snapshot = host._capture_generation_snapshot()
    assert "regions" in (snapshot.get("config") or {}), snapshot.get("config")

    # 用户中途改了区域和参考图
    host.config.regions = [{"name": "B", "prompt": "blue sky", "enabled": True}]
    host.config.ipa_refs = [{"path": "b.png", "role": "style", "enabled": True}]
    host._restore_generation_snapshot(snapshot)

    assert host.config.regions == [{"name": "A", "prompt": "red dress", "enabled": True}], \
        f"区域没还原: {host.config.regions}"
    assert host.config.ipa_refs == [{"path": "a.png", "role": "identity", "enabled": True}], \
        f"参考图没还原: {host.config.ipa_refs}"
    assert "regions" in refreshed and "ipa" in refreshed, \
        f"还原后没刷新摘要标签: {refreshed}"


# ----------------------------------------------------------------------
#  4) 步间缓存还原不能删掉别人的 forward 补丁
# ----------------------------------------------------------------------
def test_step_cache_restore_keeps_outer_patch():
    from utils.step_cache import _StepCache

    class FakeUNet:
        def forward(self, x):
            return ("original", x)

    unet = FakeUNet()

    def outer_forward(x):
        return ("outer", x)

    # 别的模块（utils/video_gen.py 就是这么干的）先打了补丁
    unet.forward = outer_forward
    cache = _StepCache(unet, threshold=0.18, max_skips=2)
    assert unet.forward == cache._forward, "安装后 forward 应指向缓存包装"
    cache.restore()
    assert unet.forward is outer_forward, \
        "restore 把别人的 forward 补丁删掉了（del __dict__ 的老写法会这样）"
    # 重写后每个 lane 各存一份上次的输入/输出，restore 必须把它们都清掉，
    # 否则几张 2K 图的 latent 会一直挂在缓存对象上。
    for lane in cache._lanes:
        assert lane["prev_in"] is None and lane["prev_out"] is None, \
            "restore 后没有释放缓存的输入/输出张量"


# ----------------------------------------------------------------------
#  5) LoRA 触发词旁文件按绝对路径查找（换 CWD 也要命中）
# ----------------------------------------------------------------------
def test_lora_trigger_sidecar_uses_absolute_root():
    from utils import paths
    from utils.lora_triggers import trigger_for_display

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lora_dir = str(getattr(paths, "LORA_DIR", "") or "")
    assert lora_dir, "paths.LORA_DIR 不存在"
    sidecar = os.path.join(lora_dir, "_tmp_trigger_regression.txt")
    os.makedirs(lora_dir, exist_ok=True)
    with open(sidecar, "w", encoding="utf-8") as handle:
        handle.write("regressiontrigger")
    previous = os.getcwd()
    try:
        os.chdir(os.environ.get("TEMP", root))
        got = trigger_for_display("_tmp_trigger_regression")
        assert got == "regressiontrigger", \
            f"换工作目录后读不到触发词: {got!r}（说明还在用相对路径 loras/）"
    finally:
        os.chdir(previous)
        if os.path.exists(sidecar):
            os.remove(sidecar)


def test_queue_restores_ui_after_drain():
    """队列跑完要把界面还原成"排队前用户自己那套"。

    用户实测反馈：排 3 张、跑完以后界面停在**最后一张**的参数上，排队前
    自己调好的设置没了，还没有任何提示。`_generation_queue_tick` 每派发
    一张都会把任务快照写进控件（否则第二张会串味），所以结束时必须补一次还原。
    """
    host = _make_host()
    host.scale_cfg.setValue(9.5)
    host.txt_prompt.setPlainText("user's own settings")
    user_snapshot = host._capture_generation_snapshot()

    # 队列派发：第一张、第二张把界面改成各自任务的参数
    task_snapshot = dict(user_snapshot)
    task_snapshot["values"] = dict(user_snapshot["values"])
    task_snapshot["values"]["scale_cfg"] = 3.0
    host._restore_generation_snapshot(task_snapshot)

    host._generation_queue_user_snapshot = user_snapshot
    host._generation_queue_last_snapshot = task_snapshot
    host._generation_queue_dispatched = 2

    assert host._restore_queue_resting_state() is True, "队列结束应该还原界面"
    assert round(host.scale_cfg.float_value(), 3) == 9.5, \
        f"排队前的 CFG 没还原: {host.scale_cfg.float_value()}"
    assert host.txt_prompt.toPlainText() == "user's own settings"
    assert host._generation_queue_user_snapshot is None, "还原后状态要清空"


def test_queue_keeps_user_edits_made_while_running():
    """队列运行期间用户改过参数 → 不能拿排队前的快照覆盖他的新设置。"""
    host = _make_host()
    host.scale_cfg.setValue(9.5)
    user_snapshot = host._capture_generation_snapshot()

    task_snapshot = dict(user_snapshot)
    task_snapshot["values"] = dict(user_snapshot["values"])
    task_snapshot["values"]["scale_cfg"] = 3.0
    host._restore_generation_snapshot(task_snapshot)

    # 用户在看结果的同时自己把 CFG 调成了 6.5（≠ 最后一张任务的 3.0）
    host.scale_cfg.setValue(6.5)

    host._generation_queue_user_snapshot = user_snapshot
    host._generation_queue_last_snapshot = task_snapshot
    host._generation_queue_dispatched = 2

    assert host._restore_queue_resting_state() is False, \
        "用户改过界面时不该还原（会覆盖他的新设置）"
    assert round(host.scale_cfg.float_value(), 3) == 6.5, \
        f"用户的 6.5 被覆盖了: {host.scale_cfg.float_value()}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            failed.append(fn.__name__)
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("=" * 60)
    print(f"总计 {len(fns)} 项：PASS {len(fns) - len(failed)} / FAIL {len(failed)}")
    for f in failed:
        print(f"  失败: {f}")
    sys.stdout.flush()
    sys.exit(1 if failed else 0)
