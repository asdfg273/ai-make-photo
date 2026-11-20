"""Anima 图生图路由 + 无管线架构回退测试 —— 可独立运行，也可被 pytest 收集。

背景
----
Anima（Cosmos DiT）加载器只建 ModularPipeline（AnimaAutoBlocks，原生支持
img2img：state 带 image/strength 即自动切换工作流）。旧代码在图生图路径
直接 `self.ai.img2img_pipe(**kwargs)` → TypeError: 'NoneType' object is
not callable，批量出图每张都崩。

修复后的行为：
- anima：image/strength 经 _adapt_call 带进 PipelineState，走原生 img2img；
- 其他无 img2img 管线的架构（如 SD3）：回退文生图并提示；
- _gt_safe_pipe_call 对 None 管线抛可读错误兜底。
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image
from utils.app_generation import GenerationMixin


class _FakeAI:
    """模拟 ModelManager。"""
    arch_profile = None
    ip_adapter_loaded = False
    ipa_ref_image = None

    def __init__(self, arch_id="anima", with_img2img=False):
        self.current_arch_id = arch_id
        self.txt2img_pipe = object()
        self.img2img_pipe = object() if with_img2img else None


class _Signal:
    def __init__(self):
        self.msgs = []

    def emit(self, *a):
        self.msgs.append(a)


class _Bridge:
    def __init__(self):
        self.status_signal = _Signal()


class _Recorder:
    """记录 _gt_safe_pipe_call 收到的 pipe 与 kwargs。"""

    def __init__(self):
        self.calls = []

    def __call__(self, pipe, ctx, **kwargs):
        self.calls.append((pipe, kwargs))
        return "FAKE_OUTPUT"


class _Host:
    """只携带 _gt_run_base_pipe / _adapt_call 所需最小属性的宿主。"""

    def __init__(self, ai):
        self.ai = ai
        self._bridge = _Bridge()
        self._rec = _Recorder()
        self._gt_safe_pipe_call = self._rec

    def run(self, ctx):
        return GenerationMixin._gt_run_base_pipe(self, ctx, {"prompt": "1girl"})


def _ctx(**over):
    base = dict(
        use_pose=False, use_inpaint=False, use_reference_only=False,
        skip_img2img=False, use_img2img=True,
        init_image=Image.new("RGB", (8, 8)), strength=0.6,
    )
    base.update(over)
    return base


def test_anima_img2img_routes_to_modular_pipe_with_image():
    """anima：img2img_pipe 为 None，但应把 image/strength 传给 ModularPipeline。"""
    host = _Host(_FakeAI(arch_id="anima", with_img2img=False))
    out = host.run(_ctx())
    assert out == "FAKE_OUTPUT"
    assert len(host._rec.calls) == 1
    pipe, kwargs = host._rec.calls[0]
    assert pipe is host.ai.txt2img_pipe
    assert isinstance(kwargs.get("image"), Image.Image)
    assert kwargs.get("strength") == 0.6
    assert host._bridge.status_signal.msgs == [], "anima 原生支持，不应有回退提示"


def test_sdxl_img2img_uses_img2img_pipe():
    host = _Host(_FakeAI(arch_id="sdxl", with_img2img=True))
    out = host.run(_ctx())
    assert out == "FAKE_OUTPUT"
    pipe, kwargs = host._rec.calls[0]
    assert pipe is host.ai.img2img_pipe
    assert "image" in kwargs and "strength" in kwargs
    assert host._bridge.status_signal.msgs == [], "正常路径不应有回退提示"


def test_sd3_no_img2img_falls_back_to_txt2img():
    """SD3 等确实没有 img2img 的架构：回退文生图并给出状态栏提示。"""
    host = _Host(_FakeAI(arch_id="sd3", with_img2img=False))
    out = host.run(_ctx())
    assert out == "FAKE_OUTPUT"
    pipe, kwargs = host._rec.calls[0]
    assert pipe is host.ai.txt2img_pipe
    assert "image" not in kwargs and "strength" not in kwargs
    assert host._bridge.status_signal.msgs, "回退时应给用户可见提示"
    assert "文生图" in host._bridge.status_signal.msgs[0][0]


def test_adapt_call_carries_image_and_strength_into_state():
    """_adapt_call 必须把 image/strength 带进 PipelineState（否则被 kwargs.clear() 丢弃）。"""
    host = _Host(_FakeAI(arch_id="anima"))
    kwargs = {
        "prompt": "1girl, fox_ears",
        "negative_prompt": "blurry",
        "num_inference_steps": 28,
        "guidance_scale": 5.0,
        "width": 512, "height": 768,
        "image": Image.new("RGB", (8, 8)),
        "strength": 0.6,
        "junk_key": "should_be_dropped",
    }
    adapted = GenerationMixin._adapt_call(host, None, {}, kwargs)
    assert "state" in adapted, "modular 架构应转成 state 调用"
    state = adapted["state"]
    values = state.values
    assert values["prompt"] == "1girl, fox_ears"
    assert values["strength"] == 0.6
    assert isinstance(values["image"], Image.Image)
    assert values["num_inference_steps"] == 28
    # 未识别的键被清空（走 state 声明式输入，不再透传）
    assert "junk_key" not in getattr(state, "values", {})


def test_safe_pipe_call_rejects_none_pipe_with_clear_error():
    host = _Host(_FakeAI(arch_id="anima"))
    try:
        GenerationMixin._gt_safe_pipe_call(host, None, {})
        raise AssertionError("None 管线应抛 RuntimeError")
    except RuntimeError as e:
        msg = str(e)
        assert "anima" in msg, "错误信息应包含架构名"
        assert "NoneType" not in msg, "不应再是费解的 TypeError 文案"


if __name__ == "__main__":
    _fns = [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]
    for n, f in _fns:
        f()
        print(f"PASS {n}")
    print(f"\n共 {len(_fns)} 项通过")
