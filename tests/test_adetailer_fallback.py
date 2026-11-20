# -*- coding: utf-8 -*-
"""ADetailer 在"没有 inpaint 管线"的架构上是否能真的工作。

背景（真实故障）：Anima 是 ModularPipeline，没有 inpaint —— 修脸/修手会打印
`⚠️ inpaint_pipe 不存在，跳过 ADetailer` 然后什么都不做。用户生成了
1536x2048 的大图、画面里明明有脸和手，两阶段却都是 0.00 秒跳过。

修法：没有 inpaint 时退化成「裁切 → 图生图 → 羽化贴回」。
这个测试锁死三件事：
  1. 没有 inpaint 但有 img2img → 必须真的跑，且不能传 mask_image；
  2. 两条管线都没有 → 返回原图且不抛异常（明确跳过）；
  3. 有 inpaint → 行为不变（仍然传 mask_image + 尺寸）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

import utils.image_processor as ip  # noqa: E402

RESULTS = []


def check(name):
    def deco(fn):
        try:
            fn()
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        else:
            RESULTS.append((name, True, ""))
        return fn
    return deco


class _Out:
    def __init__(self, image):
        self.images = [image]


class InpaintPipe:
    """蒙版 inpaint 管线：接受 mask_image / width / height。"""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt="", negative_prompt="", image=None,
                 mask_image=None, strength=0.3, num_inference_steps=24,
                 guidance_scale=6.5, width=None, height=None, **kw):
        self.calls.append(dict(prompt=prompt, image=image,
                               mask_image=mask_image, strength=strength,
                               width=width, height=height, **kw))
        size = (width or image.width, height or image.height)
        return _Out(Image.new("RGB", size, (255, 0, 0)))


class Img2ImgPipe:
    """SD1.5 风格 img2img：没有 width / height，也没有 mask_image。"""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt="", negative_prompt="", image=None,
                 strength=0.3, num_inference_steps=24, guidance_scale=6.5,
                 **kw):
        self.calls.append(dict(prompt=prompt, image=image,
                               strength=strength, **kw))
        assert "mask_image" not in kw, "img2img 不该收到 mask_image"
        return _Out(Image.new("RGB", image.size, (0, 0, 255)))


class ModularPipe:
    """Anima 风格：块式管线（有 blocks.sub_blocks），靠 **kwargs 传 state。"""

    def __init__(self):
        self.calls = []
        self.blocks = type("_Blocks", (), {"sub_blocks": {}})()

    def __call__(self, **kw):
        self.calls.append(kw)
        size = (kw.get("width") or 512, kw.get("height") or 512)
        return _Out(Image.new("RGB", size, (0, 255, 0)))


def _patch_detect(monkeypatch_box=(40, 50, 60, 70)):
    ip._yolo_detect = lambda model_path, cv_image, conf=0.3: [monkeypatch_box]
    ip._resolve_local_adetailer_model = lambda cfg: "dummy_yolo.pt"


def _target():
    return sorted(ip.ADETAILER_MODELS)[0]


@check("没有 inpaint 但有 img2img → 会跑，且不传 mask_image")
def test_falls_back_to_img2img():
    _patch_detect()
    base = Image.new("RGB", (256, 256), (10, 20, 30))
    i2i = Img2ImgPipe()
    out = ip.process_adetailer(
        base, None, "详细的脸", "模糊",
        strength=0.35, target=_target(), detail_size=128,
        img2img_pipe=i2i)
    assert len(i2i.calls) == 1, f"img2img 没被调用: {len(i2i.calls)}"
    call = i2i.calls[0]
    assert call["image"] is not None and call["image"].size != base.size, \
        "应该传裁切块而不是整图"
    assert call["strength"] == 0.35
    assert "width" not in call and "height" not in call, \
        "SD1.5 img2img 没有 width/height 形参，传了只会被静默忽略"
    # 只有检测框附近被贴回，四角保持原样
    assert out.size == base.size
    assert out.getpixel((0, 0)) == (10, 20, 30), "四角不该被改"
    cx, cy = (40 + 60) // 2, (50 + 70) // 2      # 检测框中心
    assert out.getpixel((cx, cy)) != (10, 20, 30), "检测框区域应该被重绘"


@check("ModularPipeline（Anima）→ 必须传 width/height")
def test_modular_pipe_gets_explicit_size():
    _patch_detect()
    base = Image.new("RGB", (300, 200), (10, 20, 30))
    pipe = ModularPipe()
    ip.process_adetailer(
        base, None, "脸", "", strength=0.3, target=_target(),
        detail_size=128, img2img_pipe=pipe)
    assert len(pipe.calls) == 1
    kw = pipe.calls[0]
    assert kw.get("width") and kw.get("height"), \
        f"没传尺寸，Anima 会按默认 1024 出图: {sorted(kw)}"
    assert kw["width"] % 8 == 0 and kw["height"] % 8 == 0, "尺寸要对齐 8"
    assert kw["image"] is not None


@check("两条管线都没有 → 返回原图、不抛异常")
def test_returns_original_without_any_pipe():
    _patch_detect()
    base = Image.new("RGB", (64, 64), (5, 5, 5))
    out = ip.process_adetailer(base, None, "脸", "", target=_target())
    assert isinstance(out, Image.Image)
    assert out.tobytes() == base.tobytes(), "没有管线时应原样返回"


@check("有 inpaint 时行为不变：仍然传 mask_image + 尺寸")
def test_inpaint_path_unchanged():
    _patch_detect()
    base = Image.new("RGB", (256, 256), (10, 20, 30))
    inpaint = InpaintPipe()
    ip.process_adetailer(
        base, inpaint, "脸", "", strength=0.4, target=_target(),
        detail_size=128, img2img_pipe=Img2ImgPipe())
    assert len(inpaint.calls) == 1
    kw = inpaint.calls[0]
    assert kw["mask_image"] is not None, "inpaint 路径必须带蒙版"
    assert kw["width"] and kw["height"]
    assert kw["strength"] == 0.4


@check("call 适配器优先于直连管线")
def test_custom_call_is_used():
    _patch_detect()
    base = Image.new("RGB", (128, 128), (10, 20, 30))
    seen = []

    def _call(pipe, kwargs):
        seen.append(sorted(kwargs))
        size = (kwargs.get("width") or 64, kwargs.get("height") or 64)
        return _Out(Image.new("RGB", size, (200, 100, 50)))

    ip.process_adetailer(
        base, None, "脸", "", strength=0.3, target=_target(),
        detail_size=64, img2img_pipe=ModularPipe(), call=_call)
    assert len(seen) == 1 and "image" in seen[0] and "prompt" in seen[0]


@check("主界面两个阶段都传了 img2img_pipe 与 call")
def test_app_generation_passes_fallback():
    src = (ROOT / "utils" / "app_generation.py").read_text(encoding="utf-8")
    assert src.count("img2img_pipe=getattr(") >= 2, "脸部/手部两处都要传"
    assert src.count("call=self._adetailer_call") >= 2
    assert "def _adetailer_call(" in src
    assert "def _adetailer_call(self, pipe, call_kwargs)" in src


@check("Anima 能力表承认 adetailer（走图生图兜底）")
def test_anima_caps_allow_adetailer():
    from workflow.adapters import capabilities_for
    caps = capabilities_for("anima")
    assert caps.supports("adetailer") is True, \
        "现在有图生图兜底，能力表不应再声称不支持"
    assert caps.supports("inpaint") is False, "但仍然没有真正的 inpaint"
    assert "裁切" in (caps.notes or "") or "图生图" in (caps.notes or "")


def main():
    print("=" * 60)
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}"
              + (f": {detail}" if detail else ""))
    failed = [r for r in RESULTS if not r[1]]
    print("=" * 60)
    print(f"总计 {len(RESULTS)} 项：PASS {len(RESULTS) - len(failed)} / "
          f"FAIL {len(failed)}")
    if failed:
        print("  失败:", ", ".join(n for n, _, _ in failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
