# tests/test_tiled_hires.py — 分块高清（放大→逐块 img2img→羽化拼回）
#
# 为什么要有这个文件
# ------------------
# 8GB 显存直接出 1024² 会爆（实测 Anima 峰值 9679MB > 8123MB 总量，走 WDDM 换页）。
# 分块高清的可行性在真机上验证过了，但过程中踩了两个必须锁死的坑：
#
#   1. **每块必须显式传 height/width**。ModularPipeline 不传就按默认 1024 出图，
#      等于每块都在跑整图：单块 28~30s、峰值 9679MB；传了以后单块 3.8s、
#      4 块合计 15.5s、峰值 7099MB。少传一个参数就是 7 倍慢 + 爆显存。
#   2. 调用方（主界面阶段 2）会把基础生成的 kwargs 带下来，里面含
#      `width/height/image/strength`，不清掉就会覆盖逐块的参数。
#
# 另外锁死：denoise=0 不跑扩散、块数超限提前报错、种子可复现、
# 拼回尺寸必须等于放大后的目标尺寸（不能出残图）。
#
# 用法：venv\Scripts\python.exe tests\test_tiled_hires.py
import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码，放最前）

_bootstrap.isolate_config_globally()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeResult:
    def __init__(self, image):
        self.images = [image]


class RecordingPipe:
    """假 img2img 管线：记录每次调用的 kwargs，返回与请求尺寸一致的图。"""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        from PIL import Image

        self.calls.append(dict(kwargs))
        size = (int(kwargs.get("width", 512)), int(kwargs.get("height", 512)))
        src = kwargs.get("image")
        if src is not None:
            img = src.convert("RGB").resize(size)
        else:
            img = Image.new("RGB", size, (120, 120, 120))
        # 每块叠一点不同的颜色，便于确认"拼回去用的是各块的输出"
        px = img.load()
        px[0, 0] = (len(self.calls) * 10 % 256, 7, 9)
        return _FakeResult(img)


def _base_image(w=256, h=256):
    from PIL import Image

    return Image.new("RGB", (w, h), (90, 140, 200))


def test_resolve_mode_maps_ui_and_config():
    from utils import tiled_hires as th

    # 下拉明确选了什么就听下拉的
    assert th.resolve_mode(ui_text="分块 (省显存)", cfg_mode="latent") == "tiled"
    assert th.resolve_mode(ui_text="整图 (快)", cfg_mode="tiled") == "latent"
    # 「自动」按显存判断：<10GB 分块，>=10GB 整图
    assert th.resolve_mode(ui_text="自动", cfg_mode="auto", total_gb=7.9) == "tiled"
    assert th.resolve_mode(ui_text="", cfg_mode="auto", total_gb=24.0) == "latent"
    # 配置里写死了也认
    assert th.resolve_mode(ui_text="自动", cfg_mode="tiled", total_gb=24.0) == "tiled"
    # 显存探测失败/未知时不崩，结果只能是两种模式之一（真机上 detect 会返回 7.9GB）
    assert th.resolve_mode(ui_text="自动", cfg_mode="auto",
                           total_gb=None) in ("latent", "tiled")


def test_pipe_without_width_height_does_not_get_them():
    """SD1.5/SDXL 的 img2img 没有 width/height 参数，硬传会 TypeError。

    真机验证过：`StableDiffusionImg2ImgPipeline.__call__` 与
    `StableDiffusionXLImg2ImgPipeline.__call__` 的签名里都没有 width/height
    （尺寸跟着传入的 image 走），只有 SD3 的 img2img 有。
    """
    from utils import tiled_hires as th

    class Sd15LikePipe:
        def __init__(self):
            self.calls = []

        def __call__(self, prompt, image, strength, num_inference_steps,
                     generator=None, guidance_scale=None):
            self.calls.append({"prompt": prompt, "image": image,
                               "strength": strength,
                               "num_inference_steps": num_inference_steps})
            return _FakeResult(image.convert("RGB"))

    pipe = Sd15LikePipe()
    out, meta = th.tiled_img2img(
        pipe, _base_image(128, 128), scale=2.0, tile_size=128, overlap=16,
        denoise=0.2, color_correct=False,
        call_kwargs={"prompt": "a park", "num_inference_steps": 20})
    assert out.size == (256, 256)
    assert meta["tile_pass_size"] is False, meta
    assert pipe.calls, "管线没被调用"
    for kw in pipe.calls:
        assert "width" not in kw and "height" not in kw, kw.keys()
        assert kw["image"].size == (meta["tile_size"], meta["tile_size"]), (
            kw["image"].size, meta["tile_size"])


def test_flow_lets_sd15_autodetect_and_forces_size_for_anima():
    """主界面分块路径：SD1.5/SDXL 走自动检测，Anima 强制传尺寸。"""
    import io

    src = io.open(os.path.join(ROOT, "utils", "app_generation.py"),
                  encoding="utf-8").read()
    assert "_pass_size = True if _arch == \"anima\" else None" in src
    assert "pass_size=_pass_size" in src


class _StubCtx:
    """节点只用到这几个成员（status/progress/check_cancel/metadata/device）。"""

    def __init__(self):
        self.metadata = {}
        self.device = "cpu"
        self.statuses = []
        self.progresses = []

    def status(self, text):
        self.statuses.append(text)

    def progress(self, done, total):
        self.progresses.append((done, total))

    def check_cancel(self):
        pass


class _FakeCond:
    def __init__(self, text, ready=False):
        self.text = text
        self.ready = ready
        self.embeds = "EMBEDS" if ready else None
        self.negative_embeds = None
        self.pooled = None


class _FakeModel:
    def __init__(self, arch_id, adapter, pipe):
        self.arch_id = arch_id
        self.adapter = adapter
        self._pipe = pipe

    @property
    def model_path(self):
        return "fake.safetensors"


def _run_tiled_node(arch_id, pipe, adapter, denoise=0.35, steps=20,
                    cond=None):
    """跑一遍 TiledHiresFix 节点，返回 (结果, 假管线收到的调用, ctx)。"""
    from nodes import tile_nodes

    model = _FakeModel(arch_id, adapter, pipe)
    adapter.pipe = lambda _m, _kind="img2img": pipe
    import types

    node = types.SimpleNamespace(params={
        "scale": 2.0, "denoise": denoise, "tile_size": 128, "overlap": 16,
        "steps": steps, "cfg": 7.0, "seed": 7, "upscale_method": "Lanczos",
        "blend": "羽化", "color_correct": False, "vram_mode": "自动",
        "max_tiles": 0})
    ctx = _StubCtx()
    inputs = {"model": model, "image": _base_image(128, 128),
              "positive": cond or _FakeCond("a park"), "negative": None}
    res = tile_nodes.run_tiled_hires_fix(ctx, node, inputs)
    return res, pipe.calls, ctx


def test_auto_mode_prefers_latent_when_pipe_can_offload():
    """「自动」只在不能 offload 的架构（Anima）上才选分块。

    能 offload 的 SD1.5/SDXL 在 1024² 整图重绘更快、也没有块边界，
    自动档把它们切成分块反而是倒退。
    """
    import types

    try:
        import torch
        if not torch.cuda.is_available():
            print("      (无 CUDA，跳过)")
            return
    except Exception:
        print("      (无 torch，跳过)")
        return

    from utils.app_generation import GenerationMixin

    class _Combo:
        def currentText(self):
            return "自动"

    class _Host(GenerationMixin):
        def __init__(self, pipe):
            self.combo_hires_mode = _Combo()
            self.config = types.SimpleNamespace(hires_mode="auto")
            self.ai = types.SimpleNamespace(img2img_pipe=pipe)

    offloadable = types.SimpleNamespace(enable_model_cpu_offload=lambda: None)
    assert _Host(offloadable)._hires_mode_resolved() == "latent"
    # ModularPipeline 没有 offload 钩子 → 必须分块
    assert _Host(object())._hires_mode_resolved() == "tiled"


def test_node_tiled_hires_feeds_prompt_and_size_for_anima():
    """Anima（非 Compel、ModularPipeline）：提示词不能被丢掉，每块必须带尺寸。"""
    from workflow.adapters import AnimaAdapter

    pipe = RecordingPipe()
    res, calls, _ctx = _run_tiled_node("anima", pipe, AnimaAdapter())
    assert calls, "分块节点没有调用管线"
    assert res["image"].size == (256, 256)
    for kw in calls:
        assert kw.get("prompt") == "a park", f"提示词丢了: {kw.get('prompt')!r}"
        assert "width" in kw and "height" in kw, "Anima 没拿到逐块尺寸"
    # 强度被夹到上限、步数被补足
    assert res["metadata"]["tile_denoise"] == 0.25, res["metadata"]
    assert res["metadata"]["tile_steps"] >= 12, res["metadata"]


def test_node_tiled_hires_keeps_sd15_contract():
    """SD1.5：走 prompt_embeds，且绝不传 width/height（管线没这两个参数）。"""
    from workflow.adapters import SD15Adapter

    class Sd15LikePipe:
        def __init__(self):
            self.calls = []

        def __call__(self, prompt_embeds=None, image=None, strength=None,
                     num_inference_steps=None, generator=None,
                     guidance_scale=None, negative_prompt_embeds=None):
            self.calls.append({"prompt_embeds": prompt_embeds, "image": image,
                               "strength": strength,
                               "num_inference_steps": num_inference_steps})
            return _FakeResult(image.convert("RGB"))

    pipe = Sd15LikePipe()
    _res, calls, _ctx = _run_tiled_node("sd15", pipe, SD15Adapter(),
                                        denoise=0.2,
                                        cond=_FakeCond("a park", ready=True))
    assert calls
    for kw in calls:
        assert "width" not in kw and "height" not in kw, kw.keys()
        assert kw.get("prompt_embeds") == "EMBEDS", "SD1.5 应走嵌入"


def test_plan_target_size_aligns_to_multiple_of_16():
    from utils import tiled_hires as th

    # 502x394 → 对齐到 16 的倍数 → 496x384（Anima 的 patch/VAE 要求 16）
    w, h = th.plan_target_size(_base_image(251, 197), 2.0)
    assert w % th.SIZE_MULTIPLE == 0 and h % th.SIZE_MULTIPLE == 0, \
        f"尺寸没按 {th.SIZE_MULTIPLE} 对齐: {w}x{h}"
    assert w % 16 == 0 and h % 16 == 0, f"必须能被 16 整除: {w}x{h}"
    assert abs(w - 502) <= 16 and abs(h - 394) <= 16, f"倍率算错了: {w}x{h}"
    w, h = th.plan_target_size(_base_image(512, 512), 1.0)
    assert (w, h) == (512, 512)
    # 非法倍率夹到 1.0，不放大
    w, h = th.plan_target_size(_base_image(64, 64), 0.2)
    assert (w, h) == (64, 64)
    # 每块的裁剪尺寸也必须能被 16 整除，否则 Anima 的 patch_embed 会崩
    plan = th.tiling.plan_tiles(1000, 1200, 640, 48)
    for t in plan:
        assert t["w"] % 16 == 0 and t["h"] % 16 == 0, t


def test_every_tile_gets_explicit_size_and_returns_full_canvas():
    from utils import tiled_hires as th

    pipe = RecordingPipe()
    out, meta = th.tiled_img2img(
        pipe, _base_image(256, 256),
        scale=2.0, tile_size=320, overlap=48, denoise=0.25, seed=1234,
        call_kwargs={"prompt": "a park", "num_inference_steps": 4,
                     # 这些是主界面阶段 2 会带下来的污染项，必须被清掉
                     "width": 999, "height": 888, "strength": 0.99,
                     "latents": "junk"},
        color_correct=False,
    )
    assert out.size == (512, 512), f"拼回尺寸不对: {out.size}"
    assert len(pipe.calls) == meta["tile_count"] >= 2, \
        f"块数与记录不一致: {len(pipe.calls)} vs {meta['tile_count']}"
    for i, kw in enumerate(pipe.calls):
        assert "width" in kw and "height" in kw, f"第 {i+1} 块没传尺寸"
        assert kw["width"] != 999 and kw["height"] != 888, \
            f"第 {i+1} 块被调用方的 width/height 污染了: {kw['width']}x{kw['height']}"
        assert kw["strength"] == 0.25, f"第 {i+1} 块强度被覆盖: {kw['strength']}"
        assert "latents" not in kw, f"第 {i+1} 块还带着 latents"
        assert kw["prompt"] == "a park", "prompt 没传下去"
        # 4 步 × 0.25 只有 1 步生效 → 会被抬到 12 步（见步数用例）
        assert kw["num_inference_steps"] >= 4, kw["num_inference_steps"]
        # 尺寸必须是 8 的倍数，否则 VAE 会报错
        assert kw["width"] % 8 == 0 and kw["height"] % 8 == 0, \
            f"第 {i+1} 块尺寸没对齐: {kw['width']}x{kw['height']}"
    assert meta["tiled_hires"] is True
    assert meta["hires_size"] == "512x512"
    # 实际重叠是 (x, y) 两元组：overlap 参数只是下界，均摊后常更大
    ov = meta["tile_overlap_effective"]
    assert isinstance(ov, tuple) and max(ov) >= 48, \
        f"实际重叠不对: {ov}（请求 48）"
    for key in ("tile_size", "tile_overlap", "tile_count", "tile_blend",
                "tile_method", "tile_seed", "tile_denoise"):
        assert key in meta, f"元数据缺 {key}"


def test_tile_seeds_are_reproducible_and_increment():
    from utils import tiled_hires as th

    pipe = RecordingPipe()
    _out, meta = th.tiled_img2img(pipe, _base_image(192, 192), scale=2.0,
                                  tile_size=256, overlap=32, denoise=0.4,
                                  seed=777, color_correct=False)
    assert meta["tile_seed"] == 777
    gens = [kw.get("generator") for kw in pipe.calls]
    assert all(g is not None for g in gens), "有的块没拿到 generator"
    # torch.Generator 的初始种子应当各不相同（base+i）
    seeds = [int(g.initial_seed()) if hasattr(g, "initial_seed") else None
             for g in gens]
    assert all(s is not None for s in seeds), "generator 不是 torch.Generator"
    assert len(set(seeds)) == len(seeds), f"块之间种子重复了: {seeds}"


def test_zero_denoise_is_pure_upscale_without_diffusion():
    from utils import tiled_hires as th

    pipe = RecordingPipe()
    out, meta = th.tiled_img2img(pipe, _base_image(100, 100), scale=2.0,
                                 denoise=0.0, seed=1)
    assert pipe.calls == [], "denoise=0 不该调用管线"
    # 200 → 对齐到 16 → 192
    assert out.size == (192, 192), out.size
    assert out.size[0] % th.SIZE_MULTIPLE == 0, out.size
    assert meta["tile_count"] == 0
    assert "未跑扩散" in str(meta.get("tile_note", ""))


def test_max_tiles_guard_raises_before_running():
    from utils import tiled_hires as th

    pipe = RecordingPipe()
    try:
        th.tiled_img2img(pipe, _base_image(512, 512), scale=4.0,
                         tile_size=256, overlap=32, denoise=0.3, max_tiles=3)
    except ValueError as e:
        assert "最大块数" in str(e), e
    else:
        raise AssertionError("块数超限没有报错")
    assert pipe.calls == [], "超限报错前就已经开始跑了"


def test_progress_and_status_callbacks_fire():
    from utils import tiled_hires as th

    pipe = RecordingPipe()
    progress, statuses = [], []

    def _p(done, total):
        progress.append((done, total))

    out, meta = th.tiled_img2img(
        pipe, _base_image(128, 128), scale=2.0, tile_size=128, overlap=16,
        denoise=0.3, seed=3,
        progress=_p, status=statuses.append, color_correct=False)
    assert out.size == (256, 256)
    assert progress and progress[-1][0] == meta["tile_count"], progress
    assert any("分块高清" in s for s in statuses), statuses


def test_anima_loader_shares_one_pipeline_for_txt2img_and_img2img():
    """Anima 是 ModularPipeline：同一对象按传参选 block。

    以前 build() 只返回 txt2img，img2img_pipe 恒 None →
    阶段 2 Hires.fix 直接 `'NoneType' object is not callable`。
    这里从源码层面锁死"两个槽位挂同一个对象"。
    """
    src = open(os.path.join(ROOT, "core", "loaders", "anima.py"),
               encoding="utf-8").read()
    assert "LoadResult(txt2img=pipe, img2img=pipe)" in src, \
        "Anima build() 没有把管线同时挂到 img2img 槽位"


def test_stage2_hires_uses_hires_mode_dropdown():
    """主界面阶段 2 必须真的读「高清方式」下拉（以前只写 config，没人读）。"""
    src = open(os.path.join(ROOT, "utils", "app_generation.py"),
               encoding="utf-8").read()
    assert "_hires_mode_resolved()" in src, "阶段 2 没解析高清方式"
    assert 'elif _hmode == "tiled"' in src, "阶段 2 没有分块分支"
    assert "self._gt_tiled_hires(image, ctx, kwargs)" in src, \
        "阶段 2 分块分支没调用 _gt_tiled_hires"
    assert "def _gt_tiled_hires" in src
    assert "def _hires_mode_resolved" in src


def test_effective_steps_are_boosted_but_strength_is_kept():
    """生效步数不足会把每块整块重画：抬总步数，但不动用户的强度。

    真机实测（Anima，640² 单块，MAD 相对底图裁剪）：
      steps=4  strength=0.35（生效 1 步）→ MAD 23.74  一只猫变三只
      steps=9  strength=0.35（生效 3 步）→ 正常
      steps=20 strength=0.15（生效 3 步）→ MAD  6.72
    """
    from utils import tiled_hires as th

    assert th.boost_steps(4, 0.35) == 9, th.boost_steps(4, 0.35)
    assert th.boost_steps(20, 0.15) == 20, "已经够 3 步不该再抬"
    assert th.boost_steps(20, 0.35) == 20, "生效 7 步也不该抬"
    # 强度越高，需要的总步数越少
    assert th.boost_steps(4, 0.6) == 5, th.boost_steps(4, 0.6)
    # 边界：不跑扩散 / 非法值不炸
    assert th.boost_steps(4, 0.0) == 4
    assert th.boost_steps(0, 0.35) == 0

    pipe = RecordingPipe()
    out, meta = th.tiled_img2img(
        pipe, _base_image(128, 128), scale=2.0, tile_size=128, overlap=16,
        denoise=0.2, seed=5, color_correct=False,
        call_kwargs={"num_inference_steps": 4})
    assert out.size == (256, 256)
    assert meta["tile_steps"] == 15, meta
    assert meta["tile_effective_steps"] >= th.MIN_EFFECTIVE_STEPS, meta
    assert meta["tile_denoise"] == 0.2, "强度被改了"
    for kw in pipe.calls:
        assert kw["num_inference_steps"] == 15, kw["num_inference_steps"]
        assert kw["strength"] == 0.2


def test_denoise_above_cap_is_clamped_and_reported():
    """分块强度 0.35 会把主体各画各的（实测一只猫变三只），必须夹住并告知。"""
    from utils import tiled_hires as th

    pipe = RecordingPipe()
    _out, meta = th.tiled_img2img(
        pipe, _base_image(128, 128), scale=2.0, tile_size=128, overlap=16,
        denoise=0.6, seed=9, color_correct=False,
        call_kwargs={"num_inference_steps": 20})
    assert meta["tile_denoise"] == th.TILED_MAX_DENOISE, meta["tile_denoise"]
    assert meta["tile_denoise_requested"] == 0.6, meta
    for kw in pipe.calls:
        assert kw["strength"] == th.TILED_MAX_DENOISE, kw["strength"]
    # 显式放行时才吃用户的高强度
    pipe2 = RecordingPipe()
    _o2, meta2 = th.tiled_img2img(
        pipe2, _base_image(128, 128), scale=2.0, tile_size=128, overlap=16,
        denoise=0.6, seed=9, color_correct=False, max_denoise=1.0,
        call_kwargs={"num_inference_steps": 20})
    assert meta2["tile_denoise"] == 0.6, meta2["tile_denoise"]
    assert pipe2.calls[0]["strength"] == 0.6


def test_anima_capabilities_do_not_claim_inpaint():
    """模块化管线没有 mask 输入，能力表别乱说支持（SD3 同款问题）。"""
    from workflow.adapters import capabilities_for

    caps = capabilities_for("anima")
    assert caps.supports_img2img is True, "Anima 其实支持 img2img（已实测 1.5s 出图）"
    assert caps.supports_inpaint is False, \
        "Anima 没有 mask 输入，能力表不该说支持 inpaint"


def test_tile_denoise_has_its_own_control():
    """分块强度必须是用户能调的控件，不能只吃整图那个「降噪强度」。"""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    groups = (root / "ui" / "shared_groups.py").read_text(encoding="utf-8")
    assert "scale_hires_tile_denoise" in groups, "主界面没有分块强度滑杆"
    assert "FloatSlider(0.10, 0.35" in groups, "分块强度上限应为 0.35"

    cfg = (root / "core" / "config_manager.py").read_text(encoding="utf-8")
    assert "hires_tile_denoise" in cfg, "配置里没有 hires_tile_denoise"

    contracts = (root / "ui" / "contracts.py").read_text(encoding="utf-8")
    assert "scale_hires_tile_denoise" in contracts, "控件名没进契约表"

    gen = (root / "utils" / "app_generation.py").read_text(encoding="utf-8")
    assert "self.scale_hires_tile_denoise.float_value()" in gen, \
        "分块高清没读这个滑杆"
    # 队列快照 / 预设快照都要带上，否则排队与预设切换会丢设置
    for name in ("utils/generation_queue_controller.py",
                 "ui/param_snapshot.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "scale_hires_tile_denoise" in src, f"{name} 没带分块强度"


def test_clamp_message_is_short_info():
    """夹取只说结果，不再长篇劝告（用户已有滑杆可调）。"""
    import logging
    from io import StringIO

    from utils import tiled_hires as th

    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    old_level = th.logger.level
    th.logger.addHandler(handler)
    th.logger.setLevel(logging.INFO)
    try:
        _out, meta = th.tiled_img2img(
            RecordingPipe(), _base_image(256, 256), scale=1.5, tile_size=256,
            overlap=64, denoise=0.35, call_kwargs={"num_inference_steps": 20},
            seed=1)
    finally:
        th.logger.removeHandler(handler)
        th.logger.setLevel(old_level)
    text = buf.getvalue()
    assert meta["tile_denoise"] == th.TILED_MAX_DENOISE, meta["tile_denoise"]
    assert meta["tile_denoise_requested"] == 0.35, meta
    assert "按上限" in text, text
    assert "确实要重构图" not in text, "别再塞无法采纳的长篇建议"
    assert "超过上限" not in text, text


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
