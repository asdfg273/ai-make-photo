# -*- coding: utf-8 -*-
"""视频增强回归：Lightning 加速 / 分段长视频 / LTX-Video+Wan2.1 入口 / 中文路径。

这些点以前没有任何自动化覆盖：
  · AnimateDiff-Lightning 的 4/8 步档位判断
  · "长视频"复选框到底做了什么（旧实现收参数不用）
  · cv2 在 Windows 下打不开中文路径（视频用户的目录几乎一定带中文）
  · LTX-Video / Wan2.1 只在架构层留入口，必须能识别、能说清为什么不能加载
"""
import io
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import _bootstrap  # noqa: E402
_bootstrap.isolate_config_globally()

from PIL import Image  # noqa: E402

from utils.video_gen import CONTINUATION_FRAMES  # noqa: E402

_fails = []


def check(name):
    def deco(fn):
        try:
            ok = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok = False
            print(f"   EXC {type(e).__name__}: {e}")
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            _fails.append(name)
        return fn
    return deco


# ---------------- 假管线 ----------------
class _R:
    def __init__(self, frames):
        self.frames = [frames]


class _FakeT2V:
    def __init__(self):
        self.calls = []
        self.loaded = None

    def __call__(self, **kw):
        self.calls.append(kw)
        n = int(kw.get("num_frames") or 1)
        col = (len(self.calls) * 40 % 256, 5, 5)
        return _R([Image.new("RGB", (8, 8), col) for _ in range(n)])

    def load_lora_weights(self, path, adapter_name=None):
        self.loaded = (path, adapter_name)

    def set_adapters(self, names, adapter_weights=None):
        self.adapters = (names, adapter_weights)


class _FakeV2V(_FakeT2V):
    def __call__(self, **kw):
        self.calls.append(kw)
        n = len(kw.get("video") or [])
        return _R([Image.new("RGB", (8, 8), (0, 0, 90)) for _ in range(n)])


def _gen():
    from utils.video_gen import VideoGenerator
    g = VideoGenerator.__new__(VideoGenerator)
    g.ai = None
    g.pipe = _FakeT2V()
    g.v2v_pipe = _FakeV2V()
    g.motion_adapters = []
    return g


_COMMON = {"num_inference_steps": 4, "guidance_scale": 7.0,
           "width": 8, "height": 8, "generator": None}


# ---------------- 1. Lightning ----------------
@check("lightning_defaults_parse_4_and_8_step")
def t_lightning_defaults():
    from utils.video_gen import VideoGenerator
    d = VideoGenerator.lightning_defaults
    assert d("animatediff_lightning_4step_diffusers.safetensors") == (4, 1.0)
    assert d("animatediff_lightning_8step_diffusers.safetensors") == (8, 1.5)
    assert d("AnimateDiff-Lightning-2-step.safetensors") == (2, 1.0)
    assert d("unknown.safetensors") == (4, 1.0)
    return True


@check("lightning_loads_and_registers_adapter")
def t_lightning_load():
    g = _gen()
    ok = g._apply_lightning(g.pipe, os.path.join(ROOT, "loras", "sd1.5", "setu.safetensors"),
                            log_hint=False)
    assert ok is True
    assert g.pipe.loaded[1] == "lightning"
    assert g.pipe.adapters[0] == ["lightning"]
    return True


@check("lightning_missing_file_says_where_to_put")
def t_lightning_missing():
    g = _gen()
    try:
        g._apply_lightning(g.pipe, "nope.safetensors")
    except FileNotFoundError as e:
        msg = str(e)
        assert "lightning" in msg and "models" in msg, msg
        return True
    return False


@check("lightning_overrides_steps_and_cfg")
def t_lightning_override():
    """generate() 里挂了 lightning 后，步数/CFG 必须被拉回蒸馏档位。"""
    src = io.open(os.path.join(ROOT, "utils", "video_gen.py"), encoding="utf-8").read()
    assert "num_steps, guidance = l_steps, l_cfg" in src
    assert "self._apply_lightning(self.v2v_pipe, lightning, log_hint=False)" in src
    return True


# ---------------- 2. 分段长视频 ----------------
@check("segmented_splits_and_continues_from_tail")
def t_segmented():
    g = _gen()
    frames = g._run_segmented("p", "n", 40, _COMMON, mode="txt2video",
                              segment_frames=16, continuation_strength=0.5)
    assert len(frames) == 40, len(frames)
    assert len(g.pipe.calls) == 1, len(g.pipe.calls)
    assert len(g.v2v_pipe.calls) == 2, len(g.v2v_pipe.calls)
    # 均分：40 帧 / 每段上限 16 → 3 段 14+13+13（不是 16+16+8 那种短尾巴）
    first_len = int(g.pipe.calls[0]["num_frames"])
    assert first_len == 14, first_len
    first_v2v = g.v2v_pipe.calls[0]
    # 续接输入 = 最近 CONTINUATION_FRAMES 帧历史 + 本段新帧；生成完把历史那段丢掉
    assert len(first_v2v["video"]) == CONTINUATION_FRAMES + 13, len(first_v2v["video"])
    assert abs(first_v2v["strength"] - 0.5) < 1e-9
    # 历史前缀的第 0 帧 = 第一段倒数第 CONTINUATION_FRAMES 帧（顺序不能反）
    assert first_v2v["video"][0].tobytes() == \
        frames[first_len - CONTINUATION_FRAMES].tobytes()
    return True


@check("segmented_builds_v2v_pipe_for_txt2video")
def t_segmented_needs_v2v():
    """txt2video + 分段时也要建 v2v 管线，否则尾帧续接没有管线可用。"""
    src = io.open(os.path.join(ROOT, "utils", "video_gen.py"), encoding="utf-8").read()
    assert 'need_v2v = mode in ("vid2vid", "img2video") or uses_segments' in src
    # 超过单次上限时必须走分段（不再要求用户手动勾选才能触发）
    assert "SINGLE_PASS_MAX_FRAMES" in src
    assert "自动改成分段生成" in src
    return True


@check("frames_over_16_auto_segments_with_long_mode")
def t_auto_segment():
    """>16 帧 + 勾了长视频模式 → 直接分段，不能还交给单次生成（实测会崩）。"""
    g = _gen()
    g._build_pipe = lambda **kw: None
    g._load_motion_loras = lambda cfgs: None
    g._drop_ip_adapter = lambda: None
    g._save = lambda *a, **k: ("out.mp4", 1)
    g.generate("p", num_frames=20, num_steps=1, width=8, height=8,
               output_format="MP4", output_dir="output/_tmp_vidadv",
               use_context_window=True, seed=3)
    # 20 帧 / 每段最多 16 → 均分 10+10：首段走 txt2video，第二段走 v2v 续接
    first = int(g.pipe.calls[0]["num_frames"])
    assert first == 10, f"首段应为 10 帧，实际 {first}"
    sizes = [len(c["video"]) for c in g.v2v_pipe.calls]
    assert sizes == [CONTINUATION_FRAMES + 10], \
        f"续接输入应是 4+10 帧，实际 {sizes}"
    return True


@check("frames_over_16_segments_even_without_long_mode")
def t_auto_segment_default():
    """>16 帧即使没勾长视频也要分段：单次生成必崩，不能为了"尊重输入"就出噪声。"""
    g = _gen()
    g._build_pipe = lambda **kw: None
    g._load_motion_loras = lambda cfgs: None
    g._drop_ip_adapter = lambda: None
    g._save = lambda *a, **k: ("out.mp4", 1)
    g.generate("p", num_frames=20, num_steps=1, width=8, height=8,
               output_format="MP4", output_dir="output/_tmp_vidadv",
               use_context_window=False, seed=3)
    assert int(g.pipe.calls[0]["num_frames"]) == 10, \
        f"20 帧应均分成 10+10，首段实际 {g.pipe.calls[0]['num_frames']}"
    assert [len(c["video"]) for c in g.v2v_pipe.calls] == \
        [CONTINUATION_FRAMES + 10], \
        f"第二段应续接 4+10 帧：{[len(c['video']) for c in g.v2v_pipe.calls]}"
    return True


@check("frames_above_default_max_are_clamped")
def t_clamp_over_default_max():
    """不勾长视频又要 >32 帧 → 夹到 32 并提示（再长必须显式勾选）。"""
    from utils.video_gen import DEFAULT_MAX_FRAMES, SINGLE_PASS_MAX_FRAMES

    assert DEFAULT_MAX_FRAMES == 32
    assert DEFAULT_MAX_FRAMES > SINGLE_PASS_MAX_FRAMES
    g = _gen()
    g._build_pipe = lambda **kw: None
    g._load_motion_loras = lambda cfgs: None
    g._drop_ip_adapter = lambda: None
    g._save = lambda *a, **k: ("out.mp4", 1)
    g.generate("p", num_frames=80, num_steps=1, width=8, height=8,
               output_format="MP4", output_dir="output/_tmp_vidadv",
               use_context_window=False, seed=3)
    got = sum(int(c["num_frames"]) for c in g.pipe.calls) + \
        sum(len(c["video"]) for c in g.v2v_pipe.calls) - \
        CONTINUATION_FRAMES * len(g.v2v_pipe.calls)
    assert got == 32, f"应夹到 32 帧（扣掉丢弃的历史前缀），实际 {got}"
    return True


@check("segment_controls_are_wired_in_ui")
def t_segmented_ui():
    page = io.open(os.path.join(ROOT, "ui", "pages", "video_page.py"), encoding="utf-8").read()
    panel = io.open(os.path.join(ROOT, "ui", "video_panel_mixin.py"), encoding="utf-8").read()
    assert "spin_segment_frames" in page and "spin_continuation" in page
    assert "segment_frames=_seg_frames" in panel
    assert "continuation_strength=_cont" in panel
    return True


@check("nan_guard_is_installed_and_clears_nan")
def t_nan_guard():
    """每步都要挂 NaN 守卫：fp16 溢出会产生纯黑帧（实测 10 帧黑 4 帧）。"""
    import torch
    from utils.video_gen import _make_step_guard

    calls = []

    def inner(pipe, i, t, kw):
        calls.append(i)
        return kw

    cb = _make_step_guard(inner)
    # 有 NaN → 必须就地清零
    lat = torch.tensor([[float("nan"), 1.0, float("inf")]])
    out = cb(None, 3, 0, {"latents": lat})
    assert torch.isfinite(out["latents"]).all(), "NaN 没被清掉"
    # 干净的 latents 不能被改
    lat2 = torch.ones(2, 2)
    out2 = cb(None, 4, 0, {"latents": lat2})
    assert torch.equal(out2["latents"], lat2), "干净 latents 被改了"
    assert calls == [3, 4], f"进度回调没被链上: {calls}"
    # 进度回调炸了也不能影响生成
    def boom(*a, **k):
        raise RuntimeError("ui died")
    out3 = _make_step_guard(boom)(None, 5, 0, {"latents": lat2})
    assert "latents" in out3
    return True


@check("black_frames_are_repaired")
def t_black_repair():
    """万一仍漏出纯黑帧，要用相邻好帧补上，并如实告警。"""
    from PIL import Image
    from utils.video_gen import VideoGenerator

    good = Image.new("RGB", (8, 8), (120, 120, 120))
    black = Image.new("RGB", (8, 8), (0, 0, 0))
    out = VideoGenerator._repair_black_frames([good, black, black, good, black])
    means = [float(__import__("numpy").asarray(f).mean()) for f in out]
    assert all(m > 1.0 for m in means), means
    # 全是黑帧时不能崩，也不能假装修好了
    black_only = VideoGenerator._repair_black_frames([black, black])
    assert len(black_only) == 2
    # 空输入
    assert VideoGenerator._repair_black_frames([]) == []
    return True


@check("generate_wires_guard_and_repair")
def t_wire_guard():
    src = io.open(os.path.join(ROOT, "utils", "video_gen.py"), encoding="utf-8").read()
    assert "common[\"callback_on_step_end\"] = _make_step_guard(progress_callback)" in src
    assert "frames = self._repair_black_frames(frames)" in src
    return True


@check("continuation_uses_history_prefix_then_drops_it")
def t_continuation_prefix():
    """续接把最近 k 帧当真实时间前缀喂进去，生成 k+n 帧后只留新帧。

    真机对比（384²/12 步/32 帧）：只放一张尾帧 → 段内帧差 3.0（几乎静止）；
    直接用 4 帧循环铺满 → 段内 20.6 但接缝 30（每段像换镜头）；
    前缀 + 丢弃 → 段内与接缝都在正常量级。
    """
    g = _gen()
    g._build_pipe = lambda **kw: None
    g._load_motion_loras = lambda cfgs: None
    g._drop_ip_adapter = lambda: None
    g._save = lambda *a, **k: ("out.mp4", 1)
    g.generate("p", num_frames=32, num_steps=1, width=8, height=8,
               output_format="MP4", output_dir="output/_tmp_vidadv",
               use_context_window=True, segment_frames=16, seed=5,
               continuation_frames=4)
    call = g.v2v_pipe.calls[0]
    assert len(call["video"]) == 4 + 16, \
        f"输入应是 4 帧历史 + 16 帧新帧，实际 {len(call['video'])}"
    total = sum(len(c["video"]) for c in g.v2v_pipe.calls) + \
        sum(int(c["num_frames"]) for c in g.pipe.calls)
    assert total == 32 + 4, f"生成总量应为 32+4（多出的是被丢掉的前缀），实际 {total}"
    return True


@check("continuation_frames_one_keeps_static_tail")
def t_continuation_static():
    """continuation_frames=1 时退回老行为：一张尾帧铺满整段（旧的兼容路径）。"""
    g = _gen()
    g._build_pipe = lambda **kw: None
    g._load_motion_loras = lambda cfgs: None
    g._drop_ip_adapter = lambda: None
    g._save = lambda *a, **k: ("out.mp4", 1)
    g.generate("p", num_frames=32, num_steps=1, width=8, height=8,
               output_format="MP4", output_dir="output/_tmp_vidadv",
               use_context_window=True, segment_frames=16, seed=5,
               continuation_frames=1)
    assert len(g.v2v_pipe.calls[0]["video"]) == 16
    return True


@check("segment_color_is_matched_to_previous_tail")
def t_color_match():
    """段间颜色对齐：整体偏色的段要被拉回来，偏差很小时不能乱改。"""
    from PIL import Image
    from utils.video_gen import VideoGenerator

    ref = Image.new("RGB", (16, 16), (128, 128, 128))
    dark = [Image.new("RGB", (16, 16), (40, 40, 40))] * 3
    out = VideoGenerator._match_segment_color(dark, ref)
    m = float(__import__("numpy").asarray(out[0]).mean())
    assert m > 45.0, f"偏暗的段没被拉亮（max_shift 内应至少 +18），实际 {m:.1f}"
    # 已经对齐的不要动
    same = [Image.new("RGB", (16, 16), (128, 128, 128))]
    out2 = VideoGenerator._match_segment_color(same, ref)
    assert float(__import__("numpy").asarray(out2[0]).mean()) == 128.0
    return True


@check("long_video_defaults_and_segment_progress_are_wired")
def t_long_defaults():
    from utils.video_gen import CONTINUATION_FRAMES, CONTINUATION_STRENGTH

    assert CONTINUATION_FRAMES == 4
    assert abs(CONTINUATION_STRENGTH - 0.70) < 1e-6
    panel = io.open(os.path.join(ROOT, "ui", "video_panel_mixin.py"),
                    encoding="utf-8").read()
    assert "def segment_callback(idx, total, n)" in panel
    assert "segment_callback=segment_callback" in panel
    assert "第 {idx + 1}/{total} 段" in panel, "进度文案没带段号"
    page = io.open(os.path.join(ROOT, "ui", "pages", "video_page.py"),
                   encoding="utf-8").read()
    assert "_dspin(0.30, 0.80, 0.70, 0.05, 2)" in page
    return True


# ---------------- 3. 中文路径 ----------------
@check("video_io_reads_non_ascii_path")
def t_non_ascii():
    import cv2
    import numpy as np
    from utils.video_io import probe_video, read_all_frames, needs_ascii_workaround

    d = os.path.join(ROOT, "output", "_tmp_videoio", "中文 目录")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "测试视频.avi")
    try:
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 8, (64, 64))
        for i in range(8):
            w.write(np.full((64, 64, 3), i * 20 % 255, dtype=np.uint8))
        w.release()

        assert needs_ascii_workaround(p) is True
        ok, fps, cnt, ww, hh = probe_video(p)
        assert ok and cnt == 8 and (ww, hh) == (64, 64), (ok, cnt, ww, hh)
        frames = read_all_frames(p)
        assert len(frames) == 8, len(frames)
        return True
    finally:
        shutil.rmtree(os.path.join(ROOT, "output", "_tmp_videoio"), ignore_errors=True)


@check("rife_searched_in_models_dir")
def t_rife_paths():
    from utils.rife_interpolate import RIFEInterpolator, RIFE_SEARCH_DIRS, find_rife_dirs
    assert "models/rife" in RIFE_SEARCH_DIRS
    r = RIFEInterpolator()
    if not r.is_available():
        txt = r.status_text()
        assert "models/rife" in txt, txt
    assert isinstance(find_rife_dirs(), list)
    return True


# ---------------- 4. LTX-Video / Wan2.1 入口 ----------------
@check("modern_video_arches_are_entry_only")
def t_arches():
    from core.arch import get_arch
    for a in ("ltxvideo", "wan21"):
        info = get_arch(a)
        assert info.supported is False, a
        assert "未接通" in (info.unsupported_reason or ""), a
        assert info.model_subdir == a
    return True


@check("modern_video_dir_is_detected")
def t_detect():
    from core.arch import detect
    made = []
    try:
        for name, cls, pipe_file, want in (
            ("_tmp_ltx_t", "LTXPipeline", None, "ltxvideo"),
            ("_tmp_wan_t", "WhateverWan", "pipeline_wan.py", "wan21"),
        ):
            d = os.path.join(ROOT, "models", "puton", name)
            os.makedirs(d, exist_ok=True)
            made.append(d)
            with io.open(os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
                json.dump({"_class_name": cls}, f)
            if pipe_file:
                with io.open(os.path.join(d, pipe_file), "w", encoding="utf-8") as f:
                    f.write("# fake\n")
            got = detect(d)
            assert got.arch_id == want, (name, got.arch_id)
        return True
    finally:
        for d in made:
            shutil.rmtree(d, ignore_errors=True)


@check("modern_video_market_entries_do_not_download")
def t_market():
    from utils import extension_manager as em
    for eid in ("ltxvideo_entry", "wan21_entry"):
        assert eid in em.EXTENSIONS, eid
        assert eid in em.DOWNLOADERS, eid
        assert em.EXTENSIONS[eid]["category"] == "新架构引擎"
    try:
        em.download_extension("ltxvideo_entry")
    except RuntimeError as e:
        msg = str(e)
        assert "自己下载" in msg and "models/ltxvideo" in msg, msg
        return True
    return False


@check("load_model_explains_entry_only")
def t_load_message():
    from core.model_manager import ModelManager
    d = os.path.join(ROOT, "models", "puton", "_tmp_wan_load")
    os.makedirs(d, exist_ok=True)
    try:
        with io.open(os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
            json.dump({"_class_name": "WanPipeline"}, f)
        try:
            ModelManager().load_model(d)
        except Exception as e:
            msg = str(e)
            assert "未接通" in msg, msg[:200]
            assert "models/wan21" in msg, msg[:300]
            return True
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


print()
if _fails:
    print("FAILED:", _fails)
    sys.exit(1)
print("VIDEO_ADV_OK")
