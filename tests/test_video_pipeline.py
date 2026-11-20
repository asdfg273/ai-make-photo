# -*- coding: utf-8 -*-
"""视频链路的回归测试（对应 2026-11-16 的视频审计修复）。

覆盖五件事：
1. 配音不会截短视频（旧实现用 -shortest，6s 视频 + 2s 语音只剩 2.04s）
2. "MP4 + GIF" 真的产出两个文件（旧实现只出 GIF）
3. 提示词超 77 token 会告警（旧实现静默截断 219 个词）
4. img2video 走 v2v 首帧约束，不再依赖 IP-Adapter
5. 输出目录跟随调用方、Motion LoRA 用绝对路径

跑法: venv\\Scripts\\python.exe tests\\test_video_pipeline.py
"""
import os
import re
import shutil
import subprocess
import sys
import types
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PASS, _FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            _PASS.append(name)
            print(f"PASS {name}", flush=True)
        except Exception as e:
            _FAIL.append((name, e))
            print(f"FAIL {name}: {e}", flush=True)
        return fn
    return deco


def _ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _probe_duration(path):
    r = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path],
                       capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    assert m, f"探测不到时长: {r.stderr[-300:]}"
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def _make_video(path, seconds=6, fps=8, size=(128, 128)):
    import imageio
    import numpy as np
    n = int(seconds * fps)
    frames = []
    for i in range(n):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[:, :, 0] = (i * 7) % 255
        arr[:, :, 1] = 40
        frames.append(arr)
    imageio.mimsave(path, frames, fps=fps, codec="libx264",
                    macro_block_size=1)
    return path


def _make_wav(path, seconds=2.0, rate=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


class _FakePipe:
    """记录调用参数的假扩散管线。"""

    def __init__(self, tag):
        self.tag = tag
        self.calls = []
        self.tokenizer = None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        from PIL import Image
        n = int(kwargs.get("num_frames") or len(kwargs.get("video") or []))
        frames = [Image.new("RGB", (64, 64), (10 * i % 255, 60, 90)) for i in range(n)]
        return types.SimpleNamespace(frames=[frames])


def _fake_generator():
    from utils.video_gen import VideoGenerator
    vg = VideoGenerator.__new__(VideoGenerator)   # 不跑 __init__（不加载模型）
    vg.pipe = _FakePipe("t2v")
    vg.v2v_pipe = _FakePipe("v2v")
    vg.v2v_pipe.tokenizer = None
    vg.loaded_motion_loras = []
    vg._build_pipe = lambda **kw: None
    vg._load_motion_loras = lambda cfgs: None
    vg._drop_ip_adapter = lambda: None
    vg._warn_prompt_length = types.MethodType(
        VideoGenerator._warn_prompt_length, vg)
    vg._save = types.MethodType(VideoGenerator._save, vg)
    return vg


TMP = os.path.join(ROOT, "output", "_tmp_vidtest")
if os.path.isdir(TMP):
    shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP, exist_ok=True)


# --------------------------------------------------------------- 1. 配音不截短
@check("voice_mux_keeps_full_video")
def test_voice_mux_keeps_full_video():
    from ui.video_panel_mixin import VideoPanelMixin

    class _TTS:
        def __init__(self, wav):
            self.wav = wav

        def generate_chattts(self, text, speaker_seed=2222):
            return self.wav

    vid = _make_video(os.path.join(TMP, "src6s.mp4"), seconds=6)
    wav = _make_wav(os.path.join(TMP, "voice2s.wav"), seconds=2)

    stub = VideoPanelMixin()
    stub.tts_engine = _TTS(wav)
    out = stub._add_voice_to_video(vid, {"text": "hello there", "engine": "chattts"})

    assert out != vid, "配音没有生成新文件"
    dur = _probe_duration(out)
    assert dur >= 5.5, f"视频被截短了：{dur:.2f}s（应该 ≈6s）"


# --------------------------------------------------------------- 2. MP4 + GIF
@check("mp4_and_gif_both_saved")
def test_mp4_and_gif_both_saved():
    vg = _fake_generator()
    out, _seed = vg.generate(
        prompt="cat", num_frames=8, num_steps=1,
        width=64, height=64, output_format="MP4 + GIF",
        output_dir=TMP, mode="txt2video", seed=5,
    )
    stem = os.path.splitext(out)[0]
    assert out.endswith(".mp4"), f"返回值应该是 mp4: {out}"
    assert os.path.exists(f"{stem}.mp4"), "MP4 没生成"
    assert os.path.exists(f"{stem}.gif"), "GIF 没生成"


# --------------------------------------------------------------- 3. 长提示词告警
@check("long_prompt_warns")
def test_long_prompt_warns():
    # 注意：utils/video_gen.py 用的是 logging.getLogger(__name__)，
    # 挂到 utils.system_utils.logger 上抓不到（实测 records 为空）。
    import logging
    logger = logging.getLogger("utils.video_gen")

    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _H()
    logger.addHandler(h)
    try:
        vg = _fake_generator()

        class _Tok:
            def __call__(self, text, add_special_tokens=True):
                n = len(text.split()) + 2
                return {"input_ids": [0] * n}

        vg.pipe.tokenizer = _Tok()
        vg.generate(
            prompt=" ".join(f"word{i}" for i in range(120)),
            num_frames=8, num_steps=1, width=64, height=64,
            output_format="MP4", output_dir=TMP, mode="txt2video", seed=6,
        )
    finally:
        logger.removeHandler(h)

    joined = "\n".join(records)
    assert "77" in joined and ("超过" in joined or "token" in joined), \
        f"长提示词没有告警: {joined[-400:]}"


# --------------------------------------------------------------- 4. i2v 走 v2v
@check("img2video_uses_v2v_first_frame")
def test_img2video_uses_v2v_first_frame():
    from PIL import Image
    img = os.path.join(TMP, "first.png")
    Image.new("RGB", (64, 64), (255, 0, 0)).save(img)

    vg = _fake_generator()
    vg.generate(prompt="move", num_frames=6, num_steps=1,
                width=64, height=64, output_format="MP4",
                output_dir=TMP, mode="img2video",
                input_path=img, strength=0.85, seed=7)

    assert not vg.pipe.calls, "img2video 不该再调用 txt2video 管线"
    assert vg.v2v_pipe.calls, "img2video 应该走 v2v 管线"
    kw = vg.v2v_pipe.calls[0]
    video = kw["video"]
    assert len(video) == 6, f"首帧序列长度应为 6，实际 {len(video)}"
    assert abs(float(kw["strength"]) - 0.85) < 1e-6, "strength 没传进 v2v"
    assert all(v is video[0] for v in video), "首帧应该被重复成整段序列"
    assert "ip_adapter_image" not in kw, "不该再传 IP-Adapter 图像"


# --------------------------------------------------------------- 5. 源码级接线
@check("panel_uses_configured_output_dir")
def test_panel_uses_configured_output_dir():
    src = open(os.path.join(ROOT, "ui", "video_panel_mixin.py"), encoding="utf-8").read()
    assert 'output_dir="photo/videos"' not in src, "输出目录仍然写死"
    assert "host_output_dir(self)" in src, "输出目录没有跟随设置"
    assert "preflight_ip_adapter" not in src, "i2v 仍在预检 IP-Adapter"


@check("motion_lora_scan_is_absolute")
def test_motion_lora_scan_is_absolute():
    src = open(os.path.join(ROOT, "ui", "video_panel_mixin.py"), encoding="utf-8").read()
    assert '"models/motion_lora"' not in src, "Motion LoRA 扫描仍是相对路径"
    assert 'os.path.join(MODEL_DIR, "motion_lora")' in src, "应该用 MODEL_DIR 拼绝对路径"


@check("no_shortest_in_voice_mux")
def test_no_shortest_in_voice_mux():
    src = open(os.path.join(ROOT, "ui", "video_panel_mixin.py"), encoding="utf-8").read()
    assert "apad=whole_dur" in src, "缺少补静音处理"
    assert "duration=longest" in src, "BGM 混音仍按最短截断"
    # -shortest 只允许出现在"探测不到时长"的兜底分支里；注释里提到不算
    code_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    hits = [ln for ln in code_lines if "-shortest" in ln]
    assert len(hits) <= 1, f"-shortest 出现在 {len(hits)} 行代码里: {hits}"


@check("video_build_restores_lora")
def test_video_build_restores_lora():
    """生成视频不该把用户挂在图片管线上的 LoRA 弄丢。"""
    mm_src = open(os.path.join(ROOT, "core", "model_manager.py"), encoding="utf-8").read()
    vg_src = open(os.path.join(ROOT, "utils", "video_gen.py"), encoding="utf-8").read()

    assert "def active_loras(self)" in mm_src, "ModelManager 缺 active_loras()"
    assert "_active_lora_spec" in mm_src, "ModelManager 没记录已挂载的 LoRA"
    assert "mounted_specs.append" in mm_src, "应该只记真正挂上的 LoRA"

    assert "_saved_loras" in vg_src, "视频链路没备份 LoRA"
    assert "apply_multiple_loras(_saved_loras)" in vg_src, "建完管线后没挂回去"

    # 顺序：备份必须在卸载之前
    i_save = vg_src.index("_saved_loras = self.ai.active_loras()")
    i_unload = vg_src.index("base.unload_lora_weights()")
    assert i_save < i_unload, "必须先备份再卸载"


print()
print(f"共 {len(_PASS) + len(_FAIL)} 项：{len(_PASS)} 通过，{len(_FAIL)} 失败")
if _FAIL:
    for n, e in _FAIL:
        print(f"  FAILED {n}: {e}")
    sys.exit(1)
print("VIDEO_PIPELINE_OK")
