# tests/test_anima_base_resolution.py — Anima 基座组件定位（无头，不联网、不跑 GPU）
#
# 为什么要有这个文件
# ------------------
# 用户实测：本地有一个 3.90GB 的 Anima 检查点（685 keys，架构检测正常），
# 但点"生成"直接失败，日志是：
#
#   core/loaders/anima.py:128 pipe = ModularPipeline.from_pretrained(self.BASE_REPO)
#   → HEAD https://hf-mirror.com/circlestone-labs/Anima-Base-v1.0-Diffusers/...
#   → '[Errno 11001] getaddrinfo failed'  →  Retrying [1/5] …
#   → RuntimeError: Cannot send a request, as the client has been closed.
#   → Exception: 模型加载失败: Cannot send a request, as the client has been closed.
#
# 三个问题：
#   1. Anima 检查点里**只有 transformer + llm_adapter**，文本编码器（Qwen3-0.6B）
#      和 Qwen VAE 在基座仓里 —— 所以必须能找到基座，这是设计上的依赖，不是 bug；
#   2. 但旧写法拿 **repo id** 交给 diffusers，即使本地已有缓存/已手动放好基座，
#      也照样发网络请求 → 离线必挂；
#   3. 失败信息是 httpx 的 "client has been closed"，用户完全看不出缺什么、
#      该放哪里。
#
# 所以这里锁死四件事：本地目录优先、缓存次之、都没有时给可操作的中文报错、
# **任何情况下都不会在没有基座时发网络请求**。
#
# 用法：venv\Scripts\python.exe tests\test_anima_base_resolution.py
import json
import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码，放最前）

_bootstrap.isolate_config_globally()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX = os.path.join(ROOT, "tests", "artifacts", "tmp", "anima_base")


def _reset_sandbox():
    """建一个干净的假 models/ 根目录，返回 (root, models_dir)。"""
    import shutil

    if os.path.isdir(SANDBOX):
        shutil.rmtree(SANDBOX, ignore_errors=True)
    models = os.path.join(SANDBOX, "models")
    os.makedirs(models, exist_ok=True)
    return SANDBOX, models


def _make_base_dir(parent, name, marker="modular_model_index.json",
                   cls="AnimaModularPipeline", complete=True):
    """造一个"看起来像基座"的目录。

    complete=False 时保留入口文件但把组件权重写成 0 字节 —— 用来模拟
    HF 缓存里"下到一半"的快照（真实存在：入口文件有、blob 是空的）。
    """
    path = os.path.join(parent, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, marker), "w", encoding="utf-8") as f:
        json.dump({"_class_name": cls}, f)
    for part in ("transformer", "text_encoder", "vae"):
        d = os.path.join(path, part)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "diffusion_pytorch_model.safetensors"),
                  "wb") as f:
            f.write(b"\0" * (64 if complete else 0))
    return path


class _PatchedPaths:
    """把 utils.paths.MODEL_DIR 指到沙箱里（进出都恢复）。"""

    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.old = None

    def __enter__(self):
        from utils import paths

        self.old = paths.MODEL_DIR
        paths.MODEL_DIR = self.models_dir
        return self.models_dir

    def __exit__(self, *exc):
        from utils import paths

        paths.MODEL_DIR = self.old
        return False


def _loader():
    from core.loaders.anima import AnimaLoader

    return AnimaLoader()


def _no_network_snapshot(recorded):
    """替身 snapshot_download：记录调用参数，然后按"缓存未命中"失败。"""

    def fake(repo_id, **kwargs):
        recorded.append((repo_id, dict(kwargs)))
        raise OSError("not cached (test double)")

    return fake


# ----------------------------------------------------------------------
#  1) models/anima/<任意目录名>/modular_config.json 能被找到
# ----------------------------------------------------------------------
def test_local_base_dir_in_models_anima():
    root, models = _reset_sandbox()
    base = _make_base_dir(os.path.join(models, "anima"), "MyAnimaBase")
    with _PatchedPaths(models):
        path, how = _loader().resolve_base()
    assert os.path.normcase(path) == os.path.normcase(base), \
        f"没有选中本地基座目录: {path}"
    assert how == "本地目录", f"来源说明不对: {how}"


# ----------------------------------------------------------------------
#  2) 投递箱 models/puton/<基座目录> 也认（用户按"丢进来就行"的习惯）
# ----------------------------------------------------------------------
def test_local_base_dir_in_puton():
    root, models = _reset_sandbox()
    _make_base_dir(os.path.join(models, "puton"), "Anima-Base-v1.0-Diffusers")
    with _PatchedPaths(models):
        path, how = _loader().resolve_base()
    assert how == "本地目录" and "puton" in path.lower(), \
        f"puton 里的基座没被认出来: {path}（{how}）"


# ----------------------------------------------------------------------
#  3) 目录名不限：改名后只要里面还有 modular_config.json 就行
# ----------------------------------------------------------------------
def test_base_dir_name_is_irrelevant():
    root, models = _reset_sandbox()
    _make_base_dir(os.path.join(models, "anima"), "随便起个名字")
    with _PatchedPaths(models):
        path, _how = _loader().resolve_base()
    assert os.path.basename(path) == "随便起个名字", f"选中了别的东西: {path}"


# ----------------------------------------------------------------------
#  4) 没有本地目录时用 HF 缓存快照，且必须是 local_files_only（不联网）
# ----------------------------------------------------------------------
def test_cached_snapshot_is_used_offline():
    import huggingface_hub

    root, models = _reset_sandbox()
    cached = _make_base_dir(os.path.join(root, "hf_cache"), "snapshots-abc")
    recorded = []
    original = huggingface_hub.snapshot_download

    def fake_download(repo_id, **kwargs):
        recorded.append((repo_id, dict(kwargs)))
        return cached

    huggingface_hub.snapshot_download = fake_download
    try:
        with _PatchedPaths(models):
            path, how = _loader().resolve_base()
    finally:
        huggingface_hub.snapshot_download = original

    assert os.path.normcase(path) == os.path.normcase(cached), \
        f"没用缓存快照: {path}"
    assert how == "HF 缓存", f"来源说明不对: {how}"
    assert recorded and recorded[0][1].get("local_files_only") is True, \
        f"探测缓存时没有传 local_files_only=True: {recorded}"


# ----------------------------------------------------------------------
#  5) 什么都没有 → 明确中文报错（不是 httpx 的 client has been closed）
# ----------------------------------------------------------------------
def test_missing_base_raises_actionable_message():
    import huggingface_hub

    root, models = _reset_sandbox()
    recorded = []
    original = huggingface_hub.snapshot_download
    huggingface_hub.snapshot_download = _no_network_snapshot(recorded)
    try:
        with _PatchedPaths(models):
            try:
                _loader().resolve_base()
            except RuntimeError as e:
                msg = str(e)
            else:
                raise AssertionError("缺基座时居然没报错")
    finally:
        huggingface_hub.snapshot_download = original

    assert "基座组件" in msg, f"报错没点明是缺基座: {msg}"
    assert "modular_model_index.json" in msg, f"没告诉用户放什么文件: {msg}"
    assert "puton" in msg, f"没告诉用户放哪里: {msg}"
    assert "1.7GB" in msg, f"没说明体积: {msg}"
    assert "client has been closed" not in msg, \
        f"又把 httpx 的内部报错透给用户了: {msg}"
    assert "Anima-Base-v1.0-Diffusers" in msg, f"没给出需要的仓名: {msg}"


# ----------------------------------------------------------------------
#  6) 缺基座时绝不允许"先联网试一下"
# ----------------------------------------------------------------------
def test_missing_base_never_touches_network():
    import huggingface_hub

    root, models = _reset_sandbox()
    recorded = []
    original = huggingface_hub.snapshot_download

    def fake(repo_id, **kwargs):
        recorded.append((repo_id, dict(kwargs)))
        # 只要有人去掉 local_files_only，这里就让它假装成功 —— 用来抓"偷偷联网"
        if not kwargs.get("local_files_only"):
            raise AssertionError(
                f"resolve_base 在缺基座时发起了网络请求: {repo_id} {kwargs}")
        raise OSError("not cached")

    huggingface_hub.snapshot_download = fake
    try:
        with _PatchedPaths(models):
            try:
                _loader().resolve_base()
            except RuntimeError:
                pass
    finally:
        huggingface_hub.snapshot_download = original

    assert recorded, "根本没探测缓存（应该探测一次，且只查本地）"
    assert all(kw.get("local_files_only") is True for _r, kw in recorded), \
        f"探测时允许联网了: {recorded}"


# ----------------------------------------------------------------------
#  7) 缓存里"下到一半"的快照要被认出来（入口文件在、组件 0 字节）
# ----------------------------------------------------------------------
def test_incomplete_cache_reports_zero_byte_components():
    import huggingface_hub

    root, models = _reset_sandbox()
    stub = _make_base_dir(os.path.join(root, "hf_cache"), "snapshots-stub",
                          complete=False)
    original = huggingface_hub.snapshot_download
    huggingface_hub.snapshot_download = lambda repo_id, **kw: stub
    try:
        with _PatchedPaths(models):
            try:
                _loader().resolve_base()
            except RuntimeError as e:
                msg = str(e)
            else:
                raise AssertionError("0 字节的基座居然被当成可用")
    finally:
        huggingface_hub.snapshot_download = original

    assert "0 字节" in msg, f"没指出是下载不完整: {msg}"
    assert "transformer" in msg, f"没指出哪个组件是空的: {msg}"


# ----------------------------------------------------------------------
#  8) 目录型基座能被架构检测认成 anima（否则投递箱不会把它归位）
# ----------------------------------------------------------------------
def test_detect_dir_recognizes_modular_config():
    from core import arch as arch_mod

    root, models = _reset_sandbox()
    base = _make_base_dir(os.path.join(models, "puton"), "BaseByModularConfig")
    det = arch_mod.detect(base)
    assert det.arch_id == "anima", \
        f"含 modular_config.json 的目录被判成 {det.arch_id}（{det.evidence}）"

    classic = _make_base_dir(os.path.join(models, "puton"), "Classic",
                             marker="model_index.json",
                             cls="SomeUnknownPipeline")
    det2 = arch_mod.detect(classic)
    assert det2.arch_id == "unknown", \
        "model_index.json 里没有 _class_name 时不该瞎猜架构"


# ----------------------------------------------------------------------
#  9) 投递箱会把基座目录归位到 models/anima/（用户"丢进来就行"）
# ----------------------------------------------------------------------
def test_inbox_moves_base_dir_into_models_anima():
    from utils import paths
    from utils import model_inbox

    root, models = _reset_sandbox()
    puton = os.path.join(models, "puton")
    _make_base_dir(puton, "Anima-Base-v1.0-Diffusers")
    old_lora = paths.LORA_DIR
    paths.LORA_DIR = os.path.join(SANDBOX, "loras")
    try:
        with _PatchedPaths(models):
            rep = model_inbox.run(apply=True)
    finally:
        paths.LORA_DIR = old_lora

    assert rep.moved, f"基座目录没被归位（kept={[e.name for e in rep.kept]}）"
    target = os.path.join(models, "anima", "Anima-Base-v1.0-Diffusers")
    assert os.path.isdir(target), f"没移动到 {target}"
    assert os.path.isfile(os.path.join(target, "modular_model_index.json")), \
        "移动后 modular_model_index.json 丢了"


# ----------------------------------------------------------------------
#  10) 不支持 offload 的管线（ModularPipeline）不能把选档逻辑搞崩
# ----------------------------------------------------------------------
def test_vram_strategy_survives_pipe_without_offload_hooks():
    """Anima 的 ModularPipeline 没有 enable_*_cpu_offload，旧代码直接调用会
    AttributeError（= 模型加载失败）。这里用一个同形状的替身锁死行为。"""
    import torch

    if not torch.cuda.is_available():          # 无卡机器跳过
        print("   (无 CUDA，跳过)")
        return

    from utils.vram_manager import VRAMManager

    class FakeModularPipe:
        def __init__(self):
            self.vae = None
            self.moved = None

        def to(self, where):
            self.moved = where
            return self

    pipe = FakeModularPipe()
    assert not hasattr(pipe, "enable_model_cpu_offload")
    strategy = VRAMManager.apply_optimal_strategy(
        pipe, is_sdxl=False, weights_gb=5.6, arch_id="anima")
    assert isinstance(strategy, str) and strategy, "没有返回策略说明"
    assert pipe.moved == "cuda:0", f"退回全 GPU 时没调用 to(): {pipe.moved}"
    assert "不支持 offload" in strategy or "全GPU" in strategy, \
        f"策略说明没体现该管线无法 offload: {strategy}"


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
