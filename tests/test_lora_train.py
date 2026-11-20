"""轻量 LoRA 训练的永久回归（core/lora_train.py）。

为什么需要它
------------
训练代码最容易"看起来跑完了但什么都没学到"，而且它有两条独立的链路：
1. **数据面**：扫描图片 / 读 `.txt` 标注 / 分桶 / 组 batch —— 任何一处
   静默跳过，用户只会看到"loss 在降但模型没效果"；
2. **训练面**：挂适配器、前向、反传、优化器、保存、写触发词旁文件。

真跑一遍 SD1.5（每步约 0.3s）在本机可以，但测试必须能在 CI/无卡环境跑，
所以这里用 `_prepared=` 注入一个**极小 UNet + 假缓存**，把训练面完整走一遍
（含 loss 有限、LoRA 参数确实被更新、文件确实落盘）。

跑法: venv\\Scripts\\python.exe tests\\test_lora_train.py
"""
import json
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bootstrap  # noqa: E402

_bootstrap.isolate_config_globally()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "tests", "artifacts", "tmp", "lora_train")
RESULT = []


def check(name):
    def deco(fn):
        try:
            info = fn()
            RESULT.append((name, True, info))
            print(f"[PASS] {name}" + (f" | {info}" if info else ""))
        except Exception as exc:  # noqa: BLE001 - 测试脚本自己聚合
            RESULT.append((name, False, str(exc)))
            print(f"[FAIL] {name} | {exc}")
            traceback.print_exc()
        return fn
    return deco


def _tmpdir():
    os.makedirs(ART, exist_ok=True)
    return ART


def _write_images(count=3):
    from PIL import Image

    out = os.path.join(_tmpdir(), f"ds_{count}")
    os.makedirs(out, exist_ok=True)
    sizes = [(512, 512), (500, 800), (800, 500)]
    for i in range(count):
        w, h = sizes[i % len(sizes)]
        Image.new("RGB", (w, h), (30 + i * 40, 60, 120)).save(os.path.join(out, f"img{i}.png"))
    return out


def _tiny_unet():
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 32),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=32,
    )


class _FakePipe:
    """只实现 `save_lora_weights`：把 LoRA 张量真的写进 safetensors。"""

    def __init__(self, unet):
        self.unet = unet
        self.saved_to = ""

    def save_lora_weights(self, save_directory, unet_lora_layers=None,
                          text_encoder_lora_layers=None, safe_serialization=True):
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        state = {}
        # 真实 API 收到的是 **state dict**（不是模块）；这里两种都容忍，
        # 但必须拒绝"整份 UNet 副本"（那会让 LoRA 文件变成 1.7GB 的假货）。
        items = (unet_lora_layers.items() if isinstance(unet_lora_layers, dict)
                 else unet_lora_layers.state_dict().items())
        for key, value in items:
            if "lora" in key.lower():
                state[key] = value.detach().to("cpu").contiguous()
        path = os.path.join(save_directory, "pytorch_lora_weights.safetensors")
        save_file(state, path)
        self.saved_to = path


def _prepared(steps_hint=32):
    import torch

    unet = _tiny_unet()
    dtype = next(unet.parameters()).dtype
    samples = []
    for i in range(3):
        bucket = (256 + 64 * (i % 2), 256)
        samples.append({
            "latent": torch.randn(1, 4, 32, 32, dtype=torch.float32).to(dtype),
            "embeds": torch.randn(1, 77, 32, dtype=torch.float32).to(dtype),
            "pooled": None,
            "bucket": bucket,
            "caption": "a test caption",
        })
    return {"unet": unet, "sdxl": False, "samples": samples}


@check("scan_dataset_reads_captions_and_warns")
def test_scan():
    from core.lora_train import scan_dataset

    folder = _write_images(3)
    # UTF-8 标注 + GBK 标注 + 无标注
    with open(os.path.join(folder, "img0.txt"), "w", encoding="utf-8") as fh:
        fh.write("a red cat, looking up")
    with open(os.path.join(folder, "img1.txt"), "w", encoding="gbk") as fh:
        fh.write("一只绿猫")
    records, warnings = scan_dataset(folder, trigger="mychar")
    assert len(records) == 3, f"应扫到 3 张，实际 {len(records)}"
    by_name = {os.path.basename(r["path"]): r["caption"] for r in records}
    assert by_name["img0.png"] == "mychar, a red cat, looking up", by_name["img0.png"]
    assert by_name["img1.png"] == "mychar, 一只绿猫", by_name["img1.png"]
    assert by_name["img2.png"] == "mychar", by_name["img2.png"]
    assert any("没有同名 .txt" in w for w in warnings), warnings
    return f"3 张 / warning {len(warnings)} 条"


@check("scan_dataset_reports_empty_dir")
def test_scan_empty():
    from core.lora_train import scan_dataset

    empty = os.path.join(_tmpdir(), "empty")
    os.makedirs(empty, exist_ok=True)
    records, warnings = scan_dataset(empty)
    assert records == []
    assert any("没有可用的图片" in w for w in warnings), warnings
    return "空目录有明确提示"


@check("bucket_size_is_multiple_of_64_and_keeps_aspect")
def test_bucket():
    from core.lora_train import bucket_size, SIZE_MULTIPLE

    for (w, h) in [(512, 512), (500, 800), (1920, 1080), (300, 900), (1024, 1024)]:
        bw, bh = bucket_size(w, h, 512)
        assert bw % SIZE_MULTIPLE == 0 and bh % SIZE_MULTIPLE == 0, (w, h, bw, bh)
        ratio_in = w / h
        ratio_out = bw / bh
        assert abs(math.log(ratio_out / ratio_in)) < 0.12, f"{w}x{h} → {bw}x{bh}"
        area = bw * bh
        assert 0.55 < area / (512 * 512) < 1.5, f"{w}x{h} → 面积 {area}"
    return "5 组比例都通过"


@check("batches_are_bucket_homogeneous")
def test_batches():
    import random

    from core.lora_train import build_batches

    items = []
    for i in range(7):
        items.append({"bucket": (512, 512), "i": i})
    for i in range(5):
        items.append({"bucket": (512, 768), "i": i + 100})
    rng = random.Random(0)
    batches = build_batches(items, 2, rng)
    total = sum(len(b) for b in batches)
    assert total == 12, total
    for batch in batches:
        buckets = {b["bucket"] for b in batch}
        assert len(buckets) == 1, buckets
        assert 1 <= len(batch) <= 2, len(batch)
    return f"{len(batches)} 个 batch / 12 张"


@check("lora_config_has_targets_and_alpha")
def test_config():
    from core.lora_train import build_lora_config, UNET_TARGETS, TrainConfig

    cfg = TrainConfig(rank=8)
    assert cfg.effective_alpha() == 8.0
    cfg2 = TrainConfig(rank=8, alpha=4.0)
    assert cfg2.effective_alpha() == 4.0
    lc = build_lora_config(8, 8.0, UNET_TARGETS)
    assert lc.r == 8 and lc.lora_alpha == 8.0, (lc.r, lc.lora_alpha)
    assert set(lc.target_modules) == set(UNET_TARGETS), lc.target_modules
    return f"r={lc.r} alpha={lc.lora_alpha}"


@check("vram_estimate_scales")
def test_estimate():
    from core.lora_train import estimate_vram_mb, TrainConfig

    a = estimate_vram_mb(TrainConfig(resolution=512), "sd15")
    b = estimate_vram_mb(TrainConfig(resolution=1024), "sd15")
    c = estimate_vram_mb(TrainConfig(resolution=512), "sdxl")
    d = estimate_vram_mb(TrainConfig(resolution=512, train_text_encoder=True), "sd15")
    assert b > a and c > a and d > a, (a, b, c, d)
    assert a > 1000, a
    return f"sd15@{512}={a}MB sdxl@{512}={c}MB +te={d}MB"


@check("micro_training_updates_params_and_saves")
def test_micro():
    from safetensors.torch import load_file

    from core.lora_train import TrainConfig, train_lora, _safe_name

    prep = _prepared()
    pipe = _FakePipe(prep["unet"])
    outdir = os.path.join(_tmpdir(), "loras")
    os.makedirs(outdir, exist_ok=True)
    import utils.paths as paths
    old = getattr(paths, "LORA_DIR", None)
    paths.LORA_DIR = outdir
    progress = []
    try:
        cfg = TrainConfig(dataset_dir="ignored", base_model="tiny", output_name="unit lora/v1",
                          rank=4, lr=1e-3, steps=6, batch_size=1, trigger="mychar",
                          seed=1234, resolution=32, warmup=2)
        res = train_lora(cfg, pipe=pipe, arch="sd15", device="cpu", _prepared=prep,
                         on_progress=lambda d, t, s="": progress.append((d, t)))
    finally:
        if old is not None:
            paths.LORA_DIR = old
    assert res.ok, res.error
    assert res.steps_done == 6, res.steps_done
    assert len(res.losses) == 6 and all(math.isfinite(x) for x in res.losses), res.losses
    assert progress and progress[-1][0] == progress[-1][1], progress[-1:]
    assert os.path.isfile(res.out_path), res.out_path
    assert res.out_path.endswith("unit_lora_v1.safetensors"), res.out_path
    assert os.path.isfile(res.sidecar), res.sidecar
    with open(res.sidecar, encoding="utf-8") as fh:
        assert fh.read() == "mychar"
    meta_path = os.path.splitext(res.out_path)[0] + ".train.json"
    assert os.path.isfile(meta_path)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["steps"] == 6 and meta["arch"] == "sd15" and meta["config"]["trigger"] == "mychar"
    tensors = load_file(res.out_path)
    assert tensors, "保存的 safetensors 是空的"
    assert all(float(v.float().abs().max()) >= 0 for v in tensors.values())
    assert any(float(v.float().abs().max()) > 0 for v in tensors.values()), "全零 LoRA"
    assert _safe_name("a/b:c*d") == "a_b_c_d", _safe_name("a/b:c*d")
    return f"6 步 loss {res.losses[0]:.3f}→{res.losses[-1]:.3f}，{len(tensors)} 个张量"


@check("training_actually_changes_the_weights")
def test_weights_change():
    """lr=0 与 lr=1e-2 两次训练（同种子 → 同初始化）产出的文件必须不同。"""
    import torch

    from core.lora_train import TrainConfig, train_lora

    outdir = os.path.join(_tmpdir(), "loras_diff")
    os.makedirs(outdir, exist_ok=True)
    import utils.paths as paths
    old = getattr(paths, "LORA_DIR", None)
    paths.LORA_DIR = outdir
    blobs = []
    try:
        for lr, name in ((0.0, "frozen"), (1e-2, "trained")):
            torch.manual_seed(999)
            prep = _prepared()
            pipe = _FakePipe(prep["unet"])
            cfg = TrainConfig(dataset_dir="x", output_name=name, rank=4, lr=lr,
                              steps=4, seed=11, resolution=32, warmup=1)
            res = train_lora(cfg, pipe=pipe, arch="sd15", device="cpu", _prepared=prep)
            assert res.ok, (name, res.error)
            with open(res.out_path, "rb") as fh:
                blobs.append(fh.read())
    finally:
        if old is not None:
            paths.LORA_DIR = old
    assert blobs[0] != blobs[1], "lr=0 与 lr>0 保存出完全相同的文件（优化器没生效）"
    return f"两个文件 {len(blobs[0])}B vs {len(blobs[1])}B，内容不同"


@check("cancel_stops_without_saving")
def test_cancel():
    from core.lora_train import TrainConfig, train_lora

    prep = _prepared()
    pipe = _FakePipe(prep["unet"])
    outdir = os.path.join(_tmpdir(), "loras_cancel")
    os.makedirs(outdir, exist_ok=True)
    import utils.paths as paths
    old = getattr(paths, "LORA_DIR", None)
    paths.LORA_DIR = outdir
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] > 3

    try:
        cfg = TrainConfig(dataset_dir="x", output_name="cancelled", rank=4, lr=1e-3,
                          steps=50, seed=7, resolution=32, warmup=1)
        res = train_lora(cfg, pipe=pipe, arch="sd15", device="cpu", _prepared=prep,
                         should_cancel=should_cancel)
    finally:
        if old is not None:
            paths.LORA_DIR = old
    assert res.cancelled, res.error
    assert 0 < res.steps_done < 50, res.steps_done
    assert not os.path.exists(os.path.join(outdir, "cancelled.safetensors"))
    return f"在 {res.steps_done} 步停下，未落盘"


@check("training_empty_dataset_returns_error_not_crash")
def test_empty_dataset():
    from core.lora_train import TrainConfig, train_lora

    class _Pipe:
        unet = _tiny_unet()

        def save_lora_weights(self, *a, **k):
            raise AssertionError("不该走到保存")

    empty = os.path.join(_tmpdir(), "empty2")
    os.makedirs(empty, exist_ok=True)
    cfg = TrainConfig(dataset_dir=empty, output_name="nope", steps=2, resolution=32)
    res = train_lora(cfg, pipe=_Pipe(), arch="sd15", device="cpu")
    assert not res.ok and not res.cancelled
    assert "数据集" in (res.error or ""), res.error
    return res.error[:40]


# ---------------------------------------------------------------- 多架构
@check("arch_family_maps_three_families")
def test_arch_family():
    from core.lora_train import arch_family

    assert arch_family("sd15") == "sd"
    assert arch_family("sdxl") == "sd"
    assert arch_family("sd3") == "sd3"
    assert arch_family("sd35") == "sd3"
    assert arch_family("flux") == "flux"
    assert arch_family("flux1") == "flux"
    assert arch_family("") == "sd"
    return "sd/sd3/flux 三族映射正确"


@check("backbone_targets_cover_transformer_naming")
def test_backbone_targets():
    from core.lora_train import backbone_targets

    sd = backbone_targets("sd15")
    sd3 = backbone_targets("sd3")
    flux = backbone_targets("flux")
    assert "to_q" in sd and "to_out.0" in sd
    assert "add_q_proj" in sd3 and "add_q_proj" not in sd
    assert "proj_mlp" in flux and "proj_out" in flux and "proj_mlp" not in sd3
    return f"sd={len(sd)} sd3={len(sd3)} flux={len(flux)}"


@check("pack_latents_patches_2x2_and_rejects_odd")
def test_pack_latents():
    import torch
    from core.lora_train import _pack_latents

    lat = torch.zeros(2, 32, 64, 64)
    packed = _pack_latents(lat)
    assert tuple(packed.shape) == (2, 1024, 128), packed.shape
    try:
        _pack_latents(torch.zeros(1, 32, 63, 64))
    except ValueError as exc:
        assert "偶数" in str(exc), str(exc)
        return "打包 (2,1024,128)，奇数长宽报错"
    raise AssertionError("奇数长宽应当报错")


@check("predict_dispatches_by_family")
def test_predict():
    import torch
    from core.lora_train import _predict

    class _Sample:
        def __init__(self, t):
            self.sample = t

    seen = {}

    class _Sd(torch.nn.Module):
        def forward(self, noisy, timesteps, enc, added_cond_kwargs=None):
            seen["sd"] = dict(added=bool(added_cond_kwargs))
            return _Sample(torch.zeros_like(noisy))

    class _Tf(torch.nn.Module):
        def forward(self, hidden_states=None, encoder_hidden_states=None,
                    pooled_projections=None, timestep=None, img_ids=None,
                    txt_ids=None, guidance=None, return_dict=True):
            seen["tf"] = dict(has_img=img_ids is not None, has_txt=txt_ids is not None,
                              has_g=guidance is not None,
                              t=float(timestep.reshape(-1)[0]))
            return (torch.zeros_like(hidden_states),)

    noisy = torch.zeros(1, 32, 16, 16)
    enc = torch.zeros(1, 5, 16)
    pred, _ = _predict("sd", _Sd(), noisy, torch.tensor([3]), enc, None,
                       (64, 64), 1, "cpu", torch.float32, 0.0)
    assert tuple(pred.shape) == (1, 32, 16, 16) and seen["sd"]["added"] is False
    pred2, _ = _predict("sd3", _Tf(), noisy, torch.tensor([250.0]), enc,
                        torch.zeros(1, 8), (64, 64), 1, "cpu", torch.float32, 0.0)
    assert tuple(pred2.shape) == (1, 32, 16, 16)
    assert seen["tf"]["t"] == 0.25, seen["tf"]["t"]
    pred3, _ = _predict("flux", _Tf(), noisy, torch.tensor([500.0]), enc,
                        torch.zeros(1, 8), (64, 64), 1, "cpu", torch.float32, 3.5)
    # flux 的预测在「打包后」的空间里：(B, (H/2)*(W/2), C*4)
    # 16×16 latent → 8×8=64 个 token，32 通道 → 128
    assert tuple(pred3.shape) == (1, 64, 128), pred3.shape
    assert seen["tf"]["has_img"] and seen["tf"]["has_txt"] and seen["tf"]["has_g"]
    return "sd 用 added_cond_kwargs；sd3/flux 用 transformer 签名"


@check("save_lora_picks_kwarg_by_signature")
def test_save_lora_kwarg():
    import torch
    from core import lora_train as lt

    captured = {}

    class _Unet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

    class _PipeSd:
        unet = _Unet()

        def save_lora_weights(self, save_directory=None,
                              unet_lora_layers=None, text_encoder_lora_layers=None,
                              safe_serialization=True):
            captured["sd"] = (unet_lora_layers, safe_serialization)
            raise RuntimeError("stop")

    class _PipeTf(_PipeSd):
        unet = None
        transformer = _Unet()

        def save_lora_weights(self, save_directory=None,
                              transformer_lora_layers=None,
                              text_encoder_lora_layers=None,
                              safe_serialization=True):
            captured["tf"] = (transformer_lora_layers, safe_serialization)
            raise RuntimeError("stop")

    orig = lt._lora_state_dict
    lt._lora_state_dict = lambda m: {"unet.x.lora.down.weight": torch.zeros(2)}
    try:
        cfg = lt.TrainConfig(output_name="kw", steps=1)
        for cls, key, arch in ((_PipeSd, "sd", "sd15"), (_PipeTf, "tf", "flux")):
            try:
                lt._save_lora(cls(), cfg, arch, 1, [0.1])
            except RuntimeError as exc:
                assert "stop" in str(exc), exc
            assert key in captured, key
        assert isinstance(captured["sd"][0], dict)
        assert isinstance(captured["tf"][0], dict)
    finally:
        lt._lora_state_dict = orig
    return "SD 用 unet_lora_layers，SD3/Flux 用 transformer_lora_layers"


@check("vram_estimate_grows_with_backbone")
def test_vram_by_family():
    from core.lora_train import TrainConfig, estimate_vram_mb, vram_advice

    cfg = TrainConfig(resolution=512, batch_size=1)
    sd = estimate_vram_mb(cfg, "sd15")
    sdxl = estimate_vram_mb(cfg, "sdxl")
    sd3 = estimate_vram_mb(cfg, "sd3")
    flux = estimate_vram_mb(cfg, "flux")
    assert sd < sdxl < sd3 < flux, (sd, sdxl, sd3, flux)
    advice = vram_advice(cfg, "flux", available_mb=7400)
    assert "Flux" in advice and "8GB" in advice, advice
    return f"sd={sd} sdxl={sdxl} sd3={sd3} flux={flux}"


@check("trainable_arches_include_new_families")
def test_trainable_arches():
    from core.lora_train import TRAINABLE_ARCHES

    for a in ("sd15", "sd21", "sdxl", "pony", "sd3", "sd35", "flux"):
        assert a in TRAINABLE_ARCHES, a
    return ",".join(TRAINABLE_ARCHES)


def main():
    print("=" * 60)
    totals = len(RESULT)
    passed = sum(1 for _, ok, _ in RESULT if ok)
    print(f"总计 {totals} 项：PASS {passed} / FAIL {totals - passed}")
    print("=" * 60)
    return 0 if passed == totals else 1


if __name__ == "__main__":
    sys.exit(main())
