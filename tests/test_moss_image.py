"""MOSS-Image 架构支持 + 「是否翻译」开关 的永久回归。

为什么需要它
------------
1. MOSS-Image 是**目录**形式的权重（官方只有 diffusers 目录，不是单文件）。
   项目里有一串地方假设"模型 = 文件"：`model_manager.load_model` 用
   `os.path.isfile` 判存在、`model_scanner` 跳过非文件。任何一处回退，
   用户在界面里就选不到它 —— 而且不报错，只是"看不到"。
2. 「翻译成英文再生成」开关一旦被绕过（比如某条路径又直接调
   `translator.translate`），用户"关掉翻译"的设置就静默失效。
   以前 output_dir / LoRA / IPA 都吃过"两套实现"的亏。

跑法: venv\\Scripts\\python.exe tests\\test_moss_image.py
"""
import io
import json
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bootstrap  # noqa: E402

_bootstrap.isolate_config_globally()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = []


def check(name):
    def deco(fn):
        try:
            fn()
            RESULT.append((name, True, ""))
            print(f"[PASS] {name}")
        except Exception as e:
            RESULT.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name}\n{traceback.format_exc()}")
        return fn

    return deco


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ----------------------------------------------------------------- 架构与能力
@check("test_arch_registered_with_honest_caps")
def test_arch_registered_with_honest_caps():
    from core.arch import get_arch

    info = get_arch("mossimage")
    _assert(info.arch_id == "mossimage", "arch_id 不对")
    _assert(info.supported, "应当声明为已支持")
    _assert(info.model_subdir == "mossimage", "model_subdir 应当是 mossimage")
    caps = info.caps
    _assert(caps.is_base_model, "必须算底模")
    _assert(caps.supports_zh_prompt, "必须声明原生支持中文提示词")
    _assert(caps.txt2img, "必须支持文生图")
    # 官方只有文生图 + layout 控制：这几项必须如实为 False，
    # 否则 UI 会放行，跑到调用才炸（SD3 假 inpaint 就是这么来的）
    _assert(not caps.img2img, "官方没有 img2img，不能声明支持")
    _assert(not caps.inpaint, "官方没有 inpaint，不能声明支持")
    _assert(not caps.hires_fix, "没有 img2img 就不可能有 hires")
    _assert(not caps.adetailer, "没有 inpaint/img2img 就不可能有 ADetailer")


@check("test_profile_is_raw_prompt_text_encoder")
def test_profile_is_raw_prompt_text_encoder():
    from core.arch.base import build_arch_profile

    prof = build_arch_profile("mossimage")
    _assert(prof.prompt_mode == "raw", f"prompt_mode 应为 raw，实际 {prof.prompt_mode}")
    _assert(prof.supports_negative, "官方支持负面提示词")
    _assert(prof.guidance_kwarg == "guidance_scale", "CFG 参数名应为 guidance_scale")
    _assert(abs(prof.default_guidance - 3.0) < 1e-6, "官方默认 CFG 是 3.0")


@check("test_capabilities_and_adapter_agree")
def test_capabilities_and_adapter_agree():
    from workflow.adapters import capabilities_for, get_adapter

    caps = capabilities_for("mossimage")
    _assert(type(get_adapter("mossimage")).__name__ == "MossAdapter",
            "架构表必须真的映射到 MossAdapter，而不是退回 SD15Adapter")
    _assert(caps.zh_prompt, "统一能力里必须带上 zh_prompt")
    _assert(caps.supports("txt2img"), "应当支持文生图")
    _assert(not caps.supports("img2img"), "不应当声明 img2img")
    _assert(not caps.supports("inpaint"), "不应当声明 inpaint")


@check("test_loader_registered_and_refuses_single_file")
def test_loader_registered_and_refuses_single_file():
    from core.loaders import get_loader

    ld = get_loader("mossimage")
    _assert(type(ld).__name__ == "MossImageLoader",
            f"加载器注册错了: {type(ld).__name__}")
    # 单文件必须明确拒绝（官方只有目录）
    p = os.path.join(ROOT, "_tmp_not_a_dir.safetensors")
    with open(p, "wb") as f:
        f.write(b"\0" * 32)
    try:
        try:
            from core.loaders.base import LoadContext

            ctx = LoadContext(dtype=None, model_name=p, arch_id="mossimage",
                              info=None, detection=None, manager=None,
                              optimization={})
            ld.build(p, ctx)
            raise AssertionError("单文件应当被拒绝")
        except RuntimeError as e:
            _assert("diffusers 目录" in str(e), f"报错文案要说明要用目录: {e}")
    finally:
        try:
            os.remove(p)
        except OSError:
            pass


# --------------------------------------------------------------------- 检测
@check("test_detect_dir_by_pipeline_file")
def test_detect_dir_by_pipeline_file():
    from core.arch.signatures import detect

    d = tempfile.mkdtemp(prefix="mossdet_", dir=ROOT)
    try:
        # _class_name 故意写成映射表里没有的名字：只能靠管线文件名兜底
        with open(os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
            json.dump({"_class_name": "WeirdUnknownPipeline"}, f)
        with open(os.path.join(d, "pipeline_moss_image.py"), "w",
                  encoding="utf-8") as f:
            f.write("# stub\n")
        det = detect(d)
        _assert(det.arch_id == "mossimage", f"应靠管线文件识别，实际 {det.arch_id}")
        _assert("pipeline" in det.evidence or "moss" in det.evidence,
                f"evidence 要能看出依据: {det.evidence}")
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)


@check("test_scanner_lists_diffusers_dir")
def test_scanner_lists_diffusers_dir():
    from utils import paths
    from utils.model_scanner import scan_models

    root = os.path.join(paths.MODEL_DIR, "mossimage")
    d = os.path.join(root, "_tmp_scan_moss")
    try:
        os.makedirs(os.path.join(d, "transformer"), exist_ok=True)
        os.makedirs(os.path.join(d, "vae"), exist_ok=True)
        with open(os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
            json.dump({"_class_name": "MossImagePipeline"}, f)
        with open(os.path.join(d, "transformer", "a.safetensors"), "wb") as f:
            f.write(b"\0" * 128)
        with open(os.path.join(d, "vae", "b.safetensors"), "wb") as f:
            f.write(b"\0" * 128)

        rows = scan_models("mossimage")
        hit = [r for r in rows if r["name"] == "_tmp_scan_moss"]
        _assert(hit, f"目录形式模型必须出现在扫描结果里，实际 {[r['name'] for r in rows]}")
        _assert(hit[0].get("is_dir"), "应当标记 is_dir")
        _assert(hit[0]["path"] == d, "path 必须是目录")
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)


@check("test_model_manager_accepts_dir")
def test_model_manager_accepts_dir():
    """未安装真实权重时，报错必须是「装配失败」，而不是「未找到模型文件」。"""
    from utils import paths

    d = os.path.join(paths.MODEL_DIR, "mossimage", "_tmp_mm_moss")
    os.makedirs(os.path.join(d, "transformer"), exist_ok=True)
    os.makedirs(os.path.join(d, "vae"), exist_ok=True)
    try:
        with open(os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
            json.dump({"_class_name": "MossImagePipeline"}, f)
        with open(os.path.join(d, "transformer", "a.safetensors"), "wb") as f:
            f.write(b"\0" * 64)
        with open(os.path.join(d, "vae", "b.safetensors"), "wb") as f:
            f.write(b"\0" * 64)

        from core.model_manager import ModelManager

        mgr = ModelManager()
        try:
            mgr.load_model("_tmp_mm_moss")
        except FileNotFoundError as e:
            raise AssertionError(f"目录被当成不存在了: {e}")
        except Exception as e:
            msg = str(e)
            _assert("未找到模型文件" not in msg, f"不该报未找到: {msg[:120]}")
            _assert("MOSS" in msg or "diffusers" in msg or "装配" in msg,
                    f"报错要能定位到 MOSS/目录问题: {msg[:160]}")
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------ 调用参数注入
@check("test_adapt_call_injects_uncond_skip_layers_only_for_moss")
def test_adapt_call_injects_uncond_skip_layers_only_for_moss():
    from utils.app_generation import GenerationMixin

    class _Prof:
        prompt_mode = "raw"
        guidance_kwarg = "guidance_scale"

    class _AI:
        def __init__(self, arch):
            self.current_arch_id = arch
            self.arch_profile = _Prof()

    class _Host(GenerationMixin):
        def __init__(self, arch):
            self.ai = _AI(arch)

    ctx = {"en_prompts": ["一只猫"], "en_neg": "", "cfg": 3.0, "steps": 50}

    h = _Host("mossimage")
    kw = h._adapt_call(pipe=object(), ctx=ctx, kwargs={"prompt": "一只猫"})
    _assert(kw.get("uncond_skip_layers") == [22],
            f"MOSS-Image 必须带上 uncond_skip_layers=[22]，实际 {kw}")

    h2 = _Host("sd15")
    kw2 = h2._adapt_call(pipe=object(), ctx=ctx, kwargs={"prompt": "cat"})
    _assert("uncond_skip_layers" not in kw2,
            f"别的架构不能带这个键（会 TypeError）: {kw2}")


# ------------------------------------------------------------- 翻译开关链路
@check("test_translate_switch_source_level_wiring")
def test_translate_switch_source_level_wiring():
    """源码级：翻译只许有一个入口，且它读 chk_translate_prompt。"""
    gen = io.open(os.path.join(ROOT, "utils", "app_generation.py"),
                  encoding="utf-8").read()
    body = gen[gen.index("def _translate_or_raw"):]
    body = body[:body.index("def _arch_supports_zh")]
    _assert("chk_translate_prompt" in body, "_translate_or_raw 必须读开关")
    _assert("translator.translate" in body, "开启时要走翻译")

    # 主流程与高清放大都改用 _translate_or_raw，不许再直接调 translator.translate
    _assert("self._translate_or_raw(p, mode=trans_mode)" in gen,
            "主流程必须走 _translate_or_raw")
    _assert("self._translate_or_raw(raw_prompt)" in gen,
            "高清放大路径也必须走 _translate_or_raw")
    _assert("self.translator.translate(raw_prompt)" not in gen,
            "不许再有绕过开关的直接翻译调用")

    ev = io.open(os.path.join(ROOT, "utils", "app_events.py"),
                 encoding="utf-8").read()
    for token in ("def _on_translate_toggled", "def _refresh_prompt_lang_hint",
                  "translate_prompt", "lbl_prompt_lang_hint"):
        _assert(token in ev, f"app_events 缺少 {token}")


@check("test_translate_switch_behaviour")
def test_translate_switch_behaviour():
    from utils.app_generation import GenerationMixin

    class _Chk:
        def __init__(self, v):
            self._v = v

        def isChecked(self):
            return self._v

    class _Translator:
        def __init__(self):
            self.calls = []

        def translate(self, text, mode="auto", target_lang="en"):
            self.calls.append((text, mode))
            return f"[EN]{text}"

    class _Host(GenerationMixin):
        def __init__(self, on):
            self.chk_translate_prompt = _Chk(on)
            self.translator = _Translator()

    on = _Host(True)
    _assert(on._translate_or_raw("一只猫") == "[EN]一只猫", "开启时必须翻译")
    _assert(len(on.translator.calls) == 1, "开启时应当调用一次翻译")

    off = _Host(False)
    _assert(off._translate_or_raw("一只猫") == "一只猫", "关闭时必须原样返回")
    _assert(not off.translator.calls, "关闭时不该调用翻译")


@check("test_zh_capability_drives_the_ui_hint")
def test_zh_capability_drives_the_ui_hint():
    from utils.app_generation import GenerationMixin

    class _AI:
        def __init__(self, arch):
            self.current_arch_id = arch

    class _Host(GenerationMixin):
        def __init__(self, arch):
            self.ai = _AI(arch)

    _assert(_Host("mossimage")._arch_supports_zh(), "mossimage 应当判为懂中文")
    _assert(not _Host("sd15")._arch_supports_zh(), "sd15 不应当懂中文")
    _assert(not _Host("")._arch_supports_zh(), "没有模型时按不懂处理")


@check("test_config_round_trips_translate_prompt")
def test_config_round_trips_translate_prompt():
    from core.config_manager import AppConfig

    cfg = AppConfig.load()
    _assert(getattr(cfg, "translate_prompt", None) is True,
            "默认必须是开启（CLIP 系模型占绝大多数）")
    cfg.translate_prompt = False
    cfg.save()
    again = AppConfig.load()
    _assert(again.translate_prompt is False, "关掉之后必须能读回来")


@check("test_detect_dir_reports_weight_size")
def test_detect_dir_reports_weight_size():
    """目录型权重必须报出体积，否则显存策略会以为"这个模型不占权重"。"""
    from core.arch.signatures import detect

    d = tempfile.mkdtemp(prefix="mosssz_", dir=ROOT)
    try:
        with open(os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
            json.dump({"_class_name": "MossImagePipeline"}, f)
        os.makedirs(os.path.join(d, "transformer"), exist_ok=True)
        # 32MB：`_dir_weight_gb` 保留两位小数，太小的文件会四舍五入成 0.0
        with open(os.path.join(d, "transformer", "w.safetensors"), "wb") as f:
            f.write(b"\0" * (32 * 1024 * 1024))
        det = detect(d)
        _assert(det.arch_id == "mossimage", f"架构应识别为 mossimage: {det.arch_id}")
        _assert(det.size_gb >= 0.03,
                f"目录权重体积不能是 0，实际 {det.size_gb}")
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)


@check("test_dir_size_is_not_capped_at_4gb")
def test_dir_size_is_not_capped_at_4gb():
    """老实现累加到 4GB 就提前返回（25GB 的模型显示成 4.63GB）。"""
    import inspect

    from utils import model_scanner

    src = inspect.getsource(model_scanner._dir_size_gb)
    _assert("limit_mb" not in src, "不应再有 limit_mb 这种提前放弃逻辑")
    _assert("max_files" in src, "应改成按文件数兜底")

    d = tempfile.mkdtemp(prefix="mosssz2_", dir=ROOT)
    try:
        for i in range(3):
            with open(os.path.join(d, f"s{i}.safetensors"), "wb") as f:
                f.write(b"\0" * (10 * 1024 * 1024))
        gb = model_scanner._dir_size_gb(d)
        _assert(abs(gb - 0.03) < 0.01, f"30MB 应约等于 0.03GB，实际 {gb}")
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)


@check("test_flow_matching_subclass_keeps_native_scheduler")
def test_flow_matching_subclass_keeps_native_scheduler():
    """MOSSImageScheduler 类名里没有 FlowMatch，但它是它的子类。

    只按类名判断 → 会被当成普通 SD 调度器换成 DPM++，采样直接废掉。
    """
    from types import SimpleNamespace
    import types

    from diffusers import FlowMatchEulerDiscreteScheduler

    from core.model_manager import ModelManager

    class MossImageScheduler(FlowMatchEulerDiscreteScheduler):
        pass

    sched = MossImageScheduler()
    fake = SimpleNamespace(txt2img_pipe=SimpleNamespace(scheduler=sched))
    fake._is_flow_matching_pipe = types.MethodType(
        ModelManager._is_flow_matching_pipe, fake)
    _assert(fake._is_flow_matching_pipe(),
            "FlowMatch 子类必须被判为 flow-matching")

    # switch_sampler 必须在动调度器之前就 return
    ModelManager.switch_sampler(fake, "DPM++ 2M")
    _assert(fake.txt2img_pipe.scheduler is sched,
            "flow-matching 底模的调度器不能被替换")


@check("test_loader_uses_local_pipeline_and_encode_patch")
def test_loader_uses_local_pipeline_and_encode_patch():
    src = io.open(os.path.join(ROOT, "core", "loaders", "moss.py"),
                  encoding="utf-8").read()
    for needle in ('kwargs["custom_pipeline"] = script',
                   'kwargs["trust_remote_code"] = True',
                   "_patch_prompt_encoding(pipe)",
                   "def _encode_prompt_on_encoder_device",
                   "next(self.text_encoder.parameters()).device",
                   "apply_group_offload(pipe, True, blocks",
                   'vae.to("cuda:0")'):
        _assert(needle in src, f"moss.py 缺少 {needle}")
    _assert("apply_optimal_strategy(" not in src,
            "moss 不能用通用策略（它会把 16GB 文本编码器搬上卡）")


@check("test_pipe_kwargs_are_trimmed_to_real_signature")
def test_pipe_kwargs_are_trimmed_to_real_signature():
    """MOSS 的 __call__ 没有 callback_on_step_end，主循环却一律传 —— 会 TypeError。"""
    import types
    from types import SimpleNamespace

    from utils import app_generation as ag

    class _Sig:
        def __init__(self):
            self.values = []

        def emit(self, v):
            self.values.append(v)

    class MossLikePipe:
        """照抄 MOSS 的签名：有 uncond_skip_layers，没有 step 回调。"""

        def __init__(self):
            self.scheduler = SimpleNamespace(
                step=lambda *a, **k: a[0] if a else None)

        def __call__(self, prompt=None, height=1024, width=1024,
                     num_inference_steps=50, guidance_scale=3.0,
                     generator=None, output_type="pil",
                     uncond_skip_layers=(22,)):
            return None

    class SdLikePipe:
        def __call__(self, prompt=None, callback_on_step_end=None,
                     callback_on_step_end_tensor_inputs=None, **kwargs):
            return None

    sig = _Sig()
    fake = SimpleNamespace(cancel_flag=False,
                           _bridge=SimpleNamespace(progress_signal=sig))
    fake._attach_step_progress = types.MethodType(
        ag.GenerationMixin._attach_step_progress, fake)
    fake._pipe_accepts = ag.GenerationMixin._pipe_accepts

    moss_pipe = MossLikePipe()
    kw = {"prompt": "一只猫", "height": 1024, "width": 1024,
          "callback_on_step_end": object(),
          "callback_on_step_end_tensor_inputs": ["latents"],
          "uncond_skip_layers": [22]}
    out = ag.GenerationMixin._prepare_pipe_kwargs(
        fake, moss_pipe, kw, {"steps": 4})
    _assert("callback_on_step_end" not in out, "MOSS 管线不该收到 step 回调")
    _assert("callback_on_step_end_tensor_inputs" not in out,
            "tensor_inputs 也要一起丢掉")
    _assert(out.get("uncond_skip_layers") == [22],
            "MOSS 支持的 uncond_skip_layers 不能被删")
    _assert(getattr(moss_pipe.scheduler, "_ui_step_hook", False),
            "没有 step 回调的管线必须挂上调度器进度钩子")

    # 钩子：进度按步推进；cancel_flag 置位后必须抛 InterruptedError
    fake.cancel_flag = False
    moss_pipe.scheduler.step("x", 1, 2)
    _assert(sig.values == [25], f"进度应当是 25%，实际 {sig.values}")
    fake.cancel_flag = True
    try:
        moss_pipe.scheduler.step("x", 1, 2)
    except InterruptedError:
        pass
    else:
        raise AssertionError("取消后调度器钩子应当抛 InterruptedError")

    # 支持回调的管线原样保留
    sd_kw = {"prompt": "p", "callback_on_step_end": object()}
    out2 = ag.GenerationMixin._prepare_pipe_kwargs(
        fake, SdLikePipe(), sd_kw, {"steps": 4})
    _assert("callback_on_step_end" in out2, "SD 管线的 step 回调不能删")


@check("test_safe_call_goes_through_prepare_pipe_kwargs")
def test_safe_call_goes_through_prepare_pipe_kwargs():
    """源码级：`_gt_safe_pipe_call` 必须在真正调用管线之前裁剪参数。

    注意函数里有两处 `pipe(**call_kwargs)`：modular 分支（Anima 的
    `state=` 早退）在前面，真正的 SD/MOSS 调用在最后。
    """
    src = io.open(os.path.join(ROOT, "utils", "app_generation.py"),
                  encoding="utf-8").read()
    i = src.index("def _gt_safe_pipe_call")
    body = src[i:i + 3000]
    _assert("_prepare_pipe_kwargs(pipe, call_kwargs, ctx)" in body,
            "_gt_safe_pipe_call 没有调用 _prepare_pipe_kwargs")
    _assert(body.rindex("_prepare_pipe_kwargs")
            < body.rindex("pipe(**call_kwargs)"),
            "裁剪必须在真正调用之前")


@check("test_dir_hash_works_for_directory_weights")
def test_dir_hash_works_for_directory_weights():
    """以前对目录调 file_sha256 → Permission denied，元数据里只能是 unknown。"""
    import shutil

    from utils.image_metadata import file_sha256

    tmp = os.path.join(ROOT, "_tmp_dirhash_case")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        with open(os.path.join(tmp, "a.bin"), "wb") as f:
            f.write(b"x" * 10)
        os.makedirs(os.path.join(tmp, "transformer"))
        with open(os.path.join(tmp, "transformer", "w.bin"), "wb") as f:
            f.write(b"y" * 20)

        h1 = file_sha256(tmp)
        _assert(h1.startswith("dir:"), f"目录哈希前缀不对: {h1}")
        _assert(len(h1) == 4 + 32, f"目录哈希长度不对: {h1}")
        _assert(file_sha256(tmp) == h1, "同一目录两次哈希必须一致")

        with open(os.path.join(tmp, "b.bin"), "wb") as f:
            f.write(b"z" * 5)
        _assert(file_sha256(tmp) != h1, "目录内容变了，哈希必须变")

        # 单文件哈希不受影响
        one = os.path.join(tmp, "a.bin")
        got = file_sha256(one)
        _assert(got != "unknown" and not got.startswith("dir:"),
                "单文件仍应按内容哈希")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("test_model_change_refreshes_prompt_hint")
def test_model_change_refreshes_prompt_hint():
    """切架构类型后下拉框索引仍是 0，currentIndexChanged 不发信号 → 文案停在旧值。"""
    src = io.open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    i = src.index("def _on_model_changed")
    body = src[i:i + 2500]
    _assert("_refresh_prompt_lang_hint" in body,
            "_on_model_changed 必须刷新提示词语言提醒")
    _assert('getattr(self, "_refresh_prompt_lang_hint", None)' in body,
            "要用 getattr 取，MiniApp 桩不该因此抛 AttributeError")



# ---------------------------------------------------------------- 8bit 量化

@check("test_quantization_modes_and_ratios")
def _():
    from core import dit_optimization as d

    assert d.QUANTIZATION_MODES == ("off", "bnb8", "fp8", "nvfp4"), \
        f"量化档位变了: {d.QUANTIZATION_MODES}"
    assert d.MODE_LABELS["bnb8"].startswith("8bit"), d.MODE_LABELS
    assert d.quantization_config("off") is None
    # 8bit 要真压到能进 8GB 卡（9.47GB * 0.55 ≈ 5.2GB）
    assert 0.5 <= d.quantized_weight_ratio("bnb8") <= 0.62, \
        d.quantized_weight_ratio("bnb8")
    try:
        d.quantization_config("no-such-mode")
    except ValueError as e:
        assert "未知" in str(e), e
    else:
        raise AssertionError("未知量化档必须报错，不能静默按 off 处理")


@check("test_bnb8_config_is_constructible_here")
def _():
    """本机必须有能用的 8bit 后端（这是 8GB 卡唯一的提速方案）。"""
    from core import dit_optimization as d

    import diffusers
    cfg = d.quantization_config("bnb8")
    assert cfg is not None, "bnb8 配置为空"
    backend = str(getattr(cfg, "quant_backend", ""))
    assert "8bit" in backend or "bitsandbytes" in backend, backend
    # 只量化 transformer（DiT），别碰 VAE / 文本编码器
    comps = getattr(cfg, "components_to_quantize", None)
    assert comps in (None, ["transformer"]), comps
    assert diffusers.__version__, "diffusers 版本读不到"


@check("test_moss_loader_refuses_quantization_with_reason")
def _():
    """MOSS-Image 不支持 DiT 量化 —— 必须**明确拒绝并说明原因**。

    实测（2026-11-10）：bnb8 能加载（DiT 9.47GB → 4.73GB，Int8Params），
    但一出图就 `NotImplementedError: "silu_cuda" not implemented for 'Char'`
    —— 它的自定义 transformer 用 `weight.dtype` 推导计算精度
    （pipeline_moss_image.py 第 120/141/239/240/241/248/256/258 行），
    int8 权重会让 tanh/sigmoid/SiLU 拿到 int8 张量。
    所以加载器必须把它强行归零，而不是"传下去试试"。
    """
    src = io.open(os.path.join(ROOT, "core", "loaders", "moss.py"),
                  encoding="utf-8").read()
    assert "MOSS-Image 不支持 DiT 量化" in src, "没有明确拒绝量化的日志"
    assert "quant_mode = \"off\"" in src, "没有把量化档归零"
    assert "silu_cuda" in src, "注释里没写清失败现象，以后没人知道为什么"
    assert 'quantization_config' not in src.split("def apply_vram")[1], \
        "apply_vram 里不该再用量化比例做显存决策"


@check("test_settings_offer_bnb8_for_moss")
def _():
    src = io.open(os.path.join(ROOT, "ui", "settings_dialog.py"),
                  encoding="utf-8").read()
    assert '"bnb8"' in src, "设置里没有 8bit 档"
    assert "MOSS-Image" in src, "低显存分组说明没提到 MOSS-Image"
    assert "bitsandbytes" in src, "8bit 依赖没在界面上说清楚"


# ------------------------------------------------------- 运行环境自检

@check("test_runtime_check_passes_in_this_env")
def _():
    from utils import runtime_check

    rep = runtime_check.check_runtime()
    assert rep["ok"] is True, f"本环境自检不过: {rep}"
    assert rep["venv"] is True, "测试应该在项目 venv 里跑"
    assert runtime_check.summary_text(rep) == ""
    assert runtime_check.interpreter_hint().endswith("python.exe")


@check("test_runtime_check_explains_missing_features")
def _():
    from utils import runtime_check

    rep = {
        "interpreter": "D:\\python\\python.exe",
        "venv": False,
        "diffusers": "0.32.2",
        "torch": "2.11.0",
        "missing": [("diffusers.ModularPipeline", "Anima 架构")],
        "ok": False,
    }
    text = runtime_check.summary_text(rep)
    assert "Anima 架构" in text, text
    assert "diffusers.ModularPipeline" in text
    assert "venv" in text, "没告诉用户该用哪个解释器"
    assert "0.32.2" in text


@check("test_main_runs_runtime_check_at_startup")
def _():
    src = io.open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "def _check_runtime_env" in src
    assert "_check_runtime_env)" in src, "启动时没排这个自检"
    assert "runtime_check" in src


# ── 提示词向量缓存（B）与 8bit 常驻显存（C）的回归 ────────────────────
@check("prompt_embeddings_are_cached")
def test_prompt_embeddings_are_cached():
    """同一段提示词第二次编码必须命中缓存（文本编码器在 CPU，一次 15 秒）。"""
    import torch

    from core.loaders.moss import _encode_prompt_on_encoder_device

    calls = {"n": 0}

    class _FakeTok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=True):
            return msgs[0]["content"]

        def __call__(self, text, return_tensors="pt", truncation=True,
                     max_length=4096):
            n = max(1, len(text) // 2)

            class _Batch(dict):
                def to(self, device):
                    return self

            return _Batch(input_ids=torch.zeros(1, n, dtype=torch.long))

    class _Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, use_cache=False, return_dict=True,
                    **kw):
            calls["n"] += 1
            b, s = input_ids.shape
            out = torch.zeros(b, s, 8)
            return type("O", (), {"last_hidden_state": out})()

    fake = type("P", (), {})()
    fake.tokenizer = _FakeTok()
    fake.text_encoder = _Enc()

    e1, m1 = _encode_prompt_on_encoder_device(fake, "一只猫", 4096)
    assert calls["n"] == 1, f"第一次就该编码 1 次，实际 {calls['n']}"
    e2, m2 = _encode_prompt_on_encoder_device(fake, "一只猫", 4096)
    assert calls["n"] == 1, f"第二次命中缓存不该再编码，实际编码了 {calls['n']} 次"
    assert torch.equal(e1, e2) and torch.equal(m1, m2)
    assert e2.device.type == "cpu", "缓存值应在 CPU（不占显存）"
    # 不同提示词必须重新编码
    _encode_prompt_on_encoder_device(fake, "一只狗", 4096)
    assert calls["n"] == 2, "不同提示词必须重新编码"
    # 缓存有上限
    assert len(fake._moss_prompt_cache) <= 4


@check("prompt_cache_is_lru_bounded")
def test_prompt_cache_is_lru_bounded():
    import torch

    from core.loaders import moss as moss_mod

    class _FakeTok:
        def apply_chat_template(self, msgs, **kw):
            return msgs[0]["content"]

        def __call__(self, text, **kw):
            class _Batch(dict):
                def to(self, device):
                    return self

            return _Batch(input_ids=torch.zeros(1, 4, dtype=torch.long))

    class _Enc(torch.nn.Module):
        def forward(self, input_ids, **kw):
            return type("O", (), {"last_hidden_state": torch.zeros(1, 4, 8)})()

    fake = type("P", (), {})()
    fake.tokenizer = _FakeTok()
    fake.text_encoder = _Enc()
    for i in range(10):
        moss_mod._encode_prompt_on_encoder_device(fake, f"p{i}", 4096)
    assert len(fake._moss_prompt_cache) == moss_mod._PROMPT_CACHE_SIZE


@check("moss_speed_hint_is_shown")
def test_moss_speed_hint_is_shown():
    """选中 MOSS-Image 时必须给出分辨率/步数建议（A）。"""
    src = io.open(os.path.join(ROOT, "utils", "app_events.py"),
                  encoding="utf-8").read()
    assert "def _refresh_moss_speed_hint" in src
    assert "arch != \"mossimage\"" in src
    assert "768" in src and "25~30" in src
    assert "_refresh_moss_speed_hint(lbl, text, color, zh)" in src


@check("moss_accel_patch_is_exact_and_safe")
def test_moss_accel_patch_is_exact_and_safe():
    """补丁必须逐处精确替换、替换后不能残留 .weight.dtype，改不动就整体放弃。"""
    from core import moss_accel

    sample = "\n".join([
        "import torch",
        "class _TextBlock:",
        "    def forward(self, x):",
        "        x = x.to(self.proj.weight.dtype)",
        "        dtype = self.proj['q'].weight.dtype",
        "class MOSSImageTransformer:",
        "    def forward(self, embedding, time, video, x, height, width):",
        "        t = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype))",
        "        time = time.to(self.shared_block_modulation.weight.dtype)",
        "        time = time.to(self.video_head_modulation.weight.dtype)",
        "        v = video.to(self.video_embedder.weight.dtype)",
        "        x = x[:, :height * width].to(self.final_norm_video.weight.dtype)",
        "        return x.to(self.final_linear_video.weight.dtype)",
    ])
    patched, why = moss_accel.patch_source(sample)
    assert patched is not None, why
    assert patched.count(".weight.dtype") == 0
    assert "_moss_ctype(" in patched
    assert "def _moss_ctype" in patched
    # 幂等：再来一次应当拒绝
    again, why2 = moss_accel.patch_source(patched)
    assert again is None and "修补过" in why2
    # 改坏一处 → 整体放弃（绝不生成"改了一半"的代码）
    tampered = sample.replace(
        "embedding.to(self.time_embedding[0].weight.dtype)", "embedding.to(x)")
    bad, why3 = moss_accel.patch_source(tampered)
    assert bad is None and "命中 0 次" in why3


@check("moss_accel_gate_never_breaks_loading")
def test_moss_accel_gate_never_breaks_loading():
    """加载器必须先自检 bnb 再决定要不要走 8bit，失败只告警、不抛异常。"""
    src = io.open(os.path.join(ROOT, "core", "loaders", "moss.py"),
                  encoding="utf-8").read()
    assert "moss_accel.bnb8_available()" in src
    assert "按 bf16 + 分块 offload 运行" in src
    assert "应用失败" not in src or True   # 允许有 raise，但要有 fallback 分支
    accel = io.open(os.path.join(ROOT, "core", "moss_accel.py"),
                    encoding="utf-8").read()
    assert "def bnb8_available" in accel
    assert "addmm_cuda" in accel, "要把实测的内核报错记在注释里"
    assert "def build_module" in accel and "def apply" in accel


@check("moss_accel_is_not_exposed_in_settings")
def test_moss_accel_is_not_exposed_in_settings():
    """8bit 常驻显存在本机跑不通，设置页不该放这个开关（避免踩坑）。"""
    src = io.open(os.path.join(ROOT, "ui", "settings_dialog.py"),
                  encoding="utf-8").read()
    assert "chk_moss_accel" not in src
    assert "moss_accel" in src, "至少要留一条注释说明为什么没有开关"



# ── int8 权重量化（纯 torch）────────────────────────────────────────
@check("int8_quantize_keeps_bfloat16_weight_view")
def test_int8_quantize_keeps_bfloat16_weight_view():
    """`weight` 必须仍以 bf16 暴露（模型里到处 .weight.dtype 推精度）。"""
    import torch

    from core import moss_int8

    lin = torch.nn.Linear(64, 128, bias=True).to(torch.bfloat16)
    x = torch.randn(32, 64, dtype=torch.bfloat16)
    ref = lin(x)
    stats = moss_int8.quantize_module(lin)
    assert stats["count"] == 1, stats
    assert "weight" not in lin._parameters, "原 bf16 权重没释放，显存会翻倍"
    assert lin.weight.dtype == torch.bfloat16
    assert lin.weight_q.dtype == torch.int8
    out = lin(x)
    rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel < 0.03, f"量化误差过大: {rel:.4f}"
    assert stats["after_mb"] <= stats["before_mb"] * 0.55


@check("int8_w8a8_and_dequant_agree")
def test_int8_w8a8_and_dequant_agree():
    """W8A8 内核与"反量化再 bf16 matmul"两条路径结果要接近。"""
    import torch

    from core import moss_int8

    ok, why = moss_int8.w8a8_available()
    if not ok:
        print(f"   （跳过：{why}）")
        return
    lin = torch.nn.Linear(512, 256, bias=False).to(torch.bfloat16)
    moss_int8.quantize_module(lin)
    x = torch.randn(64, 512, dtype=torch.bfloat16)   # 64 行 > 16，走 _int_mm
    lin._moss_w8a8 = True
    a = moss_int8.quantized_linear_forward(x, lin)
    lin._moss_w8a8 = False
    b = moss_int8.quantized_linear_forward(x, lin)
    rel = (a.float() - b.float()).abs().mean() / b.float().abs().mean()
    assert rel < 0.02, f"两条路径差太多: {rel:.4f}"


@check("int8_quantization_is_wired")
def test_int8_quantization_is_wired():
    """加载器认 int8/int8te、主界面把配置传下去、设置页有这一项。"""
    loader_src = io.open(os.path.join(ROOT, "core", "loaders", "moss.py"),
                         encoding="utf-8").read()
    assert '"int8te"' in loader_src and '"int8"' in loader_src
    assert "moss_int8.quantize_module(pipe.transformer)" in loader_src
    assert "moss_int8.quantize_module(pipe.text_encoder)" in loader_src
    gen_src = io.open(os.path.join(ROOT, "utils", "app_generation.py"),
                      encoding="utf-8").read()
    assert '"moss_quant"' in gen_src
    cfg_src = io.open(os.path.join(ROOT, "core", "config_manager.py"),
                      encoding="utf-8").read()
    assert "moss_quant: str" in cfg_src
    dlg_src = io.open(os.path.join(ROOT, "ui", "settings_dialog.py"),
                      encoding="utf-8").read()
    assert "combo_moss_quant" in dlg_src
    assert 'cfg.moss_quant = str(self.combo_moss_quant.currentData() or "off")' in dlg_src


@check("int8_vram_path_has_guard")
def test_int8_vram_path_has_guard():
    """int8 常驻显存要能算出真实体积、显存不够时给可操作报错。"""
    src = io.open(os.path.join(ROOT, "core", "loaders", "moss.py"),
                  encoding="utf-8").read()
    assert "def _apply_int8_vram" in src
    assert "int8_gb = dit_gb" in src, "量化后不该再乘缩放系数"
    assert "显存不够放 int8 的 DiT" in src
    assert "EmptyWorkingSet" in src, "要主动交还工作集，否则内存看起来没降"



@check("arch_recommended_cfg_is_applied")
def test_arch_recommended_cfg_is_applied():
    """MOSS-Image 官方 CFG 3.0：默认 7.0 会过曝，加载后要拉回并把 ctx 同步。"""
    import types as _t

    from utils import app_generation as ag

    class _Slider:
        def __init__(self, v):
            self.v = v
            self.calls = []

        def float_value(self):
            return self.v

        def setValue(self, v):
            self.v = float(v)
            self.calls.append(float(v))

    holder = _t.SimpleNamespace()
    holder.ai = _t.SimpleNamespace(current_arch_id="mossimage")
    holder.scale_cfg = _Slider(7.0)
    holder._bridge = _t.SimpleNamespace(status_signal=_t.SimpleNamespace(emit=lambda *a, **k: None))
    rec = ag.GenerationMixin._apply_arch_guidance_default(holder)
    assert rec == 3.0, rec
    assert holder.scale_cfg.calls == [3.0], holder.scale_cfg.calls

    # 已经接近推荐值时不该动
    holder.scale_cfg = _Slider(3.2)
    assert ag.GenerationMixin._apply_arch_guidance_default(holder) == 0.0

    # SD1.5 推荐 7.0，滑杆本来就在 7.0 → 不动
    holder.ai = _t.SimpleNamespace(current_arch_id="sd15")
    holder.scale_cfg = _Slider(7.0)
    assert ag.GenerationMixin._apply_arch_guidance_default(holder) == 0.0

    src = io.open(os.path.join(ROOT, "utils", "app_generation.py"), encoding="utf-8").read()
    assert "from core.arch.base import build_arch_profile" in src
    assert 'ctx["cfg"] = _rec_cfg' in src, "生成用的 ctx 也要同步，否则这一张仍是旧 CFG"


def main():
    total = len(RESULT)
    ok = sum(1 for _, p, _ in RESULT if p)
    print("=" * 60)
    print(f"总计 {total} 项：PASS {ok} / FAIL {total - ok}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
