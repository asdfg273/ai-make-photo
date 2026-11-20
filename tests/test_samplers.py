# tests/test_samplers.py
# ============================================================
#  采样器 / sigma 风格的回归
# ------------------------------------------------------------
#  修之前的状态（这就是本文件存在的理由）：
#    core/model_manager.switch_sampler 只有四个 if 分支，
#      Euler a / Euler / "DPM++ 2M"（无条件 karras）/ DDIM，
#    其余**全部静默落到 EulerAncestralDiscreteScheduler**。
#    于是界面上的 "DPM++ SDE Karras"、"UniPC"、"LCM"、"PNDM"
#    点了等于没点 —— 用户会觉得"怎么换采样器都一个样"。
#
#  这里锁三件事：
#    1. core/samplers.py 的每个采样器真的能构造、能排时间步；
#    2. ModelManager.switch_sampler 真的换成了对应的调度器类；
#    3. 界面 / 节点 / 设置页的候选都来自同一个列表（不许再各写一份）。
# ============================================================
import _bootstrap  # noqa: F401  （路径 + 配置隔离）

import io
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_bootstrap.isolate_config_globally()

FAILS = []


def check(name):
    def deco(fn):
        try:
            fn()
        except Exception as e:
            import traceback
            FAILS.append((name, f"{e}"))
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
        else:
            print(f"[PASS] {name}")
        return fn
    return deco


def _base_config():
    from diffusers import EulerDiscreteScheduler
    return EulerDiscreteScheduler().config


# ----------------------------------------------------------------------
@check("test_every_sampler_builds_and_steps")
def test_every_sampler_builds_and_steps():
    from core import samplers

    base = _base_config()
    built = 0
    for d in samplers.SAMPLERS:
        if not d.available():
            assert d.requires, f"{d.name} 不可用却没说缺什么"
            continue
        for style in samplers.SIGMA_STYLES:
            sch = samplers.build_scheduler(base, d.name, style)
            assert sch is not None, f"{d.name}/{style} 没构造出来"
            sch.set_timesteps(12, device="cpu")
            n = len(list(sch.timesteps))
            assert n >= 12, f"{d.name}/{style} 时间步只有 {n}"
            built += 1
    assert built >= 50, f"只构造出 {built} 个组合，采样器表可能被削了"


@check("test_unknown_sampler_returns_none")
def test_unknown_sampler_returns_none():
    from core import samplers
    assert samplers.build_scheduler(_base_config(), "完全不存在") is None
    assert samplers.find_sampler("") is None
    assert samplers.build_scheduler(_base_config(), "") is None


@check("test_legacy_karras_suffix_still_parses")
def test_legacy_karras_suffix_still_parses():
    """老配置里写的是 "DPM++ 2M Karras" —— 必须真 karras。"""
    from core import samplers

    base = _base_config()
    with_k = samplers.build_scheduler(base, "DPM++ 2M Karras")
    without = samplers.build_scheduler(base, "DPM++ 2M")
    assert with_k.config.use_karras_sigmas is True
    assert without.config.use_karras_sigmas is False

    # 显式 sigma 参数优先于名字里的后缀
    forced = samplers.build_scheduler(base, "DPM++ 2M", "默认")
    assert forced.config.use_karras_sigmas is False


@check("test_sigma_style_is_filtered_per_scheduler")
def test_sigma_style_is_filtered_per_scheduler():
    """DDIM / PNDM / LCM 没有 karras 选项，塞进去会 TypeError —— 必须被过滤。"""
    from core import samplers
    from diffusers import DDIMScheduler

    base = _base_config()
    sch = samplers.build_scheduler(base, "DDIM", "Karras")
    assert isinstance(sch, DDIMScheduler)
    assert not getattr(sch.config, "use_karras_sigmas", False)


@check("test_comfy_names_are_unique")
def test_comfy_names_are_unique():
    from core import samplers
    names = [d.comfy for d in samplers.SAMPLERS]
    assert all(names), f"有采样器没写 comfy 名: {names}"
    assert len(set(names)) == len(names), f"comfy 名重复: {names}"
    assert samplers.sigma_to_comfy("Karras") == "karras"
    assert samplers.sigma_to_comfy("默认") == "normal"


@check("test_manager_switch_really_changes_scheduler_class")
def test_manager_switch_really_changes_scheduler_class():
    """直接拿一个不加载权重的 ModelManager 实例验证切换。"""
    from core.model_manager import ModelManager
    from diffusers import (DPMSolverMultistepScheduler,
                           EulerAncestralDiscreteScheduler,
                           UniPCMultistepScheduler)

    ai = object.__new__(ModelManager)
    ai.txt2img_pipe = types.SimpleNamespace()
    ai.img2img_pipe = None
    ai.inpaint_pipe = None
    ai.controlnet_pipe = None
    ai.reference_pipe = None
    ai.is_vpred = False
    ai.zero_snr = None
    ai._sigma_style = None
    ai._is_flow_matching_pipe = lambda: False
    ai.txt2img_pipe.scheduler = EulerAncestralDiscreteScheduler()

    ai.switch_sampler("UniPC", "Karras")
    sch = ai.txt2img_pipe.scheduler
    assert isinstance(sch, UniPCMultistepScheduler), type(sch).__name__
    assert sch.config.use_karras_sigmas is True
    assert ai._sampler_name == "UniPC" and ai._sigma_style == "Karras"

    ai.switch_sampler("DPM++ 2M")
    sch2 = ai.txt2img_pipe.scheduler
    assert isinstance(sch2, DPMSolverMultistepScheduler), type(sch2).__name__
    assert sch2.config.use_karras_sigmas is False


@check("test_manager_keeps_old_scheduler_on_unknown_name")
def test_manager_keeps_old_scheduler_on_unknown_name():
    from core.model_manager import ModelManager
    from diffusers import EulerDiscreteScheduler

    ai = object.__new__(ModelManager)
    ai.txt2img_pipe = types.SimpleNamespace(scheduler=EulerDiscreteScheduler())
    ai.img2img_pipe = None
    ai.inpaint_pipe = None
    ai.controlnet_pipe = None
    ai.reference_pipe = None
    ai.is_vpred = False
    ai.zero_snr = None
    ai._sigma_style = None
    ai._is_flow_matching_pipe = lambda: False

    ai.switch_sampler("这是一个不存在的采样器")
    assert isinstance(ai.txt2img_pipe.scheduler, EulerDiscreteScheduler)


@check("test_vpred_and_zero_snr_overrides_survive")
def test_vpred_and_zero_snr_overrides_survive():
    from core.model_manager import ModelManager
    from diffusers import EulerDiscreteScheduler

    ai = object.__new__(ModelManager)
    ai.txt2img_pipe = types.SimpleNamespace(scheduler=EulerDiscreteScheduler())
    ai.img2img_pipe = None
    ai.inpaint_pipe = None
    ai.controlnet_pipe = None
    ai.reference_pipe = None
    ai.is_vpred = True
    ai.zero_snr = True
    ai._sigma_style = None
    ai._is_flow_matching_pipe = lambda: False

    ai.switch_sampler("Euler")
    cfg = ai.txt2img_pipe.scheduler.config
    assert cfg.prediction_type == "v_prediction", cfg.prediction_type
    assert cfg.rescale_betas_zero_snr is True


@check("test_ui_node_and_settings_share_one_list")
def test_ui_node_and_settings_share_one_list():
    """界面/节点/设置页都必须从 core.samplers 取候选，不许各写一份。"""
    src = {p: io.open(ROOT / p, encoding="utf-8").read()
           for p in ("ui/core_panel.py", "ui/settings_dialog.py",
                     "nodes/sampling_nodes.py", "ui/contracts.py")}

    assert "sampler_names" in src["ui/core_panel.py"]
    assert "sigma_styles" in src["ui/core_panel.py"]
    assert "sampler_names" in src["ui/settings_dialog.py"]
    assert "samplers_mod.sampler_names()" in src["nodes/sampling_nodes.py"]
    assert "combo_scheduler" in src["ui/contracts.py"]

    # 不许再出现写死的旧六项列表
    for path in ("ui/core_panel.py", "ui/settings_dialog.py"):
        assert '"DPM++ SDE Karras"' not in src[path], f"{path} 又写死了采样器列表"

    # model_manager 不许再有"猜名字"的 if 分支
    mm = io.open(ROOT / "core/model_manager.py", encoding="utf-8").read()
    assert "from core import samplers as _samplers" in mm
    assert '"欧拉A" in sampler_name' not in mm


@check("test_sampler_node_exposes_scheduler_and_eta")
def test_sampler_node_exposes_scheduler_and_eta():
    from core import samplers
    from workflow.registry import all_specs, load_builtin_nodes

    load_builtin_nodes()
    spec = all_specs().get("Sampler")
    assert spec is not None, "注册表里没有 Sampler 节点"

    params = spec.params
    assert "scheduler" in params and "eta" in params, sorted(params)
    assert tuple(params["scheduler"].choices) == samplers.sigma_styles()
    assert "Euler a" in params["sampler"].choices
    # 缺依赖的采样器不该出现在候选里
    hidden = {name for name, _need in samplers.unavailable_samplers()}
    for name in hidden:
        assert name not in params["sampler"].choices, f"{name} 缺依赖却可选"


@check("test_config_round_trips_sampler_scheduler")
def test_config_round_trips_sampler_scheduler():
    from core.config_manager import AppConfig

    cfg = AppConfig()
    assert cfg.sampler_scheduler == "默认"
    cfg.sampler_scheduler = "Exponential"
    cfg.sampler_scheduler = "Exponential"   # 幂等，不重复写
    assert AppConfig().sampler_scheduler == "默认"


@check("test_metadata_records_sigma_style")
def test_metadata_records_sigma_style():
    """出图元数据要能记住 sigma 风格，否则回填时风格永远丢。"""
    from utils.image_metadata import a1111_parameters, _a1111_schedule

    assert _a1111_schedule("默认") == "Automatic"
    assert _a1111_schedule(None) == "Automatic"
    assert _a1111_schedule("Karras") == "Karras"
    text = a1111_parameters({"prompt": "x", "sampler": "DPM++ 2M",
                             "scheduler": "Karras"})
    assert "Schedule type: Karras" in text, text
    assert "Sampler: DPM++ 2M" in text, text

    src = io.open(ROOT / "utils" / "image_metadata.py", encoding="utf-8").read()
    assert 'put("scheduler"' in src, "build_metadata 必须写 scheduler"

    # 回填时要把老写法拆开（combo 里没有 "DPM++ 2M Karras" 这一项）
    main_src = io.open(ROOT / "main.py", encoding="utf-8").read()
    assert "_samplers.resolve(name, style)" in main_src
    assert '"combo_scheduler"' in main_src


print("=" * 60)
if FAILS:
    print(f"总计：FAIL {len(FAILS)}")
    for name, err in FAILS:
        print(f"  - {name}: {err}")
    sys.exit(1)
print("全部通过")
sys.exit(0)
