# tests/test_loaders.py — 加载器注册表与派生管线结构测试（不加载真实模型）
import os, sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_registry_mapping():
    """每个架构都路由到正确的专用加载器。"""
    from core.loaders import get_loader
    from core.loaders.sd15 import SD15Loader
    from core.loaders.sdxl import SDXLLoader
    from core.loaders.anima import AnimaLoader
    from core.loaders.single_file import SingleFileLoader

    assert isinstance(get_loader("sd15"), SD15Loader)
    assert isinstance(get_loader("sd21"), SD15Loader), "sd21 应复用 SD15Loader"
    assert isinstance(get_loader("sdxl"), SDXLLoader)
    assert isinstance(get_loader("anima"), AnimaLoader)
    assert isinstance(get_loader("sd3"), SingleFileLoader)
    assert isinstance(get_loader("flux"), SingleFileLoader)
    # sd15/sdxl 不应再被通用加载器占用
    assert not isinstance(get_loader("sd15"), SingleFileLoader)
    assert not isinstance(get_loader("sdxl"), SingleFileLoader)
    print("PASS test_registry_mapping")


def test_derive_from_components():
    """共享 derive helper：从 txt2img 组件派生 img2img/inpaint，轻量优化被调用。"""
    from core.loaders.base import LoadResult, LoadContext
    from core.loaders.sd15 import SD15Loader

    class _FakePipe:
        components = {"unet": "U", "vae": "V"}

    created = {}

    class _FakeI2I:
        def __init__(self, **comps):
            created["img2img"] = comps

    class _FakeInpaint:
        def __init__(self, **comps):
            created["inpaint"] = comps

    class _FakeMgr:
        optimized = []

        def _apply_light_optimizations(self, pipe, name=""):
            self.optimized.append(name)

    result = LoadResult(txt2img=_FakePipe())
    ctx = LoadContext(dtype=None, arch_id="sd15", manager=_FakeMgr())

    loader = SD15Loader()
    # 不跑 build（需要真实模型），直接注入类并测试派生
    result.extras["classes"] = (_FakeI2I, _FakeInpaint)
    loader.derive_pipes(result, ctx)

    assert created["img2img"] == {"unet": "U", "vae": "V"}
    assert created["inpaint"] == {"unet": "U", "vae": "V"}
    assert ctx.manager.optimized == ["img2img", "inpaint"]
    assert result.img2img is not None and result.inpaint is not None
    print("PASS test_derive_from_components")


def test_build_uses_expected_pipes():
    """静态核查：sd15/sdxl 的 build 引用正确的 Pipeline 类与关键参数。"""
    import inspect
    from core.loaders.sd15 import SD15Loader
    from core.loaders.sdxl import SDXLLoader

    s15 = inspect.getsource(SD15Loader.build)
    assert "StableDiffusionPipeline" in s15
    assert "from_single_file" in s15
    assert "safety_checker=None" in s15

    sxl = inspect.getsource(SDXLLoader.build)
    assert "StableDiffusionXLPipeline" in sxl
    assert "from_single_file" in sxl
    assert SDXLLoader.CONFIG_REPO == "stabilityai/stable-diffusion-xl-base-1.0"
    assert "self.CONFIG_REPO" in sxl
    print("PASS test_build_uses_expected_pipes")


if __name__ == "__main__":
    for fn in (test_registry_mapping, test_derive_from_components,
               test_build_uses_expected_pipes):
        fn()
    sys.stdout.flush()
    print("\n✅ 全部加载器结构测试通过")
    os._exit(0)
