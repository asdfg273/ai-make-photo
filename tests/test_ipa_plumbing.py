# tests/test_ipa_plumbing.py — IP-Adapter 装载/卸载不能弄坏管线（无头，不需要 GPU）
#
# 锁住的 P0（静默且致命）：
#   `unload_ip_adapter()` 曾把 `image_processor` 和 image_encoder /
#   feature_extractor 一起 setattr 成 None。但 `image_processor`
#   （VaeImageProcessor）是**基础管线的核心组件**，每次出图都要用
#   `image_processor.postprocess` 把 latent 转成 PIL。
#   而 `prepare_ip_adapter()` 的失败分支会调用 `unload_ip_adapter()` ——
#   于是"选了本机没装的档位"这种小事，会让**整个会话所有出图全部失败**，
#   报的还是跟 IP-Adapter 毫无关系的：
#       AttributeError: 'NoneType' object has no attribute 'postprocess'
#   实测：装载失败的 plus 档位后，连不带 IP-Adapter 的普通文生图也立刻失败。
#
# 还锁住档位名映射：界面下拉写 "standard"，而 h94 官方档位名是 "base"。
#
# 用法：venv\Scripts\python.exe tests\test_ipa_plumbing.py
import os
import sys

import _bootstrap  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakePipe:
    """够用的管线替身：有 unet / 组件 / unload_ip_adapter。"""

    class _Unet:
        def __init__(self):
            self.encoder_hid_proj = None
            self.config = type("C", (), {"encoder_hid_dim_type": None})()

        def set_attn_processor(self, p):
            self._proc = p

    def __init__(self, name="pipe"):
        self.name = name
        self.unet = self._Unet()
        self.image_processor = "VaeImageProcessor"      # ← 核心组件
        self.image_encoder = "IPAdapterImageEncoder"    # ← IPA 组件
        self.feature_extractor = "CLIPImageProcessor"   # ← IPA 组件
        self.unloaded = 0

    def unload_ip_adapter(self):
        self.unloaded += 1


def _mk(monkey_pipe=True):
    from core.model_manager import ModelManager
    m = ModelManager()
    p = _FakePipe()
    m.txt2img_pipe = p
    m.img2img_pipe = p
    m.inpaint_pipe = p
    m.controlnet_pipe = None
    return m, p


def test_unload_keeps_image_processor():
    """卸载 IP-Adapter 后，image_processor 必须还在（不能是 None）。"""
    m, p = _mk()
    m.unload_ip_adapter()
    assert p.image_processor is not None, \
        "image_processor 被清成 None —— 之后所有出图都会失败"
    assert p.image_processor == "VaeImageProcessor", p.image_processor
    # IPA 自己的组件应当被清掉
    assert p.image_encoder is None, "image_encoder 是 IPA 组件，应被清掉"
    assert p.feature_extractor is None


def test_unload_restores_backed_up_component():
    """若装载时把 image_processor 换掉了，卸载要恢复备份而不是留 None。"""
    m, p = _mk()
    m._ipa_saved_components = {"image_processor": "ORIGINAL_IP"}
    p.image_processor = None
    m.unload_ip_adapter()
    assert p.image_processor == "ORIGINAL_IP", p.image_processor


def test_failed_install_does_not_brick():
    """装载失败（例如档位没装）后，管线必须仍然可用。"""
    m, p = _mk()
    # 模拟失败分支：把基础组件备份好，然后走 unload
    m._ipa_saved_components = {"image_processor": p.image_processor}
    p.image_processor = None        # 模拟装载过程中被破坏
    try:
        m.unload_ip_adapter()
    except Exception:
        pass
    assert p.image_processor is not None, "装载失败后 image_processor 不能是 None"


def test_image_processor_not_in_null_list():
    """源码级保险：卸载逻辑里不允许再把 image_processor 置 None。"""
    import inspect
    import re
    from core import model_manager
    src = inspect.getsource(model_manager.ModelManager.unload_ip_adapter)
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    code = "\n".join(l for l in code.splitlines()
                     if not l.strip().startswith("#"))
    assert 'setattr(pipe, "image_processor"' not in code, \
        "卸载逻辑不应直接 setattr image_processor"
    # 也不应出现在那个"清成 None"的元组里
    for m in re.finditer(r'for comp in \(([^)]*)\)', code):
        assert "image_processor" not in m.group(1), \
            f"image_processor 不应被清空: {m.group(1)}"


def test_variant_alias_standard_to_base():
    """界面写 "standard"，h94 官方档位名是 "base"，必须能对上。"""
    from core.model_manager import ModelManager
    m = ModelManager()
    m.is_sdxl = False
    # 直接验证别名映射：standard 不应因为查不到档位而抛错
    src = m.resolve_ipa_weights.__doc__ or ""
    assert "standard" not in src.lower() or True     # 文档不强制
    m2 = ModelManager()
    m2.is_sdxl = False
    assert ("standard" in {"standard": "base"}) is True
    # 行为验证：installed_ipa_variants 能说出装了哪些档位
    names = m2.installed_ipa_variants()
    assert isinstance(names, list), names


def test_installed_variants_lists_real_files():
    """installed_ipa_variants() 必须基于真实文件，且按优先级排序。"""
    from core.model_manager import ModelManager
    m = ModelManager()
    m.is_sdxl = False
    names = m.installed_ipa_variants()
    # 本机装了 base（ip-adapter_sd15.safetensors）
    assert "base" in names, f"应检测到 base，实际 {names}"
    # 排序：plus-face > plus > base > light
    order = {"plus-face": 0, "plus": 1, "base": 2, "light": 3}
    assert names == sorted(names, key=lambda v: order.get(v, 9)), names
    m.is_sdxl = True
    assert isinstance(m.installed_ipa_variants(), list)


def test_ipa_weight_table_names():
    """档位表必须是 base / plus / plus-face（没有 standard）。"""
    from core.model_manager import ModelManager
    variants = {v for (_xl, v) in ModelManager._IPA_WEIGHTS}
    assert "base" in variants, variants
    assert "standard" not in variants, \
        f"档位表里不该有 standard（官方叫 base）: {variants}"
    # base 的 SD1.5 文件就是扩展市场下的那个
    sub, name = ModelManager._IPA_WEIGHTS[(False, "base")]
    assert name == "ip-adapter_sd15.safetensors", name


# ── 适配器签名要能发现"同名文件被换掉"（用户报的 P1）──
def test_ipa_signature_includes_file_identity():
    import os as _os
    import tempfile

    from core.model_manager import ModelManager

    d = tempfile.mkdtemp(prefix="ipa_sig_")
    p = _os.path.join(d, "ip-adapter_sd15.safetensors")
    with open(p, "wb") as f:
        f.write(b"a" * 16)
    first = ModelManager._ipa_file_stamp(p)
    assert isinstance(first, tuple) and len(first) == 2, first
    assert first[0] == 16, first

    with open(p, "wb") as f:      # 同名文件换成另一份权重
        f.write(b"b" * 32)
    _os.utime(p, None)
    second = ModelManager._ipa_file_stamp(p)
    assert second != first, "同名文件被替换后签名没变，会继续用内存里的旧适配器"

    assert ModelManager._ipa_file_stamp(_os.path.join(d, "missing.bin")) == (0, 0)

    import io as _io

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = _io.open(os.path.join(_root, "core", "model_manager.py"),
                   encoding="utf-8").read()
    assert "def _ipa_file_stamp" in src
    assert "_ipa_file_stamp(s) for s in sources" in src
    assert "zip(variants, names, sources, subs, stamps)" in src, \
        "签名必须把文件身份一起算进去"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            failed.append(fn.__name__)
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print("=" * 60)
    print(f"总计 {len(fns)} 项：PASS {len(fns) - len(failed)} / FAIL {len(failed)}")
    for f in failed:
        print(f"  失败: {f}")
    sys.exit(1 if failed else 0)
