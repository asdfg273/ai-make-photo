# tests/test_ipa_multi.py — 多参考图 IP-Adapter（无头，纯脚本，不跑 GPU）
#
# 这个文件锁住的都是**会静默失效**的行为：错了不报错，只是"效果不对"，
# 人工点界面时极难发现。最典型的三条：
#
#   1. 三个角色节点串起来必须变成**一次** IP-Adapter 应用。
#      如果实现成"每个节点装一遍"，最后一个会覆盖前面的
#      （`_load_ip_adapter_weights` 是重建 `encoder_hid_proj` 的覆盖语义），
#      症状是"三个参考图只有最后一个有效"，而且完全不报错。
#
#   2. 本机没装 plus-face / plus 时必须**降级**（用已安装的 base）并留中文
#      警告。若不降级而是失败，用户会看到"工作流跑不了"；若静默降级不留警告，
#      用户会以为"身份参考生效了"，其实角色语义根本没生效。
#
#   3. `refs_json` 解析必须容错：界面里手写 JSON 打错一个字符，
#      绝不能把整条工作流打崩（区域功能的 `regions_json` 就是同一套约定）。
#
# 用法：venv\Scripts\python.exe tests\test_ipa_multi.py
import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码，放最前）

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402


# ----------------------------------------------------------------------
#  替身
# ----------------------------------------------------------------------
class _Caps:
    """能力门禁替身：记下 require 被怎么调的。"""

    def __init__(self, ip_adapter=True):
        self.ip_adapter = bool(ip_adapter)
        self.calls = []

    def require(self, feature):
        self.calls.append(feature)
        if feature == "ip_adapter" and not self.ip_adapter:
            from workflow.adapters import UnsupportedFeatureError
            raise UnsupportedFeatureError("IP-Adapter", "anima", "测试替身")

    def supports(self, feature):
        return bool(self.ip_adapter) if feature == "ip_adapter" else True


class _FakeAi:
    """模型管理器替身：只记录"被要求装什么"。"""

    def __init__(self, installed=("base",)):
        self._installed = list(installed)
        self.multi_calls = []
        self.image_encoder = None

    def installed_ipa_variants(self):
        return list(self._installed)

    def prepare_ip_adapter_multi(self, refs):
        from core import ipa_multi as ipa_mod
        self.multi_calls.append(list(refs))
        plan = ipa_mod.MultiAdapterPlan(ok=True, count=len(refs))
        plan.variants = [r.variant or "base" for r in refs]
        plan.scales = [float(r.weight) for r in refs]
        plan.refs = list(refs)
        return plan

    def ipa_adapter_count(self, pipe=None):
        return len(self.multi_calls[-1]) if self.multi_calls else 0


def _ctx(installed=("base",), ip_adapter=True):
    from workflow.executor import NodeContext
    ai = _FakeAi(installed)
    ctx = NodeContext(ai=ai, capabilities=_Caps(ip_adapter))
    return ctx, ai


def _model():
    from workflow.values import ModelRef
    from workflow.adapters import get_adapter
    return ModelRef(name="fake", arch_id="sd15", adapter=get_adapter("sd15"),
                    loaded=True, meta={})


class _Node:
    def __init__(self, **params):
        self.id = "n"
        self.type = "X"
        self.params = dict(params)
        self.inputs = {}


def _img(color=(200, 30, 30), size=(64, 64)):
    return Image.new("RGB", size, color)


def _mask(size=(64, 64), box=(0, 0, 32, 64)):
    m = Image.new("L", size, 0)
    m.paste(255, box)
    return m


# ======================================================================
#  1) 角色 → 权重文件的映射
# ======================================================================
def test_role_variant_mapping_is_real():
    """角色映射必须指向 h94/IP-Adapter 里**真实存在**的文件。"""
    from core import ipa_multi as ipa_mod
    from core.model_manager import ModelManager

    assert ipa_mod.ROLE_VARIANTS == {
        "identity": "plus-face", "style": "plus", "composition": "base"}

    table = ipa_mod.role_table()
    # 身份 = plus-face / 风格 = plus / 构图 = base，且文件都出自 SD1.5 那一组
    assert table["identity"]["variant"] == "plus-face"
    assert table["style"]["variant"] == "plus"
    assert table["composition"]["variant"] == "base"
    for role, info in table.items():
        assert info["file"].endswith("_sd15.safetensors"), (role, info)
        # 档位名必须真的在 ModelManager 的表里（不能是自造的名字）
        assert (False, info["variant"]) in ModelManager._IPA_WEIGHTS, role

    assert table["identity"]["file"] == "ip-adapter-plus-face_sd15.safetensors"
    assert table["style"]["file"] == "ip-adapter-plus_sd15.safetensors"
    assert table["composition"]["file"] == "ip-adapter_sd15.safetensors"

    # 档位表里没有 standard（界面下拉的别名由 normalize_variant 处理）
    assert "standard" not in {v for (_xl, v) in ModelManager._IPA_WEIGHTS}
    assert ipa_mod.normalize_variant("standard") == "base"
    assert ipa_mod.normalize_variant("plus_face") == "plus-face"
    # 中文角色名也要认
    assert ipa_mod.normalize_role("身份") == "identity"
    assert ipa_mod.normalize_role("风格") == "style"
    assert ipa_mod.normalize_role("构图") == "composition"
    assert ipa_mod.normalize_role("") == ipa_mod.DEFAULT_ROLE
    assert ipa_mod.normalize_role("乱写") == ipa_mod.DEFAULT_ROLE


# ======================================================================
#  2) 三个角色节点串起来 = 一次应用（累加）
# ======================================================================
def test_role_nodes_accumulate_into_one_application():
    """链式串接三个角色节点 → 一次安装，三个适配器各自一个强度。"""
    from nodes import ipa_nodes
    ctx, ai = _ctx(installed=("base",))
    model = _model()

    seq = [
        ("identity", 0.8, (10, 10, 10)),
        ("style", 0.6, (200, 30, 30)),
        ("composition", 0.4, (30, 30, 200)),
    ]
    for i, (role, weight, color) in enumerate(seq):
        fn = getattr(ipa_nodes, f"run_ip_adapter_{role}")
        out = fn(ctx, _Node(weight=weight),
                 {"model": model, "image": _img(color)})
        assert out["model"] is model
        # 每个角色节点都只**登记**，不安装（安装发生在采样前一刻）
        assert ai.multi_calls == [], \
            "角色节点不应该在自身里安装权重（会在链式串接时互相覆盖）"
        refs = ipa_nodes.pending_references(model)
        assert len(refs) == i + 1, refs

    refs = ipa_nodes.pending_references(model)
    assert [r.role for r in refs] == ["identity", "style", "composition"]
    assert [round(r.weight, 3) for r in refs] == [0.8, 0.6, 0.4]

    # 采样前一刻才安装：一次调用、三个适配器
    from workflow.executor import NodeContext
    from nodes.sampling_nodes import apply_ip_adapter_kwargs
    ctx2 = NodeContext(ai=ai, capabilities=_Caps())
    kw = {}
    kw, info = apply_ip_adapter_kwargs(ctx2, model, kw, width=512, height=512)
    assert len(ai.multi_calls) == 1, f"应当只安装一次，实际 {len(ai.multi_calls)} 次"
    assert len(ai.multi_calls[0]) == 3
    assert info["ip_adapter_refs"] == 3
    # ip_adapter_image 是**列表**，与适配器数量一一对应（diffusers 的硬要求）
    imgs = kw["ip_adapter_image"]
    assert isinstance(imgs, list) and len(imgs) == 3
    assert all(isinstance(im, Image.Image) for im in imgs)

    # 元数据要如实列出两条以上参考图（GPU 验证脚本也断言这一点）
    assert ctx2.metadata["ip_adapter_roles"] == ["identity", "style",
                                                 "composition"]


def test_masked_reference_adds_attention_mask():
    """带 mask 的参考图要产出 ip_adapter_masks（并且不改调用方的图）。"""
    from nodes import ipa_nodes
    from nodes.sampling_nodes import apply_ip_adapter_kwargs
    from workflow.executor import NodeContext

    ctx, ai = _ctx()
    model = _model()
    src = _img()
    before = src.tobytes()
    fn = ipa_nodes.run_ip_adapter_style
    fn(ctx, _Node(weight=0.6),
       {"model": model, "image": src, "mask": _mask()})
    assert src.tobytes() == before, "节点绝不能修改调用方传进来的 PIL 图"

    ctx2 = NodeContext(ai=ai, capabilities=_Caps())
    kw, _info = apply_ip_adapter_kwargs(ctx2, model, {}, width=512, height=512)
    masks = kw.get("cross_attention_kwargs", {}).get("ip_adapter_masks")
    assert isinstance(masks, list) and len(masks) == 1
    t = masks[0]
    assert tuple(t.shape) == (1, 1, 512 // 8, 512 // 8), tuple(t.shape)
    assert float(t.max()) > 0.5 and float(t.min()) < 0.5   # 真的有分区


def test_disabled_reference_uses_placeholder():
    """停用/零权重的参考图仍要占位（长度必须等于适配器数量）。"""
    from core import ipa_multi as ipa_mod
    from nodes.sampling_nodes import apply_ip_adapter_kwargs
    from workflow.executor import NodeContext

    ctx, ai = _ctx()
    model = _model()
    model.meta["ipa_refs"] = [
        ipa_mod.IpaReference(role="style", weight=0.0, image=_img()),
    ]
    ctx2 = NodeContext(ai=ai, capabilities=_Caps())
    kw, info = apply_ip_adapter_kwargs(ctx2, model, {}, width=256, height=256)
    imgs = kw["ip_adapter_image"]
    assert len(imgs) == 1 and imgs[0].size == (1, 1)
    assert info["ip_adapter_placeholder_refs"] == 1

# ======================================================================
#  3) 优雅降级：缺档位 → 警告 + 不抛
# ======================================================================
def test_graceful_degradation_records_warning():
    """本机只有 base 时，身份/风格都要降级到 base，并留中文警告。"""
    from nodes import ipa_nodes
    ctx, ai = _ctx(installed=("base",))
    model = _model()

    out = ipa_nodes.run_ip_adapter_identity(
        ctx, _Node(weight=0.8), {"model": model, "image": _img()})
    warns = ctx.metadata.get("warnings") or []
    assert warns, "降级必须留下警告（不能静默）"
    assert any("plus-face" in w and "base" in w for w in warns), warns
    assert out["metadata"]["ip_adapter_refs"] == 1

    # 没有异常 = 通过（降级不失败）
    ctx2, _ai2 = _ctx(installed=("plus-face", "plus", "base"))
    model2 = _model()
    ipa_nodes.run_ip_adapter_identity(
        ctx2, _Node(weight=0.8), {"model": model2, "image": _img()})
    assert not (ctx2.metadata.get("warnings") or []), "装了就不该有降级警告"


def test_no_weights_at_all_raises_clear_chinese_error():
    """一个档位都没装 → 抛中文错误（既不静默，也不去联网瞎等）。"""
    from core import ipa_multi as ipa_mod
    plan, warns = None, None
    try:
        ipa_mod.resolve_role_plan([{"role": "style", "weight": 0.6}], [])
        raise AssertionError("没有任何权重时应当抛错")
    except RuntimeError as e:
        assert "IP-Adapter" in str(e)
        assert "models/ip_adapter" in str(e), str(e)


def test_resolve_ipa_weights_is_fast_and_local_only():
    """`resolve_ipa_weights` 找不到档位时必须**毫秒级**失败且不联网、不扫缓存。

    这是被实测钉住的 P0，有两个耗时来源：

      · `_get_model_file(local_files_only=False)`（旧实现让 diffusers 自己
        在线下载）实测 **295.6 秒**；父任务量到的 126s 是同一阶段，
        随网络状况浮动。
      · `hf_hub_download(..., local_files_only=True)` 在**缓存未命中**时
        实测 **6.2 秒**（要把整个 HF 缓存目录扫一遍才能确定"真的没有"）。

    所以这里既断言"不许联网回退"，也断言"缓存命中走纯文件系统查找"。
    """
    import inspect
    import time
    from core.model_manager import ModelManager

    m = ModelManager()
    m.is_sdxl = False
    # 请求一个本机一定没有的档位（plus-face 未安装）
    t0 = time.time()
    try:
        m.resolve_ipa_weights("plus-face")
    except RuntimeError as e:
        assert "ip-adapter-plus-face_sd15.safetensors" in str(e), str(e)
    dt = time.time() - t0
    assert dt < 5.0, f"解析不该耗时 {dt:.2f}s（说明又去联网/扫缓存了）"

    src = inspect.getsource(ModelManager.resolve_ipa_weights)
    assert 'return "h94/IP-Adapter", subfolder, weight_name' not in src, \
        "不允许再返回在线仓库让 diffusers 自己去下载"
    # 只允许在注释/文档里提到它，不允许真的**调用**
    import re as _re
    code = _re.sub(r'""".*?"""', "", src, flags=_re.S)
    code = "\n".join(l for l in code.splitlines()
                     if not l.strip().startswith("#"))
    assert "hf_hub_download" not in code, \
        "缓存命中要用纯文件系统查找（hf_hub_download 未命中时实测 6.2s）"
    assert "_ipa_hf_cache_hit" in code, "缓存命中路径必须走 _ipa_hf_cache_hit"


def test_local_image_encoder_is_required():
    """只有 44MB 权重、没有 2.4GB 编码器时不能算"装了"。"""
    from core.model_manager import ModelManager
    m = ModelManager()
    m.is_sdxl = False
    enc = m._ipa_local_image_encoder()
    # 本机确实有 image_encoder（扩展市场装的）
    assert enc is not None, (
        "本机 models/ip_adapter/image_encoder 应当存在；"
        "没有的话 installed_ipa_variants() 会返回空")
    assert os.path.isfile(os.path.join(enc, "config.json"))
    # 换一个不存在的目录必须返回 None（并且不联网、不抛）
    assert m._ipa_local_image_encoder("Z:/绝对不存在的目录") is None


# ======================================================================
#  4) JSON 容错 与 往返
# ======================================================================
def test_refs_json_parsing_is_tolerant():
    """界面手写 JSON 打错字 → 空列表 + 提示，绝不抛异常。"""
    from core import ipa_multi as ipa_mod
    from nodes.adapter_nodes import run_ip_adapter_apply

    for junk in (None, "", "   ", "[]", "{}", "not json", "{",
                 '{"a":1}', "[1,2,3]", "[[[", "null"):
        assert ipa_mod.references_from_json(junk) == [], junk

    # 单个对象也认（不是数组）
    one = ipa_mod.references_from_json(
        '{"role":"identity","path":"E:/a.png","weight":0.7}')
    assert len(one) == 1 and one[0].role == "identity"
    # 外层包装（界面导出格式）
    wrapped = ipa_mod.references_from_json(
        '{"refs":[{"role":"style","path":"a.png"}]}')
    assert len(wrapped) == 1 and wrapped[0].role == "style"
    # 权重越界被夹取而不是报错
    assert ipa_mod.references_from_json(
        '[{"path":"a.png","weight":9}]')[0].weight == 2.0
    assert ipa_mod.references_from_json(
        '[{"path":"a.png","weight":-3}]')[0].weight == 0.0
    # 缺字段走默认
    r = ipa_mod.references_from_json('[{"path":"a.png"}]')[0]
    assert r.role == ipa_mod.DEFAULT_ROLE
    assert abs(r.weight - ipa_mod.DEFAULT_WEIGHT) < 1e-9

    # 节点层：坏 JSON + 没有连线图 → 抛的是"缺参考图"，不是 JSONDecodeError
    ctx, _ai = _ctx()
    try:
        run_ip_adapter_apply(ctx, _Node(refs_json="这不是 JSON"), {})
        raise AssertionError("应当抛错（没有模型）")
    except ValueError as e:
        assert "model" in str(e), str(e)

    ctx, _ai = _ctx()
    model = _model()
    try:
        run_ip_adapter_apply(ctx, _Node(refs_json="这不是 JSON"),
                             {"model": model})
        raise AssertionError("坏 JSON + 没有图 应当抛「缺少参考图」")
    except ValueError as e:
        assert "参考图" in str(e), str(e)


def test_refs_json_round_trips():
    """references_to_json → references_from_json 必须完全还原。"""
    from core import ipa_multi as ipa_mod
    import json

    refs = [
        ipa_mod.IpaReference(role="identity", weight=0.8, path="E:/a.png"),
        ipa_mod.IpaReference(role="style", weight=0.55, path="E:/b.png"),
        ipa_mod.IpaReference(role="composition", weight=0.4, path="E:/c.png",
                             mask_path="E:/m.png", enabled=False),
    ]
    text = ipa_mod.references_to_json(refs)
    assert json.loads(text)[0]["role"] == "identity"
    back = ipa_mod.references_from_json(text)
    assert len(back) == 3
    for a, b in zip(refs, back):
        assert a.role == b.role
        assert abs(a.weight - b.weight) < 1e-9
        assert a.path == b.path
        assert a.mask_path == b.mask_path
        assert bool(a.enabled) == bool(b.enabled)


def test_ip_adapter_apply_accepts_multiple_refs():
    """IPAdapterApply 的 refs_json + 连线单图会合并（向后兼容）。"""
    from nodes.adapter_nodes import run_ip_adapter_apply
    from nodes import ipa_nodes

    ctx, _ai = _ctx()
    model = _model()
    out = run_ip_adapter_apply(
        ctx,
        _Node(refs_json='[{"role":"identity","path":"E:/a.png","weight":0.7},'
                        '{"role":"composition","path":"E:/b.png","weight":0.3}]',
              variant="plus", weight=0.6),
        {"model": model, "image": _img()})
    refs = ipa_nodes.pending_references(model)
    assert [r.role for r in refs] == ["identity", "composition", "style"]
    assert out["metadata"]["ip_adapter_refs"] == 3
    # 单图模式（只有 image 连线）仍可用 → 兼容老工作流
    ctx2, _ai2 = _ctx()
    model2 = _model()
    run_ip_adapter_apply(ctx2, _Node(variant="standard", weight=0.5),
                         {"model": model2, "image": _img()})
    refs2 = ipa_nodes.pending_references(model2)
    assert len(refs2) == 1 and refs2[0].role == "style"
    assert abs(refs2[0].weight - 0.5) < 1e-9


# ======================================================================
#  5) 节点注册 / 能力门禁 / 起点清零
# ======================================================================
def test_nodes_registered_with_required_ports():
    from workflow import registry
    registry.load_builtin_nodes()
    for t in ("IPAdapterIdentity", "IPAdapterStyle", "IPAdapterComposition"):
        assert registry.exists(t), t
        spec = registry.get(t)
        assert spec.category == "适配", spec.category
        assert set(spec.inputs) == {"model", "image", "mask"}
        assert spec.inputs["model"].optional is False
        assert spec.inputs["image"].optional is False
        assert spec.inputs["mask"].optional is True
        assert set(spec.outputs) == {"model", "metadata"}
        for p in ("weight", "start_at", "end_at", "enabled"):
            assert p in spec.params, (t, p)
        for p in ("weight", "start_at", "end_at", "enabled"):
            ps = spec.params[p]
            assert ps.label and ps.help, f"{t}.{p} 需要中文 label/help"
        assert spec.params["weight"].minimum == 0.0
        assert spec.params["weight"].maximum == 2.0
        assert spec.params["start_at"].minimum == 0.0
        assert spec.params["end_at"].maximum == 1.0


def test_capability_gate_not_model_name():
    """必须用 caps.require("ip_adapter") 门禁；不支持的架构要报清楚。"""
    from nodes import ipa_nodes
    import inspect
    src = inspect.getsource(ipa_nodes.add_reference)
    assert 'caps.require("ip_adapter")' in src, \
        "必须走能力门禁（caps.require），不能判断模型名"

    ctx, _ai = _ctx(ip_adapter=False)
    try:
        ipa_nodes.run_ip_adapter_style(
            ctx, _Node(weight=0.6), {"model": _model(), "image": _img()})
        raise AssertionError("不支持 ip_adapter 的架构应当报错")
    except Exception as e:
        assert "不支持" in str(e) or "IP-Adapter" in str(e), str(e)


def test_checkpoint_clears_references():
    """LoadCheckpoint 必须清掉上一轮的参考图（否则会串味）。"""
    from nodes import ipa_nodes
    ctx, _ai = _ctx()
    model = _model()
    ipa_nodes.run_ip_adapter_style(ctx, _Node(weight=0.6),
                                  {"model": model, "image": _img()})
    assert ipa_nodes.pending_references(model)
    n = ipa_nodes.clear_references(model)
    assert n == 1
    assert ipa_nodes.pending_references(model) == []
    # 源码级保险：LoadCheckpoint 里确实调了它
    import inspect
    from nodes import model_nodes
    src = inspect.getsource(model_nodes.run_load_checkpoint)
    assert "clear_references" in src, \
        "LoadCheckpoint 必须清空累积参考图（与 LoRA 同一类串味问题）"


def test_sampler_guards_missing_ip_adapter_image():
    """装了 IPA 却没有参考图 → 必须给中文错误，而不是 diffusers 的 TypeError。

    实测（本机 GPU）：装好 base 后不带 ip_adapter_image 调管线 →
        TypeError: argument of type 'NoneType' is not iterable
        （unet_2d_condition.py:965，added_cond_kwargs 是 None）
    这条断言把"我们自己的检查"钉在源码里。
    """
    import inspect
    from nodes import sampling_nodes
    src = inspect.getsource(sampling_nodes.apply_ip_adapter_kwargs)
    assert "MultiIPAdapterImageProjection" in src
    assert "每次调用都必须提供参考图" in src

    from nodes.sampling_nodes import apply_ip_adapter_kwargs
    from workflow.executor import NodeContext

    class _P:
        class _U:
            def __init__(self):
                self.encoder_hid_proj = type("X", (), {})()
        def __init__(self):
            self.unet = self._U()

    class _Ai:
        def __init__(self):
            self.txt2img_pipe = _P()
            self.img2img_pipe = None
            self.inpaint_pipe = None
            self.ip_adapter_loaded = True
            self._proj = None

    ai = _Ai()
    # 把 encoder_hid_proj 换成一个"名字像"的对象
    class MultiIPAdapterImageProjection:      # noqa: N801  （故意同名）
        pass
    ai.txt2img_pipe.unet.encoder_hid_proj = MultiIPAdapterImageProjection()
    ctx = NodeContext(ai=ai, capabilities=_Caps())
    try:
        apply_ip_adapter_kwargs(ctx, _model(), {}, width=512, height=512)
        raise AssertionError("应当抛中文错误")
    except RuntimeError as e:
        assert "参考图" in str(e) and "TypeError" in str(e), str(e)


# ======================================================================
#  6) 区域（独立参考图区域）：core/regional 的 ip_adapter 钩子
# ======================================================================
def test_region_ip_adapter_hook_exists():
    """区域必须支持 ip_adapter 钩子（独立参考图区域）。"""
    from core import regional

    r = regional.Region.from_dict({
        "name": "衣服", "prompt": "red dress",
        "ip_adapter": {"refs": [{"role": "style", "path": "E:/cloth.png",
                                 "weight": 0.6}]},
    })
    cfg = r.merged_ip_adapter()
    assert cfg and len(cfg["refs"]) == 1
    assert r.to_dict()["ip_adapter"]["refs"][0]["role"] == "style"
    # 单条写法也要认
    r2 = regional.Region.from_dict({
        "ip_adapter": {"role": "identity", "path": "E:/face.png"}})
    assert r2.merged_ip_adapter()["refs"][0]["role"] == "identity"
    # 没配 → None
    assert regional.Region.from_dict({}).merged_ip_adapter() is None

    # 该区域的参考图要**带上本区域蒙版**
    from core.regional import _region_ip_adapter_refs
    mask = _mask()
    refs = _region_ip_adapter_refs(r, mask)
    assert len(refs) == 1 and refs[0].mask is mask

    import inspect
    src = inspect.getsource(regional._region_sampler)
    assert "_apply_region_ip_adapter" in src, \
        "区域采样必须调用局部 IP-Adapter 钩子"


def test_region_ip_adapter_pads_to_adapter_count():
    """局部参考图数量必须补齐到适配器数量，否则 diffusers 会抛长度错误。"""
    from core.model_manager import ModelManager
    m = ModelManager()

    class _Proj:
        num_ip_adapters = 3

    class _U:
        encoder_hid_proj = _Proj()

    class _Pipe:
        unet = _U()

    m.txt2img_pipe = _Pipe()
    assert m.ipa_adapter_count() == 3

    kw, info = m.build_ip_adapter_kwargs(
        [{"role": "style", "path": "E:/不存在.png", "weight": 0.6}],
        width=256, height=256, has_masks=False)
    assert len(kw["ip_adapter_image"]) == 3, len(kw.get("ip_adapter_image") or [])
    assert info["scales"] == [0.0, 0.0, 0.0], info["scales"]   # 读不到 → 占位强度 0
    assert any("占位" in n for n in info["notes"]), info["notes"]

    # ⚠️ **没有**参考图也必须给占位列表：只要 UNet 上装了 IPA，每次调用都得给，
    #    否则 diffusers 抛 "ip_adapter_image must have same length as the
    #    number of IP Adapters"（实测）。
    kw0, info0 = m.build_ip_adapter_kwargs(None, width=256, height=256)
    assert len(kw0["ip_adapter_image"]) == 3, len(kw0.get("ip_adapter_image") or [])
    assert info0["scales"] == [0.0, 0.0, 0.0], info0["scales"]
    # 没装 IPA 时不该凭空造参数（真机上有真的管线，所以用一个"干净"替身）
    m2 = ModelManager()
    class _NoIpaPipe:
        unet = None
    m2.txt2img_pipe = _NoIpaPipe()
    assert m2.ipa_adapter_count() == 0
    assert m2.build_ip_adapter_kwargs(None) == ({}, {})


def test_region_scale_is_restored_after_sampling():
    """区域采样后必须把 IPA 强度还回去（否则后面区域会继承）。"""
    from core import regional

    class _Proc:
        scale = [0.5, 0.5]

    class _Unet:
        attn_processors = {"down_blocks.0.attn2": _Proc()}

    class _Pipe:
        def __init__(self):
            self.unet = _Unet()

        def set_ip_adapter_scale(self, s):
            _Unet.attn_processors["down_blocks.0.attn2"].scale = (
                list(s) if isinstance(s, (list, tuple)) else [s])

    p = _Pipe()
    kw = {"_ipa_restore": [(p, regional.read_ipa_scales(p))]}
    assert regional.read_ipa_scales(p) == [0.5, 0.5]
    # 采样期间被改成别的值
    p.set_ip_adapter_scale([0.9, 0.1])
    assert p.unet.attn_processors["down_blocks.0.attn2"].scale == [0.9, 0.1]
    n = regional.restore_region_ip_adapter(kw, None)
    assert n == 1, n
    assert p.unet.attn_processors["down_blocks.0.attn2"].scale == [0.5, 0.5]
    assert "_ipa_restore" not in kw          # 不要留在 kwargs 里传给管线

    import inspect
    src = inspect.getsource(regional._region_sampler)
    assert "restore_region_ip_adapter" in src, \
        "采样后必须恢复 IPA 强度（否则区域之间互相污染）"


def test_topo_sort_respects_inputs_declared_inline():
    """拓扑序必须认"直接写进 inputs 的连线"（不依赖 wf.connect）。

    这是做多参考图 GPU 验证时踩出来的**真实 bug**：
    `topo_sort` 只读 `wf.links`，而 `wf.links` 只在调用 `wf.connect()` 时才
    被填。于是"用 `Node(inputs={...})` 声明连线"的工作流所有入度都是 0，
    顺序退化成**节点插入顺序**。实测：

        ckpt → pos/neg → sampler → vae      （期望顺序）
        实际 topo_sort: ['ckpt','pos','neg','vae','sampler']

    VAEDecode 被排到 Sampler 前面 → 拿不到 latent →
    `ValueError: 缺少 latent（也没有可透传的 image）`，
    而报错完全指不到真正的原因（顺序）。
    """
    from workflow import graph as G
    from workflow.schema import Node, Workflow

    wf = Workflow(name="order")
    wf.add(Node("ckpt", "LoadCheckpoint", {}, {"path": "x"}))
    wf.add(Node("pos", "TextEncode", {"model": ["ckpt", "model"]}, {"text": "a"}))
    wf.add(Node("sampler", "Sampler",
                {"model": ["ckpt", "model"], "positive": ["pos", "conditioning"]},
                {}))
    # 故意把 vae 放在 sampler **之后**声明，但它的上游是 sampler
    wf.add(Node("vae", "VAEDecode",
                {"model": ["ckpt", "model"], "latent": ["sampler", "latent"]},
                {}))
    order = G.topo_sort(wf)
    assert order.index("ckpt") < order.index("pos")
    assert order.index("pos") < order.index("sampler"), order
    assert order.index("sampler") < order.index("vae"), order

    # 反过来声明（上游在后）也要正确
    wf2 = Workflow(name="order2")
    wf2.add(Node("vae", "VAEDecode",
                 {"model": ["ckpt", "model"], "latent": ["sampler", "latent"]},
                 {}))
    wf2.add(Node("sampler", "Sampler",
                 {"model": ["ckpt", "model"], "positive": ["pos", "conditioning"]},
                 {}))
    wf2.add(Node("ckpt", "LoadCheckpoint", {}, {"path": "x"}))
    wf2.add(Node("pos", "TextEncode", {"model": ["ckpt", "model"]},
                 {"text": "a"}))
    order2 = G.topo_sort(wf2)
    assert order2.index("ckpt") < order2.index("pos") < order2.index("sampler")
    assert order2.index("sampler") < order2.index("vae"), order2

    # connect() 与内联 inputs 混用不能把同一条边算两次（否则入度虚高，
    # 会出现"明明没有环却报环路"）
    wf3 = Workflow(name="order3")
    wf3.add(Node("a", "LoadCheckpoint", {}, {"path": "x"}))
    wf3.add(Node("b", "TextEncode", {"model": ["a", "model"]}, {"text": "t"}))
    wf3.connect("a", "model", "b", "model")
    assert G.topo_sort(wf3) == ["a", "b"], G.topo_sort(wf3)

    # upstream_of / needed_for 也要认内联连线（只跑子图时不能漏上游）
    assert "sampler" in G.upstream_of(wf, "vae")
    assert set(G.needed_for(wf, ["vae"])) == {"ckpt", "pos", "sampler", "vae"}


def test_degradation_warning_is_not_duplicated():
    """同一个"缺 plus-face"不能在元数据里出现两次（节点一条、采样一条）。"""
    from nodes import ipa_nodes
    from nodes.sampling_nodes import apply_ip_adapter_kwargs
    from workflow.executor import NodeContext

    ctx, ai = _ctx(installed=("base",))
    model = _model()
    ipa_nodes.run_ip_adapter_identity(ctx, _Node(weight=0.8),
                                      {"model": model, "image": _img()})
    ipa_nodes.run_ip_adapter_style(ctx, _Node(weight=0.6),
                                   {"model": model, "image": _img()})
    ctx2 = NodeContext(ai=ai, capabilities=_Caps())
    # 采样层的 ctx 与角色节点的 ctx 是**同一个**（工作流里就一个 ctx），
    # 所以这里的 metadata 必须原样带过去（包括 ipa_degraded）。
    ctx2.metadata.update(ctx.metadata)
    apply_ip_adapter_kwargs(ctx2, model, {}, width=256, height=256)
    warns = ctx2.metadata.get("warnings") or []
    assert sum(1 for w in warns if "plus-face" in w) == 1, warns
    assert sum(1 for w in warns if "「plus」" in w) == 1, warns


def test_regional_regression_still_holds():
    """区域的核心不变量不能被这次改动带偏。"""
    from core.regional import coerce_regions, region_masks, validate_regions
    import numpy as np
    regs = coerce_regions([{"name": "左", "prompt": "a red apple",
                            "rect": (0.25, 0.25, 0.75, 0.75), "weight": 0.7}])
    m = np.asarray(region_masks(regs, 512, 512)[0][1].convert("L"))
    assert m[256, 256] > 0 and m[5, 5] == 0
    assert validate_regions(regs, 512, 512) is not None


# ======================================================================
#  共享流程（主界面 / 工作流共用一份实现）
# ======================================================================
def test_enabled_refs_handles_dicts_and_objects():
    """`enabled_refs` 必须同时吃得下 dict 与 IpaReference，缺省算启用。

    这条原来散在 4 处、两种写法（`.get("enabled")` / `getattr(...)`）。
    漏一处 = "界面上停用了，生成时仍然生效"。
    """
    from core import ipa_multi as ipa_mod

    dicts = [{"role": "style", "enabled": True},
             {"role": "identity", "enabled": False},
             {"role": "composition"}]
    assert len(ipa_mod.enabled_refs(dicts)) == 2

    refs = ipa_mod.coerce_references([
        {"role": "style", "enabled": True},
        {"role": "identity", "enabled": False},
    ])
    on = ipa_mod.enabled_refs(refs)
    assert len(on) == 1 and getattr(on[0], "role", None) == "style"
    assert ipa_mod.enabled_refs([]) == [] == ipa_mod.enabled_refs(None)


def test_collect_reference_images_keeps_one_entry_per_adapter():
    """权重 ≤0 / 读不到都要留 1x1 占位图 —— 列表长度必须等于适配器数量。

    数量塌缩的后果不是报错，而是"每张参考图对错了适配器"：图像内容与
    权重文件错位，画面看起来只是"不像参考图"，极难归因。
    """
    from core import ipa_multi as ipa_mod

    refs = ipa_mod.coerce_references([
        {"role": "style", "weight": 0.8, "path": ""},      # 读不到 → 占位
        {"role": "identity", "weight": 0.0},               # 权重 0 → 占位
        {"role": "composition", "weight": 0.5},            # 无图无路径 → 占位
    ])
    imgs, ph = ipa_mod.collect_reference_images(
        refs, on_missing="degrade", log=lambda m: None)
    assert len(imgs) == 3 == len(refs), (len(imgs), len(refs))
    assert ph == 3
    assert all(im.size == (1, 1) for im in imgs)


def test_collect_reference_images_degrade_vs_raise():
    """同一份代码，两种失败策略：主界面降级、工作流报错。"""
    from core import ipa_multi as ipa_mod

    refs = ipa_mod.coerce_references([
        {"role": "style", "weight": 0.9, "path": "Z:/definitely/not/here.png"}])
    imgs, ph = ipa_mod.collect_reference_images(
        refs, on_missing="degrade", log=lambda m: None)
    assert len(imgs) == 1 and ph == 1
    assert refs[0].weight == 0.0, "降级必须把权重归零，否则后续 scale 仍是 0.9"

    refs2 = ipa_mod.coerce_references([
        {"role": "style", "weight": 0.9, "path": "Z:/definitely/not/here.png"}])
    try:
        ipa_mod.collect_reference_images(refs2, on_missing="raise")
        raise AssertionError("on_missing='raise' 应该抛 RuntimeError")
    except RuntimeError as e:
        assert "参考图读不到" in str(e)


def test_collect_reference_images_applies_mask():
    """带掩码的参考图，掩码外内容不能进 CLIP（否则"区域参考"等于整图参考）。"""
    from core import ipa_multi as ipa_mod

    ref = ipa_mod.coerce_references([{"role": "style", "weight": 0.8}])[0]
    ref.image = _img()                              # 内存图（不走磁盘）
    ref.mask = _mask((64, 64), (0, 0, 32, 64))      # 左半保留
    imgs, ph = ipa_mod.collect_reference_images([ref], on_missing="degrade")
    assert ph == 0, "有图就应该真的出图，不该是占位图"
    px = imgs[0].convert("RGB").load()
    assert px[5, 32] == (200, 30, 30), px[5, 32]        # 掩码内：原色
    assert px[60, 32] != (200, 30, 30), px[60, 32]      # 掩码外：背景色


def test_plan_masks_uses_pipe_scale():
    """掩码要按管线真实 downscale 算：倍率错了 = 掩码只覆盖 1/N 区域。"""
    from core import ipa_multi as ipa_mod

    class _Cfg8:
        vae_scale_factor = 8

    class _Vae8:
        config = _Cfg8()

    class _Pipe8:
        vae = _Vae8()

    class _Cfg4:
        vae_scale_factor = 4

    class _Vae4:
        config = _Cfg4()

    class _Pipe4:
        vae = _Vae4()

    ref = ipa_mod.coerce_references([{"role": "style", "weight": 1.0}])[0]
    ref.mask = _mask((64, 64), (0, 0, 32, 32))

    assert ipa_mod.vae_scale_factor(_Pipe8()) == 8
    assert ipa_mod.vae_scale_factor(_Pipe4()) == 4
    m8 = ipa_mod.plan_masks([ref], 512, 512, pipe=_Pipe8())[0]
    m4 = ipa_mod.plan_masks([ref], 512, 512, pipe=_Pipe4())[0]
    assert tuple(m8.shape) == (1, 1, 64, 64), tuple(m8.shape)
    assert tuple(m4.shape) == (1, 1, 128, 128), tuple(m4.shape)

    # 没有掩码时返回 None（调用方就完全不传 cross_attention_kwargs）
    plain = ipa_mod.coerce_references([{"role": "style", "weight": 1.0}])
    assert ipa_mod.plan_masks(plain, 512, 512, pipe=_Pipe8()) is None


def _fake_ai(installed=("base",), ok=True, error=""):
    """够 `build_runtime` 用的最小 ModelManager 替身。"""
    from core import ipa_multi as ipa_mod

    class _AI:
        def installed_ipa_variants(self):
            return list(installed)

        def prepare_ip_adapter_multi(self, refs):
            plan = ipa_mod.MultiAdapterPlan()
            plan.refs = ipa_mod.coerce_references(refs)
            plan.scales = [float(r.weight) for r in plan.refs]
            plan.variants = [r.variant for r in plan.refs]
            plan.encoder_dir = "models/ip_adapter/image_encoder"
            plan.count = len(plan.refs)
            plan.ok = bool(ok)
            plan.error = error
            return plan

    return _AI()


def test_build_runtime_bundles_everything():
    """七步合一：一次调用就拿到图 / 强度 / 元数据。"""
    from core import ipa_multi as ipa_mod

    img = _img()
    refs = [ipa_mod.IpaReference(role="style", variant="plus",
                                image=img, weight=0.6)]
    rt = ipa_mod.build_runtime(refs, _fake_ai(installed=("plus",)),
                              width=512, height=768)
    assert rt.ok, rt.error
    assert len(rt.images) == 1, "每张参考图都要有一张图（长度 == 适配器数）"
    assert rt.scales == [0.6]
    assert rt.info["ip_adapter_refs"] == 1
    assert rt.info["ip_adapter_weights"] == [0.6]
    assert rt.info["ip_adapter_enabled"] == "true"
    assert rt.info["ip_adapter_summary"]
    assert rt.encoder_dir


def test_build_runtime_needs_manager():
    """没有 ModelManager 时给出中文原因，而不是抛到界面上。"""
    from core import ipa_multi as ipa_mod

    rt = ipa_mod.build_runtime([ipa_mod.IpaReference(role="style",
                                                    image=_img())], None)
    assert not rt.ok
    assert "模型管理器" in rt.error


def test_build_runtime_reports_install_failure():
    """安装失败要把原因原样带出来（两条路决定是抛还是降级）。"""
    from core import ipa_multi as ipa_mod

    rt = ipa_mod.build_runtime(
        [ipa_mod.IpaReference(role="style", image=_img())],
        _fake_ai(ok=False, error="没装 base 权重"))
    assert not rt.ok
    assert "没装 base 权重" in rt.error


def test_build_runtime_degrades_missing_image():
    """主界面语义：一张参考图读不到 → 该条按 0 强度占位，不毁整次生成。"""
    from core import ipa_multi as ipa_mod

    ref = ipa_mod.IpaReference(role="style", path="不存在的参考图.png", weight=0.8)
    rt = ipa_mod.build_runtime([ref], _fake_ai(), on_missing="degrade")
    assert rt.ok, rt.error
    assert len(rt.images) == 1, "读不到也要留占位图，长度不能塌缩"
    assert rt.placeholders == 1
    assert rt.scales == [0.0], "读不到的条目强度应被置 0"


def test_build_runtime_can_raise_on_missing():
    """工作流语义：要的图给不了就说清楚（抛中文 RuntimeError）。"""
    from core import ipa_multi as ipa_mod

    ref = ipa_mod.IpaReference(role="style", path="不存在.png")
    try:
        ipa_mod.build_runtime([ref], _fake_ai(), on_missing="raise")
    except RuntimeError as e:
        assert "参考图读不到" in str(e)
    else:
        raise AssertionError("on_missing='raise' 时必须抛错")


def test_both_paths_use_build_runtime():
    """源码级：七步流程只准有一份实现（两条路各留薄封装）。

    这条是"以后有人又在某一边复制一份七步流程"的护栏 —— 审计第 9 条说的
    就是这个漂移风险。project 里 output_dir / LoRA-CN-IPA 读数 / 超分签名
    都因为"两套实现"吃过亏。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "utils", "app_events.py"), encoding="utf-8") as f:
        ui_src = f.read()
    with open(os.path.join(root, "nodes", "sampling_nodes.py"),
              encoding="utf-8") as f:
        node_src = f.read()

    for src, name in ((ui_src, "app_events"), (node_src, "sampling_nodes")):
        # 只看代码、不看注释：注释里提到这些函数名是正常的（说明"以前自己装"）。
        code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
        assert "build_runtime(" in code, f"{name} 没走共享的 build_runtime"
        assert "prepare_ip_adapter_multi(" not in code, \
            f"{name} 又自己装适配器了（应交给 build_runtime）"
        assert "collect_reference_images(" not in code, \
            f"{name} 又自己收参考图了（应交给 build_runtime）"
        assert "plan_masks(" not in code, \
            f"{name} 又自己算掩码了（应交给 build_runtime）"
        assert 'Image.new("RGB", (1, 1)' not in code, \
            f"{name} 又自己拼占位图了（应交给 collect_reference_images）"

    # 倍率助手也只留一份实现（节点那份是转发）
    assert "ipa_mod.vae_scale_factor(pipe)" in node_src


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            import traceback
            failed.append(fn.__name__)
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("=" * 60)
    print(f"总计 {len(fns)} 项：PASS {len(fns) - len(failed)} / FAIL {len(failed)}")
    for f in failed:
        print(f"  失败: {f}")
    sys.exit(1 if failed else 0)
