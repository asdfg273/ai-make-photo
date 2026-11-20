# tests/test_regional.py — 区域提示词 / 局部重绘（无头，纯脚本，不跑 GPU）
#
# 为什么要有这个文件
# ------------------
# 这里锁住的是几个**会静默失效**的行为：错了不会报错，只是"效果不对"，
# 极难在人工点界面时发现。最典型的一条：
#
#   区域配了「局部 ADetailer」之后，该区域的**提示词重绘被整个盖掉**
#   （曾经把重绘前的图裁给 ADetailer 再贴回，实测区域差异从 43.6 掉到 1.1）
#
# 所以这些断言是回归保险，不只是"能跑就行"。
#
# 用法：venv\Scripts\python.exe tests\test_regional.py
import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码，放最前）
from _bootstrap import isolated_config as _isolated_config  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])


def _reg(name="左", rect=(0.0, 0.0, 0.5, 1.0), prompt="a red apple",
         weight=0.7, ad=None, cn=None, shape="rect", mask_path=""):
    return {"name": name, "prompt": prompt, "negative": "", "weight": weight,
            "shape": shape, "rect": rect, "mask_path": mask_path,
            "color": "#f38ba8", "enabled": True, "feather": 16,
            "controlnet": cn, "adetailer": ad}


def test_region_parsing_is_tolerant():
    """编辑器/老工作流传来的脏数据不能把引擎弄崩。"""
    from core.regional import Region, coerce_regions, regions_from_json
    assert Region.from_dict({}).weight == 0.5          # 缺字段走默认
    for junk in (None, "x", {}, [], 3, {"weight": "abc", "rect": "bad"}):
        Region.from_dict(junk)                         # 不抛就算过
    assert regions_from_json("not json") == []
    assert regions_from_json(None) == []
    assert regions_from_json("") == []
    # 带外层包装（编辑器导出格式）也要认
    assert len(regions_from_json('{"regions": [%s]}'
                                 % '{"prompt":"a"}')) == 1
    assert len(coerce_regions([_reg()])) == 1


def test_rect_masks_are_localized():
    """矩形区域必须真的只覆盖该区域（这是"局部"的根基）。"""
    import numpy as np
    from core.regional import region_masks, coerce_regions
    regs = coerce_regions([_reg(rect=(0.25, 0.25, 0.75, 0.75))])
    pairs = region_masks(regs, 512, 512)
    assert len(pairs) == 1
    m = np.asarray(pairs[0][1].convert("L"))
    assert m[256, 256] > 0                     # 中心生效
    assert m[5, 5] == 0                        # 角落不生效
    ys, xs = np.nonzero(m)
    # 白区应落在 [128,384] ±羽化 范围内
    assert 100 <= xs.min() and xs.max() <= 412, (xs.min(), xs.max())
    assert 100 <= ys.min() and ys.max() <= 412, (ys.min(), ys.max())


def test_out_of_range_rect_is_clamped_not_crashed():
    import numpy as np
    from core.regional import region_masks, coerce_regions
    regs = coerce_regions([_reg(rect=(0.9, 0.9, 1.8, 1.8))])
    m = np.asarray(region_masks(regs, 256, 256)[0][1].convert("L"))
    assert m.shape == (256, 256)
    assert m.max() > 0                          # 夹取后仍有面积，而不是消失


def test_validate_reports_problems():
    from core.regional import validate_regions, coerce_regions
    regs = coerce_regions([
        _reg(name="权重越界", weight=1.7),
        _reg(name="没有词", prompt=""),
        _reg(name="零面积", rect=(0.5, 0.5, 0.5, 0.5)),
        _reg(name="缺图", shape="mask", mask_path="Z:/绝对不存在.png"),
    ])
    msgs = " ".join(validate_regions(regs, 512, 512))
    assert "权重" in msgs
    assert "提示词" in msgs
    assert "缺图" in msgs or "不存在" in msgs
    assert validate_regions([], 512, 512)       # 空列表也要有提示


def test_adetailer_does_not_wipe_the_region_redraw():
    """回归保险：局部 ADetailer 不能把该区域的提示词重绘结果盖掉。

    这是一个**静默失效**的真实 bug：`_region_adetailer()` 曾经接收重绘**之前**
    的图（`cur`），裁它、处理后按蒙版贴回，于是区域重绘被整体覆盖。
    这里用源码断言把调用点钉死 —— 真正的端到端验证在 README 的手册里
    （需要 GPU：给区域配 ADetailer 后，区域差异应仍然很大，约 45 而不是 1）。
    """
    import inspect
    from core import regional
    src = inspect.getsource(regional.apply_regions)
    assert "_region_adetailer" in src, "apply_regions 里应调用局部 ADetailer"
    # 调用点必须传 redrawn，不能传 cur
    idx = src.index("_region_adetailer(")
    call = src[idx:idx + 220]
    assert "redrawn" in call, f"局部 ADetailer 应处理重绘后的图，实际调用: {call[:120]}"
    assert "cur," not in call.split("prompt=")[0], \
        "局部 ADetailer 不能拿重绘前的 cur（会把区域重绘盖掉）"


def test_regions_port_type_is_strong():
    """REGIONS 是要参与校验的强类型：连错端口必须在运行前报错。"""
    from workflow.schema import DataType, Workflow, Node, types_compatible
    assert DataType.REGIONS.value == "REGIONS"
    assert not types_compatible(DataType.TEXT_COND, DataType.REGIONS)
    assert not types_compatible(DataType.IMAGE, DataType.REGIONS)
    assert types_compatible(DataType.ANY, DataType.REGIONS)

    from workflow import registry
    registry.load_builtin_nodes()
    assert registry.exists("RegionalPrompt")
    assert registry.exists("RegionalRedraw")
    assert registry.get("RegionalPrompt").outputs["regions"].type \
        is DataType.REGIONS

    wf = Workflow(name="t")
    wf.add(Node("rp", "RegionalPrompt", {}, {"regions_json": '{"regions":[]}'}))
    wf.add(Node("rr", "RegionalRedraw", {}, {}))
    wf.add(Node("m", "LoadCheckpoint", {}, {"path": "x"}))
    wf.add(Node("img", "LoadImage", {}, {"path": ""}))
    wf.connect("rp", "regions", "rr", "regions")
    wf.connect("m", "model", "rr", "model")
    wf.connect("img", "image", "rr", "image")
    from workflow import graph as G
    errs = [i for i in G.validate(wf) if i.level == "error"
            and "regions" in i.message]
    assert not errs, [i.message for i in errs]

    # 类型错连必须被抓到
    wf2 = Workflow(name="t2")
    wf2.add(Node("rp", "RegionalPrompt", {}, {}))
    wf2.add(Node("rr", "RegionalRedraw", {}, {}))
    wf2.connect("rp", "regions", "rr", "image")      # REGIONS → IMAGE
    bad = [i for i in G.validate(wf2) if i.level == "error"]
    assert bad, "REGIONS 接到 IMAGE 应该报类型不匹配"


def test_main_ui_exposes_region_editor():
    """普通用户必须在主界面上就能用（不要求懂工作流）。"""
    import main
    from core.model_manager import ModelManager
    with _isolated_config():
        win = main.AIDesktopApp()
        win.ai = ModelManager()
    try:
        for attr in ("chk_use_regions", "btn_edit_regions",
                     "btn_clear_regions", "lbl_regions_summary"):
            assert hasattr(win, attr), f"主界面缺少控件 {attr}"
        for meth in ("open_region_editor", "clear_regions", "regions",
                     "_refresh_regions_summary", "_apply_regions_if_enabled"):
            assert callable(getattr(win, meth, None)), f"主界面缺少方法 {meth}"
        # 隔离配置下不该有任何区域残留（否则说明测试污染了用户配置）
        assert win.regions() == [], f"配置里有残留区域: {win.regions()}"
        win._refresh_regions_summary()
        assert "未设置" in win.lbl_regions_summary.text()
        # 状态摘要要能正确反映区域
        win.config.regions = [_reg(name="脸部", prompt="detailed face")]
        win._refresh_regions_summary()
        assert "脸部" in win.lbl_regions_summary.text()
    finally:
        with _isolated_config():
            win.close()


def test_editor_output_is_engine_readable():
    """编辑器产出的 dict 必须能被引擎直接读懂（两边格式不能漂移）。"""
    from ui.region_editor import RegionEditorDialog
    from core.regional import coerce_regions
    dlg = RegionEditorDialog([], width=512, height=768)
    try:
        dlg._on_created((0.0, 0.0, 0.5, 1.0))
        dlg._set_field(0, "name", "左边")
        dlg._set_field(0, "prompt", "detailed face, sharp eyes")
        dlg._set_field(0, "weight", 0.55)
        regs = dlg.regions()
        assert len(regs) == 1
        parsed = coerce_regions(regs)
        assert parsed[0].prompt == "detailed face, sharp eyes"
        assert abs(parsed[0].weight - 0.55) < 1e-9
    finally:
        dlg.close()


def test_builder_compiles_regions_into_workflow():
    """主界面设了区域，简易模式编译出的工作流就该带上区域节点。"""
    import main
    from core.model_manager import ModelManager
    from workflow import builder, graph as G
    with _isolated_config():
        win = main.AIDesktopApp()
        # build_from_ui 走 _gt_prepare_context()，需要模型管理器在位
        # （主程序里由 _show_main 负责赋值）
        win.ai = ModelManager()
    try:
        win.txt_prompt.setPlainText("portrait, detailed clothing")
        win.txt_neg.setPlainText("lowres")
        win.config.regions = [_reg(name="脸部", prompt="detailed face")]
        win.config.use_regions = True
        wf = builder.build_from_ui(win, name="t")
        types = [n.type for n in wf.nodes]
        assert "RegionalPrompt" in types, types
        assert "RegionalRedraw" in types, types
        assert types.index("RegionalPrompt") < types.index("RegionalRedraw")
        assert types.index("RegionalRedraw") < types.index("SaveImage")
        rp = wf.by_type("RegionalPrompt")[0]
        assert "detailed face" in str(rp.params.get("regions_json"))
        errs = [i for i in G.validate(wf) if i.level == "error"]
        assert not errs, [i.message for i in errs]
    finally:
        with _isolated_config():
            win.close()


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
