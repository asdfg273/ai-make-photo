# tests/test_builder_aux.py — 界面辅助项必须真的进工作流图（无头，纯脚本）
#
# 锁住的是一个**静默降级**的 P0：
#
#   界面勾了 LoRA / ControlNet / IP-Adapter，点「用工作流引擎生成」时
#   生成的图里一个对应节点都没有，而且**不报错、不警告**。
#
# 根因：`_gt_prepare_context()` 里 'use_ipa'/'use_pose'/'lora_meta_info'
# 是硬编码初值，真实值由 `_gt_setup_ipa` / `_gt_apply_loras` /
# `_gt_run_pose_transfer` 写入，而那三个只在 `start_generation` 里被调用。
# `builder.build_from_ui()` 只调 `_gt_prepare_context()` → 永远读到"什么都没开"。
#
# 顺带锁住第二个 bug：`_gt_run_pose_transfer` 读的是 `self.pose_image`
# ——这个属性在整个项目里**从未被赋值**，于是用户上传的骨架图被完全忽略
# （连老的手动 ControlNet 路径也是坏的）。
#
# 用法：venv\Scripts\python.exe tests\test_builder_aux.py
import os
import sys

import _bootstrap  # noqa: F401
from _bootstrap import isolated_config  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])


def _make_win():
    import main
    from core.model_manager import ModelManager
    with isolated_config():
        win = main.AIDesktopApp()
        win.ai = ModelManager()
    return win


def _close(win):
    with isolated_config():
        win.close()


def _some_lora():
    from utils import paths
    for sub in ("sd1.5", "sdxl", "_unknown_dit", ""):
        d = os.path.join(paths.LORA_DIR, sub) if sub else paths.LORA_DIR
        try:
            for f in sorted(os.listdir(d)):
                if f.endswith((".safetensors", ".pt", ".ckpt")):
                    return f
        except OSError:
            continue
    return None


def _set_lora(win, name):
    """模拟"用户在下拉框里选了某个 LoRA"。

    ⚠️ `combo_loras` 里的条目是由 `refresh_lora_by_model()` 按**当前底模架构**
    过滤后填进去的，需要一个已选中的底模。测试只关心"读不读得到下拉框的值"，
    所以这里直接把条目加进去再选中，避免依赖模型选择流程。
    非可编辑 QComboBox 上 `setCurrentText` 对不存在的条目是**静默无效**的，
    必须先 addItem —— 否则测试会假通过/假失败。
    """
    if not name or not hasattr(win, "combo_loras"):
        return False
    cb = win.combo_loras[0]
    if cb.findText(name) < 0:
        cb.addItem(name)
    cb.setCurrentText(name)
    try:
        win.scale_loras[0].setValue(0.8)
    except Exception:
        pass
    return cb.currentText() == name


def _img():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "logo", "dzbut-9fc5g-001.ico")
    return p if os.path.exists(p) else None


def test_aux_state_reads_ui():
    """_gt_aux_state() 必须真的读界面，而不是返回初值。"""
    win = _make_win()
    try:
        st = win._gt_aux_state(win._gt_prepare_context())
        assert st["use_ipa"] is False and st["use_pose"] is False
        assert st["lora_list"] == []

        win.chk_use_pose.setChecked(True)
        win.chk_use_ipa.setChecked(True)
        if hasattr(win, "spin_ipa_scale"):
            win.spin_ipa_scale.setValue(0.75)
        name = _some_lora()
        picked = _set_lora(win, name)

        st2 = win._gt_aux_state(win._gt_prepare_context())
        assert st2["use_pose"] is True, st2
        assert st2["use_ipa"] is True, st2
        assert abs(st2["ipa_scale"] - 0.75) < 1e-6, st2
        if picked:
            assert any(n == name for n, _ in st2["lora_list"]), st2["lora_list"]
            assert any(abs(w - 0.8) < 1e-6 for _, w in st2["lora_list"])
    finally:
        _close(win)


def test_pose_image_phantom_attribute_is_gone():
    """`self.pose_image` 从未被赋值 —— 代码里不能再读它。

    这是一个真实的 bug：`_gt_run_pose_transfer` 读 `getattr(self,'pose_image')`
    → 恒为 None → `prepare_controlnet("openpose")` 永不执行，
    用户上传的骨架图被静默忽略。
    """
    import inspect
    import re
    from utils import app_generation

    assert not hasattr(_make_win(), "pose_image"), "不该存在 pose_image 属性"
    src = inspect.getsource(app_generation.GenerationMixin)
    # 必须先去掉文档字符串和注释行再匹配：否则"解释这个 bug"的注释
    # 会把测试自己绊倒（这个坑我踩过一次）
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    code = "\n".join(l for l in code.splitlines()
                     if not l.strip().startswith("#"))
    assert "getattr(self, 'pose_image'" not in code, \
        "仍在读取从未被赋值的 self.pose_image"
    assert "_gt_pose_pil_image" in code, "应通过 _gt_pose_pil_image() 读骨架图"


def test_pose_pil_image_reads_path():
    """_gt_pose_pil_image() 从 pose_image_path 读图；缺失时返回 None 不抛。"""
    from PIL import Image
    win = _make_win()
    try:
        win.pose_image_path = None
        assert win._gt_pose_pil_image() is None
        win.pose_image_path = "Z:/绝对不存在.png"
        assert win._gt_pose_pil_image() is None
        p = _img()
        if p:
            win.pose_image_path = p
            im = win._gt_pose_pil_image()
            assert isinstance(im, Image.Image), type(im)
        # 也兼容有人直接塞 PIL 对象
        win.pose_image_path = Image.new("RGB", (8, 8))
        assert isinstance(win._gt_pose_pil_image(), Image.Image)
    finally:
        _close(win)


def test_build_from_ui_includes_all_aux_nodes():
    """三项都开着且给了参考图 → 图里必须有 LoRA / CN / IP-Adapter 节点。"""
    from workflow import builder
    win = _make_win()
    try:
        ref = _img()
        lora = _some_lora()
        win.txt_prompt.setPlainText("portrait, detailed clothing")
        win.txt_neg.setPlainText("lowres")
        win.chk_use_pose.setChecked(True)
        win.chk_use_ipa.setChecked(True)
        if ref:
            win.pose_image_path = ref
            win.ipa_image_path = ref
            win.ref_image_path = ref
        picked = _set_lora(win, lora)

        wf = builder.build_from_ui(win, name="aux")
        types = [n.type for n in wf.nodes]
        if picked:
            assert "LoadLoRA" in types, types
        if ref:
            assert "ControlNetPreprocess" in types, types
            assert "ControlNetApply" in types, types
            assert "IPAdapterApply" in types, types
            # 参考图要有自己的 LoadImage 源，否则必填输入缺失
            assert types.count("LoadImage") >= 2, types
        from workflow import graph as G
        errs = [i for i in G.validate(wf) if i.level == "error"]
        assert not errs, [i.message for i in errs]
    finally:
        _close(win)


def test_missing_reference_is_reported_not_silent():
    """缺参考图时不能静默：必须留下可展示的 warning。"""
    from workflow import builder
    win = _make_win()
    try:
        win.txt_prompt.setPlainText("portrait")
        win.txt_neg.setPlainText("lowres")
        win.chk_use_pose.setChecked(True)
        win.chk_use_ipa.setChecked(True)
        # 故意不给参考图
        win.pose_image_path = None
        win.ipa_image_path = None
        win.ref_image_path = None
        wf = builder.build_from_ui(win, name="noref")
        types = [n.type for n in wf.nodes]
        assert "ControlNetPreprocess" not in types, types
        assert "IPAdapterApply" not in types, types
        warns = list((wf.metadata or {}).get("warnings") or [])
        assert warns, "缺参考图时必须记 warning（否则就是静默降级）"
        joined = " ".join(warns)
        assert "ControlNet" in joined or "IP-Adapter" in joined, joined
    finally:
        _close(win)


def test_lora_slots_and_prompt_tags_both_count():
    """提示词里的 <lora:> 标签与槽位选择都要算进依赖。"""
    win = _make_win()
    try:
        st = win._gt_aux_state({"lora_tags": [{"name": "tag_lora", "weight": 0.6}]})
        assert any(n == "tag_lora" for n, _ in st["lora_list"]), st["lora_list"]
        # 同名槽位不应重复挂载
        name = _some_lora()
        if name and hasattr(win, "combo_loras"):
            win.combo_loras[0].setCurrentText(name)
            st2 = win._gt_aux_state({"lora_tags": [
                {"name": name, "weight": 0.6}]})
            hits = [n for n, _ in st2["lora_list"] if n == name]
            assert len(hits) == 1, f"同名 LoRA 被重复挂载: {st2['lora_list']}"
    finally:
        _close(win)


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
