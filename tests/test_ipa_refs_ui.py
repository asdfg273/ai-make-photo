# tests/test_ipa_refs_ui.py — 主界面「多参考图 (IP-Adapter)」（无头，纯脚本，不跑 GPU）
#
# 为什么要有这个文件
# ------------------
# 「多参考图」以前**只存在于工作流编辑器**里（三个角色节点串起来）。
# 普通用户在主界面上根本用不到。这一条链路把同一件事搬到主界面：
#
#     分组（开关/编辑/清空/摘要）→ ui/ipa_editor.IPARefEditorDialog
#     → config.ipa_refs → 生成时 prepare_ip_adapter_multi（一次装 N 个）
#
# 锁住的是几类**静默失效**（错了不报错，只是"没效果"）：
#   · 分组控件/宿主方法缺失 → 按钮点了没反应（_wire 只告警）
#   · 编辑器产出的 dict 引擎读不懂 → 到了采样才报"参考图读不到"
#   · 坏 JSON 把对话框整个抛崩 → 用户填好的内容全丢
#   · 简易模式编译出的工作流里没有角色节点 → "设了等于没设"
#
# ⚠️ 所有 AIDesktopApp() 的创建与 close() 都必须包在 isolated_config() 里：
#    closeEvent() 会把配置写回 AppConfig.config_file，不隔离就会覆盖用户
#    真实的 app_config.json（这个坑已经踩过一次）。
#
# 用法：venv\Scripts\python.exe tests\test_ipa_refs_ui.py
import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码，放最前）
from _bootstrap import isolated_config as _isolated_config  # noqa: E402

# 追加整进程隔离：被测方法（clear_ipa_refs / _toggle_ipa_refs）自己会
# config.save()，只包 _make_win/_close 挡不住，必须全程指向临时配置。
_bootstrap.isolate_config_globally()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])


def _img():
    """一张真实存在的小图（logo 里的 ico 是项目自带的，不做新文件）。"""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "logo", "dzbut-9fc5g-001.ico")
    return p if os.path.exists(p) else None


def _ref(role="identity", weight=0.5, path=None, enabled=True, mask=""):
    return {"role": role, "weight": weight, "path": path or _img() or "",
            "mask_path": mask, "enabled": enabled}


def _make_win():
    import main
    from core.model_manager import ModelManager
    with _isolated_config():
        win = main.AIDesktopApp()
        win.ai = ModelManager()
    return win


def _close(win):
    with _isolated_config():
        win.close()


# ----------------------------------------------------------------------
#  1) 主界面结构与宿主方法
# ----------------------------------------------------------------------
def test_main_ui_exposes_ipa_refs():
    """普通用户必须在主界面上就能用（不要求懂工作流）。"""
    win = _make_win()
    try:
        for attr in ("chk_use_ipa_refs", "btn_edit_ipa_refs",
                     "btn_clear_ipa_refs", "lbl_ipa_refs_summary"):
            assert hasattr(win, attr), f"主界面缺少控件 {attr}"
        for meth in ("open_ipa_ref_editor", "clear_ipa_refs", "ipa_refs",
                     "_refresh_ipa_refs_summary", "_apply_ipa_refs_if_enabled"):
            assert callable(getattr(win, meth, None)), f"主界面缺少方法 {meth}"
        # 分组必须真的挂在 shell 的折叠分组字典里（重构时不会被顺手删掉）
        secs = getattr(win, "_group_sections", {})
        assert "ipa_refs" in secs, f"折叠分组缺少 ipa_refs: {list(secs)}"
        # 隔离配置下不该有任何残留
        assert win.ipa_refs() == [], f"配置里有残留参考图: {win.ipa_refs()}"
        win._refresh_ipa_refs_summary()
        assert "未设置" in win.lbl_ipa_refs_summary.text()
    finally:
        _close(win)


def test_summary_label_reflects_refs():
    """摘要标签要真的跟着参考图变（含"3 张参考：身份(0.50) …"格式）。"""
    win = _make_win()
    try:
        win.config.ipa_refs = [
            _ref("identity", 0.5), _ref("style", 0.7), _ref("composition", 0.4)]
        win.config.use_ipa_refs = True
        win._refresh_ipa_refs_summary()
        txt = win.lbl_ipa_refs_summary.text()
        assert "3 张参考" in txt, txt
        assert "身份(0.50)" in txt, txt
        assert "风格(0.70)" in txt, txt
        assert "构图(0.40)" in txt, txt
        # 停用的条目不算
        win.config.ipa_refs = [_ref("identity", 0.5),
                               _ref("style", 0.7, enabled=False)]
        win._refresh_ipa_refs_summary()
        txt2 = win.lbl_ipa_refs_summary.text()
        assert "1 张参考" in txt2 and "风格" not in txt2, txt2
        # 未启用时要明说"生成时不生效"（否则用户以为设了就有效）
        win.config.use_ipa_refs = False
        win._refresh_ipa_refs_summary()
        assert "不生效" in win.lbl_ipa_refs_summary.text()
        # 清空
        win.clear_ipa_refs()
        assert win.ipa_refs() == []
        assert not win.config.use_ipa_refs
    finally:
        _close(win)


def test_clear_and_toggle_are_wired():
    """清空按钮/勾选框的行为：只动多参考图，不碰单张 IPA。"""
    win = _make_win()
    try:
        win.config.ipa_refs = [_ref("identity", 0.5), _ref("style", 0.7)]
        win.config.use_ipa_refs = True
        win.clear_ipa_refs()
        assert win.config.ipa_refs == []
        assert win.config.use_ipa_refs is False
        assert win.chk_use_ipa_refs.isChecked() is False
        # 勾选框 → _toggle_ipa_refs 只改配置，不碰 chk_use_ipa（单张那条路）
        win._toggle_ipa_refs(True)
        assert win.config.use_ipa_refs is True
        assert win.chk_use_ipa.isChecked() is False
        win._toggle_ipa_refs(False)
        assert win.config.use_ipa_refs is False
    finally:
        _close(win)


# ----------------------------------------------------------------------
#  2) 编辑器
# ----------------------------------------------------------------------
def test_editor_output_is_engine_readable():
    """编辑器产出的 dict 必须能被引擎直接读懂（两边格式不能漂移）。"""
    from ui.ipa_editor import IPARefEditorDialog
    from core import ipa_multi as ipa_mod
    dlg = IPARefEditorDialog([], parent=None)
    try:
        assert dlg.table.rowCount() == 0
        dlg._add_ref("identity")
        dlg._add_ref("style")
        assert dlg.table.rowCount() == 2
        dlg._set_field(0, "weight", 0.55)
        dlg._set_field(1, "weight", 0.75)
        dlg._set_field(0, "path", "E:/refs/face.png")
        dlg._set_field(1, "path", "E:/refs/style.jpg")
        dlg._set_field(1, "mask_path", "E:/masks/sky.png")
        dlg._update_summary()

        refs = dlg.refs()
        assert len(refs) == 2
        # 引擎侧：resolve_role_plan 能吃下这份数据（不抛就算通）
        plan, warns = ipa_mod.resolve_role_plan(refs, installed=["base"])
        assert len(plan) == 2, plan
        assert plan[0].role == "identity" and plan[1].role == "style"
        assert abs(plan[0].weight - 0.55) < 1e-9, plan[0].weight
        assert plan[1].mask_path == "E:/masks/sky.png", plan[1].mask_path
        # 本机只装 base → 身份/风格各降级一条中文警告（不是静默）
        assert len(warns) == 2, warns
        assert "降级" in warns[0] and "身份" in warns[0]
        # coerce_references 是采样层真正走的路径
        parsed = ipa_mod.coerce_references(refs)
        assert parsed[1].variant_requested == "plus", parsed[1].variant_requested
        # 角色下拉是中文显示、内部是英文键
        assert "身份" in dlg.lbl_summary.text() or "1 张" in dlg.lbl_summary.text()
    finally:
        dlg.close()


def test_editor_summary_and_installed_note():
    """摘要格式 + 「本机已装档位」提示（不许需要 GPU）。"""
    from ui.ipa_editor import (IPARefEditorDialog, installed_variants_text,
                               summary_text)
    note = installed_variants_text()
    assert isinstance(note, str) and note, note
    assert ("已装档位" in note) or ("未检测到" in note), note
    # 取不到 ModelManager 也不该抛（纯文件系统探测）
    txt = summary_text([_ref("identity", 0.5), _ref("style", 0.7),
                        _ref("composition", 0.4)])
    assert txt == "3 张参考：身份(0.50) 风格(0.70) 构图(0.40)", txt
    assert summary_text([]) == "（未设置参考图）"
    assert summary_text([_ref("style", 0.7, enabled=False)]) == "（未设置参考图）"

    dlg = IPARefEditorDialog([_ref("identity", 0.5)])
    try:
        assert "已装档位" in dlg.lbl_installed.text() or \
               "未检测到" in dlg.lbl_installed.text(), dlg.lbl_installed.text()
        assert "1 张参考" in dlg.lbl_summary.text(), dlg.lbl_summary.text()
        # 没选图片的条目要在摘要里点出来（否则用户不知道为什么没效果）
        dlg._set_field(0, "path", "")
        dlg._update_summary()
        assert "还没选图片" in dlg.lbl_summary.text(), dlg.lbl_summary.text()
    finally:
        dlg.close()


def test_editor_survives_bad_json_and_junk():
    """坏 JSON / 脏条目**永远不能抛**（抛了会关掉对话框、丢用户输入）。"""
    from ui.ipa_editor import IPARefEditorDialog
    dlg = IPARefEditorDialog([], parent=None)
    try:
        for junk in (None, 3, "x", {"a": 1}, [1, 2, None, "s"],
                     [{"role": "不认识的写法", "weight": "abc"}],
                     [{"role": "身份", "weight": 99, "path": None}],
                     [{"role": "plus-face", "weight": -3}]):
            dlg._set_refs(list(junk) if isinstance(junk, list) else [junk]
                          if isinstance(junk, dict) else [])
            dlg.refs()            # 不抛就算过
        # 角色/权重都要被夹到合法范围
        dlg._set_refs([{"role": "身份", "weight": 99}])
        r = dlg.refs()[0]
        assert r["role"] == "identity", r
        assert r["weight"] == 2.0, r
        # 中文别名也认
        dlg._set_refs([{"role": "画风"}])
        assert dlg.refs()[0]["role"] == "style"
    finally:
        dlg.close()


# ----------------------------------------------------------------------
#  3) 生成侧：_apply_ipa_refs_if_enabled 与 builder
# ----------------------------------------------------------------------
def test_apply_returns_none_when_disabled():
    """没开多参考图时，生成侧必须**什么都不做**（返回 None）。"""
    win = _make_win()
    try:
        ctx = {"width": 512, "height": 768}
        assert win._apply_ipa_refs_if_enabled(ctx) is None
        # 开了开关但没有条目 / 条目全停用 → 也不装
        win.config.use_ipa_refs = True
        assert win._apply_ipa_refs_if_enabled(ctx) is None
        win.config.ipa_refs = [_ref("identity", 0.5, enabled=False)]
        assert win._apply_ipa_refs_if_enabled(ctx) is None
    finally:
        _close(win)


def test_apply_gates_on_capability_not_model_name():
    """能力门禁：不支持的架构必须**明确跳过并留日志**，而不是静默。"""
    win = _make_win()
    try:
        win.config.use_ipa_refs = True
        win.config.ipa_refs = [_ref("identity", 0.5)]
        # 造一个"不支持 ip_adapter"的架构
        win.ai.current_arch_id = "anima"
        res = win._apply_ipa_refs_if_enabled({"width": 512, "height": 768})
        assert res is None, "不支持的架构不该装上适配器"
        log = ""
        try:
            log = win.txt_log_image.toPlainText()
        except Exception:
            pass
        assert "IP-Adapter" in log or "不支持" in log, log
    finally:
        _close(win)


def test_builder_emits_role_nodes_when_refs_configured():
    """主界面配了参考图，简易模式编译出的工作流就该带角色节点。"""
    from workflow import builder, graph as G
    win = _make_win()
    try:
        ref = _img()
        win.txt_prompt.setPlainText("portrait, detailed clothing")
        win.txt_neg.setPlainText("lowres")
        win.config.ipa_refs = [_ref("identity", 0.5, path=ref),
                               _ref("style", 0.7, path=ref)]
        win.config.use_ipa_refs = True
        wf = builder.build_from_ui(win, name="ipa_refs")
        types = [n.type for n in wf.nodes]
        assert "IPAdapterIdentity" in types, types
        assert "IPAdapterStyle" in types, types
        # 每张参考图都要有自己的 LoadImage 源（image 是必填输入）
        assert types.count("LoadImage") >= 2, types
        # 角色节点必须串起来（第二个挂在第一个的 model 输出上）
        # links 是 (src_id, src_port, dst_id, dst_port) 四元组
        ident = wf.by_type("IPAdapterIdentity")[0]
        style = wf.by_type("IPAdapterStyle")[0]
        assert any(l[0] == ident.id and l[2] == style.id
                   for l in wf.links), \
            [f"{l[0]}->{l[2]}" for l in wf.links]
        # 权重/角色都要落到节点参数上
        assert abs(float(ident.params.get("weight")) - 0.5) < 1e-9
        assert abs(float(style.params.get("weight")) - 0.7) < 1e-9
        # 多参考图与单张不能同时编译（load_ip_adapter 第二次是覆盖语义）
        assert "IPAdapterApply" not in types, types
        errs = [i for i in G.validate(wf) if i.level == "error"]
        assert not errs, [i.message for i in errs]
    finally:
        _close(win)


def test_builder_skips_role_nodes_when_not_configured():
    """没配参考图时不能凭空多出角色节点（否则等于偷偷改了别人的图）。"""
    from workflow import builder
    win = _make_win()
    try:
        win.txt_prompt.setPlainText("portrait")
        win.txt_neg.setPlainText("lowres")
        win.config.ipa_refs = []
        win.config.use_ipa_refs = False
        wf = builder.build_from_ui(win, name="no_refs")
        types = [n.type for n in wf.nodes]
        for t in ("IPAdapterIdentity", "IPAdapterStyle",
                  "IPAdapterComposition", "IPAdapterApply"):
            assert t not in types, types
        # 配了但开关关着 → 同样不进图
        win.config.ipa_refs = [_ref("identity", 0.5)]
        wf2 = builder.build_from_ui(win, name="off")
        types2 = [n.type for n in wf2.nodes]
        assert "IPAdapterIdentity" not in types2, types2
        # 开关开着但图片不存在 → 要有中文 warning，不能装成"没有参考图"的节点
        win.config.use_ipa_refs = True
        win.config.ipa_refs = [{"role": "identity", "weight": 0.5,
                                "path": "Z:/绝对不存在.png", "enabled": True}]
        wf3 = builder.build_from_ui(win, name="badpath")
        types3 = [n.type for n in wf3.nodes]
        assert "IPAdapterIdentity" not in types3, types3
        warns = " ".join((wf3.metadata or {}).get("warnings") or [])
        assert "参考图" in warns, warns
    finally:
        _close(win)


def test_single_image_ipa_path_still_works():
    """回归保险：旧的「单张 IP-Adapter」路径必须原样能用。"""
    from workflow import builder
    win = _make_win()
    try:
        ref = _img()
        win.txt_prompt.setPlainText("portrait")
        win.txt_neg.setPlainText("lowres")
        win.config.use_ipa_refs = False
        win.config.ipa_refs = []
        win.chk_use_ipa.setChecked(True)
        win.ipa_image_path = ref
        win.ref_image_path = ref
        st = win._gt_aux_state(win._gt_prepare_context())
        assert st["use_ipa"] is True, st
        assert st["use_ipa_refs"] is False, st
        wf = builder.build_from_ui(win, name="single")
        types = [n.type for n in wf.nodes]
        assert "IPAdapterApply" in types, types
        assert "IPAdapterIdentity" not in types, types
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
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("=" * 60)
    print(f"总计 {len(fns)} 项：PASS {len(fns) - len(failed)} / FAIL {len(failed)}")
    for f in failed:
        print(f"  失败: {f}")
    sys.stdout.flush()
    sys.exit(1 if failed else 0)
