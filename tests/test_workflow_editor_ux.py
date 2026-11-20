# -*- coding: utf-8 -*-
"""工作流编辑器的"好不好用"回归（参考 ComfyUI 补齐的那批交互）。

覆盖：
  · 5 个内置模板都能建、端口合法、图校验 0 错误
  · 撤销 / 重做真的把图恢复回去（快照式，含节点与连线）
  · 复制 / 粘贴保留内部连线、且节点 id 不重复
  · 节点搜索按类型过滤（拖线到空白时只列能接住的节点）
  · 拖图片进画布 → 自动建 LoadImage 且路径写进参数
  · 右键菜单里该有的项都在
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import _bootstrap  # noqa: E402
_bootstrap.isolate_config_globally()

from PyQt6.QtCore import QPointF  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMenu  # noqa: E402

from workflow import graph as G, registry, templates  # noqa: E402
from workflow.schema import DataType  # noqa: E402

_fails = []
_APP = QApplication.instance() or QApplication(sys.argv)


def check(name):
    def deco(fn):
        try:
            ok = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok = False
            print(f"   EXC {type(e).__name__}: {e}")
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            _fails.append(name)
        return fn
    return deco


@check("templates_build_and_validate")
def t_templates():
    assert len(templates.list_templates()) >= 5
    for key, name, _desc in templates.list_templates():
        wf = templates.build(key)
        assert wf.nodes, key
        issues = [i for i in G.validate(wf) if getattr(i, "level", "error") == "error"]
        assert not issues, f"{key}: {[str(i) for i in issues]}"
        # 位置信息必须齐全，否则打开时节点会堆在左上角
        pos = wf.metadata.get("ui_positions") or {}
        assert len(pos) == len(wf.nodes), f"{key} 缺位置: {len(pos)}/{len(wf.nodes)}"
    return True


@check("scene_loads_template")
def t_scene_load():
    from ui.workflow_editor.canvas import GraphScene
    scene = GraphScene()
    wf = templates.build("txt2img")
    scene.load_workflow(wf)
    assert len(scene.node_items) == len(wf.nodes)
    assert len(scene.link_items) == len(wf.links)
    # 数据层与画布层一致
    back = scene.to_workflow()
    assert len(back.nodes) == len(wf.nodes)
    return True


@check("undo_redo_restores_graph")
def t_undo_redo():
    from ui.workflow_editor.canvas import GraphScene
    from ui.workflow_editor.history import HistoryStack
    scene = GraphScene()
    scene.history = HistoryStack()
    scene.load_workflow(templates.build("txt2img"))
    n0 = len(scene.node_items)

    nid = scene.add_node("Upscale", QPointF(900, 500))
    assert nid and len(scene.node_items) == n0 + 1

    got = scene.history.undo(scene.snapshot())
    assert got, "撤销栈是空的"
    scene.restore_snapshot(got[1])
    assert len(scene.node_items) == n0, len(scene.node_items)

    got2 = scene.history.redo(scene.snapshot())
    assert got2, "重做栈是空的"
    scene.restore_snapshot(got2[1])
    assert len(scene.node_items) == n0 + 1, len(scene.node_items)
    # 节点类型也要对（不是"恢复出一个别的节点"）
    assert any(it.ntype == "Upscale" for it in scene.node_items.values())
    return True


@check("copy_paste_keeps_internal_links")
def t_copy_paste():
    from ui.workflow_editor.canvas import GraphScene
    from ui.workflow_editor.history import HistoryStack
    scene = GraphScene()
    scene.history = HistoryStack()
    scene.load_workflow(templates.build("upscale"))     # img→up→save，3 条线

    scene.select_all()
    payload = scene.copy_selection()
    assert payload and len(payload["nodes"]) == 3, payload
    assert len(payload["links"]) == 3, payload

    before_nodes = len(scene.node_items)
    before_links = len(scene.link_items)
    new_ids = scene.paste_payload(payload, QPointF(600, 40))
    assert len(new_ids) == 3
    assert len(scene.node_items) == before_nodes + 3
    assert len(scene.link_items) == before_links + 3
    # id 不能撞
    assert len(set(scene.node_items)) == len(scene.node_items)
    return True


@check("duplicate_selection_offsets")
def t_duplicate():
    from ui.workflow_editor.canvas import GraphScene
    from ui.workflow_editor.history import HistoryStack
    scene = GraphScene()
    scene.history = HistoryStack()
    scene.load_workflow(templates.build("upscale"))
    item = list(scene.node_items.values())[0]
    item.setSelected(True)
    p0 = item.pos()
    new_ids = scene.duplicate_selection()
    assert len(new_ids) == 1
    new_item = scene.node_items[new_ids[0]]
    assert new_item.pos() != p0
    assert abs(new_item.pos().x() - p0.x() - 36.0) < 0.01
    return True


@check("search_filters_by_type")
def t_search_filter():
    from ui.workflow_editor.search_dialog import NodeSearchDialog
    registry.load_builtin_nodes()

    all_dlg = NodeSearchDialog(None)
    n_all = all_dlg.list.count()
    assert n_all == len(registry.all_specs()), (n_all, len(registry.all_specs()))

    img_dlg = NodeSearchDialog(None, type_hint=DataType.IMAGE.value)
    types = [img_dlg.list.item(i).data(0x0100) for i in range(img_dlg.list.count())]
    assert types, "按 IMAGE 过滤后一个节点都没有"
    assert n_all > len(types) > 0, (n_all, len(types))
    # 每个结果必须确实能吃 IMAGE（免得提示"接住 IMAGE"却给出接不住的节点）
    from workflow.schema import types_compatible
    for t in types:
        spec = registry.get(str(t))
        assert spec is not None
        assert any(types_compatible(DataType.IMAGE, p.type)
                   for p in spec.inputs.values()), t

    # 过滤文本（在"全部"对话框里搜；按类型过滤时会正确地排除 LoadImage ——
    # 它没有输入口，接不住别人递过来的图）
    all_dlg.edit.setText("loadimage")
    assert all_dlg.list.count() >= 1, "全量搜索里找不到 LoadImage"
    assert all_dlg.selected_type() == "LoadImage", all_dlg.selected_type()
    # 中文也能搜（分类名 / 说明都进索引）
    all_dlg.edit.setText("放大")
    assert all_dlg.list.count() >= 1, "中文搜索没命中"
    return True


@check("finish_link_on_empty_requests_search")
def t_link_search():
    from ui.workflow_editor.canvas import GraphScene, GraphView
    scene = GraphScene()
    scene.load_workflow(templates.build("txt2img"))
    view = GraphView(scene)
    got = []
    scene.nodeSearchRequested.connect(lambda pos, hint: got.append(hint))
    # 模拟从 Sampler.latent 拖到空白
    src = scene.node_items["sampler"].ports_out["latent"]
    scene.start_link(src)
    assert scene.finish_link(QPointF(3000, 3000)) is False
    assert got == [DataType.LATENT.value], got
    assert scene._pending_link_src is src
    # 选好节点后应能自动接上
    nid = scene.add_node("VAEDecode", QPointF(3000, 3000))
    assert scene.complete_pending_link(nid) is True
    assert scene.wf.node(nid).inputs.get("latent") == ["sampler", "latent"]
    view.deleteLater()
    return True


@check("drop_image_creates_loadimage")
def t_drop_image():
    from PyQt6.QtGui import QColor, QImage
    from ui.workflow_editor.window import WorkflowEditorWindow
    win = WorkflowEditorWindow(host=None)
    tmp = os.path.join(ROOT, "output", "_tmp_drop_test.png")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    img = QImage(8, 8, QImage.Format.Format_RGB32)
    img.fill(QColor("#336699"))
    assert img.save(tmp)

    try:
        before = len(win.scene.node_items)
        win._on_files_dropped([tmp], QPointF(120, 120))
        assert len(win.scene.node_items) == before + 1
        nid = [k for k in win.scene.node_items
               if win.scene.node_items[k].ntype == "LoadImage"][0]
        node = win.scene.wf.node(nid)
        assert node.params.get("path") == tmp, node.params.get("path")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
        win.close()
    return True


@check("context_menu_has_expected_actions")
def t_context_menu():
    from ui.workflow_editor.window import WorkflowEditorWindow
    win = WorkflowEditorWindow(host=None)
    seen = []
    orig = QMenu.exec

    def fake_exec(self, *a, **k):
        seen.extend(act.text() for act in self.actions() if act.text())
        return None

    QMenu.exec = fake_exec
    try:
        win._on_context_menu(win.view.rect().center())
    finally:
        QMenu.exec = orig
        win.close()
    for want in ("添加节点…", "粘贴", "全选", "删除选中", "适应窗口", "校验"):
        assert want in seen, (want, seen)
    return True


print()
if _fails:
    print("FAILED:", _fails)
    sys.exit(1)
print("WF_EDITOR_UX_OK")
