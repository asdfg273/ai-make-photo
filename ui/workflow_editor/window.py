# ui/workflow_editor/window.py
"""工作流编辑器主窗口（**基础编辑器**，只支持本项目注册的节点）。

能力
----
· 画布上增删节点、连线、改参数（控件由 ParamSpec 自动生成）
· 「从当前界面生成」：把简易模式的设置编译成一张图（workflow.builder）
· 运行前校验 / 资源清单 / 执行 / 逐节点状态着色 / 失败自动定位
· 工作流 JSON 的保存、另存、加载、模板列表
· 只重跑某个节点（选中节点 → 只跑此节点，其余用缓存）
· ComfyUI 工作流导入/导出（workflow.comfy_compat；不认识的节点会明确列出来，
  不会静默少节点）

交互（参考 ComfyUI 补齐的部分）
----------------------------
· Ctrl+Z / Ctrl+Shift+Z   撤销 / 重做（快照式，连拖动位置也能撤）
· Ctrl+C / Ctrl+V / Ctrl+D 复制 / 粘贴 / 再制
· 双击空白 或 Ctrl+F       搜索添加节点
· 端口拖到空白松手          只列能接住该类型的节点，选中即连线
· 拖图片文件进画布          自动建 LoadImage（带预览）；拖 .json 直接打开
· 右键空白                  添加节点 / 粘贴 / 全选 / 适配 / 校验
· 工具栏「模板」            五个能直接跑的示例工作流

仍然**不**做的：子图 / 循环 / 自定义节点类型。任意 Python 节点现在有了
（`nodes/python_nodes.py`），但默认关闭，要在设置里显式打开。
"""
from __future__ import annotations

import logging
import os
import traceback

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QLabel, QMainWindow, QMenu,
                             QMessageBox, QPlainTextEdit, QSplitter,
                             QToolBar, QVBoxLayout, QWidget)

from ui.i18n import ui_i18n
from ui.workflow_editor.canvas import GraphScene, GraphView
from ui.workflow_editor.history import HistoryStack
from ui.workflow_editor.search_dialog import NodeSearchDialog
from ui.workflow_editor.palette import NodePalette
from ui.workflow_editor.param_panel import ParamPanel
from workflow import comfy_compat as comfy
from workflow import graph as G, registry, resource_manager, serializers, templates
from workflow.schema import Workflow

logger = logging.getLogger(__name__)


# ======================================================================
class WorkflowRunThread(QThread):
    """在工作线程里跑工作流，避免界面卡死。"""

    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)

    def __init__(self, wf: Workflow, host, only=None, parent=None):
        super().__init__(parent)
        self.wf = wf
        self.host = host
        self.only = only
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from workflow.executor import Executor, make_context
            ai = getattr(self.host, "ai", None)
            arch = getattr(ai, "current_arch_id", "") or ""
            #  用工作流里记下的策略预设。旧代码硬编码 preset="auto"，
            # 编辑器工具栏上选的「节省显存/平衡/高性能」完全不生效。
            preset = (self.wf.metadata or {}).get("memory_preset") or "auto"
            ctx = make_context(
                host=self.host, ai=ai, preset=preset, arch_id=arch,
                cancel_check=lambda: self._cancel,
                log=lambda m: self.progress.emit(str(m)),
                status=lambda m: self.progress.emit(str(m)),
                progress=lambda a, b: None,
                translate=ui_i18n.tr)
            self.progress.emit(ui_i18n.trf("显存策略预设: {preset}", preset=preset))
            ex = Executor(ctx)
            res = ex.run(self.wf, only=self.only)
            self.finished_ok.emit(res)
        except Exception as e:
            logger.exception("工作流线程异常")
            self.finished_ok.emit({
                "ok": False, "error": f"{type(e).__name__}: {e}",
                "failed_node": "", "states": {}, "outputs": {},
                "cached": [], "seconds": 0.0,
                "trace": traceback.format_exc()})


# ======================================================================
class WorkflowEditorWindow(QMainWindow):
    """节点编辑器。`host` 通常是主窗口（提供 ai / config）。"""

    def __init__(self, host=None, parent=None):
        super().__init__(parent)
        self.host = host
        self.setWindowTitle(ui_i18n.tr("工作流编辑器 — 基础版"))
        self.resize(1360, 860)
        self._thread: WorkflowRunThread | None = None
        self._current_path = ""
        self.history = HistoryStack()
        self.act_undo = None
        self.act_redo = None

        registry.load_builtin_nodes()

        self.scene = GraphScene(self)
        self.view = GraphView(self.scene, self)
        self.palette = NodePalette(self)
        self.params = ParamPanel(self)

        self.scene.history = self.history
        self._build_ui()
        self._connect()
        self.scene.load_workflow(Workflow(name="新工作流"))
        self._log(ui_i18n.tr("双击左侧节点添加；拖端口连线；选中节点改参数。"))
        self._log(ui_i18n.tr("也可以点「从当前界面生成」把简易模式的设置变成一张图。"))

    # ------------------------------------------------------------------
    def _build_ui(self):
        tb = QToolBar("工作流")
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, tip=""):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setToolTip(tip or text)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        act("新建", self.on_new)
        act("打开", self.on_open, "Ctrl+O")
        act("保存", self.on_save, "Ctrl+S")
        act("另存为", self.on_save_as)
        act("导入 ComfyUI", self.on_import_comfy, None,
            "导入 ComfyUI 的 API 格式或界面格式工作流；不支持的节点会列出来")
        act("导出 ComfyUI", self.on_export_comfy, None,
            "导出成 ComfyUI 的 API 格式（只含能完整表达的节点）")
        tb.addSeparator()
        act("从当前界面生成", self.on_build_from_ui, "Ctrl+G",
            "把简易模式当前的模型/提示词/LoRA/高清/修脸等设置编译成工作流")
        act("载入当前界面设置", self.on_apply_ui_to_graph, None,
            "在当前画布上按界面设置重建（会替换现有图）")
        tb.addSeparator()
        act("校验", self.on_validate)
        act("预检", self.on_preflight, "Ctrl+P",
            "运行前预检：模型/LoRA/VAE/ControlNet/显存/输出目录/依赖 一次查完")
        act("资源清单", self.on_resources)
        act("资源依赖", self.on_resources_dialog, None,
            "工作流依赖了哪些模型、哪些缺失、怎么补（按 resource_id 检查）")
        act("运行", self.on_run, "Ctrl+R")
        act("只跑选中节点", self.on_run_selected, "Ctrl+Shift+R")
        act("取消", self.on_cancel)
        tb.addSeparator()
        act("清空状态", lambda: self.scene.reset_states())
        act("删除选中", self.on_delete_selected)
        act("适应窗口", self.on_fit, "Ctrl+0")

        # ---- 参考 ComfyUI 的编辑体验 ----
        tb.addSeparator()
        self.act_undo = act("撤销", self.do_undo, "Ctrl+Z", "撤销上一步编辑")
        self.act_redo = act("重做", self.do_redo, "Ctrl+Shift+Z", "重做刚才撤销的编辑")
        self.act_undo.setEnabled(False)
        self.act_redo.setEnabled(False)
        act("搜索节点", self.on_search_nodes, "Ctrl+F",
            "搜类型 / 分类 / 说明，回车添加")
        self.menu_templates = QMenu(self)
        for key, name, desc in templates.list_templates():
            a = self.menu_templates.addAction(name)
            a.setToolTip(desc)
            a.triggered.connect(lambda _checked=False, _k=key: self.load_template(_k))
        btn_tpl = tb.addAction("模板 ▾")
        btn_tpl.setToolTip("载入一个能直接跑的内置工作流")
        btn_tpl.triggered.connect(self._show_templates_menu)

        # 显存策略选择（能力配置，不是显卡名字）
        tb.addSeparator()
        tb.addWidget(QLabel("显存策略: "))
        self.combo_policy = QComboBox()
        from workflow.memory_policy import PRESET_LABELS
        self.combo_policy.addItem("自动", "auto")
        for k, label in PRESET_LABELS.items():
            self.combo_policy.addItem(label.split("（")[0], k)
        self.combo_policy.setToolTip("执行时使用的显存策略预设")
        tb.addWidget(self.combo_policy)

        # 三栏：节点库 | 画布 | 参数 + 日志
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.params, 3)
        rl.addWidget(QLabel("运行日志"))
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(2000)
        self.txt_log.setFixedHeight(200)
        rl.addWidget(self.txt_log)

        sp1 = QSplitter(Qt.Orientation.Horizontal)
        sp1.addWidget(self.palette)
        sp1.addWidget(self.view)
        sp1.addWidget(right)
        sp1.setSizes([250, 780, 330])

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(sp1)
        self.setCentralWidget(central)

        self.statusBar().showMessage("就绪")

    def _connect(self):
        self.palette.addRequested.connect(self.on_add_node)
        self.scene.statusMessage.connect(self._log)
        self.scene.nodeSelected.connect(self._on_node_selected)
        self.scene.graphChanged.connect(self._update_title)
        # 参数两处可改：卡片内嵌控件 + 右侧面板。
        # 必须双向同步，否则两边显示不一致、而且不知道哪个才是真的。
        self.params.paramChanged.connect(self._on_param_changed)
        self.scene.paramChanged.connect(self._on_param_changed)
        self.scene.nodeSearchRequested.connect(self._on_node_search_requested)
        self.scene.filesDropped.connect(self._on_files_dropped)
        self.scene.graphChanged.connect(self._update_history_actions)
        self.view.customContextMenuRequested.connect(self._on_context_menu)

    # ------------------------------------------------------------ 撤销/重做
    def _update_history_actions(self):
        if self.act_undo is not None:
            self.act_undo.setEnabled(self.history.can_undo)
            tip = self.history.undo_label()
            self.act_undo.setToolTip(ui_i18n.trf("撤销：{what}", what=tip) if tip else "撤销")
        if self.act_redo is not None:
            self.act_redo.setEnabled(self.history.can_redo)
            tip = self.history.redo_label()
            self.act_redo.setToolTip(ui_i18n.trf("重做：{what}", what=tip) if tip else "重做")

    def do_undo(self):
        got = self.history.undo(self.scene.snapshot())
        if not got:
            self._log(ui_i18n.tr("没有可撤销的操作"))
            return
        label, snap = got
        self.scene.restore_snapshot(snap)
        self.params.show_node(self.scene.to_workflow(), "")
        self._update_history_actions()
        self._log(ui_i18n.trf("已撤销：{what}", what=label))

    def do_redo(self):
        got = self.history.redo(self.scene.snapshot())
        if not got:
            self._log(ui_i18n.tr("没有可重做的操作"))
            return
        label, snap = got
        self.scene.restore_snapshot(snap)
        self.params.show_node(self.scene.to_workflow(), "")
        self._update_history_actions()
        self._log(ui_i18n.trf("已重做：{what}", what=label))

    # ------------------------------------------------------------ 搜索添加
    def on_search_nodes(self):
        self._on_node_search_requested(None, "")

    def _on_node_search_requested(self, scene_pos, type_hint: str):
        """双击空白 / Ctrl+F / 拖线到空白 都会走这里。"""
        hint = str(type_hint or "")
        title = ui_i18n.tr("添加节点")
        if hint:
            title = ui_i18n.trf("接住 {type} 的节点", type=hint)
        ntype = NodeSearchDialog.pick(self, type_hint=hint, title=title)
        if not ntype:
            self.scene._pending_link_src = None
            return
        pos = scene_pos
        if pos is None:
            pos = self.view.mapToScene(self.view.viewport().rect().center())
        nid = self.scene.add_node(ntype, pos)
        if not nid:
            return
        if self.scene._pending_link_src is not None:
            self.scene.complete_pending_link(nid)
        for it in self.scene.node_items.values():
            it.setSelected(it.node_id == nid)
        self.scene.nodeSelected.emit(nid)

    # ------------------------------------------------------------ 拖入文件
    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    def _on_files_dropped(self, paths, scene_pos):
        paths = [p for p in (paths or []) if isinstance(p, str)]
        if not paths:
            return
        # 先处理工作流 JSON：只取第一个
        for p in paths:
            if p.lower().endswith(".json"):
                if not self._confirm_discard():
                    return
                try:
                    wf = serializers.load(p)
                except Exception as e:
                    QMessageBox.critical(self, ui_i18n.tr("打开失败"),
                                         f"{type(e).__name__}: {e}")
                    return
                self.history.push(ui_i18n.tr("打开工作流"), self.scene.snapshot())
                self.scene.load_workflow(wf)
                self._current_path = p
                self._update_title()
                self._update_history_actions()
                self._log(ui_i18n.trf("已载入 {path}（{count} 节点）",
                                      path=p, count=len(wf.nodes)))
                return
        imgs = [p for p in paths if p.lower().endswith(self.IMAGE_EXTS)]
        if not imgs:
            self._log(ui_i18n.tr("拖进来的文件里没有图片或工作流 JSON"))
            return
        base = scene_pos
        for i, p in enumerate(imgs):
            pos = base
            if pos is not None:
                from PyQt6.QtCore import QPointF
                pos = QPointF(pos.x() + i * (236 + 40), pos.y())
            nid = self.scene.add_node("LoadImage", pos)
            if nid:
                self.scene.set_node_param(nid, "path", p)
                self._log(ui_i18n.trf("已加入图片 {name}", name=os.path.basename(p)))

    # ------------------------------------------------------------ 右键菜单
    def _on_context_menu(self, pos):
        menu = QMenu(self)
        a = menu.addAction(ui_i18n.tr("添加节点…"))
        a.triggered.connect(lambda: self._on_node_search_requested(
            self.view.mapToScene(pos), ""))
        clip = getattr(self.scene, "clipboard_payload", None)
        a_paste = menu.addAction(ui_i18n.tr("粘贴"))
        a_paste.setEnabled(bool(clip))
        a_paste.triggered.connect(
            lambda: clip and self.scene.paste_payload(
                clip, self.view.mapToScene(pos)))
        menu.addSeparator()
        a = menu.addAction(ui_i18n.tr("全选"))
        a.triggered.connect(self.scene.select_all)
        a = menu.addAction(ui_i18n.tr("删除选中"))
        a.triggered.connect(self.on_delete_selected)
        a = menu.addAction(ui_i18n.tr("从当前界面生成"))
        a.triggered.connect(self.on_build_from_ui)
        menu.addSeparator()
        a = menu.addAction(ui_i18n.tr("适应窗口"))
        a.triggered.connect(self.on_fit)
        a = menu.addAction(ui_i18n.tr("校验"))
        a.triggered.connect(self.on_validate)
        menu.exec(self.view.mapToGlobal(pos))

    # ------------------------------------------------------------ 模板
    def _show_templates_menu(self):
        self.menu_templates.exec(self.mapToGlobal(self.rect().center()))

    def load_template(self, key: str):
        if not self._confirm_discard():
            return
        try:
            wf = templates.build(key)
        except Exception as e:
            QMessageBox.critical(self, ui_i18n.tr("模板载入失败"),
                                 f"{type(e).__name__}: {e}")
            return
        self.history.push(ui_i18n.trf("载入模板 {name}", name=wf.name),
                          self.scene.snapshot())
        self.scene.load_workflow(wf)
        self._current_path = ""
        self.params.show_node(self.scene.to_workflow(), "")
        self._update_title()
        self._update_history_actions()
        self._log(ui_i18n.trf("已载入模板：{name}（{count} 节点）",
                              name=wf.name, count=len(wf.nodes)))
        self.on_fit()

    def _on_param_changed(self, node_id: str, name: str, value):
        """把改动写回数据层，并同步到另一处控件。"""
        self.scene.set_node_param(node_id, name, value)
        # 同步右侧面板（若显示的就是这个节点）
        if getattr(self.params, "_node_id", "") == node_id:
            self.params.set_param(name, value)
        self.scene.graphChanged.emit()

    # ------------------------------------------------------------------
    def _log(self, msg: str):
        self.txt_log.appendPlainText(str(msg))
        self.statusBar().showMessage(str(msg)[:120])

    def _update_title(self):
        wf = self.scene.to_workflow()
        name = os.path.basename(self._current_path) if self._current_path else wf.name
        self.setWindowTitle(ui_i18n.trf(
            "工作流编辑器 — {name}（{nodes} 节点 / {links} 连线）",
            name=name, nodes=len(wf.nodes), links=len(wf.links)))

    def _on_node_selected(self, node_id: str):
        wf = self.scene.to_workflow(self.scene.wf.name)
        self.params.show_node(wf, node_id)

    # ------------------------------------------------------------------ 编辑
    def on_add_node(self, ntype: str):
        center = self.view.mapToScene(self.view.viewport().rect().center())
        self.scene.add_node(ntype, center)

    def on_delete_selected(self):
        for lk in self.scene.selected_links():
            self.scene._drop_link(lk)
            self.scene.graphChanged.emit()
        for nid in [i.node_id for i in self.scene.selectedItems()
                    if hasattr(i, "node_id")]:
            self.scene.remove_node(nid)

    def on_fit(self):
        self.view.fitInView(self.scene.itemsBoundingRect(),
                            Qt.AspectRatioMode.KeepAspectRatio)

    # ------------------------------------------------------------------ 文件
    def on_new(self):
        if not self._confirm_discard():
            return
        self.scene.load_workflow(Workflow(name="新工作流"))
        self._current_path = ""
        self._update_title()
        self._log(ui_i18n.tr("已新建工作流"))

    def _confirm_discard(self) -> bool:
        if not self.scene.node_items:
            return True
        r = QMessageBox.question(
            self, ui_i18n.tr("确认"), ui_i18n.tr("当前工作流未保存，确定要放弃吗？"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return r == QMessageBox.StandardButton.Yes

    def on_open(self):
        if not self._confirm_discard():
            return
        d = serializers.workflows_dir()
        path, _ = QFileDialog.getOpenFileName(
            self, ui_i18n.tr("打开工作流"), d, ui_i18n.tr("工作流 JSON (*.json)"))
        if not path:
            return
        try:
            wf = serializers.load(path)
        except Exception as e:
            QMessageBox.critical(self, ui_i18n.tr("打开失败"), f"{type(e).__name__}: {e}")
            return
        self.scene.load_workflow(wf)
        self._current_path = path
        self._update_title()
        self._log(ui_i18n.trf("已载入 {path}（{count} 节点）", path=path, count=len(wf.nodes)))
        self.on_validate(silent_ok=True)

    def on_save(self):
        if self._current_path:
            self._save_to(self._current_path)
            return
        self.on_save_as()

    def on_save_as(self):
        d = serializers.workflows_dir()
        default = os.path.join(d, serializers._safe_name(self.scene.wf.name))
        path, _ = QFileDialog.getSaveFileName(
            self, ui_i18n.tr("保存工作流"), default, ui_i18n.tr("工作流 JSON (*.json)"))
        if not path:
            return
        self._save_to(path)

    def _save_to(self, path: str):
        wf = self.scene.to_workflow()
        try:
            wf.name = os.path.splitext(os.path.basename(path))[0]
            serializers.save(wf, path)
        except Exception as e:
            QMessageBox.critical(self, ui_i18n.tr("保存失败"), f"{type(e).__name__}: {e}")
            return
        self._current_path = path
        self.scene.wf = wf
        self._update_title()
        self._log(ui_i18n.trf("已保存 {path}", path=path))

    # ------------------------------------------------------- ComfyUI 互通
    def on_import_comfy(self):
        """导入 ComfyUI 工作流（API 或 UI 格式）。"""
        if not self._confirm_discard():
            return
        d = serializers.workflows_dir()
        path, _ = QFileDialog.getOpenFileName(
            self, ui_i18n.tr("导入 ComfyUI 工作流"), d,
            ui_i18n.tr("ComfyUI JSON (*.json)"))
        if not path:
            return
        try:
            wf, rep = comfy.load_file(path)
        except Exception as e:
            QMessageBox.critical(self, ui_i18n.tr("导入失败"),
                                 f"{type(e).__name__}: {e}")
            return
        text = rep.summary_text()
        if not wf.nodes:
            QMessageBox.warning(self, ui_i18n.tr("没导入任何节点"), text)
            return
        self.scene.load_workflow(wf)
        self._current_path = ""
        self._update_title()
        self._log(ui_i18n.trf("已从 ComfyUI 导入 {n} 个节点：{p}",
                              n=len(wf.nodes), p=path))
        for line in text.splitlines():
            self._log("  " + line)
        if rep.unsupported:
            QMessageBox.warning(self, ui_i18n.tr("部分节点不支持"), text)
        else:
            QMessageBox.information(self, ui_i18n.tr("导入完成"), text)
        self.on_validate(silent_ok=True)

    def on_export_comfy(self):
        """导出成 ComfyUI API 格式。"""
        wf = self.scene.to_workflow()
        data, rep = comfy.export_prompt(wf)
        text = rep.summary_text()
        if not data:
            QMessageBox.warning(self, ui_i18n.tr("没有可导出的节点"), text)
            return
        d = serializers.workflows_dir()
        default = os.path.join(d, serializers._safe_name(wf.name) + "_comfy.json")
        path, _ = QFileDialog.getSaveFileName(
            self, ui_i18n.tr("导出为 ComfyUI"), default,
            ui_i18n.tr("ComfyUI JSON (*.json)"))
        if not path:
            return
        try:
            import json
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, ui_i18n.tr("导出失败"),
                                 f"{type(e).__name__}: {e}")
            return
        self._log(ui_i18n.trf("已导出 ComfyUI 格式：{p}", p=path))
        if rep.unsupported or rep.warnings:
            QMessageBox.warning(self, ui_i18n.tr("导出完成（有省略）"), text)
        else:
            QMessageBox.information(self, ui_i18n.tr("导出完成"), text)

    # ------------------------------------------------------------------ 生成
    def on_build_from_ui(self):
        if self.host is None:
            QMessageBox.information(
                self, ui_i18n.tr("无宿主"),
                ui_i18n.tr("编辑器没有绑定主窗口，无法从界面生成。"))
            return
        try:
            from workflow import builder
            wf = builder.build_from_ui(self.host)
        except Exception as e:
            QMessageBox.warning(
                self, ui_i18n.tr("生成失败"),
                f"{type(e).__name__}: {e}\n\n"
                + ui_i18n.tr("请确认主窗口已就绪（模型列表已加载）。"))
            return
        self.scene.load_workflow(wf)
        self._current_path = ""
        self._update_title()
        self._log(ui_i18n.trf(
            "已从当前界面生成工作流：{description}", description=builder.describe(wf)))
        self._log_ui_drop_warnings(wf)
        self.on_validate(silent_ok=True)

    def _log_ui_drop_warnings(self, wf):
        """把"界面上开了、但没能进工作流"的项**明确说出来**。

         这是 P0 静默降级的另一半：以前界面勾了 ControlNet/IP-Adapter
        却因为缺参考图而没进图，既不报错也不警告，用户以为用了。
        现在只要 builder 记了 warnings，就在日志里红字列出 + 弹窗提示。
        """
        warns = list((wf.metadata or {}).get("warnings") or [])
        if not warns:
            return
        for w in warns:
            self._log(ui_i18n.trf("界面上的设置没能进工作流：{warning}", warning=w))
        QMessageBox.warning(
            self, ui_i18n.tr("有设置没进工作流"),
            ui_i18n.tr("以下设置在界面上是开着的，但没有进入生成的工作流（通常是缺少参考图/姿势图）：")
            + "\n\n" + "\n".join(f"· {w}"for w in warns)
            + "\n\n" + ui_i18n.tr("补齐后重新点「从当前界面生成」即可。"))

    def on_apply_ui_to_graph(self):
        if self.scene.node_items and not self._confirm_discard():
            return
        self.on_build_from_ui()

    # ------------------------------------------------------------------ 校验
    def on_validate(self, silent_ok: bool = False) -> bool:
        wf = self.scene.to_workflow()
        issues = G.validate(wf, translate=ui_i18n.tr)
        errs = [i for i in issues if i.level == "error"]
        warns = [i for i in issues if i.level == "warning"]
        for i in issues:
            self._log(ui_i18n.tr("错误"if i.level == "error"else "提醒") + str(i))
        if errs:
            self._log(ui_i18n.trf("校验未通过：{count} 个错误", count=len(errs)))
            QMessageBox.warning(
                self, ui_i18n.tr("校验未通过"),
                ui_i18n.trf("发现 {count} 个问题：\n\n{details}",
                            count=len(errs),
                            details="\n".join(f"· {i}"for i in errs[:12])))
            return False
        if warns:
            self._log(ui_i18n.trf("校验通过（{count} 条提醒）", count=len(warns)))
        elif not silent_ok:
            self._log(ui_i18n.tr("校验通过"))
        return True

    def on_resources(self):
        wf = self.scene.to_workflow()
        try:
            rep = resource_manager.summary(wf, translate=ui_i18n.tr)
        except Exception as e:
            QMessageBox.warning(self, ui_i18n.tr("资源清单失败"), str(e))
            return
        for line in rep["text"].splitlines():
            self._log(line)
        QMessageBox.information(self, ui_i18n.tr("资源清单"), rep["text"])

    def on_resources_dialog(self):
        """打开资源依赖面板（缺失资源可在此补全）。"""
        try:
            from ui.workflow_editor.resource_panel import ResourceDialog
            dlg = ResourceDialog(self.scene.to_workflow(), self)
            dlg.exec()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, ui_i18n.tr("资源面板打开失败"), str(e))

    def on_preflight(self):
        """只做预检（不执行），把错误/警告/提示一次列清。"""
        wf = self.scene.to_workflow()
        wf.metadata["memory_preset"] = self.combo_policy.currentData()
        try:
            from workflow.preflight import run_preflight
            res = run_preflight(wf, host=self.host,
                                ai=getattr(self.host, "ai", None),
                                translate=ui_i18n.tr)
        except Exception as e:
            QMessageBox.critical(self, ui_i18n.tr("预检失败"), f"{type(e).__name__}: {e}")
            return
        text = res.to_text(translate=ui_i18n.tr)
        for line in text.splitlines():
            self._log(line)
        # 错误节点直接标红并选中，方便定位
        for it in res.errors:
            item = self.scene.node_items.get(it.node_id)
            if item is not None:
                item.set_status("error", it.message)
        if res.errors:
            QMessageBox.warning(self, ui_i18n.tr("预检未通过"), text)
        else:
            QMessageBox.information(self, ui_i18n.tr("预检结果"), text)

    # ------------------------------------------------------------------ 运行
    def _policy_kwargs(self) -> dict:
        preset = self.combo_policy.currentData()
        return {"preset": preset}

    def on_run(self):
        self._start_run(only=None)

    def on_run_selected(self):
        nid = self.scene.selected_node_id()
        if not nid:
            QMessageBox.information(
                self, ui_i18n.tr("未选中"), ui_i18n.tr("请先在画布上选中一个节点。"))
            return
        self._log(ui_i18n.trf("只重跑 {node}（其余节点用缓存）", node=nid))
        self._start_run(only=[nid])

    def _start_run(self, only=None):
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(
                self, ui_i18n.tr("正在运行"), ui_i18n.tr("上一个任务还没结束。"))
            return
        if not self.on_validate(silent_ok=True):
            return
        wf = self.scene.to_workflow()
        # 把策略预设塞进 metadata，供执行器与预检读取（不改全局配置）
        wf.metadata["memory_preset"] = self.combo_policy.currentData()

        # ---- 运行前预检：在真加载模型之前把问题一次报完 ----
        if not self._preflight(wf):
            return

        self.scene.reset_states()

        # 在画布上先标出"准备执行/复用"的样子
        for nid in self.scene.node_items:
            self.scene.node_items[nid].set_status("idle")

        self._log(ui_i18n.tr("开始执行…"))
        self._thread = WorkflowRunThread(wf, self.host, only=only, parent=self)
        self._thread.progress.connect(self._log)
        self._thread.finished_ok.connect(self._on_run_done)
        self._thread.start()

    def _preflight(self, wf) -> bool:
        """返回是否继续执行。错误级别必须处理，警告/提示只展示。"""
        try:
            from workflow.preflight import run_preflight
            res = run_preflight(wf, host=self.host,
                                ai=getattr(self.host, "ai", None),
                                translate=ui_i18n.tr)
        except Exception as e:
            self._log(ui_i18n.trf("预检失败（继续运行）: {error}", error=e))
            return True

        text = res.to_text(translate=ui_i18n.tr)
        for line in text.splitlines():
            self._log(line)
        if res.errors:
            QMessageBox.warning(self, ui_i18n.tr("预检未通过"), text)
            return False
        if res.warnings or res.hints:
            # 只是提醒，不阻断；用户可选择继续
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Information)
            box.setWindowTitle(ui_i18n.tr("预检通过（有提醒）"))
            box.setText(ui_i18n.tr("可以运行，但有以下提醒："))
            box.setInformativeText(text)
            go = box.addButton(ui_i18n.tr("继续运行"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(ui_i18n.tr("先改一下"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is not go:
                return False
        else:
            self._log(ui_i18n.tr("预检通过"))
        return True

    def on_cancel(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self._log(ui_i18n.tr("已请求取消（当前节点跑完就停）"))
        else:
            self._log(ui_i18n.tr("（当前没有运行中的任务）"))

    def _on_run_done(self, res: dict):
        states = res.get("states") or {}
        self.scene.apply_states(states, res.get("cached"))
        # 把每个节点产出的图渲染回它的卡片（"图生图加了参考图能在模块里看到图"）
        try:
            self.scene.push_previews(res.get("images") or {})
        except Exception as e:
            logger.debug(f"节点预览推送失败: {e}")
        secs = res.get("seconds", 0)
        if res.get("cancelled"):
            self._log(ui_i18n.tr("已取消"))
        elif res.get("ok"):
            self._log(ui_i18n.trf("完成，用时 {seconds}s", seconds=f"{secs:.1f}"))
            if res.get("cached"):
                self._log(ui_i18n.trf("复用缓存节点 {count} 个", count=len(res["cached"])))
            outs = res.get("outputs") or {}
            for k, v in outs.items():
                if isinstance(v, str) and os.path.exists(v):
                    self._log(f"{k} → {v}")
            if res.get("cache_stats"):
                st = res["cache_stats"]
                self._log(ui_i18n.trf(
                    "缓存 {count} 项，命中率 {rate}%",
                    count=st["entries"], rate=f"{st['hit_rate'] * 100:.0f}"))
        else:
            failed = res.get("failed_node") or "?"
            self._log(ui_i18n.trf(
                "失败于节点 [{node}]：{error}", node=failed, error=res.get("error", "")))
            QMessageBox.critical(
                self, ui_i18n.tr("执行失败"),
                ui_i18n.trf("节点 [{node}] 执行失败：\n\n{error}",
                            node=failed, error=res.get("error", "")))
        if res.get("trace"):
            logger.error("工作流 traceback:\n%s", res["trace"])
        self._thread = None

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            r = QMessageBox.question(
                self, ui_i18n.tr("正在运行"), ui_i18n.tr("工作流还在运行，确定关闭吗？"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._thread.cancel()
            self._thread.wait(3000)
        super().closeEvent(event)
