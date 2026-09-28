# ui/workflow_editor/canvas.py
"""工作流画布：QGraphicsScene 上的节点 / 端口 / 连线。

只支持**本项目自己注册的节点**（`workflow.registry`）。刻意不做
"任意 Python 节点"：那会让调试、显存管理和错误定位迅速失控。

交互
----
· 拖节点      移动位置（位置存在 workflow.metadata["ui_positions"]，不污染参数）
· 从输出端口拖到输入端口   连线（类型不兼容会拒绝并提示）
· 点连线选中 + Del        断开
· 选中节点 + Del          删除节点
· 双击节点              居中/展开（由窗口层接参数面板）

节点状态直接用颜色表示（运行中/成功/复用缓存/失败），失败节点会被
标红并自动选中 —— 这是"失败时能看出是哪个节点"的界面落点。
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath, QPen)
from PyQt6.QtWidgets import (QGraphicsItem, QGraphicsPathItem, QGraphicsScene,
                             QGraphicsView)

from workflow import registry
from workflow.schema import DataType, Node, Workflow, types_compatible

logger = logging.getLogger(__name__)

NODE_W = 190
PORT_R = 5.0
HEADER_H = 26.0
ROW_H = 20.0
PAD = 8.0

# 各状态配色
STATUS_COLORS = {
    "idle":     ("#313244", "#585b70"),   # 背景, 边框
    "running":  ("#45475a", "#f9e2af"),
    "ok":       ("#313244", "#a6e3a1"),
    "cached":   ("#2b3440", "#89b4fa"),
    "error":    ("#4a2b33", "#f38ba8"),
    "skipped":  ("#313244", "#6c7086"),
}

# 类型配色（端口小圆点）
TYPE_COLORS = {
    DataType.MODEL: "#cba6f7", DataType.VAE: "#89b4fa",
    DataType.TEXT_COND: "#f9e2af", DataType.IMAGE: "#a6e3a1",
    DataType.LATENT: "#fab387", DataType.MASK: "#eba0ac",
    DataType.CONTROL_IMAGE: "#94e2d5", DataType.LORA: "#f5c2e7",
    DataType.SCHEDULER: "#b4befe", DataType.NUMBER: "#89dceb",
    DataType.BOOLEAN: "#89dceb", DataType.METADATA: "#7f849c",
}


def _type_color(t: DataType) -> str:
    return TYPE_COLORS.get(t, "#9399b2")


# ======================================================================
class PortItem(QGraphicsItem):
    """一个端口圆点。`is_output` 决定它是连线起点还是终点。"""

    def __init__(self, node_item: "NodeItem", name: str, dtype: DataType,
                 is_output: bool):
        super().__init__(node_item)
        self.node_item = node_item
        self.name = name
        self.dtype = dtype
        self.is_output = is_output
        self.connected = False
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{name} : {dtype.value}")
        self._hover = False

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_R - 2, -PORT_R - 2, (PORT_R + 2) * 2, (PORT_R + 2) * 2)

    def paint(self, painter: QPainter, option, widget=None):
        r = PORT_R + (1.5 if self._hover else 0.0)
        painter.setBrush(QBrush(QColor(_type_color(self.dtype))))
        painter.setPen(QPen(QColor("#11111b") if not self._hover
                            else QColor("#ffffff"), 1.6))
        painter.drawEllipse(QPointF(0, 0), r, r)

    def hoverEnterEvent(self, e):
        self._hover = True
        self.update()
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e):
        self._hover = False
        self.update()
        super().hoverLeaveEvent(e)


# ======================================================================
class NodeItem(QGraphicsItem):
    """一个节点方块：标题 + 端口行。可拖动。"""

    def __init__(self, node_id: str, ntype: str):
        super().__init__()
        self.node_id = node_id
        self.ntype = ntype
        self.spec = registry.get(ntype)
        self.status = "idle"
        self.error = ""
        self.cached = False
        self.ports_in: dict[str, PortItem] = {}
        self.ports_out: dict[str, PortItem] = {}
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(1)

        self._layout_ports()
        title = self.spec.description.splitlines()[0] if self.spec.description else ""
        self.setToolTip(f"{ntype}\n{title}\n\n{self.spec.description}")

    # ---- 几何 ----
    def _layout_ports(self):
        n_in = len(self.spec.inputs)
        n_out = len(self.spec.outputs)
        rows = max(n_in, n_out, 1)
        self.h = HEADER_H + rows * ROW_H + PAD

        for i, (name, ps) in enumerate(self.spec.inputs.items()):
            p = PortItem(self, name, ps.type, is_output=False)
            p.setPos(0, HEADER_H + i * ROW_H + ROW_H / 2)
            self.ports_in[name] = p
        for i, (name, ps) in enumerate(self.spec.outputs.items()):
            p = PortItem(self, name, ps.type, is_output=True)
            p.setPos(NODE_W, HEADER_H + i * ROW_H + ROW_H / 2)
            self.ports_out[name] = p

    def boundingRect(self) -> QRectF:
        return QRectF(-2, -2, NODE_W + 4, self.h + 4)

    def paint(self, painter: QPainter, option, widget=None):
        bg, border = STATUS_COLORS.get(self.status, STATUS_COLORS["idle"])
        if self.isSelected():
            border = "#f5e0dc"

        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, NODE_W, self.h), 6, 6)
        painter.setBrush(QBrush(QColor(bg)))
        painter.setPen(QPen(QColor(border), 2.0 if self.isSelected() else 1.4))
        painter.drawPath(path)

        # 标题条
        painter.setBrush(QBrush(QColor("#181825")))
        painter.setPen(Qt.PenStyle.NoPen)
        head = QPainterPath()
        head.addRoundedRect(QRectF(0, 0, NODE_W, HEADER_H), 6, 6)
        painter.drawPath(head)

        painter.setPen(QPen(QColor("#cdd6f4")))
        f = QFont()
        f.setBold(True)
        f.setPointSize(9)
        painter.setFont(f)
        label = self.ntype
        if self.cached:
            label += " ♻"
        painter.drawText(QRectF(PAD, 0, NODE_W - PAD * 2, HEADER_H),
                         int(Qt.AlignmentFlag.AlignVCenter
                             | Qt.AlignmentFlag.AlignLeft), label)

        # 端口标签
        f2 = QFont()
        f2.setPointSize(8)
        painter.setFont(f2)
        painter.setPen(QPen(QColor("#a6adc8")))
        for i, (name, ps) in enumerate(self.spec.inputs.items()):
            y = HEADER_H + i * ROW_H
            tag = name if not ps.optional else f"{name}?"
            painter.drawText(QRectF(PAD + 8, y, NODE_W / 2, ROW_H),
                             int(Qt.AlignmentFlag.AlignVCenter
                                 | Qt.AlignmentFlag.AlignLeft), tag)
        for i, name in enumerate(self.spec.outputs):
            y = HEADER_H + i * ROW_H
            painter.drawText(QRectF(NODE_W / 2, y, NODE_W / 2 - PAD - 8, ROW_H),
                             int(Qt.AlignmentFlag.AlignVCenter
                                 | Qt.AlignmentFlag.AlignRight), name)

        if self.error:
            painter.setPen(QPen(QColor("#f38ba8")))
            painter.drawText(QRectF(0, self.h + 1, NODE_W, 14),
                             int(Qt.AlignmentFlag.AlignLeft), "⚠ 执行失败")

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            sc = self.scene()
            if sc is not None and hasattr(sc, "refresh_links"):
                sc.refresh_links()
        return super().itemChange(change, value)

    def set_status(self, status: str, error: str = "", cached: bool = False):
        self.status = status
        self.error = error
        self.cached = cached
        self.update()


# ======================================================================
class LinkItem(QGraphicsPathItem):
    """一条贝塞尔连线（带箭头方向感）。可选中。"""

    def __init__(self, src: PortItem, dst: PortItem):
        super().__init__()
        self.src = src
        self.dst = dst
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(0)
        self._shape_path = QPainterPath()
        self.update_path()

    def update_path(self):
        p1 = self.src.scenePos()
        p2 = self.dst.scenePos()
        dx = max(40.0, abs(p2.x() - p1.x()) * 0.5)
        path = QPainterPath(p1)
        path.cubicTo(QPointF(p1.x() + dx, p1.y()),
                     QPointF(p2.x() - dx, p2.y()), p2)
        self.setPath(path)
        self._shape_path = path
        self.prepareGeometryChange()

    def shape(self):
        # 加宽可点击区域：细线很难点中（用 12px 宽的描边做命中区）
        from PyQt6.QtGui import QPainterPathStroker
        s = QPainterPathStroker()
        s.setWidth(12)
        return s.createStroke(self.path())

    def paint(self, painter: QPainter, option, widget=None):
        col = QColor("#f5e0dc") if self.isSelected() else QColor(
            _type_color(self.src.dtype))
        painter.setPen(QPen(col, 2.4 if self.isSelected() else 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path())


# ======================================================================
class GraphScene(QGraphicsScene):
    """画布场景：与 `workflow.schema.Workflow` 双向转换 + 编辑操作。"""

    graphChanged = pyqtSignal()
    nodeSelected = pyqtSignal(str)          # node_id，"" 表示取消选择
    statusMessage = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(QColor("#11111b")))
        self.wf = Workflow(name="未命名工作流")
        self.node_items: dict[str, NodeItem] = {}
        self.link_items: list[LinkItem] = []
        self._temp_link: QGraphicsPathItem | None = None
        self._link_src: PortItem | None = None
        self.selectionChanged.connect(self._on_selection)

    # ---------------------------------------------------------------- 载入
    def load_workflow(self, wf: Workflow, reset_view: bool = True):
        self.clear_items()
        self.wf = wf
        positions = wf.metadata.get("ui_positions") or {}
        auto_x, auto_y = 40.0, 40.0
        for n in wf.nodes:
            if not registry.exists(n.type):
                logger.warning(f"跳过未注册节点 {n.id} ({n.type})")
                continue
            item = NodeItem(n.id, n.type)
            pos = positions.get(n.id)
            if pos:
                item.setPos(float(pos[0]), float(pos[1]))
            else:
                item.setPos(auto_x, auto_y)
                auto_y += item.h + 30
                if auto_y > 700:
                    auto_y = 40.0
                    auto_x += NODE_W + 90
            self.addItem(item)
            self.node_items[n.id] = item
        # 先摆好再连线
        for (src, sp, dst, dp) in wf.links:
            self._make_link(src, sp, dst, dp)

    def clear_items(self):
        self._link_src = None
        self._temp_link = None
        for lk in self.link_items:
            self.removeItem(lk)
        self.link_items.clear()
        for it in self.node_items.values():
            self.removeItem(it)
        self.node_items.clear()
        self.clearSelection()

    # ---------------------------------------------------------------- 导出
    def to_workflow(self, name: str | None = None) -> Workflow:
        wf = Workflow(name=name or self.wf.name)
        wf.schema_version = self.wf.schema_version
        wf.metadata = dict(self.wf.metadata)
        wf.metadata["ui_positions"] = {
            nid: [it.pos().x(), it.pos().y()]
            for nid, it in self.node_items.items()}
        for nid, it in self.node_items.items():
            src = self.wf.node(nid)
            node = Node(nid, it.ntype,
                        dict(src.inputs) if src else {},
                        dict(src.params) if src else {})
            wf.nodes.append(node)
        for lk in self.link_items:
            wf.links.append((lk.src.node_item.node_id, lk.src.name,
                             lk.dst.node_item.node_id, lk.dst.name))
        return wf

    # ---------------------------------------------------------------- 节点
    def add_node(self, ntype: str, pos: QPointF | None = None) -> str | None:
        if not registry.exists(ntype):
            self.statusMessage.emit(f"⚠️ 未注册的节点类型: {ntype}")
            return None
        nid = self._unique_id(ntype)
        spec = registry.get(ntype)
        params = dict(spec.param_defaults())
        self.wf.nodes.append(Node(nid, ntype, {}, params))
        item = NodeItem(nid, ntype)
        item.setPos(pos if pos is not None else QPointF(60, 60))
        self.addItem(item)
        self.node_items[nid] = item
        self.graphChanged.emit()
        self.statusMessage.emit(f"➕ 已添加 {ntype} ({nid})")
        return nid

    def _unique_id(self, ntype: str) -> str:
        base = ntype[:1].lower() + ntype[1:] if ntype else "node"
        used = set(self.node_items)
        i = 1
        while f"{base}_{i}" in used:
            i += 1
        return f"{base}_{i}"

    def remove_node(self, nid: str):
        item = self.node_items.pop(nid, None)
        if item is None:
            return
        for lk in list(self.link_items):
            if lk.src.node_item.node_id == nid or lk.dst.node_item.node_id == nid:
                self._drop_link(lk)
        self.removeItem(item)
        for i, n in enumerate(self.wf.nodes):
            if n.id == nid:
                self.wf.nodes.pop(i)
                break
        self.graphChanged.emit()
        self.statusMessage.emit(f"🗑 已删除 {nid}")

    # ---------------------------------------------------------------- 连线
    def _make_link(self, src_id: str, src_port: str,
                   dst_id: str, dst_port: str) -> bool:
        a = self.node_items.get(src_id)
        b = self.node_items.get(dst_id)
        if not a or not b:
            return False
        pa = a.ports_out.get(src_port)
        pb = b.ports_in.get(dst_port)
        if not pa or not pb:
            return False
        lk = LinkItem(pa, pb)
        self.addItem(lk)
        self.link_items.append(lk)
        pa.connected = pb.connected = True
        # 同步到数据层（覆盖同一输入口的旧连线）
        node = self.wf.node(dst_id)
        if node is not None:
            node.inputs[dst_port] = [src_id, src_port]
        self.wf.links = [l for l in self.wf.links
                         if not (l[2] == dst_id and l[3] == dst_port)]
        self.wf.links.append((src_id, src_port, dst_id, dst_port))
        return True

    def _drop_link(self, lk: LinkItem):
        try:
            self.link_items.remove(lk)
        except ValueError:
            pass
        self.removeItem(lk)
        dst = lk.dst.node_item.node_id
        port = lk.dst.name
        node = self.wf.node(dst)
        if node is not None:
            node.inputs.pop(port, None)
        self.wf.links = [l for l in self.wf.links
                         if not (l[2] == dst and l[3] == port)]

    def connect_ports(self, src: PortItem, dst: PortItem) -> bool:
        """类型检查后连线。类型不兼容时拒绝并提示。"""
        if src is dst or src.node_item is dst.node_item:
            self.statusMessage.emit("⚠️ 不能连接到同一个节点")
            return False
        if src.is_output == dst.is_output:
            self.statusMessage.emit("⚠️ 必须从「输出」连到「输入」")
            return False
        if not src.is_output:                 # 允许反向拖
            src, dst = dst, src
        a, b = src.dtype, dst.dtype
        if not types_compatible(a, b):
            self.statusMessage.emit(
                f"❌ 类型不匹配：{a.value} → {b.value}（不会自动转换）")
            return False
        # 一个输入口只接一条线
        for lk in list(self.link_items):
            if lk.dst is dst:
                self._drop_link(lk)
        ok = self._make_link(src.node_item.node_id, src.name,
                             dst.node_item.node_id, dst.name)
        if ok:
            self.refresh_links()
            self.graphChanged.emit()
            self.statusMessage.emit(
                f"🔗 {src.node_item.node_id}.{src.name} → "
                f"{dst.node_item.node_id}.{dst.name}")
        return ok

    def refresh_links(self):
        for lk in self.link_items:
            lk.update_path()

    # ---------------------------------------------------------------- 拖拽连线
    def start_link(self, port: PortItem):
        self._link_src = port
        pen = QPen(QColor(_type_color(port.dtype)), 2, Qt.PenStyle.DashLine)
        self._temp_link = self.addPath(QPainterPath(port.scenePos()), pen)
        self._temp_link.setZValue(5)

    def drag_link(self, scene_pos: QPointF):
        if self._link_src is None or self._temp_link is None:
            return
        p1 = self._link_src.scenePos()
        path = QPainterPath(p1)
        path.lineTo(scene_pos)
        self._temp_link.setPath(path)

    def finish_link(self, scene_pos: QPointF) -> bool:
        src = self._link_src
        self._link_src = None
        if self._temp_link is not None:
            self.removeItem(self._temp_link)
            self._temp_link = None
        if src is None:
            return False
        target = None
        for item in self.items(scene_pos):
            if isinstance(item, PortItem):
                target = item
                break
        if target is None:
            self.statusMessage.emit("（未连到端口，已取消）")
            return False
        return self.connect_ports(src, target)

    # ---------------------------------------------------------------- 状态
    def apply_states(self, states: dict, cached: list | None = None):
        """把执行器的 per-node 状态刷到画布上（失败节点自动选中）。"""
        cached = set(cached or [])
        failed = ""
        for nid, item in self.node_items.items():
            st = (states or {}).get(nid) or {}
            status = st.get("status", "idle")
            if status == "ok" and nid in cached:
                status = "cached"
            item.set_status(status, st.get("error", ""), nid in cached)
            if status == "error":
                failed = nid
        if failed:
            self.clearSelection()
            it = self.node_items.get(failed)
            if it is not None:
                it.setSelected(True)
                it.setZValue(10)
            self.nodeSelected.emit(failed)

    def reset_states(self):
        for it in self.node_items.values():
            it.set_status("idle")

    # ---------------------------------------------------------------- 选择
    def _on_selection(self):
        sel = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
        self.nodeSelected.emit(sel[0].node_id if sel else "")

    def selected_node_id(self) -> str:
        for i in self.selectedItems():
            if isinstance(i, NodeItem):
                return i.node_id
        return ""

    def selected_links(self) -> list:
        return [i for i in self.selectedItems() if isinstance(i, LinkItem)]

    # ---------------------------------------------------------------- 键盘/鼠标
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for lk in self.selected_links():
                self._drop_link(lk)
                self.graphChanged.emit()
            for nid in [i.node_id for i in self.selectedItems()
                        if isinstance(i, NodeItem)]:
                self.remove_node(nid)
            return
        super().keyPressEvent(event)


# ======================================================================
class GraphView(QGraphicsView):
    """画布视图：滚轮缩放、中键/空格平移、双击空白新建。"""

    def __init__(self, scene: GraphScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._linking = False

    def wheelEvent(self, event):
        f = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        cur = self.transform().m11()
        if 0.25 < cur * f < 3.0:
            self.scale(f, f)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # 交给基类处理平移
            super().mousePressEvent(event)
            return
        sc = self.scene()
        item = self.itemAt(event.pos())
        if isinstance(item, PortItem) and isinstance(sc, GraphScene):
            sc.start_link(item)
            self._linking = True
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        sc = self.scene()
        if self._linking and isinstance(sc, GraphScene):
            sc.drag_link(self.mapToScene(event.pos()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        sc = self.scene()
        if self._linking and isinstance(sc, GraphScene):
            self._linking = False
            sc.finish_link(self.mapToScene(event.pos()))
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().mouseReleaseEvent(event)
