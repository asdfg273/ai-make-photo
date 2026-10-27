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

from ui.i18n import ui_i18n
from workflow import registry
from workflow.schema import DataType, Node, Workflow, types_compatible

logger = logging.getLogger(__name__)

NODE_W = 236
PORT_R = 5.0
HEADER_H = 26.0
ROW_H = 20.0
PAD = 8.0
INLINE_H = 22          # 内嵌参数控件高度
PARAM_LABEL_W = 76     # 卡片内参数标签占的宽度
PREVIEW_SIZE = 104     # 卡片内图片预览的边长

# 内嵌控件配色（跟随主题的暗色，避免默认白底在深色画布上刺眼）
_INLINE_QSS = """
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background:#1e1e2e; color:#cdd6f4; border:1px solid #45475a;
    border-radius:3px; padding:1px 4px; font-size:11px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border:1px solid #89b4fa;
}
QComboBox::drop-down { width:14px; border:0; }
QComboBox QAbstractItemView {
    background:#1e1e2e; color:#cdd6f4; selection-background-color:#45475a;
}
QCheckBox { color:#cdd6f4; font-size:11px; }
"""

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
    DataType.REGIONS: "#f38ba8",
}


def _type_color(t: DataType) -> str:
    return TYPE_COLORS.get(t, "#9399b2")


# ---- 缩略图缓存：同一个路径不重复解码（画布重绘很频繁）----
_THUMB_CACHE: dict = {}
_THUMB_ORDER: list = []
_THUMB_MAX = 64


def load_thumbnail(path: str, size: int = PREVIEW_SIZE):
    """把图片读成小缩略图；失败返回 (None, 原因)。带 LRU 缓存。"""
    if not path:
        return None, ""
    key = (str(path), size)
    if key in _THUMB_CACHE:
        pix, err = _THUMB_CACHE[key]
        return pix, ui_i18n.tr(err)
    result = (None, "")
    try:
        import os
        if not os.path.isfile(path):
            result = (None, "文件不存在")
        else:
            from PIL import Image
            from PyQt6.QtGui import QImage, QPixmap
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((size, size), Image.LANCZOS)
                data = im.tobytes("raw", "RGB")
                qimg = QImage(data, im.width, im.height, im.width * 3,
                              QImage.Format.Format_RGB888)
                result = (QPixmap.fromImage(qimg.copy()), "")
    except Exception as e:
        logger.debug(f"缩略图加载失败 {path}: {e}")
        result = (None, str(e)[:40])
    _THUMB_CACHE[key] = result
    _THUMB_ORDER.append(key)
    while len(_THUMB_ORDER) > _THUMB_MAX:
        old = _THUMB_ORDER.pop(0)
        _THUMB_CACHE.pop(old, None)
    return result[0], ui_i18n.tr(result[1])


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
        # 内嵌参数控件（QGraphicsProxyWidget 承载真实 Qt 控件）
        self.param_widgets: dict[str, object] = {}
        self.param_proxies: list = []
        self.param_rows: list = []
        self._param_base_y = 0.0
        # 卡片内图片预览：参考图（来自路径）或运行后节点自己产出的图
        self.preview_pixmap = None
        self.preview_note = ""
        self._preview_param = ""
        self._preview_path = ""
        self._preview_h = 0.0
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(1)

        self._layout_ports()
        description = ui_i18n.tr(self.spec.description or "")
        title = description.splitlines()[0] if description else ""
        self.setToolTip(f"{ntype}\n{title}\n\n{description}")

    # ---- 几何 ----
    def _layout_ports(self):
        n_in = len(self.spec.inputs)
        n_out = len(self.spec.outputs)
        rows = max(n_in, n_out, 1)
        self._ports_h = HEADER_H + rows * ROW_H + PAD
        self.h = self._ports_h

        for i, (name, ps) in enumerate(self.spec.inputs.items()):
            p = PortItem(self, name, ps.type, is_output=False)
            p.setPos(0, HEADER_H + i * ROW_H + ROW_H / 2)
            self.ports_in[name] = p
        for i, (name, ps) in enumerate(self.spec.outputs.items()):
            p = PortItem(self, name, ps.type, is_output=True)
            p.setPos(NODE_W, HEADER_H + i * ROW_H + ROW_H / 2)
            self.ports_out[name] = p

    # ------------------------------------------------------------------
    #  内嵌参数：把参数做进节点卡片里，用户不用来回看右侧面板
    # ------------------------------------------------------------------
    def build_param_widgets(self, node: "Node", on_change) -> None:
        """为每个参数建一个紧凑控件，直接嵌进卡片。

        `on_change(param_name, value)` 由场景层接上，用于写回数据层。
        长文本（text）在卡片里用单行输入框（省空间），完整多行编辑仍在右侧面板。
        """
        from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                                     QGraphicsProxyWidget, QLineEdit,
                                     QSpinBox)

        self.param_widgets.clear()
        self.param_proxies = []
        self.param_rows = []

        # ---- 图片预览区（图生图/参考图节点用）----
        # 规则：参数里带 source="images" 的节点（LoadImage）→ 按路径直接出缩略图；
        # 其余节点等运行完由 set_preview_image() 填。这样"加了参考图就能在
        # 模块里看到图"，不用去猜路径填对没有。
        self._preview_param = ""
        for nm, pspec in self.spec.params.items():
            if getattr(pspec, "source", "") == "images":
                self._preview_param = nm
                break
        self._preview_h = (PREVIEW_SIZE + 6) if self._preview_param else 0.0
        if self._preview_param:
            self.refresh_preview_from_path(
                node.params.get(self._preview_param, ""))

        if not self.spec.params:
            self.h = self._ports_h + self._preview_h
            return

        y = self._ports_h + self._preview_h + 4
        for name, pspec in self.spec.params.items():
            label = (pspec.label or name)
            self.param_rows.append((name, label, y))
            cur = node.params.get(name, pspec.default)
            w = None
            try:
                if pspec.kind == "bool":
                    w = QCheckBox()
                    w.setChecked(bool(cur))
                    w.toggled.connect(
                        lambda v, n=name: on_change(n, bool(v)))
                elif pspec.kind in ("int", "seed"):
                    w = QSpinBox()
                    lo = int(pspec.minimum) if pspec.minimum is not None else (
                        -1 if pspec.kind == "seed" else -10 ** 9)
                    hi = int(pspec.maximum) if pspec.maximum is not None else (
                        2_147_483_647 if pspec.kind == "seed" else 10 ** 9)
                    w.setRange(lo, hi)
                    w.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
                    if pspec.kind == "seed":
                        w.setSpecialValueText(ui_i18n.tr("随机"))
                    try:
                        w.setValue(int(cur))
                    except Exception:
                        w.setValue(lo)
                    w.valueChanged.connect(
                        lambda v, n=name: on_change(n, int(v)))
                elif pspec.kind == "float":
                    w = QDoubleSpinBox()
                    w.setDecimals(3)
                    w.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                    lo = float(pspec.minimum) if pspec.minimum is not None else -1e9
                    hi = float(pspec.maximum) if pspec.maximum is not None else 1e9
                    w.setRange(lo, hi)
                    if pspec.step:
                        w.setSingleStep(float(pspec.step))
                    try:
                        w.setValue(float(cur))
                    except Exception:
                        w.setValue(lo)
                    w.valueChanged.connect(
                        lambda v, n=name: on_change(n, float(v)))
                elif pspec.kind == "choice":
                    w = QComboBox()
                    w.addItems([str(c) for c in pspec.choices])
                    i = w.findText("" if cur is None else str(cur))
                    w.setCurrentIndex(i if i >= 0 else 0)
                    w.currentTextChanged.connect(
                        lambda v, n=name: on_change(n, str(v)))
                elif pspec.source:
                    # ★ 本地文件参数（底模 / LoRA / VAE / 图片）→ 下拉框，
                    #   用户从**已下载的本地文件**里选，不用手打文件名。
                    #   仍然可编辑，方便填绝对路径或还没进目录的文件。
                    w = QComboBox()
                    w.setEditable(True)
                    w.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
                    opts = []
                    try:
                        from ui.workflow_editor import param_sources
                        opts = param_sources.options_for(pspec.source)
                    except Exception as e:
                        logger.debug(f"参数选项加载失败 {pspec.source}: {e}")
                    w.addItem("")
                    w.addItems(opts)
                    i = w.findText("" if cur is None else str(cur))
                    if i >= 0:
                        w.setCurrentIndex(i)
                    else:
                        w.setEditText("" if cur is None else str(cur))
                    w.currentTextChanged.connect(
                        lambda v, n=name: on_change(n, str(v)))
                    src_label = ""
                    try:
                        from ui.workflow_editor import param_sources
                        src_label = param_sources.describe(pspec.source)
                    except Exception:
                        pass
                    w.setToolTip(ui_i18n.trf(
                        "{label}：{source}共 {count} 个可选\n（也可直接粘贴绝对路径）",
                        label=label, source=src_label, count=len(opts))
                        + (f"\n{pspec.help}" if pspec.help else ""))
                else:
                    w = QLineEdit()
                    w.setText("" if cur is None else str(cur))
                    w.textChanged.connect(
                        lambda v, n=name: on_change(n, str(v)))
            except Exception as e:
                logger.warning(f"内嵌控件创建失败 {self.ntype}.{name}: {e}")
                w = None

            self.param_widgets[name] = w
            if w is None:
                continue

            if not pspec.source:
                w.setToolTip(f"{label}"
                             + (f"\n{pspec.help}" if pspec.help else ""))
            w.setStyleSheet(_INLINE_QSS)
            w.setFixedHeight(INLINE_H)
            proxy = QGraphicsProxyWidget(self)
            proxy.setWidget(w)
            proxy.setPos(PARAM_LABEL_W, y)
            proxy.resize(NODE_W - PARAM_LABEL_W - PAD, INLINE_H)
            self.param_proxies.append(proxy)
            y += INLINE_H + 2

        self.h = y + PAD

    # ------------------------------------------------------------------
    #  图片预览
    # ------------------------------------------------------------------
    def refresh_preview_from_path(self, path: str) -> None:
        """参考图路径变化 → 重新出缩略图（LoadImage 这类节点）。"""
        self._preview_path = str(path or "")
        pix, err = load_thumbnail(self._preview_path)
        self.preview_pixmap = pix
        if not path:
            self.preview_note = ui_i18n.tr("未选择参考图")
        elif pix is None:
            self.preview_note = err or ui_i18n.tr("无法读取")
        else:
            self.preview_note = ""
        self.update()

    def retranslate_dynamic(self) -> None:
        """刷新 graphics 项内不会经过普通控件树翻译的文字。"""
        description = ui_i18n.tr(self.spec.description or "")
        title = description.splitlines()[0] if description else ""
        self.setToolTip(f"{self.ntype}\n{title}\n\n{description}")
        for name, widget in self.param_widgets.items():
            pspec = self.spec.params.get(name)
            if widget is None or pspec is None:
                continue
            if pspec.kind == "seed":
                widget.setSpecialValueText(ui_i18n.tr("随机"))
            if pspec.source:
                try:
                    from ui.workflow_editor import param_sources
                    opts = param_sources.options_for(pspec.source)
                    src_label = param_sources.describe(pspec.source)
                except Exception:
                    opts, src_label = [], ""
                label = pspec.label or name
                widget.setToolTip(ui_i18n.trf(
                    "{label}：{source}共 {count} 个可选\n（也可直接粘贴绝对路径）",
                    label=label, source=src_label, count=len(opts))
                    + (f"\n{pspec.help}" if pspec.help else ""))
        if self._preview_param:
            self.refresh_preview_from_path(self._preview_path)
        self.update()

    def set_preview_image(self, image, note: str = "") -> None:
        """运行后把节点产出的图渲染进卡片（缩略图，避免占内存）。"""
        pix = None
        try:
            from PyQt6.QtGui import QImage, QPixmap
            from PIL import Image
            if isinstance(image, Image.Image):
                im = image.convert("RGB")
                im.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)
                data = im.tobytes("raw", "RGB")
                qimg = QImage(data, im.width, im.height, im.width * 3,
                              QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg.copy())
            elif isinstance(image, str) and image:
                pix, err = load_thumbnail(image)
                if pix is None:
                    note = note or err
        except Exception as e:
            logger.debug(f"节点预览失败 {self.node_id}: {e}")
            note = note or str(e)[:40]
        self.preview_pixmap = pix
        self.preview_note = note if pix is None else ""
        if self._preview_param == "" and pix is not None:
            self._preview_h = PREVIEW_SIZE + 6
        self.update()

    def clear_preview_image(self) -> None:
        if self._preview_param:
            return          # 参考图节点由路径驱动，不清
        self.preview_pixmap = None
        self.preview_note = ""
        self._preview_h = 0.0
        self.update()

    def _paint_preview(self, painter: QPainter, top: float) -> None:
        """在卡片里画一块预览区（图 + 说明）。"""
        box = QRectF(PAD, top + 2, NODE_W - PAD * 2, PREVIEW_SIZE)
        painter.setBrush(QBrush(QColor("#181825")))
        painter.setPen(QPen(QColor("#313244"), 1))
        painter.drawRoundedRect(box, 4, 4)

        if self.preview_pixmap is not None and not self.preview_pixmap.isNull():
            pm = self.preview_pixmap
            # 等比缩放居中放进去
            scaled = pm.scaled(
                int(box.width()) - 4, int(box.height()) - 4,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(
                int(box.x() + (box.width() - scaled.width()) / 2),
                int(box.y() + (box.height() - scaled.height()) / 2),
                scaled)
            if self.preview_note:
                painter.setPen(QPen(QColor("#f9e2af")))
                f = QFont()
                f.setPointSize(7)
                painter.setFont(f)
                painter.drawText(box.adjusted(3, 0, -3, -2),
                                 int(Qt.AlignmentFlag.AlignBottom
                                     | Qt.AlignmentFlag.AlignHCenter),
                                 self.preview_note)
        else:
            painter.setPen(QPen(QColor("#6c7086")))
            f = QFont()
            f.setPointSize(8)
            painter.setFont(f)
            painter.drawText(box,
                             int(Qt.AlignmentFlag.AlignCenter),
                             self.preview_note or ui_i18n.tr("（无预览）"))

    def set_param_widget(self, name: str, value) -> None:
        w = self.param_widgets.get(name)
        if w is None:
            return
        w.blockSignals(True)
        try:
            from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                                         QLineEdit, QSpinBox)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QSpinBox):
                w.setValue(int(value))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(value))
            elif isinstance(w, QComboBox):
                i = w.findText(str(value))
                if i >= 0:
                    w.setCurrentIndex(i)
            elif isinstance(w, QLineEdit):
                w.setText("" if value is None else str(value))
        except Exception as e:
            logger.debug(f"同步参数控件失败 {name}: {e}")
        finally:
            w.blockSignals(False)

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
            label += ui_i18n.tr("（重跑）")
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

        # ---- 内嵌参数区：分隔线 + 每行左侧标签 ----
        if self._preview_h:
            self._paint_preview(painter, self._ports_h)
        if self.param_rows:
            sep_y = self._ports_h + 1
            painter.setPen(QPen(QColor("#45475a"), 1))
            painter.drawLine(QPointF(PAD, sep_y), QPointF(NODE_W - PAD, sep_y))
            f3 = QFont()
            f3.setPointSize(8)
            painter.setFont(f3)
            painter.setPen(QPen(QColor("#9399b2")))
            for _name, plabel, py in self.param_rows:
                painter.drawText(
                    QRectF(PAD, py, PARAM_LABEL_W - PAD - 4, INLINE_H),
                    int(Qt.AlignmentFlag.AlignVCenter
                        | Qt.AlignmentFlag.AlignLeft),
                    painter.fontMetrics().elidedText(
                        plabel, Qt.TextElideMode.ElideRight,
                        int(PARAM_LABEL_W - PAD - 6)))

        if self.error:
            painter.setPen(QPen(QColor("#f38ba8")))
            painter.drawText(QRectF(0, self.h + 1, NODE_W, 14),
                             int(Qt.AlignmentFlag.AlignLeft), ui_i18n.tr("执行失败"))

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
    paramChanged = pyqtSignal(str, str, object)   # node_id, param, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(QColor("#11111b")))
        self.wf = Workflow(name="未命名工作流")
        self.node_items: dict[str, NodeItem] = {}
        self.link_items: list[LinkItem] = []
        self._temp_link: QGraphicsPathItem | None = None
        self._link_src: PortItem | None = None
        self.selectionChanged.connect(self._on_selection)
        ui_i18n.language_changed.connect(self.retranslate_dynamic)

    def retranslate_dynamic(self, *_):
        for item in self.node_items.values():
            if isinstance(item, NodeItem):
                item.retranslate_dynamic()

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
            # 参数控件要在算高度**之前**建好，否则节点框会按没有参数的高度画
            item.build_param_widgets(
                n, lambda name, val, nid=n.id: self._on_param_edit(nid, name, val))
            pos = positions.get(n.id)
            if pos is not None:
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

    def _on_param_edit(self, node_id: str, name: str, value):
        """卡片内控件被改动 → 写回数据层。"""
        node = self.wf.node(node_id)
        if node is None:
            return
        node.params[name] = value
        self._maybe_refresh_preview(node_id, name, value)
        self.paramChanged.emit(node_id, name, value)

    def set_node_param(self, node_id: str, name: str, value):
        """外部（右侧面板）改动 → 写回数据层并同步卡片内控件。"""
        node = self.wf.node(node_id)
        if node is None:
            return
        node.params[name] = value
        item = self.node_items.get(node_id)
        if item is not None:
            item.set_param_widget(name, value)
        self._maybe_refresh_preview(node_id, name, value)

    def _maybe_refresh_preview(self, node_id: str, name: str, value):
        """改的是参考图路径 → 立刻重画卡片里的预览图。"""
        item = self.node_items.get(node_id)
        if item is None or not item._preview_param:
            return
        if name == item._preview_param:
            item.refresh_preview_from_path(str(value or ""))

    def push_previews(self, images: dict):
        """运行结束后把每个节点产出的图渲染进它的卡片。"""
        for nid, img in (images or {}).items():
            item = self.node_items.get(nid)
            if item is not None:
                item.set_preview_image(img)

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
            self.statusMessage.emit(ui_i18n.trf("未注册的节点类型：{type}", type=ntype))
            return None
        nid = self._unique_id(ntype)
        spec = registry.get(ntype)
        params = dict(spec.param_defaults())
        node = self.wf.add(Node(nid, ntype, {}, params))
        item = NodeItem(nid, ntype)
        item.build_param_widgets(
            node, lambda name, val, _nid=nid: self._on_param_edit(_nid, name, val))
        item.setPos(pos if pos is not None else QPointF(60, 60))
        self.addItem(item)
        self.node_items[nid] = item
        self.graphChanged.emit()
        self.statusMessage.emit(ui_i18n.trf("已添加 {type} ({id})", type=ntype, id=nid))
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
        self.statusMessage.emit(ui_i18n.trf("已删除 {id}", id=nid))

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
            self.statusMessage.emit(ui_i18n.tr("不能连接到同一个节点"))
            return False
        if src.is_output == dst.is_output:
            self.statusMessage.emit(ui_i18n.tr("必须从「输出」连到「输入」"))
            return False
        if not src.is_output:                 # 允许反向拖
            src, dst = dst, src
        a, b = src.dtype, dst.dtype
        if not types_compatible(a, b):
            self.statusMessage.emit(ui_i18n.trf(
                "类型不匹配：{source} → {target}（不会自动转换）",
                source=a.value, target=b.value))
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
                f"{src.node_item.node_id}.{src.name} → "
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
            self.statusMessage.emit(ui_i18n.tr("（未连到端口，已取消）"))
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
