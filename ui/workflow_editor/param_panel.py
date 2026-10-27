# ui/workflow_editor/param_panel.py
"""参数面板：从 `ParamSpec` **自动生成**控件，不手写任何一个节点的表单。

这是"加功能只需要加节点"的关键一环：节点在 `nodes/*.py` 里声明了
参数类型与范围，界面这边零改动就会出现对应控件（滑块/下拉/勾选/文本框），
并且带上下限与非法值拦截。
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                             QFrame, QLabel, QLineEdit, QPlainTextEdit,
                             QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ui.i18n import ui_i18n
from workflow import registry

logger = logging.getLogger(__name__)


class ParamPanel(QScrollArea):
    """选中节点后显示它的参数（以及输入端口当前连到了哪里）。"""

    paramChanged = pyqtSignal(str, str, object)   # node_id, param, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._node_id = ""
        self._widgets: dict = {}
        self._wf = None
        self._host = QWidget()
        self._lay = QVBoxLayout(self._host)
        self._lay.setContentsMargins(8, 8, 8, 8)
        self.setWidget(self._host)
        ui_i18n.language_changed.connect(self.retranslate_dynamic)
        self._show_placeholder()

    # ------------------------------------------------------------------
    def _clear(self):
        """清空面板。

        ⚠️ 必须 `setParent(None)` 再 `deleteLater()`。
        只调 `deleteLater()` 的话：它只是**排队**等事件循环删除，而这期间
        控件仍然是 `_host` 的**可见子控件**，会停在原来的坐标继续绘制。
        于是快速切换节点（或 show_node 连续调用）时，上一个节点的控件会
        和新控件叠在一起 —— 表现就是"面板里出现了两个采样器/两组参数"。
        `setParent(None)` 立即解除父子关系（控件随之隐藏），deleteLater 再收尾。
        """
        self._widgets.clear()
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
                continue
            lay = item.layout()
            if lay is not None:
                # QFormLayout 这类"子布局"也要连里面的控件一起摘掉
                while lay.count():
                    sub = lay.takeAt(0)
                    sw = sub.widget()
                    if sw is not None:
                        sw.setParent(None)
                        sw.deleteLater()
                lay.setParent(None)
                lay.deleteLater()

    def _show_placeholder(self, text=None):
        self._clear()
        lbl = QLabel(text if text is not None else ui_i18n.tr("选中一个节点查看参数"))
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#7f849c;")
        self._lay.addWidget(lbl)
        self._lay.addStretch(1)

    def retranslate_dynamic(self, *_):
        """语言切换后重建只读参数视图，业务值仍从节点数据读取。"""
        if self._wf is not None and self._node_id:
            self.show_node(self._wf, self._node_id)
        else:
            self._show_placeholder()


    # ------------------------------------------------------------------
    def show_node(self, wf, node_id: str, node_item=None):
        self._wf = wf
        self._node_id = node_id or ""
        node = wf.node(node_id) if (wf and node_id) else None
        if node is None:
            self._show_placeholder()
            return
        if not registry.exists(node.type):
            self._show_placeholder(ui_i18n.trf("未注册的节点类型：{type}", type=node.type))
            return

        spec = registry.get(node.type)
        self._clear()

        title = QLabel(f"<b>{node_id}</b> · {node.type}")
        title.setStyleSheet("color:#cdd6f4; font-size:13px;")
        self._lay.addWidget(title)
        desc = QLabel(spec.description or "")
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#7f849c; font-size:11px;")
        self._lay.addWidget(desc)

        # ---- 输入端口现状（只读，连线在画布上做）----
        if spec.inputs:
            self._lay.addWidget(self._section(ui_i18n.tr("输入")))
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            for name, ps in spec.inputs.items():
                link = node.link_refs().get(name)
                txt = (f"← {link[0]}.{link[1]}" if link else ui_i18n.tr(
                    "未连接" if not ps.optional else "未连接（可选）"))
                lbl = QLabel(txt)
                col = "#a6e3a1" if link else "#6c7086"
                lbl.setStyleSheet(f"color:{col}; font-size:11px;")
                form.addRow(f"{name} ({ps.type.value})", lbl)
            self._lay.addLayout(form)

        # ---- 参数 ----
        if spec.params:
            self._lay.addWidget(self._section(ui_i18n.tr("参数")))
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            for name, pspec in spec.params.items():
                cur = node.params.get(name, pspec.default)
                w = self._make_widget(name, pspec, cur)
                if w is None:
                    continue
                self._widgets[name] = w
                tip = pspec.help or ""
                if pspec.minimum is not None or pspec.maximum is not None:
                    tip += "\n" + ui_i18n.trf(
                        "范围 [{minimum}, {maximum}]",
                        minimum=pspec.minimum, maximum=pspec.maximum)
                if tip:
                    w.setToolTip(tip.strip())
                form.addRow(pspec.label or name, w)
            self._lay.addLayout(form)
        else:
            self._lay.addWidget(self._muted(ui_i18n.tr("该节点没有可调参数")))

        self._lay.addWidget(self._section(ui_i18n.tr("资源声明")))
        res = spec.resources
        support = ui_i18n.tr("支持")
        unsupported = ui_i18n.tr("不支持")
        info = [ui_i18n.trf("显存峰值 ≈ {value} MB", value=res.estimated_vram_mb),
                ui_i18n.trf("CPU offload: {status}",
                            status=support if res.supports_cpu_offload else unsupported),
                ui_i18n.trf("分块: {status}",
                            status=support if res.supports_tiled else unsupported)]
        if res.exclusive_gpu:
            info.append(ui_i18n.tr("独占 GPU"))
        if res.requires:
            info.append(ui_i18n.trf("依赖: {requirements}", requirements=", ".join(res.requires)))
        lbl = QLabel("\n".join(info))
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#94e2d5; font-size:11px;")
        self._lay.addWidget(lbl)
        self._lay.addStretch(1)

    # ------------------------------------------------------------------
    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#89b4fa; font-weight:bold; "
                          "padding-top:8px; font-size:12px;")
        return lbl

    @staticmethod
    def _muted(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#7f849c; font-size:11px;")
        return lbl

    def _emit(self, name: str, value):
        if self._node_id:
            self.paramChanged.emit(self._node_id, name, value)

    def _make_widget(self, name: str, pspec, cur):
        """按 ParamSpec.kind 生成控件。"""
        kind = pspec.kind
        try:
            if kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(cur))
                w.toggled.connect(lambda v, n=name: self._emit(n, bool(v)))
                return w

            if kind in ("int", "seed"):
                w = QSpinBox()
                lo = int(pspec.minimum) if pspec.minimum is not None else (
                    -1 if kind == "seed" else 0)
                hi = int(pspec.maximum) if pspec.maximum is not None else (
                    2_147_483_647 if kind == "seed" else 100000)
                w.setRange(lo, hi)
                if kind == "seed":
                    w.setSpecialValueText(ui_i18n.tr("随机 (-1)"))
                try:
                    w.setValue(int(cur))
                except Exception:
                    w.setValue(lo)
                w.valueChanged.connect(lambda v, n=name: self._emit(n, int(v)))
                return w

            if kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(3)
                lo = float(pspec.minimum) if pspec.minimum is not None else -1e6
                hi = float(pspec.maximum) if pspec.maximum is not None else 1e6
                w.setRange(lo, hi)
                if pspec.step:
                    w.setSingleStep(float(pspec.step))
                try:
                    w.setValue(float(cur))
                except Exception:
                    w.setValue(lo)
                w.valueChanged.connect(lambda v, n=name: self._emit(n, float(v)))
                return w

            if kind == "choice":
                w = QComboBox()
                w.addItems([str(c) for c in pspec.choices])
                s = "" if cur is None else str(cur)
                i = w.findText(s)
                w.setCurrentIndex(i if i >= 0 else 0)
                w.currentTextChanged.connect(
                    lambda v, n=name: self._emit(n, str(v)))
                return w

            if kind == "text":
                w = QPlainTextEdit()
                w.setPlainText("" if cur is None else str(cur))
                w.setFixedHeight(70)
                w.textChanged.connect(
                    lambda n=name, e=w: self._emit(n, e.toPlainText()))
                return w

            # str 及其它
            w = QLineEdit()
            w.setText("" if cur is None else str(cur))
            w.textChanged.connect(lambda v, n=name: self._emit(n, str(v)))
            return w
        except Exception as e:
            logger.warning(f"参数控件创建失败 {name}: {e}")
            return None

    # ------------------------------------------------------------------
    def set_param(self, name: str, value):
        """外部（如程序化改图）同步到控件。"""
        w = self._widgets.get(name)
        if w is None:
            return
        w.blockSignals(True)
        try:
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
            elif isinstance(w, QPlainTextEdit):
                w.setPlainText("" if value is None else str(value))
            elif isinstance(w, QLineEdit):
                w.setText("" if value is None else str(value))
        except Exception as e:
            logger.debug(f"set_param 失败 {name}: {e}")
        finally:
            w.blockSignals(False)
