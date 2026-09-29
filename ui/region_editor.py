# ui/region_editor.py
"""区域提示词编辑器（主界面用，不要求用户懂工作流）。

为什么要有这个界面
------------------
区域提示词如果只能在节点编辑器里用，那它就只服务于高级用户。这个对话框
把同一件事变成"在图上拖几个框、每个框写一句话"：

    全局提示词：portrait, detailed clothing
    ├─ 脸部区域：detailed face, sharp eyes
    ├─ 衣服区域：intricate fabric, embroidery
    └─ 背景区域：forest, soft light

矩形框直接拖出来；也可以导入黑白蒙版图（白=生效区域）做不规则区域。
每个区域可以单独设置提示词、局部负面提示词、权重（=该区域重绘强度）、
羽化，以及可选的局部 ControlNet / 局部 ADetailer。

输出是一串 `dict`（存进配置的 `regions` 字段），引擎侧由
`core.regional.Region.from_dict()` 读懂，两边格式完全一致。
"""
from __future__ import annotations

import logging
import os

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                             QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

logger = logging.getLogger(__name__)

# 新建区域时轮换的配色（画布上便于区分）
PALETTE = ["#f38ba8", "#a6e3a1", "#89b4fa", "#f9e2af", "#cba6f7",
           "#fab387", "#94e2d5", "#f5c2e7"]

COL_ENABLE, COL_NAME, COL_PROMPT, COL_NEG, COL_WEIGHT, COL_SHAPE = range(6)


# ======================================================================
class RegionCanvas(QWidget):
    """在按输出分辨率等比缩放的画布上拖拽矩形区域。"""

    regionPicked = pyqtSignal(int)          # 选中第几个
    regionCreated = pyqtSignal(tuple)       # 新拖出的 (x0,y0,x1,y1) 归一化

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.img_w, self.img_h = 512, 768
        self.regions: list = []             # [{"rect":(..),"color":..,"name":..}]
        self.selected = -1
        self._drag_from = None
        self._drag_to = None
        self.setMouseTracking(True)
        self.setToolTip("按住左键拖出一个矩形区域；点选已有区域")

    # ---- 尺寸换算 ----
    def set_image_size(self, w: int, h: int):
        self.img_w = max(1, int(w))
        self.img_h = max(1, int(h))
        self.update()

    def _view_rect(self) -> QRect:
        """图在控件里的显示矩形（等比居中，留边距）。"""
        pad = 10
        aw = max(1, self.width() - pad * 2)
        ah = max(1, self.height() - pad * 2)
        k = min(aw / self.img_w, ah / self.img_h)
        w = int(self.img_w * k)
        h = int(self.img_h * k)
        return QRect(pad + (aw - w) // 2, pad + (ah - h) // 2, w, h)

    def _to_norm(self, pos: QPoint) -> tuple:
        r = self._view_rect()
        x = (pos.x() - r.x()) / max(1, r.width())
        y = (pos.y() - r.y()) / max(1, r.height())
        return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))

    def set_regions(self, regions: list):
        self.regions = regions
        if self.selected >= len(regions):
            self.selected = -1
        self.update()

    def select(self, idx: int):
        self.selected = idx
        self.update()

    # ---- 绘制 ----
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._view_rect()
        # 画布底
        p.setBrush(QBrush(QColor("#181825")))
        p.setPen(QPen(QColor("#45475a"), 1))
        p.drawRect(r)
        # 三分线（构图参考）
        p.setPen(QPen(QColor("#313244"), 1, Qt.PenStyle.DashLine))
        for i in (1, 2):
            p.drawLine(r.x() + r.width() * i // 3, r.y(),
                       r.x() + r.width() * i // 3, r.y() + r.height())
            p.drawLine(r.x(), r.y() + r.height() * i // 3,
                       r.x() + r.width(), r.y() + r.height() * i // 3)

        # 已有区域
        for i, reg in enumerate(self.regions):
            x0, y0, x1, y1 = reg.get("rect", (0, 0, 0, 0))
            rc = QRect(int(r.x() + x0 * r.width()), int(r.y() + y0 * r.height()),
                       max(2, int((x1 - x0) * r.width())),
                       max(2, int((y1 - y0) * r.height())))
            col = QColor(reg.get("color") or PALETTE[i % len(PALETTE)])
            on = reg.get("enabled", True)
            fill = QColor(col)
            fill.setAlpha(50 if on else 18)
            p.setBrush(QBrush(fill))
            p.setPen(QPen(col, 3 if i == self.selected else 1,
                          Qt.PenStyle.SolidLine if on else Qt.PenStyle.DashLine))
            p.drawRect(rc)
            label = reg.get("name") or f"区域{i + 1}"
            p.setPen(QPen(col))
            p.drawText(rc.adjusted(4, 2, -4, -2),
                       int(Qt.AlignmentFlag.AlignTop
                           | Qt.AlignmentFlag.AlignLeft), label)

        # 正在拖的框
        if self._drag_from and self._drag_to:
            x0, y0 = self._drag_from
            x1, y1 = self._drag_to
            rc = QRect(int(r.x() + min(x0, x1) * r.width()),
                       int(r.y() + min(y0, y1) * r.height()),
                       max(2, int(abs(x1 - x0) * r.width())),
                       max(2, int(abs(y1 - y0) * r.height())))
            p.setBrush(QBrush(QColor(255, 255, 255, 30)))
            p.setPen(QPen(QColor("#f9e2af"), 2, Qt.PenStyle.DashLine))
            p.drawRect(rc)

        p.setPen(QPen(QColor("#6c7086")))
        p.drawText(4, self.height() - 4, f"{self.img_w}x{self.img_h}")

    # ---- 鼠标 ----
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        n = self._to_norm(pos)
        # 先看点没点到已有区域（从后往前，方便选到上层）
        hit = -1
        r = self._view_rect()
        for i in range(len(self.regions) - 1, -1, -1):
            x0, y0, x1, y1 = self.regions[i].get("rect", (0, 0, 0, 0))
            rc = QRect(int(r.x() + x0 * r.width()), int(r.y() + y0 * r.height()),
                       int((x1 - x0) * r.width()), int((y1 - y0) * r.height()))
            if rc.contains(pos):
                hit = i
                break
        if hit >= 0:
            self.select(hit)
            self.regionPicked.emit(hit)
            return
        self._drag_from = n
        self._drag_to = n

    def mouseMoveEvent(self, event):
        if self._drag_from:
            self._drag_to = self._to_norm(event.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, event):
        if not self._drag_from:
            return
        x0, y0 = self._drag_from
        x1, y1 = self._drag_to or self._drag_from
        self._drag_from = self._drag_to = None
        # 太小的拖动当作误点
        if abs(x1 - x0) < 0.02 or abs(y1 - y0) < 0.02:
            self.update()
            return
        self.regionCreated.emit((min(x0, x1), min(y0, y1),
                                 max(x0, x1), max(y0, y1)))
        self.update()


# ======================================================================
class RegionEditorDialog(QDialog):
    """区域提示词编辑器。exec() 后用 `regions()` 取结果。"""

    def __init__(self, regions: list, width: int = 512, height: int = 768,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎯 区域提示词")
        self.resize(1040, 660)
        self._regions: list = []
        self._loading = False

        lay = QVBoxLayout(self)
        head = QLabel(
            "在左侧画布上<b>拖出矩形</b>即可新增区域；每个区域可单独设置提示词、"
            "局部负面提示词、权重（=该区域重绘强度）。<br>"
            "需要不规则形状时，用「导入蒙版」选一张黑白图（<b>白色</b>=生效区域）。")
        head.setWordWrap(True)
        lay.addWidget(head)

        body = QHBoxLayout()
        lay.addLayout(body, 1)

        # ---- 左：画布 ----
        left = QVBoxLayout()
        self.canvas = RegionCanvas()
        self.canvas.set_image_size(width, height)
        self.canvas.regionCreated.connect(self._on_created)
        self.canvas.regionPicked.connect(self._on_picked)
        left.addWidget(self.canvas, 1)
        hint = QLabel(f"画布按输出分辨率等比显示（当前 {width}×{height}）")
        hint.setStyleSheet("color:#7f849c; font-size:11px;")
        left.addWidget(hint)
        body.addLayout(left, 3)

        # ---- 右：区域列表 ----
        right = QVBoxLayout()
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["启用", "名称", "区域提示词", "局部负面", "权重", "形状"])
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setDefaultSectionSize(28)
        h = self.table.horizontalHeader()
        for c in (COL_ENABLE, COL_WEIGHT, COL_SHAPE):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        for c in (COL_NAME, COL_PROMPT, COL_NEG):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        right.addWidget(self.table, 1)

        btns = QHBoxLayout()
        for text, slot, tip in (
                ("➕ 加矩形", lambda: self._add_region("rect"),
                 "也可以直接在画布上拖"),
                ("🖼 导入蒙版", self._add_mask_region,
                 "选一张黑白图，白色=生效区域"),
                ("🗑 删除", self._remove_selected, ""),
                ("⏫", lambda: self._move(-1), "上移"),
                ("⏬", lambda: self._move(1), "下移")):
            b = QPushButton(text)
            b.clicked.connect(slot)
            if tip:
                b.setToolTip(tip)
            btns.addWidget(b)
        btns.addStretch(1)
        right.addLayout(btns)

        # ---- 局部附加能力（选中区域生效）----
        grp = QGroupBox("选中区域的局部附加能力（可选）")
        gf = QFormLayout(grp)
        self.chk_cn = QCheckBox("局部 ControlNet")
        self.combo_cn = QComboBox()
        self.combo_cn.addItems(["openpose", "canny", "depth"])
        self.spin_cn = QDoubleSpinBox()
        self.spin_cn.setRange(0.0, 2.0)
        self.spin_cn.setSingleStep(0.05)
        self.spin_cn.setValue(0.8)
        cn_row = QHBoxLayout()
        cn_row.addWidget(self.chk_cn)
        cn_row.addWidget(self.combo_cn)
        cn_row.addWidget(self.spin_cn)
        gf.addRow("ControlNet:", cn_row)

        self.chk_ad = QCheckBox("局部 ADetailer")
        self.combo_ad = QComboBox()
        self.combo_ad.addItems(["二次元脸部", "二次元手部"])
        self.spin_ad = QDoubleSpinBox()
        self.spin_ad.setRange(0.05, 0.9)
        self.spin_ad.setSingleStep(0.05)
        self.spin_ad.setValue(0.35)
        ad_row = QHBoxLayout()
        ad_row.addWidget(self.chk_ad)
        ad_row.addWidget(self.combo_ad)
        ad_row.addWidget(self.spin_ad)
        gf.addRow("ADetailer:", ad_row)

        self.spin_feather = QDoubleSpinBox()
        self.spin_feather.setRange(0.0, 128.0)
        self.spin_feather.setSingleStep(4)
        self.spin_feather.setValue(16)
        gf.addRow("边缘羽化(px):", self.spin_feather)
        right.addWidget(grp)
        body.addLayout(right, 4)

        # ---- 底部 ----
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("color:#a6adc8;")
        lay.addWidget(self.lbl_summary)

        bottom = QHBoxLayout()
        b_save = QPushButton("💾 保存区域集")
        b_save.clicked.connect(self._save_set)
        b_load = QPushButton("📂 载入区域集")
        b_load.clicked.connect(self._load_set)
        bottom.addWidget(b_save)
        bottom.addWidget(b_load)
        bottom.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bottom.addWidget(bb)
        lay.addLayout(bottom)

        self._set_regions(list(regions or []), width, height)

    # ------------------------------------------------------------------
    def _set_regions(self, regions: list, w=None, h=None):
        self._loading = True
        try:
            self._regions = []
            for i, r in enumerate(regions):
                if not isinstance(r, dict):
                    continue
                d = dict(r)
                d.setdefault("name", f"区域{i + 1}")
                d.setdefault("prompt", "")
                d.setdefault("negative", "")
                d.setdefault("weight", 0.5)
                d.setdefault("shape", "rect")
                d.setdefault("rect", (0.0, 0.0, 0.5, 0.5))
                d.setdefault("mask_path", "")
                d.setdefault("color", PALETTE[i % len(PALETTE)])
                d.setdefault("enabled", True)
                d.setdefault("feather", 16)
                self._regions.append(d)
            if w:
                self.canvas.set_image_size(w, h or 768)
            self._rebuild_table()
            self._refresh_canvas()
        finally:
            self._loading = False
        self._update_summary()

    def _rebuild_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._regions))
        for i, r in enumerate(self._regions):
            chk = QCheckBox()
            chk.setChecked(bool(r.get("enabled", True)))
            chk.toggled.connect(
                lambda v, idx=i: self._set_field(idx, "enabled", bool(v)))
            self.table.setCellWidget(i, COL_ENABLE, chk)

            for col, key in ((COL_NAME, "name"), (COL_PROMPT, "prompt"),
                             (COL_NEG, "negative")):
                ed = QLineEdit(str(r.get(key, "")))
                ed.textChanged.connect(
                    lambda v, idx=i, k=key: self._set_field(idx, k, str(v)))
                self.table.setCellWidget(i, col, ed)

            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1.0)
            sp.setSingleStep(0.05)
            sp.setValue(float(r.get("weight", 0.5)))
            sp.setToolTip("区域权重 = 该区域的重绘强度（越大改动越多）")
            sp.valueChanged.connect(
                lambda v, idx=i: self._set_field(idx, "weight", float(v)))
            self.table.setCellWidget(i, COL_WEIGHT, sp)

            shape = str(r.get("shape", "rect"))
            it = QTableWidgetItem("矩形" if shape == "rect" else "蒙版")
            it.setToolTip(r.get("mask_path", "") or "拖拽产生的矩形区域")
            self.table.setItem(i, COL_SHAPE, it)
        self.table.blockSignals(False)

    def _set_field(self, idx, key, value):
        if self._loading or not (0 <= idx < len(self._regions)):
            return
        self._regions[idx][key] = value
        self._refresh_canvas()
        self._update_summary()

    def _refresh_canvas(self):
        self.canvas.set_regions(self._regions)

    def _update_summary(self):
        enabled = [r for r in self._regions if r.get("enabled", True)]
        if not enabled:
            self.lbl_summary.setText("未启用任何区域（生成时不做区域重绘）")
            return
        parts = []
        for i, r in enumerate(enabled[:6]):
            nm = r.get("name") or f"区域{i + 1}"
            parts.append(f"{nm}({float(r.get('weight', 0.5)):.2f})")
        more = f" …共 {len(enabled)} 个" if len(enabled) > 6 else ""
        self.lbl_summary.setText(
            f"已启用 {len(enabled)} 个区域：" + "  ".join(parts) + more)

    # ------------------------------------------------------------------
    def _on_created(self, rect):
        idx = len(self._regions)
        self._regions.append({
            "name": f"区域{idx + 1}", "prompt": "", "negative": "",
            "weight": 0.5, "shape": "rect", "rect": tuple(rect),
            "mask_path": "", "color": PALETTE[idx % len(PALETTE)],
            "enabled": True, "feather": 16})
        self._rebuild_table()
        self._refresh_canvas()
        self._update_summary()
        self.table.selectRow(idx)
        # 直接聚焦到"区域提示词"输入框，少点一次
        w = self.table.cellWidget(idx, COL_PROMPT)
        if w is not None:
            w.setFocus()

    def _on_picked(self, idx):
        if 0 <= idx < self.table.rowCount():
            self.table.blockSignals(True)
            self.table.selectRow(idx)
            self.table.blockSignals(False)
            self._load_region_extras(idx)

    def _on_row_selected(self):
        idx = self.table.currentRow()
        self.canvas.select(idx)
        self._load_region_extras(idx)

    def _load_region_extras(self, idx):
        """把选中区域的局部 CN/ADetailer/羽化 回填到下面的表单。"""
        if not (0 <= idx < len(self._regions)):
            return
        r = self._regions[idx]
        self._loading = True
        try:
            cn = r.get("controlnet") or {}
            self.chk_cn.setChecked(bool(cn))
            if cn.get("kind"):
                j = self.combo_cn.findText(str(cn["kind"]))
                if j >= 0:
                    self.combo_cn.setCurrentIndex(j)
            if cn.get("weight") is not None:
                self.spin_cn.setValue(float(cn["weight"]))
            ad = r.get("adetailer") or {}
            self.chk_ad.setChecked(bool(ad))
            if ad.get("target"):
                j = self.combo_ad.findText(str(ad["target"]))
                if j >= 0:
                    self.combo_ad.setCurrentIndex(j)
            if ad.get("strength") is not None:
                self.spin_ad.setValue(float(ad["strength"]))
            self.spin_feather.setValue(float(r.get("feather", 16)))
        finally:
            self._loading = False

    def _commit_extras(self):
        """表单 → 当前选中区域。在 accept 前统一调用。"""
        idx = self.table.currentRow()
        if not (0 <= idx < len(self._regions)):
            return
        r = self._regions[idx]
        r["controlnet"] = ({"kind": self.combo_cn.currentText(),
                            "weight": float(self.spin_cn.value())}
                           if self.chk_cn.isChecked() else None)
        r["adetailer"] = ({"target": self.combo_ad.currentText(),
                           "strength": float(self.spin_ad.value())}
                          if self.chk_ad.isChecked() else None)
        r["feather"] = int(self.spin_feather.value())

    # ------------------------------------------------------------------
    def _add_region(self, shape):
        idx = len(self._regions)
        # 默认放在中间偏上，避免叠在一起看不出来
        off = min(0.3, 0.05 * idx)
        self._regions.append({
            "name": f"区域{idx + 1}", "prompt": "", "negative": "",
            "weight": 0.5, "shape": shape,
            "rect": (0.1 + off, 0.1 + off, 0.5 + off, 0.4 + off),
            "mask_path": "", "color": PALETTE[idx % len(PALETTE)],
            "enabled": True, "feather": 16})
        self._rebuild_table()
        self._refresh_canvas()
        self._update_summary()

    def _add_mask_region(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择蒙版图（白色=生效区域）", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        idx = len(self._regions)
        name = os.path.splitext(os.path.basename(path))[0][:16]
        self._regions.append({
            "name": name or f"区域{idx + 1}", "prompt": "", "negative": "",
            "weight": 0.5, "shape": "mask", "rect": (0.0, 0.0, 1.0, 1.0),
            "mask_path": path, "color": PALETTE[idx % len(PALETTE)],
            "enabled": True, "feather": 16})
        self._rebuild_table()
        self._refresh_canvas()
        self._update_summary()

    def _remove_selected(self):
        idx = self.table.currentRow()
        if 0 <= idx < len(self._regions):
            self._regions.pop(idx)
            self._rebuild_table()
            self._refresh_canvas()
            self._update_summary()

    def _move(self, delta):
        idx = self.table.currentRow()
        j = idx + delta
        if 0 <= idx < len(self._regions) and 0 <= j < len(self._regions):
            self._regions[idx], self._regions[j] = \
                self._regions[j], self._regions[idx]
            self._rebuild_table()
            self._refresh_canvas()
            self.table.selectRow(j)

    # ------------------------------------------------------------------
    def _sets_dir(self) -> str:
        try:
            from utils import paths
            d = os.path.join(paths.PROJECT_ROOT, "regions")
        except Exception:
            d = os.path.abspath("regions")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_set(self):
        self._commit_extras()
        path, _ = QFileDialog.getSaveFileName(
            self, "保存区域集", os.path.join(self._sets_dir(), "区域集.json"),
            "JSON (*.json)")
        if not path:
            return
        try:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"regions": self._regions}, f,
                          ensure_ascii=False, indent=2)
            QMessageBox.information(self, "已保存", path)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _load_set(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "载入区域集", self._sets_dir(), "JSON (*.json)")
        if not path:
            return
        try:
            import json
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            regs = data.get("regions") if isinstance(data, dict) else data
            self._set_regions(list(regs or []))
        except Exception as e:
            QMessageBox.warning(self, "载入失败", str(e))

    # ------------------------------------------------------------------
    def regions(self) -> list:
        """返回可直接写进配置 / 交给引擎的区域列表。"""
        self._commit_extras()
        return [dict(r) for r in self._regions]

    def accept(self):
        self._commit_extras()
        super().accept()
