# ui/ipa_editor.py
"""多参考图编辑器（主界面用，普通用户不需要懂工作流）。

为什么要有这个界面
-----------------
「多参考图」原本只在工作流编辑器里能用（IPAdapterIdentity /
IPAdapterStyle / IPAdapterComposition 三个节点串起来）。那等于只有高级用户
享受得到这个能力。这个对话框把它变成**一张表**：

    启用 | 图片路径 | 角色 | 权重 | 区域蒙版
     ☑   E:/refs/face.png    身份   0.50   （空）
     ☑   E:/refs/style.jpg   风格   0.70   （空）
     ☐   E:/refs/sky.png     构图   0.40   E:/masks/sky.png

角色不是一个随便起的名字：它对应 h94/IP-Adapter 里**训练目标不同**的三个
权重文件（身份=plus-face / 风格=plus / 构图=base），映射表在
`core/ipa_multi.py` 里，本机缺哪个档位就自动降级并给中文提示。

输出是一串 `dict`（存进配置的 `ipa_refs` 字段），引擎侧由
`core.ipa_multi.coerce_references()` / `resolve_role_plan()` 读懂 ——
与 `nodes/ipa_nodes.py`、`nodes/adapter_nodes.py` 的 `refs_json`
**完全同一套格式**，两边不会漂移。

本对话框**不需要 GPU、不加载任何模型**：只在打开时做几次文件系统探测，
报告"本机装了哪些档位"（`ModelManager.installed_ipa_variants()`）。
"""
from __future__ import annotations

import json
import logging
import os

from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QTableWidget,
                             QVBoxLayout, QWidget)

logger = logging.getLogger(__name__)

# 列号（与表头一一对应）
COL_ENABLE, COL_PATH, COL_ROLE, COL_WEIGHT, COL_MASK = range(5)

#: 下拉里展示的角色顺序（身份最常用放最前）
ROLE_ORDER = ("identity", "style", "composition")

#: 图片选择过滤串（参考图与蒙版共用）
_IMG_FILTER = "图片 (*.png *.jpg *.jpeg *.bmp *.webp)"


# ======================================================================
#  纯函数（不依赖 Qt，主界面摘要与测试都复用它）
# ======================================================================
def installed_variants_text() -> str:
    """本机已安装的档位说明（**只查本地文件，不联网、不需要 GPU**）。"""
    try:
        from core.model_manager import ModelManager
        installed = list(ModelManager().installed_ipa_variants())
    except Exception as e:                          # pragma: no cover - 防御
        logger.warning(f"⚠️ 探测 IP-Adapter 档位失败: {e}")
        installed = []
    if not installed:
        return ("⚠️ 本机未检测到任何 IP-Adapter 档位"
                "（权重 + image_encoder 都要有）→ 参考图会被跳过。"
                "请到「工具 → 扩展市场 → IP-Adapter SD1.5」安装。")
    missing = []
    try:
        from core import ipa_multi as ipa_mod
        for role in ROLE_ORDER:
            if ipa_mod.variant_for_role(role) not in installed:
                missing.append(f"{ipa_mod.role_label(role)}"
                               f"（{ipa_mod.variant_for_role(role)}）")
    except Exception:                               # pragma: no cover - 防御
        pass
    txt = f"本机已装档位：{'/'.join(installed)}"
    if missing:
        txt += (f"（{'、'.join(missing)}档位未安装，"
                f"将自动降级并提示）")
    return txt


def summary_text(refs) -> str:
    """参考图中文摘要，例如 `3 张参考：身份(0.50) 风格(0.70) 构图(0.40)`。

    ⚠️ 与 `core.ipa_multi.describe_references` 的分工：
      那个带文件名（写进元数据，便于回溯用了哪张图）；
      这个是给**界面标签**看的（短、只看角色与权重）。
    """
    try:
        from core import ipa_multi as ipa_mod
    except Exception:                               # pragma: no cover - 防御
        return "（无法读取角色表）"

    items = []
    for r in (refs or []):
        if not isinstance(r, dict):
            continue
        if not r.get("enabled", True):
            continue
        items.append(r)
    if not items:
        return "（未设置参考图）"
    parts = []
    for r in items[:6]:
        role = ipa_mod.role_label(r.get("role"))
        try:
            wt = float(r.get("weight", ipa_mod.DEFAULT_WEIGHT))
        except (TypeError, ValueError):
            wt = float(ipa_mod.DEFAULT_WEIGHT)
        parts.append(f"{role}({wt:.2f})")
    more = f" …共 {len(items)} 张" if len(items) > 6 else ""
    return f"{len(items)} 张参考：" + " ".join(parts) + more


# ======================================================================
class IPARefEditorDialog(QDialog):
    """多参考图编辑器。`exec()` 之后用 `refs()` 取结果。"""

    def __init__(self, refs: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("多参考图 (IP-Adapter)")
        self.resize(1000, 560)
        self._refs: list = []
        self._loading = False

        lay = QVBoxLayout(self)

        head = QLabel(
            "给 IP-Adapter 配多张参考图，<b>每张各管一件事</b>："
            "身份（人脸专用权重）/ 风格（画风质感）/ 构图（布局色彩）。<br>"
            "每张图有自己的权重（0~2，最终是每个适配器一个独立强度）；"
            "「区域蒙版」可选，填了就只让这张图影响蒙版区域（白色=生效）。<br>"
            "多张参考图是<b>一次</b>装进模型、跑<b>一次</b>采样，"
            "不会变成跑好几遍。")
        head.setWordWrap(True)
        lay.addWidget(head)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["启用", "图片路径", "角色", "权重", "区域蒙版（可选）"])
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setDefaultSectionSize(30)
        h = self.table.horizontalHeader()
        for c in (COL_ENABLE, COL_ROLE, COL_WEIGHT):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        for c in (COL_PATH, COL_MASK):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)

        btns = QHBoxLayout()
        for text, slot, tip in (
                ("添加", self._add_ref, "新增一条参考图"),
                ("删除", self._remove_selected, "删除选中的那条"),
                ("↑", lambda: self._move(-1), "上移"),
                ("↓", lambda: self._move(1), "下移")):
            b = QPushButton(text)
            b.clicked.connect(slot)
            if tip:
                b.setToolTip(tip)
            btns.addWidget(b)
        btns.addStretch(1)
        self.btn_add_identity = QPushButton("加身份图")
        self.btn_add_identity.setToolTip("新增一条「身份」参考图（plus-face 权重）")
        self.btn_add_identity.clicked.connect(lambda: self._add_ref("identity"))
        btns.addWidget(self.btn_add_identity)
        self.btn_add_style = QPushButton("加风格图")
        self.btn_add_style.setToolTip("新增一条「风格」参考图（plus 权重）")
        self.btn_add_style.clicked.connect(lambda: self._add_ref("style"))
        btns.addWidget(self.btn_add_style)
        lay.addLayout(btns)

        self.lbl_installed = QLabel(installed_variants_text())
        self.lbl_installed.setWordWrap(True)
        self.lbl_installed.setStyleSheet("color:#7f849c; font-size:11px;")
        lay.addWidget(self.lbl_installed)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("color:#a6adc8;")
        lay.addWidget(self.lbl_summary)

        bottom = QHBoxLayout()
        b_save = QPushButton("保存参考集")
        b_save.clicked.connect(self._save_set)
        b_load = QPushButton("载入参考集")
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

        self._set_refs(list(refs or []))

    # ------------------------------------------------------------------
    #  数据层
    # ------------------------------------------------------------------
    def _set_refs(self, refs: list):
        """外部数据 → 内部条目（**容错**：脏数据一律跳过，绝不抛）。"""
        self._loading = True
        try:
            self._refs = []
            for r in refs:
                if not isinstance(r, dict):
                    continue
                self._refs.append(self._normalize(r))
            self._rebuild_table()
        finally:
            self._loading = False
        self._update_summary()

    @staticmethod
    def _normalize(r: dict) -> dict:
        """一条参考图 → 界面用的规范 dict（缺字段/怪类型都不抛）。"""
        from core import ipa_multi as ipa_mod
        try:
            weight = float(r.get("weight", ipa_mod.DEFAULT_WEIGHT))
        except (TypeError, ValueError):
            weight = float(ipa_mod.DEFAULT_WEIGHT)
        weight = max(0.0, min(2.0, weight))
        return {
            "role": ipa_mod.normalize_role(r.get("role")),
            "weight": weight,
            "path": str(r.get("path") or "").strip(),
            "mask_path": str(r.get("mask_path") or "").strip(),
            "enabled": bool(r.get("enabled", True)),
        }

    def _new_ref(self, role: str = "style") -> dict:
        from core import ipa_multi as ipa_mod
        return {"role": ipa_mod.normalize_role(role),
                "weight": float(ipa_mod.DEFAULT_WEIGHT),
                "path": "", "mask_path": "", "enabled": True}

    def refs(self) -> list:
        """返回可直接写进配置 / 交给引擎的参考图列表（纯 dict）。"""
        return [dict(r) for r in self._refs]

    # ------------------------------------------------------------------
    #  表格
    # ------------------------------------------------------------------
    def _rebuild_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._refs))
        for i, r in enumerate(self._refs):
            chk = QCheckBox()
            chk.setChecked(bool(r.get("enabled", True)))
            chk.setToolTip("取消勾选 = 这条仍占位（保持适配器编号对齐），"
                           "但强度按 0 处理，不影响画面")
            chk.toggled.connect(
                lambda v, idx=i: self._set_field(idx, "enabled", bool(v)))
            self.table.setCellWidget(i, COL_ENABLE, chk)

            self.table.setCellWidget(i, COL_PATH, self._path_cell(
                i, "path", "选择参考图", _IMG_FILTER))

            combo = QComboBox()
            from core import ipa_multi as ipa_mod
            for role in ROLE_ORDER:
                combo.addItem(ipa_mod.role_label(role), role)
            j = combo.findData(str(r.get("role", "style")))
            combo.setCurrentIndex(j if j >= 0 else 0)
            combo.setToolTip(
                "身份 = plus-face（人脸专用）\n"
                "风格 = plus（画风/质感）\n"
                "构图 = base（布局/色彩）\n"
                "本机没装的档位会自动降级并提示，不会让生成失败")
            combo.currentIndexChanged.connect(
                lambda _x, idx=i, c=combo: self._set_field(
                    idx, "role", str(c.currentData() or "style")))
            self.table.setCellWidget(i, COL_ROLE, combo)

            sp = QDoubleSpinBox()
            sp.setRange(0.0, 2.0)
            sp.setSingleStep(0.05)
            sp.setDecimals(2)
            sp.setValue(float(r.get("weight", 0.6)))
            sp.setToolTip("该参考图的独立强度（最终是每个适配器一个 scale）")
            sp.valueChanged.connect(
                lambda v, idx=i: self._set_field(idx, "weight", float(v)))
            self.table.setCellWidget(i, COL_WEIGHT, sp)

            self.table.setCellWidget(i, COL_MASK, self._path_cell(
                i, "mask_path", "选择区域蒙版（白色=生效）", _IMG_FILTER))
        self.table.blockSignals(False)

    def _path_cell(self, idx: int, key: str, title: str,
                   filter_str: str) -> QWidget:
        """一行里的「路径 + 浏览…」小控件（QLineEdit 的编辑照样回写数据）。"""
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        ed = QLineEdit(str(self._refs[idx].get(key, "") or ""))
        ed.setPlaceholderText("（空）" if key == "mask_path" else "选择图片…")
        ed.textChanged.connect(
            lambda v, i=idx, k=key: self._set_field(i, k, str(v)))
        row.addWidget(ed, 1)
        btn = QPushButton("浏览…")
        btn.setFixedWidth(60)
        btn.setToolTip(title)
        btn.clicked.connect(lambda _x, i=idx, k=key, t=title, f=filter_str:
                            self._pick_path(i, k, t, f))
        row.addWidget(btn)
        return wrap

    def _pick_path(self, idx: int, key: str, title: str, filter_str: str):
        if not (0 <= idx < len(self._refs)):
            return
        cur = str(self._refs[idx].get(key, "") or "")
        start = os.path.dirname(cur) if cur and os.path.isdir(
            os.path.dirname(cur)) else ""
        path, _ = QFileDialog.getOpenFileName(self, title, start, filter_str)
        if not path:
            return
        self._loading = True
        try:
            self._refs[idx][key] = path
            self._rebuild_table()
        finally:
            self._loading = False
        self._update_summary()

    def _set_field(self, idx, key, value):
        if self._loading or not (0 <= idx < len(self._refs)):
            return
        self._refs[idx][key] = value
        self._update_summary()

    def _update_summary(self):
        if not hasattr(self, "lbl_summary"):
            return
        enabled = [r for r in self._refs if r.get("enabled", True)]
        if not enabled:
            self.lbl_summary.setText(
                "未启用任何参考图（生成时不做多参考图）")
            return
        missing = sum(1 for r in enabled if not str(r.get("path") or "").strip())
        txt = summary_text(self._refs)
        if missing:
            txt += f"（其中 {missing} 张还没选图片，生成时会被跳过）"
        self.lbl_summary.setText(txt)

    # ------------------------------------------------------------------
    #  增删改序
    # ------------------------------------------------------------------
    def _add_ref(self, role: str = "style"):
        self._refs.append(self._new_ref(role))
        self._rebuild_table()
        self._update_summary()
        row = len(self._refs) - 1
        self.table.selectRow(row)
        cell = self.table.cellWidget(row, COL_PATH)
        ed = cell.findChild(QLineEdit) if cell is not None else None
        if ed is not None:
            ed.setFocus()

    def _remove_selected(self):
        idx = self.table.currentRow()
        if 0 <= idx < len(self._refs):
            self._refs.pop(idx)
            self._rebuild_table()
            self._update_summary()

    def _move(self, delta):
        idx = self.table.currentRow()
        j = idx + delta
        if 0 <= idx < len(self._refs) and 0 <= j < len(self._refs):
            self._refs[idx], self._refs[j] = self._refs[j], self._refs[idx]
            self._rebuild_table()
            self.table.selectRow(j)

    # ------------------------------------------------------------------
    #  参考集（JSON 文件）
    # ------------------------------------------------------------------
    def _sets_dir(self) -> str:
        try:
            from utils import paths
            d = os.path.join(paths.PROJECT_ROOT, "ipa_refs")
        except Exception:
            d = os.path.abspath("ipa_refs")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_set(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存参考集", os.path.join(self._sets_dir(), "参考集.json"),
            "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"refs": self.refs()}, f,
                          ensure_ascii=False, indent=2)
            QMessageBox.information(self, "已保存", path)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _load_set(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "载入参考集", self._sets_dir(), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            # ⚠️ 坏 JSON 只提示，绝不抛 —— 抛出去会让整个对话框关掉、
            # 用户已经填好的内容全丢。
            QMessageBox.warning(self, "载入失败", f"这个文件读不出来：\n{e}")
            return
        refs = data
        if isinstance(data, dict):
            for key in ("refs", "references", "参考图"):
                if isinstance(data.get(key), list):
                    refs = data[key]
                    break
            else:
                refs = []
        if not isinstance(refs, list):
            refs = []
        self._set_refs(refs)

    # ------------------------------------------------------------------
    def accept(self):
        # 空路径的条目留着（用户可能只是想先排好角色再选图），
        # 但至少要有一条能用的，否则提醒一句（不阻止确定）。
        enabled = [r for r in self._refs if r.get("enabled", True)]
        usable = [r for r in enabled if str(r.get("path") or "").strip()]
        if enabled and not usable:
            r = QMessageBox.question(
                self, "还没选图片",
                "启用的参考图都没有图片路径，生成时会被跳过。\n仍要保存吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return
        super().accept()


__all__ = ["IPARefEditorDialog", "summary_text", "installed_variants_text",
           "ROLE_ORDER", "COL_ENABLE", "COL_PATH", "COL_ROLE", "COL_WEIGHT",
           "COL_MASK"]
