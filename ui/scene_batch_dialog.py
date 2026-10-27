# ui/scene_batch_dialog.py
"""场景批量出图对话框 —— 内置场景编辑 + 批量出图。
"""

from __future__ import annotations

import logging
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QProgressBar, QMessageBox, QLineEdit,
    QPlainTextEdit, QSpinBox, QCheckBox, QGroupBox, QSplitter, QWidget,
    QDoubleSpinBox,
)

from ui.i18n import ui_i18n
logger = logging.getLogger(__name__)

# ── 场景字段表：唯一需要维护字段名的地方 ───────────────────────
# (Scene 属性名, 显示名, 控件类型, 默认值)
_SCENE_FIELDS = (
    ("name",           "场景名",        "str",  ""),
    ("pose_prompt",    "姿势/正向补充",  "text", ""),
    ("negative_extra", "负面补充",       "text", ""),
    ("count",          "出图张数",       "int",  1),
    ("seed",           "种子(-1随机)",   "int",  -1),
    ("enabled",        "启用",           "bool", True),
)

_OPT_FIELDS = (
    ("steps",  "步数", "int",   1,   150),
    ("cfg",    "CFG",  "float", 0.0, 30.0),
    ("width",  "宽",   "int",   64,  4096),
    ("height", "高",   "int",   64,  4096),
)


def _make_widget(kind, lo=0, hi=100):
    if kind == "int":
        w = QSpinBox()
        w.setRange(int(lo), int(hi))
    elif kind == "float":
        w = QDoubleSpinBox()
        w.setRange(float(lo), float(hi))
        w.setSingleStep(0.5)
        w.setDecimals(1)
    else:
        w = QLineEdit()
    return w

def _new_scene():
    """构造空白 Scene。兼容『无参构造』和『from_dict 构造』两种形态。"""
    from utils.scene_library import Scene
    try:
        return Scene()
    except TypeError:
        return Scene.from_dict({})


class SceneBatchDialog(QDialog):
    def __init__(self, shell=None, character=None, library=None,
                 runner=None, parent=None):
        super().__init__(parent)
        self.library = library
        self.character = character
        self.shell = shell
        self._runner = runner
        self._owns_runner = runner is None
        self._attached = False
        self._scenes: list = []
        self._widgets: dict = {}
        self._loading = False

        name = getattr(character, "name", ui_i18n.tr("未命名角色"))
        self.setWindowTitle(ui_i18n.trf("批量出图 — {name}", name=name))
        self.setMinimumSize(760, 520)
        self._build_ui()
        self._reload(select_uuid=None)

    # ---------------- UI ----------------
    def _build_ui(self):
        v = QVBoxLayout(self)

        self.lbl_head = QLabel()
        self.lbl_head.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.lbl_head)

        split = QSplitter(Qt.Orientation.Horizontal)

        # 左：场景列表
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.lst = QListWidget()
        self.lst.itemChanged.connect(self._on_item_changed)
        self.lst.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.lst.currentItemChanged.connect(self._on_row_changed)
        lv.addWidget(self.lst, 1)

        row = QHBoxLayout()
        for text, slot in (("新建", self._on_add), ("复制", self._on_dup),
                           ("删除", self._on_del), ("↑", lambda: self._on_move(-1)),
                           ("↓", lambda: self._on_move(1))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lv.addLayout(row)

        row2 = QHBoxLayout()
        b_all = QPushButton("全选")
        b_none = QPushButton("全不选")
        b_all.clicked.connect(lambda: self._set_all(True))
        b_none.clicked.connect(lambda: self._set_all(False))
        row2.addWidget(b_all)
        row2.addWidget(b_none)
        lv.addLayout(row2)
        split.addWidget(left)

        # 右：字段表单
        box = QGroupBox("场景参数")
        g = QGridLayout(box)
        for r, (key, label, kind, default) in enumerate(_SCENE_FIELDS):
            g.addWidget(QLabel(label + ":"), r, 0)
            if kind == "bool":
                w = QCheckBox()
            elif kind == "int":
                w = QSpinBox()
                w.setRange(-1, 10_000_000) if key == "seed" else w.setRange(1, 999)
            elif kind == "text":
                w = QPlainTextEdit()
                w.setFixedHeight(72)
            else:
                w = QLineEdit()
            g.addWidget(w, r, 1)
            self._widgets[key] = w
            self._connect_field(w, kind)
        g.setRowStretch(len(_SCENE_FIELDS), 1)

        # ── 可选覆盖参数（不勾选 = 沿用页面当前值）──
        r = len(_SCENE_FIELDS)
        g.addWidget(QLabel("── 覆盖页面参数 ──"), r, 0, 1, 2)
        self._opt_widgets = {}
        for i, (key, label, kind, lo, hi) in enumerate(_OPT_FIELDS, start=r + 1):
            cb = QCheckBox(label)
            sw = _make_widget(kind, lo, hi)
            sw.setEnabled(False)
            cb.toggled.connect(
                lambda on, w=sw: (w.setEnabled(on), self._on_field_changed()))
            g.addWidget(cb, i, 0)
            g.addWidget(sw, i, 1)
            self._opt_widgets[key] = (cb, sw)

        btn_save = QPushButton("保存场景")
        btn_save.clicked.connect(self._save_current)
        g.addWidget(btn_save, len(_SCENE_FIELDS) + 1, 1)
        split.addWidget(box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        v.addWidget(split, 1)

        self.bar = QProgressBar()
        v.addWidget(self.bar)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color:#7d8187; font-size:12px;")
        self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)

        row3 = QHBoxLayout()
        self.btn_run = QPushButton("开始批量出图")
        self.btn_open_out = QPushButton("打开输出目录")
        self.btn_close = QPushButton("关闭")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_open_out.clicked.connect(self._open_output_dir)
        self.btn_close.clicked.connect(self.close)
        row3.addWidget(self.btn_run)
        row3.addStretch(1)
        row3.addWidget(self.btn_open_out)
        row3.addWidget(self.btn_close)
        v.addLayout(row3)

    def _connect_field(self, w, kind):
        if kind == "bool":
            w.toggled.connect(self._on_field_changed)
        elif kind == "int":
            w.valueChanged.connect(self._on_field_changed)
        elif kind == "text":
            w.textChanged.connect(self._on_field_changed)   # 由 _loading 挡住回灌
        else:
            w.textEdited.connect(self._on_field_changed)

    # ---------------- 数据 ----------------
    def _char_uuid(self) -> str:
        return getattr(self.character, "uuid", "") or ""

    def _reload(self, select_uuid=None):
        self.lst.blockSignals(True)
        self.lst.clear()
        self._scenes = []
        if self.library is not None:
            try:
                self._scenes = list(self.library.list_for_character(self._char_uuid()) or [])
            except Exception as e:
                logger.exception("读取场景失败")
                self.lbl_status.setText(ui_i18n.trf("读取场景失败：{error}", error=e))

        for sc in self._scenes:
            it = QListWidgetItem(self._row_text(sc))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if getattr(sc, "enabled", True)
                             else Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, sc)
            self.lst.addItem(it)
        self.lst.blockSignals(False)

        shots = sum(max(1, int(getattr(s, "count", 1) or 1))
                    for s in self._scenes if getattr(s, "enabled", True))
        self.lbl_head.setText(ui_i18n.trf(
            "角色：<b>{character}</b>　场景：<b>{scenes}</b> 个　启用出图：<b>{shots}</b> 张",
            character=getattr(self.character, "name", "—"),
            scenes=len(self._scenes), shots=shots))
        self.btn_run.setEnabled(bool(self._scenes))

        # 选中目标行
        idx = 0
        if select_uuid:
            for i, sc in enumerate(self._scenes):
                if getattr(sc, "uuid", "") == select_uuid:
                    idx = i
                    break
        if self.lst.count():
            self.lst.setCurrentRow(idx)
        else:
            self._load_form(None)

    def _row_text(self, sc) -> str:
        label = getattr(sc, "name", "") or ui_i18n.tr("(未命名)")
        text = f"{label}　×{max(1, int(getattr(sc, 'count', 1) or 1))}"
        st = getattr(sc, "status", "") or ""
        if st and st != "pending":
            text += f"　[{ui_i18n.tr(st)}]"
        return text

    def _current(self):
        it = self.lst.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def _load_form(self, sc):
        self._loading = True
        try:
            for key, _label, kind, default in _SCENE_FIELDS:
                w = self._widgets[key]
                val = getattr(sc, key, default) if sc is not None else default
                if kind == "bool":
                    w.setChecked(bool(val))
                elif kind == "int":
                    try:
                        w.setValue(int(val))
                    except (TypeError, ValueError):
                        w.setValue(int(default))
                elif kind == "text":
                    w.setPlainText(str(val or ""))
                else:
                    w.setText(str(val or ""))
            for key in ("name", "pose_prompt", "negative_extra", "count", "seed", "enabled"):
                self._widgets[key].setEnabled(sc is not None)
            for key, _l, _k, lo, _hi in _OPT_FIELDS:
                cb, sw = self._opt_widgets[key]
                val = getattr(sc, key, None) if sc is not None else None
                cb.setChecked(val is not None)
                sw.setValue(val if val is not None else lo)
                cb.setEnabled(sc is not None)
                sw.setEnabled(sc is not None and val is not None)
        finally:
            self._loading = False

    def _on_row_changed(self, *_):
        self._load_form(self._current())

    def _on_field_changed(self, *_):
        if self._loading:
            return
        sc = self._current()
        if sc is None:
            return
        self._collect(sc)
        it = self.lst.currentItem()
        if it is not None:
            it.setText(self._row_text(sc))

    def _refresh_header(self):
        shots = sum(max(1, int(getattr(s, "count", 1) or 1))
                    for s in self._scenes if getattr(s, "enabled", True))
        self.lbl_head.setText(ui_i18n.trf(
            "角色：<b>{character}</b>　场景：<b>{scenes}</b> 个　启用出图：<b>{shots}</b> 张",
            character=getattr(self.character, "name", "—"),
            scenes=len(self._scenes), shots=shots))

    def _on_item_changed(self, it):
        if self._loading or self.library is None:
            return
        sc = it.data(Qt.ItemDataRole.UserRole)
        if sc is None:
            return
        sc.enabled = (it.checkState() == Qt.CheckState.Checked)
        self.library.save()
        self._refresh_header()
        if sc is self._current():          # 同步右侧的「启用」
            self._loading = True
            self._widgets["enabled"].setChecked(sc.enabled)
            self._loading = False

    def _collect(self, sc):
        for key, _label, kind, _default in _SCENE_FIELDS:
            w = self._widgets[key]
            if kind == "bool":
                setattr(sc, key, bool(w.isChecked()))
            elif kind == "int":
                setattr(sc, key, int(w.value()))
            elif kind == "text":
                setattr(sc, key, w.toPlainText().strip())
            else:
                setattr(sc, key, w.text().strip())
        for key, _l, _k, _lo, _hi in _OPT_FIELDS:
            cb, sw = self._opt_widgets[key]
            setattr(sc, key, sw.value() if cb.isChecked() else None)

    # ---------------- 场景增删改 ----------------
    def _save_current(self):
        sc = self._current()
        if sc is None:
            QMessageBox.information(
                self, ui_i18n.tr("提示"), ui_i18n.tr("请先在左侧选择一个场景。"))
            return
        self._collect(sc)
        if self.library is not None:
            self.library.save()
        it = self.lst.currentItem()
        if it is not None:
            it.setText(self._row_text(sc))
        name = getattr(sc, "name", "") or ui_i18n.tr("(未命名)")
        self.lbl_status.setText(ui_i18n.trf("已保存：{name}", name=name))

    def _on_add(self):
        if self.library is None:
            return
        sc = _new_scene()
        sc.character_uuid = self._char_uuid()
        # 默认值灌进去，避免控件默认和 Scene 默认不一致
        for key, _label, _kind, default in _SCENE_FIELDS:
            if getattr(sc, key, None) in (None, ""):
                setattr(sc, key, default)
        setattr(sc, "name", getattr(sc, "name", "") or "新场景")
        try:
            self.library.add(sc)
            self.library.save()
        except Exception as e:
            logger.exception("新增场景失败")
            QMessageBox.critical(
                self, ui_i18n.tr("错误"), ui_i18n.trf("新增场景失败：{error}", error=e))
            return
        self._reload(select_uuid=getattr(sc, "uuid", None))

    def _on_dup(self):
        sc = self._current()
        if sc is None or self.library is None:
            return
        try:
            import copy
            new = copy.deepcopy(sc)
            setattr(new, "uuid", "")           # 让库分配新 uuid
            setattr(new, "name", (getattr(sc, "name", "") or "场景") + " 副本")
            self.library.add(new)
            self.library.save()
        except Exception as e:
            logger.exception("复制场景失败")
            QMessageBox.critical(
                self, ui_i18n.tr("错误"), ui_i18n.trf("复制场景失败：{error}", error=e))
            return
        self._reload(select_uuid=getattr(new, "uuid", None))

    def _on_del(self):
        sc = self._current()
        if sc is None or self.library is None:
            return
        nm = getattr(sc, "name", "") or ui_i18n.tr("(未命名)")
        if QMessageBox.question(
                self, ui_i18n.tr("确认"), ui_i18n.trf("删除场景「{name}」？", name=nm)) \
                != QMessageBox.StandardButton.Yes:
            return
        try:
            self.library.delete(getattr(sc, "uuid", ""))
            self.library.save()
        except Exception as e:
            logger.exception("删除场景失败")
            QMessageBox.critical(
                self, ui_i18n.tr("错误"), ui_i18n.trf("删除场景失败：{error}", error=e))
            return
        self._reload()

    def _on_move(self, delta: int):
        sc = self._current()
        if sc is None or self.library is None:
            return
        try:
            self.library.move(getattr(sc, "uuid", ""), delta)
            self.library.save()
        except Exception:
            logger.exception("移动场景失败")
            return
        self._reload(select_uuid=getattr(sc, "uuid", None))

    def _set_all(self, on: bool):
        self.lst.blockSignals(True)
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            sc = it.data(Qt.ItemDataRole.UserRole)
            if sc is not None:
                setattr(sc, "enabled", on)
        self.lst.blockSignals(False)
        if self.library is not None:
            self.library.save()
        self._refresh_header()

    # ---------------- 运行 ----------------
    def _selected(self) -> list:
        out = []
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            if it.checkState() != Qt.CheckState.Checked:
                continue
            sc = it.data(Qt.ItemDataRole.UserRole)
            if sc is not None:
                out.append(sc)
        return out

    def _open_output_dir(self):
        try:
            from utils.batch_runner import resolve_output_dir
            d = str(resolve_output_dir())
            os.startfile(d)
        except Exception as e:
            logger.error(f"打开输出目录失败: {e}")
            QMessageBox.critical(
                self, ui_i18n.tr("错误"), ui_i18n.trf("打开输出目录失败：{error}", error=e))

    def _on_run(self):
        chosen = self._selected()
        if not chosen:
            QMessageBox.information(
                self, ui_i18n.tr("提示"), ui_i18n.tr("请至少勾选一个场景。"))
            return
        if self.shell is None:
            QMessageBox.critical(
                self, ui_i18n.tr("错误"), ui_i18n.tr("缺少宿主对象（shell），无法启动生成。"))
            return
        self._save_current()

        if self._runner is None:
            try:
                from utils.batch_runner import BatchRunner
            except Exception as e:
                QMessageBox.critical(
                    self, ui_i18n.tr("错误"),
                    ui_i18n.trf("BatchRunner 导入失败：{error}", error=e))
                return
            self._runner = BatchRunner(self.shell, self.library, parent=self)
            self._owns_runner = True

        if self._runner.running:
            QMessageBox.information(
                self, ui_i18n.tr("提示"), ui_i18n.tr("已有批量任务在运行中。"))
            return

        self._attach()
        shots = sum(max(1, int(getattr(s, "count", 1) or 1)) for s in chosen)
        self.bar.setRange(0, shots)
        self.bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_close.setEnabled(False)

        if not self._runner.start(self.character, chosen):
            self.lbl_status.setText(ui_i18n.tr("启动失败（可能已有任务在运行）"))
            self.btn_run.setEnabled(True)
            self.btn_close.setEnabled(True)
            self._detach()

    def _attach(self):
        self._detach()
        self._runner.progress.connect(self._on_progress)
        self._runner.scene_done.connect(self._on_scene_done)
        self._runner.log.connect(self._on_log)
        self._runner.finished.connect(self._on_finished)
        self._attached = True

    def _detach(self):
        if not self._attached or self._runner is None:
            return
        for sig, slot in ((self._runner.progress, self._on_progress),
                          (self._runner.scene_done, self._on_scene_done),
                          (self._runner.log, self._on_log),
                          (self._runner.finished, self._on_finished)):
            try:
                sig.disconnect(slot)
            except TypeError:
                pass
        self._attached = False

    def _on_progress(self, done: int, total: int, msg: str = ""):
        if total > 0:
            self.bar.setRange(0, total)
        self.bar.setValue(done)
        if msg:
            self.lbl_status.setText(msg)

    def _on_scene_done(self, scene_uuid: str, status: str):
        sc = self.library.get(scene_uuid) if self.library else None
        nm = getattr(sc, "name", scene_uuid[:8])
        self.lbl_status.setText(ui_i18n.trf(
            "场景完成：{name} — {status}", name=nm, status=ui_i18n.tr(status)))
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            if it.data(Qt.ItemDataRole.UserRole) is sc and sc is not None:
                it.setText(self._row_text(sc))
                break

    def _on_log(self, text: str):
        if text:
            self.lbl_status.setText(text)

    def _on_finished(self, ok: int, fail: int):
        if self.bar.maximum() > 0:
            self.bar.setValue(self.bar.maximum())
        self.lbl_status.setText(ui_i18n.trf("完成：成功 {ok} / 失败 {fail}", ok=ok, fail=fail))
        self.btn_run.setEnabled(True)
        self.btn_close.setEnabled(True)
        self._detach()
        self._reload(select_uuid=None)
        logger.info(f"[BatchDialog] 完成 成功={ok} 失败={fail}")

    def closeEvent(self, ev):
        if self._runner is not None and self._runner.running:
            r = QMessageBox.question(
                self, ui_i18n.tr("确认"), ui_i18n.tr("批量任务正在运行，确定中止并关闭？"))
            if r != QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
            self._runner.stop()
        if self._owns_runner:
            self._runner = None
        else:
            self._detach()
        super().closeEvent(ev)