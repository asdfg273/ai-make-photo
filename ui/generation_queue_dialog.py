"""生成队列管理对话框：编辑仅作用于等待中的参数快照。"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSpinBox, QSplitter, QTextEdit,
                             QVBoxLayout, QWidget)


class GenerationQueueDialog(QDialog):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.queue = host._generation_queue_state()
        self._loading = False
        self.setWindowTitle("生成队列")
        self.resize(760, 520)

        root = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_promote = QPushButton("插队到最前")
        self.btn_promote.clicked.connect(lambda: self._change("promote"))
        self.btn_up = QPushButton("上移")
        self.btn_up.clicked.connect(lambda: self._change("move", -1))
        self.btn_down = QPushButton("下移")
        self.btn_down.clicked.connect(lambda: self._change("move", 1))
        self.btn_remove = QPushButton("移除等待项")
        self.btn_remove.clicked.connect(lambda: self._change("remove"))
        for button in (self.btn_pause, self.btn_promote, self.btn_up,
                       self.btn_down, self.btn_remove):
            toolbar.addWidget(button)
        toolbar.addStretch()
        root.addLayout(toolbar)

        splitter = QSplitter()
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._selection_changed)
        splitter.addWidget(self.list)

        editor = QWidget()
        form = QFormLayout(editor)
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setAcceptRichText(False)
        form.addRow("正向提示词:", self.txt_prompt)
        self.txt_neg = QTextEdit()
        self.txt_neg.setAcceptRichText(False)
        form.addRow("负向提示词:", self.txt_neg)
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 150)
        form.addRow("步数:", self.spin_steps)
        self.spin_cfg = QDoubleSpinBox()
        self.spin_cfg.setRange(0, 30)
        self.spin_cfg.setSingleStep(0.5)
        form.addRow("CFG:", self.spin_cfg)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 2147483647)
        form.addRow("种子:", self.spin_seed)
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 32)
        form.addRow("数量:", self.spin_count)
        self.btn_save = QPushButton("保存等待项参数")
        self.btn_save.clicked.connect(self._save)
        form.addRow(self.btn_save)
        splitter.addWidget(editor)
        splitter.setSizes([300, 460])
        root.addWidget(splitter, 1)

        self.lbl_status = QLabel("暂停在采样步边界生效；运行中任务请用主窗口“中断生成”。")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self._editor_widgets = [self.txt_prompt, self.txt_neg, self.spin_steps,
                                self.spin_cfg, self.spin_seed, self.spin_count,
                                self.btn_save]
        self._timer = QTimer(self)
        self._timer.setInterval(600)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def _selected_id(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _toggle_pause(self):
        self.host.toggle_generation_queue_pause()
        self.refresh()

    def _change(self, method, *args):
        job_id = self._selected_id()
        if not job_id:
            return
        try:
            getattr(self.queue, method)(job_id, *args)
            self.host._generation_queue_timer.start()
        except (KeyError, ValueError) as exc:
            self.lbl_status.setText(str(exc))
        self.refresh()

    def _save(self):
        job_id = self._selected_id()
        if not job_id:
            return
        values = {
            "txt_prompt": self.txt_prompt.toPlainText(),
            "txt_neg": self.txt_neg.toPlainText(),
            "spin_steps": self.spin_steps.value(),
            "scale_cfg": self.spin_cfg.value(),
            "spin_seed": self.spin_seed.value(),
            "spin_count": self.spin_count.value(),
        }
        try:
            self.queue.update(job_id, values)
            self.lbl_status.setText("等待项参数已保存，派发时会恢复整套快照。")
        except (KeyError, ValueError) as exc:
            self.lbl_status.setText(str(exc))
        self.refresh()

    def _selection_changed(self, *_):
        if self._loading:
            return
        job_id = self._selected_id()
        job = next((item for item in self.queue.jobs() if item.id == job_id), None)
        editable = bool(job and job.status == "pending")
        for widget in self._editor_widgets:
            widget.setEnabled(editable)
        if job_id == getattr(self, "_loaded_job_id", None):
            return
        self._loaded_job_id = job_id
        if not job:
            return
        values = job.snapshot.get("values", {})
        self._loading = True
        try:
            self.txt_prompt.setPlainText(str(values.get("txt_prompt", "")))
            self.txt_neg.setPlainText(str(values.get("txt_neg", "")))
            self.spin_steps.setValue(int(values.get("spin_steps", 30)))
            self.spin_cfg.setValue(float(values.get("scale_cfg", 7.0)))
            self.spin_seed.setValue(int(values.get("spin_seed", -1)))
            self.spin_count.setValue(int(values.get("spin_count", 1)))
        finally:
            self._loading = False

    def refresh(self):
        selected = self._selected_id()
        status_names = {"pending": "等待", "running": "运行中", "done": "完成",
                        "failed": "失败", "cancelled": "已取消"}
        self._loading = True
        try:
            self.list.clear()
            for job in self.queue.jobs()[-100:]:
                text = f"[{status_names.get(job.status, job.status)}] {job.label}"
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, job.id)
                if job.error:
                    item.setToolTip(job.error)
                self.list.addItem(item)
                if job.id == selected:
                    self.list.setCurrentItem(item)
            if self.list.currentItem() is None and self.list.count():
                self.list.setCurrentRow(0)
            self.btn_pause.setText("恢复" if self.queue.paused else "暂停")
        finally:
            self._loading = False
        self._selection_changed()
