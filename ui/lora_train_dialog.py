# ui/lora_train_dialog.py
"""轻量 LoRA 训练窗口。

界面只做三件事：收集参数、跑训练线程、把进度/日志显示出来。所有训练逻辑
都在 `core/lora_train.py`，UI 不碰 torch。

线程模型与项目其它地方一致：QThread 里跑训练，回调通过 pyqtSignal 回主线程。
"""
from __future__ import annotations

import logging
import os
import time

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QDoubleSpinBox, QSpinBox, QVBoxLayout, QWidget,
)

from core import lora_train as lt

logger = logging.getLogger(__name__)


class TrainerThread(QThread):
    """把 `train_lora` 包成线程；回调全部转成信号。"""

    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(object)

    def __init__(self, cfg: lt.TrainConfig, base_model: str, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.base_model = base_model
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102 - QThread 约定
        try:
            self.status.emit("正在加载底模（约 10~30 秒）...")
            pipe, arch = lt.load_pipeline_for_training(self.base_model)
            res = lt.train_lora(
                self.cfg,
                pipe=pipe,
                arch=arch,
                device="cuda",
                on_progress=lambda d, t, s="": self.progress.emit(d, t),
                on_status=lambda m: self.status.emit(m),
                on_log=lambda m: self.log.emit(m),
                should_cancel=lambda: self._cancel,
            )
        except Exception as exc:  # noqa: BLE001 - 交给界面显示
            logger.exception("LoRA 训练线程异常")
            res = lt.TrainResult(ok=False, error=str(exc) or exc.__class__.__name__)
        self.finished_ok.emit(res)


class LoraTrainDialog(QDialog):
    """LoRA 训练主窗口。"""

    def __init__(self, host: QWidget, dataset_default: str = ""):
        super().__init__(host)
        self.host = host
        self.thread: TrainerThread | None = None
        self._built = False
        self.setWindowTitle("🎓 训练 LoRA（轻量）")
        self.resize(760, 720)
        self._build_ui(dataset_default)
        self._built = True
        self._refresh_estimate()

    # ------------------------------------------------------------ UI
    def _build_ui(self, dataset_default: str) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        tip = QLabel(
            "只训练 LoRA 附加权重（底模冻结），8GB 显存可训 SD1.5。\n"
            "数据集：一个文件夹，放图片 + **同名 .txt** 标注（如 cat_01.png / cat_01.txt）。"
        )
        tip.setWordWrap(True)
        tip.setProperty("role", "hint")
        root.addWidget(tip)

        # --- 数据与底模 ---
        grp = QGroupBox("① 数据与底模")
        form = QFormLayout(grp)
        row = QHBoxLayout()
        self.edit_dataset = QLineEdit(dataset_default)
        btn_browse = QPushButton("浏览…")
        btn_browse.clicked.connect(self._pick_dataset)
        btn_scan = QPushButton("检查")
        btn_scan.setToolTip("统计图片数量、尺寸桶、缺标注的图片")
        btn_scan.clicked.connect(self._scan_now)
        row.addWidget(self.edit_dataset, 1)
        row.addWidget(btn_browse)
        row.addWidget(btn_scan)
        form.addRow("数据集目录:", row)

        self.combo_base = QComboBox()
        self._fill_models()
        form.addRow("底模:", self.combo_base)

        row_out = QHBoxLayout()
        self.edit_name = QLineEdit("my_lora")
        self.edit_trigger = QLineEdit("")
        self.edit_trigger.setPlaceholderText("触发词，如 mys（存成同名 .txt，主界面会自动带出）")
        row_out.addWidget(self.edit_name, 1)
        row_out.addWidget(self.edit_trigger, 1)
        form.addRow("输出名 / 触发词:", row_out)
        root.addWidget(grp)

        # --- 超参 ---
        grp2 = QGroupBox("② 训练参数")
        form2 = QFormLayout(grp2)
        row_r = QHBoxLayout()
        self.spin_rank = QSpinBox()
        self.spin_rank.setRange(1, 128)
        self.spin_rank.setValue(8)
        self.spin_rank.setToolTip("rank 越大越能学细节，也越容易过拟合；8 是常用值")
        self.spin_alpha = QSpinBox()
        self.spin_alpha.setRange(0, 128)
        self.spin_alpha.setValue(0)
        self.spin_alpha.setToolTip("0 = 跟随 rank（社区惯例 alpha=rank）")
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 20000)
        self.spin_steps.setValue(400)
        row_r.addWidget(QLabel("rank"))
        row_r.addWidget(self.spin_rank)
        row_r.addWidget(QLabel("alpha"))
        row_r.addWidget(self.spin_alpha)
        row_r.addWidget(QLabel("总步数"))
        row_r.addWidget(self.spin_steps)
        form2.addRow("容量 / 步数:", row_r)

        row_t = QHBoxLayout()
        self.combo_res = QComboBox()
        for r in (512, 640, 768, 1024):
            self.combo_res.addItem(f"{r}×{r}", r)
        self.combo_res.setToolTip("512 最快；768 以上 8GB 卡要更久，SDXL 建议 512")
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 8)
        self.spin_batch.setValue(1)
        self.spin_lr = QLineEdit("1e-4")
        self.spin_lr.setToolTip("UNet 侧学习率；LoRA 常用 1e-4 ~ 5e-4")
        row_t.addWidget(QLabel("分辨率"))
        row_t.addWidget(self.combo_res)
        row_t.addWidget(QLabel("batch"))
        row_t.addWidget(self.spin_batch)
        row_t.addWidget(QLabel("学习率"))
        row_t.addWidget(self.spin_lr)
        form2.addRow("尺寸 / 优化:", row_t)

        # ---- 学习率调度 / 优化器 ----
        row_s = QHBoxLayout()
        self.combo_lr_sched = QComboBox()
        for key, label in lt.LR_SCHEDULERS.items():
            self.combo_lr_sched.addItem(label, key)
        self.combo_lr_sched.setCurrentIndex(
            max(0, self.combo_lr_sched.findData("cosine")))
        self.combo_lr_sched.setToolTip(
            "余弦退火是最常用的：前期大步、后期小步收敛，末端保留一点学习率避免死锁。\n"
            "「余弦退火 + 重启」适合训练久一点、容易陷局部最优的情况；\n"
            "「按 loss 自动降」不看步数，只在 loss 不降时把学习率减半。")
        self.spin_min_lr = QDoubleSpinBox()
        self.spin_min_lr.setRange(0.0, 50.0)
        self.spin_min_lr.setDecimals(0)
        self.spin_min_lr.setValue(5.0)
        self.spin_min_lr.setSuffix(" %")
        self.spin_min_lr.setToolTip("末端学习率 = 学习率 × 这个百分比。0% = 一路降到 0")
        row_s.addWidget(QLabel("调度"))
        row_s.addWidget(self.combo_lr_sched, 2)
        row_s.addWidget(QLabel("末端"))
        row_s.addWidget(self.spin_min_lr, 1)
        form2.addRow("学习率调度:", row_s)

        row_o = QHBoxLayout()
        self.combo_optim = QComboBox()
        for key, label in lt.OPTIMIZERS.items():
            self.combo_optim.addItem(label, key)
        self.combo_optim.setToolTip(
            "8bit 优化器把优化器状态从 fp32 压到 int8，SD1.5 能省约 0.2GB、SDXL 约 0.8GB；"
            "没有 bitsandbytes 会自动回退普通 AdamW。\n"
            "Adafactor 最省显存，但收敛曲线更抖。")
        self.spin_wd = QDoubleSpinBox()
        self.spin_wd.setRange(0.0, 1.0)
        self.spin_wd.setDecimals(3)
        self.spin_wd.setSingleStep(0.005)
        self.spin_wd.setValue(0.01)
        self.spin_wd.setToolTip("权重衰减；过拟合（出图千篇一律）时调到 0.05~0.1")
        self.spin_accum = QSpinBox()
        self.spin_accum.setRange(1, 16)
        self.spin_accum.setValue(1)
        self.spin_accum.setToolTip(
            "梯度累积：每 N 步才更新一次权重，等效放大 batch。\n"
            "显存紧张但想要大 batch 效果时用它（速度略慢）")
        row_o.addWidget(QLabel("优化器"))
        row_o.addWidget(self.combo_optim, 2)
        row_o.addWidget(QLabel("权重衰减"))
        row_o.addWidget(self.spin_wd, 1)
        row_o.addWidget(QLabel("累积"))
        row_o.addWidget(self.spin_accum, 1)
        form2.addRow("优化细节:", row_o)

        self.chk_te = QCheckBox("同时训练文本编码器（更吃显存，SD1.5 约 +0.4GB）")
        self.chk_ckpt = QCheckBox("梯度检查点（省显存，默认开）")
        self.chk_ckpt.setChecked(True)
        self.chk_save = QCheckBox("每 50 步存一次中间结果")
        form2.addRow(self.chk_te)
        form2.addRow(self.chk_ckpt)
        form2.addRow(self.chk_save)
        root.addWidget(grp2)

        # --- 显存提示 + 按钮 ---
        self.lbl_estimate = QLabel("")
        self.lbl_estimate.setWordWrap(True)
        self.lbl_estimate.setProperty("role", "hint")
        root.addWidget(self.lbl_estimate)

        row_btn = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.setDefault(True)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.reject)
        row_btn.addWidget(self.btn_start)
        row_btn.addWidget(self.btn_stop)
        row_btn.addStretch(1)
        row_btn.addWidget(btn_close)
        root.addLayout(row_btn)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        root.addWidget(self.bar)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(160)
        root.addWidget(self.log_view, 1)

        # 数值变化时刷新估算
        for w in (self.spin_rank, self.spin_steps, self.spin_batch):
            w.valueChanged.connect(self._refresh_estimate)
        self.combo_res.currentIndexChanged.connect(self._refresh_estimate)
        self.combo_base.currentIndexChanged.connect(self._refresh_estimate)
        self.spin_accum.valueChanged.connect(self._refresh_estimate)
        self.combo_optim.currentIndexChanged.connect(self._refresh_estimate)
        self.chk_te.toggled.connect(self._refresh_estimate)
        self.chk_ckpt.toggled.connect(self._refresh_estimate)

    def _fill_models(self) -> None:
        from utils.model_scanner import scan_all_models

        self.combo_base.clear()
        try:
            groups = scan_all_models()
        except Exception as exc:  # noqa: BLE001 - 扫模型失败不该让窗口打不开
            logger.warning("扫描底模失败: %s", exc)
            groups = {}
        for arch in lt.TRAINABLE_ARCHES:
            for item in groups.get(arch, []) or []:
                if item.get("is_dir"):
                    continue
                self.combo_base.addItem(f"{arch} · {item['name']}", item["path"])
        if self.combo_base.count() == 0:
            self.combo_base.addItem("（models/ 下没有可训练的底模）", "")

    def _pick_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集文件夹", self.edit_dataset.text() or "")
        if path:
            self.edit_dataset.setText(path)
            self._scan_now()

    def _scan_now(self) -> None:
        records, warnings = lt.scan_dataset(self.edit_dataset.text(), self.edit_trigger.text().strip())
        if not records:
            self._log("扫描结果：" + ("；".join(warnings) or "没有图片"))
            return
        buckets = {}
        for rec in records:
            w, h = lt.bucket_size(rec["size"][0], rec["size"][1],
                                  self.combo_res.currentData() or 512)
            buckets[f"{w}x{h}"] = buckets.get(f"{w}x{h}", 0) + 1
        self._log(f"扫描到 {len(records)} 张图；尺寸桶：" +
                  "、".join(f"{k}×{v}张" for k, v in sorted(buckets.items())))
        for w in warnings:
            self._log("⚠️ " + w)

    def _current_cfg(self) -> lt.TrainConfig:
        try:
            lr = float(self.spin_lr.text())
        except ValueError:
            lr = 1e-4
        return lt.TrainConfig(
            dataset_dir=self.edit_dataset.text().strip(),
            base_model=self.combo_base.currentData() or "",
            output_name=self.edit_name.text().strip() or "my_lora",
            rank=int(self.spin_rank.value()),
            alpha=float(self.spin_alpha.value()),
            lr=lr,
            steps=int(self.spin_steps.value()),
            batch_size=int(self.spin_batch.value()),
            resolution=int(self.combo_res.currentData() or 512),
            train_text_encoder=self.chk_te.isChecked(),
            gradient_checkpointing=self.chk_ckpt.isChecked(),
            trigger=self.edit_trigger.text().strip(),
            save_every=50 if self.chk_save.isChecked() else 0,
            lr_scheduler=str(self.combo_lr_sched.currentData() or "cosine"),
            min_lr_ratio=float(self.spin_min_lr.value()) / 100.0,
            optimizer=str(self.combo_optim.currentData() or "adamw"),
            weight_decay=float(self.spin_wd.value()),
            grad_accumulation=int(self.spin_accum.value()),
        )

    def _refresh_estimate(self, *_args) -> None:
        if not self._built:
            return
        cfg = self._current_cfg()
        path = cfg.base_model
        arch = "sd15"
        if " · " in self.combo_base.currentText():
            arch = self.combo_base.currentText().split(" · ", 1)[0]
        mins = cfg.steps * (0.42 if arch in lt._SDXL_ARCHES else 0.22) * (
            1.25 if cfg.gradient_checkpointing else 1.0) * (cfg.resolution / 512) ** 2
        mb_free = 0
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                mb_free = int(_torch.cuda.mem_get_info()[0] / (1024 * 1024))
        except Exception:  # noqa: BLE001 - 拿不到显存就只报估算值
            mb_free = 0
        text = (lt.vram_advice(cfg, arch, available_mb=mb_free)
                + f"；大约 {mins / 60:.0f} 分钟（{cfg.steps} 步，不含加载与编码）")
        if arch in lt._SDXL_ARCHES:
            text += "。SDXL 在 8GB 上属于勉强能跑，建议 512 分辨率 + rank≤8。"
        if not path:
            text += "  ⚠️ 先选一个底模。"
        self.lbl_estimate.setText(text)

    # ------------------------------------------------------------ 运行
    def _start(self) -> None:
        cfg = self._current_cfg()
        if not cfg.dataset_dir or not os.path.isdir(cfg.dataset_dir):
            QMessageBox.warning(self, "缺少数据集", "请先选择一个包含图片（和同名 .txt 标注）的文件夹。")
            return
        if not cfg.base_model or not os.path.isfile(cfg.base_model):
            QMessageBox.warning(self, "缺少底模", "请在 models/ 下选择要训练的底模（SD1.5 / SDXL）。")
            return
        records, _ = lt.scan_dataset(cfg.dataset_dir, cfg.trigger)
        if not records:
            QMessageBox.warning(self, "数据集为空", "这个文件夹里没有可用图片。")
            return
        if len(records) < 5:
            if QMessageBox.question(
                self, "样本很少",
                f"只有 {len(records)} 张图，LoRA 很可能学不到东西（建议 ≥15 张）。仍然继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return
        self.log_view.clear()
        self.bar.setValue(0)
        self._log(f"开始：{len(records)} 张图，{cfg.steps} 步，rank {cfg.rank}，"
                  f"{cfg.resolution}px，lr {cfg.lr:g}")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.thread = TrainerThread(cfg, cfg.base_model, self)
        self.thread.progress.connect(self._on_progress)
        self.thread.status.connect(self._on_status)
        self.thread.log.connect(self._log)
        self.thread.finished_ok.connect(self._on_done)
        self.thread.start()

    def _stop(self) -> None:
        if self.thread is not None:
            self.thread.cancel()
            self._on_status("正在停止（当前这一步跑完就停）...")
            self.btn_stop.setEnabled(False)

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(min(done, total))

    def _on_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{stamp}] {text}")

    def _on_done(self, res: lt.TrainResult) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        for w in res.warnings or []:
            self._log("⚠️ " + w)
        if res.ok:
            self._on_status("完成")
            self._log(res.summary_text())
            QMessageBox.information(
                self, "训练完成",
                f"{res.summary_text()}\n\n"
                f"LoRA 已放到 loras/，主界面「LoRA」下拉框里可以直接选。"
                + ("\n触发词旁文件已写好。" if res.sidecar else ""),
            )
        elif res.cancelled:
            self._on_status("已取消")
            self._log("已取消（没有保存）")
        else:
            self._on_status("失败")
            self._log("❌ " + (res.error or "未知错误"))
            QMessageBox.critical(self, "训练失败", res.error or "未知错误")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if self.thread is not None and self.thread.isRunning():
            if QMessageBox.question(
                self, "训练还在进行", "训练还没结束，关闭窗口会中断它。确定关闭？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.thread.cancel()
            self.thread.wait(5000)
        super().closeEvent(event)
