# ui/workflow_editor/resource_panel.py
"""工作流资源面板：工作流依赖了哪些模型、哪些缺失、怎么补。

为什么需要它
------------
工作流原来只存一个硬编码路径，换台电脑/挪个目录就失效，而且用户
"不知道缺哪些模型"。现在工作流按 `resource_id` 记录依赖，这里把它
检查结果摊开给用户看，形成闭环：

    检查资源 → 存在 → 继续执行
             → 缺失 → 显示资源卡片 → 用户下载 → 校验 → 重新预检

 下载入口的边界（遵守项目既定策略）
------------------------------------
**底模 / LoRA / VAE 不给下载按钮** —— 模型由用户自行下载，这是项目明确
的策略。这几类缺失时只告诉用户"缺什么、放哪儿"。
只有**辅助组件**（ControlNet / ADetailer / IP-Adapter / 插帧 / Flux 引擎等，
即扩展市场里的 18 个条目）才提供「下载」按钮，走既有的
`utils.extension_manager.download_extension()`，不新增任何下载入口。
"""
from __future__ import annotations

import logging
import os
import sys

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout,
                             QHeaderView, QLabel, QMessageBox, QProgressBar,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout)
from ui.i18n import ui_i18n

logger = logging.getLogger(__name__)

# 状态 → (中文标签, 颜色)
STATUS_LABELS = {
    "ok": ("就绪", "#a6e3a1"),
    "found_elsewhere": ("位置已变", "#f9e2af"),
    "missing": ("缺失", "#f38ba8"),
    "arch_mismatch": ("架构不符", "#f38ba8"),
    "sha_mismatch": ("校验失败", "#f38ba8"),
    "unknown": ("未检查", "#6c7086"),
}

# 资源类型 → 用户可读
TYPE_LABELS = {
    "checkpoint": "底模", "lora": "LoRA", "vae": "VAE",
    "controlnet": "ControlNet", "upscaler": "放大/插帧",
    "engine": "新架构引擎", "adetailer": "ADetailer",
    "ip_adapter": "IP-Adapter",
}

# 资源类型 → 该放到哪个目录（缺失时告诉用户）
TYPE_DIRS = {
    "checkpoint": "models/（可放 models/puton/ 让程序自动归位）",
    "lora": "loras/（按架构分子目录：sd1.5 / sdxl）",
    "vae": "models/ 或 weights/",
}


def extension_id_for(res_type: str, architecture: str = "",
                     kind: str = "") -> str:
    """把"辅助组件类资源"映射到扩展市场的条目 id；底模/LoRA/VAE 返回 ""。

    返回空字符串 = **不提供下载入口**（遵守"模型由用户自行下载"的策略）。
    """
    t = (res_type or "").lower()
    is_xl = architecture in ("sdxl", "pony")
    if t == "controlnet":
        k = (kind or "openpose").lower()
        if k not in ("openpose", "canny", "depth"):
            return ""
        return f"controlnet_{k}_sdxl" if is_xl else f"controlnet_{k}"
    if t == "adetailer":
        return "adetailer_hand" if (kind or "") == "hand" else "adetailer_face"
    if t == "ip_adapter":
        return "ip_adapter_sd15"
    if t == "upscaler":
        return "rife_v46"
    if t == "engine":
        return "flux_engine_schnell"
    return ""


class _DownloadThread(QThread):
    """在后台下载辅助组件，避免界面卡死。"""

    progress = pyqtSignal(float, str)
    done = pyqtSignal(bool, str)

    def __init__(self, ext_id: str, parent=None):
        super().__init__(parent)
        self.ext_id = ext_id

    def run(self):
        try:
            from utils.extension_manager import download_extension
            download_extension(
                self.ext_id,
                lambda pct, msg: self.progress.emit(float(pct), str(msg)))
            self.done.emit(True, ui_i18n.trf("{extension} 下载完成", extension=self.ext_id))
        except Exception as e:
            self.done.emit(False, ui_i18n.trf(
                "{extension} 下载失败: {error}", extension=self.ext_id, error=e))


class ResourceDialog(QDialog):
    """工作流资源清单 + 缺失资源补全。"""

    def __init__(self, wf, parent=None):
        super().__init__(parent)
        self.wf = wf
        self.setWindowTitle(ui_i18n.tr("工作流资源"))
        self.resize(920, 520)
        self._report = {}
        self._rows: list = []
        self._thread = None

        lay = QVBoxLayout(self)
        self.lbl_summary = QLabel(ui_i18n.tr("正在检查…"))
        self.lbl_summary.setWordWrap(True)
        lay.addWidget(self.lbl_summary)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["名称", "类型", "架构", "状态", "大小", "说明"])
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3, 4):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)

        btns = QHBoxLayout()
        self.btn_recheck = QPushButton(" 重新检测")
        self.btn_recheck.clicked.connect(self.reload)
        btns.addWidget(self.btn_recheck)

        self.btn_download = QPushButton(" 下载选中的辅助组件")
        self.btn_download.setToolTip(
            "只对辅助组件（ControlNet / ADetailer / IP-Adapter / 插帧 / Flux 引擎）\n"
            "提供下载。底模、LoRA、VAE 请自行下载 —— 这是项目既定策略。")
        self.btn_download.clicked.connect(self.on_download)
        btns.addWidget(self.btn_download)

        self.btn_open_dir = QPushButton(" 打开模型目录")
        self.btn_open_dir.clicked.connect(self.on_open_dir)
        btns.addWidget(self.btn_open_dir)

        btns.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_close)
        lay.addLayout(btns)

        self.reload()

    # ------------------------------------------------------------------
    def reload(self):
        """重新解析工作流的资源依赖。"""
        try:
            from workflow import resource_store as RS
            RS.invalidate()
            refs = RS.collect_refs(self.wf)
            rep = RS.missing_report(refs)
        except Exception as e:
            logger.warning(f"资源检查失败: {e}")
            self.lbl_summary.setText(ui_i18n.trf("资源检查不可用：{error}", error=e))
            self.table.setRowCount(0)
            return

        self._report = rep
        rows = rep.get("present", []) + rep.get("mismatched", []) \
            + rep.get("missing", [])
        self._rows = rows
        self._fill_table(rows)

        total = rep.get("total", 0)
        miss = len(rep.get("missing", []))
        bad = len(rep.get("mismatched", []))
        dl = len(rep.get("downloadable", []))
        if not total:
            head = ui_i18n.tr("这个工作流没有引用任何本地模型")
        elif not miss and not bad:
            head = ui_i18n.trf("依赖 {total} 项资源，全部就绪", total=total)
        else:
            head = ui_i18n.trf(
                "依赖 {total} 项资源：缺失 {missing} 项，不匹配 {bad} 项",
                total=total, missing=miss, bad=bad)
            if dl:
                head += ui_i18n.trf("（其中 {count} 项可从扩展市场下载）", count=dl)
        self.lbl_summary.setText(head + "\n" + str(rep.get("summary", "")))

    def _fill_table(self, rows):
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            res = r.get("resource")
            ref = r.get("ref") or {}
            status = r.get("status", "unknown")
            label, color = STATUS_LABELS.get(status, STATUS_LABELS["unknown"])
            name = (getattr(res, "name", "") or ref.get("filename")
                    or ref.get("resource_id") or "?")
            rtype = getattr(res, "type", "") or ref.get("model_type") or "?"
            arch = (getattr(res, "architecture", "")
                    or ref.get("architecture") or "?")
            size = getattr(res, "size_bytes", 0) or 0
            size_txt = (f"{size / 1024 ** 3:.2f} GB" if size > 1024 ** 3
                        else (f"{size / 1024 ** 2:.0f} MB" if size else "—"))

            cells = [name, ui_i18n.tr(TYPE_LABELS.get(rtype, rtype)), arch,
                     ui_i18n.tr(label), size_txt, r.get("detail", "")]
            for c, txt in enumerate(cells):
                item = QTableWidgetItem(str(txt))
                if c == 3:
                    item.setForeground(Qt.GlobalColor.white)
                    item.setData(Qt.ItemDataRole.ForegroundRole, None)
                    from PyQt6.QtGui import QColor
                    item.setForeground(QColor(color))
                self.table.setItem(i, c, item)

    # ------------------------------------------------------------------
    def _selected_row(self) -> dict | None:
        i = self.table.currentRow()
        if i < 0 or i >= len(self._rows):
            return None
        return self._rows[i]

    def on_download(self):
        row = self._selected_row()
        if row is None:
            QMessageBox.information(
                self, ui_i18n.tr("未选中"), ui_i18n.tr("请先在列表里选一项。"))
            return
        ref = row.get("ref") or {}
        rtype = ref.get("model_type") or ""
        ext_id = extension_id_for(rtype, ref.get("architecture", ""),
                                  ref.get("kind", ""))
        if not ext_id:
            directory = ui_i18n.tr(TYPE_DIRS.get(rtype, "models/ 目录"))
            QMessageBox.information(
                self, ui_i18n.tr("需要自行下载"),
                ui_i18n.tr("底模 / LoRA / VAE 需要你自己下载（项目不提供这些下载入口）。")
                + "\n\n" + ui_i18n.trf("请把文件放到：{directory}", directory=directory)
                + "\n" + ui_i18n.tr("放好后点「重新检测」。"))
            return
        try:
            from utils.extension_manager import is_installed
            if is_installed(ext_id):
                QMessageBox.information(
                    self, ui_i18n.tr("已安装"),
                    ui_i18n.trf("{extension} 已经在本地了。", extension=ext_id))
                self.reload()
                return
        except Exception:
            pass
        self._start_download(ext_id)

    def _start_download(self, ext_id: str):
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(
                self, ui_i18n.tr("正在下载"), ui_i18n.tr("上一个下载还没结束。"))
            return
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self.btn_download.setEnabled(False)
        self._thread = _DownloadThread(ext_id, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.done.connect(self._on_download_done)
        self._thread.start()

    def _on_progress(self, pct: float, msg: str):
        self.bar.setValue(int(max(0.0, min(100.0, pct))))
        self.lbl_summary.setText(ui_i18n.tr(msg))

    def _on_download_done(self, ok: bool, msg: str):
        self.bar.setVisible(False)
        self.btn_download.setEnabled(True)
        self._thread = None
        self.lbl_summary.setText(ui_i18n.tr(msg))
        # 下载完 → 重新检测（用户要求的闭环：下载完成 → 重新预检）
        self.reload()

    def on_open_dir(self):
        row = self._selected_row()
        ref = (row or {}).get("ref") or {}
        rtype = ref.get("model_type") or "checkpoint"
        try:
            from utils import paths
            d = {"checkpoint": paths.MODEL_DIR, "vae": paths.WEIGHTS_DIR,
                 "lora": paths.LORA_DIR}.get(rtype, paths.MODEL_DIR)
            os.makedirs(d, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(d)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", d])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            QMessageBox.warning(self, ui_i18n.tr("无法打开目录"), str(e))

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(2000)
        super().closeEvent(event)
