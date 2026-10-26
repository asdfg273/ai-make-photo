# ui/extension_market.py
"""
🧩 扩展市场 GUI
- 分类折叠面板
- 每个扩展一张卡片:名称/描述/大小/状态/按钮
- 后台线程下载/卸载,不卡主界面
- 实时进度条 + 状态文本
"""
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QProgressBar, QFrame, QMessageBox,
)
import logging

logger = logging.getLogger(__name__)

from utils.extension_manager import (
    EXTENSIONS, get_status_summary,
    download_extension, uninstall_extension,
)
from ui.theme import PALETTE

# 语义状态色（从 PALETTE 派生，深浅同族）
_C_OK_BG, _C_OK_FG   = "#274a35", "#7fd69a"
_C_BAD_BG, _C_BAD_FG = "#5a2e35", PALETTE["danger"]
_C_DIM_BG, _C_DIM_FG = PALETTE["bg_soft"], PALETTE["fg_mute"]


# ============================================================
#  下载/卸载工作线程
# ============================================================
# 模块级注册表：worker **故意不挂到对话框上**。
# 用户关闭扩展市场后，下载要继续在后台跑（需求原话），
# 所以 worker 的生命周期必须独立于对话框；对话框关闭时只解除 UI 连接。
_ACTIVE_WORKERS: set = set()


def running_worker_count() -> int:
    """当前仍在运行的下载/卸载任务数（供主窗口退出前判断）。"""
    return sum(1 for w in list(_ACTIVE_WORKERS) if w.isRunning())


def shutdown_workers(timeout_ms: int = 8000) -> int:
    """退出程序前收尾：尽量等任务结束，超时则安全终止线程。

    必须做这一步的原因：QThread 在**运行中**被销毁时 Qt 会 qFatal
    直接终止进程（表现为"关掉程序后弹错/闪退"）。
    这里先给宽限期，再 terminate()+wait()，保证析构时线程已停止。
    终止是安全的：HF 下载器基于文件，已下载部分会保留，下次可续传。
    """
    import time as _t
    alive = [w for w in list(_ACTIVE_WORKERS) if w.isRunning()]
    if not alive:
        return 0
    logger.info("⏳ 等待 %d 个后台下载任务收尾（最多 %ds）…",
                len(alive), timeout_ms // 1000)
    deadline = _t.time() + timeout_ms / 1000.0
    while _t.time() < deadline:
        if not any(w.isRunning() for w in alive):
            logger.info("✅ 后台下载已自然结束")
            return 0
        _t.sleep(0.1)

    stopped = 0
    for w in alive:
        if w.isRunning():
            try:
                w.terminate()
                w.wait(2000)
                stopped += 1
            except Exception as e:
                logger.warning(f"⚠️ 终止下载线程失败: {e}")
    logger.warning("⚠️ 已强制停止 %d 个未完任务（已下载部分保留，下次可续传）", stopped)
    return stopped


class ExtensionWorker(QThread):
    progress = pyqtSignal(float, str)   # pct, msg
    finished_ok = pyqtSignal(str)       # ext_id
    failed = pyqtSignal(str, str)       # ext_id, error

    def __init__(self, ext_id: str, action: str, parent=None):
        super().__init__(parent)
        self.ext_id = ext_id
        self.action = action  # "install" or "uninstall"

    def run(self):
        try:
            def cb(pct, msg):
                self.progress.emit(float(pct), str(msg))

            if self.action == "install":
                download_extension(self.ext_id, cb)
            else:
                uninstall_extension(self.ext_id, cb)
            self.finished_ok.emit(self.ext_id)
        except Exception as e:
            self.failed.emit(self.ext_id, str(e))


# ============================================================
#  单个扩展卡片
# ============================================================
class ExtensionCard(QFrame):
    action_requested = pyqtSignal(str, str)  # ext_id, action

    def __init__(self, ext_id: str, ext_info: dict, installed: bool):
        super().__init__()
        self.ext_id = ext_id
        self.ext_info = ext_info
        self.installed = installed
        self.busy = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background: {PALETTE['bg_soft']};
                border-radius: 8px;
                padding: 10px;
            }}
            QLabel {{ background: transparent; color: {PALETTE['fg']}; }}
        """)

        lay = QVBoxLayout(self)
        lay.setSpacing(6)

        # 第一行: 名称 + 大小 + 状态
        row1 = QHBoxLayout()
        req_tag = " [必需]" if ext_info.get("required") else ""
        self.lbl_name = QLabel(f"<b>{ext_info['name']}</b>{req_tag}")
        self.lbl_name.setProperty("role", "title")
        row1.addWidget(self.lbl_name)
        row1.addStretch()

        self.lbl_size = QLabel(f"{ext_info['size_mb']} MB")
        self.lbl_size.setProperty("role", "hint")
        row1.addWidget(self.lbl_size)

        self.lbl_status = QLabel()
        self.lbl_status.setStyleSheet("font-size:12px; padding:2px 8px; border-radius:4px;")
        row1.addWidget(self.lbl_status)
        lay.addLayout(row1)

        # 第二行: 描述
        self.lbl_desc = QLabel(ext_info.get("desc", ""))
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setProperty("role", "hint")
        lay.addWidget(self.lbl_desc)

        # 第三行: 进度条 (默认隐藏)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(18)
        lay.addWidget(self.progress)

        self.lbl_progress_msg = QLabel("")
        self.lbl_progress_msg.setProperty("role", "hint")
        self.lbl_progress_msg.setVisible(False)
        lay.addWidget(self.lbl_progress_msg)

        # 第四行: 按钮
        row_btn = QHBoxLayout()
        row_btn.addStretch()
        self.btn_action = QPushButton()
        self.btn_action.setFixedWidth(110)
        self.btn_action.clicked.connect(self._on_click)
        row_btn.addWidget(self.btn_action)
        lay.addLayout(row_btn)

        self._refresh_ui()

    def _refresh_ui(self):
        from ui.i18n import ui_i18n
        if self.installed:
            self.lbl_status.setText(ui_i18n.tr("已安装"))
            self.lbl_status.setStyleSheet(
                f"background:{_C_OK_BG}; color:{_C_OK_FG}; font-size:12px; padding:2px 8px; border-radius:4px;"
            )
            self.btn_action.setText(ui_i18n.tr("卸载"))
            self.btn_action.setStyleSheet(f"""
                QPushButton {{
                    background:{_C_BAD_BG}; color:{_C_BAD_FG}; border:none;
                    padding:6px 12px; border-radius:4px; font-weight:bold;
                }}
                QPushButton:hover {{ background:#6a3a42; }}
            """)
        else:
            required = self.ext_info.get("required", False)
            if required:
                self.lbl_status.setText(ui_i18n.tr("缺失（必需）"))
                self.lbl_status.setStyleSheet(
                    f"background:{_C_BAD_BG}; color:{_C_BAD_FG}; font-size:12px; padding:2px 8px; border-radius:4px;"
                )
            else:
                self.lbl_status.setText(ui_i18n.tr("未安装"))
                self.lbl_status.setStyleSheet(
                    f"background:{_C_DIM_BG}; color:{_C_DIM_FG}; font-size:12px; padding:2px 8px; border-radius:4px;"
                )
            self.btn_action.setText(ui_i18n.tr("下载"))
            self.btn_action.setStyleSheet(f"""
                QPushButton {{
                    background:{PALETTE['accent']}; color:#0d1620; border:none;
                    padding:6px 12px; border-radius:4px; font-weight:bold;
                }}
                QPushButton:hover {{ background:{PALETTE['accent_hi']}; }}
            """)

    def _on_click(self):
        if self.busy:
            return
        action = "uninstall" if self.installed else "install"

        # 卸载必需项二次确认
        if action == "uninstall" and self.ext_info.get("required"):
            reply = QMessageBox.warning(
                self, "确认卸载",
                f"「{self.ext_info['name']}」是必需扩展,卸载后相关功能将无法使用!\n\n确定继续?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        elif action == "uninstall":
            reply = QMessageBox.question(
                self, "确认卸载",
                f"确定卸载「{self.ext_info['name']}」吗?\n(可随时重新下载)",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.action_requested.emit(self.ext_id, action)

    def set_busy(self, busy: bool):
        self.busy = busy
        self.btn_action.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.lbl_progress_msg.setVisible(busy)
        if not busy:
            self.progress.setValue(0)
            self.lbl_progress_msg.setText("")

    def update_progress(self, pct: float, msg: str):
        self.progress.setValue(int(pct))
        # 截断长消息
        if len(msg) > 80:
            msg = msg[:77] + "..."
        self.lbl_progress_msg.setText(msg)

    def refresh_installed(self, installed: bool):
        self.installed = installed
        self._refresh_ui()


# ============================================================
#  扩展市场主对话框
# ============================================================
class ExtensionMarketDialog(QDialog):
    def retranslate_dynamic(self):
        """刷新安装状态和汇总数字，保留当前界面语言。"""
        for card in self.cards.values():
            card._refresh_ui()
        self._update_summary()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🧩 扩展市场")
        self.resize(880, 720)

        self.cards = {}       # ext_id -> ExtensionCard
        self.workers = {}     # ext_id -> ExtensionWorker

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # 顶部标题
        title = QLabel("🧩 扩展市场")
        title.setStyleSheet("font-size:20px; font-weight:bold;")
        root.addWidget(title)

        hint = QLabel(
            "所有扩展均为可选下载。带 <b>[必需]</b> 标记的是核心功能所需模型。<br>"
            "下载使用 <b>hf-mirror.com</b> 国内镜像,失败自动回退官方源。"
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # 全局状态栏
        self.lbl_summary = QLabel()
        self.lbl_summary.setStyleSheet(
            f"background:{PALETTE['bg_soft']}; color:{PALETTE['fg']}; padding:8px; border-radius:6px; font-size:13px;"
        )
        root.addWidget(self.lbl_summary)

        # 滚动区
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.content_lay = QVBoxLayout(content)
        self.content_lay.setSpacing(12)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # 底部按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_refresh = QPushButton("🔄 刷新状态")
        btn_refresh.clicked.connect(self.refresh_all)
        btn_refresh.setStyleSheet("padding:8px 16px;")
        btn_row.addWidget(btn_refresh)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn_close.setStyleSheet("padding:8px 16px;")
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        self._build_cards()
        self._update_summary()
        from ui.i18n import ui_i18n
        ui_i18n.language_changed.connect(self.retranslate_dynamic)
        ui_i18n.translate_widget_tree(self)

    def _build_cards(self):
        """按分类构建卡片"""
        from utils.extension_manager import is_installed

        # 按分类分组
        by_cat = {}
        for ext_id, ext in EXTENSIONS.items():
            by_cat.setdefault(ext["category"], []).append((ext_id, ext))

        # 分类顺序
        cat_order = ["图片增强", "控制", "修复", "视频", "音频", "新架构引擎"]
        ordered_cats = [c for c in cat_order if c in by_cat] + [c for c in by_cat if c not in cat_order]

        for cat in ordered_cats:
            # 分类标题
            cat_lbl = QLabel(f"📂 {cat}")
            cat_lbl.setStyleSheet(
                f"font-size:15px; font-weight:bold; color:{PALETTE['accent']}; padding:6px 0 2px 0;"
            )
            self.content_lay.addWidget(cat_lbl)

            for ext_id, ext in by_cat[cat]:
                installed = is_installed(ext_id)  # 🔧 改这里
                card = ExtensionCard(ext_id, ext, installed)
                card.action_requested.connect(self._on_action)
                self.cards[ext_id] = card
                self.content_lay.addWidget(card)

        self.content_lay.addStretch()

    def _set_status(self, text: str, color: str = "#89dceb"):
        """在汇总标签上显示一句话状态（没有独立状态栏，复用顶部汇总行）。"""
        try:
            if getattr(self, "_orig_summary_text", None) is None:
                self._orig_summary_text = self.lbl_summary.text()
            self.lbl_summary.setText(text)
            self.lbl_summary.setStyleSheet(f"color:{color}; font-weight:bold;")
        except Exception:
            pass

    def _update_summary(self):
        """更新顶部汇总"""
        from ui.i18n import ui_i18n
        summary = get_status_summary()
        installed = summary["installed"]
        total = summary["total"]
        label = ui_i18n.tr("已安装:")
        self.lbl_summary.setText(f"{label} {installed} / {total}")
        self.lbl_summary.setStyleSheet("")


    def refresh_all(self):
        """重新扫描所有扩展状态"""
        from utils.extension_manager import is_installed
        for ext_id, card in self.cards.items():
            card.refresh_installed(is_installed(ext_id))
        self._update_summary()

    def _on_action(self, ext_id: str, action: str):
        """处理下载/卸载请求"""
        if ext_id in self.workers and self.workers[ext_id].isRunning():
            return

        card = self.cards[ext_id]
        card.set_busy(True)

        # 不传 parent：让 worker 独立于对话框存活，关窗后继续下载
        worker = ExtensionWorker(ext_id, action)
        worker.progress.connect(lambda p, m, c=card: c.update_progress(p, m))
        worker.finished_ok.connect(lambda eid: self._on_worker_done(eid, True))
        worker.failed.connect(lambda eid, err: self._on_worker_done(eid, False, err))
        worker.finished.connect(lambda eid=ext_id: self._cleanup_worker(eid))
        self.workers[ext_id] = worker
        _ACTIVE_WORKERS.add(worker)
        worker.start()

    def _cleanup_worker(self, ext_id: str):
        w = self.workers.pop(ext_id, None)
        if w is not None:
            _ACTIVE_WORKERS.discard(w)
            w.deleteLater()

    def _on_worker_done(self, ext_id: str, success: bool, error: str = ""):
        card = self.cards.get(ext_id)
        if card:
            card.set_busy(False)
            # 重新检测安装状态
            from utils.extension_manager import is_installed
            card.refresh_installed(is_installed(ext_id))
        self._update_summary()

        if not success:
            QMessageBox.critical(self, "操作失败", f"[{ext_id}]\n\n{error}")

    def closeEvent(self, event):
        """关闭窗口 = 收工，但下载继续在后台跑。

        线程生命周期不依赖本对话框（worker 无 parent、且登记在
        _ACTIVE_WORKERS），所以关闭是安全的；真正需要收尾的是
        退出程序时，由 shutdown_workers() 统一处理。
        """
        running = [wid for wid, w in self.workers.items() if w.isRunning()]
        if running:
            logger.info("⏳ 关闭扩展市场：%d 个任务转入后台继续", len(running))
            try:
                self._set_status(
                    f"⏳ {len(running)} 个下载任务已在后台继续，完成后会自动安装",
                    "#89dceb")
            except Exception:
                pass
        event.accept()


# ============================================================
#  独立测试
# ============================================================
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dlg = ExtensionMarketDialog()
    dlg.exec()
