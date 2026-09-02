# ui/shell.py
# ============================================================
#  新主窗口外壳 v6.0 — NavRail + 中央Stacked + 右侧参数面板
#  生成核心控件全局单例；页面专属区随导航切换
# ============================================================
import logging
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
                             QStackedWidget, QProgressBar, QLabel, QStatusBar,
                             QScrollArea)
from PyQt6.QtCore import Qt

from ui.theme import PALETTE
from ui.nav import NavRail
from ui.pages import PAGES
from ui.contracts import install_aliases, check_contract, apply_degradation

logger = logging.getLogger(__name__)


class ShellMixin:
    """替代 UIBuilderMixin 的 setup_ui 实现。业务 mixin 零改动。"""

    def setup_ui(self):
        self.setMinimumSize(1320, 820)
        self.setWindowTitle("AI 绘画工作站 v6.0")
        # 主题由 main.py 的 apply_theme(QApplication) 统一应用

        from ui.core_panel import build_status_widgets
        build_status_widgets(self)   # lbl_status / progress_gen 先建，状态栏复用

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── 主体三栏 ──
        body = QHBoxLayout()
        body.setSpacing(6)

        self.nav = NavRail()
        body.addWidget(self.nav)

        self.center_stack = QStackedWidget()
        body.addWidget(self.center_stack, 1)

        # 右侧参数面板：核心区(常驻) + 专属区(切换) + 折叠分组 + 生成按钮
        self.params_panel = self._build_params_panel()
        body.addWidget(self.params_panel)

        root.addLayout(body, 1)

        # ── 底部胶片条（Task 11 接入真实组件，先占位）──
        self.filmstrip = QLabel("胶片条占位")
        self.filmstrip.setFixedHeight(110)
        root.addWidget(self.filmstrip)

        # ── 页面注册与切换 ──
        self._pages: dict = {}
        self._ws_added: set = set()
        self.nav.set_pages(PAGES)
        for cls in PAGES:
            page = cls()
            try:
                page.build(self)              # 页面专属控件挂到 self
            except Exception:
                logger.exception(f"❌ 页面构建失败: {cls.page_id}")
                self.center_stack.addWidget(QLabel(f"页面 {cls.title} 加载失败"))
                continue
            self._pages[cls.page_id] = page
            ws = page.workspace()
            if ws not in self._ws_added:      # 多页可共享同一中央工作区实例
                self._ws_added.add(ws)
                self.center_stack.addWidget(ws)
            pw = page.params_widget()
            if pw is not None:
                self.params_stack.addWidget(pw)
        self.nav.page_selected.connect(self._on_page_selected)
        if PAGES:
            self.nav.select(PAGES[0].page_id)

        # ── 契约自检 + 分级降级 ──
        install_aliases(self)
        crit, minor = check_contract(self)
        if minor:
            logger.warning(f"⚠️ 契约自检（非关键缺失）: {minor}")
        apply_degradation(self, crit)

        self._build_statusbar_v6()

    # ---------- 页面切换 ----------
    def _on_page_selected(self, page_id: str):
        page = self._pages.get(page_id)
        if page is None:
            return
        self.center_stack.setCurrentWidget(page.workspace())
        pw = page.params_widget()
        if pw is not None:
            self.params_stack.setCurrentWidget(pw)
        # 只切换/隐藏"页面专属区"；生成核心区与生成按钮永远常驻
        self.params_scroll.setVisible(pw is not None)

    # ---------- 右侧面板（本任务先骨架，Task 6 填核心区）----------
    def _build_params_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(360)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.core_area = QVBoxLayout()       # 生成核心区（全局单例）
        from ui.core_panel import build_core, build_gen_area
        build_core(self, self.core_area)
        lay.addLayout(self.core_area)
        self.params_stack = QStackedWidget() # 页面专属区
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setWidget(self.params_stack)
        lay.addWidget(self.params_scroll, 1)
        self.shared_groups = QVBoxLayout()   # Task 9 填充共享折叠分组
        lay.addLayout(self.shared_groups)
        self.gen_area = QVBoxLayout()        # 生成/停止按钮
        build_gen_area(self, self.gen_area)
        lay.addLayout(self.gen_area)
        return w

    # ---------- 状态栏 ----------
    def _build_statusbar_v6(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_status = getattr(self, "lbl_status", None) or QLabel("就绪")
        self.progress_gen = getattr(self, "progress_gen", None) or QProgressBar()
        self.progress_gen.setMaximumWidth(220)
        sb.addWidget(self.lbl_status, 1)
        sb.addPermanentWidget(self.progress_gen)
        sb.showMessage("AI 绘画工作站 v6.0 已就绪")

    # ---------- 方法契约 ----------
    def append_log(self, text: str, color: str = "#dfe5ec"):
        """从旧 append_log 迁入：写图片/动画日志框，带颜色，自动滚底。"""
        html = (
            f'<span style="color:{color}; font-family:Consolas;">'
            f'{text}</span>'
        )
        wrote = False
        for attr in ("txt_log_image", "txt_log_video"):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.append(html)
                sb = widget.verticalScrollBar()
                sb.setValue(sb.maximum())
                wrote = True
        if not wrote:
            logger.info(text)

    def set_status(self, text: str, color: str = "#dfe5ec"):
        if getattr(self, "lbl_status", None) is not None:
            self.lbl_status.setText(text)

    def set_progress(self, value: int):
        if getattr(self, "progress_gen", None) is not None:
            self.progress_gen.setValue(value)

    def play_video(self, video_path: str):
        logger.info(f"play_video 占位: {video_path}")  # Task 10 实现
