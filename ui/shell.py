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

        # ── 底部胶片条（真实组件，联动画廊 items_changed）──
        from ui.components.filmstrip import FilmStrip
        self.filmstrip = FilmStrip()
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

        # ── 画廊 ↔ 胶片条联动（gallery 由画廊页创建，此处已存在）──
        if getattr(self, "gallery", None) is not None:
            self.filmstrip._gallery = self.gallery   # 右键菜单动作委托
            self.gallery.items_changed.connect(self._refresh_filmstrip)
            self.gallery.video_selected.connect(self._on_gallery_video_picked)
            self.filmstrip.media_clicked.connect(self._on_filmstrip_clicked)
            self._refresh_filmstrip()

        # ── 契约自检 + 分级降级 ──
        install_aliases(self)
        crit, minor = check_contract(self)
        if minor:
            logger.warning(f"⚠️ 契约自检（非关键缺失）: {minor}")
        apply_degradation(self, crit)

        self._build_statusbar_v6()
        self._build_menu_v6()
        self._init_defaults()
        self._setup_shortcuts()

    # ---------- 全局快捷键 ----------
    def _setup_shortcuts(self):
        from PyQt6.QtGui import QShortcut, QKeySequence
        self._shortcuts = []   # 驻留防 GC

        def _add(seq, fn):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

        _add("Ctrl+Return", lambda: getattr(self, "btn_generate", None)
             and self.btn_generate.isEnabled() and self.btn_generate.click())
        _add("Escape", lambda: getattr(self, "btn_interrupt", None)
             and self.btn_interrupt.isEnabled() and self.btn_interrupt.click())
        # Ctrl+1..4 切换页面
        for i, cls in enumerate(PAGES, 1):
            _add(f"Ctrl+{i}",
                 lambda pid=cls.page_id: self.nav.select(pid))

    # ---------- 页面切换 ----------
    def _on_page_selected(self, page_id: str):
        page = self._pages.get(page_id)
        if page is None:
            return
        # 离开动画页时停止播放；进入时刷新视频历史
        prev = getattr(self, "_current_page_id", None)
        if prev == "video" and page_id != "video" and hasattr(self, "stop_video"):
            try:
                self.stop_video()
            except Exception:
                pass
        self._current_page_id = page_id
        self.center_stack.setCurrentWidget(page.workspace())

        # 画廊页：隐藏整个右侧面板，画廊全宽
        self.params_panel.setVisible(page_id != "gallery")

        # 动画页：图片核心区与图片生成按钮不适用（它有专属参数和生成按钮）
        is_video = (page_id == "video")
        self.core_wrap.setVisible(not is_video)
        self.gen_wrap.setVisible(not is_video)

        # 只切换/隐藏"页面专属区"；生成核心区与生成按钮永远常驻
        pw = page.params_widget()
        if pw is not None:
            self.params_stack.setCurrentWidget(pw)
            pw.adjustSize()
            # 按当前页内容撑高 stack，外层滚动区负责滚动，互不挤压
            self.params_stack.setMinimumHeight(max(pw.sizeHint().height(), 80))
        else:
            self.params_stack.setMinimumHeight(0)
        self.params_stack.setVisible(pw is not None)
        # 共享折叠分组是图片专属：动画页隐藏
        for sec in getattr(self, "_group_sections", {}).values():
            sec.setVisible(not is_video)
        if is_video and hasattr(self, "_refresh_video_gallery"):
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self._refresh_video_gallery)
        # 进入画廊页：重新扫描输出目录（图片 + videos 子目录）
        if page_id == "gallery" and getattr(self, "gallery", None) is not None:
            from PyQt6.QtCore import QTimer
            def _reload():
                try:
                    from utils.paths import OUTPUT_DIR
                    self.gallery.reload_from_dir(OUTPUT_DIR, limit=80)
                except Exception as e:
                    logger.warning(f"画廊刷新失败: {e}")
            QTimer.singleShot(100, _reload)

    # ---------- 胶片条联动 ----------
    def _refresh_filmstrip(self):
        gallery = getattr(self, "gallery", None)
        if gallery is None or getattr(self, "filmstrip", None) is None:
            return
        # 跟随画廊当前过滤后的可见项（全部/图片/动画 + 搜索 + 收藏过滤）
        paths = []
        for i in range(gallery.list_widget.count()):
            p = gallery.list_widget.item(i).data(Qt.ItemDataRole.UserRole)
            if p:
                paths.append(p)
            if len(paths) >= 24:
                break
        self.filmstrip.refresh(paths)

    def _on_filmstrip_clicked(self, path: str):
        """点击胶片条 → 跳画廊页选中对应项，并直接复用该图参数。"""
        self.nav.select("gallery")
        gallery = getattr(self, "gallery", None)
        if gallery is None:
            return
        for i in range(gallery.list_widget.count()):
            item = gallery.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                gallery.list_widget.setCurrentItem(item)
                gallery.list_widget.scrollToItem(item)
                break
        # 图片直接复用参数回填生成区；视频无参数可复用
        from ui.gallery_panel import GalleryPanel
        if GalleryPanel.media_kind(path) == "image" \
                and hasattr(self, "reuse_params_from_path"):
            try:
                self.reuse_params_from_path(path)
            except Exception as e:
                logger.warning(f"胶片条复用参数失败: {e}")

    def _on_gallery_video_picked(self, path: str):
        """画廊双击视频 → 跳动画页播放。"""
        self.nav.select("video")
        self.play_video(path)

    # ---------- 右侧面板（整体可滚动，生成按钮固定底部）----------
    def _build_params_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(400)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # 单一滚动区：核心区 + 页面专属区 + 共享折叠分组，内容再多也不挤压
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        clay = QVBoxLayout(content)
        clay.setContentsMargins(4, 4, 4, 4)
        clay.setSpacing(8)

        # 生成核心区（全局单例，包一层容器便于整区显隐）
        self.core_wrap = QWidget()
        self.core_area = QVBoxLayout(self.core_wrap)
        self.core_area.setContentsMargins(0, 0, 0, 0)
        from ui.core_panel import build_core, build_gen_area
        build_core(self, self.core_area)
        clay.addWidget(self.core_wrap)

        self.params_stack = QStackedWidget() # 页面专属区
        clay.addWidget(self.params_stack)

        self.shared_groups = QVBoxLayout()   # 共享折叠分组（LoRA/CN/高级/X-Y）
        from ui.shared_groups import build_shared_groups
        self._group_sections = build_shared_groups(self, self.shared_groups)
        clay.addLayout(self.shared_groups)
        clay.addStretch()

        self.params_scroll.setWidget(content)
        lay.addWidget(self.params_scroll, 1)

        self.gen_wrap = QWidget()            # 生成/停止按钮（固定底部）
        self.gen_area = QVBoxLayout(self.gen_wrap)
        self.gen_area.setContentsMargins(0, 0, 0, 0)
        build_gen_area(self, self.gen_area)
        lay.addWidget(self.gen_wrap)
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
        self._setup_resource_monitor()

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
        """播放指定路径的视频（从旧 UIBuilderMixin 迁入）"""
        import os as _os
        from PyQt6.QtCore import QUrl
        _status = getattr(self, "_set_status", None) or \
            (lambda msg, color=None: logger.info(msg))
        logger.info(f"🎥 尝试播放视频: {video_path}")
        if not _os.path.exists(video_path):
            _status(f"⚠️ 视频文件不存在: {video_path}", "#e06c75")
            return
        try:
            self.video_player.stop()
            self.video_player.setSource(QUrl.fromLocalFile(video_path))
            if hasattr(self, 'video_stacked'):
                self.video_stacked.setCurrentIndex(1)
            else:
                self.lbl_video_placeholder.hide()
                self.video_widget.show()
            self.video_player.play()
            _status(f"🎥 正在播放: {_os.path.basename(video_path)}", "#dfe5ec")
            self.current_video_path = video_path
        except Exception as e:
            import traceback
            _status(f"⚠️ 视频播放失败: {e}", "#e06c75")
            logger.error(f"❌ 视频播放失败: {e}")
            logger.error(traceback.format_exc())
            if hasattr(self, 'video_stacked'):
                self.video_stacked.setCurrentIndex(0)

    # ============================================================
    #  以下方法从旧 UIBuilderMixin 迁入（业务层/菜单依赖）
    # ============================================================

    # ---------- 默认值 + 控件联动 ----------
    def _init_defaults(self):
        if hasattr(self, 'refresh_models'):
            try:
                self.refresh_models()
            except Exception as e:
                logger.warning(f"refresh_models 失败: {e}")
        self._toggle_adetailer()
        self._toggle_ad_hand()
        self._toggle_hires()
        self._toggle_xy()
        self._toggle_cn()

    def _toggle_adetailer(self):
        if not hasattr(self, 'chk_use_adetailer'):
            return
        on = self.chk_use_adetailer.isChecked()
        for c in (self.combo_ad_target, self.combo_adetailer_model,
                  self.scale_adetailer_strength):
            c.setEnabled(on)

    def _toggle_ad_hand(self):
        if not hasattr(self, 'chk_use_ad_hand'):
            return
        on = self.chk_use_ad_hand.isChecked()
        for c in (self.combo_ad_hand, self.scale_ad_hand,
                  self.scale_ad_hand_blend):
            c.setEnabled(on)

    def _toggle_hires(self):
        if not hasattr(self, 'chk_hires'):
            return
        on = self.chk_hires.isChecked()
        for c in (self.combo_hires_scale, self.scale_hires_denoise,
                  self.combo_hires_upscaler):
            c.setEnabled(on)

    def _toggle_xy(self):
        if not hasattr(self, 'chk_enable_xy'):
            return
        on = self.chk_enable_xy.isChecked()
        for w in (self.combo_x_type, self.entry_x_vals,
                  self.combo_y_type, self.entry_y_vals):
            w.setEnabled(on)

    def _toggle_cn(self):
        if not hasattr(self, 'chk_use_pose'):
            return
        on = self.chk_use_pose.isChecked()
        for c in (self.combo_cn_type, self.scale_cn_strength,
                  self.btn_load_cn_img):
            c.setEnabled(on)

    def _on_pose_transfer_toggled(self, checked: bool):
        """Pose Transfer 开关切换 → 自动联动其他控件"""
        if checked:
            if hasattr(self, 'combo_cn_type'):
                idx = self.combo_cn_type.findText("OpenPose")
                if idx >= 0:
                    self.combo_cn_type.setCurrentIndex(idx)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Pose Transfer 已启用",
                "✅ 工作流程:\n\n"
                "1️⃣ AI 用提示词生成动作参考图\n"
                "2️⃣ 自动提取 OpenPose 骨架\n"
                "3️⃣ 骨架 + IP-Adapter 角色图 → 最终图\n\n"
                "⚠️ 请确保已上传【IP-Adapter 角色参考图】\n"
                "💡 推荐: 影响力 0.6~0.8")

    def _on_ai_model_changed(self, index: int = 0):
        """AI 模型档位切换 - 立即生效，无需重启"""
        if not getattr(self, "_ui_ready", False):
            return
        key = self.combo_ai_model.currentData()
        if not key:
            return
        from utils.prompt_enhancer import PromptEnhancer
        PromptEnhancer().set_model_key(key)
        self.config.qwen_model_key = key
        self.config.save()
        logger.info(f"🎚️ AI 模型档位 → {self.combo_ai_model.currentText()}")

    # ---------- 修前/修后对比 ----------
    def _on_prefix_image(self, path: str):
        """生成管线发来的修前快照（Hires/ADetailer 前的阶段 1 图）。"""
        self._prefix_image_path = path
        btn = getattr(self, "btn_compare", None)
        if btn is not None:
            btn.setEnabled(True)
            btn.setToolTip(f"修前 / 修后 对比滑条\n修前: {path}")

    def _on_compare_toggled(self, checked: bool):
        stack = getattr(self, "preview_stack", None)
        if stack is None:
            return
        if checked:
            prefix = getattr(self, "_prefix_image_path", None)
            result = getattr(self, "last_generated_path", None) \
                or getattr(self, "current_generated_path", None)
            from PyQt6.QtGui import QPixmap
            import os as _os
            if prefix and result and _os.path.exists(prefix) and _os.path.exists(result):
                self.compare_canvas.set_images(QPixmap(prefix), QPixmap(result))
                stack.setCurrentIndex(1)
            else:
                self.set_status("⚠️ 没有可对比的修前快照", "#ff7a17")
                btn = getattr(self, "btn_compare", None)
                if btn is not None:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
        else:
            stack.setCurrentIndex(0)

    # ---------- 显存/内存常驻监控 ----------
    def _setup_resource_monitor(self):
        from PyQt6.QtCore import QTimer
        self.lbl_resource = QLabel("💾 --")
        self.lbl_resource.setProperty("role", "hint")
        self.statusBar().addPermanentWidget(self.lbl_resource)
        self._resource_timer = QTimer(self)
        self._resource_timer.setInterval(2000)
        self._resource_timer.timeout.connect(self._update_resource_label)
        self._resource_timer.start()
        self._update_resource_label()

    def _update_resource_label(self):
        try:
            import psutil
            ram = psutil.Process().memory_info().rss / 1024**3
            vram = ""
            try:
                import torch
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    vram = f" | 显存 {used:.1f}/{total:.1f}GB"
            except Exception:
                pass
            self.lbl_resource.setText(f"💾 内存 {ram:.1f}GB{vram}")
        except Exception:
            self.lbl_resource.setText("💾 --")

    # ---------- 画廊回调 ----------
    def _on_gallery_picked(self, path: str):
        from PyQt6.QtGui import QPixmap
        # 视频条目跳过图片预览（Image.open 会抛异常被吞，白读一次文件）
        from ui.gallery_panel import GalleryPanel
        if GalleryPanel.media_kind(path) == "video":
            return
        if hasattr(self, 'show_preview'):
            self.show_preview(path)
        else:
            try:
                pix = QPixmap(path)
                if not pix.isNull():
                    self.lbl_preview.set_pixmap(pix)
            except Exception:
                pass
        self.last_generated_path = path
        if hasattr(self, 'btn_edit'):
            self.btn_edit.setEnabled(True)
        if hasattr(self, 'btn_upscale'):
            self.btn_upscale.setEnabled(True)

    # ---------- LoRA 触发词 ----------
    def _insert_lora_triggers(self, slot_idx=None):
        """插入 LoRA 触发词到提示词框。slot_idx=None 插入全部槽。"""
        import os
        triggers_list = []

        for i, combo in enumerate(self.combo_loras):
            if slot_idx is not None and i != slot_idx:
                continue
            lora_name = combo.currentText().strip()
            if not lora_name or lora_name in ("无", "None", ""):
                continue
            if "[" in lora_name:
                lora_name = lora_name.split("[")[0].strip()
            base = os.path.splitext(lora_name)[0]
            for sub in ["sdxl", "sd1.5", "sd15", ""]:
                txt_path = os.path.join("loras", sub, base + ".txt") if sub \
                    else os.path.join("loras", base + ".txt")
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                triggers_list.append(content)
                                break
                    except Exception as e:
                        logger.warning(f"⚠️ 读取 {txt_path} 失败: {e}")

        if not triggers_list:
            self._set_status("⚠️ 没有可插入的触发词", "#ff7a17")
            return

        all_triggers = ", ".join(triggers_list)
        cur = self.txt_prompt.toPlainText().strip()
        new_text = f"{all_triggers}, {cur}" if cur else all_triggers
        self.txt_prompt.setPlainText(new_text)
        self._set_status(f"✅ 已插入 {len(triggers_list)} 组触发词", "#dadbdf")

    # ---------- 菜单 ----------
    def _build_menu_v6(self):
        from PyQt6.QtGui import QAction
        mb = self.menuBar()

        m_file = mb.addMenu("📁 文件")
        a_open = QAction("加载图片", self)
        if hasattr(self, "select_image"):
            a_open.triggered.connect(self.select_image)
        m_file.addAction(a_open)
        a_out = QAction("打开输出目录", self)
        a_out.triggered.connect(self._open_output_folder)
        m_file.addAction(a_out)
        m_file.addSeparator()
        a_quit = QAction("退出", self)
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_tool = mb.addMenu("🔧 工具")
        a_clear_log = QAction("清空日志", self)
        a_clear_log.triggered.connect(lambda: (
            getattr(self, 'txt_log_image', None) and self.txt_log_image.clear(),
            getattr(self, 'txt_log_video', None) and self.txt_log_video.clear()))
        m_tool.addAction(a_clear_log)
        m_tool.addSeparator()
        act_market = QAction("🛒 扩展市场...", self)
        act_market.setShortcut("Ctrl+E")
        act_market.triggered.connect(self._open_extension_market)
        m_tool.addAction(act_market)
        act_refresh = QAction("🔄 刷新扩展状态", self)
        act_refresh.triggered.connect(self._refresh_extension_count)
        m_tool.addAction(act_refresh)

        m_memory = mb.addMenu("🧹 内存")
        a_release = QAction("释放内存", self)
        a_release.triggered.connect(self.on_unload_models)
        m_memory.addAction(a_release)
        a_show = QAction("查看当前内存", self)
        a_show.triggered.connect(self._show_memory)
        m_memory.addAction(a_show)

        m_about = mb.addMenu("❓ 关于")
        a_about = QAction("关于本软件", self)
        a_about.triggered.connect(self._show_about)
        m_about.addAction(a_about)

        # 状态栏扩展计数
        self.lbl_ext_count = QLabel()
        self.statusBar().addPermanentWidget(self.lbl_ext_count)
        self._refresh_extension_count()

    def _refresh_extension_count(self):
        try:
            from utils.extension_manager import get_status_summary
            s = get_status_summary()
            self.lbl_ext_count.setText(f"🧩 扩展: {s['installed']}/{s['total']}")
        except Exception as e:
            self.lbl_ext_count.setText("🧩 扩展: --")
            logger.warning(f"[EXT-COUNT] 刷新失败: {e}")

    def _open_extension_market(self):
        try:
            from ui.extension_market import ExtensionMarketDialog
            dlg = ExtensionMarketDialog(self)
            dlg.exec()
            self._refresh_extension_count()
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "错误", f"打开扩展市场失败:\n{e}")

    def _open_output_folder(self):
        import subprocess, sys, os
        from utils.app_utils import OUTPUT_DIR
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile(os.path.abspath(OUTPUT_DIR))
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', OUTPUT_DIR])
            else:
                subprocess.Popen(['xdg-open', OUTPUT_DIR])
        except Exception as e:
            logger.warning(f"打开目录失败: {e}")

    def _show_about(self):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(
            self, "关于",
            "<b>AI 绘画工作站 v6.0</b><br>"
            "PyQt6 重构版 — GPU 加速<br><br>"
            "基于 Stable Diffusion + ADetailer<br>"
            "支持 LoRA / ControlNet / Hires.fix / IP-Adapter / Pose Transfer")

    def on_unload_models(self):
        from PyQt6.QtWidgets import QMessageBox
        if getattr(self, 'is_generating', False):
            QMessageBox.warning(self, "提示", "请先停止当前生成任务")
            return
        try:
            self._set_status("🧹 正在释放模型...", "#ff7a17")
            if hasattr(self, 'ai'):
                self.ai.unload_all()
            try:
                import psutil
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                self._set_status(
                    f"✅ 模型已释放 (当前内存 {mem:.0f} MB)", "#dadbdf")
            except ImportError:
                self._set_status("✅ 模型已释放", "#dadbdf")
        except Exception as e:
            QMessageBox.critical(self, "释放失败", str(e))

    def _show_memory(self):
        from PyQt6.QtWidgets import QMessageBox
        try:
            import psutil
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            QMessageBox.information(
                self, "内存使用情况",
                f"当前进程内存: {mem:.1f} MB\n\n如果数值过大,可以点'释放内存'清理。")
        except ImportError:
            QMessageBox.information(
                self, "提示", "请安装 psutil: pip install psutil")
