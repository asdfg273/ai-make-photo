# ui/shell.py
# ============================================================
#  新主窗口外壳 v6.0 — NavRail + 中央Stacked + 右侧参数面板
#  生成核心控件全局单例；页面专属区随导航切换
# ============================================================
import logging
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QProgressBar, QLabel, QStatusBar,
                             QScrollArea, QSizePolicy, QApplication, QMessageBox)
from PyQt6.QtCore import Qt

from ui.nav import NavRail
from ui.pages import PAGES
from ui.contracts import install_aliases, check_contract, apply_degradation
from ui.theme import PALETTE
from ui.i18n import ui_i18n

_DEFAULT_FG = PALETTE["fg"]
_OK = PALETTE["success"]
_WARN = PALETTE["warning"]
_ERR = PALETTE["danger"]
_INFO = PALETTE["info"]

logger = logging.getLogger(__name__)


def _describe_chain(wf) -> str:
    """把工作流按执行顺序写成一行「A → B → C」，给弹窗与日志用。"""
    try:
        from workflow import graph as _G
        types = []
        for nid in _G.topo_sort(wf):
            n = wf.node(nid)
            if n is not None:
                types.append(n.type)
        return "→ ".join(types) or "(空)"
    except Exception:
        return "→ ".join(n.type for n in wf.nodes) or "(空)"


class ShellMixin:
    """替代 UIBuilderMixin 的 setup_ui 实现。业务 mixin 零改动。"""

    def setup_ui(self):
        cfg = getattr(self, "config", None)
        ui_i18n.set_language(getattr(cfg, "ui_language", "zh"))
        ui_i18n.install(QApplication.instance())
        self.setMinimumSize(1180, 760)
        self.setWindowTitle(ui_i18n.tr("AI 绘画工作站 v6.0"))
        # 主题由 main.py 的 apply_theme(QApplication) 统一应用

        from ui.core_panel import build_status_widgets
        build_status_widgets(self)   # lbl_status / progress_gen 先建，状态栏复用

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        # ── 主体三栏 ──
        body = QHBoxLayout()
        body.setSpacing(12)

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
                logger.exception(f"页面构建失败: {cls.page_id}")
                self.center_stack.addWidget(QLabel(f"页面 {cls.title} 加载失败"))
                continue
            self._pages[cls.page_id] = page
            ws = page.workspace()
            if ws not in self._ws_added:      # 多页可共享同一中央工作区实例
                self._ws_added.add(ws)
                self.center_stack.addWidget(ws)
            pw = page.params_widget()
            if pw is not None:
                # 关键：水平策略 Ignored —— 页面内长文本控件（复选框/下拉框）
                # 不会把滚动区撑爆，否则 400px 面板右侧控件被整体裁掉
                sp = pw.sizePolicy()
                sp.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
                pw.setSizePolicy(sp)
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
            self.filmstrip.reuse_requested.connect(self._on_filmstrip_reuse)
            self._refresh_filmstrip()

        # ── 契约自检 + 分级降级 ──
        install_aliases(self)
        crit, minor = check_contract(self)
        if minor:
            logger.warning(f"契约自检（非关键缺失）: {minor}")
        apply_degradation(self, crit)

        self._build_statusbar_v6()
        self._build_menu_v6()
        self._init_defaults()
        self._setup_shortcuts()
        self.lbl_ext_count = QLabel()
        self.lbl_ext_count.setProperty("role", "hint")
        self.statusBar().addPermanentWidget(self.lbl_ext_count)
        self._refresh_extension_count()
        ui_i18n.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    def retranslate_ui(self):
        """语言切换后就地更新文字，不重建页面或清空用户输入。"""
        # 这些下拉框由业务层读取 currentIndex/currentData，允许翻译显示文本。
        # 其他下拉框可能以 currentText 作为参数值，默认保持原文。
        for name in ("combo_trans_mode", "combo_enhance_mode",
                     "combo_video_mode", "combo_travel_mode"):
            combo = getattr(self, name, None)
            if combo is not None:
                combo.setProperty("i18nItems", True)
        ui_i18n.translate_widget_tree(self)
        self.setWindowTitle(ui_i18n.tr("AI 绘画工作站 v6.0"))
        self._render_resource_label()
        self._render_extension_count()
        retranslate_video = getattr(self, "_retranslate_video_dynamic", None)
        if callable(retranslate_video):
            retranslate_video()

    # ---------- 全局快捷键（键位可在 设置→快捷键 改）----------
    def _setup_shortcuts(self):
        self._rebuild_shortcuts()

    def _shortcut_map(self) -> dict:
        from core.config_manager import DEFAULT_SHORTCUTS
        m = dict(DEFAULT_SHORTCUTS)
        saved = getattr(getattr(self, "config", None), "shortcuts", None) or {}
        m.update({k: v for k, v in saved.items() if k in m})
        return m

    def _rebuild_shortcuts(self):
        """按当前配置重建全部全局快捷键（设置改键后立即生效）。"""
        from PyQt6.QtGui import QShortcut, QKeySequence
        for sc in getattr(self, "_shortcuts", []):
            try:
                sc.setEnabled(False)
                sc.setParent(None)
                sc.deleteLater()
            except Exception:
                pass
        self._shortcuts = []   # 驻留防 GC

        def _add(seq, fn):
            if not seq:          # 清空 = 禁用该快捷键
                return
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

        m = self._shortcut_map()
        _add(m.get("generate"), lambda: getattr(self, "btn_generate", None)
             and self.btn_generate.isEnabled() and self.btn_generate.click())
        _add(m.get("interrupt"), lambda: getattr(self, "btn_interrupt", None)
             and self.btn_interrupt.isEnabled() and self.btn_interrupt.click())
        # 页面切换（顺序 = PAGES 注册顺序）
        for cls in PAGES:
            _add(m.get(f"page_{cls.page_id}", ""),
                 lambda pid=cls.page_id: self.nav.select(pid))
        # 扩展市场菜单项的键位同步
        act = getattr(self, "_act_market", None)
        if act is not None:
            act.setShortcut(QKeySequence(m.get("extension_market", "")))

    # ---------- 页面切换 ----------
    def _on_page_selected(self, page_id: str):
        page = self._pages.get(page_id)
        if page is None:
            return
        # 离开动画页时停止播放；进入时刷新视频历史
        prev = getattr(self, "_current_page_id", None)
        if prev == "video"and page_id != "video"and hasattr(self, "stop_video"):
            try:
                self.stop_video()
            except Exception:
                pass
        self._current_page_id = page_id
        if page_id == "txt2img"and getattr(self, "ref_image_path", None):
            self.clear_reference()

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
        if page_id == "gallery"and getattr(self, "gallery", None) is not None:
            from PyQt6.QtCore import QTimer
            def _reload():
                try:
                    from utils.paths import host_output_dir
                    self.gallery.reload_from_dir(host_output_dir(self), limit=80)
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
        """单击胶片条 → 原地全量回填参数 + 更新预览，不跳页。

        之前会跳到画廊页，但画廊页没有参数面板，用户看不到回填结果，
        体验等同"没回填"。现在停在当前页（通常在文生图页），
        回填后可直接点生成。"""
        gallery = getattr(self, "gallery", None)
        if gallery is not None:
            for i in range(gallery.list_widget.count()):
                item = gallery.list_widget.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == path:
                    gallery.list_widget.setCurrentItem(item)
                    break
        from ui.gallery_panel import GalleryPanel
        if GalleryPanel.media_kind(path) != "image":
            return
        # 预览区同步显示该图
        try:
            from PyQt6.QtGui import QPixmap
            if hasattr(self, "lbl_preview"):
                self.lbl_preview.set_pixmap(QPixmap(path))
            self.last_generated_path = path
            self.current_generated_path = path
        except Exception:
            pass
        if hasattr(self, "reuse_params_from_path"):
            try:
                self.reuse_params_from_path(path)
            except Exception as e:
                logger.warning(f"胶片条复用参数失败: {e}")

    def _on_filmstrip_reuse(self, path: str):
        """胶片条右键「套用参数」→ 直接全量回填，不跳页。"""
        from ui.gallery_panel import GalleryPanel
        if GalleryPanel.media_kind(path) == "video":
            return
        if hasattr(self, "reuse_params_from_path"):
            try:
                self.reuse_params_from_path(path)
            except Exception as e:
                logger.warning(f"胶片条套用参数失败: {e}")

    def _on_gallery_video_picked(self, path: str):
        """画廊双击视频 → 跳动画页播放。"""
        self.nav.select("video")
        self.play_video(path)

    # ---------- 右侧面板（整体可滚动，生成按钮固定底部）----------
    def _build_params_panel(self) -> QWidget:
        w = QWidget()
        w.setObjectName("paramsPanel")
        w.setFixedWidth(400)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(8)

        # 单一滚动区：核心区 + 页面专属区 + 共享折叠分组，内容再多也不挤压
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.params_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        clay = QVBoxLayout(content)
        clay.setContentsMargins(2, 2, 2, 2)
        clay.setSpacing(10)

        # 生成核心区（全局单例，包一层容器便于整区显隐）
        self.core_wrap = QWidget()
        self.core_area = QVBoxLayout(self.core_wrap)
        self.core_area.setContentsMargins(0, 0, 0, 0)
        from ui.core_panel import build_core, build_gen_area
        build_core(self, self.core_area)
        # 水平 Ignored：长文本控件不再撑爆 400px 面板（否则右侧控件被裁掉）
        _sp = self.core_wrap.sizePolicy()
        _sp.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.core_wrap.setSizePolicy(_sp)
        clay.addWidget(self.core_wrap)

        self.params_stack = QStackedWidget() # 页面专属区
        _sp = self.params_stack.sizePolicy()
        _sp.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.params_stack.setSizePolicy(_sp)
        clay.addWidget(self.params_stack)

        self.shared_groups = QVBoxLayout()   # 共享折叠分组（LoRA/CN/高级/X-Y）
        from ui.shared_groups import build_shared_groups
        self._group_sections = build_shared_groups(self, self.shared_groups)
        shared_wrap = QWidget()
        shared_wrap.setLayout(self.shared_groups)
        _sp = shared_wrap.sizePolicy()
        _sp.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        shared_wrap.setSizePolicy(_sp)
        clay.addWidget(shared_wrap)
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
    def append_log(self, text: str, color: str = _DEFAULT_FG):
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

    def set_status(self, text: str, color: str = _DEFAULT_FG):
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
        logger.info(f"尝试播放视频: {video_path}")
        if not _os.path.exists(video_path):
            _status(f"视频文件不存在: {video_path}", _ERR)
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
            _status(f"正在播放: {_os.path.basename(video_path)}", _DEFAULT_FG)
            self.current_video_path = video_path
        except Exception as e:
            import traceback
            _status(f"视频播放失败: {e}", _ERR)
            logger.error(f"视频播放失败: {e}")
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

    def _toggle_freeu(self):
        """FreeU 复选框切换回调（无额外控件联动，仅留钩子以备未来加 s1/s2/b1/b2 滑块）"""
        pass

    def _on_clip_skip_changed(self, index: int = 0):
        """CLIP Skip 下拉框切换 → 立即应用到已加载的文本编码器。

        之前这个能力只在 Capabilities.clip_skip 里声明、从未接线，
        等于动漫底模的用户永远拿不到应有的质量。
        """
        skip = max(0, int(index))
        try:
            self.config.clip_skip = skip
        except Exception:
            pass
        try:
            from utils.clip_skip import apply_clip_skip_to_pipes, supports_clip_skip
            ok, reason = supports_clip_skip(self.ai)
            if skip and not ok:
                # 架构不支持时明确说明原因，避免"调了没用"的假象
                self._set_status(f"CLIP Skip 不适用于当前模型：{reason}", _WARN)
                return
            n = apply_clip_skip_to_pipes(self.ai, skip)
            if skip:
                if n:
                    self._set_status(f"CLIP Skip = {skip}（已作用于 {n} 个文本编码器）",
                                     _OK)
                else:
                    self._set_status(
                        "当前架构不支持 CLIP Skip（仅 CLIP 系：SD1.5/SD2.1）", _WARN)
            else:
                self._set_status("CLIP Skip 已关闭（使用最后一层）", _INFO)
        except Exception as e:
            logger.warning(f"CLIP Skip 应用失败: {e}")
            self._set_status(f"CLIP Skip 应用失败: {e}", _ERR)

    def _toggle_hires(self):
        if not hasattr(self, 'chk_hires'):
            return
        on = self.chk_hires.isChecked()
        for name in ("combo_hires_scale", "scale_hires_denoise",
                     "scale_hires_tile_denoise", "combo_hires_upscaler",
                     "combo_hires_mode", "combo_hires_order"):
            c = getattr(self, name, None)
            if c is not None:
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
                "工作流程:\n\n"
                "1. AI 用提示词生成动作参考图\n"
                "2. 自动提取 OpenPose 骨架\n"
                "3. 骨架 + IP-Adapter 角色图 → 最终图\n\n"
                "请确保已上传【IP-Adapter 角色参考图】\n"
                "推荐: 影响力 0.6~0.8")

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
        logger.info(f"AI 模型档位 → {self.combo_ai_model.currentText()}")

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
                self.set_status("没有可对比的修前快照", _WARN)
                btn = getattr(self, "btn_compare", None)
                if btn is not None:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
        else:
            stack.setCurrentIndex(0)

    # ---------- 翻译回译对比 ----------
    def _on_trans_compare(self):
        """中→英→中 回译对比：检查 AI 翻译是否有幻觉/漏词。"""
        raw = self.txt_prompt.toPlainText().strip() \
            if hasattr(self, "txt_prompt") else ""
        if not raw:
            self.set_status("提示词为空，无法回译对比", _WARN)
            return
        translator = getattr(self, "translator", None)
        if translator is None:
            self.set_status("翻译服务未就绪", _WARN)
            return

        mode_idx = self.combo_trans_mode.currentIndex() \
            if hasattr(self, "combo_trans_mode") else 2
        mode = ["dict", "ai", "auto"][mode_idx]

        btn = getattr(self, "btn_trans_compare", None)
        if btn is not None:
            btn.setEnabled(False)
            btn.setText("翻译中…")
        self.set_status("正在 中→英→中 回译...", _INFO)

        import threading
        holder = {"phase": "init"}

        def _work():
            try:
                from utils.prompt_enhancer import get_enhancer
                enh = get_enhancer()
                was_loaded = getattr(enh, "model", None) is not None
                prev_qwen = getattr(translator, "qwen_enhancer", None)
                try:
                    # AI 相关模式：确保翻译服务手里有 Qwen 实例，
                    # 否则 zh→en 会悄悄降级成词典，对比结果不代表 AI 水平
                    if mode != "dict"and prev_qwen is None:
                        translator.qwen_enhancer = enh
                    holder["phase"] = "zh2en"
                    en = translator.translate(raw, mode=mode)
                    holder["en"] = en
                    holder["already_en"] = (en.strip() == raw.strip())
                    # 回译必须走 AI：模型已卸载则现载（点对比才载入），
                    # 完成后若原本是卸载状态则再次卸下，不占生图显存。
                    # 注意：即使提示词已是英文（en==raw）也要回译——
                    # ③展示的是 AI 对这段英文的"理解"，同样是幻觉检查
                    if en and en.strip():
                        holder["phase"] = "en2zh"
                        back = enh.translate(en, target_lang="zh")
                        holder["back"] = back \
                            if back and back.strip() != en.strip() else ""
                    else:
                        holder["back"] = ""
                finally:
                    translator.qwen_enhancer = prev_qwen
                    if not was_loaded and getattr(enh, "model", None) is not None:
                        enh.unload(reason="回译对比完成，释放显存")
            except Exception as e:
                holder["err"] = str(e)

        t = threading.Thread(target=_work, daemon=True)
        t.start()

        from PyQt6.QtCore import QTimer
        timer = QTimer(self)
        timer.setInterval(200)

        def _poll():
            if t.is_alive():
                phase = holder.get("phase")
                if phase == "zh2en":
                    self.set_status("中→英 翻译中...", _INFO)
                elif phase == "en2zh":
                    self.set_status("载入 AI 回译 英→中（首次需加载模型，请稍候）...",
                                    _INFO)
                return
            timer.stop()
            if btn is not None:
                btn.setEnabled(True)
                btn.setText("回译对比")
            self._show_trans_compare(raw, holder, mode)

        timer.timeout.connect(_poll)
        timer.start()
        self._trans_poll_timer = timer  # 防 GC

    def _show_trans_compare(self, raw: str, holder: dict, mode: str):
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTextEdit,
                                     QLabel, QDialogButtonBox)
        if "err"in holder:
            self.set_status(f"回译失败: {holder['err']}", _ERR)
            return
        en = holder.get("en", "")
        back = holder.get("back", "")
        if not en:
            self.set_status("翻译结果为空", _WARN)
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("翻译回译对比（检查幻觉）")
        dlg.resize(560, 520)
        v = QVBoxLayout(dlg)

        def _block(title, text, hint=None):
            lbl = QLabel(title)
            lbl.setProperty("role", "title")
            v.addWidget(lbl)
            te = QTextEdit()
            te.setReadOnly(True)
            te.setPlainText(text)
            v.addWidget(te, 1)
            if hint:
                h = QLabel(hint)
                h.setProperty("role", "hint")
                h.setWordWrap(True)
                v.addWidget(h)

        _block("① 原文", raw)
        if holder.get("already_en"):
            _block(f"② 英文（提示词已是英文，未做中→英 · 模式 {mode}）", en)
        else:
            _block(f"② AI 英文（实际送入模型 · 模式 {mode}）", en)
        if back and back.strip() and back.strip() != en.strip():
            _block("③ 回译（英→中，对照①检查是否词不达意）", back,
                   "若③与①语义偏差大，说明②有幻觉/漏词，"
                   "建议换「AI 智能改写」模式或手动修改英文")
        else:
            _block("③ 回译（英→中）",
                   "（AI 正忙或回译失败；②即实际送入模型的英文，"
                   "可直接对照①检查）")
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        v.addWidget(bb)
        dlg.show()   # 非模态，方便对照主窗口
        self._trans_compare_dlg = dlg  # 防 GC
        self.set_status("回译完成，请对照 ①/③ 检查", _OK)

    # ---------- 显存/内存常驻监控 ----------
    def _setup_resource_monitor(self):
        from PyQt6.QtCore import QTimer
        self.lbl_resource = QLabel("— —")
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
            self._resource_ram_gb = psutil.Process().memory_info().rss / 1024**3
            self._resource_vram_gb = None
            try:
                import torch
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    self._resource_vram_gb = (used, total)
            except Exception:
                pass
        except Exception:
            self._resource_ram_gb = None
            self._resource_vram_gb = None
        self._render_resource_label()

    def _render_resource_label(self):
        lbl = getattr(self, "lbl_resource", None)
        if lbl is None:
            return
        ram = getattr(self, "_resource_ram_gb", None)
        if ram is None:
            lbl.setText("— —")
            return
        text = f"{ui_i18n.tr('内存')} {ram:.1f}GB"
        vram = getattr(self, "_resource_vram_gb", None)
        if vram is not None:
            used, total = vram
            text += f" ·  {ui_i18n.tr('显存')} {used:.1f}/{total:.1f}GB"
        lbl.setText(text)

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
            if "["in lora_name:
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
                        logger.warning(f"读取 {txt_path} 失败: {e}")

        if not triggers_list:
            self._set_status("没有可插入的触发词", _WARN)
            return

        all_triggers = ", ".join(triggers_list)
        cur = self.txt_prompt.toPlainText().strip()
        new_text = f"{all_triggers}, {cur}"if cur else all_triggers
        self.txt_prompt.setPlainText(new_text)
        self._set_status(f"已插入 {len(triggers_list)} 组触发词", _DEFAULT_FG)

    # ---------- 菜单 ----------
    def _build_menu_v6(self):
        from PyQt6.QtGui import QAction
        mb = self.menuBar()

        m_file = mb.addMenu("文件")
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

        m_tool = mb.addMenu("工具")
        a_clear_log = QAction("清空日志", self)
        a_clear_log.triggered.connect(lambda: (
            getattr(self, 'txt_log_image', None) and self.txt_log_image.clear(),
            getattr(self, 'txt_log_video', None) and self.txt_log_video.clear()))
        m_tool.addAction(a_clear_log)
        m_tool.addSeparator()
        act_market = QAction("扩展市场...", self)
        act_market.setShortcut("Ctrl+E")
        act_market.triggered.connect(self._open_extension_market)
        m_tool.addAction(act_market)
        self._act_market = act_market   # 设置改键后同步
        act_refresh = QAction("刷新扩展状态", self)
        act_refresh.triggered.connect(self._refresh_extension_count)
        m_tool.addAction(act_refresh)
        m_tool.addSeparator()
        act_train = QAction("训练 LoRA…", self)
        act_train.setToolTip("用自己的图片训练一个轻量 LoRA（只训附加权重，8GB 显存可跑）")
        act_train.triggered.connect(self._open_lora_trainer)
        m_tool.addAction(act_train)
        self._act_lora_train = act_train

        # ---------- 工作流（节点编辑 / 用引擎生成）----------
        m_wf = mb.addMenu("工作流")
        a_wf_edit = QAction("打开工作流编辑器…", self)
        a_wf_edit.setShortcut("Ctrl+W")
        a_wf_edit.setToolTip("可视化节点编辑器：看/改这张「界面自动生成的流程图」")
        a_wf_edit.triggered.connect(self._open_workflow_editor)
        m_wf.addAction(a_wf_edit)

        a_wf_gen = QAction("用当前界面设置生成流程图", self)
        a_wf_gen.setToolTip("把当前模型/提示词/LoRA/高清/修脸等设置编译成工作流并打开编辑器")
        a_wf_gen.triggered.connect(self._open_workflow_from_ui)
        m_wf.addAction(a_wf_gen)

        # 从内置模板新建：给"不知道节点怎么连"的人一个能直接跑的起点
        self._wf_template_menu = m_wf.addMenu("从模板新建…")
        self._wf_template_menu.setToolTip("文生图 / 高清 / 局部重绘 / 参考图 / 放大，五个能直接跑的流程")
        try:
            from workflow import templates as _tpl
            for _key, _name, _desc in _tpl.list_templates():
                _act = QAction(_name, self)
                _act.setToolTip(_desc)
                _act.triggered.connect(
                    lambda _checked=False, _k=_key: self._open_workflow_template(_k))
                self._wf_template_menu.addAction(_act)
        except Exception as _e:      # 模板加载失败不该让主窗口起不来
            logger.warning(f"工作流模板菜单构建失败: {_e}")
        m_wf.addSeparator()

        a_wf_run = QAction("用工作流引擎生成", self)
        a_wf_run.setShortcut("Ctrl+Shift+W")
        a_wf_run.setToolTip("按当前界面设置生成工作流并**立即执行**（走 workflow 引擎）")
        a_wf_run.triggered.connect(self._run_workflow_from_ui)
        m_wf.addAction(a_wf_run)

        a_wf_check = QAction("预检当前工作流", self)
        a_wf_check.setToolTip("运行前预检：模型/LoRA/VAE/ControlNet/显存/输出目录/依赖")
        a_wf_check.triggered.connect(self._check_workflow_from_ui)
        m_wf.addAction(a_wf_check)
        m_wf.addSeparator()

        a_wf_dir = QAction("打开工作流目录", self)
        a_wf_dir.triggered.connect(self._open_workflow_dir)
        m_wf.addAction(a_wf_dir)

        m_memory = mb.addMenu("内存")
        a_release = QAction("释放内存", self)
        a_release.triggered.connect(self.on_unload_models)
        m_memory.addAction(a_release)
        a_show = QAction("查看当前内存", self)
        a_show.triggered.connect(self._show_memory)
        m_memory.addAction(a_show)

        m_about = mb.addMenu("关于")
        a_about = QAction("关于本软件", self)
        a_about.triggered.connect(self._show_about)
        m_about.addAction(a_about)

        m_setting = mb.addMenu("设置")
        a_pref = QAction("偏好设置...", self)
        a_pref.setShortcut("Ctrl+,")
        a_pref.triggered.connect(self._open_settings)
        m_setting.addAction(a_pref)

    def _open_settings(self):
        """设置对话框：保存后立即应用默认值并重建快捷键。"""
        from ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        if not dlg.exec():
            return

        cfg = self.config
        prompt_bak = self.txt_prompt.toPlainText() \
            if hasattr(self, "txt_prompt") else ""
        neg_bak = self.txt_neg.toPlainText() \
            if hasattr(self, "txt_neg") else ""
        try:
            if hasattr(self, "apply_config_to_ui"):
                self.apply_config_to_ui()
        finally:
            if hasattr(self, "txt_prompt"):
                self.txt_prompt.setPlainText(prompt_bak)
            if hasattr(self, "txt_neg"):
                self.txt_neg.setPlainText(neg_bak)

        if hasattr(self, "combo_device"):
            idx = self.combo_device.findText(
                str(getattr(cfg, "device_preference", "auto")).upper())
            if idx >= 0:
                self.combo_device.setCurrentIndex(idx)

        self._rebuild_shortcuts()
        self.retranslate_ui()
        self.set_status(ui_i18n.tr("设置已保存并应用"), _OK)
        self._refresh_extension_count()   # 只刷新文字，不再重建控件

    def _refresh_extension_count(self):
        if getattr(self, 'lbl_ext_count', None) is None:
            return
        try:
            from utils.extension_manager import get_status_summary
            s = get_status_summary()
            self._extension_count = (s['installed'], s['total'])
        except Exception as e:
            self._extension_count = None
            logger.warning(f"[EXT-COUNT] 刷新失败: {e}")
        self._render_extension_count()

    def _render_extension_count(self):
        lbl = getattr(self, "lbl_ext_count", None)
        if lbl is None:
            return
        count = getattr(self, "_extension_count", None)
        values = "--" if count is None else f"{count[0]}/{count[1]}"
        lbl.setText(f"{ui_i18n.tr('扩展:')} {values}")

    def _open_extension_market(self):
        try:
            from PyQt6.QtCore import Qt
            from ui.extension_market import ExtensionMarketDialog
            dlg = ExtensionMarketDialog(self)
            # 关闭即销毁：原来每次打开都 new 一个对话框且从不释放，
            # 反复打开会累积对话框与全部卡片（内存单向增长）。
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            dlg.exec()
            self._refresh_extension_count()
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self, ui_i18n.tr("错误"),
                ui_i18n.trf("打开扩展市场失败:\n{error}", error=e))

    # ==================================================================
    #  LoRA 训练
    # ==================================================================
    def _open_lora_trainer(self):
        """打开轻量 LoRA 训练窗口（数据集目录默认给上次用过的）。"""
        try:
            from ui.lora_train_dialog import LoraTrainDialog
            last = str(getattr(self.config, "lora_dataset_dir", "") or "")
            dlg = LoraTrainDialog(self, dataset_default=last)
            self._lora_train_dialog = dlg
            dlg.exec()
            # 读控件必须在控件还活着的时候做：窗口若带 WA_DeleteOnClose，
            # exec() 返回时 C++ 对象可能已被销毁，读 edit_dataset 会抛
            # RuntimeError: wrapped C/C++ object of type QLineEdit has been deleted
            try:
                picked = dlg.edit_dataset.text().strip()
            except RuntimeError:
                picked = last
            if picked and picked != last:
                self.config.lora_dataset_dir = picked
                try:
                    self.config.save()
                except Exception as exc:  # noqa: BLE001 - 记不住目录不算错误
                    logger.debug("保存 lora_dataset_dir 失败: %s", exc)
            try:
                dlg.deleteLater()
            except Exception:  # noqa: BLE001
                pass
            self._lora_train_dialog = None
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self, ui_i18n.tr("错误"),
                ui_i18n.trf("打开 LoRA 训练窗口失败:\n{error}", error=e))

    # ==================================================================
    #  工作流（节点编辑器 / 用引擎生成）
    # ==================================================================
    def _open_workflow_editor(self):
        """打开节点编辑器（不带入当前界面设置）。"""
        try:
            from ui.workflow_editor import open_editor
            self._workflow_editor_win = open_editor(host=self, parent=None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self, ui_i18n.tr("错误"),
                ui_i18n.trf("打开工作流编辑器失败:\n{error}", error=e))

    def _open_workflow_template(self, key: str):
        """从内置模板新建一张图并打开编辑器。"""
        try:
            from ui.workflow_editor import open_editor
            win = open_editor(host=self, parent=None)
            self._workflow_editor_win = win
            win.load_template(key)
        except Exception as e:
            logger.exception("从模板新建工作流失败")
            QMessageBox.warning(
                self, ui_i18n.tr("打开模板失败"),
                ui_i18n.trf("打开模板失败:\n{error}", error=e))

    def _open_workflow_from_ui(self):
        """把当前界面设置编译成工作流图并打开编辑器。"""
        try:
            from ui.workflow_editor import open_editor
            win = open_editor(host=self, parent=None)
            win.on_build_from_ui()
        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self, ui_i18n.tr("错误"),
                ui_i18n.trf("生成工作流失败:\n{error}", error=e))

    def _workflow_from_ui_or_warn(self):
        """编译当前界面 → Workflow；失败时弹窗并返回 None。

        另外把"界面上开了但没能进工作流"的项（通常是缺参考图）**明确告诉你**——
        这类静默降级以前不报错也不警告，用户会以为功能生效了。
        """
        from PyQt6.QtWidgets import QMessageBox
        try:
            from workflow import builder
            wf = builder.build_from_ui(self)
        except Exception as e:
            QMessageBox.warning(
                self, ui_i18n.tr("无法生成工作流"),
                f"{type(e).__name__}: {e}\n\n"
                + ui_i18n.tr("请确认模型列表已加载（等主界面显示「系统就绪」后再试）。"))
            return None
        warns = list((wf.metadata or {}).get("warnings") or [])
        if warns:
            for w in warns:
                try:
                    self.append_log(f"界面上的设置没能进工作流：{w}",
                                    "#fab387")
                except Exception:
                    pass
            QMessageBox.warning(
                self, ui_i18n.tr("有设置没进工作流"),
                ui_i18n.tr("以下设置在界面上是开着的，但没有进入生成的工作流"
                           "（通常是缺少参考图/姿势图）：")
                + "\n\n" + "\n".join(f"· {w}" for w in warns)
                + "\n\n" + ui_i18n.tr("补齐后重试即可。"))
        return wf

    def _check_workflow_from_ui(self):
        """预检当前界面会生成的工作流：不执行，只报错误/警告/提示。"""
        from PyQt6.QtWidgets import QMessageBox
        wf = self._workflow_from_ui_or_warn()
        if wf is None:
            return
        try:
            from workflow.preflight import run_preflight
            res = run_preflight(
                wf, host=self, ai=getattr(self, "ai", None), translate=ui_i18n.tr)
            text = (ui_i18n.tr("节点链：") + f"\n{_describe_chain(wf)}\n\n"
                    + res.to_text(translate=ui_i18n.tr))
            if res.errors:
                QMessageBox.warning(self, ui_i18n.tr("预检未通过"), text)
            else:
                QMessageBox.information(self, ui_i18n.tr("预检结果"), text)
        except Exception as e:
            QMessageBox.critical(
                self, ui_i18n.tr("错误"), ui_i18n.trf("预检失败:\n{error}", error=e))

    def _run_workflow_from_ui(self):
        """按当前界面设置生成工作流并立即执行（在工作线程里跑）。"""
        from PyQt6.QtWidgets import QMessageBox
        if getattr(self, "is_generating", False):
            QMessageBox.information(
                self, ui_i18n.tr("正在生成"), ui_i18n.tr("当前已有生成任务在跑，请等它结束。"))
            return
        if getattr(self, "ai", None) is None:
            QMessageBox.warning(
                self, ui_i18n.tr("引擎未就绪"), ui_i18n.tr("AI 引擎还在预热，请稍后再试。"))
            return
        wf = self._workflow_from_ui_or_warn()
        if wf is None:
            return
        try:
            from ui.workflow_editor import open_editor
            win = open_editor(host=self, parent=None)
            win.scene.load_workflow(wf)
            win._log(f"由主界面触发执行：{_describe_chain(wf)}")
            win.on_run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, ui_i18n.tr("错误"), ui_i18n.trf("执行工作流失败:\n{error}", error=e))

    def _open_workflow_dir(self):
        import os
        import subprocess
        import sys
        try:
            from workflow import serializers
            d = serializers.workflows_dir()
            if sys.platform.startswith("win"):
                os.startfile(os.path.abspath(d))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "无法打开目录", str(e))

    def _open_output_folder(self):
        import subprocess
        import sys
        import os
        from utils.paths import host_output_dir
        OUTPUT_DIR = host_output_dir(self)
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
        from utils.version import collect_versions
        try:
            _v = collect_versions()
            _ver_line = f"torch {_v.get('torch', '?')} · diffusers {_v.get('diffusers', '?')}"
        except Exception:
            _ver_line = ""
        QMessageBox.about(
            self, ui_i18n.tr("关于"),
            f"<b>{ui_i18n.tr('AI 绘画工作站 v6.0')}</b><br>"
            + ui_i18n.tr(
                "基于 Stable Diffusion + ADetailer<br>"
                "支持 LoRA / ControlNet / Hires.fix / IP-Adapter / Pose Transfer")
            + (f"<br><br><small>{_ver_line}</small>" if _ver_line else "")
            + "<br><br><small>"
            + ui_i18n.tr("本项目以 AGPL-3.0 发布，完整许可见 LICENSE")
            + "</small>")

    def on_unload_models(self):
        from PyQt6.QtWidgets import QMessageBox
        if getattr(self, 'is_generating', False):
            QMessageBox.warning(
                self, ui_i18n.tr("提示"), ui_i18n.tr("请先停止当前生成任务"))
            return
        try:
            from utils import mem as _mem

            before = _mem.rss_mb()
            self._set_status(ui_i18n.tr("正在释放模型..."), _WARN)
            if hasattr(self, 'ai'):
                self.ai.unload_all()
            # unload_all 内部只清引用；MOSS 这类 mmap 权重留下的文件页
            # 要显式 EmptyWorkingSet 才会还给系统，这里再补一刀。
            delta = _mem.release_memory()
            after = delta.get("rss_after", _mem.rss_mb())
            vram = delta.get("vram_after") or (-1, -1, -1)
            parts = [ui_i18n.trf(
                "模型已释放：进程内存 {a} MB → {b} MB", a=f"{before:.0f}", b=f"{after:.0f}")]
            if vram[2] >= 0:
                parts.append(ui_i18n.trf(
                    "显存空闲 {free} MB", free=f"{vram[2]:.0f}"))
            self._set_status("；".join(parts), _DEFAULT_FG)
        except Exception as e:
            QMessageBox.critical(self, ui_i18n.tr("释放失败"), str(e))

    def _show_memory(self):
        from PyQt6.QtWidgets import QMessageBox
        from utils import mem as _mem

        rss = _mem.rss_mb()
        alloc, reserved, free = _mem.vram_mb()
        lines = [ui_i18n.trf("进程内存: {v} MB", v=f"{rss:.0f}") if rss >= 0
                 else ui_i18n.tr("进程内存: 不可用")]
        if free >= 0:
            lines.append(ui_i18n.trf(
                "显存: 已分配 {a} MB / 已保留 {r} MB / 空闲 {f} MB",
                a=f"{alloc:.0f}", r=f"{reserved:.0f}", f=f"{free:.0f}"))
        lines.append("")
        lines.append(ui_i18n.tr(
            "换了模型之后如果这两个数字没有降下来，点『释放内存』："
            "它会卸载模型、回收显存，并把进程工作集还给系统。"))
        QMessageBox.information(
            self, ui_i18n.tr("内存使用情况"), "\n".join(lines))

    def _goto_page(self, page_id: str):
        page = getattr(self, "_pages", {}).get(page_id)
        if page is None:
            logger.warning(f"页面不存在: {page_id}")
            return None
        if hasattr(self, "nav"):
            self.nav.select(page_id)
        return page

    def _page_call(self, page_id: str, method: str, *args):
        page = self._goto_page(page_id)
        fn = getattr(page, method, None) if page else None
        if callable(fn):
            return fn(*args)
        logger.warning(f"页面 {page_id} 无方法 {method}")
        return None

