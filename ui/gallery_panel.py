# utils/gallery.py
import os
import json
import shutil
import subprocess
import sys
from PyQt6.QtWidgets import (
    QListWidget, QListWidgetItem, QMenu, QMessageBox,
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QLabel, QTextEdit, QDialog, QScrollArea, QFileDialog, QApplication
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QIcon, QWheelEvent

NSFW_KEYWORDS = {
    # 英文
    "nude", "naked", "nsfw", "sex", "pussy", "penis", "cum",
    "nipple", "nipples", "explicit", "rating:explicit", "uncensored",
    "hentai", "r18", "r-18", "18+", "topless", "bottomless",
    "pubic", "vagina", "anus", "areola", "orgasm",
    # 中文
    "裸", "露出", "性交", "情色", "色情","测试"
}

def is_nsfw_prompt(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(kw in low for kw in NSFW_KEYWORDS)


# ============================================================
#  G7: 大图预览弹窗 (支持滚轮缩放)
# ============================================================
class ImageViewerDialog(QDialog):
    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"🖼️ {os.path.basename(path)}")
        self.resize(1000, 800)
        self.path = path
        self.scale_factor = 1.0
        self.orig_pixmap = QPixmap(path)

        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        btn_zoom_in = QPushButton("🔍+ 放大")
        btn_zoom_out = QPushButton("🔍- 缩小")
        btn_fit = QPushButton("📐 适配窗口")
        btn_actual = QPushButton("1:1 原始")
        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setStyleSheet("color:#a6e3a1; padding:0 12px;")
        for b in (btn_zoom_in, btn_zoom_out, btn_fit, btn_actual):
            b.setStyleSheet("padding:6px 12px;")
            toolbar.addWidget(b)
        toolbar.addWidget(self.lbl_zoom)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setStyleSheet("background:#1e1e2e;")
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.lbl_img)
        layout.addWidget(self.scroll, 1)

        btn_zoom_in.clicked.connect(lambda: self._zoom(1.25))
        btn_zoom_out.clicked.connect(lambda: self._zoom(0.8))
        btn_fit.clicked.connect(self._fit_window)
        btn_actual.clicked.connect(lambda: self._set_scale(1.0))

        self.scroll.viewport().installEventFilter(self)
        self._fit_window()

    def _set_scale(self, factor: float):
        self.scale_factor = max(0.1, min(8.0, factor))
        w = int(self.orig_pixmap.width() * self.scale_factor)
        h = int(self.orig_pixmap.height() * self.scale_factor)
        scaled = self.orig_pixmap.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_img.setPixmap(scaled)
        self.lbl_img.resize(scaled.size())
        self.lbl_zoom.setText(f"{int(self.scale_factor * 100)}%")

    def _zoom(self, ratio: float):
        self._set_scale(self.scale_factor * ratio)

    def _fit_window(self):
        if self.orig_pixmap.isNull():
            return
        vw = self.scroll.viewport().width() - 20
        vh = self.scroll.viewport().height() - 20
        ratio = min(vw / self.orig_pixmap.width(),
                    vh / self.orig_pixmap.height())
        self._set_scale(ratio)

    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport() and isinstance(event, QWheelEvent):
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                delta = event.angleDelta().y()
                self._zoom(1.15 if delta > 0 else 0.87)
                return True
        return super().eventFilter(obj, event)


# ============================================================
#  G6: 元数据浮窗
# ============================================================
class MetadataPanel(QWidget):
    copy_prompt_signal = pyqtSignal(str)
    apply_params_signal = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_meta = {}
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("📋 元数据")
        title.setStyleSheet("font-weight:bold; color:#cdd6f4; font-size:13px;")
        layout.addWidget(title)

        lbl_p = QLabel("Prompt:")
        lbl_p.setStyleSheet("color:#a6e3a1; font-weight:bold; font-size:11px;")
        layout.addWidget(lbl_p)
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setReadOnly(True)
        self.txt_prompt.setFixedHeight(80)
        self.txt_prompt.setStyleSheet(
            "background:#181825; color:#cdd6f4; border:1px solid #313244; "
            "padding:4px; font-size:11px;"
        )
        layout.addWidget(self.txt_prompt)

        lbl_n = QLabel("Negative:")
        lbl_n.setStyleSheet("color:#f38ba8; font-weight:bold; font-size:11px;")
        layout.addWidget(lbl_n)
        self.txt_neg = QTextEdit()
        self.txt_neg.setReadOnly(True)
        self.txt_neg.setFixedHeight(60)
        self.txt_neg.setStyleSheet(
            "background:#181825; color:#cdd6f4; border:1px solid #313244; "
            "padding:4px; font-size:11px;"
        )
        layout.addWidget(self.txt_neg)

        lbl_param = QLabel("参数:")
        lbl_param.setStyleSheet("color:#89b4fa; font-weight:bold; font-size:11px;")
        layout.addWidget(lbl_param)
        self.lbl_params = QLabel()
        self.lbl_params.setWordWrap(True)
        self.lbl_params.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.lbl_params.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.lbl_params.setStyleSheet(
            "background:#181825; color:#cdd6f4; border:1px solid #313244; "
            "padding:6px; font-family:Consolas; font-size:10px;"
        )
        scroll = QScrollArea()
        scroll.setWidget(self.lbl_params)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border:none;")
        layout.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_copy_prompt = QPushButton("📋 复制 Prompt")
        self.btn_apply = QPushButton("↩️ 套用全部参数")
        for b in (self.btn_copy_prompt, self.btn_apply):
            b.setStyleSheet(
                "background:#45475a; color:#cdd6f4; padding:6px; "
                "border-radius:3px; font-size:11px;"
            )
            b.setFixedHeight(30)
        btn_row.addWidget(self.btn_copy_prompt)
        btn_row.addWidget(self.btn_apply)
        layout.addLayout(btn_row)

        self.btn_copy_prompt.clicked.connect(self._on_copy_prompt)
        self.btn_apply.clicked.connect(self._on_apply)
        self.clear()

    def load_from_path(self, path: str):
        self.current_meta = {}
        if not path or not os.path.exists(path):
            self.clear()
            return
        try:
            from PIL import Image
            img = Image.open(path)
            raw = img.info.get("parameters") or img.info.get("meta", "")
            if not raw:
                self.clear(hint="(本图无嵌入参数)")
                return
            try:
                meta = json.loads(raw)
            except Exception:
                meta = self._parse_a1111(raw)
            self.current_meta = meta
            self._render(meta)
        except Exception as e:
            self.clear(hint=f"(读取失败: {e})")

    def _parse_a1111(self, text: str) -> dict:
        lines = text.split("\n")
        meta = {"prompt": "", "negative_prompt": ""}
        if lines:
            meta["prompt"] = lines[0]
        for line in lines[1:]:
            if line.startswith("Negative prompt:"):
                meta["negative_prompt"] = line.replace("Negative prompt:", "").strip()
            elif ":" in line:
                for part in line.split(","):
                    if ":" in part:
                        k, v = part.split(":", 1)
                        meta[k.strip().lower().replace(" ", "_")] = v.strip()
        return meta

    def _render(self, meta: dict):
        self.txt_prompt.setPlainText(meta.get("prompt", "") or meta.get("p", ""))
        self.txt_neg.setPlainText(meta.get("negative_prompt", "") or meta.get("n", ""))
        skip = {"prompt", "negative_prompt", "p", "n"}
        lines = []
        for k, v in meta.items():
            if k in skip:
                continue
            val = str(v)
            if len(val) > 60:
                val = val[:57] + "..."
            lines.append(f"<b style='color:#89b4fa'>{k}</b>: {val}")
        self.lbl_params.setText("<br>".join(lines) if lines else "(无其他参数)")

    def clear(self, hint: str = ""):
        self.txt_prompt.setPlainText("")
        self.txt_neg.setPlainText("")
        self.lbl_params.setText(hint or "(未选中图片)")
        self.current_meta = {}

    def _on_copy_prompt(self):
        p = self.txt_prompt.toPlainText().strip()
        if not p:
            return
        QApplication.clipboard().setText(p)
        self.copy_prompt_signal.emit(p)

    def _on_apply(self):
        if not self.current_meta:
            return
        self.apply_params_signal.emit(self.current_meta)


# ============================================================
#  GalleryPanel (终极增强版)
# ============================================================
class GalleryPanel(QWidget):
    image_selected      = pyqtSignal(str)
    image_deleted       = pyqtSignal(str)
    apply_params_signal = pyqtSignal(dict)
    reuse_params_signal = pyqtSignal(str)   # 🔁 复用 PNG 参数
    send_to_i2i_signal  = pyqtSignal(str)   # 🛠 发送到 img2img
    send_to_face_signal = pyqtSignal(str)   # 😀 发送到修脸

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_items = []

        # 收藏持久化
        self._favs_path = os.path.join("data", "gallery_favs.json")
        self._favs = self._load_favs()

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── 搜索栏 ──
        search_row = QHBoxLayout()
        search_row.setSpacing(4)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 搜索文件名或 prompt 关键字...")
        self.search_box.setStyleSheet(
            "background:#313244; color:#cdd6f4; border:1px solid #45475a; "
            "padding:6px; border-radius:4px;"
        )
        self.search_box.textChanged.connect(self._apply_filter)

        self.btn_clear_search = QPushButton("✖")
        self.btn_clear_search.setToolTip("清空搜索")
        self.btn_clear_search.clicked.connect(lambda: self.search_box.clear())

        self.btn_only_fav = QPushButton("⭐")
        self.btn_only_fav.setCheckable(True)
        self.btn_only_fav.setToolTip("仅显示收藏")

        self.btn_only_fav.toggled.connect(self._apply_filter)

        self.btn_show_nsfw = QPushButton("废")
        self.btn_show_nsfw.setCheckable(True)
        self.btn_show_nsfw.setChecked(False)   # 默认关闭
        self.btn_show_nsfw.setToolTip("显示废弃内容（默认过滤）")
        self.btn_show_nsfw.toggled.connect(self._apply_filter)


        self.btn_show_meta = QPushButton("📋")
        self.btn_show_meta.setToolTip("显示/隐藏元数据面板")
        self.btn_show_meta.clicked.connect(self._toggle_meta_panel)

        self.lbl_count = QLabel("0 张")
        self.lbl_count.setStyleSheet("color:#a6e3a1; padding:0 6px;")

        search_row.addWidget(self.search_box, 1)
        search_row.addWidget(self.btn_clear_search)
        search_row.addWidget(self.btn_only_fav)
        search_row.addWidget(self.btn_show_nsfw)
        search_row.addWidget(self.btn_show_meta)
        search_row.addWidget(self.lbl_count)
        root.addLayout(search_row)

        # ── 缩略图列表 ──
        self.setMinimumHeight(280)
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(100, 100))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSpacing(6)
        self.list_widget.setMovement(QListWidget.Movement.Static)
        # ⭐ 开启多选
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_menu)
        self.list_widget.itemClicked.connect(self._on_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_double_clicked)
        root.addWidget(self.list_widget, 1)

        # ── 元数据浮窗 ──
        self.meta_panel = MetadataPanel()
        self.meta_panel.setWindowFlags(Qt.WindowType.Tool)
        self.meta_panel.setWindowTitle("📋 图片元数据")
        self.meta_panel.resize(380, 520)
        self.meta_panel.apply_params_signal.connect(self.apply_params_signal)
        self.meta_panel.hide()

    # ========== 收藏持久化 ==========
    def _load_favs(self) -> set:
        try:
            os.makedirs(os.path.dirname(self._favs_path), exist_ok=True)
            if os.path.exists(self._favs_path):
                with open(self._favs_path, "r", encoding="utf-8") as f:
                    return set(json.load(f))
        except Exception as e:
            print(f"⚠️ 加载收藏失败: {e}")
        return set()

    def _save_favs(self):
        try:
            with open(self._favs_path, "w", encoding="utf-8") as f:
                json.dump(sorted(self._favs), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存收藏失败: {e}")

    # ========== 数据管理 ==========
    def add_image(self, path: str, prepend: bool = False):
        if not os.path.exists(path):
            return
        # 去重
        for p, _, _ in self._all_items:
            if os.path.abspath(p) == os.path.abspath(path):
                return
        prompt_text = self._extract_prompt(path)
        nsfw = is_nsfw_prompt(prompt_text)
        if prepend:
            self._all_items.insert(0, (path, prompt_text, nsfw))
        else:
            self._all_items.append((path, prompt_text, nsfw))
        self._apply_filter()

    def reload_from_dir(self, directory: str, limit: int = 80):
        self._all_items.clear()
        if not os.path.isdir(directory):
            return
        files = []
        for f in os.listdir(directory):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                full = os.path.join(directory, f)
                files.append((full, os.path.getmtime(full)))
        files.sort(key=lambda x: -x[1])
        for path, _ in files[:limit]:
            prompt_text = self._extract_prompt(path)
            nsfw = is_nsfw_prompt(prompt_text)
            self._all_items.append((path, prompt_text, nsfw))  
        self._apply_filter()

    def _extract_prompt(self, path: str) -> str:
        try:
            from PIL import Image
            img = Image.open(path)
            raw = img.info.get("parameters") or img.info.get("meta", "")
            if not raw:
                return ""
            try:
                meta = json.loads(raw)
                return meta.get("prompt", "") or meta.get("p", "")
            except Exception:
                return raw[:500]
        except Exception:
            return ""

    # ========== 搜索过滤 ==========
    def _apply_filter(self):
        keyword = self.search_box.text().strip().lower()
        only_fav = self.btn_only_fav.isChecked()
        show_nsfw = self.btn_show_nsfw.isChecked()  
        self.list_widget.clear()
        shown = 0
        nsfw_hidden = 0
        for path, prompt_text, nsfw in self._all_items:   
            if nsfw and not show_nsfw:
                nsfw_hidden += 1
                continue
            if only_fav and os.path.abspath(path) not in self._favs:
                continue
            if keyword:
                name = os.path.basename(path).lower()
                if keyword not in name and keyword not in prompt_text.lower():
                    continue
            self._add_to_list(path)
            shown += 1

        if nsfw_hidden > 0 and not show_nsfw:
            self.lbl_count.setText(f"{shown}/{len(self._all_items)} 张 (🔞 隐藏 {nsfw_hidden})")
        else:
            self.lbl_count.setText(f"{shown}/{len(self._all_items)} 张")

    def _add_to_list(self, path: str):
        item = QListWidgetItem()
        pix = QPixmap(path)
        if not pix.isNull():
            icon = QIcon(pix.scaled(
                110, 110,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            item.setIcon(icon)
        name = os.path.basename(path)[:20]
        if os.path.abspath(path) in self._favs:
            name = f"⭐ {name}"
        item.setText(name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self.list_widget.addItem(item)

    def _smart_position_meta(self):
        """智能定位元数据浮窗，确保完全在屏幕内"""
        from PyQt6.QtGui import QGuiApplication
    
        # 目标位置：画廊右侧
        anchor = self.mapToGlobal(self.rect().topRight())
        x = anchor.x() + 10
        y = anchor.y()
    
        # 获取屏幕可用区域
        screen = QGuiApplication.screenAt(anchor) or QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
    
        pw = self.meta_panel.width() or 380
        ph = self.meta_panel.height() or 520
    
        # 右边放不下 → 改放画廊左边
        if x + pw > geo.right():
            left_anchor = self.mapToGlobal(self.rect().topLeft())
            x = left_anchor.x() - pw - 10
    
        # 还是不行 → 强制贴右边
        if x < geo.left():
            x = geo.right() - pw - 20
        if x < geo.left():
            x = geo.left() + 20
    
        # 垂直方向越界
        if y + ph > geo.bottom():
            y = geo.bottom() - ph - 20
        if y < geo.top():
            y = geo.top() + 20
    
        self.meta_panel.move(x, y)


    def _toggle_meta_panel(self):
        if self.meta_panel.isVisible():
            self.meta_panel.hide()
        else:
            self._smart_position_meta()
            self.meta_panel.show()


    def _show_meta_for(self, path: str):
        self.meta_panel.load_from_path(path)
        if not self.meta_panel.isVisible():
            self._smart_position_meta()
            self.meta_panel.show()
        self.meta_panel.raise_()

    # ========== 事件处理 ==========
    def _on_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        self.image_selected.emit(path)
        self.meta_panel.load_from_path(path)

    def _on_double_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.exists(path):
            dlg = ImageViewerDialog(path, self)
            dlg.exec()

    # ========== 右键菜单 (单选/多选自动切换) ==========
    def _show_menu(self, pos: QPoint):
        selected = self.list_widget.selectedItems()
        if not selected:
            item = self.list_widget.itemAt(pos)
            if not item:
                return
            selected = [item]

        selected_paths = [it.data(Qt.ItemDataRole.UserRole) for it in selected]
        selected_paths = [p for p in selected_paths if p]
        if not selected_paths:
            return

        is_multi = len(selected_paths) > 1
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background:#1e1e2e; color:#cdd6f4; border:1px solid #45475a; padding:4px; }
            QMenu::item { padding:6px 24px; }
            QMenu::item:selected { background:#45475a; }
            QMenu::item:disabled { color:#6c7086; }
            QMenu::separator { height:1px; background:#45475a; margin:4px 8px; }
        """)

        # ─────── 多选菜单 ───────
        if is_multi:
            menu.addAction(f"📦 已选 {len(selected_paths)} 张").setEnabled(False)
            menu.addSeparator()
            act_batch_fav    = menu.addAction("⭐ 批量切换收藏")
            act_batch_export = menu.addAction("📤 批量导出到…")
            menu.addSeparator()
            act_batch_remove = menu.addAction("🗑 从画廊移除")
            act_batch_delete = menu.addAction("❌ 删除文件 (不可恢复)")

            chosen = menu.exec(self.list_widget.viewport().mapToGlobal(pos))
            if chosen == act_batch_fav:
                self._toggle_fav(selected_paths)
            elif chosen == act_batch_export:
                self._batch_export(selected_paths)
            elif chosen == act_batch_remove:
                self._remove_from_view(selected_paths)
            elif chosen == act_batch_delete:
                self._batch_delete_files(selected_paths)
            return

        # ─────── 单选菜单 ───────
        path = selected_paths[0]
        act_open   = menu.addAction("🖼️ 大图查看")
        act_edit   = menu.addAction("✏️ 载入预览/发送到编辑")
        menu.addSeparator()
        act_reuse  = menu.addAction("🔁 复用参数")
        act_i2i    = menu.addAction("🛠 发送到 img2img")
        act_face   = menu.addAction("😀 发送到修脸")
        menu.addSeparator()
        is_fav = os.path.abspath(path) in self._favs
        act_fav    = menu.addAction("💔 取消收藏" if is_fav else "⭐ 加入收藏")
        act_copy_p = menu.addAction("📋 复制 Prompt")
        act_meta   = menu.addAction("📋 查看元数据")
        act_folder = menu.addAction("📁 打开所在文件夹")
        menu.addSeparator()
        act_remove = menu.addAction("🗑 从画廊移除")
        act_del    = menu.addAction("❌ 删除文件")

        chosen = menu.exec(self.list_widget.viewport().mapToGlobal(pos))
        if chosen == act_open:
            self._on_double_clicked(selected[0])
        elif chosen == act_edit:
            self.image_selected.emit(path)
        elif chosen == act_reuse:
            self.reuse_params_signal.emit(path)
        elif chosen == act_i2i:
            self.send_to_i2i_signal.emit(path)
        elif chosen == act_face:
            self.send_to_face_signal.emit(path)
        elif chosen == act_fav:
            self._toggle_fav([path])
        elif chosen == act_copy_p:
            self._copy_prompt(path)
        elif chosen == act_meta:
            self._show_meta_for(path)
        elif chosen == act_folder:
            self._open_folder(path)
        elif chosen == act_remove:
            self._remove_from_view([path])
        elif chosen == act_del:
            self._batch_delete_files([path])

    # ========== 收藏切换 ==========
    def _toggle_fav(self, paths):
        n_add, n_del = 0, 0
        for p in paths:
            ap = os.path.abspath(p)
            if ap in self._favs:
                self._favs.discard(ap); n_del += 1
            else:
                self._favs.add(ap); n_add += 1
        self._save_favs()
        self._apply_filter()  # 刷新 ⭐ 前缀
        QMessageBox.information(
            self, "收藏",
            f"⭐ 新增 {n_add} 张 | 💔 取消 {n_del} 张\n共 {len(self._favs)} 张收藏"
        )

    # ========== 批量导出 ==========
    def _batch_export(self, paths):
        dst_dir = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not dst_dir:
            return
        ok, fail = 0, 0
        for p in paths:
            try:
                shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))
                ok += 1
            except Exception as e:
                print(f"⚠️ 导出失败 {p}: {e}")
                fail += 1
        QMessageBox.information(
            self, "批量导出",
            f"✅ 成功 {ok} 张\n❌ 失败 {fail} 张\n📁 目录: {dst_dir}"
        )

    # ========== 从画廊移除 (不删文件) ==========
    def _remove_from_view(self, paths):
        paths_set = {os.path.abspath(p) for p in paths}
        self._all_items = [
            (p, t, n) for p, t, n in self._all_items
            if os.path.abspath(p) not in paths_set
        ]
        self._apply_filter()

    # ========== 批量删除文件 ==========
    def _batch_delete_files(self, paths):
        ret = QMessageBox.warning(
            self, "确认删除",
            f"⚠️ 即将永久删除 {len(paths)} 个文件！\n此操作不可恢复，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        ok, fail = 0, 0
        deleted = []
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
                deleted.append(p)
                ok += 1
            except Exception as e:
                print(f"⚠️ 删除失败 {p}: {e}")
                fail += 1
        self._remove_from_view(deleted)
        # 同步清空收藏中已删除的条目
        for p in deleted:
            self._favs.discard(os.path.abspath(p))
            self.image_deleted.emit(p)
        self._save_favs()
        self.meta_panel.clear()
        QMessageBox.information(self, "删除完成", f"✅ 已删除 {ok} 张\n❌ 失败 {fail} 张")

    # ========== 复制 Prompt ==========
    def _copy_prompt(self, path: str):
        try:
            from PIL import Image
            img = Image.open(path)
            raw = img.info.get("parameters") or img.info.get("meta", "")
            if not raw:
                QMessageBox.information(self, "提示", "图片中没有 prompt 信息")
                return
            try:
                meta = json.loads(raw)
                prompt = meta.get("prompt", "") or meta.get("p", "")
            except Exception:
                prompt = raw.split("Negative prompt:")[0].strip()
            if not prompt:
                QMessageBox.information(self, "提示", "未找到 prompt")
                return
            QApplication.clipboard().setText(prompt)
            preview = prompt[:200] + ("..." if len(prompt) > 200 else "")
            QMessageBox.information(self, "✅ 已复制", f"Prompt 已复制到剪贴板\n\n{preview}")
        except Exception as e:
            QMessageBox.warning(self, "失败", f"读取失败: {e}")

    # ========== 文件夹 ==========
    def _open_folder(self, path: str):
        try:
            if sys.platform == "win32":
                subprocess.Popen(['explorer', '/select,', os.path.normpath(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(['open', '-R', path])
            else:
                subprocess.Popen(['xdg-open', os.path.dirname(path)])
        except Exception as e:
            print(f"⚠️ 打开文件夹失败: {e}")

    def closeEvent(self, event):
        if hasattr(self, 'meta_panel'):
            self.meta_panel.close()
        super().closeEvent(event)