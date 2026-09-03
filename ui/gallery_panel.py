# ui/gallery_panel.py
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
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QPoint, QTimer
from PyQt6.QtGui import QPixmap, QIcon, QWheelEvent

from utils.paths import DATA_DIR
import logging

logger = logging.getLogger(__name__)

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
        self.lbl_zoom.setProperty("role", "value")
        for b in (btn_zoom_in, btn_zoom_out, btn_fit, btn_actual):
            b.setStyleSheet("padding:6px 12px;")
            toolbar.addWidget(b)
        toolbar.addWidget(self.lbl_zoom)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        from ui.theme import PALETTE
        self.scroll.setStyleSheet(f"background:{PALETTE['bg']};")
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
        title.setProperty("role", "title")
        layout.addWidget(title)

        lbl_p = QLabel("Prompt:")
        lbl_p.setProperty("role", "value")
        layout.addWidget(lbl_p)
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setReadOnly(True)
        self.txt_prompt.setFixedHeight(80)
        layout.addWidget(self.txt_prompt)

        lbl_n = QLabel("Negative:")
        lbl_n.setProperty("role", "hint")
        layout.addWidget(lbl_n)
        self.txt_neg = QTextEdit()
        self.txt_neg.setReadOnly(True)
        self.txt_neg.setFixedHeight(60)
        layout.addWidget(self.txt_neg)

        lbl_param = QLabel("参数:")
        lbl_param.setProperty("role", "hint")
        layout.addWidget(lbl_param)
        self.lbl_params = QLabel()
        self.lbl_params.setWordWrap(True)
        self.lbl_params.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.lbl_params.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.lbl_params.setProperty("role", "hint")
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
    send_to_editor_signal = pyqtSignal(str) # ✏️ 载入预览并打开修图编辑器
    video_selected      = pyqtSignal(str)   # ▶️ 双击视频条目（交由宿主播放）
    items_changed       = pyqtSignal()      # 🔄 可见列表重建完成（供胶片条联动）

    IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    VIDEO_EXT = {".mp4", ".gif", ".webm", ".mov"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_items = []
        self._media_filter = "all"          # all / image / video

        # 批量添加防洪：200ms 防抖合并刷新（X-Y 矩阵一次几十张不疯狂重排）
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(200)
        self._refresh_timer.timeout.connect(self._apply_filter)

        # 收藏持久化
        self._favs_path = os.path.join(DATA_DIR, "gallery_favs.json")
        self._favs = self._load_favs()

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── 搜索栏 ──
        search_row = QHBoxLayout()
        search_row.setSpacing(4)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 搜索文件名或 prompt 关键字...")
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)   # 防抖 300ms
        self._search_timer.timeout.connect(self._apply_filter)
        self.search_box.textChanged.connect(self._search_timer.start)

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
        self.lbl_count.setProperty("role", "value")

        # 排序 + 时间过滤
        from PyQt6.QtWidgets import QComboBox
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["最新优先", "最旧优先", "按文件名"])
        self.combo_sort.setToolTip("排序方式")
        self.combo_sort.currentIndexChanged.connect(self._apply_filter)
        self.combo_time = QComboBox()
        self.combo_time.addItems(["全部时间", "今天", "近 7 天", "近 30 天"])
        self.combo_time.setToolTip("按修改时间过滤")
        self.combo_time.currentIndexChanged.connect(self._apply_filter)

        search_row.addWidget(self.search_box, 1)
        search_row.addWidget(self.btn_clear_search)
        search_row.addWidget(self.combo_time)
        search_row.addWidget(self.combo_sort)
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

        # ── 悬浮大图预览（hover 500ms 触发，跟随鼠标的浮层）──
        self.list_widget.setMouseTracking(True)
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(500)
        self._hover_timer.timeout.connect(self._show_hover_preview)
        self._hover_path = None
        self._hover_label = None
        self.list_widget.itemEntered.connect(self._on_item_entered)
        self.list_widget.viewport().installEventFilter(self)

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
            logger.warning(f"⚠️ 加载收藏失败: {e}")
        return set()

    def _save_favs(self):
        try:
            with open(self._favs_path, "w", encoding="utf-8") as f:
                json.dump(sorted(self._favs), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ 保存收藏失败: {e}")

    # ========== 数据管理 ==========
    @classmethod
    def media_kind(cls, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in cls.VIDEO_EXT:
            return "video"
        if ext in cls.IMAGE_EXT:
            return "image"
        return "other"

    def set_media_filter(self, mode: str):
        """切换媒体类型过滤：all / image / video。"""
        assert mode in ("all", "image", "video"), f"未知媒体过滤模式: {mode}"
        self._media_filter = mode
        self._apply_filter()

    def add_media(self, path: str, prepend: bool = False):
        """统一入口：图片/视频都走这里。防抖刷新，批量添加不逐张重排。"""
        if not os.path.exists(path):
            return
        # 去重
        for p, _, _ in self._all_items:
            if os.path.abspath(p) == os.path.abspath(path):
                return
        if self.media_kind(path) == "video":
            prompt_text, nsfw = "", False
        else:
            prompt_text = self._extract_prompt(path)
            nsfw = is_nsfw_prompt(prompt_text)
        if prepend:
            self._all_items.insert(0, (path, prompt_text, nsfw))
        else:
            self._all_items.append((path, prompt_text, nsfw))
        self._refresh_timer.start()   # 200ms 合并刷新

    def add_image(self, path: str, prepend: bool = False):
        """兼容包装：保持旧行为（不存在路径静默 return）。"""
        self.add_media(path, prepend)

    def reload_from_dir(self, directory: str, limit: int = 80):
        self._all_items.clear()
        if not os.path.isdir(directory):
            return
        files = []
        for f in os.listdir(directory):
            if os.path.splitext(f)[1].lower() in self.IMAGE_EXT:
                full = os.path.join(directory, f)
                files.append((full, os.path.getmtime(full)))
        # 统一画廊：同时扫描 videos 子目录
        vdir = os.path.join(directory, "videos")
        if os.path.isdir(vdir):
            for f in os.listdir(vdir):
                if os.path.splitext(f)[1].lower() in self.VIDEO_EXT:
                    full = os.path.join(vdir, f)
                    files.append((full, os.path.getmtime(full)))
        files.sort(key=lambda x: -x[1])
        for path, _ in files[:limit]:
            if self.media_kind(path) == "video":
                prompt_text, nsfw = "", False
            else:
                prompt_text = self._extract_prompt(path)
                nsfw = is_nsfw_prompt(prompt_text)
            self._all_items.append((path, prompt_text, nsfw))
        self._apply_filter()

    @staticmethod
    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

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
        import time as _time
        keyword = self.search_box.text().strip().lower()
        only_fav = self.btn_only_fav.isChecked()
        show_nsfw = self.btn_show_nsfw.isChecked()
        media = self._media_filter
        time_idx = self.combo_time.currentIndex() if hasattr(self, "combo_time") else 0
        sort_idx = self.combo_sort.currentIndex() if hasattr(self, "combo_sort") else 0

        # 时间过滤阈值
        cutoff = None
        if time_idx == 1:
            cutoff = _time.time() - 86400
        elif time_idx == 2:
            cutoff = _time.time() - 7 * 86400
        elif time_idx == 3:
            cutoff = _time.time() - 30 * 86400

        items = list(self._all_items)
        # 排序
        if sort_idx == 0:
            items.sort(key=lambda it: -self._mtime(it[0]))
        elif sort_idx == 1:
            items.sort(key=lambda it: self._mtime(it[0]))
        elif sort_idx == 2:
            items.sort(key=lambda it: os.path.basename(it[0]).lower())

        self.list_widget.clear()
        shown = 0
        nsfw_hidden = 0
        for path, prompt_text, nsfw in items:
            if media != "all" and self.media_kind(path) != media:
                continue
            if cutoff is not None and self._mtime(path) < cutoff:
                continue
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
        self.items_changed.emit()

    _video_thumb_cache = None

    @classmethod
    def _video_placeholder_icon(cls) -> QIcon:
        """视频占位缩略图：纯色底 + ▶ 标记。"""
        if cls._video_thumb_cache is None:
            from PyQt6.QtGui import QColor, QPainter
            pix = QPixmap(110, 110)
            pix.fill(QColor("#2a2a3c"))
            p = QPainter(pix)
            p.setPen(QColor("#cdd6f4"))
            f = p.font(); f.setPointSize(28); p.setFont(f)
            p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "▶")
            p.end()
            cls._video_thumb_cache = QIcon(pix)
        return cls._video_thumb_cache

    @staticmethod
    def video_frame_icon(path: str, size: int = 110) -> QIcon | None:
        """cv2 抽视频首帧做缩略图，右下角画时长角标。失败返回 None（调用方回落占位图）。"""
        try:
            import cv2
            from PyQt6.QtGui import QImage, QPainter, QColor
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            if not ret:
                return None
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w,
                          QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg).scaled(
                size, size,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation)
            # 居中裁剪成方形
            if pix.width() > size or pix.height() > size:
                x = max(0, (pix.width() - size) // 2)
                y = max(0, (pix.height() - size) // 2)
                pix = pix.copy(x, y, min(size, pix.width()), min(size, pix.height()))
            # 时长角标
            if fps > 0 and frames > 0:
                secs = frames / fps
                label = f"{int(secs // 60)}:{int(secs % 60):02d}"
                p = QPainter(pix)
                f = p.font(); f.setPointSize(9); f.setBold(True); p.setFont(f)
                tw = p.fontMetrics().horizontalAdvance(label) + 8
                p.fillRect(pix.width() - tw - 2, pix.height() - 18, tw, 16,
                           QColor(0, 0, 0, 160))
                p.setPen(QColor("#ffffff"))
                p.drawText(pix.width() - tw, pix.height() - 5, label)
                # 左下 ▶ 标记
                p.setPen(QColor(255, 255, 255, 200))
                f.setPointSize(12); p.setFont(f)
                p.drawText(6, pix.height() - 8, "▶")
                p.end()
            return QIcon(pix)
        except Exception as e:
            logger.debug(f"视频抽帧失败 {path}: {e}")
            return None

    def _add_to_list(self, path: str):
        item = QListWidgetItem()
        # 缩略图缓存：同一路径只解码/缩放一次
        if not hasattr(self, '_thumb_cache'):
            self._thumb_cache = {}
        icon = self._thumb_cache.get(path)
        if icon is None:
            if self.media_kind(path) == "video":
                icon = self.video_frame_icon(path) or self._video_placeholder_icon()
                self._thumb_cache[path] = icon
            else:
                pix = QPixmap(path)
                if not pix.isNull():
                    icon = QIcon(pix.scaled(
                        110, 110,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    ))
                    self._thumb_cache[path] = icon
        if icon is not None:
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

    # ========== 悬浮大图预览 ==========
    def _on_item_entered(self, item):
        path = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not path or self.media_kind(path) == "video":
            self._hover_timer.stop()
            self._hide_hover_preview()
            return
        self._hover_path = path
        self._hover_timer.start()

    def _show_hover_preview(self):
        path = self._hover_path
        if not path or not os.path.exists(path):
            return
        pix = QPixmap(path)
        if pix.isNull():
            return
        if self._hover_label is None:
            self._hover_label = QLabel(None, Qt.WindowType.ToolTip)
        # 限制浮层最大边长 512
        scaled = pix.scaled(512, 512,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self._hover_label.setPixmap(scaled)
        self._hover_label.resize(scaled.size())
        from PyQt6.QtGui import QCursor, QGuiApplication
        pos = QCursor.pos()
        screen = QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = pos.x() + 24
        y = pos.y() + 24
        if x + scaled.width() > geo.right():
            x = pos.x() - scaled.width() - 16
        if y + scaled.height() > geo.bottom():
            y = geo.bottom() - scaled.height() - 8
        self._hover_label.move(x, y)
        self._hover_label.show()

    def _hide_hover_preview(self):
        if self._hover_label is not None:
            self._hover_label.hide()

    def leaveEvent(self, event):
        self._hover_timer.stop()
        self._hide_hover_preview()
        super().leaveEvent(event)

    def eventFilter(self, obj, event):
        # 鼠标离开列表视口（如移到滚动条/空白间隙）时收起悬浮预览
        from PyQt6.QtCore import QEvent
        if (obj is self.list_widget.viewport()
                and event.type() == QEvent.Type.Leave):
            self._hover_timer.stop()
            self._hide_hover_preview()
        return super().eventFilter(obj, event)

    # ========== 事件处理 ==========
    def _on_clicked(self, item):
        self._hover_timer.stop()
        self._hide_hover_preview()
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        self.image_selected.emit(path)
        self.meta_panel.load_from_path(path)

    def _on_double_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not os.path.exists(path):
            return
        if self.media_kind(path) == "video":
            self.video_selected.emit(path)   # 交给宿主播放（跳动画页/播放）
            return
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

        # 视频条目：简化菜单（播放/文件夹/收藏/移除/删除）
        if self.media_kind(path) == "video":
            act_play   = menu.addAction("▶️ 播放")
            is_fav = os.path.abspath(path) in self._favs
            act_fav    = menu.addAction("💔 取消收藏" if is_fav else "⭐ 加入收藏")
            act_folder = menu.addAction("📁 打开所在文件夹")
            menu.addSeparator()
            act_remove = menu.addAction("🗑 从画廊移除")
            act_del    = menu.addAction("❌ 删除文件")

            chosen = menu.exec(self.list_widget.viewport().mapToGlobal(pos))
            if chosen == act_play:
                self.video_selected.emit(path)
            elif chosen == act_fav:
                self._toggle_fav([path])
            elif chosen == act_folder:
                self._open_folder(path)
            elif chosen == act_remove:
                self._remove_from_view([path])
            elif chosen == act_del:
                self._batch_delete_files([path])
            return

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
            self.send_to_editor_signal.emit(path)
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
                logger.warning(f"⚠️ 导出失败 {p}: {e}")
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
                logger.warning(f"⚠️ 删除失败 {p}: {e}")
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
            logger.warning(f"⚠️ 打开文件夹失败: {e}")

    def closeEvent(self, event):
        self._hover_timer.stop()
        self._hide_hover_preview()
        if hasattr(self, 'meta_panel'):
            self.meta_panel.close()
        super().closeEvent(event)