# ui/pages/characters_page.py
"""角色库页面 — 遵循 PAGES Page 协议（非 QWidget 子类，由 shell 托管）"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import QEvent 
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QPushButton, QLineEdit, QTextEdit, QScrollArea, QFrame,
    QFileDialog, QMessageBox, QSlider, QGroupBox, QComboBox,
    QSizePolicy,
)
from PyQt6.QtCore import QObject
from utils.character_library import CharacterLibrary, Character
from utils.scene_library import SceneLibrary
from utils.batch_runner import BatchRunner
from ui.i18n import ui_i18n
logger = logging.getLogger(__name__)

# ── 主题色兼容层：design_tokens 可能是 dict 也可能是对象 ──
try:
    from ui.design_tokens import COLORS as _TOKENS
except Exception:
    _TOKENS = {}


def C(key: str, fallback: str) -> str:
    if isinstance(_TOKENS, dict):
        return _TOKENS.get(key, fallback)
    return getattr(_TOKENS, key, fallback) or fallback

BG      = C("mantle",   "#181825")
SURF0   = C("surface0", "#313244")
SURF1   = C("surface1", "#45475a")
TEXT    = C("text",     "#cdd6f4")
SUB     = C("subtext0", "#a6adc8")
BLUE    = C("blue",     "#89b4fa")
GREEN   = C("green",    "#a6e3a1")
RED     = C("red",      "#f38ba8")
YELLOW  = C("yellow",   "#f9e2af")

CARD_W, CARD_H = 180, 292
GRID_GAP = 12


# ═══════════════════════════════════════════════════════════
#  角色卡片
# ═══════════════════════════════════════════════════════════
class CharacterCard(QFrame):   # ← 改名：原本错写成 CharactersPage
    selected      = pyqtSignal(str)
    apply_request = pyqtSignal(str)
    delete_request = pyqtSignal(str)

    def __init__(self, character: Character, parent=None):
        super().__init__(parent)
        self.character = character
        self._active = False
        self.setFixedSize(CARD_W, CARD_H)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._restyle()
        self._build()


    def _status_text(self) -> str:
        c = self.character
        bits = []
        bits.append("🖼✓" if c.ref_image_path and Path(c.ref_image_path).exists() else "🖼✗")
        bits.append("LoRA✓" if c.lora_path else "LoRA✗")
        bits.append("CN✓" if c.controlnet_type else "CN✗")
        return "  ".join(bits)

    def _tags_text(self) -> str:
        tags = getattr(self.character, "tags", None) or []
        if not tags:
            return ""
        shown = [f"#{t}" for t in tags[:3]]
        more = f" +{len(tags) - 3}" if len(tags) > 3 else ""
        return " ".join(shown) + more

    def _restyle(self):
        border = BLUE if self._active else SURF1
        self.setStyleSheet(
            f"CharacterCard {{ background:{SURF0}; border:2px solid {border};"
            f" border-radius:8px; }}"
        )

    def set_active(self, on: bool):
        self._active = on
        self._restyle()

    def enterEvent(self, e):
        if not self._active:
            self.setStyleSheet(
                f"CharacterCard {{ background:{SURF1}; border:2px solid {BLUE};"
                f" border-radius:8px; }}"
            )
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._restyle()
        super().leaveEvent(e)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        thumb = QLabel()
        thumb.setFixedSize(CARD_W - 20, 160)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setStyleSheet(f"background:{BG}; border-radius:4px; color:{SUB};")
        thumb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._load_thumb(thumb)
        lay.addWidget(thumb)

        name = QLabel(self.character.name or "未命名")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setWordWrap(True)
        name.setFixedHeight(34)
        name.setStyleSheet(f"color:{TEXT}; font-weight:bold; font-size:13px;")
        name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay.addWidget(name)

        status = QLabel(self._status_text())
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status.setFixedHeight(16)
        status.setStyleSheet(f"color:{SUB}; font-size:10px;")
        status.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay.addWidget(status)

        tags = QLabel(self._tags_text())
        tags.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tags.setFixedHeight(16)
        tags.setStyleSheet(f"color:{GREEN}; font-size:10px;")
        tags.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay.addWidget(tags)

        row = QHBoxLayout()
        row.setSpacing(4)
        row.addStretch()
        btn_del = QPushButton("🗑")
        btn_del.setFixedSize(32, 24)
        btn_del.setToolTip("删除角色")
        btn_del.clicked.connect(lambda: self.delete_request.emit(self.character.uuid))
        row.addWidget(btn_del)
        row.addStretch()
        lay.addLayout(row)

    def _load_thumb(self, label: QLabel):
        path = self.character.thumbnail_path or self.character.ref_image_path
        if not path or not Path(path).exists():
            label.setText("无预览")
            return
        pm = QPixmap(path)
        if pm.isNull():
            label.setText("无预览")
            return
        label.setPixmap(pm.scaled(
            CARD_W - 20, 160,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.character.uuid)
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.apply_request.emit(self.character.uuid)
        super().mouseDoubleClickEvent(e)


# ═══════════════════════════════════════════════════════════
#  角色编辑器
# ═══════════════════════════════════════════════════════════
class CharacterEditor(QWidget):
    batch_request = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.char: Optional[Character] = None
        self._build()
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        head = QLabel("角色详情")
        head.setStyleSheet(f"color:{BLUE}; font-size:15px; font-weight:bold;")
        root.addWidget(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setSpacing(10)

        # —— 基础 ——
        g1 = QGroupBox("基础信息")
        l1 = QVBoxLayout(g1)

        r = QHBoxLayout(); r.addWidget(QLabel("名称:"))
        self.in_name = QLineEdit(); self.in_name.setPlaceholderText("角色名称")
        r.addWidget(self.in_name, 1); l1.addLayout(r)

        r = QHBoxLayout(); r.addWidget(QLabel("参考图:"))
        self.lbl_ref = QLabel("未选择")
        self.lbl_ref.setStyleSheet(f"color:{SUB};")
        self.lbl_ref.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        r.addWidget(self.lbl_ref, 1)
        b = QPushButton("📁"); b.setFixedWidth(36)
        b.clicked.connect(self.browse_ref); r.addWidget(b)
        l1.addLayout(r)

        self.lbl_preview = QLabel("无预览")
        self.lbl_preview.setFixedHeight(180)
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet(f"background:{BG}; border-radius:4px; color:{SUB};")
        l1.addWidget(self.lbl_preview)

        r = QHBoxLayout(); r.addWidget(QLabel("标签:"))
        self.in_tags = QLineEdit()
        self.in_tags.setPlaceholderText("逗号分隔，如: 日常, 泳装, 夜景")
        r.addWidget(self.in_tags, 1)
        l1.addLayout(r)
        v.addWidget(g1)

        # —— 批量出图 ——
        self.btn_batch = QPushButton("🎬 批量出图")
        self.btn_batch.setEnabled(False)
        self.btn_batch.clicked.connect(self.batch_request.emit)
        v.addWidget(self.btn_batch)

        # —— LoRA ——
        g2 = QGroupBox("LoRA（可选）")
        l2 = QVBoxLayout(g2)
        r = QHBoxLayout(); r.addWidget(QLabel("文件:"))
        self.lbl_lora = QLabel("未选择")
        self.lbl_lora.setStyleSheet(f"color:{SUB};")
        self.lbl_lora.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        r.addWidget(self.lbl_lora, 1)
        b = QPushButton("📁"); b.setFixedWidth(36)
        b.clicked.connect(self.browse_lora); r.addWidget(b)
        l2.addLayout(r)

        self.sl_lora, self.lbl_lora_w = self._scale_row(l2, "权重:", 75)
        v.addWidget(g2)

        # —— IP-Adapter ——
        g3 = QGroupBox("IP-Adapter 强度")
        l3 = QVBoxLayout(g3)
        self.sl_ipa, self.lbl_ipa = self._scale_row(l3, "强度:", 60)
        v.addWidget(g3)

        # —— ControlNet ——
        g4 = QGroupBox("ControlNet（可选）")
        l4 = QVBoxLayout(g4)
        r = QHBoxLayout(); r.addWidget(QLabel("类型:"))
        self.cb_cn = QComboBox()
        self.cb_cn.addItems(["", "openpose", "canny", "depth", "lineart", "scribble", "normal"])
        r.addWidget(self.cb_cn, 1); l4.addLayout(r)

        r = QHBoxLayout(); r.addWidget(QLabel("图像:"))
        self.lbl_cn = QLabel("未选择")
        self.lbl_cn.setStyleSheet(f"color:{SUB};")
        self.lbl_cn.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        r.addWidget(self.lbl_cn, 1)
        b = QPushButton("📁"); b.setFixedWidth(36)
        b.clicked.connect(self.browse_cn); r.addWidget(b)
        l4.addLayout(r)

        self.sl_cn, self.lbl_cn_s = self._scale_row(l4, "强度:", 100)
        v.addWidget(g4)

        # —— 提示词 ——
        g5 = QGroupBox("提示词")
        l5 = QVBoxLayout(g5)
        l5.addWidget(QLabel("正面（角色特征）:"))
        self.in_prompt = QTextEdit(); self.in_prompt.setMaximumHeight(80)
        self.in_prompt.setPlaceholderText("1girl, long silver hair, blue eyes...")
        l5.addWidget(self.in_prompt)
        l5.addWidget(QLabel("负面:"))
        self.in_neg = QTextEdit(); self.in_neg.setMaximumHeight(60)
        self.in_neg.setPlaceholderText("blurry, low quality...")
        l5.addWidget(self.in_neg)
        l5.addWidget(QLabel("备注:"))
        self.in_notes = QTextEdit(); self.in_notes.setMaximumHeight(56)
        l5.addWidget(self.in_notes)
        v.addWidget(g5)

        v.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

    def _scale_row(self, parent_layout, label: str, default: int):
        r = QHBoxLayout()
        r.addWidget(QLabel(label))
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(0, 200)
        sl.setValue(default)
        r.addWidget(sl, 1)
        val = QLabel(f"{default / 100:.2f}")
        val.setFixedWidth(40)
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        sl.valueChanged.connect(lambda x: val.setText(f"{x / 100:.2f}"))
        r.addWidget(val)
        parent_layout.addLayout(r)
        return sl, val

    def new_character(self) -> Character:
        self.char = Character(name="新角色")
        self.load(self.char)
        self.in_name.setFocus()
        self.in_name.selectAll()
        return self.char

    def load(self, char: Character):
        self.char = char
        self.in_name.setText(char.name)
        self.in_tags.setText(", ".join(char.tags or []))
        self._set_path_label(self.lbl_ref, char.ref_image_path)
        self._set_path_label(self.lbl_lora, char.lora_path)
        self._set_path_label(self.lbl_cn, char.controlnet_image_path)
        self._show_preview(char.ref_image_path)
        self.sl_lora.setValue(int(round(char.lora_weight * 100)))
        self.sl_ipa.setValue(int(round(char.ip_adapter_scale * 100)))
        self.sl_cn.setValue(int(round(char.controlnet_scale * 100)))
        self.cb_cn.setCurrentText(char.controlnet_type or "")
        self.in_prompt.setPlainText(char.base_prompt or "")
        self.in_neg.setPlainText(char.negative_prompt or "")
        self.in_notes.setPlainText(char.notes or "")

    def collect(self) -> Optional[Character]:
        if self.char is None:
            return None
        c = self.char
        c.name = self.in_name.text().strip() or "未命名角色"
        c.tags = [t.strip() for t in self.in_tags.text().split(",") if t.strip()]
        c.lora_weight      = self.sl_lora.value() / 100
        c.ip_adapter_scale = self.sl_ipa.value() / 100
        c.controlnet_type  = self.cb_cn.currentText().strip()
        c.controlnet_scale = self.sl_cn.value() / 100
        c.base_prompt      = self.in_prompt.toPlainText().strip()
        c.negative_prompt  = self.in_neg.toPlainText().strip()
        c.notes            = self.in_notes.toPlainText().strip()
        return c

    @staticmethod
    def _set_path_label(label: QLabel, path: str):
        if path and Path(path).exists():
            label.setText(Path(path).name)
            label.setToolTip(path)
        else:
            label.setText("未选择")
            label.setToolTip("")

    def _show_preview(self, path: str):
        if not path or not Path(path).exists():
            self.lbl_preview.clear()
            self.lbl_preview.setText("无预览")
            return
        pm = QPixmap(path)
        if pm.isNull():
            self.lbl_preview.clear()
            self.lbl_preview.setText("无法读取图片")
            return
        self.lbl_preview.setPixmap(pm.scaled(
            360, 180,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def browse_ref(self):
        if self.char is None:
            self.new_character()
        p, _ = QFileDialog.getOpenFileName(
            self, "选择参考图", "", "图像 (*.png *.jpg *.jpeg *.webp *.bmp)")
        if p:
            self.char.ref_image_path = p
            self.char.thumbnail_path = ""
            self._set_path_label(self.lbl_ref, p)
            self._show_preview(p)

    def browse_lora(self):
        if self.char is None:
            self.new_character()
        p, _ = QFileDialog.getOpenFileName(
            self, "选择 LoRA", "models/lora", "LoRA (*.safetensors *.pt *.bin)")
        if p:
            self.char.lora_path = p
            self._set_path_label(self.lbl_lora, p)

    def browse_cn(self):
        if self.char is None:
            self.new_character()
        p, _ = QFileDialog.getOpenFileName(
            self, "选择 ControlNet 图像", "", "图像 (*.png *.jpg *.jpeg *.webp)")
        if p:
            self.char.controlnet_image_path = p
            self._set_path_label(self.lbl_cn, p)


# ═══════════════════════════════════════════════════════════
#  角色库页面（Page 协议）
# ═══════════════════════════════════════════════════════════
class CharactersPage(QObject): 
    page_id, title, icon = "characters", "角色库", "🧍"

    def __init__(self):
        super().__init__()
        self._shell = None
        self._ws = None
        self._params = None
        self.library = CharacterLibrary()
        self.scenes = SceneLibrary()        
        self._batch_runner = None           
        self._cards = []
        self._selected_uuid = None
        self._last_cols = 0
        self.grid_scroll = None

    def build(self, shell):
        self._shell = shell
        self._ws = self._build_workspace()
        self._params = self._build_params()
        self._character_count = 0
        self.refresh()
        ui_i18n.language_changed.connect(self._render_character_count)

    def _render_character_count(self, *_):
        """角色数量使用结构化数值，避免语言切换后遗留旧文案。"""
        count = getattr(self, "_character_count", 0)
        if ui_i18n.language == "en":
            text = f"{count} character" + ("" if count == 1 else "s")
        elif ui_i18n.language == "ja":
            text = f"{count} キャラクター"
        else:
            text = f"{count} 个角色"
        self.lbl_count.setText(text)

    def dispose(self):
        """页面销毁时调用。shell 重建页面时必须走到这里。"""
        tgt = getattr(self, "_filter_target", None)
        if tgt is not None:
            try:
                tgt.removeEventFilter(self)
            except RuntimeError:
                pass
        self._filter_target = None
        for name in ("grid_scroll", "grid_host", "grid_layout"):
            setattr(self, name, None)

    def workspace(self) -> QWidget:
        if self._ws is None:
            self._ws = self._build_workspace()
        return self._ws

    def params_widget(self) -> Optional[QWidget]:
        return self._params

    def _build_workspace(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        bar = QWidget()
        bar.setStyleSheet(f"background:{BG};")
        hb = QHBoxLayout(bar)
        hb.setContentsMargins(12, 8, 12, 8)

        t = QLabel("🎭 角色库")
        t.setStyleSheet(f"color:{TEXT}; font-size:17px; font-weight:bold;")
        hb.addWidget(t)
        self.lbl_count = QLabel("0 个角色")
        self.lbl_count.setStyleSheet(f"color:{SUB}; font-size:12px; margin-left:8px;")
        hb.addWidget(self.lbl_count)
        hb.addStretch()

        self.in_search = QLineEdit()
        self.in_search.setPlaceholderText("🔍 搜索名称 / 提示词 / 标签")
        self.in_search.setFixedWidth(220)
        self.in_search.textChanged.connect(lambda _: self.refresh())
        hb.addWidget(self.in_search)

        self.cbo_tag = QComboBox()
        self.cbo_tag.setFixedWidth(130)
        self.cbo_tag.currentIndexChanged.connect(lambda _: self.refresh())
        hb.addWidget(self.cbo_tag)
        v.addWidget(bar)

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setChildrenCollapsible(False)

        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        self.grid = QGridLayout(holder)
        self.grid.setSpacing(GRID_GAP)
        self.grid.setContentsMargins(12, 12, 12, 12)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.grid_scroll.setWidget(holder)
        sp.addWidget(self.grid_scroll)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.editor = CharacterEditor()
        self.editor.batch_request.connect(self.open_batch)
        rv.addWidget(self.editor, 1)

        srow = QHBoxLayout()
        srow.setContentsMargins(10, 6, 10, 10)
        srow.addStretch()
        self.btn_save = QPushButton("💾 保存角色")
        self.btn_save.setStyleSheet(
            f"QPushButton {{ background:{GREEN}; color:#11111b; font-weight:bold;"
            f" padding:8px 18px; border-radius:4px; }}"
        )
        self.btn_save.clicked.connect(self.save_current)
        srow.addWidget(self.btn_save)
        rv.addLayout(srow)
        sp.addWidget(right)

        sp.setStretchFactor(0, 3)
        sp.setStretchFactor(1, 2)
        sp.setSizes([760, 460])
        v.addWidget(sp, 1)

        self._filter_target = self.grid_scroll.viewport()
        self._filter_target.installEventFilter(self)
        return w

    def _build_params(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        head = QLabel("角色操作")
        head.setStyleSheet(f"color:{BLUE}; font-size:14px; font-weight:bold;")
        v.addWidget(head)

        self.lbl_sel = QLabel("未选择角色")
        self.lbl_sel.setWordWrap(True)
        self.lbl_sel.setStyleSheet(
            f"color:{SUB}; background:{BG}; border-radius:4px; padding:8px; font-size:12px;")
        v.addWidget(self.lbl_sel)

        self.btn_apply = QPushButton("🎨 应用到画布")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setStyleSheet(
            f"QPushButton {{ background:{BLUE}; color:#11111b; font-weight:bold;"
            f" padding:9px 12px; border-radius:4px; }}"
            f"QPushButton:disabled {{ background:{SURF1}; color:{SUB}; }}"
        )
        self.btn_apply.clicked.connect(self.apply_selected)
        v.addWidget(self.btn_apply)

        r = QHBoxLayout()
        b = QPushButton("➕ 新建")
        b.clicked.connect(self.new_character)
        r.addWidget(b)
        bc = QPushButton("⧉ 复制")
        bc.clicked.connect(self.duplicate_selected)
        r.addWidget(bc)
        v.addLayout(r)

        r = QHBoxLayout()
        bi = QPushButton("📥 导入")
        bi.clicked.connect(self.import_dialog)
        r.addWidget(bi)
        be = QPushButton("📤 导出")
        be.clicked.connect(self.export_dialog)
        r.addWidget(be)
        v.addLayout(r)

        hint = QLabel("单击卡片 = 载入编辑\n双击卡片 = 直接应用\n\n"
                      "应用会覆盖参考图 / LoRA / 提示词，\n并自动切换到图生图页。")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{SUB}; font-size:11px; padding-top:6px;")
        v.addWidget(hint)
        v.addStretch()
        return w

    def refresh(self):
        for c in self._cards:
            c.setParent(None)
            c.deleteLater()
        self._cards.clear()
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        all_chars = self.library.list_characters()
        self._refresh_tag_filter(all_chars)
        selected_tag = self._selected_tag()
        chars = (self.library.list_characters(tag=selected_tag)
                 if selected_tag else all_chars)
        kw = (self.in_search.text() if hasattr(self, "in_search") else "").strip().lower()
        if kw:
            chars = [
                c for c in chars
                if kw in (c.name or "").lower()
                or kw in (c.base_prompt or "").lower()
                or any(kw in t.lower() for t in (c.tags or []))
            ]

        if hasattr(self, "lbl_count"):
            self._character_count = len(chars)
            self._render_character_count()

        if not chars:
            ph = QLabel("暂无角色，点击右侧 [➕ 新建角色] 开始创建")
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph.setStyleSheet(f"color:{SUB}; font-size:13px; padding:48px;")
            self.grid.addWidget(ph, 0, 0, 1, 4)
            self._cards.clear()
            return

        for c in chars:
            card = CharacterCard(c)
            card.selected.connect(self._on_card_selected)
            card.apply_request.connect(self._on_card_apply)
            card.delete_request.connect(self._on_card_delete)
            self._cards.append(card)

        cols = max(1, (self.grid_scroll.viewport().width() - 24) // (CARD_W + GRID_GAP))
        self._lay_out(cols)

    def eventFilter(self, obj, event):
        if obj is self._filter_target and event.type() == QEvent.Type.Resize:
            vp = self.grid_scroll.viewport()
            if vp is not None:
                self._relayout(vp.width())
        return super().eventFilter(obj, event)

    def _relayout(self, viewport_w: int):
        cols = max(1, (viewport_w - 24) // (CARD_W + GRID_GAP))
        if cols != self._last_cols and self._cards:
            self._lay_out(cols)

    def _refresh_tag_filter(self, chars):
        if not hasattr(self, "cbo_tag"):
            return
        tags = set()
        for c in chars:
            for t in (c.tags or []):
                tags.add(t)
        current = self._selected_tag()
        self.cbo_tag.blockSignals(True)
        self.cbo_tag.clear()
        self.cbo_tag.addItem("全部标签", "")
        for t in sorted(tags):
            self.cbo_tag.addItem(t, t)
        if current:
            idx = self.cbo_tag.findData(current)
            if idx >= 0:
                self.cbo_tag.setCurrentIndex(idx)
        self.cbo_tag.blockSignals(False)

    def _selected_tag(self) -> str:
        if hasattr(self, "cbo_tag") and self.cbo_tag.count():
            return self.cbo_tag.currentData() or ""
        return ""

    def _lay_out(self, cols: int):
        self._last_cols = cols
        for i, card in enumerate(self._cards):
            self.grid.addWidget(card, i // cols, i % cols)

    def _on_card_selected(self, uuid: str):
        char = self.library.get_character(uuid)
        if char is None:
            return
        self._select(uuid)
        self.editor.load(char)

    def _select(self, uuid: Optional[str]):
        self._selected_uuid = uuid
        for c in self._cards:
            c.set_active(c.character.uuid == uuid)
        char = self.library.get_character(uuid) if uuid else None
        if hasattr(self, "lbl_sel"):
            if char:
                bits = [f"<b>{char.name}</b>"]
                bits.append("参考图 ✅" if char.ref_image_path else "参考图 ❌")
                bits.append("LoRA ✅" if char.lora_path else "LoRA ❌")
                bits.append("ControlNet ✅" if char.controlnet_type else "ControlNet ❌")
                self.lbl_sel.setText("<br>".join(bits))
            else:
                self.lbl_sel.setText("未选择角色")
        if hasattr(self, "btn_apply"):
            self.btn_apply.setEnabled(char is not None)
        if hasattr(self.editor, "btn_batch"):
            self.editor.btn_batch.setEnabled(char is not None)

    def _on_card_apply(self, uuid: str):
        self._on_card_selected(uuid)
        self.apply_selected()

    def _on_card_delete(self, uuid: str):
        char = self.library.get_character(uuid)
        if char is None:
            return
        if QMessageBox.question(
            self._ws, "确认删除",
            f"确定删除角色 '{char.name}'？此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        if self.library.delete_character(uuid):
            if self._selected_uuid == uuid:
                self._select(None)
                self.editor.char = None
                self.editor.load(Character(name="新角色"))
            self.refresh()

    def new_character(self):
        char = self.editor.new_character()
        self._select(None)
        return char

    def duplicate_selected(self):
        if not self._selected_uuid:
            QMessageBox.warning(self._ws, "提示", "请先选择一个角色")
            return
        src = self.library.get_character(self._selected_uuid)
        if src is None:
            return
        d = src.to_dict()
        d.pop("uuid", None)
        d.pop("created_at", None)
        d.pop("modified_at", None)
        d.pop("thumbnail_path", None)
        d["name"] = (d.get("name") or "角色") + " 副本"
        dup = Character.from_dict(d)
        if self.library.save_character(dup):
            self.refresh()
            self._select(dup.uuid)
            if self._shell is not None and hasattr(self._shell, "lbl_status"):
                self._shell.lbl_status.setText(f"⧉ 已复制角色: {dup.name}")
                self._shell.lbl_status.setStyleSheet(f"color:{GREEN}; font-size:13px;")

    def save_current(self):
        char = self.editor.collect()
        if char is None:
            QMessageBox.warning(self._ws, "提示", "请先新建或选择一个角色")
            return
        if not char.name.strip():
            QMessageBox.warning(self._ws, "提示", "角色名称不能为空")
            return
        if self.library.save_character(char):
            self._select(char.uuid)
            self.refresh()
            if self._shell is not None and hasattr(self._shell, "lbl_status"):
                self._shell.lbl_status.setText(f"✅ 角色已保存: {char.name}")
                self._shell.lbl_status.setStyleSheet(f"color:{GREEN}; font-size:13px;")

    def import_dialog(self):
        p, _ = QFileDialog.getOpenFileName(
            self._ws, "导入角色卡", "", "角色卡 (*.character.json *.json)")
        if not p:
            return
        char = self.library.import_character(p)
        if char:
            self.refresh()
            self._on_card_selected(char.uuid)
            QMessageBox.information(self._ws, "成功", f"已导入角色 '{char.name}'")
        else:
            QMessageBox.critical(self._ws, "错误", "导入失败，详见日志")

    def open_batch(self):
        char = self.library.get_character(self._selected_uuid) if self._selected_uuid else None
        if char is None:
            QMessageBox.information(self._ws, "提示", "请先选择一个角色")
            return

        pid, _page = _find_target_page(self._shell)
        if pid is None:
            QMessageBox.warning(self._ws, "提示", "找不到可驱动的工作流页面")
            return
        if pid and hasattr(self._shell, "nav"):
            try:
                self._shell.nav.select(pid)
            except Exception:
                pass

        if self._batch_runner is None:
            # 注意：必须是 SceneLibrary（有 .save()），传 CharacterLibrary 会让
            # 批量状态永远落不了盘（'CharacterLibrary' object has no attribute 'save'）
            self._batch_runner = BatchRunner(self._shell, self.scenes, parent=self)

        from ui.scene_batch_dialog import SceneBatchDialog
        dlg = SceneBatchDialog(shell=self._shell, character=char,
                           library=self.scenes, runner=self._batch_runner,
                           parent=self._ws)
        dlg.exec()
        self.refresh()

    def export_dialog(self):
        if not self._selected_uuid:
            QMessageBox.warning(self._ws, "提示", "请先选择一个角色")
            return
        char = self.editor.collect()
        if char is None:
            return
        self.library.save_character(char)
        d = QFileDialog.getExistingDirectory(self._ws, "选择导出目录")
        if not d:
            return
        if self.library.export_character(char.uuid, d):
            QMessageBox.information(self._ws, "成功", f"已导出到:\n{d}")
        else:
            QMessageBox.critical(self._ws, "错误", "导出失败，详见日志")

    def apply_selected(self):
        if not self._selected_uuid:
            return
        char = self.editor.collect()
        if char is None:
            return
        self.library.save_character(char)
        self.refresh()
        self._select(char.uuid)
        _apply_character_to_shell(self._shell, char, self._ws)


# ═══════════════════════════════════════════════════════════
#  与工作流页面的桥接
# ═══════════════════════════════════════════════════════════
def _find_target_page(shell):
    """找到"应用角色"要切过去的页面。

    返回 (page_id, page)。page 只用于判断是否找到目标；
    真正要写的控件（参考图/提示词/IP-Adapter/ControlNet）都挂在 shell 上。

    ⚠️ 旧兜底分支判断的是 `hasattr(page, "ref_image_path")`，
    但这些属性属于 shell，页面实例上永远没有 → 兜底恒为 no-op。
    现在改为按"哪些页支持角色工作流"判断，并回退到 shell 上是否有生成控件。
    """
    if shell is None:
        return None, None
    pages = getattr(shell, "_pages", None) or {}

    # 角色工作流优先落在图生图，其次文生图
    for pid in ("img2img", "txt2img"):
        if pages.get(pid) is not None:
            return pid, pages[pid]

    # 没有已知的图生图/文生图页时：只要 shell 上还有生成核心控件，
    # 就仍可套用参数（只是不切页）
    if any(hasattr(shell, a) for a in ("txt_prompt", "txt_neg", "combo_model")):
        return (next(iter(pages), None), None)
    return None, None


def _set_scale(widget, value_0_1: float) -> bool:
    if widget is None:
        return False
    try:
        if hasattr(widget, "maximum") and widget.maximum() > 2:
            widget.setValue(int(round(value_0_1 * 100)))
        elif hasattr(widget, "setValue"):
            widget.setValue(value_0_1)
        else:
            return False
        return True
    except Exception:
        return False


def _apply_character_to_shell(shell, char: Character, parent):
    if shell is None:
        return
    pid, _page = _find_target_page(shell)
    if pid is None:
        QMessageBox.warning(parent, "提示", "找不到可应用的工作流页面")
        return

    applied, skipped = [], []

    if hasattr(shell, "nav") and pid:
        try:
            shell.nav.select(pid)
        except Exception:
            pass

    if char.ref_image_path and Path(char.ref_image_path).exists():
        # 参考图状态/缩略图都挂在 shell 上（页面实例只是工作区容器）
        if hasattr(shell, "set_reference_image"):
            shell.set_reference_image(char.ref_image_path, None)
            applied.append("参考图")
        else:
            shell.ref_image_path = char.ref_image_path
            applied.append("参考图（未刷新缩略图）")
    elif char.ref_image_path:
        skipped.append("参考图（文件不存在）")

    combos = getattr(shell, "combo_loras", None) or []
    scales = getattr(shell, "scale_loras", None) or []

    def _lora_key(text: str) -> str:
        """下拉框文本 → 可比较的文件名。
        形如 'name[trigger]' 的先切掉 [ 后缀（同 shell._insert_lora_triggers）。"""
        text = (text or "").strip()
        if "[" in text:
            text = text.split("[")[0].strip()
        return Path(text).name

    if not combos or not scales:
        if char.lora_path:
            skipped.append("LoRA（shell 上找不到 combo_loras/scale_loras，别名未安装？）")
    elif char.lora_path and not Path(char.lora_path).exists():
        skipped.append("LoRA（文件不存在）")
    else:
        target_idx = None
        if char.lora_path:
            want = _lora_key(Path(char.lora_path).name)
            for i in range(combos[0].count()):
                if _lora_key(combos[0].itemText(i)) == want:
                    target_idx = i
                    break

        if char.lora_path and target_idx is None:
            skipped.append(
                f"LoRA（下拉框里没有 {Path(char.lora_path).name}，可能底模架构不匹配）")
        elif target_idx is not None:
            combos[0].setCurrentIndex(target_idx)
            scales[0].setValue(int(round(char.lora_weight * 100)))
            applied.append(f"LoRA {combos[0].itemText(target_idx)}:{char.lora_weight:.2f}")
        else:
            # 角色没配 LoRA：必须清空第 0 槽。
            # 不清的话，_gt_apply_loras 会读到上一个角色留下的值并重新挂上
            # （它只在控件列表全空时才走 else 卸载，槽里有残留值就不会触发）
            sentinel = next(
                (i for i in range(combos[0].count())
                 if combos[0].itemText(i).strip() in ("无", "None", "")),
                None,
            )
            if sentinel is not None:
                combos[0].setCurrentIndex(sentinel)
                scales[0].setValue(0)
                applied.append("LoRA 已清空")
            else:
                skipped.append("LoRA（找不到空位哨兵，无法清空旧选择）")


    # IP-Adapter / 强度：控件在主窗口 shell 上（spin_ipa_scale / spin_pt_cn 等）
    for attr in ("spin_ipa_scale", "slider_ipa_scale", "ipa_scale_slider",
                 "ip_adapter_scale"):
        w = getattr(shell, attr, None)
        if _set_scale(w, char.ip_adapter_scale):
            if hasattr(shell, "chk_use_ipa"):
                shell.chk_use_ipa.setChecked(True)
            applied.append(f"IP-Adapter {char.ip_adapter_scale:.2f}")
            break
    else:
        skipped.append("IP-Adapter（未找到控件）")

    if char.controlnet_image_path and Path(char.controlnet_image_path).exists():
        hit = False
        # 姿态图路径：shell.pose_image_path（img2img 页也叫「姿态图」）
        for attr in ("pose_image_path", "controlnet_image_path", "cn_image_path"):
            if hasattr(shell, attr):
                setattr(shell, attr, char.controlnet_image_path)
                hit = True
                break
        if char.controlnet_type and hasattr(shell, "combo_cn_type"):
            idx = shell.combo_cn_type.findText(char.controlnet_type)
            if idx >= 0:
                shell.combo_cn_type.setCurrentIndex(idx)
                hit = True
        for attr in ("scale_cn_strength", "scale_cn_weight",
                     "controlnet_scale", "cn_scale_slider"):
            if _set_scale(getattr(shell, attr, None), char.controlnet_scale):
                hit = True
                break
        if hasattr(shell, "chk_use_pose"):
            shell.chk_use_pose.setChecked(True)
        if hit:
            applied.append("ControlNet")
        else:
            skipped.append("ControlNet（未找到控件）")

    def _merge(box_getter, new_text):
        if not new_text:
            return False
        box = box_getter()
        if box is None:
            return False
        cur = box.toPlainText().strip()
        if new_text in cur:
            return True
        box.setPlainText(f"{new_text}, {cur}" if cur else new_text)
        return True

    if char.base_prompt:
        if _merge(lambda: getattr(shell, "txt_prompt", None), char.base_prompt):
            applied.append("正面提示词")
        else:
            skipped.append("正面提示词（未找到控件）")
    if char.negative_prompt:
        # 负面框在 shell 上叫 txt_neg（不是 txt_neg_prompt / txt_negative）
        if _merge(lambda: getattr(shell, "txt_neg", None)
                  or getattr(shell, "txt_negative", None), char.negative_prompt):
            applied.append("负面提示词")
        else:
            skipped.append("负面提示词（未找到控件）")

    msg = f"✅ 已应用角色「{char.name}」"
    if hasattr(shell, "lbl_status"):
        shell.lbl_status.setText(msg)
        shell.lbl_status.setStyleSheet(f"color:{GREEN}; font-size:13px;")

    lines = [msg, ""]
    if applied:
        lines.append("已应用: " + "、".join(applied))
    if skipped:
        lines.append("已跳过: " + "、".join(skipped))
    QMessageBox.information(parent, "应用角色", "\n".join(lines))