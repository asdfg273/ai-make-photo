# ui/theme.py
# ============================================================
#  主题中枢 — 全项目唯一主题真源
#  单一 PALETTE 令牌源 + 全套 APP_QSS（由令牌拼接，无硬编码色）
#  风格：中性石墨底 + 鼠尾草绿强调色，以内容为中心的桌面工作台
#  其他文件一律从这里取色，禁止自行硬编码颜色
# ============================================================
import logging

logger = logging.getLogger(__name__)

PALETTE = {
    # ── 中性石墨色基底：让图片本身成为视觉中心 ──
    "bg":        "#111416",   # 窗口/画布底色
    "bg_panel":  "#191d20",   # 输入控件/状态栏/面板
    "bg_soft":   "#22272b",   # 悬浮与选中底衬
    "bg_hover":  "#2b3136",   # 悬浮态
    "border":    "#333a40",   # 边框/分割线
    # ── 文字 ──
    "fg":        "#e3e7e8",   # 主文字
    "fg_mute":   "#9aa4a9",   # 次要文字
    "fg_faint":  "#68747a",   # 弱化/禁用文字
    # ── 克制的鼠尾草绿强调色，仅用于关键动作和选中态 ──
    "accent":      "#7da58f",
    "accent_hi":   "#a1c4af",
    "accent_soft": "#293b33",
    "on_accent":   "#101713",
    # ── 语义色 ──
    "success":   "#79b188",
    "warning":   "#d2a55e",
    "danger":    "#d27676",
    "info":      "#78a6b8",
    # ── 尺寸 ──
    "radius":    4,
    "space":     8,
}

# 全套样式。%(key)s 由 PALETTE 填充；QSS 自身的 {} 保持单括号。
_APP_QSS_TEMPLATE = """
/* ================= 基底 ================= */
QWidget {
    font-family: "Segoe UI", "Yu Gothic UI", "Meiryo",
                 "Microsoft YaHei UI", "PingFang SC", sans-serif;
    font-size: 13px;
}
QMainWindow::separator { background: %(border)s; width: 1px; height: 1px; }
QToolTip {
    background: %(bg_soft)s; color: %(fg)s;
    border: 1px solid %(border)s; padding: 5px 8px; border-radius: 3px;
}
QScrollArea { background: transparent; border: none; }
QWidget#navRail { background: %(bg_panel)s; }
QWidget#paramsPanel {
    background: %(bg_panel)s; border-left: 1px solid %(border)s;
}

/* ================= 菜单栏 ================= */
QMenuBar {
    background: %(bg)s; color: %(fg_mute)s;
    border-bottom: 1px solid %(border)s;
}
QMenuBar::item { background: transparent; padding: 6px 11px; }
QMenuBar::item:selected { background: %(bg_soft)s; color: %(fg)s; }
QMenuBar::item:pressed { color: %(accent_hi)s; }
QMenu {
    background: %(bg_panel)s; color: %(fg)s;
    border: 1px solid %(border)s; padding: 4px;
}
QMenu::item { padding: 6px 24px 6px 12px; border-radius: 3px; }
QMenu::item:selected { background: %(bg_hover)s; color: %(accent_hi)s; }
QMenu::separator { height: 1px; background: %(border)s; margin: 4px 8px; }

/* ================= 按钮 ================= */
QPushButton {
    background: %(bg_panel)s; color: %(fg)s;
    border: 1px solid %(border)s; border-radius: 4px;
    padding: 4px 10px; min-height: 26px;
}
QPushButton:hover { background: %(bg_hover)s; border-color: %(fg_faint)s; }
QPushButton:pressed { background: %(bg_soft)s; }
QPushButton:disabled {
    color: %(fg_faint)s; background: %(bg_panel)s; border-color: %(border)s;
}
QPushButton:checked {
    background: %(accent_soft)s; color: %(accent_hi)s; border-color: %(accent)s;
}
QPushButton[kind="primary"], QPushButton#btnGenVideo {
    background: %(accent)s; color: %(on_accent)s; font-weight: 600;
    border: none; border-radius: 4px; padding: 10px 18px; font-size: 14px;
}
QPushButton[kind="primary"]:hover, QPushButton#btnGenVideo:hover {
    background: %(accent_hi)s;
}
QPushButton[kind="primary"]:disabled, QPushButton#btnGenVideo:disabled {
    background: %(bg_soft)s; color: %(fg_faint)s;
}
/* 历史属性名保留为 pill，但视觉改为普通小型操作按钮。 */
QPushButton[role="pill"] {
    background: transparent; color: %(fg_mute)s;
    border: 1px solid %(border)s; border-radius: 4px;
    padding: 4px 10px; font-size: 12px;
}
QPushButton[role="pill"]:hover { color: %(accent_hi)s; border-color: %(accent)s; }
QPushButton[role="pill"]:disabled { color: %(fg_faint)s; }
QPushButton#btnMediaAll:checked,
QPushButton#btnMediaImage:checked,
QPushButton#btnMediaVideo:checked {
    background: %(accent_soft)s; color: %(accent_hi)s;
    border-color: %(accent)s; font-weight: 600;
}

/* ================= 左侧导航 ================= */
QToolButton#navBtn {
    color: %(fg_mute)s; background: transparent;
    border: none; border-left: 2px solid transparent; border-radius: 0;
    padding: 10px 8px; font-size: 13px; text-align: left;
}
QToolButton#navBtn:hover { color: %(fg)s; background: %(bg_soft)s; }
QToolButton#navBtn:checked {
    color: %(accent_hi)s; background: %(accent_soft)s;
    border-left-color: %(accent)s; font-weight: 600;
}

/* ================= 折叠分组标题 ================= */
QToolButton#collapsibleHeader {
    background: transparent; color: %(fg_mute)s;
    border: none; border-bottom: 1px solid %(border)s; border-radius: 0;
    padding: 8px 4px; font-weight: 600; text-align: left;
}
QToolButton#collapsibleHeader:hover { color: %(fg)s; background: %(bg_soft)s; }
QToolButton#collapsibleHeader:checked { color: %(accent_hi)s; }

/* ================= 输入控件 ================= */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
    background: %(bg_panel)s; color: %(fg)s;
    border: 1px solid %(border)s; border-radius: 4px; padding: 4px 7px;
    min-height: 24px;
    selection-background-color: %(accent_soft)s; selection-color: %(fg)s;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus { border-color: %(accent)s; }
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled { color: %(fg_faint)s; }
QComboBox {
    background: %(bg_panel)s; color: %(fg)s;
    border: 1px solid %(border)s; border-radius: 4px;
    padding: 4px 7px; min-height: 24px;
}
QComboBox:hover { border-color: %(fg_faint)s; }
QComboBox:focus { border-color: %(accent)s; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow {
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid %(fg_mute)s; margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: %(bg_panel)s; color: %(fg)s; border: 1px solid %(border)s;
    selection-background-color: %(accent_soft)s; selection-color: %(accent_hi)s;
    outline: none;
}

/* ================= 复选 / 单选 ================= */
QCheckBox, QRadioButton { color: %(fg)s; spacing: 7px; background: transparent; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 15px; height: 15px; border: 1px solid %(border)s;
    background: %(bg_panel)s; border-radius: 3px;
}
QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: %(accent)s; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: %(accent)s; border-color: %(accent)s;
}
QCheckBox:disabled, QRadioButton:disabled { color: %(fg_faint)s; }

/* ================= 滑块 ================= */
QSlider::groove:horizontal { height: 3px; background: %(border)s; }
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
    background: %(fg_mute)s; border: 2px solid %(bg)s;
}
QSlider::handle:horizontal:hover { background: %(accent_hi)s; }
QSlider::sub-page:horizontal { background: %(accent)s; }
QSlider::add-page:horizontal { background: %(border)s; }
QSlider::handle:horizontal:disabled { background: %(fg_faint)s; }
QSlider::sub-page:horizontal:disabled { background: %(bg_soft)s; }

/* ================= 滚动条 ================= */
QScrollBar:vertical { background: transparent; width: 9px; margin: 1px; }
QScrollBar::handle:vertical {
    background: %(bg_hover)s; min-height: 30px; border-radius: 4px;
}
QScrollBar::handle:vertical:hover { background: %(fg_faint)s; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QScrollBar:horizontal { background: transparent; height: 9px; margin: 1px; }
QScrollBar::handle:horizontal {
    background: %(bg_hover)s; min-width: 30px; border-radius: 4px;
}
QScrollBar::handle:horizontal:hover { background: %(fg_faint)s; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }

/* ================= 分组框：用分隔线代替层层卡片 ================= */
QGroupBox {
    background: transparent; border: none; border-top: 1px solid %(border)s;
    margin-top: 15px; padding: 11px 2px 4px 2px;
    color: %(fg)s; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 0; padding: 0 6px 0 0; color: %(fg_mute)s; font-size: 12px;
}
QGroupBox[accent="true"] { border-top-color: %(accent)s; }
QGroupBox[accent="true"]::title { color: %(accent_hi)s; }

/* ================= 标签语义 ================= */
QLabel { background: transparent; }
QLabel[role="hint"]  { color: %(fg_mute)s; font-size: 12px; }
QLabel[role="title"] { color: %(fg)s; font-weight: 600; font-size: 12px; }
QLabel[role="value"] { color: %(accent_hi)s; font-weight: 600; }
QLabel[role="field"] { color: %(fg)s; font-weight: 600; font-size: 12px; }
QLabel[role="body"]  { color: %(fg_mute)s; font-size: 12px; }

/* ================= 进度与状态 ================= */
QProgressBar { background: %(bg_soft)s; border: none; color: transparent; }
QProgressBar::chunk { background: %(accent)s; }
QStatusBar {
    background: %(bg_panel)s; color: %(fg_mute)s;
    border-top: 1px solid %(border)s;
}
QStatusBar QLabel { background: transparent; color: %(fg_mute)s; font-size: 12px; }

/* ================= 列表与胶片条 ================= */
QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { border-radius: 3px; }
QListWidget::item:selected { background: %(accent_soft)s; }
QListWidget#filmStrip {
    background: %(bg_panel)s; border: 1px solid %(border)s; border-radius: 4px;
}

/* ================= 标签页 ================= */
QTabWidget::pane {
    border: none; border-top: 1px solid %(border)s; background: transparent;
}
QTabBar::tab {
    background: transparent; color: %(fg_mute)s;
    border: none; border-bottom: 2px solid transparent;
    padding: 7px 14px; margin-right: 3px;
}
QTabBar::tab:selected {
    background: transparent; color: %(accent_hi)s; border-bottom-color: %(accent)s;
}
QTabBar::tab:hover:!selected { color: %(fg)s; }

/* ================= 分割条 / 对话框 ================= */
QSplitter::handle { background: %(border)s; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QDialog { background: %(bg)s; }
QMessageBox { background: %(bg)s; }
"""

APP_QSS = _APP_QSS_TEMPLATE % PALETTE


def apply_theme(app) -> str:
    """应用主题。返回实际使用的主题名。qdarkstyle 失败时回退 Fusion 深色。"""
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6") + APP_QSS)
        logger.info("主题: qdarkstyle (dark) + 项目令牌覆盖层")
        return "qdarkstyle"
    except Exception as e:
        logger.warning(f"⚠ qdarkstyle 加载失败，回退 Fusion 深色: {e}")
        from PyQt6.QtGui import QPalette, QColor
        app.setStyle("Fusion")
        p = QPalette()
        p.setColor(QPalette.ColorRole.Window, QColor(PALETTE["bg"]))
        p.setColor(QPalette.ColorRole.WindowText, QColor(PALETTE["fg"]))
        p.setColor(QPalette.ColorRole.Base, QColor(PALETTE["bg_panel"]))
        p.setColor(QPalette.ColorRole.Text, QColor(PALETTE["fg"]))
        p.setColor(QPalette.ColorRole.Button, QColor(PALETTE["bg_panel"]))
        p.setColor(QPalette.ColorRole.ButtonText, QColor(PALETTE["fg"]))
        p.setColor(QPalette.ColorRole.Highlight, QColor(PALETTE["accent"]))
        app.setPalette(p)
        app.setStyleSheet(APP_QSS)
        return "fusion-fallback"
