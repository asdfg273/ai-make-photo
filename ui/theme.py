# ui/theme.py
# ============================================================
#  主题中枢 — qdarkstyle 深蓝灰基底 + 项目自定义覆盖层
#  全项目唯一主题入口；组件一律从 PALETTE 取色，杜绝硬编码
# ============================================================
import logging

logger = logging.getLogger(__name__)

PALETTE = {
    "accent":      "#4a9eff",   # qdarkstyle 原生蓝，导航选中/主按钮
    "accent_hi":   "#6cb2ff",
    "bg":          "#19232d",
    "bg_soft":     "#22303d",
    "fg":          "#dfe5ec",
    "fg_mute":     "#8fa1b3",
    "danger":      "#e06c75",
    "radius":      6,
    "space":       8,
}

# 项目自定义覆盖层：只做品牌微调 + 语义类，不重画控件
APP_QSS = """
/* 语义类：组件用 setProperty 打标，样式集中在此 */
QLabel[role="hint"]  { color: #8fa1b3; font-size: 12px; }
QLabel[role="title"] { color: #dfe5ec; font-weight: bold; font-size: 13px; }
QLabel[role="value"] { color: #4a9eff; font-weight: bold; }

QPushButton[kind="primary"] {
    background: #4a9eff; color: #0d1620; font-weight: bold;
    border: none; border-radius: 6px; padding: 10px 20px; font-size: 15px;
}
QPushButton[kind="primary"]:hover    { background: #6cb2ff; }
QPushButton[kind="primary"]:disabled { background: #37414b; color: #8fa1b3; }

/* 左侧导航栏 */
QToolButton#navBtn {
    color: #8fa1b3; border: none; border-radius: 6px;
    padding: 8px 4px; font-size: 12px;
}
QToolButton#navBtn:hover   { color: #dfe5ec; background: #22303d; }
QToolButton#navBtn:checked { color: #ffffff; background: #4a9eff; }

/* 折叠分组 */
QWidget#collapsibleHeader { background: #22303d; border-radius: 6px; }
QGroupBox { border-radius: 6px; }
"""


def apply_theme(app) -> str:
    """应用主题。返回实际使用的主题名。qdarkstyle 失败时回退 Fusion 深色。"""
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6") + APP_QSS)
        logger.info("主题: qdarkstyle (dark)")
        return "qdarkstyle"
    except Exception as e:
        logger.warning(f"⚠️ qdarkstyle 加载失败，回退 Fusion 深色: {e}")
        from PyQt6.QtGui import QPalette, QColor
        app.setStyle("Fusion")
        p = QPalette()
        p.setColor(QPalette.ColorRole.Window, QColor(25, 35, 45))
        p.setColor(QPalette.ColorRole.WindowText, QColor(223, 229, 236))
        p.setColor(QPalette.ColorRole.Base, QColor(34, 48, 61))
        p.setColor(QPalette.ColorRole.Text, QColor(223, 229, 236))
        p.setColor(QPalette.ColorRole.Button, QColor(34, 48, 61))
        p.setColor(QPalette.ColorRole.ButtonText, QColor(223, 229, 236))
        p.setColor(QPalette.ColorRole.Highlight, QColor(74, 158, 255))
        app.setPalette(p)
        app.setStyleSheet(APP_QSS)
        return "fusion-fallback"
