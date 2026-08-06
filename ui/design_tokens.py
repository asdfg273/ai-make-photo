# ui/design_tokens.py
# ============================================================
#  xAI 设计令牌 (Design Tokens) — 从 ui_builder.py 提取
#  基于 DESIGN.md 规范定义的设计系统常量
# ============================================================

DESIGN_TOKENS = {
    # 颜色
    'colors': {
        'primary': '#ffffff',
        'on-primary': '#0a0a0a',
        'ink': '#ffffff',
        'ink-hover': '#fafaf7',
        'body': '#dadbdf',
        'body-mid': '#7d8187',
        'mute': '#7d8187',
        'hairline': '#212327',
        'canvas': '#0a0a0a',
        'canvas-soft': '#1a1c20',
        'canvas-card': '#191919',
        'canvas-mid': '#363a3f',
        'accent-sunset': '#ff7a17',
        'accent-sunset-soft': '#ffc285',
        'accent-dusk': '#7c3aed',
        'accent-twilight': '#c4b5fd',
        'accent-breeze': '#a0c3ec',
        'accent-midnight': '#0d1726',
    },
    # 圆角
    'rounded': {
        'none': 0,
        'sm': 8,
        'pill': 9999,
        'full': 9999,
    },
    # 间距
    'spacing': {
        'xxs': 2,
        'xs': 4,
        'sm': 8,
        'md': 12,
        'lg': 16,
        'xl': 24,
        '2xl': 32,
        '3xl': 48,
        '4xl': 64,
    },
    # 字体
    'fonts': {
        'display': '"Segoe UI", "Microsoft YaHei", sans-serif',
        'mono': 'Consolas, "JetBrains Mono", "IBM Plex Mono", monospace',
    },
}


# ============================================================
#  全局深色样式 (DARK_STYLE)
# ============================================================
DARK_STYLE = """
QMainWindow, QDialog, QWidget {
    background:#0a0a0a; color:#ffffff;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 14px;
}
QTabWidget::pane { border:1px solid #212327; border-radius:8px; background:#191919; }
QTabBar::tab { background:#0a0a0a; color:#7d8187; padding:6px 16px; border:1px solid #212327; border-radius:9999px; margin-right:4px; }
QTabBar::tab:selected { background:#ffffff; color:#0a0a0a; border-color:#ffffff; }
QTabBar::tab:hover { color:#ffffff; border-color:#363a3f; }
QPushButton { background:#0a0a0a; color:#ffffff; border:1px solid #212327; border-radius:9999px; padding:6px 16px; }
QPushButton:hover { border-color:#363a3f; }
QPushButton:pressed { background:#1a1c20; }
QPushButton:disabled { background:#0a0a0a; color:#363a3f; border-color:#212327; }
QComboBox { background:#1a1c20; color:#ffffff; border:1px solid #212327; border-radius:8px; padding:4px 8px; }
QComboBox QAbstractItemView { background:#191919; color:#ffffff; selection-background-color:#363a3f; selection-color:#ffffff; border:1px solid #212327; }
QTextEdit, QLineEdit { background:#1a1c20; color:#ffffff; border:1px solid #212327; border-radius:8px; padding:8px 12px; }
QTextEdit:focus, QLineEdit:focus { border-color:#ffffff; }
QSpinBox, QDoubleSpinBox { background:#1a1c20; color:#ffffff; border:1px solid #212327; border-radius:8px; padding:4px 8px; }
QSlider::groove:horizontal { height:4px; background:#212327; border-radius:2px; }
QSlider::handle:horizontal { background:#ffffff; width:14px; height:14px; margin:-5px 0; border-radius:7px; }
QSlider::sub-page:horizontal { background:#ffffff; border-radius:2px; }
QCheckBox { color:#ffffff; spacing:8px; }
QCheckBox::indicator { width:16px; height:16px; border:1px solid #212327; border-radius:3px; background:#191919; }
QCheckBox::indicator:checked { background:#ffffff; border-color:#ffffff; }
QScrollBar:vertical { background:#0a0a0a; width:8px; border-radius:4px; }
QScrollBar::handle:vertical { background:#363a3f; border-radius:4px; min-height:20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QGroupBox { border:1px solid #212327; border-radius:8px; margin-top:16px; padding:12px; color:#dadbdf; font-size:13px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 8px; color:#ffffff; font-size:13px; }
QLabel { background:transparent; }
QProgressBar { background:#1a1c20; border-radius:4px; border:1px solid #212327; text-align:center; color:#ffffff; }
QProgressBar::chunk { background:#ffffff; border-radius:4px; }
QToolTip {
    background-color: #191919;
    color: #ffffff;
    border: 1px solid #212327;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    font-family: "Segoe UI", Consolas;
}
"""

VIDEO_TAB_QSS = """
    QWidget#animRoot { background: transparent; }

    QGroupBox {
        color:#dadbdf; font-weight:bold; font-size:14px;
        border:1px solid #212327; border-radius:10px;
        margin-top:12px; padding:14px 10px 10px 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin; subcontrol-position: top left;
        left:12px; padding:0 6px;
    }
    QGroupBox[accent="true"] { color:#ff7a17; border-color:#3a2a14; }
    QGroupBox::indicator { width:16px; height:16px; }

    QLabel[role="field"] { color:#ffffff; font-weight:bold; font-size:13px; }
    QLabel[role="hint"]  { color:#7d8187; font-weight:normal; font-size:12px; }
    QLabel[role="body"]  { color:#dadbdf; font-weight:normal; font-size:12px; }
    QLabel[role="value"] { color:#ff7a17; font-weight:bold; font-size:12px; }

    QCheckBox { color:#dadbdf; font-size:13px; spacing:6px; }

    QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QLineEdit { font-size:13px; }

    QPushButton[role="pill"] {
        background:#0a0a0a; color:#dadbdf;
        border:1px solid #212327; border-radius:9999px;
        padding:6px 14px; font-size:12px; font-weight:normal;
    }
    QPushButton[role="pill"]:hover { border-color:#ff7a17; color:#ffffff; }

    QPushButton#btnGenVideo {
        background:#ff7a17; color:#0a0a0a; font-weight:bold;
        font-size:15px; border-radius:9999px; border:none;
    }
    QPushButton#btnGenVideo:hover  { background:#ff9040; }
    QPushButton#btnGenVideo:disabled { background:#3a3a3a; color:#7d8187; }
    """