# ui/splash.py
# ============================================================
#  启动画面 — 从 ui_builder.py 提取
# ============================================================

import os

from PyQt6.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QLabel, QProgressBar,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap


class SplashScreen(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedSize(480, 320)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._setup_ui()
        self._center()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setObjectName("splash_container")
        container.setStyleSheet("""
            #splash_container {
                background: #0a0a0a;
                border-radius: 8px;
                border: 1px solid #212327;
            }
        """)
        inner = QVBoxLayout(container)
        inner.setContentsMargins(40, 40, 40, 30)
        inner.setSpacing(14)

        ico_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "logo", "dzbut-9fc5g-001.ico"
        )
        lbl_icon = QLabel()
        lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.exists(ico_path):
            pix = QPixmap(ico_path).scaled(
                80, 80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            lbl_icon.setPixmap(pix)
        else:
            lbl_icon.setText("🎨")
            lbl_icon.setStyleSheet("font-size:64px;")
        inner.addWidget(lbl_icon)

        lbl_title = QLabel("AI 绘画工作站")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet(
            "color:#ffffff; font-size:22px;")
        inner.addWidget(lbl_title)

        lbl_sub = QLabel("v5.0  PyQt6 Edition")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet("color:#7d8187; font-size:12px;")
        inner.addWidget(lbl_sub)

        self.lbl_msg = QLabel("正在初始化...")
        self.lbl_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_msg.setStyleSheet("color:#7d8187; font-size:11px;")
        inner.addWidget(self.lbl_msg)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar { background:#1a1c20; border-radius:3px; }
            QProgressBar::chunk { background:#ffffff; border-radius:3px; }
        """)
        inner.addWidget(self.progress)
        layout.addWidget(container)

    def _center(self):
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )

    def set_message(self, msg: str):
        self.lbl_msg.setText(msg)
        QApplication.processEvents()

    def finish_loading(self, main_window):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        QTimer.singleShot(300, lambda: (main_window.show(), self.close()))


def create_splash() -> SplashScreen:
    splash = SplashScreen()
    splash.show()
    QApplication.processEvents()
    return splash