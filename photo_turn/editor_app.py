# main.py
"""
程序入口：创建 QApplication 和 MainWindow。
使用 Fusion 样式获得跨平台一致外观。
"""
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from main_window import MainWindow


def main():
    # 启用高 DPI 支持（PyQt6 默认启用，但显式声明以保证一致性）
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Photo Editor")
    app.setOrganizationName("LocalTools")

    # 全局 QSS：美化滑块、按钮和面板
    app.setStyleSheet("""
        QMainWindow, QDockWidget, QWidget {
            background-color: #2b2b2b;
            color: #e0e0e0;
        }
        QDockWidget::title {
            background: #3a3a3a;
            padding: 6px;
            font-weight: bold;
        }
        QPushButton {
            background-color: #4a4a4a;
            border: 1px solid #5a5a5a;
            border-radius: 4px;
            padding: 6px 12px;
            color: #e0e0e0;
        }
        QPushButton:hover {
            background-color: #5a5a5a;
        }
        QPushButton:pressed {
            background-color: #3a3a3a;
        }
        QPushButton:disabled {
            background-color: #333;
            color: #777;
        }
        QSlider::groove:horizontal {
            border: 1px solid #555;
            height: 6px;
            background: #3a3a3a;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #4a9eff;
            border: 1px solid #3a8edf;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #5aaeff;
        }
        QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #3a3a3a;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px 4px;
            color: #e0e0e0;
        }
        QLabel {
            color: #e0e0e0;
        }
        QMenuBar {
            background-color: #333;
            color: #e0e0e0;
        }
        QMenuBar::item:selected {
            background-color: #4a9eff;
        }
        QMenu {
            background-color: #3a3a3a;
            color: #e0e0e0;
            border: 1px solid #555;
        }
        QMenu::item:selected {
            background-color: #4a9eff;
        }
        QToolBar {
            background-color: #333;
            border: none;
            spacing: 4px;
            padding: 4px;
        }
        QStatusBar {
            background-color: #333;
            color: #e0e0e0;
        }
        QScrollArea {
            background-color: #1e1e1e;
            border: 1px solid #444;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            background: #3a3a3a;
            text-align: center;
            color: #e0e0e0;
        }
        QProgressBar::chunk {
            background: #4a9eff;
        }
    """)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    win = MainWindow(input_path, output_path) # 传入路径
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()