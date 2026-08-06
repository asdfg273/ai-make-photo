# utils/gpu_init.py
# ============================================================
#  GPU 加速初始化 — 从 ui_builder.py 提取
# ============================================================

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtCore import Qt


def enable_gpu_acceleration():
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setVersion(3, 3)
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)