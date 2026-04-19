# worker_thread.py
"""
后台工作线程：
- 使用 QThread + Worker(QObject) 模式，避免大图处理时界面卡顿。
- 处理完成后通过 pyqtSignal 传递 PIL Image（或 QPixmap）。

内存泄漏规避要点：
1. Worker 处理完毕后 emit 结果，MainWindow 接收后覆盖当前引用，
   旧 PIL Image / QPixmap 会被 Python GC 自动回收。
2. 滑块频繁触发时使用 QTimer 防抖，避免大量任务堆积占用内存。
3. 每次处理均以"基础图像 + 参数"为输入重新合成，不在内部累积状态。
"""
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PIL import Image
from image_processor import ImageProcessor


class AdjustmentWorker(QObject):
    """对基础图像施加调整参数（brightness 等），返回处理后的 PIL Image。"""
    finished = pyqtSignal(object)   # emit PIL Image
    error = pyqtSignal(str)

    def __init__(self, base_image: Image.Image, params: dict):
        super().__init__()
        self.base_image = base_image
        self.params = params

    def run(self):
        try:
            result = ImageProcessor.process_adjustments(self.base_image, self.params)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class FilterWorker(QObject):
    """对当前图像应用预设滤镜。"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, image: Image.Image, filter_name: str, blur_radius: float = 2.0):
        super().__init__()
        self.image = image
        self.filter_name = filter_name
        self.blur_radius = blur_radius

    def run(self):
        try:
            result = ImageProcessor.apply_filter(
                self.image, self.filter_name, self.blur_radius
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


def run_in_thread(worker: QObject, on_finished, on_error=None):
    """
    通用方法：把一个 QObject Worker 放到新 QThread 中执行。
    返回 (thread, worker)，调用者保留引用直到线程结束。
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    if on_error:
        worker.error.connect(on_error)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker