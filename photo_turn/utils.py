# utils.py
"""
辅助函数：
- PIL Image 与 QImage / QPixmap 的互相转换。
- 转换时保持 RGBA 透明度通道（若有）。
"""
from PIL import Image
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


def pil_to_qimage(pil_img: Image.Image) -> QImage:
    """
    PIL Image -> QImage。
    - 若图像带 alpha，则使用 RGBA8888 格式。
    - 否则使用 RGB888 格式。
    注意：必须 .copy()，因为 QImage 底层指向 bytes 缓冲区，
    如果 bytes 被 GC 会导致悬空指针崩溃。
    """
    if pil_img.mode == "RGBA":
        data = pil_img.tobytes("raw", "RGBA")
        qimg = QImage(data, pil_img.width, pil_img.height,
                      pil_img.width * 4, QImage.Format.Format_RGBA8888)
        return qimg.copy()
    elif pil_img.mode == "RGB":
        data = pil_img.tobytes("raw", "RGB")
        qimg = QImage(data, pil_img.width, pil_img.height,
                      pil_img.width * 3, QImage.Format.Format_RGB888)
        return qimg.copy()
    else:
        # 其他模式统一转换为 RGBA
        converted = pil_img.convert("RGBA")
        data = converted.tobytes("raw", "RGBA")
        qimg = QImage(data, converted.width, converted.height,
                      converted.width * 4, QImage.Format.Format_RGBA8888)
        return qimg.copy()


def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    """PIL Image -> QPixmap"""
    return QPixmap.fromImage(pil_to_qimage(pil_img))


def qimage_to_pil(qimg: QImage) -> Image.Image:
    """
    QImage -> PIL Image。
    统一转换为 RGBA 便于后续处理。
    """
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    width, height = qimg.width(), qimg.height()
    ptr = qimg.constBits()
    ptr.setsize(height * width * 4)
    arr = bytes(ptr)
    return Image.frombytes("RGBA", (width, height), arr)


def qpixmap_to_pil(pix: QPixmap) -> Image.Image:
    """QPixmap -> PIL Image"""
    return qimage_to_pil(pix.toImage())


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小显示"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"