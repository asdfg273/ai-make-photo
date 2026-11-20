# utils/video_io.py
# ============================================================
#  视频读写小工具：解决 Windows 下 OpenCV 读不了中文路径的问题
#  cv2.VideoCapture("D:/我的视频/a.mp4") 在 Windows 上会**静默失败**
#  （isOpened() 返回 False，或帧数为 0），因为 OpenCV 内部用窄字符 API。
#  rife_interpolate.py 已经为图片踩过这个坑（_imread/_imwrite 用
#  np.fromfile + imdecode 绕开），但视频没有等价 API，只能把文件复制到
#  一个纯 ASCII 的临时路径再打开。
# ============================================================

import os
import shutil
import tempfile

from utils.paths import OUTPUT_DIR
from utils.system_utils import logger

_ASCII_SAFE = set(b"abcdefghijklmnopqrstuvwxyz"
                  b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/:\\")


def needs_ascii_workaround(path: str) -> bool:
    """路径里有非 ASCII 字符就会被 OpenCV 坑到。"""
    try:
        return bool(set(os.fsencode(path)) - _ASCII_SAFE)
    except Exception:
        return False


def _temp_dir():
    base = os.path.join(OUTPUT_DIR, ".video_io_tmp")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(dir=base)


class SafeVideoCapture:
    """cv2.VideoCapture 的包装：非 ASCII 路径先拷到临时目录再打开。

    用法与 cv2.VideoCapture 一致，支持 with 语句：

        with SafeVideoCapture(path) as cap:
            ok, frame = cap.read()
    """

    def __init__(self, path: str):
        import cv2
        self._cv2 = cv2
        self._tmp_dir = None
        self._tmp_path = None
        self._cap = None

        real = path
        if needs_ascii_workaround(path):
            try:
                self._tmp_dir = _temp_dir()
                ext = os.path.splitext(path)[1] or ".mp4"
                self._tmp_path = os.path.join(self._tmp_dir, "video" + ext)
                shutil.copyfile(path, self._tmp_path)
                real = self._tmp_path
                logger.debug(f"[video_io] 非 ASCII 路径，已复制到临时文件: {self._tmp_path}")
            except Exception as e:
                logger.warning(f"[video_io] 复制临时文件失败，直接用原路径尝试: {e}")
                real = path

        self._cap = cv2.VideoCapture(real)

    # ---- 代理 ----
    def __getattr__(self, item):
        if self._cap is None:
            raise AttributeError(item)
        return getattr(self._cap, item)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def release(self):
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None
            self._tmp_path = None


def probe_video(path: str):
    """返回 (ok, fps, frame_count, width, height)。"""
    try:
        with SafeVideoCapture(path) as cap:
            if not cap.isOpened():
                return False, 0.0, 0, 0, 0
            cv2 = cap._cv2
            return (True,
                    float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    except Exception as e:
        logger.warning(f"[video_io] 探测视频失败 {path}: {e}")
        return False, 0.0, 0, 0, 0


def read_all_frames(path: str):
    """把整条视频读成 BGR 帧列表（失败返回 []）。"""
    frames = []
    try:
        with SafeVideoCapture(path) as cap:
            if not cap.isOpened():
                return []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
    except Exception as e:
        logger.warning(f"[video_io] 读取视频失败 {path}: {e}")
    return frames
