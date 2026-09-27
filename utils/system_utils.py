import os
import sys
import time
import logging
import threading
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from typing import Optional

from utils import paths

_LOG_READY = False
_LOG_LOCK = threading.Lock()


class _SafeStreamHandler(logging.StreamHandler):
    """控制台/重定向流的编码可能不是 UTF-8（中文 Windows 是 GBK/cp936）。

    logging 默认的 emit 遇到无法编码的字符会抛 UnicodeEncodeError，
    再由 handleError 打印一大段 `--- Logging error ---` 回溯 —— 本项目
    日志里 emoji（📦✅⚠️）极多，一旦 stdout 被重定向到文件或管道就会
    刷屏。这里降级为 "用流自身编码 + errors='replace'"：中文正常保留，
    emoji 变成 '?'，绝不抛异常。
    """

    def emit(self, record):
        stream = self.stream
        if stream is None:
            return
        try:
            msg = self.format(record)
            enc = getattr(stream, "encoding", None) or "utf-8"
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                stream.write(msg.encode(enc, "replace").decode(enc, "replace")
                            + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(level=logging.INFO, force: bool = True) -> None:
    """配置根日志。幂等，可安全重复调用。

    不用 basicConfig：它在 root logger 已有 handler 时会静默失效，
    而 transformers / ChatTTS 都可能先动 root logger。
    """
    global _LOG_READY
    with _LOG_LOCK:
        if _LOG_READY and not force:
            return

        root = logging.getLogger()
        root.setLevel(level)

        # 清掉第三方库或重复调用留下的 handler
        for h in root.handlers[:]:
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        fmt = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        try:
            os.makedirs(paths.LOG_DIR, exist_ok=True)
            fh = RotatingFileHandler(
                paths.LOG_FILE,              # 绝对路径，chdir 免疫
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            pass  # 只读目录时退化为仅控制台

        # pythonw / 无控制台启动时 stdout 可能为 None，此时跳过控制台 handler
        console = sys.stdout if sys.stdout is not None else sys.stderr
        if console is not None:
            sh = _SafeStreamHandler(console)
            sh.setFormatter(fmt)
            root.addHandler(sh)

        # 顺带让**普通 print** 也不会因编码崩掉。
        # 中文 Windows 上 stdout 被重定向时是 GBK，`print("✅ ...")` 会抛
        # UnicodeEncodeError；在测试脚本/批处理里这会让进程以非 0 退出，
        # 看起来像"测试失败"，其实逻辑全过。
        # 只放宽 errors，不改 encoding —— 中文在 GBK 下仍正常渲染，
        # emoji 降级成 '?'。
        for _s in (sys.stdout, sys.stderr):
            try:
                if _s is not None and hasattr(_s, "reconfigure"):
                    _s.reconfigure(errors="replace")
            except Exception:
                pass

        _LOG_READY = True


logger = logging.getLogger(__name__)

@contextmanager
def performance_timer(name: str, log_level: str = "info"):
    start = time.perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        elapsed = time.perf_counter() - start
        status = "失败后" if failed else ""
        message = f"⏱️ {name} {status}耗时: {elapsed:.2f}秒"
        if failed:
            logger.warning(message)
        elif log_level == "debug":
            logger.debug(message)
        elif log_level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

def ensure_directory(directory: str) -> None:
    """确保目录存在。"""
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"📁 创建目录: {directory}")


class SingletonMeta(type):
    """线程安全的单例元类。"""
    _instances = {}
    _meta_lock = threading.RLock()
    def __call__(cls, *args, **kwargs):
        with SingletonMeta._meta_lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

def get_file_size(file_path: str) -> str:
    """获取文件大小的可读字符串"""
    if not os.path.exists(file_path):
        return "文件不存在"

    size_bytes = os.path.getsize(file_path)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def generate_unique_filename(prefix: str = "output", extension: str = "png") -> str:
    """生成唯一的文件名"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def clamp(value: float, min_val: float, max_val: float) -> float:
    """将值限制在指定范围内"""
    return max(min_val, min(value, max_val))

def format_seed(seed: int) -> str:
    """格式化种子值为十六进制字符串"""
    return f"0x{seed:08X}"

def get_available_memory() -> Optional[float]:
    """获取可用内存（GB）"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except ImportError:
        return None

def get_gpu_memory() -> Optional[dict]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return {
            "name":  torch.cuda.get_device_name(0),
            "total": total / (1024 ** 3),
            "used":  (total - free) / (1024 ** 3),
            "free":  free / (1024 ** 3),
        }
    except Exception as e:
        logger.debug(f"GPU 信息读取失败: {e}")
        return None

def log_system_info():
    """记录系统信息"""
    logger.info("=" * 50)
    logger.info("🎯 系统信息")
    logger.info("=" * 50)

    import platform
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"Python版本: {platform.python_version()}")

    gpu_info = get_gpu_memory()
    if gpu_info:
        logger.info(f"GPU 内存 - 总: {gpu_info['total']:.2f}GB, 已用: {gpu_info['used']:.2f}GB, 可用: {gpu_info['free']:.2f}GB")

    mem_info = get_available_memory()
    if mem_info:
        logger.info(f"可用内存: {mem_info:.2f}GB")

    logger.info("=" * 50)

def log_gpu_info():
    """torch 导入完成后调用"""
    gpu = get_gpu_memory()
    if gpu:
        logger.info(f"GPU: {gpu.get('name','?')} — 总 {gpu['total']:.2f}GB / "
                    f"已用 {gpu['used']:.2f}GB / 可用 {gpu['free']:.2f}GB")
    else:
        logger.warning("⚠️ 未检测到可用 CUDA 设备，将使用 CPU（速度极慢）")

