import os
import time
import logging
from contextlib import contextmanager
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@contextmanager
def performance_timer(name: str, log_level: str = "info"):
    """性能计时器装饰器"""
    start = time.time()
    yield
    elapsed = time.time() - start
    
    message = f"⏱️ {name} 耗时: {elapsed:.2f}秒"
    if log_level == "debug":
        logger.debug(message)
    elif log_level == "warning":
        logger.warning(message)
    else:
        logger.info(message)

def ensure_directory(directory: str) -> None:
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"📁 创建目录: {directory}")

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
    """获取GPU内存信息"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "total": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
                "used": torch.cuda.memory_allocated(0) / (1024 ** 3),
                "free": torch.cuda.memory_free(0) / (1024 ** 3)
            }
    except Exception:
        pass
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

class SingletonMeta(type):
    """单例元类"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
