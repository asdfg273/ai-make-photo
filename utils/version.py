# utils/version.py
"""程序版本与依赖版本采集（元数据写入用）。

以前版本号只硬编码在 main.py 的 setApplicationVersion("6.0") 里，
出图元数据完全没有版本信息 —— 事后排查"这张图是哪个版本、哪套依赖出的"
只能靠猜。这里给出唯一出处。
"""

APP_NAME = "AI 绘画工作站"
APP_VERSION = "6.0"


def collect_versions() -> dict:
    """采集程序与关键依赖版本。任何一项取不到都不应影响出图。"""
    import platform

    v = {"app": APP_VERSION}

    def _mod(name, attr="__version__"):
        try:
            m = __import__(name)
            return str(getattr(m, attr, "unknown"))
        except Exception:
            return "not installed"

    v["python"] = platform.python_version()
    v["torch"] = _mod("torch")
    v["diffusers"] = _mod("diffusers")
    v["transformers"] = _mod("transformers")
    v["safetensors"] = _mod("safetensors")
    v["pillow"] = _mod("PIL")
    v["numpy"] = _mod("numpy")

    # CUDA / GPU
    try:
        import torch
        v["cuda"] = getattr(torch.version, "cuda", None) or "none"
        if torch.cuda.is_available():
            v["gpu"] = torch.cuda.get_device_name(0)
            v["gpu_total_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}"
        else:
            v["gpu"] = "cpu"
    except Exception:
        v["cuda"] = "unknown"

    return v


def versions_text() -> str:
    """一行式摘要，便于塞进 A1111 兼容串。"""
    v = collect_versions()
    return (f"App: {v.get('app')}, diffusers: {v.get('diffusers')}, "
            f"torch: {v.get('torch')}, transformers: {v.get('transformers')}")
