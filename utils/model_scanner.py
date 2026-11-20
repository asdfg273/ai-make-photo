# utils/model_scanner.py
"""
🔍 模型扫描器 —— 按类型管理模型
"""
import os
from utils import paths
from core.arch import REGISTRY

MODELS_ROOT = paths.MODEL_DIR

_DEFAULT_EXT = [".safetensors", ".ckpt", ".gguf", ".pt"]
from utils import paths

MODELS_ROOT = paths.MODEL_DIR

# 模型类型配置
MODEL_TYPES = {
    "sd15":  {"label": "SD 1.5",   "ext": [".safetensors", ".ckpt"]},
    "sdxl":  {"label": "SDXL",     "ext": [".safetensors"]},
    "sd3":   {"label": "SD3/SD3.5","ext": [".safetensors"]},
    "flux":  {"label": "Flux",     "ext": [".safetensors", ".gguf"]},
}


def ensure_model_dirs():
    """确保所有模型子目录存在"""
    for t in MODEL_TYPES:
        os.makedirs(os.path.join(MODELS_ROOT, t), exist_ok=True)


def _build_model_types() -> dict:
    """从 REGISTRY 派生扫描配置，避免和架构定义脱节。
    按 model_subdir 去重（如 sd15/sd21 共用同一目录）。"""
    result = {}
    for arch_id, info in REGISTRY.items():
        if not info.caps.is_base_model:
            continue
        subdir = info.model_subdir or arch_id
        if subdir in result:
            continue
        result[subdir] = {"label": info.display_name, "ext": _DEFAULT_EXT}
    return result


MODEL_TYPES = _build_model_types()

def _dir_size_gb(path: str, max_files: int = 4000) -> float:
    """目录体积（GB）。只统计常见权重文件，避免把 .git 之类也算进去。

    max_files 只防"误指向一个巨大目录"这种极端情况；正常模型目录
    （哪怕 25GB / 几十个分片）会完整累加 —— 之前的实现到 4GB 就提前返回，
    导致 MOSS-Image 这种大模型在下拉框里显示成 4.63 GB。
    """
    total = 0
    seen = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            if not name.lower().endswith((".safetensors", ".bin", ".pt", ".ckpt")):
                continue
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
            seen += 1
            if seen >= max_files:
                return round(total / (1024 ** 3), 2)
    return round(total / (1024 ** 3), 2)


def _is_diffusers_dir(path: str) -> bool:
    """diffusers 目录（官方权重常常是目录而不是单文件，如 MOSS-Image）。"""
    if not os.path.isdir(path):
        return False
    return any(os.path.isfile(os.path.join(path, m))
               for m in ("model_index.json", "modular_model_index.json"))


def scan_models(model_type: str) -> list[dict]:
    """
    扫描某类型下的所有模型
    返回: [{"name": "xxx.safetensors", "path": "models/sd15/xxx.safetensors", 
            "note": "备注内容", "size_gb": 2.0}, ...]

    也支持 **diffusers 目录**（MOSS-Image 这类），此时 name 是目录名。
    """
    sub_dir = os.path.join(MODELS_ROOT, model_type)
    if not os.path.exists(sub_dir):
        return []

    exts = MODEL_TYPES.get(model_type, {}).get("ext", _DEFAULT_EXT)
    results = []

    for fname in sorted(os.listdir(sub_dir)):
        fpath = os.path.join(sub_dir, fname)

        if _is_diffusers_dir(fpath):
            from utils.model_previews import find_model_preview
            note = ""
            txt_path = os.path.join(fpath, "README.txt")
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        note = f.read().strip()[:200]
                except Exception:
                    pass
            results.append({
                "name": fname,
                "path": fpath,
                "preview": find_model_preview(fpath) or "",
                "note": note,
                "size_gb": _dir_size_gb(fpath),
                "type": model_type,
                "is_dir": True,
            })
            continue

        if not os.path.isfile(fpath):
            continue
        if not any(fname.lower().endswith(e) for e in exts):
            continue

        # 读取同名 .txt 备注
        note = ""
        txt_path = os.path.splitext(fpath)[0] + ".txt"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    note = f.read().strip()
            except Exception:
                pass

        from utils.model_previews import find_model_preview
        size_gb = os.path.getsize(fpath) / (1024 ** 3)
        results.append({
            "name": fname,
            "path": fpath,
            "preview": find_model_preview(fpath) or "",
            "note": note,
            "size_gb": round(size_gb, 2),
            "type": model_type,
        })

    return results


def scan_all_models() -> dict:
    """扫描所有类型 → {type: [models...]}"""
    ensure_model_dirs()
    return {t: scan_models(t) for t in MODEL_TYPES}


def find_model_path(model_name: str) -> str | None:
    """全目录查找一个模型的真实路径（兼容旧配置）"""
    # 先在子目录找
    for t in MODEL_TYPES:
        p = os.path.join(MODELS_ROOT, t, model_name)
        if os.path.exists(p):
            return p
    # 兼容旧位置（根目录）
    p = os.path.join(MODELS_ROOT, model_name)
    if os.path.exists(p):
        return p
    return None
