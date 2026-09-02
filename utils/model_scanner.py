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

def scan_models(model_type: str) -> list[dict]:
    """
    扫描某类型下的所有模型
    返回: [{"name": "xxx.safetensors", "path": "models/sd15/xxx.safetensors", 
            "note": "备注内容", "size_gb": 2.0}, ...]
    """
    sub_dir = os.path.join(MODELS_ROOT, model_type)
    if not os.path.exists(sub_dir):
        return []

    exts = MODEL_TYPES.get(model_type, {}).get("ext", _DEFAULT_EXT)
    results = []
    
    for fname in sorted(os.listdir(sub_dir)):
        fpath = os.path.join(sub_dir, fname)
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
        
        size_gb = os.path.getsize(fpath) / (1024 ** 3)
        results.append({
            "name": fname,
            "path": fpath,
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