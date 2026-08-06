# utils/model_scanner.py
"""
🔍 模型扫描器 —— 按类型管理模型
"""
import os

MODELS_ROOT = "models"

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


def scan_models(model_type: str) -> list[dict]:
    """
    扫描某类型下的所有模型
    返回: [{"name": "xxx.safetensors", "path": "models/sd15/xxx.safetensors", 
            "note": "备注内容", "size_gb": 2.0}, ...]
    """
    sub_dir = os.path.join(MODELS_ROOT, model_type)
    if not os.path.exists(sub_dir):
        return []
    
    exts = MODEL_TYPES[model_type]["ext"]
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