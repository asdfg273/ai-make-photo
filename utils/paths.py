# utils/paths.py
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 用户内容
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "photo")
VIDEO_DIR    = os.path.join(PROJECT_ROOT, "photo", "videos")
LORA_DIR     = os.path.join(PROJECT_ROOT, "loras")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
CONTROLNET_DIR = os.path.join(PROJECT_ROOT, "controlnets")
WEIGHTS_DIR  = os.path.join(PROJECT_ROOT, "weights")

# 运行时
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
LOG_DIR      = os.path.join(PROJECT_ROOT, "logs")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "models_cache")
ASSETS_DIR   = os.path.join(PROJECT_ROOT, "assets")

# 第三方
THIRD_PARTY_DIR = os.path.join(PROJECT_ROOT, "third_party")
SOVITS_DIR      = os.path.join(THIRD_PARTY_DIR, "GPT-SoVITS")
VOICES_DIR      = os.path.join(PROJECT_ROOT, "voices")

# 数据文件
CONFIG_FILE  = os.path.join(PROJECT_ROOT, "app_config.json")
DICT_FILE    = os.path.join(DATA_DIR, "zh_to_en_dict.json")
LOG_FILE     = os.path.join(LOG_DIR, "app.log")

_INIT_FAILURES = []
_RUNTIME_DIRS  = [OUTPUT_DIR, VIDEO_DIR, LORA_DIR, MODEL_DIR,
                  DATA_DIR, LOG_DIR, CACHE_DIR,CONTROLNET_DIR,]

for _d in _RUNTIME_DIRS:
    try:
        os.makedirs(_d, exist_ok=True)
    except OSError as _e:
        _INIT_FAILURES.append((_d, str(_e)))