import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 用户内容
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "photo")
VIDEO_DIR    = os.path.join(PROJECT_ROOT, "photo", "videos")
LORA_DIR     = os.path.join(PROJECT_ROOT, "loras")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

# 运行时
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
LOG_DIR      = os.path.join(PROJECT_ROOT, "logs")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "models_cache")

# 数据文件
CONFIG_FILE  = os.path.join(PROJECT_ROOT, "app_config.json")
DICT_FILE    = os.path.join(DATA_DIR, "zh_to_en_dict.json")

for d in [OUTPUT_DIR, VIDEO_DIR, LORA_DIR, MODEL_DIR, DATA_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)