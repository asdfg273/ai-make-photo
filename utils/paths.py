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


def resolve_output_dir(config=None):
    """返回有效输出目录（Path）。**全项目唯一实现。**

    - 优先 config.output_dir（非空即用，不存在则创建）
    - 未配置 / 创建失败时回落 OUTPUT_DIR（photo/）

    ⚠️ 历史遗留值 "outputs/" 特殊处理
    ---------------------------------
    下拉框 `combo_output_dir` 的占位默认值一直是字面量 "outputs/"，
    `AppConfig.output_dir` 的默认值也是它。也就是说**绝大多数用户
    配置里存的是这个占位值，而不是自己选过的目录**。

    如果直接照它执行，升级后主出图会从 photo/ 改写到 outputs/，
    画廊跟着换目录 → 用户会看到"我的图全没了"。
    所以：把 "outputs/" 视为「未设置」，**除非**那个目录里确实已有文件
    （说明用户真的在用它）。
    """
    from pathlib import Path

    custom = getattr(config, "output_dir", None) if config is not None else None
    cand = str(custom).strip() if custom else ""

    if cand and cand.replace("\\", "/").rstrip("/").lower() == "outputs":
        probe = Path(cand).expanduser()
        if not probe.is_absolute():
            probe = Path(PROJECT_ROOT) / probe
        try:
            has_content = probe.is_dir() and any(probe.iterdir())
        except OSError:
            has_content = False
        if not has_content:
            cand = ""            # 占位值 → 当作未设置

    if cand:
        try:
            p = Path(cand).expanduser()
            if not p.is_absolute():
                # 相对路径锚定项目根，避免受 CWD 影响
                p = Path(PROJECT_ROOT) / p
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError as _e:
            _INIT_FAILURES.append((cand, str(_e)))
    return Path(OUTPUT_DIR)


def default_output_dir() -> str:
    """给 UI 下拉框用的**默认输出目录文本**（不再用占位值 "outputs/"）。

    以前下拉框第一项是 "outputs/"，导致：
      · 新用户看到的是一个跟实际 photo/ 不一致的目录名；
      · 保存配置后又把占位值写回 config，形成"设置不生效"的观感。
    """
    return OUTPUT_DIR


def host_output_dir(host) -> str:
    """取宿主对象（App / 各 Mixin）当前的输出目录字符串。

    Mixin 拿不到 config 时回落 OUTPUT_DIR，绝不抛异常 ——
    输出目录这种东西不该让主流程挂掉。

    ⚠️ 为什么需要它：`OUTPUT_DIR` 常量在 app_events / app_generation 里
    被硬编码用了十几处，主出图写了配置目录、tiled 放大和临时文件却仍写
    photo/，同一个设置两套去向。统一从这一个函数取。
    """
    try:
        return str(resolve_output_dir(getattr(host, "config", None)))
    except Exception:
        return OUTPUT_DIR
