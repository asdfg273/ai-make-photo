# utils/paths.py
import logging
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

# ── ADetailer 检测模型：**唯一位置** ────────────────────────────────
# 历史问题：市场条目、模型下载器、image_processor 都写的是 models/adetailer/，
# 但旧版本把文件实际下到了 models_cache/adetailer/，于是"文件明明在盘上、
# 市场却显示未安装"，用户点了下载还会重复拉一遍。
# 现在统一到 models/adetailer/（用户能自己放、也能被市场认出来），
# 旧的 models_cache/adetailer/ 只作为**迁移来源**保留一个常量。
ADETAILER_DIR        = os.path.join(MODEL_DIR, "adetailer")
LEGACY_ADETAILER_DIR = os.path.join(CACHE_DIR, "adetailer")

_ADETAILER_EXTS = (".pt", ".onnx", ".engine")

_INIT_FAILURES = []
_RUNTIME_DIRS  = [OUTPUT_DIR, VIDEO_DIR, LORA_DIR, MODEL_DIR,
                  DATA_DIR, LOG_DIR, CACHE_DIR, CONTROLNET_DIR,
                  ADETAILER_DIR, ]

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


# ── ADetailer 模型路径 ─────────────────────────────────────────────
def adetailer_path(name: str) -> str:
    """把任意写法的 ADetailer 模型名解析成**唯一绝对路径**。

    接受这几种写法（历史上配置里都出现过）：
      ``face_yolov8n.pt``
      ``models/adetailer/face_yolov8n.pt``
      ``models\\adetailer\\face_yolov8n.pt``
      ``models_cache/adetailer/face_yolov8n.pt``   （旧位置 → 归一到新位置）
    绝对路径原样返回（用户可能在设置里填了别处的模型）。
    """
    raw = str(name or "").replace("\\", "/").strip()
    if not raw:
        return ADETAILER_DIR

    if os.path.isabs(raw):
        return os.path.normpath(raw)

    # 只取文件名：不管带的是哪种前缀，最终都落在 models/adetailer/
    base = os.path.basename(raw)
    if base.lower().endswith(_ADETAILER_EXTS) or base:
        return os.path.join(ADETAILER_DIR, base)
    return ADETAILER_DIR


def migrate_adetailer_models() -> list:
    """把旧位置 ``models_cache/adetailer/`` 里的模型搬到 ``models/adetailer/``。

    幂等：目标已存在同名文件就不动（不覆盖用户后来下的新版本）。
    返回真正被移动的文件名列表（空列表 = 什么都没做）。
    迁移失败不抛异常 —— 检测器那边还有"旧位置兜底"。
    """
    _log = logging.getLogger(__name__)
    moved = []
    try:
        if not os.path.isdir(LEGACY_ADETAILER_DIR):
            return moved
        os.makedirs(ADETAILER_DIR, exist_ok=True)
        for fname in sorted(os.listdir(LEGACY_ADETAILER_DIR)):
            if not fname.lower().endswith(_ADETAILER_EXTS):
                continue
            src = os.path.join(LEGACY_ADETAILER_DIR, fname)
            dst = os.path.join(ADETAILER_DIR, fname)
            if not os.path.isfile(src) or os.path.exists(dst):
                continue
            try:
                os.replace(src, dst)
            except OSError:
                # 跨盘/被占用 → 退回复制
                import shutil
                shutil.copy2(src, dst)
                try:
                    os.remove(src)
                except OSError:
                    pass
            moved.append(fname)
        if moved:
            _log.info(
                "📦 已把 ADetailer 模型从旧位置 %s 移到 %s：%s",
                LEGACY_ADETAILER_DIR, ADETAILER_DIR, ", ".join(moved))
            # 旧目录空了就顺手删掉，免得下次又当成"旧位置有东西"
            try:
                if not os.listdir(LEGACY_ADETAILER_DIR):
                    os.rmdir(LEGACY_ADETAILER_DIR)
            except OSError:
                pass
    except Exception as e:                     # noqa: BLE001
        _log.warning("ADetailer 模型迁移跳过：%s", e)
    return moved
