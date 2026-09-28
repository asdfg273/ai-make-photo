# utils/image_metadata.py
"""出图元数据的**唯一实现**：采集 → 写 PNG → 兼容 A1111 参数串。

为什么要有这个模块
------------------
以前有两套各写各的：
  · `app_generation._gt_save_image` 手写 16 个 PngInfo 键；
  · `app_generation._save_with_meta`（XY 网格 / 放大用）只拼了一个
    `parameters` 文本块，**没有单独键**，而且键名还不一致
    （一个用 `neg`、一个用 `negative`）。
于是不同入口出的图元数据完整度天差地别，XY 图连提示词都读不全。

这里统一成：一份 dict → 写进 PNG 文本块（可被本程序与工具读取）
+ A1111 兼容的 `parameters` 串（可被 WebUI / Civitai 识别）。

完整字段见 `build_metadata()`。任何一项取不到都写 "unknown"，
绝不因为元数据采集失败而让出图失败。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os

from PIL.PngImagePlugin import PngInfo

logger = logging.getLogger(__name__)

# 单个文本块上限，超长提示词不至于把 PNG 撑爆
_MAX_VAL = 8000

# 哈希缓存：2GB 底模每次出图都重算 sha256 是不可接受的
_CACHE_FILE = None
_CACHE: dict = {}
_CACHE_LOADED = False


def _cache_file() -> str:
    global _CACHE_FILE
    if _CACHE_FILE is None:
        from utils import paths
        _CACHE_FILE = os.path.join(paths.CACHE_DIR, "file_hashes.json")
    return _CACHE_FILE


def _load_cache():
    global _CACHE_LOADED, _CACHE
    if _CACHE_LOADED:
        return
    _CACHE_LOADED = True
    try:
        with open(_cache_file(), encoding="utf-8") as f:
            _CACHE = json.load(f)
    except Exception:
        _CACHE = {}


def _save_cache():
    try:
        os.makedirs(os.path.dirname(_cache_file()), exist_ok=True)
        with open(_cache_file(), "w", encoding="utf-8") as f:
            json.dump(_CACHE, f)
    except Exception as e:
        logger.debug(f"哈希缓存写入失败（忽略）: {e}")


def file_sha256(path: str, block: int = 1 << 20) -> str:
    """文件的 sha256（十六进制）。

    结果按 (路径, 大小, mtime) 缓存到磁盘 —— 底模动辄 2~7GB，
    每次出图都重算会让保存卡好几秒。同一文件只算一次。
    失败返回 "unknown"，绝不抛异常。
    """
    if not path or not os.path.exists(path):
        return "unknown"
    try:
        st = os.stat(path)
        key = os.path.abspath(path)
        sig = f"{st.st_size}:{int(st.st_mtime)}"

        _load_cache()
        hit = _CACHE.get(key)
        if isinstance(hit, dict) and hit.get("sig") == sig and hit.get("sha256"):
            return hit["sha256"]

        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(block)
                if not chunk:
                    break
                h.update(chunk)
        digest = h.hexdigest()
        _CACHE[key] = {"sig": sig, "sha256": digest}
        _save_cache()
        return digest
    except Exception as e:
        logger.warning(f"⚠️ 计算文件哈希失败 {path}: {e}")
        return "unknown"


def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _fmt(v, digits=2) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return _s(v)


def build_metadata(host, ctx, seed, *, image=None, kind="txt2img",
                   index=0, total=1, extra=None) -> dict:
    """采集全部元数据。返回扁平 dict（键名即 PNG 文本块名）。

    `host` 需要能提供 config / ai / 各 UI 控件；缺失项一律写 "unknown"。
    """
    from utils.version import collect_versions

    cfg = getattr(host, "config", None)
    ai = getattr(host, "ai", None)
    ctx = ctx or {}
    meta: dict = {}

    def put(k, v):
        meta[k] = _s(v) if not isinstance(v, str) else v

    # ── 提示词：原始 vs 最终 ──────────────────────────────
    raw_list = ctx.get("parsed_raw_prompts") or []
    en_list = ctx.get("en_prompts") or []
    put("prompt", raw_list[index] if index < len(raw_list) else "")
    put("prompt_final", en_list[index] if index < len(en_list) else "")
    put("negative", ctx.get("raw_neg", ""))
    put("negative_final", ctx.get("en_neg", ""))
    # 原始整段（多提示词时保留用户原输入）
    put("prompt_raw_all", ctx.get("raw_prompt", ""))
    put("prompt_mode", getattr(getattr(ai, "arch_profile", None),
                               "prompt_mode", "unknown"))
    put("clip_skip", getattr(cfg, "clip_skip", 0))

    # ── 模型 ─────────────────────────────────────────────
    model_name = ctx.get("model_name") or getattr(ai, "current_model_name", "")
    model_path = (getattr(ai, "current_model_path", None)
                  or ctx.get("model_path") or "")
    put("model", model_name)
    put("model_path", model_path or "unknown")
    digest = file_sha256(model_path) if model_path else "unknown"
    put("model_hash", digest)
    put("model_hash_short", digest[:10] if digest != "unknown" else "unknown")
    put("architecture", getattr(ai, "current_arch_id", None) or "unknown")
    put("vpred", getattr(ai, "is_vpred", False))
    put("zero_snr", getattr(ai, "zero_snr", None))

    # ── LoRA ────────────────────────────────────────────
    loras = ctx.get("lora_meta_info") or []
    if isinstance(loras, (list, tuple)):
        loras = list(loras)
    elif loras:
        loras = [str(loras)]
    else:
        loras = []
    put("loras", ", ".join(str(x) for x in loras))
    put("lora_count", len(loras))
    put("lora_tags", ctx.get("lora_tags", ""))

    # ── ControlNet / IP-Adapter ─────────────────────────
    cn_type = (getattr(ai, "current_cn_type", None)
               or ctx.get("cn_type") or "")
    put("controlnet_type", cn_type or "none")
    put("controlnet_weight", _fmt(ctx.get("cn_strength", 0.0)))
    put("pose_transfer_used", bool(ctx.get("pose_transfer_used", False)))
    put("pose_image", ctx.get("pose_image_path", "") or "none")
    use_ipa = bool(ctx.get("use_ipa", False))
    put("ip_adapter_enabled", use_ipa)
    put("ip_adapter_variant", ctx.get("ipa_variant", "none") if use_ipa else "none")
    put("ip_adapter_weight", _fmt(ctx.get("ipa_scale", 0.0)) if use_ipa else "none")

    # ── 采样 ────────────────────────────────────────────
    put("sampler", ctx.get("sampler_name", "unknown"))
    sched = None
    try:
        pipe = getattr(ai, "txt2img_pipe", None)
        if pipe is not None and getattr(pipe, "scheduler", None) is not None:
            sched = type(pipe.scheduler).__name__
    except Exception:
        pass
    put("scheduler", sched or "unknown")
    put("steps", ctx.get("steps", ""))
    put("cfg", ctx.get("cfg", ""))
    put("seed", seed)
    put("width", ctx.get("width", ""))
    put("height", ctx.get("height", ""))
    put("size", f"{ctx.get('width', '')}x{ctx.get('height', '')}")
    put("batch_index", index + 1)
    put("batch_total", total)
    put("guidance_note", "")

    # ── 流程 ────────────────────────────────────────────
    put("workflow", kind)
    put("strength", ctx.get("strength", ""))
    put("init_image", ctx.get("init_image_path") or "none")
    put("reference_only", bool(ctx.get("use_reference_only", False)))
    put("ref_fidelity", ctx.get("ref_fidelity", ""))

    # ── Hires.fix ───────────────────────────────────────
    try:
        hires_on = bool(host.chk_hires.isChecked())
    except Exception:
        hires_on = bool(getattr(cfg, "use_hires", False))
    put("hires_enabled", hires_on)
    if hires_on:
        def _txt(w, default=""):
            try:
                return w.currentText()
            except Exception:
                return default

        def _flt(w, default=0.0):
            try:
                return w.float_value()
            except Exception:
                return default
        put("hires_scale", _txt(getattr(host, "combo_hires_scale", None),
                                "2.0"))
        put("hires_denoise", _fmt(_flt(getattr(host, "scale_hires_denoise", None),
                                       getattr(cfg, "hires_denoise", 0.45))))
        put("hires_upscaler", _txt(getattr(host, "combo_hires_upscaler", None),
                                   "Latent"))
        try:
            # 用 float 算：1.5 倍不能用 split(".")[0] 取整数部分（会变成 1 倍）
            factor = float(meta["hires_scale"])
            w = int(round(int(ctx.get("width", 0)) * factor))
            h = int(round(int(ctx.get("height", 0)) * factor))
            put("hires_size", f"{w}x{h}")
        except Exception:
            put("hires_size", "unknown")
    else:
        put("hires_scale", "off")
        put("hires_denoise", "off")
        put("hires_upscaler", "off")
        put("hires_size", "off")

    # ── ADetailer ───────────────────────────────────────
    def _checked(name):
        try:
            return bool(getattr(host, name).isChecked())
        except Exception:
            return False

    def _float(name, default=0.0):
        try:
            return getattr(host, name).float_value()
        except Exception:
            return default

    ad_face = _checked("chk_use_adetailer")
    ad_hand = _checked("chk_use_ad_hand")
    put("adetailer_enabled", ad_face or ad_hand)
    put("adetailer_face", ad_face)
    put("adetailer_face_strength",
        _fmt(_float("scale_adetailer_strength",
                    getattr(cfg, "adetailer_strength", 0.35))) if ad_face else "off")
    put("adetailer_hand", ad_hand)
    put("adetailer_hand_strength",
        _fmt(_float("scale_ad_hand", getattr(cfg, "ad_hand_strength", 0.25)))
        if ad_hand else "off")
    put("adetailer_hand_blend",
        _fmt(_float("scale_ad_hand_blend", getattr(cfg, "ad_hand_blend", 0.65)))
        if ad_hand else "off")

    # ── FreeU ───────────────────────────────────────────
    put("freeu_enabled", bool(getattr(cfg, "freeu_enabled", False)))

    # ── 水印 ────────────────────────────────────────────
    try:
        from utils.watermark import describe as _wm_desc
        put("watermark", _wm_desc(cfg))
    except Exception:
        put("watermark", "unknown")

    # ── 版本 ────────────────────────────────────────────
    for k, v in collect_versions().items():
        put(f"version_{k}", v)

    # ── 其它 ────────────────────────────────────────────
    put("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if extra:
        for k, v in extra.items():
            put(k, v)

    return meta


def a1111_parameters(meta: dict) -> str:
    """拼 A1111 兼容串，便于 WebUI / Civitai 读取。"""
    parts = [
        meta.get("prompt_final") or meta.get("prompt", ""),
        f"Negative prompt: {meta.get('negative_final') or meta.get('negative', '')}",
        (f"Steps: {meta.get('steps', '')}, "
         f"Sampler: {meta.get('sampler', '')}, "
         f"Schedule type: {meta.get('scheduler', '')}, "
         f"CFG scale: {meta.get('cfg', '')}, "
         f"Seed: {meta.get('seed', '')}, "
         f"Size: {meta.get('size', '')}, "
         f"Model hash: {meta.get('model_hash_short', '')}, "
         f"Model: {meta.get('model', '')}"),
    ]
    if meta.get("clip_skip"):
        parts[-1] += f", Clip skip: {meta['clip_skip']}"
    if meta.get("loras"):
        parts[-1] += f", LoRAs: {meta['loras']}"
    if meta.get("hires_enabled") == "true":
        parts[-1] += (f", Hires upscale: {meta.get('hires_scale')}"
                      f", Hires upscaler: {meta.get('hires_upscaler')}"
                      f", Denoising strength: {meta.get('hires_denoise')}")
    if meta.get("adetailer_enabled") == "true":
        parts.append(f"ADetailer: face={meta.get('adetailer_face')} "
                     f"({meta.get('adetailer_face_strength')}), "
                     f"hand={meta.get('adetailer_hand')} "
                     f"({meta.get('adetailer_hand_strength')})")
    if meta.get("controlnet_type") and meta["controlnet_type"] != "none":
        parts.append(f"ControlNet: {meta['controlnet_type']} "
                     f"weight={meta.get('controlnet_weight')}")
    if meta.get("ip_adapter_enabled") == "true":
        parts.append(f"IP-Adapter: {meta.get('ip_adapter_variant')} "
                     f"weight={meta.get('ip_adapter_weight')}")
    parts.append(f"App: {meta.get('version_app')} | "
                 f"diffusers: {meta.get('version_diffusers')} | "
                 f"torch: {meta.get('version_torch')}")
    return "\n".join(parts)


def to_pnginfo(meta: dict) -> PngInfo:
    """dict → PngInfo。单键失败不影响其它键。"""
    info = PngInfo()
    for k, v in meta.items():
        try:
            s = "" if v is None else str(v)
            if len(s) > _MAX_VAL:
                s = s[:_MAX_VAL] + "...(truncated)"
            info.add_text(str(k), s)
        except Exception as e:
            logger.warning(f"⚠️ 元数据 [{k}] 写入失败: {e}")
    try:
        info.add_text("parameters", a1111_parameters(meta))
    except Exception as e:
        logger.warning(f"⚠️ A1111 参数串写入失败: {e}")
    return info


def enrich_metadata(meta: dict, model_path: str = "") -> dict:
    """补全通用字段：模型哈希 + 依赖版本 + 水印说明。

    工作流路径（nodes/image_nodes.py 的 SaveImage）是自己拼元数据的，
    不像引导式界面那样走 `build_metadata()`，于是 **model_hash 和版本号
    会缺**（实测出来是 unknown）。这里提供同一个补全入口，两条路径共用，
    避免又出现"两套元数据、字段不一致"的老问题。
    """
    if not isinstance(meta, dict):
        return meta

    mp = model_path or meta.get("model_path") or ""
    if mp and meta.get("model_hash") in (None, "", "unknown"):
        d = file_sha256(mp)
        meta["model_hash"] = d
        meta["model_hash_short"] = d[:10] if d != "unknown" else "unknown"

    try:
        from utils.version import collect_versions
        for k, v in collect_versions().items():
            meta.setdefault(f"version_{k}", v)
    except Exception as e:
        logger.debug(f"版本信息采集失败（忽略）: {e}")

    return meta


def save_image(host, image, save_path: str, meta: dict | None = None):
    """带元数据（+水印）保存。返回最终保存路径。

    水印在这里统一施加 —— 所有保存入口共用一条路径，
    避免出现"主图有水印、放大图没水印"这种不一致。
    """
    cfg = getattr(host, "config", None)
    try:
        from utils.watermark import apply_watermark
        image = apply_watermark(image, cfg) if cfg is not None else image
    except Exception as e:
        logger.warning(f"⚠️ 水印处理失败（继续保存原图）: {e}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    ext = os.path.splitext(save_path)[1].lower()

    # 补全通用字段（模型哈希/版本）—— 工作流路径与引导式路径共用
    try:
        meta = enrich_metadata(dict(meta) if meta else {},
                              (meta or {}).get("model_path", "")) or meta
    except Exception as e:
        logger.debug(f"元数据补全失败（忽略）: {e}")

    try:
        if ext == ".png":
            pnginfo = to_pnginfo(meta) if meta else None
            if pnginfo is not None:
                image.save(save_path, format="PNG", pnginfo=pnginfo,
                           optimize=False, compress_level=4)
            else:
                image.save(save_path, format="PNG", optimize=False,
                           compress_level=4)
        else:
            # ⚠️ 扩展名不能直接当 PIL 格式名：`.jpg` → "JPG" 不是合法格式，
            # PIL 抛异常后会被下面的 except 降级成 image.save(path)，
            # **元数据就悄悄丢了**（节点侧实测到过）。
            _EXT_FMT = {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP",
                        "bmp": "BMP", "tif": "TIFF", "tiff": "TIFF",
                        "png": "PNG"}
            fmt = _EXT_FMT.get(ext.lstrip(".").lower(),
                               ext.lstrip(".").upper() or "PNG")
            image.save(save_path, format=fmt, quality=95)
        return save_path
    except Exception as e:
        logger.warning(f"⚠️ 带元数据保存失败，降级为普通保存: {e}")
        try:
            image.save(save_path)
            return save_path
        except Exception as e2:
            logger.error(f"❌ 保存失败: {save_path}: {e2}")
            return None
