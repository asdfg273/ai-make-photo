# core/arch/signatures.py
"""从权重键名识别架构。支持单文件 safetensors 与 diffusers 目录。"""
from __future__ import annotations
import os
import json
import logging
from collections import Counter
from .registry import REGISTRY, get_arch
from .base import DetectResult

logger = logging.getLogger(__name__)


def _read_keys(path: str) -> list[str]:
    """只读 header，不加载权重（毫秒级）"""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return list(f.keys())

def _read_header(path: str) -> dict[str, list[int]]:
    """只读 header 的键名与 shape，不加载权重张量"""
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            try:
                out[k] = list(f.get_slice(k).get_shape())
            except Exception:
                out[k] = []
    return out


def _has(keys, *frags) -> bool:
    """任一 key 同时包含所有 frag"""
    return any(all(fr in k for fr in frags) for k in keys)


def _any_prefix(keys, *prefixes) -> bool:
    return any(k.startswith(p) for k in keys for p in [*prefixes])

_XATTN_FRAGS = ("attn2_to_k", "attn2_to_v", "attn2.to_k", "attn2.to_v",
                "attn2_processor_to_k", "attn2_processor_to_v")

_DIM_TO_ARCH = {768: "lora_sd15", 1024: "lora_sd21", 2048: "lora_sdxl"}
_DIT_STYLES = (
    # (判定片段, 用于取 hidden 的键后缀, 风格名)
    ("blocks.", "attn.wq.lora_A.weight", "wq/wk/wv/wo + mlp.gate"),
    ("blocks.", "self_attn.q.lora_A.weight", "self_attn/cross_attn + ffn"),
    ("blocks.", "attn.to_q.lora_A.weight", "attn.to_q 风格"),
)
# (hidden, ffn, blocks) → (arch_id 候选, 家族名)
# arch_id 若未在 REGISTRY 注册，自动回退 lora_dit
_DIT_FINGERPRINTS = {
    (5120, 13824, 40): ("lora_wan14b", "Wan 2.1 14B 系"),
    (6144, 16384, 28): ("lora_dit_6144", "DiT GQA hidden=6144/ffn=16384/28层"),
}

# 仅凭 hidden 的弱匹配（ffn/blocks 读不到时用）
_DIT_HIDDEN_HINT = {
    5120: "Wan 14B 系",
    6144: "DiT GQA（hidden=6144）",
}

def _cross_attn_dim(shape) -> tuple[int, int]:
    """从 LoRA down/A 权重形状里取 hidden 维度。

    两种格式：
      kernel (kohya)  lora_down.weight / lora_up.weight : [rank, hidden]
      diffusers       .lora_A.weight / .lora_B.weight   : [rank, hidden]
    两种都是 [rank, hidden]，rank 通常 4~256，hidden 是 768/1024/2048。

    ⚠️ 旧代码取 shp[-1] 拿到的是 **rank**（4/8/16/…），
    所以 _DIM_TO_ARCH 基本永不命中，lora_sd15/sd21/sdxl 三个分支是死代码。
    这里同时看两个轴，只要有一轴落在已知集合里就采用它（更稳，
    也兼容个别导出工具把转置权重写成 [hidden, rank] 的情况）。
    """
    dims = [int(v) for v in shape[:2]]
    for d in dims:
        if d in _DIM_TO_ARCH:
            return d, dims.index(d)
    # 都不在已知集合：取较大的一轴作为 hidden（rank 一定远小于 hidden）
    idx = 0 if dims[0] >= dims[1] else 1
    return dims[idx], idx


def _lora_arch(shapes: dict[str, list[int]]) -> tuple[str, str]:
    ks = list(shapes)

    # ---- DiT 类 LoRA：diffusion_model.blocks.N.*，非 UNet 命名 ----
    if _has(ks, "diffusion_model.blocks.") and not _has(ks, "lora_unet_"):
        hidden, ffn, style = 0, 0, "未知子风格"
        for frag, hint, name in _DIT_STYLES:
            for k, shp in shapes.items():
                if k.endswith(hint) and len(shp) >= 2:
                    hidden, style = int(shp[-1]), name
                    break
            if hidden:
                break
        # 顺带取 FFN 宽度，和 block 数一起作为架构指纹
        for k, shp in shapes.items():
            if ".ffn.0.lora_B.weight" in k or ".mlp.up.lora_B.weight" in k:
                if len(shp) >= 2:
                    ffn = int(shp[0])
                    break
        blocks = len({k.split(".")[2] for k in ks
                      if k.startswith("diffusion_model.blocks.")})

        arch_id, family = "lora_dit", ""
        fp = _DIT_FINGERPRINTS.get((hidden, ffn, blocks))
        if fp:
            cand, family = fp
            if cand in REGISTRY:
                arch_id = cand
        elif hidden in _DIT_HIDDEN_HINT:
            family = _DIT_HIDDEN_HINT[hidden]

        desc = (f"DiT LoRA（{style}，hidden={hidden or '?'}"
                f"，ffn={ffn or '?'}，blocks={blocks}）")
        if family:
            desc = f"{family} — {desc}"
        return arch_id, desc

    for k, shp in shapes.items():
        if "lora_down" not in k and ".lora_A." not in k:
            continue
        if not any(f in k for f in _XATTN_FRAGS):
            continue
        if len(shp) >= 2:
            dim, _axis = _cross_attn_dim(shp)
            arch = _DIM_TO_ARCH.get(dim)
            if arch:
                return arch, f"cross-attn 维度 {dim}（{k.split('.')[0][:40]}）"
            return "lora", f"cross-attn 维度 {dim}（未知架构）"
    return "lora", "LoRA 权重，但未找到 cross-attn 层，无法判定架构"

# 顺序敏感：特化在前，泛化在后
def _classify(shapes: dict[str, list[int]]) -> tuple[str, str]:
    ks = list(shapes)

    # ---- 先算结构特征，后续规则复用 ----
    has_te1 = (_has(ks, "cond_stage_model") or _has(ks, "text_model")
               or _any_prefix(ks, "text_encoder.", "conditioner.embedders.0"))
    has_te2 = (_has(ks, "conditioner.embedders.1")
               or _any_prefix(ks, "text_encoder_2"))
    has_unet = (_has(ks, "model.diffusion_model")
                or _any_prefix(ks, "unet.", "down_blocks.", "input_blocks."))
    is_full_model = has_unet and (has_te1 or has_te2)

    # ---- Anima：DiT 主干 + LLM adapter，无 CLIP ----
    if _has(ks, "llm_adapter") and _has(ks, "adaln_modulation_cross_attn"):
        return "anima", "llm_adapter + adaln_modulation_cross_attn"

    # ---- Flux ----
    if _has(ks, "double_blocks") and _has(ks, "img_attn"):
        return "flux", "double_blocks + img_attn"

    # ---- SD3 ----
    if _has(ks, "joint_blocks"):
        return "sd3", "joint_blocks (MMDiT)"

    # ---- 完整底模优先判定，避免被 LoRA 规则误吞 ----
    if is_full_model:
        if has_te2:
            return "sdxl", "UNet + 双文本编码器（OpenCLIP-G）"
        return "sd15", "UNet + 单 CLIP 文本编码器"

    # ---- LoRA：仅在「非完整模型」时才判，用 shape 细分 ----
    if _has(ks, "lora_down") or _has(ks, "lora_up") or _has(ks, ".lora_A."):
        return _lora_arch(shapes)

    # ---- ControlNet ----
    if not has_te1 and not has_te2 and (_has(ks, "controlnet_cond_embedding")
                                        or _has(ks, "input_hint_block")):
        return "controlnet", "controlnet_cond_embedding / input_hint_block"

    # ---- 独立 VAE ----
    if not has_unet and _any_prefix(ks, "encoder.", "decoder.",
                                    "first_stage_model"):
        return "vae", "仅含 encoder/decoder，无 UNet"

    return "unknown", "无匹配特征"


def detect(path: str) -> DetectResult:
    """入口：单文件或 diffusers 目录，返回 DetectResult"""
    if os.path.isdir(path):
        return _detect_dir(path)
    return _detect_file(path)

def _detect_file(path: str) -> DetectResult:
    size_gb = os.path.getsize(path) / 1024 ** 3
    ext = os.path.splitext(path)[1].lower()

    if ext != ".safetensors":
        # .ckpt 无法快速读 header，退回体积启发式
        arch_id = "sdxl" if size_gb > 5.0 else "sd15"
        logger.warning(f"⚠️ {ext} 格式无法读取键名，按体积推断: {arch_id}")
        return DetectResult(arch_id, get_arch(arch_id),
                            f"{ext} 体积启发式 ({size_gb:.2f}GB)",
                            size_gb=size_gb)

    try:
        shapes = _read_header(path)
    except Exception as e:
        logger.error(f"❌ 读取 safetensors header 失败: {e}")
        return DetectResult("unknown", get_arch("unknown"),
                            f"header 读取失败: {e}", size_gb=size_gb)

    keys = list(shapes)
    arch_id, evidence = _classify(shapes)

    top = ()
    if arch_id == "unknown":
        c = Counter(".".join(k.split(".")[:3]) for k in keys)
        top = tuple(f"{p}({n})" for p, n in c.most_common(5))
        logger.error(f"❌ 未识别架构，键名前缀 top5: {', '.join(top)}")

    return DetectResult(arch_id, get_arch(arch_id), evidence,
                        key_count=len(keys), size_gb=size_gb, top_prefixes=top)


# diffusers 目录：model_index.json 里的 _class_name 直接给出答案
_CLASS_TO_ARCH = {
    "StableDiffusionPipeline": "sd15",
    "StableDiffusionXLPipeline": "sdxl",
    "StableDiffusion3Pipeline": "sd3",
    "FluxPipeline": "flux",
    "AnimaModularPipeline": "anima",
    # MOSS-Image（OpenMOSS 5B DiT）的 diffusers 仓自带一个自定义管线文件
    "MossImagePipeline": "mossimage",
    # 现代视频架构（入口预留，引擎未接通；识别出来才能给出准确提示）
    "LTXPipeline": "ltxvideo",
    "LTXImageToVideoPipeline": "ltxvideo",
    "LTXConditionPipeline": "ltxvideo",
    "WanPipeline": "wan21",
    "WanImageToVideoPipeline": "wan21",
}

# 目录里出现这些**管线文件**就说明是对应架构。比只看 _class_name 更稳：
# 自定义管线仓库有时把 _class_name 写成别的（甚至留空）。
_DIR_PIPELINE_HINTS = {
    "pipeline_moss_image.py": "mossimage",
    "pipeline_ltx_video.py": "ltxvideo",
    "pipeline_ltx_condition.py": "ltxvideo",
    "pipeline_wan.py": "wan21",
    "pipeline_wan_i2v.py": "wan21",
}


def _detect_by_pipeline_file(path: str) -> str:
    for fn, arch in _DIR_PIPELINE_HINTS.items():
        if os.path.isfile(os.path.join(path, fn)):
            return arch
    return ""


def _dir_weight_gb(path: str) -> float:
    """目录里权重文件的合计体积（GB）。目录型权重（MOSS-Image）靠它报体积，
    否则 `detect()` 会给 0 —— 显存策略会以为"这个模型不占权重"。"""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            if not name.lower().endswith((".safetensors", ".bin", ".pt", ".ckpt")):
                continue
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return round(total / (1024 ** 3), 2)


def _detect_dir(path: str) -> DetectResult:
    weight_gb = _dir_weight_gb(path)
    # modular_config.json 是「模块化管线」的入口；本项目的模块化管线目前只有
    # Anima，所以即使里面没写 _class_name 也能判定（基座仓靠它识别，
    # 用户把基座目录丢进 models/puton 时才不会判成"无法识别"）。
    for fn in ("model_index.json", "modular_model_index.json",
               "modular_config.json"):
        fp = os.path.join(path, fn)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ 读取 {fn} 失败: {e}")
            continue
        cls = cfg.get("_class_name", "") or cfg.get("_class", "")
        arch_id = _CLASS_TO_ARCH.get(cls, "unknown")
        if arch_id == "unknown" and fn == "modular_config.json":
            arch_id = "anima"
            return DetectResult(arch_id, get_arch(arch_id),
                                f"{fn}（模块化管线）", size_gb=weight_gb)
        if arch_id == "unknown":
            # _class_name 没在映射表里：再看目录里的自定义管线文件名
            hint = _detect_by_pipeline_file(path)
            if hint:
                return DetectResult(hint, get_arch(hint),
                                    f"{fn}: _class_name={cls} + "
                                    f"自定义管线文件 → {hint}",
                                    size_gb=weight_gb)
        return DetectResult(arch_id, get_arch(arch_id),
                            f"{fn}: _class_name={cls}", size_gb=weight_gb)

    return DetectResult("unknown", get_arch("unknown"),
                        "目录中无 model_index.json", size_gb=weight_gb)
