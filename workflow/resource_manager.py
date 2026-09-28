# workflow/resource_manager.py
"""资源清单：从工作流里汇总"这条流程需要什么"，并检查本地是否齐备。

与下载策略的关系（重要）
------------------------
本项目约定「模型由用户自行下载」。所以这里**只做两件事**：
  1. 执行前把缺什么列清楚；
  2. 给 UI 提供"去扩展市场下载"的入口 id。
执行工作流时**只读本地文件**，绝不联网、绝不静默下载。

`requires` 字符串约定
---------------------
    "cuda"                  需要 GPU
    "model"                 需要底模（用户自备）
    "model:sdxl"            需要特定架构的底模
    "vae:<name>"            需要某个 VAE
    "asset:<key>"           可下载资产，key 映射到扩展市场条目
    "tool:<name>"           外部工具（如 rife / ffmpeg）
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# asset key -> 扩展市场条目 id（用于"去下载"按钮）
ASSET_TO_EXTENSION = {
    "adetailer_face": "adetailer_face",
    "adetailer_hand": "adetailer_hand",
    "controlnet_openpose": "controlnet_openpose",
    "controlnet_canny": "controlnet_canny",
    "controlnet_depth": "controlnet_depth",
    "controlnet_openpose_sdxl": "controlnet_openpose_sdxl",
    "controlnet_canny_sdxl": "controlnet_canny_sdxl",
    "controlnet_depth_sdxl": "controlnet_depth_sdxl",
    "ip_adapter_sd15": "ip_adapter_sd15",
    "rife": "rife_v46",
    "qwen2vl": "qwen2vl_2b",
    "nllb": "nllb_600m",
    "chattts": "chattts",
    "gpt_sovits": "gpt_sovits",
    "flux_engine": "flux_engine_schnell",
}

ASSET_LABELS = {
    "adetailer_face": "ADetailer 人脸检测器",
    "adetailer_hand": "ADetailer 手部检测器",
    "controlnet_openpose": "ControlNet OpenPose (SD1.5)",
    "controlnet_canny": "ControlNet Canny (SD1.5)",
    "controlnet_depth": "ControlNet Depth (SD1.5)",
    "controlnet_openpose_sdxl": "ControlNet OpenPose (SDXL)",
    "controlnet_canny_sdxl": "ControlNet Canny (SDXL)",
    "controlnet_depth_sdxl": "ControlNet Depth (SDXL)",
    "ip_adapter_sd15": "IP-Adapter (SD1.5)",
    "rife": "RIFE 帧插值",
    "qwen2vl": "Qwen2-VL 提示词增强",
    "nllb": "NLLB 翻译模型",
    "chattts": "ChatTTS 语音合成",
    "gpt_sovits": "GPT-SoVITS 日语克隆",
    "flux_engine": "Flux 引擎组件",
}


@dataclass
class Requirement:
    kind: str                  # cuda | model | vae | asset | tool
    key: str                   # 具体标识
    label: str = ""
    optional: bool = True
    present: bool = False
    size_mb: int = 0
    extension_id: str = ""     # 有则可一键跳扩展市场
    nodes: List[str] = field(default_factory=list)
    hint: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "key": self.key,
                "label": self.label or self.key, "optional": self.optional,
                "present": self.present, "size_mb": self.size_mb,
                "extension_id": self.extension_id,
                "nodes": list(self.nodes), "hint": self.hint}


# ---------------------------------------------------------------- 本地检查
def _check_asset(key: str) -> bool:
    """该资产本地是否就绪。复用扩展市场的检测，避免两处判据漂移。"""
    ext_id = ASSET_TO_EXTENSION.get(key)
    if not ext_id:
        return False
    try:
        from utils.extension_manager import is_installed
        return bool(is_installed(ext_id))
    except Exception as e:
        logger.debug(f"资源检测失败 {key}: {e}")
        return False


def _check_vae(name: str) -> bool:
    try:
        from utils import paths
        for root, _dirs, files in os.walk(paths.MODEL_DIR):
            for f in files:
                if name.lower() in f.lower() and f.endswith(
                        (".safetensors", ".ckpt", ".pt")):
                    return True
        for root, _dirs, files in os.walk(paths.WEIGHTS_DIR):
            for f in files:
                if name.lower() in f.lower() and f.endswith(
                        (".safetensors", ".ckpt", ".pt")):
                    return True
    except Exception:
        pass
    return False


def _check_tool(name: str) -> bool:
    if name in ("rife",):
        try:
            from utils.rife_interpolate import RIFEInterpolator
            return bool(RIFEInterpolator().is_available())
        except Exception:
            return False
    if name == "ffmpeg":
        try:
            from utils.rife_interpolate import _ffmpeg_exe
            return bool(_ffmpeg_exe())
        except Exception:
            return False
    return True     # 未知识别不了的工具不阻断


def _device_has_cuda() -> bool:
    try:
        from workflow.memory_policy import DeviceCapabilities
        return DeviceCapabilities.detect().has_cuda
    except Exception:
        return False


# ---------------------------------------------------------------- 汇总
def collect(wf, *, check: bool = True) -> List[Requirement]:
    """遍历工作流所有节点的 `resources.requires`，合并成去重清单。"""
    from workflow import registry

    merged: Dict[Tuple[str, str], Requirement] = {}

    def _add(kind: str, key: str, node_id: str, optional: bool,
             extension_id: str = "", hint: str = ""):
        rec = merged.get((kind, key))
        if rec is None:
            rec = Requirement(kind=kind, key=key,
                              label=ASSET_LABELS.get(key, key),
                              optional=optional, extension_id=extension_id,
                              hint=hint)
            merged[(kind, key)] = rec
        if node_id and node_id not in rec.nodes:
            rec.nodes.append(node_id)
        # 只要有一个节点把它当必需，就算必需
        rec.optional = rec.optional and optional

    for n in wf.nodes:
        if not registry.exists(n.type):
            continue
        reqs = registry.get(n.type).resources.requires or ()
        for raw in reqs:
            kind, _, key = str(raw).partition(":")
            kind = kind.strip()
            key = key.strip()
            if kind == "cuda":
                _add("cuda", "cuda", n.id, optional=False,
                     hint="需要 NVIDIA GPU")
            elif kind == "model":
                _add("model", key or "any", n.id, optional=False,
                     hint="底模由你自行下载后放入 models/putin 或对应目录")
            elif kind == "vae":
                _add("vae", key or "vae", n.id, optional=True)
            elif kind == "asset":
                _add("asset", key, n.id, optional=True,
                     extension_id=ASSET_TO_EXTENSION.get(key, ""),
                     hint="可在扩展市场下载")
            elif kind == "tool":
                _add("tool", key, n.id, optional=True)
            else:
                _add(kind or "other", key or kind, n.id, optional=True)

    out = list(merged.values())

    if check:
        for r in out:
            if r.kind == "cuda":
                r.present = _device_has_cuda()
            elif r.kind == "model":
                r.present = _any_checkpoint(r.key)
            elif r.kind == "vae":
                r.present = _check_vae(r.key)
            elif r.kind == "asset":
                r.present = _check_asset(r.key)
                if r.extension_id:
                    try:
                        from utils.extension_manager import EXTENSIONS
                        info = EXTENSIONS.get(r.extension_id) or {}
                        r.size_mb = int(info.get("size_mb", 0) or 0)
                    except Exception:
                        pass
            elif r.kind == "tool":
                r.present = _check_tool(r.key)
    return out


def _any_checkpoint(arch: str) -> bool:
    """本地有没有可用底模。arch 为空表示"任意架构都行"。"""
    try:
        from utils.model_scanner import MODEL_TYPES, scan_models
        for sub in MODEL_TYPES:
            if arch and arch not in ("any", sub) and arch != sub:
                continue
            if scan_models(sub):
                return True
    except Exception as e:
        logger.debug(f"底模检测失败: {e}")
    return False


def missing(wf, *, include_optional: bool = False) -> List[Requirement]:
    reqs = collect(wf)
    return [r for r in reqs
            if not r.present and (include_optional or not r.optional)]


def summary(wf) -> dict:
    """给 UI 用的资源报告。"""
    reqs = collect(wf)
    miss_required = [r for r in reqs if not r.present and not r.optional]
    miss_optional = [r for r in reqs if not r.present and r.optional]
    return {
        "requirements": [r.to_dict() for r in reqs],
        "missing_required": [r.to_dict() for r in miss_required],
        "missing_optional": [r.to_dict() for r in miss_optional],
        "total_missing_mb": sum(r.size_mb for r in miss_required + miss_optional),
        "can_run": not miss_required,
        "text": format_report(reqs),
    }


def format_report(reqs: List[Requirement]) -> str:
    """中文清单，可直接放进弹窗。"""
    if not reqs:
        return "该工作流不需要额外资源。"
    lines = ["📦 资源清单", ""]
    for r in reqs:
        mark = "✅" if r.present else ("⚠️" if r.optional else "❌")
        label = r.label or r.key
        line = f"  {mark} {label}"
        if not r.present and r.size_mb:
            line += f"（约 {r.size_mb / 1024:.1f}GB）"
        if not r.present and r.optional:
            line += "  [可选]"
        lines.append(line)
        if not r.present and r.hint:
            lines.append(f"      {r.hint}")
    miss = [r for r in reqs if not r.present and not r.optional]
    if miss:
        lines += ["", f"缺少 {len(miss)} 项必需资源，工作流无法运行。",
                  "提示：程序只读取本地文件，不会自动联网下载。"]
    else:
        lines += ["", "必需资源齐备，可以运行。"]
    return "\n".join(lines)
