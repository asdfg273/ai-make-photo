# utils/flux_engine.py
"""Flux「引擎组件」的定位与状态查询。

为什么要拆成两部分
------------------
Flux 出图需要两样东西：

  1. **transformer（主权重，bf16 约 24GB）** —— 由用户自行下载的单文件，
     符合本项目「模型由用户自行下载」的约定；
  2. **引擎组件（T5-XXL + CLIP-L + VAE + tokenizer + scheduler，约 10GB）**
     —— 这才是"引擎"，走扩展市场下载。

⚠️ 当前**不接通** Flux 引擎（`core/arch/registry.py` 里 `supported=False`）。
   本模块只负责回答"组件是否就位"，供加载器与提示弹窗使用。

   之所以要它，是因为旧 `core/loaders/flux.py` 直接
   `FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")` ——
   一旦放开 supported，就会**静默下载约 24GB**，既慢又违反模型策略。

许可证（重要，选变体时请看清）
------------------------------
  · FLUX.1-schnell : Apache-2.0
                     ✅ 可商用，无附加限制
  · FLUX.1 [dev]   : FLUX.1 [dev] Non-Commercial License v1.1.1
                     ❌ **不可商用**：禁止"营利性活动"、
                        禁止"直接面向终端用户或对终端用户产生影响"的使用。
                        商用需另行向 Black Forest Labs 申请授权。

  ⚠️ 两者的**生成图片**都可以商用（BFL 声明不对 Outputs 主张权利，
     只限制"使用模型"本身）。受限的是拿 dev 权重做产品/提供服务。

  本项目是「开源 + 打赏」形态：schnell 无争议；dev 权重请自行确认
  你的使用方式是否落在非商业范围内，风险自负。
"""
from __future__ import annotations

import logging
import os

from utils import paths

logger = logging.getLogger(__name__)

ENGINE_ROOTNAME = "flux_engine"

VARIANTS: dict[str, dict] = {
    "schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "label": "FLUX.1-schnell",
        "license": "Apache-2.0",
        "commercial": True,
        "size_mb": 10500,
        "note": "Apache-2.0，可商用，推荐",
    },
    "dev": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "label": "FLUX.1 [dev]",
        "license": "FLUX.1 [dev] Non-Commercial License v1.1.1",
        "commercial": False,
        "size_mb": 10500,
        "note": "非商用许可，商用需向 Black Forest Labs 申请授权",
    },
}

# 引擎（组件）是否就位，看这些文件。transformer 不在其中 ——
# 主权重由用户自己提供。
REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "tokenizer/tokenizer_config.json",
    "vae/config.json",
)


def engine_root() -> str:
    return os.path.join(paths.MODEL_DIR, ENGINE_ROOTNAME)


def engine_dir(variant: str) -> str:
    return os.path.join(engine_root(), variant)


def missing_files(variant: str) -> list[str]:
    """返回该变体缺失的必需文件（相对路径）。"""
    d = engine_dir(variant)
    return [f for f in REQUIRED_FILES
            if not os.path.exists(os.path.join(d, f))]


def is_installed(variant: str) -> bool:
    if variant not in VARIANTS:
        return False
    return not missing_files(variant)


def installed_variant() -> str | None:
    """返回第一个已装好的变体；都没有则 None。"""
    for v in VARIANTS:
        if is_installed(v):
            return v
    return None


def find_engine() -> tuple[str, str] | None:
    """(variant, 引擎目录)。未安装返回 None。"""
    v = installed_variant()
    if v is None:
        return None
    return v, engine_dir(v)


def status_text() -> str:
    """给弹窗/报错用的一段中文说明。"""
    v = installed_variant()
    if v:
        info = VARIANTS[v]
        warn = ("" if info["commercial"]
                else "\n⚠️ 注意：该变体为非商用许可，商用请改用 schnell "
                     "或向 Black Forest Labs 申请授权。")
        return (f"✅ Flux 引擎组件已就位：{info['label']}"
                f"（{info['license']}）{warn}")
    lines = [
        "❌ Flux 引擎组件尚未下载，无法使用 Flux 模型。",
        "",
        "请在「扩展市场」中安装 Flux 引擎（约 10GB）：",
    ]
    for vid, info in VARIANTS.items():
        mark = "✅ 可商用" if info["commercial"] else "❌ 不可商用"
        lines.append(f"   · {info['label']}（{info['license']}）{mark}")
    lines += [
        "",
        "说明：Flux 的 transformer 主权重（约 24GB）需要你自行下载，",
        "      扩展市场只提供 T5/CLIP/VAE 等引擎组件（约 10GB）。",
        "      主权重放进 models/puton/ 或 models/flux/ 即可。",
    ]
    return "\n".join(lines)
