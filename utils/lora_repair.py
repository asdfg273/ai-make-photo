# -*- coding: utf-8 -*-
"""LoRA 权重键名修复：让"别人的 LoRA"也能在本项目里挂上。

背景（都是真机撞出来的，不是猜的）：

1. **训练器自己存出来的文件读不回去**
   `core/lora_train._lora_state_dict()` 早期调
   `convert_state_dict_to_diffusers(get_peft_model_state_dict(module))`，
   当 `train_text_encoder=True` 时文本编码器那半边会残留 peft 的适配器名，
   键名长这样：

       text_encoder.encoder.layers.0.self_attn.k_proj.lora_A.default.weight

   diffusers 的 `load_lora_weights` 一看到 `.default.` 就
   `ValueError: Could not automatically infer state dict type`
   （`diffusers/utils/state_dict_utils.py:275`）。去掉 `.default` 就能读。

2. **civitai 下载的 kohya LoRA 文本编码器部分挂不上**
   这类文件是 `lora_unet_*` / `lora_te_*` 键名。diffusers 0.40 会把它转成
   `unet.*` + `text_encoder.*`，其中 UNet 部分完全正常；但文本编码器部分被
   转成了 `text_encoder.text_model.encoder.layers.0.self_attn.to_k_lora.down.weight`
   这种 **UNet 风格的 `to_k` 命名**（CLIP 里其实叫 `k_proj`），于是
   `diffusers/loaders/lora_base.py:386-390` 用 `named_modules()` 里以
   `.k_proj` 结尾的名字去找 `lora_B` 时一条都匹配不上，`rank` 是空字典，
   `get_peft_kwargs()` 里 `list(rank_dict.values())[0]`
   → `IndexError: list index out of range`（`peft_utils.py:158`）。

   这是上游的转换问题，本项目改不了。实测的取舍：
   - 只挂 UNet 部分 → **成功**（384 个键，效果大头都在这里）
   - 连文本编码器一起挂 → 必炸

   所以这里的选择是：**能挂多少挂多少**，把文本编码器那半跳掉并如实告诉用户，
   而不是整块放弃。

注意：本模块只做"键名整形 + 取舍"，不加载任何东西，方便单测。
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

# peft 的适配器名后缀：`.default.weight` / `.default.bias`
_ADAPTER_SUFFIX = re.compile(r"\.default(\.|$)")
# kohya / A1111 的键名前缀
_KOHYA_PREFIXES = ("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")


def strip_adapter_suffix(state_dict: Dict[str, object]) -> Tuple[Dict[str, object], int]:
    """去掉键名里的 `.default`（peft 适配器名），返回 (新字典, 改了多个键)。"""
    out: Dict[str, object] = {}
    changed = 0
    for key, value in state_dict.items():
        new_key = _ADAPTER_SUFFIX.sub(r"\1", key)
        if new_key != key:
            changed += 1
        out[new_key] = value
    return out, changed


def looks_kohya(state_dict: Dict[str, object]) -> bool:
    """判断是不是 kohya / A1111 的 `lora_unet_*` 格式。"""
    keys = list(state_dict.keys())
    if not keys:
        return False
    return all(k.startswith(_KOHYA_PREFIXES) for k in keys)


def has_unet_keys(state_dict: Dict[str, object]) -> bool:
    return any(k.startswith("unet.") for k in state_dict)


def has_text_encoder_keys(state_dict: Dict[str, object]) -> bool:
    return any(k.startswith("text_encoder.") for k in state_dict)


def _convert_kohya(state_dict: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    """用 diffusers 自己的 kohya 转换器转成 `unet.*` / `text_encoder.*` 键名。"""
    from diffusers.loaders.lora_conversion_utils import (
        _convert_non_diffusers_lora_to_diffusers,
    )

    return _convert_non_diffusers_lora_to_diffusers(dict(state_dict))


def repair_lora_state_dict(
    state_dict: Dict[str, object],
    *,
    drop_text_encoder: bool = True,
) -> Tuple[Dict[str, object], List[str], List[str]]:
    """把一份"读不了"的 LoRA 权重整形成可加载的样子。

    Args:
        state_dict: 原始权重（键名不做任何假设）。
        drop_text_encoder: 转完 kohya 格式后是否丢掉文本编码器部分。
            实测"一起挂"必炸（见模块说明），所以默认丢。

    Returns:
        (新的权重字典, 做了哪些改动的说明, 警告列表)
        第一个元素为空字典表示"没救"。
    """
    notes: List[str] = []
    warnings: List[str] = []
    sd: Dict[str, object] = dict(state_dict)
    if not sd:
        return {}, notes, ["权重是空的"]

    # 1) 去 peft 适配器名后缀
    sd, changed = strip_adapter_suffix(sd)
    if changed:
        notes.append(f"去掉 {changed} 个键名里的 .default")

    # 2) kohya 格式 → diffusers 键名
    if looks_kohya(sd):
        try:
            sd, _alphas = _convert_kohya(sd)
            notes.append("按 kohya 格式重写了键名")
            # `_convert_non_diffusers_lora_to_diffusers` 会把 alpha 塞进
            # 单独的网络 alpha 字典，这里不需要（rank 能从张量形状推）。
            sd = {k: v for k, v in sd.items() if not k.endswith(".alpha")}
        except Exception as exc:  # noqa: BLE001
            logger.debug("kohya 转换失败: %s", exc, exc_info=True)
            return {}, notes, [f"kohya 格式转换失败：{exc}"]

    # 3) 文本编码器半边：能挂就挂，挂不了就丢（上游转换缺陷）
    if drop_text_encoder and has_text_encoder_keys(sd) and has_unet_keys(sd):
        sd = {k: v for k, v in sd.items() if not k.startswith("text_encoder.")}
        notes.append("跳过文本编码器部分")
        warnings.append(
            "这个 LoRA 的文本编码器部分无法在当前 diffusers 版本加载（上游转换缺陷），"
            "已跳过；U-Net 部分正常生效，所以画风/人物仍会变化")

    if not sd:
        return {}, notes, warnings + ["清掉不可用的键之后什么都不剩了"]
    return sd, notes, warnings


def describe_repair(notes: Iterable[str]) -> str:
    items = [n for n in notes if n]
    return "、".join(items) if items else ""
