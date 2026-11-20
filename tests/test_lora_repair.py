# -*- coding: utf-8 -*-
"""LoRA 键名修复的回归测试。

覆盖两类真机撞出来的文件：
1. 训练器早期存出来的 `.default` 残留（`ValueError: Could not automatically
   infer state dict type`）；
2. civitai 下载的 kohya 格式（`IndexError: list index out of range`）。

不依赖 GPU、不加载模型 —— 只验证"整形逻辑"和"接线"。
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import _bootstrap  # noqa: E402,F401

import torch  # noqa: E402

from utils import lora_repair  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name):
    def deco(fn):
        global _PASS, _FAIL
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _FAIL += 1
            print(f"FAIL  {name}: {exc}")
        else:
            _PASS += 1
            print(f"PASS  {name}")
    return deco


def _t(shape=(4, 4)):
    return torch.zeros(shape, dtype=torch.float16)


def _setu_like():
    """模拟训练器存出来的文件：unet 正常，text_encoder 带 .default。"""
    sd = {
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight": _t((4, 8)),
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora.up.weight": _t((8, 4)),
        "text_encoder.encoder.layers.0.self_attn.k_proj.lora_A.default.weight": _t((4, 8)),
        "text_encoder.encoder.layers.0.self_attn.k_proj.lora_B.default.weight": _t((8, 4)),
    }
    return sd


def _kohya_like():
    sd = {
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight": _t((4, 8)),
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight": _t((8, 4)),
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.alpha": torch.tensor(4.0),
        "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight": _t((4, 8)),
        "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight": _t((8, 4)),
        "lora_te_text_model_encoder_layers_0_self_attn_k_proj.alpha": torch.tensor(4.0),
    }
    return sd


@check("strip_adapter_suffix_removes_default")
def _():
    sd, changed = lora_repair.strip_adapter_suffix(_setu_like())
    assert changed == 2, changed
    assert not any(".default" in k for k in sd), [k for k in sd if ".default" in k]
    assert "text_encoder.encoder.layers.0.self_attn.k_proj.lora_A.weight" in sd
    # 不碰其它键
    assert "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight" in sd


@check("strip_adapter_suffix_keeps_unrelated_keys")
def _():
    sd = {"a.default.b": 1, "c.defaults.d": 2, "e.f": 3}
    out, changed = lora_repair.strip_adapter_suffix(sd)
    assert out["a.b"] == 1
    assert "c.defaults.d" in out, out.keys()   # `.defaults` 不是适配器名
    assert changed == 1, changed


@check("looks_kohya_detects_prefixes")
def _():
    assert lora_repair.looks_kohya(_kohya_like()) is True
    assert lora_repair.looks_kohya(_setu_like()) is False
    assert lora_repair.looks_kohya({}) is False


@check("repair_strips_default_and_drops_text_encoder")
def _():
    fixed, notes, warnings = lora_repair.repair_lora_state_dict(_setu_like())
    assert fixed, notes
    assert not any(k.startswith("text_encoder.") for k in fixed), fixed.keys()
    assert any(k.startswith("unet.") for k in fixed)
    assert any("文本编码器" in w for w in warnings), warnings
    assert lora_repair.describe_repair(notes)


@check("repair_keeps_text_encoder_when_unet_absent")
def _():
    sd = {
        "text_encoder.encoder.layers.0.self_attn.k_proj.lora_A.default.weight": _t((4, 8)),
    }
    fixed, notes, warnings = lora_repair.repair_lora_state_dict(sd)
    assert fixed, notes
    assert any(k.startswith("text_encoder.") for k in fixed), fixed.keys()
    assert not warnings, warnings


@check("repair_converts_kohya_keys")
def _():
    fixed, notes, warnings = lora_repair.repair_lora_state_dict(_kohya_like())
    assert fixed, (notes, warnings)
    assert any(k.startswith("unet.") for k in fixed), list(fixed)[:3]
    assert not any(k.endswith(".alpha") for k in fixed), [k for k in fixed if k.endswith(".alpha")]
    assert any("kohya" in n for n in notes), notes


@check("repair_empty_returns_empty_with_note")
def _():
    fixed, notes, warnings = lora_repair.repair_lora_state_dict({})
    assert fixed == {}
    assert warnings


@check("repair_is_idempotent_on_clean_dict")
def _():
    sd = {
        "unet.x.lora.down.weight": _t((4, 8)),
        "unet.x.lora.up.weight": _t((8, 4)),
    }
    fixed, notes, warnings = lora_repair.repair_lora_state_dict(sd)
    assert set(fixed) == set(sd)
    assert not warnings, warnings


@check("model_manager_wires_repair_into_lora_loading")
def _():
    src = io.open(os.path.join(ROOT, "core", "model_manager.py"), encoding="utf-8").read()
    assert "_load_one_lora" in src
    assert "_adapter_registered" in src
    # 半成功（UNet 挂上、TE 炸）必须被认下来
    assert "U-Net 部分已挂载" in src
    assert "文本编码器部分没挂上" in src
    # 不许谎报"TE 已补挂"——实测补挂后 TE 并不参与编码
    assert "_mount_text_encoder_part" not in src
    # 修复路径必须真的调用 lora_repair
    assert "lora_repair.repair_lora_state_dict" in src
    # 旧的"整块跳过"不该再是唯一出路
    assert "logger.error(f\"❌ [跳过] 插件 {lora_name} 不兼容或损坏: {e}\")" not in src


@check("trainer_saves_keys_without_adapter_name")
def _():
    src = io.open(os.path.join(ROOT, "core", "lora_train.py"), encoding="utf-8").read()
    idx = src.find("def _lora_state_dict")
    assert idx > 0
    body = src[idx:idx + 2000]
    assert "strip_adapter_suffix" in body, "训练器必须去掉 .default，否则自己存的读不回来"


@check("apply_multiple_loras_accepts_tuples")
def _():
    src = io.open(os.path.join(ROOT, "core", "model_manager.py"), encoding="utf-8").read()
    assert "_normalize_lora_list(lora_list)" in src


print(f"\n{_PASS} passed, {_FAIL} failed")
sys.exit(0 if _FAIL == 0 else 1)
