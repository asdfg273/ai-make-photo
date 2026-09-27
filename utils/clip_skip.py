# utils/clip_skip.py
"""
CLIP Skip —— 让文本编码器使用倒数第 N 层输出，而不是最后一层

为什么需要：
  大量动漫向底模（Pony / Illustrious / NoobAI / 很多 SD1.5 二次元模型）
  都是在 CLIP 倒数第二层输出上训练的。用默认的最后一层会明显掉质量：
  线条变脏、色彩发灰、脸更容易崩。社区的 "CLIP skip 2" 说的就是这个。

实现方式（已针对 transformers 5.16 实测）：
  该版本的 `CLIPTextModel.forward` 内部是
      encoder_outputs = self.encoder(inputs_embeds=..., is_causal=True)
      last_hidden_state = self.final_layer_norm(encoder_outputs.last_hidden_state)
  而 `CLIPEncoder.forward` **只返回最后一层**，没有 output_hidden_states
  开关。所以只能自己遍历 `encoder.layers` 取中间层。

  这里只对「CLIP 系 + truncate/clip 层数 > 1」生效，且：
    · 不改动任何管线结构，只替换 text_encoder.forward；
    · 幂等：同一实例重复启用不会叠加包装；
    · 提供 restore_clip_skip() 可完整还原；
    · SDXL 的 text_encoder_2（OpenCLIP）结构不同，单独判断、失败即跳过，
      绝不影响主流程。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MARK = "_ai_clip_skip"          # 标记属性，用于幂等与还原


def _has_encoder_layers(model) -> bool:
    enc = getattr(model, "encoder", None)
    return enc is not None and hasattr(enc, "layers") and len(enc.layers) > 0


def apply_clip_skip(text_encoder, skip: int = 1) -> bool:
    """让 text_encoder 输出倒数第 (skip+1) 层。

    skip=1 → 倒数第二层（社区常说 CLIP skip 2）
    skip=0 → 最后一层（等价于默认行为，直接还原）

    返回 True 表示已生效。
    """
    if text_encoder is None:
        return False
    if skip is None:
        return False
    skip = int(skip)
    if skip <= 0:
        restore_clip_skip(text_encoder)
        return False

    if not _has_encoder_layers(text_encoder):
        logger.debug("[clip_skip] 该文本编码器结构不支持（非 CLIP 系），跳过")
        return False

    n_layers = len(text_encoder.encoder.layers)
    if skip >= n_layers:
        logger.warning(f"[clip_skip] 层数 {n_layers} 不足以跳过 {skip} 层，忽略")
        return False

    # 幂等：已包装过就直接返回
    if getattr(text_encoder, _MARK, None) is not None:
        if getattr(text_encoder, _MARK) == skip:
            return True
        # 层数变了 → 先还原再重新包装
        restore_clip_skip(text_encoder)

    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    original = text_encoder.forward
    target_layer = n_layers - 1 - skip      # layers 是 0-based

    def _forward(input_ids=None, attention_mask=None, position_ids=None,
                 **kwargs):
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        input_shape = input_ids.size()
        flat_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = text_encoder.embeddings(
            input_ids=flat_ids, position_ids=position_ids)

        # 注意力掩码：不同版本签名不一样，取不到就退化成原实现
        try:
            from transformers.masking_utils import create_causal_mask
            attn_mask = create_causal_mask(
                config=text_encoder.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
            )
        except Exception:
            attn_mask = attention_mask

        kwargs.pop("is_causal", None)

        # 逐层跑到目标层，同时收集各层输出（供 output_hidden_states 使用）
        collect = bool(kwargs.pop("output_hidden_states", False))
        enc_layers = text_encoder.encoder.layers
        for i, layer in enumerate(enc_layers):
            hidden_states = layer(hidden_states, attn_mask, **kwargs)
            if i == target_layer:
                break

        hidden_states = text_encoder.final_layer_norm(hidden_states)

        # ── pooled 输出 ──
        # transformers 5.16 的官方实现在 `eos_token_id is None` 时会
        # `(input_ids == None).int()` → AttributeError（已实测：Anima 的
        # text_encoder 就是 eos_token_id=None）。
        # 这里健壮化：拿不到 eos 就退回"序列最后一位"（tokenizer 常规行为）。
        eos_id = getattr(text_encoder, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(text_encoder.config, "eos_token_id", None)
        try:
            if isinstance(eos_id, int):
                pos = (flat_ids.to(dtype=torch.int,
                                   device=hidden_states.device) == eos_id) \
                    .int().argmax(dim=-1)
            else:
                pos = torch.full((hidden_states.shape[0],),
                                 hidden_states.shape[1] - 1,
                                 dtype=torch.long,
                                 device=hidden_states.device)
            pooled = hidden_states[
                torch.arange(hidden_states.shape[0],
                             device=hidden_states.device), pos]
        except Exception:
            pooled = hidden_states[:, -1]

        # ── hidden_states 栈 ──
        # ⚠️ 这是关键：diffusers 的 SDXL 管线用 `hidden_states[-2]` 取
        # **最后一层**。所以栈里必须满足「倒数第 2 个 = 最后一层」。
        # 我们的循环已经停在目标层，因此返回两元组：
        #     (-2) = 目标层输出（= 我们想用的 CLIP skip 层）
        #     (-1) = 与 -2 相同（占位，保证索引语义自洽）
        # 这样 `[-2]` 取到的就是用户选的层；不开启时无副作用
        # （SD1.5 用 last_hidden_state，SDXL 用 [-2]，都拿到目标层）。
        hs = None
        if collect:
            try:
                hs = (hidden_states, hidden_states)
            except Exception:
                hs = None

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=hs,
        )

    text_encoder.forward = _forward
    setattr(text_encoder, _MARK, skip)
    setattr(text_encoder, "_ai_clip_skip_orig", original)
    logger.info(f"🎚️ CLIP Skip 已启用: 使用倒数第 {skip + 1} 层 "
                f"(共 {n_layers} 层)")
    return True


def restore_clip_skip(text_encoder) -> bool:
    """还原为默认（最后一层）。"""
    if text_encoder is None:
        return False
    orig = getattr(text_encoder, "_ai_clip_skip_orig", None)
    if orig is None:
        return False
    try:
        text_encoder.forward = orig
    except Exception as e:
        logger.warning(f"[clip_skip] 还原失败: {e}")
        return False
    try:
        delattr(text_encoder, "_ai_clip_skip_orig")
        delattr(text_encoder, _MARK)
    except Exception:
        pass
    logger.info("🎚️ CLIP Skip 已还原（使用最后一层）")
    return True


def supports_clip_skip(manager) -> tuple[bool, str]:
    """判断当前架构是否真的支持 CLIP Skip。返回 (是否支持, 原因)。

    ⚠️ 实测结论（很重要，避免给用户"调了没用"的假象）：
      · **SDXL：不支持，也不需要。** diffusers 的 SDXL 管线在 encode_prompt
        里硬编码使用 `prompt_embeds.hidden_states[-2]`（倒数第二层），
        这本来就是 SDXL 训练时的选择。所以 SDXL 没有"skip 几层"可调，
        实测 skip=0 与 skip=1 出图像素差为 0。
        另外 `text_encoder_2` 是 CLIPTextModelWithProjection，结构与
        CLIPTextModel 不同，也包不上。
      · **SD1.5 / SD2.1：支持。** 它们用 last_hidden_state，可以切层。
      · **flux / sd3 / anima：不支持**（非 CLIP 文本编码器 / DiT 架构）。
    """
    if manager is None:
        return False, "引擎未就绪"
    arch = getattr(manager, "current_arch_id", None) or "未知"
    if getattr(manager, "is_sdxl", False) or arch == "sdxl":
        return False, ("SDXL 本身就固定使用倒数第二层作为文本条件"
                       "（diffusers 管线内部行为），无需也无法调整")
    if getattr(manager, "is_flux", False) or getattr(manager, "is_sd3", False) \
            or arch in ("flux", "sd3", "anima"):
        return False, f"{arch} 不是 CLIP 文本编码器架构，不支持 CLIP Skip"
    # 结构校验：必须真的能拿到 encoder.layers
    for attr in ("txt2img_pipe", "img2img_pipe", "inpaint_pipe"):
        pipe = getattr(manager, attr, None)
        if pipe is None:
            continue
        te = getattr(pipe, "text_encoder", None)
        if te is not None and _has_encoder_layers(te):
            return True, f"{arch} 支持（CLIP 系）"
    return False, "当前管线没有可切层的 CLIP 文本编码器"


def apply_clip_skip_to_pipes(manager, skip: int) -> int:
    """把 CLIP skip 应用到管理器持有的所有管线。

    只处理**结构兼容**的 CLIP 文本编码器（text_encoder）。
    SDXL 的 text_encoder_2（CLIPTextModelWithProjection）会被安全跳过；
    SDXL 整体也会由调用方通过 supports_clip_skip() 提前拦下并说明原因。
    返回成功应用的编码器数量。
    """
    targets = []
    for attr in ("txt2img_pipe", "img2img_pipe", "inpaint_pipe",
                 "controlnet_pipe", "reference_pipe"):
        pipe = getattr(manager, attr, None)
        if pipe is None:
            continue
        # 只取主 text_encoder：text_encoder_2 结构不同且 SDXL 不需要
        te = getattr(pipe, "text_encoder", None)
        if te is not None and all(te is not t for t in targets):
            targets.append(te)

    n = 0
    for te in targets:
        try:
            if apply_clip_skip(te, int(skip)):
                n += 1
        except Exception as e:
            logger.warning(f"[clip_skip] 应用失败（已跳过该编码器）: {e}")
    if n == 0 and int(skip or 0) > 0:
        logger.info("[clip_skip] 没有结构兼容的 CLIP 文本编码器，"
                    "该设置对当前架构无效（详见 supports_clip_skip）")
    return n
