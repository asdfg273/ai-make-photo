# nodes/text_nodes.py
"""文本条件节点：TextEncode / NegativeTextEncode。

设计要点
--------
· 编码**只做一次**：把结果放进 Conditioning，采样节点直接复用 Embeds。
  这既是缓存友好的，也让"改采样步数不用重新编码"成立。
· 走 ModelManager.encode_prompt（内部有 Compel 与缓存），不自己造一套；
  非 Compel 架构（Anima/Flux）返回原始文本，由适配器决定怎么用。
"""
from __future__ import annotations

import logging

from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import Conditioning

logger = logging.getLogger(__name__)


def _encode(ctx, text: str, negative: str) -> Conditioning:
    """用当前模型编码提示词。失败时退化成"原始文本"条件。"""
    text = text or ""
    negative = negative or ""
    if ctx.ai is None:
        return Conditioning(text=text, negative_text=negative)

    try:
        emb = ctx.ai.encode_prompt(text, negative)
    except Exception as e:
        logger.warning(f"⚠️ Compel 编码失败，退回原始提示词: {e}")
        emb = {"prompt": text, "negative_prompt": negative}

    if not emb:
        return Conditioning(text=text, negative_text=negative)

    if emb.get("prompt_embeds") is not None:
        return Conditioning(
            embeds=emb.get("prompt_embeds"),
            pooled=emb.get("pooled_prompt_embeds"),
            negative_embeds=emb.get("negative_prompt_embeds"),
            negative_pooled=emb.get("negative_pooled_prompt_embeds"),
            text=text, negative_text=negative)

    # 非 Compel 架构：只有原始文本
    return Conditioning(text=emb.get("prompt", text),
                        negative_text=emb.get("negative_prompt", negative))


@node(
    "TextEncode",
    category="文本",
    description="编码正面提示词为文本条件（含翻译/改写后的最终提示词）。",
    inputs={"model": PortSpec(DataType.MODEL, "模型", optional=True)},
    outputs={"conditioning": PortSpec(DataType.TEXT_COND, "文本条件"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "text": ParamSpec("text", default="", label="正面提示词"),
        "text_final": ParamSpec("text", default="", label="最终提示词",
                                help="留空则用上面的原文（可填翻译/改写后的结果）"),
    },
    resources=ResourceRequirement(requires=("model",),
                                  estimated_vram_mb=600,
                                  supports_cpu_offload=True),
)
def run_text_encode(ctx, node, inputs):
    raw = node.params.get("text") or ""
    final = (node.params.get("text_final") or "").strip() or raw
    cond = _encode(ctx, final, "")
    cond.text = final
    ctx.metadata["prompt"] = raw
    ctx.metadata["prompt_final"] = final
    return {"conditioning": cond,
            "metadata": {"prompt": raw, "prompt_final": final}}


@node(
    "NegativeTextEncode",
    category="文本",
    description="编码负面提示词；若已给定条件则把负向附加进去。",
    inputs={"model": PortSpec(DataType.MODEL, "模型", optional=True),
            "conditioning": PortSpec(DataType.TEXT_COND, "文本条件",
                                     optional=True)},
    outputs={"conditioning": PortSpec(DataType.TEXT_COND, "文本条件"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "text": ParamSpec("text", default="", label="负面提示词"),
        "text_final": ParamSpec("text", default="", label="最终负面提示词"),
    },
    resources=ResourceRequirement(requires=("model",),
                                  estimated_vram_mb=600,
                                  supports_cpu_offload=True),
)
def run_negative_text_encode(ctx, node, inputs):
    raw = node.params.get("text") or ""
    final = (node.params.get("text_final") or "").strip() or raw
    base = inputs.get("conditioning")

    if base is not None:
        cond = _encode(ctx, base.text or "", final)
        cond.text = base.text
    else:
        cond = _encode(ctx, "", final)

    ctx.metadata["negative"] = raw
    ctx.metadata["negative_final"] = final
    return {"conditioning": cond,
            "metadata": {"negative": raw, "negative_final": final}}
