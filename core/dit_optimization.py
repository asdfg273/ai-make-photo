"""Flux / SD3 DiT 量化与分块卸载的可选后端适配。"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

QUANTIZATION_MODES = ("off", "fp8", "nvfp4")


def _fp8_config():
    try:
        from diffusers import TorchAoConfig
        from torchao.quantization import Float8WeightOnlyConfig
        return TorchAoConfig(Float8WeightOnlyConfig())
    except Exception as exc:
        raise RuntimeError(
            "FP8 权重量化需要可用的 torchao 与支持它的 diffusers；"
            "请在项目环境安装 torchao，或将 DiT 量化改回“关闭”。"
            f"原始错误: {exc}") from exc


def _nvfp4_config():
    try:
        from diffusers import NVIDIAModelOptConfig
        return NVIDIAModelOptConfig(quant_type="NVFP4")
    except Exception as exc:
        raise RuntimeError(
            "NVFP4 需要支持 NVIDIA ModelOpt 的 diffusers、modelopt "
            "以及兼容的 NVIDIA GPU/内核；当前环境无法创建该配置。"
            f"原始错误: {exc}") from exc


def quantization_config(mode: str):
    """创建只作用于 DiT transformer 的管线量化配置。"""
    mode = (mode or "off").lower()
    if mode == "off":
        return None
    if mode not in QUANTIZATION_MODES:
        raise ValueError(f"未知 DiT 量化模式: {mode}")
    try:
        from diffusers import PipelineQuantizationConfig
    except ImportError as exc:
        raise RuntimeError("当前 diffusers 不支持管线量化，请升级后再使用。") from exc
    config = _fp8_config() if mode == "fp8" else _nvfp4_config()
    return PipelineQuantizationConfig(quant_mapping={"transformer": config})


def apply_group_offload(pipe, enabled: bool, blocks: int, device: str) -> bool:
    """对 DiT 主干按 block 分组 offload；返回管线是否已安装该策略。"""
    if not enabled:
        return False
    if getattr(pipe, "_dit_group_offloaded", False):
        return True
    if not str(device).lower().startswith("cuda"):
        logger.info("非 CUDA 设备不安装 DiT group offload")
        return False
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise RuntimeError("当前管线没有 transformer，无法应用 DiT 分块 offload。")
    try:
        from diffusers.hooks import apply_group_offloading
    except ImportError as exc:
        raise RuntimeError("当前 diffusers 不支持 group offload，请升级后再使用。") from exc
    blocks = max(1, min(32, int(blocks or 1)))
    try:
        apply_group_offloading(
            transformer,
            onload_device=device,
            offload_device="cpu",
            offload_type="block_level",
            num_blocks_per_group=blocks,
            use_stream=True,
        )
    except Exception as exc:
        raise RuntimeError(f"DiT 分块 offload 安装失败: {exc}") from exc
    pipe._dit_group_offloaded = True
    logger.info("DiT group offload 已启用，每组 %s 个 block", blocks)
    return True


def optimization_signature(options: dict | None) -> tuple:
    options = options or {}
    return (
        str(options.get("dit_quantization", "off") or "off").lower(),
        bool(options.get("dit_group_offload", False)),
        max(1, min(32, int(options.get("dit_offload_blocks", 1) or 1))),
    )


def quantized_weight_ratio(mode: str) -> float:
    """仅供显存策略估算；不代表实测模型文件大小。"""
    return {"fp8": 0.5, "nvfp4": 0.25}.get((mode or "off").lower(), 1.0)
