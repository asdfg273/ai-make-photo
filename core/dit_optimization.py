"""Flux / SD3 / MOSS-Image 等 DiT 的量化与分块卸载后端适配。

三种量化后端，按"本机装了什么"降级选择：

* ``bnb8`` —— bitsandbytes 8bit（本项目环境里已装）。权重压到约 55%，
  **能整块放进 8GB 显存**，因此不需要每步把 9.5GB 权重过一遍 PCIe。
  这是 8GB 卡跑 MOSS-Image 最实际的一档。
* ``fp8`` —— torchao Float8 权重（需要 torchao）。
* ``nvfp4`` —— NVIDIA ModelOpt FP4（需要 modelopt + 兼容内核）。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

QUANTIZATION_MODES = ("off", "bnb8", "fp8", "nvfp4")

MODE_LABELS = {
    "off": "关闭（原样 bf16）",
    "bnb8": "8bit（bitsandbytes，8GB 显卡推荐）",
    "fp8": "FP8（torchao）",
    "nvfp4": "NVFP4（NVIDIA ModelOpt）",
}


def _bnb8_config():
    """bitsandbytes 8bit：只量化 transformer。

    8bit 是"权重压一半多、几乎不掉精度"的折中，且在 8GB 卡上刚好
    能把 DiT（9.7GB bf16）塞进显存，省掉每步的权重搬运。
    """
    try:
        from diffusers import PipelineQuantizationConfig
    except ImportError as exc:
        raise RuntimeError("当前 diffusers 不支持管线量化，请升级后再使用。") from exc
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "8bit 量化需要 bitsandbytes；请在项目环境执行 "
            "`venv\\Scripts\\pip install bitsandbytes`。"
            f"原始错误: {exc}") from exc
    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True},
        components_to_quantize=["transformer"],
    )


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
    if mode == "bnb8":
        return _bnb8_config()
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
    """仅供显存策略估算；不代表实测模型文件大小。

    实测参考：MOSS-Image 的 DiT bf16 是 9.47GB，bitsandbytes 8bit 后
    约 5.2GB（0.55 倍），刚好落进 8GB 卡的可用范围。
    """
    return {"bnb8": 0.55, "fp8": 0.5, "nvfp4": 0.25}.get(
        (mode or "off").lower(), 1.0)
