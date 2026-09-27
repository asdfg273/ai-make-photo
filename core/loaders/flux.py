# core/loaders/flux.py
import logging
import torch
from .base import BaseLoader, LoadResult

logger = logging.getLogger(__name__)


class FluxLoader(BaseLoader):
    """Flux 单文件（transformer-only）加载器。

    ⚠️ 注意：本加载器当前**未在 core/loaders/__init__.py 中注册**，
    因此实际不可达 —— 活的 Flux 路径是 SingleFileLoader
    （它走 FluxPipeline.from_single_file，面向**完整** Flux 检查点）。
    两者面向的检查点格式不同：
        · 本加载器   : 只有 transformer 的权重，组件从 HF 仓库补
        · SingleFile : 完整 checkpoint（含 TE/VAE）
    保留本类是为了它自带的 GGUF 量化分支，等确定主推哪种格式再接。
    """
    arch_ids = ("flux",)
    # 引擎组件（T5/CLIP/VAE 等）从扩展市场下载到 models/flux_engine/<变体>。
    # 这里**不再**直接写 HF 仓库 id，避免 from_pretrained 静默下载约 24GB。
    BASE_REPO = "black-forest-labs/FLUX.1-schnell"   # 仅作兜底提示文本

    def build(self, model_path, ctx):
        from diffusers import FluxPipeline, FluxTransformer2DModel

        from utils.flux_engine import find_engine, status_text

        engine = find_engine()
        if engine is None:
            # 缺组件时必须**明确报错并指路**，而不是偷偷联网下 24GB
            raise RuntimeError(
                "无法装配 Flux：引擎组件缺失。\n\n" + status_text())

        variant, engine_path = engine
        logger.info(f"🧩 使用本地 Flux 引擎组件: {engine_path}")

        logger.info(f"🔄 加载 Flux transformer 单文件: {ctx.model_name}")
        transformer = FluxTransformer2DModel.from_single_file(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        # 若选 GGUF 量化版，改为:
        #   from diffusers import GGUFQuantizationConfig
        #   transformer = FluxTransformer2DModel.from_single_file(
        #       model_path,
        #       quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        #       torch_dtype=torch.bfloat16,
        #   )

        pipe = FluxPipeline.from_pretrained(
            engine_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        # ⚠️ 这里**不再**写死 enable_model_cpu_offload()。
        # 显存档位统一由 VRAMManager.apply_optimal_strategy 依据
        # 「主干有多大 + 还要留多少给激活值」决定（见 apply_vram）。
        return LoadResult(txt2img=pipe)

    def apply_vram(self, result, ctx):
        """与 SD 系共用同一套显存决策（不再写死 offload）。

        Flux 的主干是 pipe.transformer 而不是 pipe.unet，
        apply_optimal_strategy 已能识别两者。
        """
        from utils.vram_manager import VRAMManager
        _w = getattr(getattr(ctx, "detection", None), "size_gb", 0.0)
        VRAMManager.apply_optimal_strategy(
            result.txt2img, is_sdxl=False, weights_gb=_w, arch_id="flux")
        VRAMManager.print_status()
