# core/loaders/flux.py
import logging
import torch
from .base import BaseLoader, LoadResult

logger = logging.getLogger(__name__)


class FluxLoader(BaseLoader):
    arch_ids = ("flux",)
    BASE_REPO = "black-forest-labs/FLUX.1-dev"   # schnell 更快更省，dev 质量好

    def build(self, model_path, ctx):
        from diffusers import FluxPipeline, FluxTransformer2DModel

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

        logger.info(f"🧩 装配 Flux 组件: {self.BASE_REPO}")
        pipe = FluxPipeline.from_pretrained(
            self.BASE_REPO,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        # 8GB 卡必须 offload，否则 VAE+TE 一起就爆
        pipe.enable_model_cpu_offload()
        return LoadResult(txt2img=pipe)

    def apply_vram(self, result, ctx):
        # Flux 走 offload 架构，不用 SD 式 VAE tiling/slicing 逻辑
        pass