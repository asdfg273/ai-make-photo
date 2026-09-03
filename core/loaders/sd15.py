# core/loaders/sd15.py
# ============================================================
#  SD 1.5 / SD 2.1 单文件加载器
#  与 AnimaLoader 同范式：build / apply_vram / derive_pipes 分层
# ============================================================
import logging
from .base import BaseLoader, LoadResult

logger = logging.getLogger(__name__)


class SD15Loader(BaseLoader):
    """SD1.5 与 SD2.1 共用同一套 Pipeline 类（from_single_file 自动推断配置）。"""
    arch_ids = ("sd15", "sd21")

    def build(self, model_path, ctx):
        from diffusers import (StableDiffusionPipeline,
                               StableDiffusionImg2ImgPipeline,
                               StableDiffusionInpaintPipeline)

        logger.info(f"⏳ 正在加载 {ctx.info.display_name} 模型: {ctx.model_name}")
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=ctx.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            safety_checker=None,          # 本地工作站，跳过安全审查器省显存
        )
        r = LoadResult(txt2img=pipe)
        r.extras["classes"] = (StableDiffusionImg2ImgPipeline,
                               StableDiffusionInpaintPipeline)
        return r

    def derive_pipes(self, result, ctx):
        i2i_cls, inp_cls = result.extras["classes"]
        self.derive_from_components(result, ctx, i2i_cls, inp_cls)
