# core/loaders/sdxl.py
# ============================================================
#  SDXL 单文件加载器（含 Pony / Illustrious / NoobAI 等衍生）
#  与 AnimaLoader 同范式：build / apply_vram / derive_pipes 分层
# ============================================================
import logging
from .base import BaseLoader, LoadResult

logger = logging.getLogger(__name__)


class SDXLLoader(BaseLoader):
    arch_ids = ("sdxl",)

    # 单文件缺 scheduler/配置时，回退官方 XL base 配置
    CONFIG_REPO = "stabilityai/stable-diffusion-xl-base-1.0"

    def build(self, model_path, ctx):
        from diffusers import (StableDiffusionXLPipeline,
                               StableDiffusionXLImg2ImgPipeline,
                               StableDiffusionXLInpaintPipeline)

        logger.info(f"⏳ 正在加载 {ctx.info.display_name} 模型: {ctx.model_name}")
        logger.info(f"📋 使用标准 config: {self.CONFIG_REPO}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=ctx.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            config=self.CONFIG_REPO,
        )

        # ⚠️ 必须开启 output_hidden_states：
        # diffusers 的 SDXL 管线在 encode_prompt 里直接访问
        # `prompt_embeds.hidden_states[-2]`（用倒数第二层作为文本条件）。
        # 若该标志为 False，transformers 返回的 hidden_states 是 None →
        # `TypeError: 'NoneType' object is not subscriptable`。
        try:
            for name in ("text_encoder", "text_encoder_2"):
                te = getattr(pipe, name, None)
                if te is not None and hasattr(te, "config"):
                    te.config.output_hidden_states = True
            logger.info("✅ 已开启 text_encoder.output_hidden_states（SDXL 管线需要）")
        except Exception as e:
            logger.warning(f"⚠️ 开启 output_hidden_states 失败: {e}")

        r = LoadResult(txt2img=pipe)
        r.extras["classes"] = (StableDiffusionXLImg2ImgPipeline,
                               StableDiffusionXLInpaintPipeline)
        return r

    def derive_pipes(self, result, ctx):
        i2i_cls, inp_cls = result.extras["classes"]
        self.derive_from_components(result, ctx, i2i_cls, inp_cls)
