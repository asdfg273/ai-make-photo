import logging
from .base import BaseLoader, LoadResult

logger = logging.getLogger(__name__)


class SingleFileLoader(BaseLoader):
    arch_ids = ("sd15", "sdxl", "sd3", "flux")

    CONFIG_REPO = {
        "flux": "black-forest-labs/FLUX.1-schnell",
        "sd3":  "stabilityai/stable-diffusion-3-medium-diffusers",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    }

    def _pipe_classes(self, arch_id):
        import diffusers as D
        return {
            "flux": (D.FluxPipeline, D.FluxImg2ImgPipeline, D.FluxInpaintPipeline),
            "sd3":  (D.StableDiffusion3Pipeline, D.StableDiffusion3Img2ImgPipeline,
                     D.StableDiffusion3Pipeline),
            "sdxl": (D.StableDiffusionXLPipeline, D.StableDiffusionXLImg2ImgPipeline,
                     D.StableDiffusionXLInpaintPipeline),
            "sd15": (D.StableDiffusionPipeline, D.StableDiffusionImg2ImgPipeline,
                     D.StableDiffusionInpaintPipeline),
        }[arch_id]

    def build(self, model_path, ctx):
        txt_cls, i2i_cls, inp_cls = self._pipe_classes(ctx.arch_id)

        kwargs = {
            "torch_dtype": ctx.dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
        if ctx.arch_id == "sd15":
            kwargs["safety_checker"] = None
        if ctx.arch_id in self.CONFIG_REPO:
            kwargs["config"] = self.CONFIG_REPO[ctx.arch_id]
            logger.info(f"📋 使用标准 config: {kwargs['config']}")

        logger.info(f"⏳ 正在加载 {ctx.info.display_name} 模型...")
        pipe = txt_cls.from_single_file(model_path, **kwargs)

        r = LoadResult(txt2img=pipe)
        r.extras["classes"] = (i2i_cls, inp_cls)
        return r

    def derive_pipes(self, result, ctx):
        i2i_cls, inp_cls = result.extras["classes"]
        comps = result.txt2img.components
        for attr, cls, name in ((("img2img"), i2i_cls, "img2img"),
                                (("inpaint"), inp_cls, "inpaint")):
            try:
                p = cls(**comps)
                ctx.manager._apply_light_optimizations(p, name=name)
                setattr(result, attr, p)
            except Exception as e:
                logger.warning(f"⚠️ {name} pipe 创建失败: {e}")