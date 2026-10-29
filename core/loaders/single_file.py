import logging
from .base import BaseLoader, LoadResult

logger = logging.getLogger(__name__)


class SingleFileLoader(BaseLoader):
    """通用单文件加载器：sd3 / flux。

    sd15/sd21 与 sdxl 已拆分为独立加载器（sd15.py / sdxl.py），
    本类只保留尚未拆分架构的兜底逻辑。
    """
    arch_ids = ("sd3", "flux")

    CONFIG_REPO = {
        "flux": "black-forest-labs/FLUX.1-schnell",
        "sd3":  "stabilityai/stable-diffusion-3-medium-diffusers",
    }

    def _pipe_classes(self, arch_id):
        import diffusers as D
        return {
            "flux": (D.FluxPipeline, D.FluxImg2ImgPipeline, D.FluxInpaintPipeline),
            # ⚠️ diffusers 没有 StableDiffusion3InpaintPipeline。
            # 以前这里用 StableDiffusion3Pipeline 顶替 inpaint 槽位：管线能建起来，
            # 但 SD3 的 __call__ 没有 image/mask_image/strength 参数，一调用就
            # TypeError。改成 None（schema 明确表达"该架构无 inpaint"），
            # 配合 SD3Adapter(inpaint=False)，上层会提前拦住并给出可操作提示。
            "sd3":  (D.StableDiffusion3Pipeline, D.StableDiffusion3Img2ImgPipeline,
                     None),
        }[arch_id]

    def build(self, model_path, ctx):
        from core.dit_optimization import quantization_config

        txt_cls, i2i_cls, inp_cls = self._pipe_classes(ctx.arch_id)

        kwargs = {
            "torch_dtype": ctx.dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
        if ctx.arch_id in self.CONFIG_REPO:
            kwargs["config"] = self.CONFIG_REPO[ctx.arch_id]
            logger.info(f"📋 使用标准 config: {kwargs['config']}")

        quant = quantization_config(ctx.optimization.get("dit_quantization", "off"))
        if quant is not None:
            kwargs["quantization_config"] = quant

        logger.info(f"⏳ 正在加载 {ctx.info.display_name} 模型...")
        pipe = txt_cls.from_single_file(model_path, **kwargs)

        r = LoadResult(txt2img=pipe)
        r.extras["classes"] = (i2i_cls, inp_cls)
        return r

    def apply_vram(self, result, ctx):
        """DiT 使用独立量化/分块策略，避免重复安装通用 offload hook。"""
        from core.dit_optimization import apply_group_offload, quantized_weight_ratio
        from utils.vram_manager import VRAMManager

        options = ctx.optimization
        grouped = apply_group_offload(
            result.txt2img,
            bool(options.get("dit_group_offload", False)),
            int(options.get("dit_offload_blocks", 1) or 1),
            getattr(ctx.manager, "device", "cuda"),
        )
        if grouped:
            VRAMManager.tune_for_resolution(result.txt2img, 1024, 1024)
            return
        weight = getattr(getattr(ctx, "detection", None), "size_gb", 0.0)
        weight *= quantized_weight_ratio(options.get("dit_quantization", "off"))
        VRAMManager.apply_optimal_strategy(
            result.txt2img, is_sdxl=True, weights_gb=weight, arch_id=ctx.arch_id)
        VRAMManager.print_status()

    def derive_pipes(self, result, ctx):
        i2i_cls, inp_cls = result.extras["classes"]
        self.derive_from_components(result, ctx, i2i_cls, inp_cls)
