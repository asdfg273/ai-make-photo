# core/loaders/anima.py
import logging
import torch
from .base import BaseLoader, LoadResult

logger = logging.getLogger(__name__)


class AnimaLoader(BaseLoader):
    arch_ids = ("anima",)
    BASE_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
    PREFIX = "model.diffusion_model."   # 探测确认: 685/685 key 都在此前缀下

    def build(self, model_path, ctx):
        from diffusers import ModularPipeline
        from safetensors.torch import load_file

        logger.info(f"⏳ 装配 Anima 组件: {self.BASE_REPO}")
        pipe = ModularPipeline.from_pretrained(self.BASE_REPO)
        pipe.load_components(dtype=torch.bfloat16)
        pipe.to("cuda")

        logger.info(f"🔄 替换 transformer 权重: {ctx.model_name}")
        sd = load_file(model_path)
        sd = {k[len(self.PREFIX):]: v for k, v in sd.items()
              if k.startswith(self.PREFIX)}
        logger.info(f"🔧 剥离前缀后共 {len(sd)} keys")

        # 加载前先比对一次 key 命名，避免用 strict=False 硬吞错误
        expected = set(pipe.transformer.state_dict().keys()) - {
            k for k in pipe.transformer.state_dict() if "dtype" in k or k.endswith("_metadata")}
        got = set(sd.keys())
        missing = expected - got
        unexpected = got - expected
        logger.info(f"📊 对照: missing={len(missing)} unexpected={len(unexpected)}")
        if len(unexpected) > len(sd) * 0.3:
            logger.warning(f"⚠️ unexpected 过多，key 命名可能不匹配: {list(unexpected)[:6]}")
        if len(missing) > len(expected) * 0.3:
            logger.warning(f"⚠️ missing 过多，可能漏掉关键层: {list(missing)[:6]}")

        _, _ = pipe.transformer.load_state_dict(sd, strict=False)
        return LoadResult(txt2img=pipe)

    def apply_vram(self, result, ctx):
        pipe = result.txt2img
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            logger.info("✅ Anima: VAE tiling+slicing 已启用")
        except Exception as e:
            logger.warning(f"⚠️ Anima VAE 优化失败: {e}")