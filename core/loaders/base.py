from dataclasses import dataclass, field
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoadContext:
    """加载环境，由 ModelManager 注入"""
    dtype: Any
    model_name: str = ""
    arch_id: str = ""
    info: Any = None          # ArchInfo
    detection: Any = None     # arch_mod.detect() 的结果
    manager: Any = None       # ModelManager 自身，供复用既有 helper


@dataclass
class LoadResult:
    txt2img: Any = None
    img2img: Any = None
    inpaint: Any = None
    extras: dict = field(default_factory=dict)


class BaseLoader:
    arch_ids: tuple = ()

    def load(self, model_path: str, ctx: LoadContext) -> LoadResult:
        result = self.build(model_path, ctx)
        self.apply_vram(result, ctx)
        self.derive_pipes(result, ctx)
        self.post_load(result, ctx, model_path)
        return result

    # ---- 子类必须实现 ----
    def build(self, model_path, ctx) -> LoadResult:
        raise NotImplementedError

    # ---- 可选钩子 ----
    def apply_vram(self, result, ctx):
        from utils.vram_manager import VRAMManager
        heavy = ctx.arch_id in ("sdxl", "pony", "sd3")
        VRAMManager.apply_optimal_strategy(result.txt2img, is_sdxl=heavy)
        VRAMManager.print_status()

    def derive_pipes(self, result, ctx):
        pass

    def derive_from_components(self, result, ctx, i2i_cls, inp_cls):
        """从 txt2img 的组件派生 img2img / inpaint 管线（SD 系通用）。

        派生管线共享 unet/vae/text_encoder 等组件，只做轻量优化，
        不重复 cpu_offload。子类在 derive_pipes 里调用本方法即可。
        """
        comps = result.txt2img.components
        for attr, cls in (("img2img", i2i_cls), ("inpaint", inp_cls)):
            try:
                p = cls(**comps)
                ctx.manager._apply_light_optimizations(p, name=attr)
                setattr(result, attr, p)
            except Exception as e:
                logger.warning(f"⚠️ {attr} pipe 创建失败: {e}")

    def post_load(self, result, ctx, model_path):
        """caps 驱动的通用收尾，各架构共用"""
        caps, mgr = ctx.info.caps, ctx.manager
        pipes = [p for p in (result.txt2img, result.img2img, result.inpaint) if p]

        if caps.supports_vpred and mgr._detect_vpred(model_path, ctx.model_name):
            mgr.is_vpred = True
            try:
                cfg = dict(result.txt2img.scheduler.config)
                cfg["prediction_type"] = "v_prediction"
                cfg["rescale_betas_zero_snr"] = True
                sch = result.txt2img.scheduler.__class__.from_config(cfg)
                for p in pipes:
                    p.scheduler = sch
                logger.info("✅ 已启用 v_prediction + zero-SNR")
            except Exception as e:
                logger.error(f"❌ v_prediction 设置失败: {e}")
        else:
            mgr.is_vpred = False

        if caps.needs_fp32_vae:
            mgr._force_fp32_vae(result.txt2img)