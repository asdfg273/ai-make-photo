# model_manager.py
# ============================================================
#  模型管理器 — 多 Pipeline / LoRA / ControlNet / Sampler
# ============================================================

import os
import gc
import torch
import cv2
import numpy as np
from contextlib import contextmanager
from typing import Optional
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
)

from controlnet_aux import OpenposeDetector
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image

from utils.system_utils import SingletonMeta
from utils import paths
import logging
from core import arch as arch_mod
from core.identity import resolver as identity_resolver
logger = logging.getLogger(__name__)


SDXL_KEYWORDS = (
    "xl", "sdxl", "pony", "turbo", "lightning", "illustrious", "noobai",
)


ARCH_LABELS = {
    "flux": "FLUX",
    "sd3":  "SD3",
    "sdxl": "SDXL",
    "sd15": "SD 1.5",
}


# ============================================================
class ModelManager(metaclass=SingletonMeta):
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.device = self._normalize_device("auto")
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32

        self.current_model_name = None
        self.current_model_path = None
        self.current_lora_name  = None
        self.is_sdxl            = False
        self.is_vpred = False
        self.txt2img_pipe       = None
        self.img2img_pipe       = None
        self.inpaint_pipe       = None
        self.controlnet_pipe    = None
        self.reference_pipe     = None   # Reference-Only（community pipeline，惰性创建）
        self.detected_arch      = None
        self.current_arch_id    = None
        self.pose_detector      = None
        self.depth_estimator    = None
        self.loaded_controlnets = {}
        self.current_cn_type    = None
        # ★ IP-Adapter 状态
        self.ip_adapter_loaded   = False
        self.ip_adapter_variant  = None
        # 多参考图：本次实际装上的档位与**每适配器**强度（顺序与参考图一致）
        self._ipa_adapters       = []
        self._ipa_scales         = []
        # 已安装适配器的"指纹"（档位/文件名/来源/子目录）。
        # 完全相同时跳过重复 load_ip_adapter（实测每次约 1.1s）。
        self._ipa_signature      = None
        self.is_sd3  = False
        self.is_flux = False
        self.current_ipa_scale = 0.7

        self._compel_cache = {}
        self._model_cache  = {}

        self._freeu_cfg     = None   # 已选择的参数（关掉也保留，便于恢复）
        self._freeu_enabled = False

        # CLIP Skip（0=关闭）。由 UI 设置后写入，load_model 时自动应用。
        self.clip_skip = 0
        # zero-SNR：None=跟随模型自带配置（推荐）；True/False=强制。
        # 与 v-pred 解耦 —— v-pred ≠ zero-SNR。
        self.zero_snr = None
        self._sampler_name = None

    # ------------------------------------------------------------
    #  资源枚举
    # ------------------------------------------------------------
    def get_available_models(self, model_type="sd15"):
        from utils.model_scanner import scan_models
        return [m["name"] for m in scan_models(model_type)]

    def get_available_loras(self, model_type=None):
        from utils import paths
        sub_dir = self._normalize_lora_dir(model_type or self.current_lora_subdir())
        base_dir = os.path.join(paths.LORA_DIR, sub_dir)
        if not os.path.isdir(base_dir):
            logger.warning(f"⚠️ LoRA 目录不存在: {base_dir}")
            return ["无"]
        return ["无"] + sorted(
            f for f in os.listdir(base_dir)
            if f.endswith((".safetensors", ".ckpt", ".pt"))
        )


    # ------------------------------------------------------------
    #  显存清理
    # ------------------------------------------------------------
    def clear_memory(self):
        for attr in ('txt2img_pipe', 'img2img_pipe',
                     'inpaint_pipe', 'controlnet_pipe',
                     'reference_pipe'):
            setattr(self, attr, None)
        self.loaded_controlnets.clear()

        # _compel_cache 里存的是 prompt_embeds 等 **GPU 张量**：
        # 只 clear() 字典不会归还显存，必须逐个搬到 CPU 再丢引用。
        try:
            for _v in list(self._compel_cache.values()):
                if isinstance(_v, torch.Tensor):
                    _v.data = _v.data.cpu()
        except Exception:
            logger.debug("compel 缓存迁移 CPU 跳过", exc_info=True)
        self._compel_cache.clear()

        # 状态必须跟着一起清（含架构，否则卸载后 LoRA 目录/架构判断错位）
        self.current_model_name  = None
        self.current_lora_name   = None
        self.current_cn_type     = None
        self.ip_adapter_loaded   = False
        self.ip_adapter_variant  = None
        self._ipa_loaded         = False
        self._ipa_variant        = None
        self._ipa_adapters       = []
        self._ipa_scales         = []
        self._ipa_signature      = None
        self._freeu_enabled = False
        self._freeu_cfg = None
        self.detected_arch = None
        self.current_arch_id = None
        self.arch_profile = None
        self.is_flux = self.is_sd3 = self.is_sdxl = False

        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        # 基线一定要丢：留着旧模型的 beta/alpha/prediction_type，
        # 下一个模型切采样器时会被污染（v-pred 模型尤其致命）。
        self._scheduler_base_cfg = None
        # 只 gc + empty_cache 不够：MOSS 这类 mmap 权重会把整份文件页留在
        # 进程工作集里（实测换模型后任务管理器仍显示 21GB），必须显式
        # EmptyWorkingSet 才还给系统。
        try:
            from utils import mem as _mem

            _delta = _mem.release_memory()
            _line = _mem.describe(_delta)
            if _line:
                logger.info("🧹 已释放旧模型：%s", _line)
        except Exception:  # noqa: BLE001 - 释放失败不能挡住加载
            logger.debug("释放内存统计跳过", exc_info=True)

    # ------------------------------------------------------------
    #  显存 / 性能优化 (A-2)
    # ------------------------------------------------------------
    def apply_optimizations(self, pipe, name: str = ""):
        """
        完整优化：Model CPU Offload + VAE tiling/slicing +
        Attention slicing + xformers。用于主 txt2img 管线。
        """
        applied = []

        # 1) CPU 卸载 (仅 CUDA 有效，共享组件的其他 pipe 不要重复开)
        if self.device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
                applied.append("cpu_offload")
            except Exception as e:
                logger.warning(f"⚠️ enable_model_cpu_offload 失败: {e}")

        # 2) 剩余轻量优化
        for item in self._apply_light_optimizations(pipe, _return=True):
            applied.append(item)

        if applied:
            logger.info(f"⚙️ [{name or 'pipe'}] 已启用优化: {', '.join(applied)}")
        return pipe

    def _apply_light_optimizations(self, pipe, name: str = "",
                                   _return: bool = False):
        """
        共享 components 的管线用这个，只做 VAE / Attention / xformers，
        不再重复 cpu_offload，避免冲突。
        """
        applied = []

        # VAE 分块解码 / 切片（大图与低显存关键）。
        # ⚠️ diffusers 0.40 的 Pipeline 上**没有** enable_vae_tiling/slicing，
        # 只有 pipe.vae.enable_tiling()/enable_slicing()，
        # 所以统一走 VRAMManager._set_vae_option（内部做能力探测）。
        # 旧写法会全部抛 AttributeError 被 except 吞掉，日志却打"已启用"。
        from utils.vram_manager import VRAMManager
        if VRAMManager._set_vae_option(pipe, "tiling", True):
            applied.append("vae_tiling")
        if VRAMManager._set_vae_option(pipe, "slicing", True):
            applied.append("vae_slicing")

        # xformers：本环境未安装，且 diffusers 0.40 里
        # enable_xformers_* 已是废弃 shim（先告警再尝试 import）。
        # 只在真正装了 xformers 时才调用，避免每次加载都刷废弃警告。
        try:
            import importlib.util
            if importlib.util.find_spec("xformers") is not None:
                pipe.enable_xformers_memory_efficient_attention()
                applied.append("xformers")
        except Exception:
            pass

        if _return:
            return applied
        if applied:
            logger.info(f"⚙️ [{name or 'pipe'}] 已启用优化: {', '.join(applied)}")
        return pipe

    # ------------------------------------------------------------
    #  加载底模
    # ------------------------------------------------------------
    def _detect_vpred(self, model_path, model_name):
        """检测是否 v-prediction 模型：优先读 safetensors 元数据，回退文件名。"""
        try:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt") as f:
                meta = f.metadata() or {}
            for k, v in meta.items():
                kl, vl = k.lower(), str(v).lower()
                if "predict" in kl and vl in ("v", "v_prediction", "vpred"):
                    logger.info(f"🔍 元数据标记 v-prediction: {k}={v}")
                    return True
                if kl == "modelspec.architecture" and "-v" in vl:
                    logger.info(f"🔍 元数据标记 v-prediction: {k}={v}")
                    return True
        except Exception as e:
            logger.warning(f"⚠️ 读取 v-pred 元数据失败: {e}")

        n = (model_name or "").lower()
        hit = any(t in n for t in ("vpred", "v-pred", "v_pred"))
        if hit:
            logger.info("🔍 文件名命中 v-pred 关键字")
        return hit

    def _is_flow_matching_pipe(self) -> bool:
        """当前底模是否为 flow-matching 架构（自带原生 scheduler，不可换 SD 系采样器）。"""
        pipe = getattr(self, "txt2img_pipe", None)
        sched = getattr(pipe, "scheduler", None)
        if sched is None:
            return False
        cls_name = type(sched).__name__
        if "FlowMatch" in cls_name or "RectifiedFlow" in cls_name:
            return True
        # 子类改名很常见：MOSS-Image 的调度器叫 MOSSImageScheduler，名字里
        # 没有 "FlowMatch"，但它是 FlowMatchEulerDiscreteScheduler 的子类。
        # 只按类名判断会把它当成普通 SD 调度器，从而被换成 DPM++ 之类，
        # 直接把采样搞坏（MOSS 的 step() 语义与 SD 系不同）。
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
            return isinstance(sched, FlowMatchEulerDiscreteScheduler)
        except Exception:
            return False

    # ------------------------------------------------------------------
    #  统一能力描述（模型加载后生成）
    # ------------------------------------------------------------------
    @property
    def capabilities(self):
        """当前模型的能力描述（`workflow.adapters.ModelCapabilities`）。

        界面与节点都该问它"能不能做某事"，而不是判断模型名：
            if ai.capabilities.supports_controlnet: ...
            ai.capabilities.require("hires_fix")   # 不支持就抛明确的异常

        没加载模型时返回 None —— 调用方要自己处理"还没加载"这种情况，
        不要在这里编一个默认能力出来（那样会掩盖真正的问题）。
        """
        arch = getattr(self, "current_arch_id", None)
        if not arch:
            return None
        try:
            from workflow import adapters
            return adapters.capabilities_for(arch)
        except Exception as e:
            logger.debug(f"能力描述生成失败（arch={arch}）: {e}")
            return None

    def load_model(self, model_name):
        import warnings
        warnings.filterwarnings("ignore", message=".*kept in float32.*")

        # 同模型且推理配置未变化时才跳过；切换量化/offload 必须重建管线。
        from core.dit_optimization import optimization_signature
        opt_signature = optimization_signature(getattr(self, "inference_options", None))
        if (self.current_model_name == model_name
                and self.txt2img_pipe is not None
                and getattr(self, "_loaded_optimization_signature", None) == opt_signature):
            logger.info("⚡ 模型与推理配置未改变，跳过加载。")
            return

        # === 智能查找: 支持 models/ 根目录 + 所有架构子目录 ===
        MODELS_ROOT = paths.MODEL_DIR
        _SUBDIRS = [
            "sd15", "sdxl", "sd3", "pony", "flux",
            "anima", "mossimage",
            # 以后新架构目录也加到这
        ]

        def _is_model_entry(p: str) -> bool:
            """文件，或「diffusers 目录」（目录里有 model_index.json）。

            MOSS-Image 这类官方权重是**目录**，不是单文件；只按
            os.path.isfile 判存在会让它们永远选不中。
            """
            if os.path.isfile(p):
                return True
            if os.path.isdir(p):
                return any(
                    os.path.isfile(os.path.join(p, m))
                    for m in ("model_index.json", "modular_model_index.json"))
            return False

        candidates = [os.path.join(MODELS_ROOT, model_name)] + [
            os.path.join(MODELS_ROOT, sub, model_name) for sub in _SUBDIRS
        ]

        # 若 model_name 不带扩展名，尝试自动补全
        _exts = [".safetensors", ".ckpt", ".pt"]
        _got = None
        for c in candidates:
            if _is_model_entry(c):
                _got = c
                break
        if _got is None:
            for c in candidates:
                for ext in _exts:
                    p = c + ext
                    if os.path.isfile(p):
                        _got = p
                        break
                if _got:
                    break
        model_path = _got

        if model_path is None:
            # 最后的 os.walk 兜底
            for root, dirs, files in os.walk(MODELS_ROOT):
                for f in files:
                    if f == model_name or f == model_name + ".safetensors":
                        model_path = os.path.join(root, f)
                        break
                if model_path:
                    break
                # 目录形式的模型（diffusers 仓库，如 MOSS-Image）
                for d in dirs:
                    if d != model_name:
                        continue
                    cand = os.path.join(root, d)
                    if _is_model_entry(cand):
                        model_path = cand
                        break
                if model_path:
                    break

        # 找不到文件必须给出可诊断的报错。
        # 否则 detect(None) 会在 try 之外抛 TypeError(_path_isdir)，
        # 用户看到一个跟自己的操作毫无关联的异常。
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"未找到模型文件「{model_name}」。\n"
                f"已搜索: {MODELS_ROOT} 及子目录 "
                f"{', '.join(_SUBDIRS)}\n"
                f"请确认文件已放入对应目录（支持 .safetensors/.ckpt/.pt），\n"
                f"或放入一个含 model_index.json 的 diffusers 目录"
                f"（MOSS-Image 就是这种）。"
            )

        try:
            # ========== 架构检测（唯一来源）==========
            det = arch_mod.detect(model_path)
            self.detected_arch = det                 # 供 UI / LoRA 目录推断读取
            self.current_arch_id = det.arch_id
            from core.arch.base import build_arch_profile
            self.arch_profile = build_arch_profile(det.arch_id)
            logger.info(f"🔬 架构检测: {det.info.display_name} "
                        f"[{det.arch_id}] | 依据: {det.evidence} "
                        f"| {det.key_count} keys / {det.size_gb:.2f}GB")

            if not det.info.caps.is_base_model:
                raise ValueError(f"无法加载「{model_name}」："
                                 f"[{det.info.display_name}] "
                                 f"{det.info.unsupported_reason}")

            if not det.info.supported:
                extra = ""
                if det.info.extra_components:
                    extra = (f"\n需额外准备（自行下载）: "
                             f"{', '.join(det.info.extra_components)}"
                             f" ({det.info.extra_download_size or '大小未知'})")
                # 架构检测失败时，evidence 里往往带着**真正的原因**。
                # 例如系统提交内存不足会让 `_read_header` 抛
                # "页面文件太小 (os error 1455)"，若只显示
                # "可能是损坏文件"，用户会去删/重下好文件，白折腾。
                diag = ""
                if det.arch_id == "unknown" and det.evidence:
                    diag = f"\n🔎 诊断依据: {det.evidence}"
                    if det.top_prefixes:
                        diag += f"\n🔑 键名前缀 top5: {', '.join(det.top_prefixes)}"
                    if "header 读取失败" in det.evidence:
                        diag += ("\n💡 这不是文件损坏，而是读取文件头时出错"
                                 "（常见于内存/页面文件耗尽或权限不足）。"
                                 "关闭占内存的程序后重试；若仍失败请检查文件权限。")
                raise ValueError(f"无法加载「{model_name}」："
                                 f"[{det.info.display_name}] "
                                 f"{det.info.unsupported_reason}{extra}{diag}")

            arch           = det.arch_id
            self.is_flux   = (arch == "flux")
            self.is_sd3    = (arch == "sd3")
            self.is_sdxl   = (arch == "sdxl")


            from core.loaders import get_loader
            from core.loaders.base import LoadContext

            ctx = LoadContext(
                dtype=self.dtype, model_name=model_name,
                arch_id=det.arch_id, info=det.info,
                detection=det, manager=self,
                optimization=dict(getattr(self, "inference_options", {}) or {}),
            )

            # 换模型前先释放旧管线：否则新旧两套权重同时驻留，
            # 8GB 卡上换 SDXL 必然 OOM / offload 抖动。
            if self.txt2img_pipe is not None or self.img2img_pipe is not None:
                logger.info("🧹 加载新模型前释放旧管线（旧架构=%s）...",
                            getattr(self, "current_arch_id", "?"))
                self.clear_memory()
                # ⚠️ 关键：clear_memory() 会一并清掉架构状态
                # （detected_arch / current_arch_id / arch_profile / is_sdxl…），
                # 而上面的架构检测在它**之前**执行 —— 若不重新写回，
                # 加载完 SDXL 后 is_sdxl 会变回 False，
                # 导致：SDXL 判断错误、LoRA 目录选错、IP-Adapter 选到 SD1.5 权重、
                # ControlNet 用错架构的管线。这里必须重新赋值。
                self.detected_arch = det
                self.current_arch_id = det.arch_id
                from core.arch.base import build_arch_profile
                self.arch_profile = build_arch_profile(det.arch_id)
                _arch = det.arch_id
                self.is_flux = (_arch == "flux")
                self.is_sd3 = (_arch == "sd3")
                self.is_sdxl = (_arch == "sdxl")

            result = get_loader(det.arch_id).load(model_path, ctx)

            # ===== torch.compile 加速 UNet（暂时禁用：Windows 多 GPU 环境不兼容） =====
            #对于多 GPU 用户，torch.compile 会导致 UNet 共享失败，报错：
            #如果你在 Windows 上使用多 GPU，建议暂时禁用 torch.compile，或者升级到 PyTorch 2.1+。当然没有多 GPU 的用户可以尝试开启，首次推理会慢 30-60 秒，之后会快很多。
            #当然这个可能只是我的电脑的问题，其他人可能没事。总之先禁用，等 PyTorch 2.1+ 再启用。或者有大佬改成多 GPU 也能共享的方式也行。
            # try:
            #     import torch
            #     if torch.__version__ >= "2.0":
            #         if hasattr(result.txt2img, "unet") and result.txt2img.unet is not None:
            #             logger.info("⚡ 正在编译 UNet（首次推理会慢 30-60 秒）...")
            #             result.txt2img.unet = torch.compile(
            #                 result.txt2img.unet,
            #                 backend="cudagraphs",
            #                 fullgraph=False,
            #             )
            #             if (result.img2img is not None
            #                 and hasattr(result.img2img, "unet")
            #                 and result.img2img.unet is not result.txt2img.unet):
            #                 result.img2img.unet = torch.compile(
            #                     result.img2img.unet,
            #                     backend="cudagraphs",
            #                     fullgraph=False,
            #                 )
            #             logger.info("✅ UNet 编译完成")
            # except Exception as e:
            #     logger.warning(f"⚠️ torch.compile 跳过: {e}")
            # ===== 编译结束 =====

            self.txt2img_pipe = result.txt2img
            self.img2img_pipe = result.img2img
            self.inpaint_pipe = result.inpaint
            # 新底模自带自己的 scheduler 配置：基线要重取，否则沿用旧模型的
            # beta/alpha/prediction_type（v-pred 模型尤其致命）。
            self._scheduler_base_cfg = None
            self.controlnet_pipe = None
            self.current_cn_type  = None
            self.current_model_name = model_name
            self._loaded_optimization_signature = opt_signature
            # 出图元数据要记录底模的**真实磁盘路径 + 哈希**，以前只存了
            # 文件名，事后无法确认到底用的哪个文件（同名不同版很常见）。
            self.current_model_path = model_path
            self.current_lora_name  = None
            self._compel_cache.clear()
            self.compel_txt2img = self.compel_img2img = self.compel_controlnet = None

            if not self.is_vpred:
                # 二次检测：用文件名（不区分大小写）
                fn_lower = (model_name or "").lower()
                if any(kw in fn_lower for kw in ("vpred", "v-pred", "v_pred")):
                    logger.warning(f"⚠️ 首次 vpred 检测未命中，文件名二次检测命中: {model_name}")
                    self.is_vpred = True

            # ⚠️ 这里**不再**就地改 txt2img_pipe 的 scheduler。
            # 旧行为有两个问题（都是评审指出的）：
            #   1. 只替换了 txt2img 的 scheduler，img2img / inpaint 仍是旧实例，
            #      同一份 config 出现两种 prediction_type；
            #   2. 无条件加 rescale_betas_zero_snr=True —— v-pred ≠ zero-SNR，
            #      很多 v-pred 模型并不是 zero-SNR 训练的，强加会掉质量。
            # 现在采样器/预测类型统一由 switch_sampler() 一个入口处理。
            if self.is_vpred:
                logger.info("ℹ️ 检测到 v-pred 模型；采样器与 zero-SNR 由 "
                            "switch_sampler() 统一应用（不再就地改单条管线）")

            # FreeU 参数随旧 UNet 一起消失，换模型后按用户选择重新挂上。
            # 只有用户确实"想要"时才重挂，否则取消勾选后会被静默重新打开。
            self._freeu_enabled = False
            if self._freeu_cfg and getattr(self, "_freeu_wanted", False):
                self._apply_freeu()

            # CLIP Skip 也随新 text_encoder 一起失效 → 按用户选择重新应用。
            # 不重挂的话：用户设了 skip 2，第一次生效，换模型后悄悄回默认。
            # ⚠️ 换到 SDXL / flux / sd3 / anima 时要明确跳过并说明原因，
            # 否则用户会以为设置生效了（实测 SDXL 的 skip 不会改变输出）。
            _cs = int(getattr(self, "clip_skip", 0) or 0)
            if _cs > 0:
                try:
                    from utils.clip_skip import (apply_clip_skip_to_pipes,
                                                 supports_clip_skip)
                    _ok, _why = supports_clip_skip(self)
                    if not _ok:
                        logger.info(f"ℹ️ CLIP Skip 不适用于当前模型: {_why}")
                    else:
                        _n = apply_clip_skip_to_pipes(self, _cs)
                        if _n:
                            logger.info(f"🎚️ CLIP Skip={_cs} 已重新应用（{_n} 个编码器）")
                except Exception as _e:
                    logger.warning(f"⚠️ CLIP Skip 重新应用失败: {_e}")

            logger.info(f"✅ {det.info.display_name} 加载与显存优化完成！")

        except Exception as e:
            logger.exception(f"❌ 模型加载失败: {model_name}")
            # 失败后不能留脏状态，否则后续生成会按"已加载 SDXL"去要 text_encoder_2
            self.detected_arch = None
            self.current_arch_id = None
            self.arch_profile = None
            self.is_flux = self.is_sd3 = self.is_sdxl = False
            self.current_model_name = None
            self.current_model_path = None
            raise Exception(f"模型加载失败: {str(e)}") from e

    # ------------------------------------------------------------
    #  FreeU —— 免训练细节增强
    # ------------------------------------------------------------
    # 参数存在 unet.config.freeu_params 上，而本项目多管线共享同一个 UNet，
    _FREEU_PRESETS = {
        "sdxl": {"s1": 0.6, "s2": 0.4, "b1": 1.1, "b2": 1.2},
        "sd15": {"s1": 0.9, "s2": 0.2, "b1": 1.2, "b2": 1.4},
    }

    def _iter_unets(self):
        """遍历所有管线持有的 UNet，按 id 去重（共享的只返回一次）。"""
        seen = set()
        for attr in ("txt2img_pipe", "img2img_pipe",
                     "inpaint_pipe", "controlnet_pipe"):
            pipe = getattr(self, attr, None)
            unet = getattr(pipe, "unet", None) if pipe is not None else None
            if unet is None or id(unet) in seen:
                continue
            seen.add(id(unet))
            yield attr, unet

    def _iter_pipes(self):
        """遍历本管理器持有的所有管线，按 id 去重（多管线共享组件的只返回一次）。"""
        seen = set()
        for attr in ("txt2img_pipe", "img2img_pipe",
                     "inpaint_pipe", "controlnet_pipe"):
            pipe = getattr(self, attr, None)
            if pipe is None or id(pipe) in seen:
                continue
            seen.add(id(pipe))
            yield attr, pipe

    def freeu_supported(self):
        """FLUX / SD3 是 DiT 架构，没有 UNet 的 skip 连接，不适用。

        注意：diffusers 里 enable_freeu 同时存在于 Pipeline 与
        UNet2DConditionModel 上，但**管线未加载时两者都不在**，
        所以只要任一侧存在就说明当前架构支持（否则 UI 勾选会永远无效）。
        """
        if getattr(self, "is_flux", False) or getattr(self, "is_sd3", False):
            return False
        if any(hasattr(p, "enable_freeu") for _, p in self._iter_pipes()):
            return True
        return any(hasattr(u, "enable_freeu") for _, u in self._iter_unets())

    def get_freeu_defaults(self):
        key = "sdxl" if getattr(self, "is_sdxl", False) else "sd15"
        return dict(self._FREEU_PRESETS[key])

    def enable_freeu(self, **overrides):
        """
        开启 FreeU，overrides 可传 s1/s2/b1/b2 覆盖推荐值。
        返回 True 表示至少成功作用到一个 UNet。
        """
        if not self.freeu_supported():
            logger.info("⏭️ 当前架构不支持 FreeU（仅 SD1.5 / SDXL 系），已跳过")
            return False

        cfg = self.get_freeu_defaults()
        for k in cfg:
            if k in overrides and overrides[k] is not None:
                cfg[k] = float(overrides[k])
            cfg[k] = max(0.01, cfg[k])   # diffusers 对 <=0 的值会告警拒绝
        self._freeu_cfg = cfg
        self._freeu_wanted = True
        return self._apply_freeu()

    def _apply_freeu(self):
        cfg = getattr(self, "_freeu_cfg", None)
        if not cfg:
            return False
        n = 0
        errs = []
        # 优先调用管线级 enable_freeu（会一直持有到管线生命周期结束）
        for attr, pipe in self._iter_pipes():
            fn = getattr(pipe, "enable_freeu", None)
            if not callable(fn):
                continue
            try:
                fn(**cfg)
                n += 1
            except Exception as e:
                errs.append(f"{attr}: {e}")
        # 兜底：直接作用到 UNet
        if n == 0:
            for attr, unet in self._iter_unets():
                try:
                    unet.enable_freeu(**cfg)
                    n += 1
                except Exception as e:
                    errs.append(f"{attr}: {e}")
        self._freeu_enabled = n > 0
        if n:
            logger.info(f"✨ FreeU 已启用 {cfg}（作用于 {n} 个管线）")
        elif errs:
            logger.warning(f"⚠️ FreeU 启用失败: {errs}")
        return self._freeu_enabled

    def disable_freeu(self, keep_wanted: bool = False):
        """关闭 FreeU。

        keep_wanted=True：只是临时摘掉（ADetailer/Hires 修复阶段），
                          保留"用户想要 FreeU"的意图，退出后自动恢复。
        keep_wanted=False：用户真的取消勾选，下次加载模型不再自动挂上。
        """
        disabled = 0
        for attr, pipe in self._iter_pipes():
            fn = getattr(pipe, "disable_freeu", None)
            if not callable(fn):
                continue
            try:
                fn()
                disabled += 1
            except Exception as e:
                logger.debug(f"[{attr}] pipe.disable_freeu 失败: {e}")
        # 兜底：直接作用到 UNet
        if disabled == 0:
            for u_attr, unet in self._iter_unets():
                try:
                    if hasattr(unet, "disable_freeu"):
                        unet.disable_freeu()
                    else:
                        unet.config.freeu_params = None
                except Exception as e:
                    logger.warning(f"⚠️ [{u_attr}] disable_freeu 失败: {e}")
        if getattr(self, "_freeu_enabled", False):
            logger.info("🌙 FreeU 已临时关闭" if keep_wanted else "🌙 FreeU 已关闭")
        self._freeu_enabled = False
        if not keep_wanted:
            self._freeu_wanted = False

    @contextmanager
    def freeu_disabled(self):
        """
        修复阶段（ADetailer 局部重绘 / Hires）临时摘掉 FreeU，退出后自动恢复。
        未开启 FreeU 时是纯 no-op，调用点可以无条件包。
        """
        was = getattr(self, "_freeu_enabled", False)
        if was:
            self.disable_freeu(keep_wanted=True)
        try:
            yield
        finally:
            if was:
                self._apply_freeu()


    # ------------------------------------------------------------
    #  多 LoRA 挂载
    # ------------------------------------------------------------
    # LoRA 子目录名归一化：容忍 sd15 / sd1.5 / SD1.5 等写法
    _LORA_DIR_ALIASES = {
        "sd15": "sd1.5",
        "sd1.5": "sd1.5",
        "sd-1.5": "sd1.5",
        "sdxl": "sdxl",
        "xl": "sdxl",
        "pony": "pony",
        "flux": "flux",
        "sd3": "sd3",
    }

    def _normalize_lora_dir(self, name):
        """把各种模型类写法映射到实际的 loras/ 子目录名"""
        key = (name or "").strip().lower()
        resolved = self._LORA_DIR_ALIASES.get(key)
        if resolved is None:
            logger.warning(f"⚠️ 未知的 LoRA 子目录 '{name}'，按原样使用")
            return name
        return resolved

    def current_lora_subdir(self):
        """根据当前已加载底模推断该用哪个 LoRA 目录"""
        det = getattr(self, "detected_arch", None)
        if det is not None and det.info.lora_subdir:
            return det.info.lora_subdir
        # 兜底：detected_arch 尚未设置（模型还没加载过）
        if getattr(self, "is_flux", False):
            return "flux"
        if getattr(self, "is_sd3", False):
            return "sd3"
        if getattr(self, "is_sdxl", False):
            return "sdxl"
        return "sd1.5"

    @staticmethod
    def _normalize_lora_list(lora_list):
        """把 LoRA 列表规范成 `[(名字, 权重), ...]`。

        接受 (名字, 权重) 元组/列表、`{"name":..,"weight":..}` 字典、
        以及 `{"lora":..,"weight":..}` 这类别名。无法识别的条目**丢弃并报错**，
        绝不把字典的键当成文件名（那会静默跳过整个 LoRA）。
        """
        out = []
        for item in (lora_list or []):
            name = weight = None
            if isinstance(item, dict):
                name = (item.get("name") or item.get("lora")
                        or item.get("lora_name") or item.get("file"))
                weight = item.get("weight", item.get("scale", 1.0))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                name, weight = item[0], item[1]
            elif isinstance(item, str):
                # 允许 "名字:权重" 这种字符串形式
                if ":" in item:
                    name, _, w = item.rpartition(":")
                    try:
                        weight = float(w)
                    except ValueError:
                        name, weight = item, 1.0
                else:
                    name, weight = item, 1.0
            else:
                logger.error(f"❌ LoRA 条目格式无法识别，已丢弃: {item!r}")
                continue
            if not name or not isinstance(name, str):
                logger.error(f"❌ LoRA 名称为空，已丢弃: {item!r}")
                continue
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                logger.warning(f"⚠️ LoRA 权重无法解析，按 1.0 处理: {item!r}")
                weight = 1.0
            out.append((name, weight))
        return out

    def active_loras(self):
        """当前挂在图片管线上的 LoRA 规格 [(文件名, 权重), ...]。

        视频生成会把 LoRA 从底层 pipe 上卸掉（motion module 包了一层 UNet，
        挂着 LoRA 的 hook 会互相干扰），卸之前用它备份、建完再挂回去。
        """
        return list(getattr(self, "_active_lora_spec", []) or [])

    def apply_multiple_loras(self, lora_list, sub_dir=None):
        """
        现代版多 LoRA 加载机制 (极速秒切 + 独立权重)
        格式: [("lora1.safetensors", 0.8), ("lora2.safetensors", 0.5)]
        也接受 [{"name": ..., "weight": ...}]（会被规范化，见下）。
        sub_dir=None 时按当前底模架构自动推断，避免调用方传错命名。

        ⚠️ 为什么必须做规范化
        --------------------
        旧代码直接 `for i, (lora_name, weight) in enumerate(lora_list)`。
        如果调用方传的是 **dict**，这一步不会报错，而是把 dict 的**键**解包出来：
            {"name": "x.safetensors", "weight": 0.8}  →  (lora_name, weight) = ("name", "weight")
        于是它去找一个名叫 "name" 的文件、找不到、`logger.error` 后 `continue`
        —— **LoRA 被静默跳过，出图和不挂 LoRA 一模一样**。
        更糟的是节点里有 `except TypeError` 兜底想兼容元组，而 2 键字典解包
        根本不抛 TypeError，兜底形同虚设。
        实测：`apply_multiple_loras([{"name": l, "weight": 0.9}])` 之后
        `pipe.get_active_adapters()` 仍是 `[]`。
        """
        lora_list = self._normalize_lora_list(lora_list)

        try:
            self.txt2img_pipe.unload_lora_weights()
            logger.info("🧹 已清空旧的 LoRA 缓存。")
        except Exception:
            pass
        self._active_lora_spec = []

        if not lora_list:
            return

        from core.arch import get_arch
        from utils import paths

        if sub_dir is None:
            info = get_arch(getattr(self, 'current_arch_id', None) or "unknown")
            sub_dir = info.lora_subdir
            if not sub_dir:
                logger.warning(f"⚠️ 当前架构 {self.current_arch_id} 无对应 LoRA 目录，跳过挂载")
                return

        logger.info(f"🎨 LoRA 目录: loras/{sub_dir}/")

        lora_root = os.path.join(paths.LORA_DIR, sub_dir)
        if not os.path.isdir(lora_root):
            logger.warning(f"⚠️ LoRA 目录不存在: {lora_root}")
            return

        adapter_names   = []
        adapter_weights = []
        mounted_specs   = []          # 只记真正挂上的（中途失败的不记）

        for i, (lora_name, weight) in enumerate(lora_list):
            lora_path = os.path.join(lora_root, lora_name)
            if not os.path.isfile(lora_path):
                logger.error(f"❌ [跳过] LoRA 文件不存在: {lora_path}")
                continue

            adapter_name = f"lora_slot_{i}"
            if self._load_one_lora(lora_root, lora_name, lora_path, adapter_name):
                adapter_names.append(adapter_name)
                adapter_weights.append(weight)
                mounted_specs.append((lora_name, weight))
                logger.info(f"✅ 挂载插件: {lora_name} (独立权重: {weight})")

        if adapter_names:
            # 记下来，供视频链路"卸载后恢复"使用
            self._active_lora_spec = mounted_specs
            try:
                self.txt2img_pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                logger.info(f"⚡ 已激活插件通道: {adapter_names}，对应权重: {adapter_weights}")
            except Exception as e:
                logger.warning(f"⚠️ 激活多 LoRA 权重时异常: {e}")
        else:
            logger.warning("⚠️ 没有任何 LoRA 成功挂载")

    def _adapter_registered(self, adapter_name: str) -> bool:
        """这个适配器名是不是已经挂到 UNet/Transformer 上了。"""
        pipe = self.txt2img_pipe
        for holder_name in ("unet", "transformer"):
            holder = getattr(pipe, holder_name, None)
            configs = getattr(holder, "peft_config", None)
            if isinstance(configs, dict) and adapter_name in configs:
                return True
        return False

    def _load_one_lora(self, lora_root: str, lora_name: str, lora_path: str,
                       adapter_name: str) -> bool:
        """挂载单个 LoRA：先按原样加载，失败就修键名再试。

        为什么需要"修"：见 `utils/lora_repair.py` 的模块说明 ——
        ① 训练器早期存出来的文件键名带 `.default`，diffusers 直接拒绝；
        ② civitai 的 kohya 格式 LoRA，文本编码器部分在上游转换里必炸
           （`IndexError: list index out of range`）。
        这两种情况的共同点是：**UNet 已经挂上了，只是文本编码器那半没挂**。
        diffusers 把两半放在同一个 `load_lora_weights` 里，所以异常传出来时
        UNet 其实已经生效 —— 先认这个"半成功"，别整块丢掉。
        """
        pipe = self.txt2img_pipe
        try:
            pipe.load_lora_weights(
                lora_root,
                weight_name=lora_name,
                adapter_name=adapter_name,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - 交给修复流程再试一次
            first_error = exc

        if self._adapter_registered(adapter_name):
            logger.warning(
                "🔧 插件 %s：U-Net 部分已挂载，文本编码器部分没挂上（%s）。"
                "这是当前 diffusers 转换链的已知缺陷（本机实测：即使把键名"
                "修成 peft_config 认的样子，文本编码器也不参与编码）。"
                "画风/人物仍会变化，只是提示词里的触发词效果会弱一些。"
                "用本程序重新训练的 LoRA 没有这个问题。", lora_name, first_error)
            return True

        try:
            from safetensors.torch import load_file

            from utils import lora_repair

            state_dict = load_file(lora_path, device="cpu")
            fixed, notes, warnings = lora_repair.repair_lora_state_dict(state_dict)
            if not fixed:
                logger.error(
                    "❌ [跳过] 插件 %s 无法解析：%s | 修复也没救：%s",
                    lora_name, first_error, "；".join(warnings) or "未知原因")
                return False

            pipe.load_lora_weights(fixed, adapter_name=adapter_name)
            for note in notes:
                logger.info("   ↳ 修复：%s", note)
            for warn in warnings:
                logger.warning("   ⚠️ %s", warn)
            logger.info("🔧 插件 %s 已按修复后的键名挂载", lora_name)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "❌ [跳过] 插件 %s 不兼容或损坏: %s（修复尝试也失败: %s）",
                lora_name, first_error, exc)
            return False

    # ------------------------------------------------------------
    #  ControlNet
    # ------------------------------------------------------------
    # 本地 ControlNet 探测顺序（唯一来源，扩展市场的 is_installed 也复用它）。
    # 以前只认 controlnets/{type}/config.json 一个路径，导致：
    #   · 权重明明下好了、只缺 config.json → 判"未安装"，反复提示下载
    #   · HF 缓存里已有完整仓库（models_cache/huggingface/hub/...）→ 看不见
    #   · 单文件 .safetensors（社区最常见的分发格式）→ 完全不支持
    @staticmethod
    def _controlnet_candidates(control_type: str, is_sdxl: bool):
        prefix = "sdxl_" if is_sdxl else ""
        root = paths.CONTROLNET_DIR
        cands = [
            os.path.join(root, f"{prefix}{control_type}"),          # 直接放权重
            os.path.join(root, control_type),                        # 无前缀回退
            os.path.join(root, f"{prefix}{control_type}_diffusers"),
        ]
        return cands

    @classmethod
    def describe_local_controlnet(cls, control_type: str, is_sdxl: bool = False):
        """返回 (可用源, 说明)。源可以是目录、单文件路径或 None。

        探测原则（按优先级，**先找完整可用的，残缺的只作最后兜底**）：
          1. controlnets/{sdxl_}{type}/ 且含 config.json（diffusers 目录）
          2. HF 本地缓存快照（models_cache/huggingface/hub/...）含 config.json
          3. controlnets/ 下的**单文件 checkpoint**（文件名需与当前架构一致）
          4. 都不行 → 返回残缺目录的信息，明确指出缺什么

        ⚠️ 旧实现遇到"目录存在但缺 config.json"就立刻返回失败，
        结果：本地 HF 缓存里明明有完整仓库、单文件也有，却被判"需要下载"。
        """
        import glob as _glob
        root = paths.CONTROLNET_DIR
        prefix = "sdxl_" if is_sdxl else ""
        partial_dirs = []

        # ── 1) 直接放置的 diffusers 目录 ──
        for d in cls._controlnet_candidates(control_type, is_sdxl):
            if os.path.isdir(d):
                if os.path.exists(os.path.join(d, "config.json")):
                    return d, f"diffusers 目录 {d}"
                w = [f for f in os.listdir(d)
                     if f.endswith((".safetensors", ".bin", ".ckpt"))]
                if w:
                    partial_dirs.append(d)

        # ── 2) HF 本地缓存（main.py 已把 HF_HOME 指向 models_cache）──
        # ⚠️ 必须按 control_type 精确一对多映射。曾写成"遍历候选仓库、
        #    第一个完整就返回"，结果 SDXL 的 openpose/depth 会错拿到 canny
        #    的缓存（静默用错模型，比报错更糟）。
        _repo_by_type = {
            ("openpose", False): "lllyasviel/sd-controlnet-openpose",
            ("canny", False):    "lllyasviel/sd-controlnet-canny",
            ("depth", False):    "lllyasviel/sd-controlnet-depth",
            ("openpose", True):  "thibaud/controlnet-openpose-sdxl-1.0",
            ("canny", True):     "diffusers/controlnet-canny-sdxl-1.0",
            ("depth", True):     "diffusers/controlnet-depth-sdxl-1.0",
        }
        repo = _repo_by_type.get((control_type, bool(is_sdxl)))
        if repo:
            try:
                from huggingface_hub import snapshot_download
                p = snapshot_download(repo, local_files_only=True)
                # 双保险：仓库名里必须出现该 control_type，防止再串味
                if p and control_type in repo \
                        and os.path.exists(os.path.join(p, "config.json")):
                    return p, f"HF 本地缓存 {repo}"
            except Exception:
                pass

        # ── 3) 单文件 checkpoint（社区常见分发格式）──
        # 文件名按架构匹配：SDXL 认 xl/sdxl/pony/illu/noob，SD1.5 反而要排除它们，
        # 否则会出现"SD1.5 底模挂了 SDXL ControlNet"的静默错误结果。
        xl_marks = ("xl", "sdxl", "pony", "illu", "noob", "animagine")
        hits = sorted(_glob.glob(os.path.join(root, "*.safetensors")))
        arch_hits, wrong_hits = [], []
        for h in hits:
            base = os.path.basename(h).lower()
            if control_type not in base or "controlnet" not in base \
                    and control_type not in base:
                continue
            looks_xl = any(m in base for m in xl_marks)
            (arch_hits if looks_xl == bool(is_sdxl) else wrong_hits).append(h)
        if arch_hits:
            return arch_hits[0], f"单文件 checkpoint {os.path.basename(arch_hits[0])}"
        if wrong_hits:
            kind = "SD1.5" if is_sdxl else "SDXL"
            return None, (f"只找到 {kind} 系的单文件 "
                          f"{os.path.basename(wrong_hits[0])}，"
                          f"与当前架构（{'SDXL' if is_sdxl else 'SD1.5'}）不匹配。"
                          f"请放入对应的 ControlNet，或换用匹配的底模。")

        # ── 4) 兜底：报告残缺目录，说清缺什么 ──
        if partial_dirs:
            d = partial_dirs[0]
            return None, (f"目录 {d} 里有权重但缺 config.json（下载未完成）。"
                          f"删掉该目录后重新下载，或改用单文件方式放入 "
                          f"{os.path.join(root, prefix + control_type)}.safetensors")
        return None, "本地未找到（可从扩展市场下载，或手动放入 controlnets/）"

    def prepare_controlnet(self, control_type="openpose"):
        CN_ALIAS = {
            "openpose": "openpose", "OpenPose": "openpose", "Openpose": "openpose",
            "OPENPOSE": "openpose", "姿势": "openpose", "骨架": "openpose",
            "canny": "canny", "Canny": "canny", "CANNY": "canny",
            "线稿": "canny", "边缘": "canny",
            "depth": "depth", "Depth": "depth", "DEPTH": "depth",
            "深度": "depth", "深度图": "depth",
        }

        raw_type = str(control_type).strip()
        control_type = CN_ALIAS.get(raw_type, raw_type.lower())
        _sdxl_now = bool(getattr(self, 'is_sdxl', False))
        _cn_key = (_sdxl_now, control_type)

        # 已加载同类型 **且同架构** → 复用（原来只看 control_type，
        # SD1.5 ⇄ SDXL 切换时会复用到错误架构的 ControlNet）
        if (getattr(self, '_cn_key', None) == _cn_key
                and getattr(self, 'controlnet_pipe', None) is not None):
            # 🔧 即便复用,也要确保 image_encoder 同步 (IP-Adapter 可能后加载)
            self._sync_ipa_components_to_controlnet()
            if getattr(self, "_freeu_cfg", None):
                self._apply_freeu()
            return

        logger.info(f"🔄 正在配置 ControlNet: {control_type} "
              f"(原始输入: {raw_type}, SDXL={_sdxl_now})...")

        model_id_map = {
            "openpose": "lllyasviel/sd-controlnet-openpose",
            "canny":    "lllyasviel/sd-controlnet-canny",
            "depth":    "lllyasviel/sd-controlnet-depth",
        }
        sdxl_model_id_map = {
            "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
            "canny":    "diffusers/controlnet-canny-sdxl-1.0",
            "depth":    "diffusers/controlnet-depth-sdxl-1.0",
        }

        use_map = (sdxl_model_id_map
                   if getattr(self, 'is_sdxl', False)
                   else model_id_map)

        if control_type not in use_map:
            raise ValueError(
                f"❌ 未知 ControlNet 类型: '{raw_type}'\n"
                f"   归一化后: '{control_type}'\n"
                f"   可用类型: {list(use_map.keys())}"
            )

        if _cn_key not in self.loaded_controlnets:
            cn_source, how = self.describe_local_controlnet(control_type, _sdxl_now)
            local_only = bool(cn_source)
            if cn_source:
                logger.info(f"📂 ControlNet 本地命中 → {how}")
            else:
                cn_model_id = use_map[control_type]
                logger.info(f"🌐 本地无可用 ControlNet（{how}），"
                            f"从在线获取: {cn_model_id} "
                            f"(via {os.environ.get('HF_ENDPOINT', 'huggingface.co')})")
                cn_source = cn_model_id
                local_only = False

            try:
                if local_only and cn_source.lower().endswith(
                        (".safetensors", ".ckpt", ".bin")):
                    # 单文件 checkpoint（社区最常见的 ControlNet 分发格式）
                    from diffusers import ControlNetModel as _CNM
                    self.loaded_controlnets[_cn_key] = _CNM.from_single_file(
                        cn_source, torch_dtype=self.dtype,
                    )
                else:
                    self.loaded_controlnets[_cn_key] = \
                        ControlNetModel.from_pretrained(
                            cn_source,
                            torch_dtype=self.dtype,
                            local_files_only=local_only,
                        )
            except Exception as e:
                _cands = self._controlnet_candidates(control_type, _sdxl_now)
                raise RuntimeError(
                    f"❌ ControlNet 加载失败: {e}\n\n"
                    f"💡 三种可用做法（任选其一）:\n"
                    f"   1. 扩展市场下载（会自动落到 {_cands[0]}）\n"
                    f"   2. 手动放 diffusers 目录: {_cands[0]}/"
                    f"（需 config.json + diffusion_pytorch_model.safetensors）\n"
                    f"   3. 手动放单文件: {os.path.join(paths.CONTROLNET_DIR, control_type)}"
                    f".safetensors（自动兼容）\n"
                    f"   4. 或在 UI 中关闭 ControlNet，改用普通图生图/文生图"
                ) from e

        controlnet = self.loaded_controlnets[_cn_key]

        pipe_class = (StableDiffusionXLControlNetPipeline
                      if _sdxl_now
                      else StableDiffusionControlNetPipeline)

        common_kwargs = dict(
            vae          = self.txt2img_pipe.vae,
            text_encoder = self.txt2img_pipe.text_encoder,
            tokenizer    = self.txt2img_pipe.tokenizer,
            unet         = self.txt2img_pipe.unet,
            scheduler    = self.txt2img_pipe.scheduler,
            controlnet   = controlnet,
        )

        if _sdxl_now:
            self.controlnet_pipe = pipe_class(
                **common_kwargs,
                text_encoder_2 = self.txt2img_pipe.text_encoder_2,
                tokenizer_2    = self.txt2img_pipe.tokenizer_2,
            )
        else:
            self.controlnet_pipe = pipe_class(
                **common_kwargs,
                safety_checker          = None,
                feature_extractor       = None,
                requires_safety_checker = False,
            )

        # ── 显存策略统一 ──
        # 以前是「ControlNet .to(device) + 新管线 .to(device)」两处硬搬：
        # 主 SDXL 管线若已启用 CPU offload，这一步会把**共享的** UNet/VAE/文本
        # 编码器重新搬上 GPU，造成显存峰值或 hook 状态错乱（8GB 上尤其明显）。
        # 现在：ControlNet 自己按预算安置；共享组件跟随主管线已有策略。
        _shared_offloaded = any(
            getattr(m, "_hf_hook", None) is not None
            for m in (self.txt2img_pipe.unet, self.txt2img_pipe.vae,
                      self.txt2img_pipe.text_encoder)
        ) if self.txt2img_pipe is not None else False

        if _shared_offloaded:
            logger.info("⏭️ 共享组件已挂 offload hook → ControlNet 走 CPU offload")
            try:
                # ControlNet 不在主管线的 offload 计划里，单独挂 hook
                self.controlnet_pipe.enable_model_cpu_offload(gpu_id=0)
            except Exception as e:
                logger.warning(f"⚠️ ControlNet offload 失败，改为仅未共享组件上卡: {e}")
                try:
                    controlnet.to(self.device)
                except Exception:
                    pass
        else:
            # 全 GPU 模式：ControlNet 与整条新管线一起上卡
            controlnet.to(self.device)
            self.controlnet_pipe.to(self.device)

        # 轻量优化
        self._apply_light_optimizations(self.controlnet_pipe, name="controlnet")

        self.current_cn_type = control_type
        self._cn_key = _cn_key
        logger.info(f"✅ ControlNet ({control_type}) 加载完成（SDXL={_sdxl_now}）")

        self._sync_ipa_components_to_controlnet()

        # ===== 预处理器(检测器) =====
        if control_type == "openpose" and not self.pose_detector:
            logger.info("⏳ 正在加载 OpenPose 骨架提取器...")
            local_annot = os.path.join(paths.CONTROLNET_DIR, "Annotators")
            attempts = []
            # 1) controlnets/Annotators（用户直接放置）
            if os.path.exists(os.path.join(local_annot, "body_pose_model.pth")):
                attempts.append(("本地目录", local_annot, True))
            # 2) HF 本地缓存（实测本机已有 lllyasviel/Annotators）
            try:
                from huggingface_hub import snapshot_download
                p = snapshot_download("lllyasviel/Annotators",
                                      local_files_only=True)
                if p:
                    attempts.append((f"HF 缓存 {p}", p, True))
            except Exception:
                pass

            for label, src, local_only in attempts:
                try:
                    self.pose_detector = OpenposeDetector.from_pretrained(
                        src, local_files_only=local_only)
                    logger.info(f"✅ OpenPose 提取器就绪（{label}）")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ OpenPose 从此处加载失败（{label}）: {e}")

            if self.pose_detector is None:
                # 不再静默联网下载：明确告知缺什么、怎么获取。
                logger.warning(
                    "⚠️ OpenPose 提取器本地不可用（controlnets/Annotators "
                    "与 HF 缓存都没有）。姿势迁移/骨架提取将不可用。\n"
                    "   自行获取：huggingface-cli download lllyasviel/Annotators\n"
                    "   或改用 Canny（无需任何模型）")

        elif control_type == "depth" and not self.depth_estimator:
            logger.info("⏳ 正在加载 Depth 深度图提取器...")
            # 显式指定 repo（原来是 pipeline('depth-estimation')，
            # 用 transformers 的隐式默认模型，用户完全看不见也难离线准备）。
            #
            # ⚠️ 实测：本项目缓存里**没有** Intel/dpt-hybrid-midas，
            # 所以 depth 目前必然需要联网下载（约 470MB）。
            # 按"模型由用户自行下载、不静默联网"的定位，这里只尝试本地缓存；
            # 本地没有就**明确报错并给出获取方式**，不偷偷下载。
            depth_repo = "Intel/dpt-hybrid-midas"
            local_only = True
            try:
                from transformers import pipeline
                self.depth_estimator = pipeline(
                    'depth-estimation', model=depth_repo,
                    local_files_only=local_only)
                logger.info(f"✅ Depth 提取器就绪（本地缓存 {depth_repo}）")
            except Exception as e:
                self.depth_estimator = None
                logger.warning(f"⚠️ Depth 提取器本地不可用: {e}")
                raise RuntimeError(
                    "❌ Depth 深度图提取器未就绪：本地缓存里没有 "
                    f"{depth_repo}（约 470MB）。\n"
                    "💡 三种做法（任选其一）：\n"
                    "   1. 自行下载到缓存后重试：\n"
                    "      huggingface-cli download Intel/dpt-hybrid-midas\n"
                    "   2. 改用 Canny（无需任何模型）或 OpenPose\n"
                    "   3. 关闭 ControlNet，用普通图生图/文生图\n"
                    "（本程序不会在生成流程里静默下载该模型）") from e

    def _mark_ipa_state(self, loaded, scale=None):
        """
        给各个 Pipeline 写入明确的 IPA 状态。

        ADetailer 不再通过 hasattr() 猜测 IPA 是否加载，
        而是读取 _ai_ipa_loaded。
        """
        pipe_names = (
            "txt2img_pipe",
            "img2img_pipe",
            "inpaint_pipe",
            "controlnet_pipe",
        )

        visited = set()

        for pipe_name in pipe_names:
            pipe = getattr(self, pipe_name, None)

            if pipe is None:
                continue

            pipe_id = id(pipe)

            if pipe_id in visited:
                continue

            visited.add(pipe_id)

            pipe._ai_ipa_loaded = bool(loaded)

            if scale is not None:
                pipe._ai_ipa_scale = scale


    def _sync_ipa_components_to_controlnet(self):
        """
        将已经加载到 txt2img_pipe 的 IPA 组件同步给
        controlnet_pipe。

        这里只同步 pipeline 级别组件，不重复调用
        load_ip_adapter()，避免重复加载权重。
        """
        src = getattr(self, "txt2img_pipe", None)
        dst = getattr(self, "controlnet_pipe", None)

        if src is None or dst is None:
            return

        if not getattr(self, "ip_adapter_loaded", False):
            logger.info(
                "ℹ️ [sync] 当前没有加载 IPA，跳过同步"
            )
            return

        # IP-Adapter 通常使用共享 UNet。
        # 如果不是共享 UNet，仍然允许同步 pipeline 组件，
        # 但记录警告，方便检查模型管理逻辑。
        src_unet = getattr(src, "unet", None)
        dst_unet = getattr(dst, "unet", None)

        if (
            src_unet is not None
            and dst_unet is not None
            and src_unet is not dst_unet
        ):
            logger.warning(
                "⚠️ [sync] txt2img_pipe 与 controlnet_pipe "
                "没有共享同一个 UNet"
            )

        # 同步视觉编码器
        src_image_encoder = getattr(
            src,
            "image_encoder",
            None,
        )

        if src_image_encoder is not None:
            dst.image_encoder = src_image_encoder
            logger.info(
                "🔧 [sync] image_encoder 已同步到 controlnet_pipe"
            )

        # 同步 CLIP 图像预处理器
        src_feature_extractor = getattr(
            src,
            "feature_extractor",
            None,
        )

        if src_feature_extractor is not None:
            dst.feature_extractor = src_feature_extractor
            logger.info(
                "🔧 [sync] feature_extractor 已同步到 controlnet_pipe"
            )

        # 某些 diffusers 版本使用 image_processor，
        # 这里一并兼容。
        src_image_processor = getattr(
            src,
            "image_processor",
            None,
        )

        if src_image_processor is not None:
            dst.image_processor = src_image_processor

        current_scale = getattr(
            self,
            "current_ipa_scale",
            getattr(src, "_ai_ipa_scale", 0.7),
        )

        try:
            setter = getattr(
                dst,
                "set_ip_adapter_scale",
                None,
            )

            if callable(setter):
                setter(current_scale)
                logger.info(
                    "🔧 [sync] controlnet_pipe IPA scale = %.3f",
                    float(current_scale),
                )
            else:
                dst_unet_setter = getattr(
                    dst_unet,
                    "set_ip_adapter_scale",
                    None,
                )

                if callable(dst_unet_setter):
                    dst_unet_setter(current_scale)

        except Exception as error:
            logger.warning(
                "⚠️ [sync] 设置 controlnet_pipe IPA scale 失败：%s",
                error,
            )

        # 写入明确状态
        dst._ai_ipa_loaded = True
        dst._ai_ipa_scale = current_scale

        # 共享 UNet 时，inpaint_pipe 的注意力处理器实际上
        # 也可能已经受到 IPA 影响，所以需要让 ADetailer
        # 能够识别这个状态。
        inpaint_pipe = getattr(
            self,
            "inpaint_pipe",
            None,
        )

        if inpaint_pipe is not None:
            inpaint_unet = getattr(
                inpaint_pipe,
                "unet",
                None,
            )

            if inpaint_unet is src_unet:
                inpaint_pipe._ai_ipa_loaded = True
                inpaint_pipe._ai_ipa_scale = current_scale

        logger.info(
            "✅ [sync] IPA 状态同步完成"
        )

    def get_control_image(self, input_image, control_type="openpose"):
        control_type = str(control_type).strip().lower()
        if self.current_cn_type != control_type:
            self.prepare_controlnet(control_type)

        if control_type == "openpose":
            # 检测器加载失败时 prepare_controlnet 只 warning，
            # pose_detector 仍是 None → 这里会抛 "NoneType object is not callable"，
            # 用户完全看不出是"缺预处理器"。
            if self.pose_detector is None:
                raise RuntimeError(
                    "❌ OpenPose 骨架提取器未就绪（通常是首次使用需要联网下载）。\n"
                    f"   可手动放置: {os.path.join(paths.CONTROLNET_DIR, 'Annotators')}"
                    "（需含 body_pose_model.pth）\n"
                    "   或在 UI 中改用 Canny / Depth，或关闭 ControlNet。")
            return self.pose_detector(input_image)

        elif control_type == "canny":
            image = np.array(input_image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            return Image.fromarray(image)

        elif control_type == "depth":
            if self.depth_estimator is None:
                raise RuntimeError(
                    "❌ Depth 深度图提取器未就绪（通常是首次使用需要联网下载）。\n"
                    "   可改用 Canny / OpenPose，或关闭 ControlNet。")
            return self.depth_estimator(input_image)['depth']

        return input_image

    # ------------------------------------------------------------
    #  IP-Adapter 权重解析（架构感知）
    # ------------------------------------------------------------
    # SDXL 必须用 SDXL 版权重（h94/IP-Adapter 的 sdxl 子目录）。
    # 旧实现无论 SD1.5 还是 SDXL 都硬编码 "*_sd15.safetensors"，
    # 会导致"看起来装上了、实际输出错误"——比报错更危险。
    _IPA_WEIGHTS = {
        # (is_sdxl, variant) -> (repo_subfolder, weight_name)
        (False, "plus"):      ("models",       "ip-adapter-plus_sd15.safetensors"),
        (False, "plus-face"): ("models",       "ip-adapter-plus-face_sd15.safetensors"),
        (False, "base"):      ("models",       "ip-adapter_sd15.safetensors"),
        (True,  "plus"):      ("sdxl_models",  "ip-adapter-plus_sdxl_vit-h.safetensors"),
        (True,  "plus-face"): ("sdxl_models",  "ip-adapter-plus-face_sdxl_vit-h.safetensors"),
        (True,  "base"):      ("sdxl_models",  "ip-adapter_sdxl_vit-h.safetensors"),
    }

    def installed_ipa_variants(self):
        """本机 `models/ip_adapter/` 下**实际装了**哪些档位（当前架构）。

        用途：
          · 档位不匹配时自动改用装了的那个（见 resolve_ipa_weights）；
          · 多参考图里按角色降级（见 core/ipa_multi.resolve_role_plan）；
          · 界面/测试可以据此显示"你装了哪些"。

        ⚠️ 必须连 **image_encoder 一起看**才算"装了"：IP-Adapter 的权重
        只有 44MB，真正不可替代的是 2.4GB 的 CLIP-ViT-H 视觉编码器。
        只有权重文件而没有 encoder 时，`load_ip_adapter()` 会在加载
        encoder 那一步失败 —— 那种"看起来装了"的档位不能用，否则多参考图
        会走到一个必然失败的路径上。
        """
        is_sdxl = bool(getattr(self, "is_sdxl", False))
        local_dir = os.path.join(paths.MODEL_DIR, "ip_adapter")
        has_encoder = self._ipa_local_image_encoder() is not None
        out = []
        for (xl, variant), (sub, name) in self._IPA_WEIGHTS.items():
            if xl != is_sdxl:
                continue
            if has_encoder and (
                    os.path.isfile(os.path.join(local_dir, name))
                    or os.path.isfile(os.path.join(local_dir, sub, name))):
                if variant not in out:
                    out.append(variant)
        # 常见档位优先：plus-face（身份）> plus > base > light
        order = {"plus-face": 0, "plus": 1, "base": 2, "light": 3}
        out.sort(key=lambda v: order.get(v, 9))
        return out

    # ------------------------------------------------------------
    #  本地视觉编码器（IP-Adapter 的 image_encoder）
    # ------------------------------------------------------------
    #: 判定"这个目录是一个可用的 CLIPVisionModelWithProjection"所需的文件。
    #: diffusers 在目录里只认 config.json；权重可能是 .safetensors 或 .bin
    #: （旧仓库/老版本 transformers 导出的就是 bin）。
    _IMAGE_ENCODER_MARKERS = ("config.json",)

    @classmethod
    def _ipa_local_image_encoder(cls, base_dir: Optional[str] = None):
        """返回本机可用的 IP-Adapter 视觉编码器目录；没有返回 None。

        探测顺序（**全部只查本地，绝不联网**）：
          1. models/ip_adapter/image_encoder
          2. models/ip_adapter/models/image_encoder（h94 仓库布局照搬下来的）
          3. models/image_encoder

        为什么要这一步：diffusers 的 `load_ip_adapter()` 在
        `image_encoder` 为空时会用 `CLIPVisionModelWithProjection.from_pretrained`
        去加载；如果传进去的路径里没有 config.json，它就会去 Hub 找 —— 
        离线时这一步实测要 **295 秒**才失败（见任务记录里的
        `_get_model_file(local_files_only=False)` 计时）。先在本机把目录
        找出来并校验过，才能保证失败是**秒级**的。
        """
        root = base_dir or os.path.join(paths.MODEL_DIR, "ip_adapter")
        cands = [
            os.path.join(root, "image_encoder"),
            os.path.join(root, "models", "image_encoder"),
            os.path.join(root, "sdxl_models", "image_encoder"),
            os.path.join(paths.MODEL_DIR, "image_encoder"),
        ]
        for d in cands:
            if not os.path.isdir(d):
                continue
            if not all(os.path.isfile(os.path.join(d, m))
                       for m in cls._IMAGE_ENCODER_MARKERS):
                logger.warning(
                    f"⚠️ {d} 缺 config.json，不能当作 IP-Adapter 的视觉编码器")
                continue
            has_weight = any(
                f.startswith(("model.safetensors", "pytorch_model"))
                and f.endswith((".safetensors", ".bin"))
                for f in os.listdir(d))
            if not has_weight:
                logger.warning(
                    f"⚠️ {d} 里没有 model.safetensors/pytorch_model.bin，"
                    f"不能当作 IP-Adapter 的视觉编码器")
                continue
            return d
        return None

    def _ipa_hf_cache_hit(self, subfolder: str, weight_name: str):
        """在 HF 本地缓存里找 h94/IP-Adapter 的某个权重文件（**纯文件系统查找**）。

        返回 (repo_id, subfolder, weight_name) 或 None。

        ⚠️ 为什么不用 `hf_hub_download(..., local_files_only=True)`：
        本机实测，**缓存未命中**时它一次要 **6.2 秒**（`[hf_download] 6.216s
        -> LocalEntryNotFoundError`）。原因就是题目里怀疑的那条：即便
        local_files_only=True，huggingface_hub 仍要把整个缓存目录走一遍
        才能确定"真的没有"（本机 models_cache 里有好几个 GB 的模型仓库）。
        而我们**每次**解析一个没装的档位都会走到这里 —— 于是"选了个没装的
        档位"就要白等 6 秒多，再叠加真正联网那条路（实测 295.6 秒）。

        直接按 HF 的缓存布局拼路径是 **0.00 秒**（只做几次 os.path.isfile）：
            models_cache/huggingface/hub/models--h94--IP-Adapter/
                snapshots/<rev>/models/ip-adapter-plus_sd15.safetensors
        这样"失败要快"才真正成立。
        """
        hub = os.environ.get("HF_HUB_CACHE")
        if not hub:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                hub = os.path.join(hf_home, "hub")
            else:
                hub = os.path.join(os.path.expanduser("~"), ".cache",
                                   "huggingface", "hub")
        repo_dir = os.path.join(hub, "models--h94--IP-Adapter")
        snaps = os.path.join(repo_dir, "snapshots")
        if not os.path.isdir(snaps):
            return None
        try:
            revs = sorted(os.listdir(snaps))
        except Exception:
            return None
        rel = os.path.join(subfolder, weight_name) if subfolder else weight_name
        for rev in revs:
            base = os.path.join(snaps, rev)
            if not os.path.isdir(base):
                continue
            cand = os.path.join(base, rel)
            if os.path.isfile(cand):
                logger.info(f"📂 使用 HF 缓存的 IP-Adapter: {cand}")
                return "h94/IP-Adapter", subfolder, weight_name
        return None

    def resolve_ipa_weights(self, variant: str = "plus"):
        """返回 (repo_or_local_dir, subfolder, weight_name)。**只查本地，绝不联网。**

        优先级：
          1. 项目内 models/ip_adapter/<weight>（用户自行下载，可离线）
          2. HF 本地缓存（`hf_hub_download(local_files_only=True)` 命中快照）
          3. 请求的档位没有、但**装了别的档位** → 用装了的那一个（带中文警告）
          4. 都没有 → **立刻**抛错，说清缺哪个文件、怎么装

        ⚠️ 关于"绝不联网"（这是实测出来的 P0 修复）
        -------------------------------------------
        旧实现第 3 步是"交给 diffusers 在线拉取"，也就是返回
        `"h94/IP-Adapter"` 让 `load_ip_adapter()` 自己下载。代价实测如下
        （本机 8GB RTX 5060，直接计时 `_get_model_file`）：

            local_files_only=True  → 0.0s，OSError（本地没有）
            local_files_only=False → **295.6s**，OSError（联网超时/重试后失败）

        而 `load_ip_adapter()` 调 `_get_model_file` 时**不带** local_files_only，
        所以"选了一个本机没装的档位"= 卡 2~10 分钟才报错。父任务量到的
        126s 就是这个阶段（网络状况不同，时长会变，但量级一致）。
        另外这条路径失败会连带触发 `unload_ip_adapter()`，历史上会把
        `image_processor` 清成 None 从而弄坏整个会话。

        现在：本机没有就在**毫秒级**抛 `RuntimeError`，且消息里写明缺哪个
        文件、去哪装。想在线拉权重的用户请显式走扩展市场（那条路有进度条）。

        找不到匹配架构的权重时也绝不悄悄退回 SD1.5 权重（那会"看起来装上了、
        实际输出错误"，比报错更危险）。
        """
        is_sdxl = bool(getattr(self, "is_sdxl", False))
        arch_label = "SDXL" if is_sdxl else "SD1.5"
        variant = (variant or "plus").lower()
        # 界面下拉写的是 "standard"，而 h94 官方档位名是 "base" —— 两者同一个文件。
        # 不做这个映射的话，选"standard (轻量)"会查不到任何档位。
        variant = {"standard": "base", "light": "base"}.get(variant, variant)

        entry = self._IPA_WEIGHTS.get((is_sdxl, variant))
        if entry is None:
            # 请求的 variant 在该架构下没有官方发布 → 回退到同架构的安全档位
            fallback = "base" if variant == "plus-face" else "plus"
            entry = self._IPA_WEIGHTS.get((is_sdxl, fallback))
            if entry is None:
                raise RuntimeError(
                    f"❌ 没有适用于 {arch_label} 的 IP-Adapter 权重"
                    f"（variant={variant}）")
            logger.warning(f"⚠️ variant={variant} 在 "
                           f"{arch_label} 下不可用，"
                           f"回退到 {fallback}")
        subfolder, weight_name = entry

        local_dir = os.path.join(paths.MODEL_DIR, "ip_adapter")

        # ── 0) 视觉编码器先落地检查：没有 encoder 的"权重已安装"是假象 ──
        # 只有 44MB 的权重、没有 2.4GB 的 CLIP-ViT-H 时，
        # load_ip_adapter() 会在加载 encoder 那一步去 Hub 找（离线时几百秒）。
        # 所以这里一并说清楚，让失败发生在**毫秒级**且原因准确。
        encoder_dir = self._ipa_local_image_encoder(local_dir)
        encoder_hint = ""
        if encoder_dir is None:
            encoder_hint = (
                f"\n   ⚠️ 同时没有找到视觉编码器（image_encoder）。IP-Adapter 需要\n"
                f"      {os.path.join(local_dir, 'image_encoder')}"
                f"（config.json + model.safetensors，约 2.5GB）。\n"
                f"      只放 44MB 的 ip-adapter*.safetensors 是**不够**的。")

        # ── 1) 项目内本地权重 ──
        if os.path.isfile(os.path.join(local_dir, weight_name)):
            logger.info(f"📂 使用本地 IP-Adapter 权重: {weight_name}")
            return local_dir, "", weight_name
        if os.path.isfile(os.path.join(local_dir, subfolder, weight_name)):
            logger.info(f"📂 使用本地 IP-Adapter 权重: {subfolder}/{weight_name}")
            return local_dir, subfolder, weight_name

        # ── 2) HF 本地缓存（**纯文件系统查找**，不调 hf_hub_download）──
        # 缓存未命中时 hf_hub_download(local_files_only=True) 实测要 6.2 秒，
        # 而我们每次解析没装的档位都会走到这里。详见 _ipa_hf_cache_hit。
        hit = self._ipa_hf_cache_hit(subfolder, weight_name)
        if hit is not None:
            return hit

        # ── 3) 请求的档位本机没有，但装了别的档位 → 用装了的那一个 ──
        #
        # 为什么加这一步：界面默认档位是 "plus (推荐)"，而扩展市场里的
        # `ip_adapter_sd15` 装下来的是 **standard**（models/ip-adapter_sd15.safetensors）。
        # 也就是说"从自家市场装好、按默认打开"必然命中不了 plus。旧代码于是
        # 去在线拉取，离线时一挂就是几分钟到十几分钟（实测 126s / 669s）
        # 最后失败。现在：本机有可用档位就直接用，并明确告诉你用的是哪个档位。
        try:
            installed = self.installed_ipa_variants()
        except Exception:
            installed = []
        if installed and self._IPA_WEIGHTS.get((is_sdxl, installed[0])):
            alt = installed[0]
            alt_sub, alt_name = self._IPA_WEIGHTS[(is_sdxl, alt)]
            for cand, sub in ((os.path.join(local_dir, alt_name), ""),
                              (os.path.join(local_dir, alt_sub, alt_name),
                               alt_sub)):
                if os.path.isfile(cand):
                    logger.warning(
                        f"⚠️ 本机没有 IP-Adapter 档位「{variant}」，"
                        f"改用已安装的「{alt}」（{alt_name}）。"
                        f"想要 {variant} 请在扩展市场安装或在 "
                        f"models/ip_adapter/ 放入 {weight_name}")
                    return local_dir, sub, alt_name
        else:
            installed = installed or []

        # ── 4) 本机确实没有 → **立刻**给出可操作的错误，不去在线拉取 ──
        raise RuntimeError(
            f"❌ 本机没有可用的 IP-Adapter 权重（档位 {variant}，{arch_label}）。\n"
            f"   需要的文件：{os.path.join(local_dir, weight_name)}\n"
            f"   请到「工具 → 🛒 扩展市场 → IP-Adapter {arch_label}」安装，"
            f"或把该文件手动放入 models/ip_adapter/。\n"
            f"   已安装的档位：{installed or '（无）'}"
            f"{encoder_hint}\n"
            f"   （本节点不会自动联网下载，也不会为此等待网络超时。）")

    @staticmethod
    def _ipa_clean_attn_processor(unet) -> bool:
        """把 UNet 的注意力处理器复位成干净的标准实现。

        为什么每次安装前都要复位：`load_ip_adapter()` 是**叠加**语义
        （`unet._load_ip_adapter_weights(state_dicts)` 会按传入的 state_dict
        数量重建 `encoder_hid_proj`）。如果上一次已经装过 IPA、attn processor
        上还挂着 `IPAdapterAttnProcessor2_0`（带 to_k_ip/to_v_ip），再装一次
        会出现两组 IP 权重而 scale 列表只覆盖一组，症状是"图变了但不受控制"。
        复位的代价只有几毫秒（只是换 processor 对象，不碰权重）。
        """
        if unet is None:
            return False
        try:
            from diffusers.models.attention_processor import (
                AttnProcessor2_0, AttnProcessor)
            import torch.nn.functional as F
            proc = (AttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else AttnProcessor)
            unet.set_attn_processor(proc())
            return True
        except Exception as e:
            logger.warning(f"  ⚠️ 复位共享 UNet processor 失败: {e}")
            return False

    def _ipa_prepare_pipes(self):
        """安装 IPA 前的共用准备：关 attention slicing + 复位共享 UNet。

        返回 (primary_pipe, shared_unet)。
        """
        primary = getattr(self, "txt2img_pipe", None)
        if primary is None:
            raise RuntimeError("❌ 尚未加载底模，无法安装 IP-Adapter")

        # IPA 与 attention slicing 不兼容（切片会把 IP 的额外 token 切散）
        for attr in ("txt2img_pipe", "img2img_pipe",
                     "inpaint_pipe", "controlnet_pipe"):
            p = getattr(self, attr, None)
            if p is None:
                continue
            try:
                if hasattr(p, "disable_attention_slicing"):
                    p.disable_attention_slicing()
            except Exception:
                pass

        shared_unet = getattr(primary, "unet", None)
        self._ipa_clean_attn_processor(shared_unet)
        return primary, shared_unet

    def _ipa_encoder_kwargs(self):
        """`load_ip_adapter(..., image_encoder_folder=)` 该传什么。

        找到本机 image_encoder 目录就传**绝对路径**（diffusers 会在
        `Path(subfolder, image_encoder_folder)` 上取，绝对路径会直接覆盖
        subfolder，从而彻底杜绝"去 Hub 找 encoder"的联网等待）；
        找不到就返回 `{}`，让调用方按原样走（并由调用方给出明确报错）。
        """
        d = self._ipa_local_image_encoder()
        if d:
            return {"image_encoder_folder": os.path.abspath(d)}
        return {}

    def _ipa_sync_pipe_components(self, primary):
        """把管线级 IPA 组件（image_encoder / feature_extractor /
        image_processor）补齐给其它共享同一 UNet 的管线。

        不重复调用 `load_ip_adapter()`：attn processor / encoder_hid_proj
        已经在共享 UNet 上，其它管线天然可见。
        """
        shared_img_enc = getattr(primary, "image_encoder", None)
        shared_feat = getattr(primary, "feature_extractor", None)
        shared_img_proc = getattr(primary, "image_processor", None)

        for attr in ("img2img_pipe", "inpaint_pipe", "controlnet_pipe"):
            p = getattr(self, attr, None)
            if p is None or p is primary:
                continue
            try:
                if shared_img_enc is not None and getattr(p, "image_encoder", None) is None:
                    p.image_encoder = shared_img_enc
                if shared_feat is not None and getattr(p, "feature_extractor", None) is None:
                    p.feature_extractor = shared_feat
                if shared_img_proc is not None and getattr(p, "image_processor", None) is None:
                    p.image_processor = shared_img_proc
                logger.debug(f"  → {attr}: 组件已同步（未重复安装权重）")
            except Exception as e:
                logger.warning(f"  ⚠️ {attr} 组件同步失败: {e}")

    def _install_ipa_once(self, variant: str):
        """在**共享 UNet 上只安装一次** IP-Adapter，然后同步各管线级组件。

        为什么必须这样做：
          `load_ip_adapter()` 会改写 `self.unet`（set_attn_processor +
          encoder_hid_proj，已对 diffusers 0.40 源码核实）。而
          txt2img / img2img / inpaint / controlnet 四条管线**共享同一个 UNet**，
          所以对每条管线都调一次 = 对同一 UNet 重复安装，
          后一次会覆盖前一次的状态，容易出 device mismatch 或结构错乱。

        返回 (ok, source, subfolder, weight_name)
        """
        primary, _shared_unet = self._ipa_prepare_pipes()
        src, subfolder, weight_name = self.resolve_ipa_weights(variant)

        logger.info(f"🎭 在共享 UNet 上安装 IP-Adapter 一次 "
                    f"({subfolder}/{weight_name})")
        primary.load_ip_adapter(src, subfolder=subfolder,
                                weight_name=weight_name,
                                **self._ipa_encoder_kwargs())
        self._ipa_sync_pipe_components(primary)
        return True, src, subfolder, weight_name

    @staticmethod
    def _ipa_file_stamp(path: str) -> tuple:
        """适配器权重文件的"身份"：(体积, mtime)。

        为什么不是 sha256：SD1.5/SDXL 的适配器 40~700MB，每次生成都哈希一遍
        太贵（实测一个 base 适配器 0.3s 起）；体积 + mtime 已经能发现
        "同名文件被替换/重新下载"这类真正会用到旧权重的情况。
        """
        try:
            st = os.stat(path)
            return (int(st.st_size), int(st.st_mtime))
        except OSError:
            return (0, 0)

    def prepare_ip_adapter_multi(self, refs):
        """**一次安装 N 个 IP-Adapter**（每个一条参考图、一个独立 scale）。

        这是多参考图功能的落点：`IPAdapterIdentity` / `IPAdapterStyle` /
        `IPAdapterComposition` 三个节点累加出 `refs` 列表，最后在这里
        变成"一次安装 + 每适配器独立强度"，而不是跑 N 遍采样。

        —— diffusers 0.40 到底支持什么（已逐行核实，含文件:行号）
        ----------------------------------------------------------
        1) **可以一次装多个适配器**：`load_ip_adapter()` 接受列表。
           `diffusers/loaders/ip_adapter.py:59-66`
               def load_ip_adapter(self, pretrained_model_name_or_path_or_dict:
                       str | list[str] | dict..., subfolder: str | list[str],
                       weight_name: str | list[str], ...)
           `:117-135` 把三者规范成等长列表并校验；`:163-165` 逐项 zip 读取
           state_dict；`:196` 收集成 `state_dicts = [...]`；
           `:244-245`
               unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
               unet._load_ip_adapter_weights(state_dicts, low_cpu_mem_usage=...)
           `diffusers/loaders/unet.py:696-724` 为**每个** state_dict 建一个
           投影层，最后
               self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
               self.config.encoder_hid_dim_type = "ip_image_proj"
           → 结论：**装 N 个是一个调用，不是 N 次调用**。
             （⚠️ 反复调 `load_ip_adapter()` 是**覆盖**语义：
               `_load_ip_adapter_weights()` 里 `self.encoder_hid_proj = None`
               然后重建，所以"调 N 次不同 adapter_name"在 SD1.5 这条路上
               并不存在 adapter_name 参数，也达不到 N 个适配器的效果。）

        2) **`set_ip_adapter_scale` 收 per-adapter 列表**：
           `diffusers/loaders/ip_adapter.py:516-550`（IPAdapterMixin，SD1.5 走这条）
               def set_ip_adapter_scale(self, scale):
                   ...
                   if not isinstance(scale, list):
                       scale = [scale]
                   scale_configs = _maybe_expand_lora_scales(unet, scale, default_scale=0.0)
                   for attn_name, attn_processor in unet.attn_processors.items():
                       if isinstance(attn_processor, (IPAdapterAttnProcessor,
                                   IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor)):
                           if len(scale_configs) != len(attn_processor.scale):
                               raise ValueError(...)
                           elif len(scale_configs) == 1:
                               scale_configs = scale_configs * len(attn_processor.scale)
                           for i, scale_config in enumerate(scale_configs): ...
                   → 结论：`pipe.set_ip_adapter_scale([0.8, 0.4])` = 第 1 个适配器
                     0.8、第 2 个 0.4。**已实测通过**（见 scratch 记录）。

        3) **`ip_adapter_image` 收列表**：
           `diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py:538-559`
               def prepare_ip_adapter_image_embeds(self, ip_adapter_image, ...):
                   if not isinstance(ip_adapter_image, list):
                       ip_adapter_image = [ip_adapter_image]
                   if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                       raise ValueError(
                           f"`ip_adapter_image` must have same length as the number of IP Adapters. ...")
                   for single_ip_adapter_image, image_proj_layer in zip(...):
                       single_image_embeds, single_negative_image_embeds = self.encode_image(...)
           → 结论：**支持**，长度必须等于适配器数量。**已实测**：传 1 张给
             2 个适配器时确实抛
             ``ValueError: `ip_adapter_image` must have same length as the number of IP Adapters. Got 1 images and 2 IP Adapters.``

        4) **每条参考图独立区域**：公开的 `__call__` 没有这个入参，但注意力
           处理器有 `ip_adapter_masks`：
           `diffusers/models/attention_processor.py:4459`
               ip_adapter_masks: torch.Tensor | None = None,
           `:4528-4557` 校验"长度 == scale 长度 == ip_hidden_states 长度"、
           每个元素 shape `[1, num_images_for_ip_adapter, height, width]`；
           `models/attention.py:1305-1327` 的 BasicTransformerBlock 会把
           `cross_attention_kwargs` 原样展开进 attn2，所以可以从**管线公开
           入参** `cross_attention_kwargs={"ip_adapter_masks": [...]}` 传进去。
           → 结论：per-reference 区域**真的可实现**，且不需要改 diffusers。
             见 `core/ipa_multi.build_ip_adapter_masks`（本机已实测非零影响）。

        参数
        ----
        refs : list[IpaReference | dict]
            参考图列表。角色→档位的降级、权重夹取都在这里完成。

        返回
        ----
        `core.ipa_multi.MultiAdapterPlan`（**不抛异常**：失败写进 .error，
        这样节点可以决定是"整体失败"还是"降级继续"）。
        """
        from core import ipa_multi as ipa_mod

        plan = ipa_mod.MultiAdapterPlan()
        try:
            items = ipa_mod.coerce_references(refs)
        except Exception as e:                      # pragma: no cover - 防御
            plan.error = f"参考图列表无法解析: {e}"
            return plan
        if not items:
            plan.error = "没有可用的参考图（列表为空）"
            return plan

        primary = getattr(self, "txt2img_pipe", None)
        if primary is None:
            plan.error = "尚未加载底模，无法安装 IP-Adapter"
            return plan

        try:
            installed = self.installed_ipa_variants()
        except Exception:
            installed = []

        # ── 1) 角色 → 实际档位（缺文件就降级 + 中文警告，一个都没有才抛）──
        try:
            resolved, warns = ipa_mod.resolve_role_plan(items, installed)
        except Exception as e:
            plan.error = str(e)
            return plan
        plan.warnings.extend(warns)
        plan.refs = resolved

        # ── 2) 逐个解析成 (source, subfolder, weight_name) ──
        # ⚠️ 这里可能出现第二次降级：`resolve_role_plan` 只在"角色档位没装"时
        # 降级到 installed[0]；如果连 `installed[0]` 也解析不出来（例如用户
        # 把文件删了但目录还在），`resolve_ipa_weights` 会再降级一次并打警告。
        # 我们**不在这里编造**"实际用了哪个档位" —— 直接按返回的**文件名**
        # 反查档位，避免元数据说谎。
        variants, sources, subs, names, scales = [], [], [], [], []
        for r in resolved:
            try:
                src, sub, name = self.resolve_ipa_weights(r.variant)
            except Exception as e:
                plan.error = str(e)
                return plan
            got = next(
                (var for (xl, var), (_sub, nm) in self._IPA_WEIGHTS.items()
                 if xl == bool(getattr(self, "is_sdxl", False)) and nm == name),
                r.variant)
            if got != r.variant:
                msg = (f"参考图（{ipa_mod.role_label(r.role)}）实际使用档位"
                       f"「{got}」（请求 {r.variant}）。")
                plan.warnings.append(msg)
                r.variant = got
            variants.append(got)
            sources.append(src)
            subs.append(sub)
            names.append(name)
            scales.append(float(r.weight))

        # ── 3) 安装（共享 UNet 只装一次，N 个权重一起传）──
        encoder_dir = self._ipa_local_image_encoder() or ""

        # 已经装过**完全相同**的适配器组合 → 跳过重复安装。
        # 为什么值得做：`load_ip_adapter()` 每次都要重新读 safetensors 并重建
        # encoder_hid_proj，实测约 1.1s；而"同一张工作流跑第二次"
        # （改提示词、改种子、批量出图）会命中这条路径。
        # ⚠️ 只有 (档位, 文件名, 来源, 子目录) 全都一样才能跳过 ——
        # 只比档位会漏掉"同一个 base 装两次"这种多适配器场景。
        # 只比 (档位, 文件名, 来源, 子目录) 不够：用户把同名文件换掉
        # （重新下载一版、覆盖成别的适配器）时这几项全不变，签名不变，
        # 于是会继续用内存里那份**旧权重**。加上文件体积 + mtime，
        # 换文件就能被发现，而不用每次哈希几百 MB 的权重。
        stamps = [self._ipa_file_stamp(s) for s in sources]
        sig = tuple(zip(variants, names, sources, subs, stamps))
        if (getattr(self, "_ipa_adapters", None)
                and getattr(self, "_ipa_signature", None) == sig
                and getattr(self, "ip_adapter_loaded", False)
                and getattr(self, "ipa_adapter_count", lambda: 0)() == len(names)):
            logger.info(f"⚡ 适配器组合未变（{'+'.join(names)} → "
                        f"{len(names)} 个），跳过重复安装")
            self._set_multi_scale(scales)
            self._ipa_scales = list(scales)
            self.current_ipa_scale = max(scales) if scales else 0.7
            self._ipa_scale = self.current_ipa_scale
            self._mark_ipa_state(loaded=True, scale=self.current_ipa_scale)
            plan.ok, plan.count = True, len(names)
            plan.variants, plan.scales = variants, scales
            plan.sources, plan.weight_names, plan.subfolders = sources, names, subs
            plan.encoder_dir = encoder_dir
            return plan

        try:
            self._ipa_prepare_pipes()
            logger.info(f"🎭 在共享 UNet 上一次性安装 {len(names)} 个 IP-Adapter: "
                        f"{', '.join(names)}")
            primary.load_ip_adapter(
                sources, subfolder=subs, weight_name=names,
                **self._ipa_encoder_kwargs())
            self._ipa_sync_pipe_components(primary)
        except Exception as e:
            logger.error(f"❌ 多适配器安装失败: {type(e).__name__}: {e}")
            plan.error = (f"{type(e).__name__}: {e}")
            try:
                self.unload_ip_adapter()
            except Exception:
                pass
            return plan

        # ── 4) 每适配器独立强度 ──
        self._set_multi_scale(scales)

        # ── 5) 状态 ──
        self._ipa_adapters = list(variants)
        self._ipa_scales = list(scales)
        self._ipa_signature = sig
        self.ip_adapter_loaded = True
        self.ip_adapter_variant = "+".join(variants)
        self._ipa_loaded = True
        self._ipa_variant = self.ip_adapter_variant
        self._ipa_weight_name = "+".join(names)
        self._ipa_source = sources[0] if sources else ""
        self.current_ipa_scale = max(scales) if scales else 0.7
        self._ipa_scale = self.current_ipa_scale
        self._mark_ipa_state(loaded=True, scale=self.current_ipa_scale)

        plan.ok = True
        plan.count = len(names)
        plan.variants = variants
        plan.scales = scales
        plan.sources = sources
        plan.weight_names = names
        plan.subfolders = subs
        plan.encoder_dir = encoder_dir
        plan.warnings = list(plan.warnings)

        if any((r.mask is not None or r.mask_path) for r in resolved):
            plan.warnings.append(
                "有参考图带独立区域掩码：调用采样时要带 "
                "cross_attention_kwargs={'ip_adapter_masks': [...]}"
                "（见 core.ipa_multi.build_ip_adapter_masks）")

        logger.info(f"✅ 多适配器就绪：{plan.summary()}"
                    f"（每适配器独立强度 {scales}）")
        return plan

    def _set_multi_scale(self, scales, pipes=None):
        """把 per-adapter scale 列表设到管线（失败只警告，不抛）。

        ⚠️ 只在装了 IPA 的 UNet 上设：scale 是写在 attn processor 上的，
        没装 IPA 的管线去设会抛 `AttributeError`，被 except 吞掉后
        看起来"设过了"，其实什么都没发生（老代码就有这个毛病）。
        """
        scales = [float(s) for s in (scales or [])]
        if not scales:
            return
        payload = scales[0] if len(scales) == 1 else scales
        targets = pipes if pipes is not None else (
            self.txt2img_pipe, self.img2img_pipe,
            self.inpaint_pipe, self.controlnet_pipe)
        for p in targets:
            if p is None:
                continue
            setter = getattr(p, "set_ip_adapter_scale", None)
            if not callable(setter):
                continue
            try:
                setter(payload)
                p._ai_ipa_loaded = True
                p._ai_ipa_scale = max(scales)
                # 每适配器的强度都留一份：ADetailer / Pose Transfer 那种
                # "临时把 IPA 关掉再恢复"的场景必须能原样还回去
                # （只知道最大值是不够的）。
                p._ai_ipa_scales = list(scales)
            except Exception as e:
                logger.warning(f"  ⚠️ set_ip_adapter_scale({payload}) 失败: {e}")

    def ipa_adapter_count(self, pipe=None):
        """当前共享 UNet 上装了几个 IP-Adapter（没装返回 0）。

        为什么要问这个：diffusers 的 `prepare_ip_adapter_image_embeds` 会
        **严格校验** `len(ip_adapter_image) == 适配器数量`，否则抛
        `ValueError: ip_adapter_image must have same length as the number of IP Adapters`。
        局部重绘（区域）想给某个区域单独换参考图时，必须按这个数量补齐占位图。
        """
        p = pipe or getattr(self, "txt2img_pipe", None)
        for cand in (p, getattr(self, "inpaint_pipe", None),
                     getattr(self, "img2img_pipe", None)):
            unet = getattr(cand, "unet", None) if cand is not None else None
            proj = getattr(unet, "encoder_hid_proj", None)
            n = getattr(proj, "num_ip_adapters", None)
            try:
                if n:
                    return int(n)
            except Exception:
                continue
            layers = getattr(proj, "image_projection_layers", None)
            if layers is not None:
                try:
                    return len(layers)
                except Exception:
                    continue
        return 0

    def build_ip_adapter_kwargs(self, refs=None, *, width: int = 0,
                               height: int = 0, latent_scale: int = 8,
                               has_masks: bool = False):
        """构造**一次采样调用**要用的 IP-Adapter 入参（供区域重绘等调用方用）。

        与 `prepare_ip_adapter_multi` 的分工：
          · `prepare_ip_adapter_multi` 负责"装"（N 个权重一次性上 UNet）；
          · 本方法负责"调"（这一轮给哪些参考图、各什么强度、哪些区域掩码）。

        规则（都是为了让 diffusers 不报错）：
          1. 适配器数量 N = `ipa_adapter_count()`；
          2. 参考图数量 != N 时**补齐/截断**到 N（用 1x1 黑图占位），
             否则 diffusers 会抛长度不匹配的 ValueError；
          3. scale 按每条参考图的 weight 给（占位图权重 0 → 该适配器跳过）；
          4. `has_masks=True` 时把每条的掩码转成注意力掩码张量放进
             `cross_attention_kwargs["ip_adapter_masks"]`。

        返回 `(kwargs_dict, info_dict)`；没有装 IPA 时返回 `({}, {})`。
        """
        from core import ipa_multi as ipa_mod

        n = self.ipa_adapter_count()
        if n <= 0:
            return {}, {}

        # ⚠️ 即使 refs 为空也要返回占位图列表：只要 UNet 上装了 IPA，
        # **每次采样调用**都必须给 ip_adapter_image（长度 == 适配器数量），
        # 否则 diffusers 会抛 `ValueError: ip_adapter_image must have same
        # length as the number of IP Adapters`（本机已实测）。
        items = ipa_mod.coerce_references(refs) if refs else []

        from PIL import Image as _PIL
        images, scales, notes = [], [], []
        mask_w = mask_h = 0
        for i in range(n):
            if i < len(items):
                r = items[i]
                if not getattr(r, "enabled", True) or float(r.weight) <= 0.0:
                    images.append(_PIL.new("RGB", (1, 1), (0, 0, 0)))
                    scales.append(0.0)
                    continue
                img = ipa_mod.load_reference_image(r)
                if img is not None and (r.mask is not None or r.mask_path):
                    img = ipa_mod.apply_reference_mask(r, img)
                if img is None:
                    img = _PIL.new("RGB", (1, 1), (0, 0, 0))
                    notes.append(f"参考图 {i + 1} 读不到，已用占位图（强度 0）")
                    scales.append(0.0)
                else:
                    scales.append(float(r.weight))
                    if mask_w == 0 and mask_h == 0:
                        mask_w, mask_h = img.width, img.height
                images.append(img)
            else:
                images.append(_PIL.new("RGB", (1, 1), (0, 0, 0)))
                scales.append(0.0)
                if items:
                    notes.append("补充占位图以对齐适配器数量")
                else:
                    notes.append("本区域无参考图，用占位图（强度 0）")

        kwargs = {"ip_adapter_image": images}
        if has_masks and items:
            w = int(width or mask_w)
            h = int(height or mask_h)
            masks = ipa_mod.build_ip_adapter_masks(
                items[:n], w, h, scale=latent_scale, device="cpu")
            if masks:
                kwargs["cross_attention_kwargs"] = {"ip_adapter_masks": masks}
        info = {"scales": scales, "notes": notes, "adapters": n}
        return kwargs, info
        scales = [float(s) for s in (scales or [])]
        if not scales:
            return
        payload = scales[0] if len(scales) == 1 else scales
        for attr in ("txt2img_pipe", "img2img_pipe",
                     "inpaint_pipe", "controlnet_pipe"):
            p = getattr(self, attr, None)
            if p is None:
                continue
            setter = getattr(p, "set_ip_adapter_scale", None)
            if not callable(setter):
                continue
            try:
                setter(payload)
                p._ai_ipa_loaded = True
                p._ai_ipa_scale = max(scales)
            except Exception as e:
                logger.warning(f"  ⚠️ [{attr}] set_ip_adapter_scale({payload}) 失败: {e}")

    def prepare_ip_adapter(self, variant="plus"):
        """安装 IP-Adapter（共享 UNet 只安装一次，架构感知选权重）。"""
        if getattr(self, 'ip_adapter_loaded', False) \
           and getattr(self, 'ip_adapter_variant', None) == variant:
            logger.info("✅ IP-Adapter 已加载,复用")
            return True

        logger.info(f"🎭 加载 IP-Adapter (variant={variant}, "
                    f"SDXL={bool(getattr(self, 'is_sdxl', False))}) ...")

        # 装载前备份**基础组件**（尤其 image_processor）。
        # 这样即使装载失败、走到 unload_ip_adapter()，也能把它恢复回来，
        # 不会把整个会话的出图能力弄坏。
        if not getattr(self, "_ipa_saved_components", None):
            saved = {}
            for attr in ("txt2img_pipe", "img2img_pipe", "inpaint_pipe",
                         "controlnet_pipe"):
                p = getattr(self, attr, None)
                if p is None:
                    continue
                for comp in ("image_processor",):
                    v = getattr(p, comp, None)
                    if v is not None and comp not in saved:
                        saved[comp] = v
            self._ipa_saved_components = saved
            if saved:
                logger.debug(f"  💾 已备份基础组件: {list(saved)}")

        try:
            ok, src, subfolder, weight_name = self._install_ipa_once(variant)

            # 🆕 inpaint_pipe：**同共享 UNet 无法摘除 IPA**，
            # 但 ADetailer 调用层会用 dummy 图 + scale=0 规避（见 image_processor）。
            inpaint_pipe = getattr(self, 'inpaint_pipe', None)
            primary = getattr(self, 'txt2img_pipe', None)
            if inpaint_pipe is not None and primary is not None:
                if getattr(inpaint_pipe, 'unet', None) is getattr(primary, 'unet', None):
                    logger.info("  ℹ️ inpaint_pipe 与主管线共享 UNet："
                                "IPA 结构天然可见，ADetailer 用 scale=0 + 占位图规避")
                else:
                    try:
                        from diffusers.models.attention_processor import (
                            AttnProcessor2_0, AttnProcessor)
                        import torch.nn.functional as F
                        proc = (AttnProcessor2_0
                                if hasattr(F, "scaled_dot_product_attention")
                                else AttnProcessor)
                        inpaint_pipe.unet.set_attn_processor(proc())
                        logger.info("  ✅ inpaint_pipe 独立 UNet 已复位为标准 processor")
                    except Exception as e:
                        logger.warning(f"  ⚠️ inpaint_pipe 复位失败: {e}")

            # 🆕 设置所有状态标志(统一两套变量名)
            self.ip_adapter_loaded  = True
            self.ip_adapter_variant = variant
            self._ipa_loaded        = True
            self._ipa_variant       = variant
            self._ipa_scale         = 0.7
            self._ipa_weight_name   = weight_name
            self._ipa_source        = src
            # 单档位路径：清掉多参考图状态，避免残留的 N 个 scale 被误用
            self._ipa_adapters      = [variant]
            self._ipa_scales        = [float(getattr(self, "current_ipa_scale", 0.7))]

            self.current_ipa_scale = float(
                getattr(self, "current_ipa_scale", 0.7))
            # ⚠️ 同步到 _ipa_scale，否则 ADetailer 读到的永远是初始 0.7
            self._ipa_scale = self.current_ipa_scale

            self._mark_ipa_state(
                loaded=True,
                scale=self.current_ipa_scale,
            )

            logger.info(f"✅ IP-Adapter 加载完成 "
                        f"({subfolder}/{weight_name}，共享 UNet 仅安装一次)")

            _resolver = identity_resolver()
            if _resolver.on_before_load is None:
                _resolver.on_before_load = self._free_gpu_cache

            # controlnet_pipe 若已存在，同步管线级组件（不重复安装权重）
            if getattr(self, 'controlnet_pipe', None) is not None:
                try:
                    self._sync_ipa_components_to_controlnet()
                except Exception as e:
                    logger.warning(f"⚠️ 同步 IPA 到 controlnet 失败: {e}")

            return True

        except Exception as e:
            logger.error(f"❌ IP-Adapter 加载失败: {e}")
            import traceback
            traceback.print_exc()

            try:
                self.unload_ip_adapter()
            except Exception as e2:
                logger.warning(f"⚠️ 清理失败: {e2}")

            self.ip_adapter_loaded  = False
            self.ip_adapter_variant = None
            self._ipa_loaded        = False
            self._ipa_variant       = None
            self._ipa_adapters      = []
            self._ipa_scales        = []
            self._ipa_signature     = None
            return False


    def unload_ip_adapter(self):
        """卸载 IP-Adapter，恢复 UNet 到普通模式。

        ⚠️ 旧的"遍历四条管线各调一次 unload_ip_adapter()"是错的：
        四条管线**共享同一个 UNet**，每次调用都会对同一 UNet 重写
        attn processor / encoder_hid_proj。重复卸载不仅浪费，
        还可能在中途把别的管线正在用的结构改坏。

        正确顺序：
          1. 先按 **UNet 去重**，共享的只卸一次；
          2. 再清理各管线持有的**管线级** IPA 组件与状态。
        """
        logger.info("🧹 卸载 IP-Adapter ...")

        pipes = [p for p in (self.txt2img_pipe, self.img2img_pipe,
                             self.inpaint_pipe, self.controlnet_pipe)
                 if p is not None]

        # ── 1. 按 UNet 去重，只卸一次 ──
        seen_unets = set()
        unloaded = 0
        for pipe in pipes:
            unet = getattr(pipe, "unet", None)
            unet_id = id(unet) if unet is not None else None
            if unet_id is not None and unet_id in seen_unets:
                logger.debug("  → 共享 UNet 已卸载过，跳过该管线")
                continue
            if unet_id is not None:
                seen_unets.add(unet_id)
            try:
                if hasattr(pipe, "unload_ip_adapter"):
                    pipe.unload_ip_adapter()
                else:
                    from diffusers.models.attention_processor import (
                        AttnProcessor2_0, AttnProcessor)
                    import torch.nn.functional as F
                    proc = (AttnProcessor2_0
                            if hasattr(F, "scaled_dot_product_attention")
                            else AttnProcessor)
                    if unet is not None:
                        unet.set_attn_processor(proc())
                        unet.encoder_hid_proj = None
                        unet.config.encoder_hid_dim_type = None
                unloaded += 1
            except Exception as e:
                logger.warning(f"  ⚠️ 卸载异常: {e}")

        # 兜底：若没有任何管线可卸但 UNet 上仍有 IPA 痕迹，手动清理一次
        if unloaded == 0:
            for pipe in pipes:
                unet = getattr(pipe, "unet", None)
                if unet is not None and getattr(unet, "encoder_hid_proj", None) is not None:
                    try:
                        from diffusers.models.attention_processor import (
                            AttnProcessor2_0, AttnProcessor)
                        import torch.nn.functional as F
                        proc = (AttnProcessor2_0
                                if hasattr(F, "scaled_dot_product_attention")
                                else AttnProcessor)
                        unet.set_attn_processor(proc())
                        unet.encoder_hid_proj = None
                        unet.config.encoder_hid_dim_type = None
                        unloaded += 1
                    except Exception as e:
                        logger.warning(f"  ⚠️ 兜底清理失败: {e}")
                    break

        # ── 2. 清理管线级组件与所有状态标记 ──
        #
        # ⚠️ 绝对不要把 `image_processor` 清成 None！
        # 它是**基础管线的核心组件**（VaeImageProcessor，每次出图都要用
        # `image_processor.postprocess` 把 latent 转成 PIL），不是 IP-Adapter 的组件。
        # 旧代码把它和 image_encoder / feature_extractor 一起置 None，后果是：
        #   · 任何一次 IPA 装载失败（例如选了本机没装的 plus 档位）都会走到
        #     `unload_ip_adapter()`，
        #   · 于是 `txt2img_pipe.image_processor = None`，
        #   · 之后**整个会话所有出图全部失败**，报的还是与 IPA 毫无关系的
        #     `AttributeError: 'NoneType' object has no attribute 'postprocess'`。
        # 实测：装载 plus 失败后，连不带 IPA 的普通文生图也立刻失败。
        # 正确做法：只清 IPA 自己的组件；image_processor 恢复成装载前的对象。
        for pipe in pipes:
            for comp in ("image_encoder", "feature_extractor"):
                try:
                    if hasattr(pipe, comp):
                        setattr(pipe, comp, None)
                except Exception:
                    pass
            # 恢复基础组件的备份（若装载时被换掉过）
            try:
                for comp, saved in (getattr(self, "_ipa_saved_components", None)
                                    or {}).items():
                    if saved is not None and getattr(pipe, comp, None) is None:
                        setattr(pipe, comp, saved)
            except Exception:
                pass
            for flag in ("_ai_ipa_loaded", "_ai_ipa_scale"):
                try:
                    if hasattr(pipe, flag):
                        delattr(pipe, flag)
                except Exception:
                    pass

        self.ip_adapter_loaded  = False
        self.ip_adapter_variant = None
        self._ipa_loaded        = False
        self._ipa_variant       = None
        self._ipa_weight_name   = None
        self._ipa_source        = None
        self._ipa_adapters      = []
        self._ipa_scales        = []
        self._ipa_signature     = None
        logger.info(f"✅ IP-Adapter 卸载完成（卸载了 {unloaded} 个独立 UNet）")

        # 🆕 同步清理身份后端（缓存 / 已加载权重）
        try:
            identity_resolver().release()
        except Exception:
            pass

    def _free_gpu_cache(self):
        """身份后端加载前腾出设备缓存（无副作用，安全可重复调用）。"""
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass

    @staticmethod
    def _is_offloaded(module):
        """
        该模块是否已被 accelerate 接管（model/sequential CPU offload）。
        被接管后不能再手动 .to()：
          - sequential 下权重是 meta 占位张量，.to() 直接抛 NotImplementedError
          - model_cpu_offload 下 .to() 会绕过 hook 的权重簿记，
            下一次 forward 时 hook 又把它搬回去，白搬一趟还污染状态
        """
        return getattr(module, "_hf_hook", None) is not None

    # ------------------------------------------------------------
    #  提示词编码 (Compel,支持 SD1.5 / SDXL)
    # ------------------------------------------------------------
    def set_ip_adapter_scale(self, scale: float = 0.6):
        """调整 IP-Adapter 影响强度 0.0~1.5（覆盖全部 4 个 pipeline）"""
        if not self.ip_adapter_loaded:
            return
        scale = max(0.0, min(1.5, float(scale)))
        self._ipa_scale = float(scale)
        targets = [
            self.txt2img_pipe, self.img2img_pipe,
            self.inpaint_pipe, self.controlnet_pipe,
        ]
        for pipe in targets:
            if pipe is None:
                continue
            try:
                if hasattr(pipe, 'set_ip_adapter_scale'):
                    pipe.set_ip_adapter_scale(scale)
            except Exception:
                pass
        logger.info(f"🎛️ IP-Adapter scale = {scale}")

    # ------------------------------------------------------------
    #  提示词编码 (Compel,支持 SD1.5 / SDXL)
    # ------------------------------------------------------------
    def encode_prompt(self, prompt, negative_prompt, pipe=None):
        prof = getattr(self, 'arch_profile', None)
        if getattr(prof, 'prompt_mode', None) != "compel":
            # ModularPipeline 系（Anima/Cosmos）等非 Compel 架构：
            # 不做 Compel，直接退原始 prompt，交给 _adapt_call 转 PipelineState
            return {"prompt": prompt, "negative_prompt": negative_prompt}
        if not self.txt2img_pipe:
            return {}

        if pipe is None:
            pipe = self.txt2img_pipe

        # 仅在全 GPU 模式下手动搬 TE；offload 模式交给 accelerate 的 hook
        if hasattr(pipe, "text_encoder") and pipe.text_encoder:
            if not self._is_offloaded(pipe.text_encoder):
                pipe.text_encoder.to(self.device)
        if (self.is_sdxl
                and hasattr(pipe, "text_encoder_2")
                and pipe.text_encoder_2):
            if not self._is_offloaded(pipe.text_encoder_2):
                pipe.text_encoder_2.to(self.device)

        # ── 缓存键设计 ──
        # ⚠️ 旧键含 `current_model_name`：换任何底模都会让整个缓存失效，
        # 即使新底模用的是**同一个 CLIP 文本编码器**（SD1.5 系各 checkpoint
        # 几乎都共用同一套 CLIP-L），批量出图跨模型时会白算很多次。
        # 决定 embedding 的是「编码器实例 + 文本」，所以按编码器身份建键。
        #
        # ⚠️ 只缓存 **pad 之前**的原始 embeds：pad_conditioning_tensors_to_same_length
        # 会把正向按负向长度补齐，若把补齐后的张量存进缓存，下次换一个
        # 不同长度的负向词就会形状不匹配。
        te = getattr(pipe, "text_encoder", None)
        te2 = getattr(pipe, "text_encoder_2", None)
        cache_key = (id(te), id(te2), prompt, negative_prompt or "")

        # Compel 实例先构造（缓存命中路径也要用它做 pad）
        if self.is_sdxl:
            compel = Compel(
                tokenizer    = [pipe.tokenizer, pipe.tokenizer_2],
                text_encoder = [pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=
                    ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                device=self.device,
                truncate_long_prompts = False,
            )
        else:
            compel = Compel(
                tokenizer    = pipe.tokenizer,
                text_encoder = pipe.text_encoder,
                device       = self.device,
                truncate_long_prompts = False,
            )

        raw = self._compel_cache.get(cache_key)
        if raw is not None:
            logger.debug(f"♻️ Compel 缓存命中（enc={id(te)}）")
        else:
            raw = self._compute_compel_embeds(compel, prompt, negative_prompt,
                                              self.is_sdxl)
            self._compel_cache_put(cache_key, raw)

        # 统一在这里做 pad 后返回（缓存里存的始终是未 pad 的原始张量）
        result = dict(raw)
        try:
            pe, ne = compel.pad_conditioning_tensors_to_same_length(
                [result["prompt_embeds"], result["negative_prompt_embeds"]])
            result["prompt_embeds"] = pe
            result["negative_prompt_embeds"] = ne
        except Exception as e:
            logger.debug(f"[compel] pad 跳过: {e}")
        return result

    def _compute_compel_embeds(self, compel, prompt, negative_prompt, is_sdxl):
        """计算**未 pad** 的原始 embeds（供缓存复用）。"""
        if is_sdxl:
            prompt_embeds, pooled = compel(prompt)
            neg_embeds, neg_pooled = compel(negative_prompt)
            return {
                "prompt_embeds":                 prompt_embeds,
                "pooled_prompt_embeds":          pooled,
                "negative_prompt_embeds":        neg_embeds,
                "negative_pooled_prompt_embeds": neg_pooled,
            }
        return {
            "prompt_embeds":          compel(prompt),
            "negative_prompt_embeds": compel(negative_prompt),
        }

    def _compel_cache_put(self, key, value):
        """写入 Compel 缓存（LRU，上限 24）。"""
        try:
            self._compel_cache[key] = value
            while len(self._compel_cache) > 24:
                oldest = next(iter(self._compel_cache))
                del self._compel_cache[oldest]
        except Exception:
            logger.debug("Compel 缓存写入跳过", exc_info=True)

    # ------------------------------------------------------------
    #  采样器切换
    # ------------------------------------------------------------
    def switch_sampler(self, sampler_name, sigma=None):
        """根据 UI 传来的名称（+ sigma 风格），切换底层的调度器。

        真正的"名字 → 调度器类"映射在 `core/samplers.py`：这里以前只有
        四个 if 分支，其它名字**静默落到 Euler a**（界面选了 UniPC / LCM /
        PNDM 也一样），现在统一交给那张表，未知名字会明确警告而不是装傻。
        """
        if self._is_flow_matching_pipe():
            logger.info(
                f"⏭️ 跳过采样器切换（flow-matching 底模保持原生调度器: "
                f"{type(self.txt2img_pipe.scheduler).__name__}）"
            )
            return
        # 能力表里写明不支持换调度器（如 MOSS-Image）时也别动它
        try:
            caps = self.capabilities
            if caps is not None and not caps.supports("scheduler_switch"):
                logger.info("⏭️ 跳过采样器切换（%s 不支持替换调度器）",
                            getattr(self, "current_arch_id", "?"))
                return
        except Exception:
            logger.debug("能力表检查失败，继续按流程切换采样器", exc_info=True)
        if (not hasattr(self, 'txt2img_pipe')
                or self.txt2img_pipe is None):
            return

        # 基线 config 只取**一次**（第一次切换时）。
        #
        # 直接从 `self.txt2img_pipe.scheduler.config` 取是错的：上一步刚切到
        # "UniPC + Karras" 之后，config 里就带着 use_karras_sigmas=True，
        # 再切到 "DPM++ 2M"（不带 Karras）时会被这个残留值污染 —— 界面说
        # 默认，实际还在 Karras。换底模时由 load_model 清空重取。
        base_cfg = getattr(self, "_scheduler_base_cfg", None)
        if base_cfg is None:
            base_cfg = dict(self.txt2img_pipe.scheduler.config)
            self._scheduler_base_cfg = base_cfg

        # ── 预测类型 / zero-SNR 必须用**显式 kwargs** 传入 ──
        # ⚠️ 实测（diffusers 0.40）：把值先写进 config dict 再 from_config(cfg)
        # 是**无效**的 —— config 里自带的 prediction_type='epsilon' 会覆盖你写的值
        # （EulerDiscreteScheduler 的 _compatibles 会显式传入 prediction_type）。
        # 只有 from_config(cfg, prediction_type=...) 这种显式 kwargs 才能生效。
        overrides = {}

        # v-pred 模型必须用 v_prediction，否则输出直接是废图
        if getattr(self, 'is_vpred', False):
            overrides["prediction_type"] = "v_prediction"

        # zero-SNR 单独配置，不再与 v-pred 绑定：
        # v-pred ≠ zero-SNR，很多 v-pred 模型并非 zero-SNR 训练，强开会掉质量。
        # None = 跟随模型自带配置。
        _zsnr = getattr(self, "zero_snr", None)
        if _zsnr is not None:
            overrides["rescale_betas_zero_snr"] = bool(_zsnr)

        # sigma 风格（默认 / Karras / Exponential / Beta）：名字里带
        # "Karras" 的老写法也认，见 core/samplers.resolve()。
        # 只认显式传入的 sigma，不传就是"默认"。
        # 以前这里回落到 self._sigma_style：先切「UniPC + Karras」，再按名字
        # 切回「DPM++ 2M」，Karras 会被一起带过去 —— 界面显示默认、实际还在
        # Karras。同一采样器重复 switch（zero_snr 重放）会把风格显式传回来，
        # 所以回落没有存在必要。
        style = sigma

        try:
            from core import samplers as _samplers

            def _build():
                built = _samplers.build_scheduler(
                    base_cfg, sampler_name, style, **overrides)
                if built is None:
                    raise ValueError(
                        f"未知采样器「{sampler_name}」，可选："
                        f"{'、'.join(_samplers.sampler_names())}")
                return built

            # 每条管线各建一个实例：
            # scheduler 内部会记录 timesteps / sigmas 等推理期状态，
            # 多条管线共享同一实例时，并发或交替调用会互相污染状态。
            assigned = 0
            for attr in ("txt2img_pipe", "img2img_pipe",
                         "inpaint_pipe", "controlnet_pipe",
                         "reference_pipe"):
                p = getattr(self, attr, None)
                if p is None:
                    continue
                try:
                    p.scheduler = _build()
                    assigned += 1
                except Exception as e:
                    logger.warning(f"  ⚠️ {attr} 切换采样器失败: {e}")

            self._sampler_name = sampler_name
            self._sigma_style = style or "默认"
            _pt = overrides.get("prediction_type", "模型自带")
            try:
                _desc = _samplers.describe(sampler_name, style)
            except Exception:                     # pragma: no cover - 防御
                _desc = sampler_name
            logger.info(f"🔄 采样器已切换为 -> {_desc}"
                        f"（{assigned} 条管线各持独立实例，"
                        f"prediction_type={_pt}）")

        except Exception as e:
            logger.warning(f"⚠️ 切换采样器失败: {e}，将使用原默认采样器")

    def set_zero_snr(self, enabled):
        """显式开关 zero-SNR（与 v-pred 解耦）。

        None = 跟随模型自带配置；True/False = 强制。
        """
        self.zero_snr = None if enabled is None else bool(enabled)
        logger.info(f"🎚️ zero-SNR = {self.zero_snr}")
        # 立即按当前采样器重建一遍，让设置生效
        name = getattr(self, "_sampler_name", None)
        if name:
            self.switch_sampler(name, getattr(self, "_sigma_style", None))

    def _normalize_device(self, raw) -> str:
        import re
        import torch
        s = str(raw or "").lower()
        m = re.search(r'\(([^)]+)\)', s)
        if m:
            s = m.group(1).strip()
        s = s.strip()

        if "cuda" in s or "gpu" in s or "显卡" in s:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if "mps" in s:
            try:
                if torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        if "cpu" in s:
            return "cpu"
        # auto / 自动 / 空 / 未知
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def unload_all(self):
        """彻底释放所有管线和内存"""
        logger.info("🧹 开始释放所有模型...")

        # 释放检测器（走 __dict__.pop，属性会被彻底移除）
        for attr in ['pose_detector', 'depth_estimator']:
            self.__dict__.pop(attr, None)

        # 释放 IP-Adapter 参考图
        if hasattr(self, 'ipa_ref_image'):
            self.ipa_ref_image = None

        # 管线 / ControlNet / 缓存 / 架构状态统一交给 clear_memory()：
        # 以前两套卸载路径各清一半，导致 current_arch_id / is_sdxl 等残留，
        # 卸载后 LoRA 目录与架构判断错位。
        self.clear_memory()

        # 强制 GC + 工作集回收（clear_memory 里已做一次，这里再收一遍：
        # 上面刚 pop 掉检测器，它们的显存/内存要等这次 gc 才真正归还）。
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        try:
            from utils import mem as _mem

            _delta = _mem.release_memory()
            _line = _mem.describe(_delta)
            if _line:
                logger.info("✅ 所有模型已释放：%s", _line)
        except Exception:  # noqa: BLE001
            logger.info("✅ 所有模型已释放")

    def sync_ipa_to_controlnet(self):
        """让 controlnet_pipe 具备 IPA 能力。

        ⚠️ 不再调用 load_ip_adapter()。已对 diffusers 0.40 源码核实：
        `load_ip_adapter()` 会改写 `self.unet`（set_attn_processor +
        encoder_hid_proj），而 controlnet_pipe 与 txt2img_pipe **共享同一个
        UNet** —— 再装一次就是对同一 UNet 重复安装，会覆盖已有状态
        （典型症状：device mismatch / attn processor 结构错乱）。

        正确做法：UNet 侧已经装好了（共享可见），这里只补齐**管线级**组件
        （image_encoder / feature_extractor / image_processor）并设置 scale，
        然后校验共享 UNet 上确实挂着 IPA processor。
        """
        if self.controlnet_pipe is None:
            logger.warning("⚠️ controlnet_pipe 未初始化")
            return False
        if self.txt2img_pipe is None or not getattr(self, '_ipa_loaded', False):
            logger.warning("⚠️ txt2img_pipe 未加载 IPA，无法同步")
            return False

        try:
            # ── 1. 关闭 attention slicing（IPA 不兼容）──
            try:
                self.controlnet_pipe.disable_attention_slicing()
            except Exception:
                pass

            src = self.txt2img_pipe
            dst = self.controlnet_pipe
            shared = getattr(src, "unet", None) is getattr(dst, "unet", None)

            # ── 2. 同步管线级组件（省显存，不重复装权重）──
            copied = []
            for comp in ("image_encoder", "feature_extractor", "image_processor"):
                val = getattr(src, comp, None)
                if val is not None and getattr(dst, comp, None) is None:
                    setattr(dst, comp, val)
                    copied.append(comp)
            if copied:
                logger.info(f"  → 共享组件到 controlnet_pipe: {', '.join(copied)}")

            # ── 3. 设置 scale（作用在共享 UNet 的 IPA 投影上）──
            scale = getattr(self, 'current_ipa_scale',
                            getattr(self, '_ipa_scale', 0.7))
            try:
                dst.set_ip_adapter_scale(scale)
            except Exception as e:
                logger.warning(f"  ⚠️ 设置 controlnet IPA scale 失败: {e}")

            # ── 4. 校验共享 UNet 上 IPA processor 是否就位 ──
            ok_count = err_count = 0
            unet = getattr(dst, "unet", None)
            if unet is not None:
                for name, proc in unet.attn_processors.items():
                    if 'attn2' in name:
                        if 'IPAdapter' in type(proc).__name__:
                            ok_count += 1
                        else:
                            err_count += 1

            if ok_count > 0 and err_count == 0:
                logger.info(f"✅ controlnet_pipe IPA 就绪（共享 UNet，"
                            f"{ok_count} 个 IPA processor，未重复安装权重）")
                return True

            logger.warning(f"⚠️ controlnet_pipe IPA 校验异常: "
                           f"{ok_count} 正确 / {err_count} 错误"
                           f"（共享 UNet={shared}）")
            return False

        except Exception as e:
            logger.error(f"❌ IPA 同步到 controlnet 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _enable_memory_efficient(self, pipe, name="pipe"):
        """统一启用 VAE 分块解码 + 切片，省显存出大图。

        走 VRAMManager 的能力探测接口（diffusers 0.40 的 Pipeline 上
        没有 enable_vae_tiling/slicing，只有 pipe.vae 上才有）。
        """
        if pipe is None:
            return
        from utils.vram_manager import VRAMManager
        if VRAMManager._set_vae_option(pipe, "tiling", True):
            logger.info(f"  ✅ [{name}] VAE Tiling 已启用（支持大图）")
        else:
            logger.warning(f"  ⚠️ [{name}] VAE Tiling 启用失败")
        if VRAMManager._set_vae_option(pipe, "slicing", True):
            logger.info(f"  ✅ [{name}] VAE Slicing 已启用（省显存）")
        else:
            logger.warning(f"  ⚠️ [{name}] VAE Slicing 启用失败")

    def prepare_reference_only(self):
        """加载 Reference-Only 管线（复用现有组件，不额外下载权重）。

        ⚠️ diffusers 0.40 起 `StableDiffusionReferencePipeline` 已从主包移除
        （它一直是 community pipeline）。以前 `from diffusers import ...`
        会直接 ImportError，导致"Reference-Only 生成"必崩。
        这里改为按 community pipeline 加载，并**复用主管线组件**
        （不重新加载权重，省显存）。
        """
        from diffusers import DiffusionPipeline

        if self.txt2img_pipe is None:
            raise RuntimeError("底模尚未加载，无法准备 Reference-Only 管线")

        last_err = None
        for local_only in (True, False):     # 先查本地，缺失才联网
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    custom_pipeline="stable_diffusion_reference",
                    unet=self.txt2img_pipe.unet,
                    vae=self.txt2img_pipe.vae,
                    text_encoder=self.txt2img_pipe.text_encoder,
                    tokenizer=self.txt2img_pipe.tokenizer,
                    scheduler=self.txt2img_pipe.scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                    local_files_only=local_only,
                )
                # 组件是共享的（可能已被 enable_model_cpu_offload 挂上 hook），
                # 所以这里不能再 to(device)，否则会与 offload hook 冲突。
                self.reference_pipe = pipe
                logger.info("✅ Reference-Only 管线就绪"
                            f"（community pipeline, local_only={local_only}）")
                return
            except Exception as e:
                last_err = e
                logger.debug(f"Reference-Only local_only={local_only} 失败: {e}")

        raise RuntimeError(
            "❌ Reference-Only 管线加载失败（需要联网获取 community pipeline）\n"
            f"原因: {last_err}\n"
            "💡 可手动准备后离线使用：\n"
            "   huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \\\n"
            "     --include '*reference*' --local-dir models_cache/huggingface/hub\n"
            "   或改用 IP-Adapter / Pose Transfer 做角色一致性。")


    def _force_fp32_vae(self, pipe):
        if hasattr(pipe, "_all_hooks") and pipe._all_hooks:
            logger.info("⏭️ 已启用 CPU offload，跳过 VAE fp32 patch")
            return

        vae = pipe.vae
        if getattr(vae, '_fp32_patched', False):
            return

        _orig_decode = vae.decode
        _orig_dtype  = next(vae.parameters()).dtype  # 记住原始 dtype（通常 fp16）

        def _decode_fp32(z, *a, **k):
            vae.to(dtype=torch.float32)
            try:
                result = _orig_decode(z.to(torch.float32), *a, **k)
            finally:
                vae.to(dtype=_orig_dtype)   # 解码后立即还原，不影响 encode
            return result

        vae.decode = _decode_fp32
        vae._fp32_patched = True
        logger.info(f"🛡️ VAE decode 临时升 fp32（encode 保持 {_orig_dtype}）")
