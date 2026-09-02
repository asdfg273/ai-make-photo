# model_manager.py
# ============================================================
#  模型管理器 — 多 Pipeline / LoRA / ControlNet / Sampler
# ============================================================

import os
import gc
import torch
import cv2
import numpy as np

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)

from controlnet_aux import OpenposeDetector
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image

from utils.system_utils import SingletonMeta
from utils import paths 
import logging
from core import arch as arch_mod
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
        self.current_lora_name  = None
        self.is_sdxl            = False

        self.txt2img_pipe       = None
        self.img2img_pipe       = None
        self.inpaint_pipe       = None
        self.controlnet_pipe    = None
        self.detected_arch      = None
        self.current_arch_id    = None
        self.pose_detector      = None
        self.depth_estimator    = None
        self.loaded_controlnets = {}
        self.current_cn_type    = None
        # ★ IP-Adapter 状态
        self.ip_adapter_loaded   = False
        self.ip_adapter_variant  = None
        self.is_sd3  = False
        self.is_flux = False
        self.current_ipa_scale = 0.6

        self._compel_cache = {}
        self._model_cache  = {}

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
                     'inpaint_pipe', 'controlnet_pipe'):
            setattr(self, attr, None)
        self.loaded_controlnets.clear()
        self._compel_cache.clear()

        # 状态必须跟着一起清
        self.current_model_name  = None
        self.current_lora_name   = None
        self.current_cn_type     = None
        self.ip_adapter_loaded   = False
        self.ip_adapter_variant  = None
        self._ipa_loaded         = False
        self._ipa_variant        = None

        gc.collect()
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

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

        # VAE 分块解码 (大图关键)
        try:
            pipe.enable_vae_tiling()
            applied.append("vae_tiling")
        except Exception:
            pass

        # VAE 按 batch 切片
        try:
            pipe.enable_vae_slicing()
            applied.append("vae_slicing")
        except Exception:
            pass

        # xformers (装了才用，PyTorch 2.x SDPA 也已是默认快通道)
        try:
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

    def detect_arch_from_checkpoint(model_path):
        """
        读 safetensors 头部的键名判断架构，返回 'sdxl'/'sd15'/'sd3'/'flux'/None。
        只读元数据不载权重，失败返回 None 交给调用方回退启发式。
        """
        if not model_path.lower().endswith(".safetensors"):
            return None
        try:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt") as f:
                keys = list(f.keys())
        except Exception as e:
            logger.warning(f"⚠️ 读取 checkpoint 头部失败: {e}")
            return None

        joined = "\n".join(keys)
        if "double_blocks." in joined or "model.diffusion_model.double_blocks." in joined:
            return "flux"
        if "joint_blocks." in joined:
            return "sd3"
        # SDXL 的第二个文本编码器 (OpenCLIP-G)
        if "conditioner.embedders.1." in joined or "text_encoder_2." in joined:
            return "sdxl"
        if "cond_stage_model.transformer." in joined or "text_encoder." in joined:
            return "sd15"
        return None

    def _is_flow_matching_pipe(self) -> bool:
        """当前底模是否为 flow-matching 架构（自带原生 scheduler，不可换 SD 系采样器）。"""
        pipe = getattr(self, "txt2img_pipe", None)
        sched = getattr(pipe, "scheduler", None)
        if sched is None:
            return False
        cls_name = type(sched).__name__
        return "FlowMatch" in cls_name or "RectifiedFlow" in cls_name

    def load_model(self, model_name):
        # 同模型跳过
        if (self.current_model_name == model_name
                and self.txt2img_pipe is not None):
            logger.info("⚡ 模型未改变，跳过加载。")
            return

        # === 智能查找: 支持 models/ 根目录 + 所有架构子目录 ===
        MODELS_ROOT = paths.MODEL_DIR
        _SUBDIRS = [
            "sd15", "sdxl", "sd3", "pony", "flux",
            "anima",       
            # 以后新架构目录也加到这
        ]

        candidates = [os.path.join(MODELS_ROOT, model_name)] + [
            os.path.join(MODELS_ROOT, sub, model_name) for sub in _SUBDIRS
        ]

        # 若 model_name 不带扩展名，尝试自动补全
        _exts = [".safetensors", ".ckpt", ".pt"]
        _got = None
        for c in candidates:
            if os.path.isfile(c):
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
            for root, _, files in os.walk(MODELS_ROOT):
                for f in files:
                    if f == model_name or f == model_name + ".safetensors":
                        model_path = os.path.join(root, f)
                        break
                if model_path:
                    break

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
                extra = (f"\n需额外下载: {', '.join(det.info.extra_components)}"
                         f" ({det.info.extra_download_size or '大小未知'})")
            raise ValueError(f"无法加载「{model_name}」："
                             f"[{det.info.display_name}] "
                             f"{det.info.unsupported_reason}{extra}")

        arch           = det.arch_id
        file_size_gb   = det.size_gb
        self.is_flux   = (arch == "flux")
        self.is_sd3    = (arch == "sd3")
        self.is_sdxl   = (arch == "sdxl")
        model_type     = det.info.display_name

       
        from core.loaders import get_loader
        from core.loaders.base import LoadContext

        ctx = LoadContext(
            dtype=self.dtype, model_name=model_name,
            arch_id=det.arch_id, info=det.info,
            detection=det, manager=self,
        )

        try:
            result = get_loader(det.arch_id).load(model_path, ctx)

            self.txt2img_pipe = result.txt2img
            self.img2img_pipe = result.img2img
            self.inpaint_pipe = result.inpaint
            self.controlnet_pipe = None
            self.current_cn_type  = None
            self.current_model_name = model_name
            self.current_lora_name  = None
            self._compel_cache.clear()
            self.compel_txt2img = self.compel_img2img = self.compel_controlnet = None
            logger.info(f"✅ {det.info.display_name} 加载与显存优化完成！")
        except Exception as e:
            import traceback; traceback.print_exc()
            raise Exception(f"模型加载失败: {str(e)}")

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

    def apply_multiple_loras(self, lora_list, sub_dir=None):
        """
        现代版多 LoRA 加载机制 (极速秒切 + 独立权重)
        lora_list 格式: [("lora1.safetensors", 0.8), ("lora2.safetensors", 0.5)]
        sub_dir=None 时按当前底模架构自动推断，避免调用方传错命名。
        """
        try:
            self.txt2img_pipe.unload_lora_weights()
            logger.info("🧹 已清空旧的 LoRA 缓存。")
        except Exception:
            pass

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

        for i, (lora_name, weight) in enumerate(lora_list):
            lora_path = os.path.join(lora_root, lora_name)
            if not os.path.isfile(lora_path):
                logger.error(f"❌ [跳过] LoRA 文件不存在: {lora_path}")
                continue

            adapter_name = f"lora_slot_{i}"
            try:
                self.txt2img_pipe.load_lora_weights(
                    lora_root,
                    weight_name=lora_name,
                    adapter_name=adapter_name,
                )
                adapter_names.append(adapter_name)
                adapter_weights.append(weight)
                logger.info(f"✅ 挂载插件: {lora_name} (独立权重: {weight})")
            except Exception as e:
                logger.error(f"❌ [跳过] 插件 {lora_name} 不兼容或损坏: {e}")

        if adapter_names:
            try:
                self.txt2img_pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                logger.info(f"⚡ 已激活插件通道: {adapter_names}，对应权重: {adapter_weights}")
            except Exception as e:
                logger.warning(f"⚠️ 激活多 LoRA 权重时异常: {e}")
        else:
            logger.warning("⚠️ 没有任何 LoRA 成功挂载")

    # ------------------------------------------------------------
    #  ControlNet
    # ------------------------------------------------------------
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

        # 已加载同类型 → 仍需检查 image_encoder 是否需要同步
        if (self.current_cn_type == control_type
                and getattr(self, 'controlnet_pipe', None) is not None):
            # 🔧 即便复用,也要确保 image_encoder 同步 (IP-Adapter 可能后加载)
            self._sync_ipa_components_to_controlnet()
            return

        logger.info(f"🔄 正在配置 ControlNet: {control_type} "
              f"(原始输入: {raw_type})...")

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

        if control_type not in self.loaded_controlnets:
            local_dir = os.path.abspath(
                f"{paths.CONTROLNET_DIR}/{'sdxl_' if getattr(self, 'is_sdxl', False) else ''}{control_type}"

            )
            if os.path.exists(os.path.join(local_dir, "config.json")):
                logger.info(f"📂 从本地加载 ControlNet: {local_dir}")
                cn_source = local_dir
                local_only = True
            else:
                cn_model_id = use_map[control_type]
                logger.info(f"🌐 从在线下载 ControlNet: {cn_model_id} "
                      f"(via {os.environ.get('HF_ENDPOINT', 'huggingface.co')})")
                cn_source = cn_model_id
                local_only = False

            try:
                self.loaded_controlnets[control_type] = \
                    ControlNetModel.from_pretrained(
                        cn_source,
                        torch_dtype=self.dtype,
                        local_files_only=local_only,
                    ).to(self.device)
            except Exception as e:
                raise RuntimeError(
                    f"❌ ControlNet 加载失败: {e}\n\n"
                    f"💡 解决办法:\n"
                    f"   1. 检查网络是否通畅,镜像 https://hf-mirror.com 能否访问\n"
                    f"   2. 或手动下载模型放到: {local_dir}\n"
                    f"      只需 config.json + diffusion_pytorch_model.safetensors\n"
                    f"   3. 或在 UI 中关闭 ControlNet,改用普通图生图"
                ) from e

        controlnet = self.loaded_controlnets[control_type]

        pipe_class = (StableDiffusionXLControlNetPipeline
                      if getattr(self, 'is_sdxl', False)
                      else StableDiffusionControlNetPipeline)

        common_kwargs = dict(
            vae          = self.txt2img_pipe.vae,
            text_encoder = self.txt2img_pipe.text_encoder,
            tokenizer    = self.txt2img_pipe.tokenizer,
            unet         = self.txt2img_pipe.unet,
            scheduler    = self.txt2img_pipe.scheduler,
            controlnet   = controlnet,
        )

        if getattr(self, 'is_sdxl', False):
            self.controlnet_pipe = pipe_class(
                **common_kwargs,
                text_encoder_2 = self.txt2img_pipe.text_encoder_2,
                tokenizer_2    = self.txt2img_pipe.tokenizer_2,
            ).to(self.device)
        else:
            self.controlnet_pipe = pipe_class(
                **common_kwargs,
                safety_checker          = None,
                feature_extractor       = None,
                requires_safety_checker = False,
            ).to(self.device)

        # 轻量优化
        self._apply_light_optimizations(self.controlnet_pipe, name="controlnet")

        self.current_cn_type = control_type
        logger.info(f"✅ ControlNet ({control_type}) 加载完成")

        self._sync_ipa_components_to_controlnet()

        # ===== 预处理器(检测器) =====
        if control_type == "openpose" and not self.pose_detector:
            logger.info("⏳ 正在加载 OpenPose 骨架提取器...")
            local_annot = os.path.join(paths.CONTROLNET_DIR, "Annotators")
            if os.path.exists(os.path.join(local_annot, "body_pose_model.pth")):
                logger.info(f"📂 从本地加载 OpenPose 检测器: {local_annot}")
                try:
                    self.pose_detector = OpenposeDetector.from_pretrained(local_annot)
                except Exception as e:
                    logger.warning(f"⚠️ 本地 OpenPose 加载失败: {e}")
            else:
                try:
                    self.pose_detector = OpenposeDetector.from_pretrained(
                        "lllyasviel/Annotators")
                except Exception as e:
                    logger.warning(f"⚠️ 加载 OpenPose 失败: {e}")

        elif control_type == "depth" and not self.depth_estimator:
            logger.info("⏳ 正在加载 Depth 深度图提取器...")
            try:
                from transformers import pipeline
                self.depth_estimator = pipeline('depth-estimation')
            except Exception as e:
                logger.warning(f"⚠️ 加载 Depth 失败: {e}")

    def _sync_ipa_components_to_controlnet(self):
        """
        把 txt2img_pipe 上的 IP-Adapter 组件 (image_encoder/feature_extractor)
        同步给 controlnet_pipe。
    
        IP-Adapter 加载在 txt2img_pipe 上时,会注入 image_encoder。
        controlnet_pipe 共享了 unet (含 attn processor),但 image_encoder 是
        pipeline 级别的属性,不会自动共享 → 需要手动赋值。
        """
        if getattr(self, 'controlnet_pipe', None) is None:
            return
        if getattr(self, 'txt2img_pipe', None) is None:
            return

        src = self.txt2img_pipe
        dst = self.controlnet_pipe

        # 1. image_encoder (IP-Adapter 的视觉编码器)
        src_img_enc = getattr(src, 'image_encoder', None)
        if src_img_enc is not None:
            dst.image_encoder = src_img_enc
            logger.info("🔧 [sync] image_encoder → controlnet_pipe")

        # 2. feature_extractor (CLIP 图像预处理器)
        src_feat = getattr(src, 'feature_extractor', None)
        if src_feat is not None:
            dst.feature_extractor = src_feat
            logger.info("🔧 [sync] feature_extractor → controlnet_pipe")

        # 3. 同步 IP-Adapter scale (UNet 共享,所以会自动生效,这里只为保险)
        try:
            if getattr(self, 'ip_adapter_loaded', False):
                current_scale = getattr(self, 'current_ipa_scale', 0.6)
                dst.set_ip_adapter_scale(current_scale)
                logger.info(f"🔧 [sync] IPA scale={current_scale} → controlnet_pipe")
        except Exception as e:
            logger.warning(f"⚠️ [sync] 设置 controlnet_pipe IPA scale 失败: {e}")


    def get_control_image(self, input_image, control_type="openpose"):
        control_type = str(control_type).strip().lower()
        if self.current_cn_type != control_type:
            self.prepare_controlnet(control_type)

        if control_type == "openpose":
            return self.pose_detector(input_image)

        elif control_type == "canny":
            image = np.array(input_image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            return Image.fromarray(image)

        elif control_type == "depth":
            return self.depth_estimator(input_image)['depth']

        return input_image

    def prepare_ip_adapter(self, variant="plus"):
        """
        加载 IP-Adapter 到 txt2img / img2img 管线
        🔧 inpaint_pipe 不装载 IPA,避免 ADetailer 局部重绘冲突
           (ADetailer 只需修手/脸细节,不需要参考角色特征)
        """
        if getattr(self, 'ip_adapter_loaded', False) \
           and getattr(self, 'ip_adapter_variant', None) == variant:
            logger.info("✅ IP-Adapter 已加载,复用")
            return True

        weight_name = (
            "ip-adapter-plus_sd15.safetensors"
            if variant == "plus"
            else "ip-adapter_sd15.safetensors"
        )
        logger.info(f"🎭 加载 IP-Adapter ({variant}) ...")

        pipes = []
        for attr in ("txt2img_pipe", "img2img_pipe"):   
            pipe = getattr(self, attr, None)
            if pipe is not None:
                pipes.append((attr, pipe))

        # ── 重置 attn processor (只对装载 IPA 的 pipe) ──
        logger.info("  → 重置 attn processor (关闭 attention slicing) ...")
        try:
            from diffusers.models.attention_processor import (
                AttnProcessor2_0, AttnProcessor
            )
            import torch.nn.functional as F
            proc_class = (
                AttnProcessor2_0
                if hasattr(F, 'scaled_dot_product_attention')
                else AttnProcessor
            )
            for attr, pipe in pipes:
                try:
                    if hasattr(pipe, 'disable_attention_slicing'):
                        pipe.disable_attention_slicing()
                except Exception:
                    pass
                try:
                    pipe.unet.set_attn_processor(proc_class())
                except Exception as e:
                    logger.warning(f"    ⚠️ {attr} 重置 processor 失败: {e}")
        except Exception as e:
            logger.warning(f"  ⚠️ 重置 processor 异常: {e}")

        # ── 装载 IP-Adapter (仅 txt2img / img2img) ──
        try:
            for attr, pipe in pipes:
                logger.info(f"  → 装载 IP-Adapter 到 {attr} ...")
                pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name=weight_name,
                )

            # 🆕 inpaint_pipe 只同步 encoder/feature_extractor,不装载 IPA
            # 这样它能正常推理,且 ADetailer 不会触发 IPA 冲突
            inpaint_pipe = getattr(self, 'inpaint_pipe', None)
            if inpaint_pipe is not None:
                try:
                    # 同步基础组件 (复用显存)
                    if getattr(self.txt2img_pipe, 'image_encoder', None) is not None:
                        inpaint_pipe.image_encoder = self.txt2img_pipe.image_encoder
                    if getattr(self.txt2img_pipe, 'feature_extractor', None) is not None:
                        inpaint_pipe.feature_extractor = self.txt2img_pipe.feature_extractor

                    from diffusers.models.attention_processor import (
                        AttnProcessor2_0, AttnProcessor
                    )
                    import torch.nn.functional as F
                    proc_class = (
                        AttnProcessor2_0
                        if hasattr(F, 'scaled_dot_product_attention')
                        else AttnProcessor
                    )

                    # 检查 inpaint 是否和 txt2img 共享 UNet
                    shares_unet = (inpaint_pipe.unet is self.txt2img_pipe.unet)
                    if shares_unet:
                        logger.warning("  ⚠️ inpaint_pipe 与 txt2img 共享 UNet,"
                              "ADetailer 时需在调用层处理 IPA 占位")
                    else:
                        # 独立 UNet,直接重置回干净的 processor
                        inpaint_pipe.unet.set_attn_processor(proc_class())
                        logger.info("  ✅ inpaint_pipe 已重置为标准 attn_processor "
                              "(无 IPA 干扰)")
                except Exception as e:
                    logger.warning(f"  ⚠️ inpaint_pipe 同步失败: {e}")

            # 🆕 设置所有状态标志(统一两套变量名)
            self.ip_adapter_loaded  = True
            self.ip_adapter_variant = variant
            self._ipa_loaded        = True
            self._ipa_variant       = variant
            self._ipa_scale         = 0.7

            logger.info("✅ IP-Adapter 加载完成 (inpaint_pipe 已排除)")

            # 🆕 如果 controlnet_pipe 已存在,自动同步
            if getattr(self, 'controlnet_pipe', None) is not None:
                try:
                    self.sync_ipa_to_controlnet()
                except Exception as e:
                    logger.warning(f"⚠️ 自动同步 IPA 到 controlnet 失败: {e}")

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
            return False


    def unload_ip_adapter(self):
        """彻底卸载 IP-Adapter，恢复 UNet 到普通模式（覆盖全部 4 个 pipeline）"""
        logger.info("🧹 卸载 IP-Adapter ...")
        pipes = [
            self.txt2img_pipe,
            self.img2img_pipe,
            self.inpaint_pipe,
            self.controlnet_pipe,
        ]
        for pipe in pipes:
            if pipe is None:
                continue
            try:
                # 方式1: diffusers 0.31+ 提供的官方接口
                if hasattr(pipe, 'unload_ip_adapter'):
                    pipe.unload_ip_adapter()
                else:
                    # 方式2: 手动恢复 attn processor（兼容旧版 diffusers）
                    from diffusers.models.attention_processor import AttnProcessor2_0, AttnProcessor
                    proc_class = AttnProcessor2_0 if hasattr(
                        __import__('torch.nn.functional', fromlist=['scaled_dot_product_attention']),
                        'scaled_dot_product_attention') else AttnProcessor
                    pipe.unet.set_attn_processor(proc_class())
                    pipe.unet.encoder_hid_proj = None
                    pipe.unet.config.encoder_hid_dim_type = None
            except Exception as e:
                logger.warning(f"  ⚠️ 卸载异常: {e}")

        self.ip_adapter_loaded  = False
        self.ip_adapter_variant = None
        self._ipa_loaded        = False
        self._ipa_variant       = None
        logger.info("✅ IP-Adapter 卸载完成")


    def set_ip_adapter_scale(self, scale: float = 0.6):
        """调整 IP-Adapter 影响强度 0.0~1.5（覆盖全部 4 个 pipeline）"""
        if not self.ip_adapter_loaded:
            return
        self._ipa_scale = float(scale)
        scale = max(0.0, min(1.5, float(scale)))
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

        if hasattr(pipe, "text_encoder") and pipe.text_encoder:
            pipe.text_encoder.to(self.device)
        if (self.is_sdxl
                and hasattr(pipe, "text_encoder_2")
                and pipe.text_encoder_2):
            pipe.text_encoder_2.to(self.device)

        cache_key = (prompt, negative_prompt, self.current_model_name,
             getattr(self, 'current_arch_id', None))
        if cache_key in self._compel_cache:
            return dict(self._compel_cache[cache_key]) 

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
            prompt_embeds, pooled     = compel(prompt)
            neg_embeds,    neg_pooled = compel(negative_prompt)
            prompt_embeds, neg_embeds = \
                compel.pad_conditioning_tensors_to_same_length(
                    [prompt_embeds, neg_embeds])

            result = {
                "prompt_embeds":                prompt_embeds,
                "pooled_prompt_embeds":         pooled,
                "negative_prompt_embeds":       neg_embeds,
                "negative_pooled_prompt_embeds":neg_pooled,
            }
        else:
            compel = Compel(
                tokenizer    = pipe.tokenizer,
                text_encoder = pipe.text_encoder,
                device       = self.device,
                truncate_long_prompts = False,
            )
            prompt_embeds = compel(prompt)
            neg_embeds    = compel(negative_prompt)
            prompt_embeds, neg_embeds = \
                compel.pad_conditioning_tensors_to_same_length(
                    [prompt_embeds, neg_embeds])
            result = {
                "prompt_embeds":          prompt_embeds,
                "negative_prompt_embeds": neg_embeds,
            }

        self._compel_cache[cache_key] = dict(result)    
        if len(self._compel_cache) > 16:
            oldest = next(iter(self._compel_cache))
            del self._compel_cache[oldest]

        return result

    # ------------------------------------------------------------
    #  采样器切换
    # ------------------------------------------------------------
    def switch_sampler(self, sampler_name):
        """根据 UI 传来的名称，切换底层的扩散调度器 (Sampler)"""
        if self._is_flow_matching_pipe():
            logger.info(
                f"⏭️ 跳过采样器切换（flow-matching 底模保持原生调度器: "
                f"{type(self.txt2img_pipe.scheduler).__name__}）"
            )
            return
        if (not hasattr(self, 'txt2img_pipe')
                or self.txt2img_pipe is None):
            return

        config = dict(self.txt2img_pipe.scheduler.config)
        if getattr(self, 'is_vpred', False):
            config["prediction_type"] = "v_prediction"
            config["rescale_betas_zero_snr"] = True

        try:
            if "欧拉A" in sampler_name or "Euler a" in sampler_name:
                new_scheduler = EulerAncestralDiscreteScheduler.from_config(config)
            elif "欧拉" in sampler_name or "Euler" in sampler_name:
                new_scheduler = EulerDiscreteScheduler.from_config(config)
            elif "DPM++ 2M" in sampler_name:
                new_scheduler = DPMSolverMultistepScheduler.from_config(
                    config, use_karras_sigmas=True)
            elif "DDIM" in sampler_name:
                new_scheduler = DDIMScheduler.from_config(config)
            else:
                # 默认保底
                new_scheduler = EulerAncestralDiscreteScheduler.from_config(config)

            self.txt2img_pipe.scheduler = new_scheduler
            if getattr(self, 'img2img_pipe', None) is not None:
                self.img2img_pipe.scheduler = new_scheduler
            if getattr(self, 'inpaint_pipe', None) is not None:
                self.inpaint_pipe.scheduler = new_scheduler
            if getattr(self, 'controlnet_pipe', None) is not None:
                self.controlnet_pipe.scheduler = new_scheduler

            logger.info(f"🔄 采样器已切换为 -> {sampler_name}")

        except Exception as e:
            logger.warning(f"⚠️ 切换采样器失败: {e}，将使用原默认采样器")

    def _normalize_device(self, raw) -> str:
        import re, torch
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

        # 释放管线
        for attr in ['txt2img_pipe', 'img2img_pipe', 'inpaint_pipe',
                     'controlnet_pipe']:
            pipe = getattr(self, attr, None)
            if pipe is not None:
                try:
                    # 解绑组件,加速 GC
                    for sub in ['unet', 'vae', 'text_encoder',
                                'text_encoder_2', 'tokenizer',
                                'tokenizer_2', 'scheduler', 'safety_checker',
                                'feature_extractor']:
                        if hasattr(pipe, sub):
                            try:
                                setattr(pipe, sub, None)
                            except Exception:
                                pass
                    del pipe
                except Exception:
                    pass
                setattr(self, attr, None)

        # 释放 ControlNet
        if hasattr(self, 'loaded_controlnets'):
            for k in list(self.loaded_controlnets.keys()):
                try:
                    del self.loaded_controlnets[k]
                except Exception:
                    pass
            self.loaded_controlnets = {}

        # 释放检测器
        for attr in ['pose_detector', 'depth_estimator']:
            if getattr(self, attr, None) is not None:
                try:
                    del self.__dict__[attr]
                except Exception:
                    pass
                setattr(self, attr, None)

        # 释放 IP-Adapter 参考图
        if hasattr(self, 'ipa_ref_image'):
            self.ipa_ref_image = None

        # 释放缓存
        if hasattr(self, '_compel_cache'):
            self._compel_cache = {}
        if hasattr(self, '_model_cache'):
            self._model_cache = {}

        self.current_model_name = None
        self.current_lora_name = None
        self.current_cn_type = None

        # 强制 GC
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        logger.info("✅ 所有模型已释放")

    def sync_ipa_to_controlnet(self):
        """
        给 controlnet_pipe 独立装载 IP-Adapter（不依赖共享 UNet）。
        """
        if self.controlnet_pipe is None:
            logger.warning("⚠️ controlnet_pipe 未初始化")
            return False

        if self.txt2img_pipe is None or not getattr(self, '_ipa_loaded', False):
            logger.warning("⚠️ txt2img_pipe 未加载 IPA，无法同步")
            return False

        logger.info("🔄 给 controlnet_pipe 独立装载 IP-Adapter...")

        try:
            # ── 1. 关闭 attention slicing（IPA 不兼容） ──
            try:
                self.controlnet_pipe.disable_attention_slicing()
                logger.info("  → 关闭 controlnet_pipe attention_slicing")
            except Exception:
                pass

            # ── 2. 共享 image_encoder 和 feature_extractor（节省显存） ──
            if hasattr(self.txt2img_pipe, 'image_encoder') and self.txt2img_pipe.image_encoder is not None:
                self.controlnet_pipe.image_encoder = self.txt2img_pipe.image_encoder
                logger.info("  → 共享 image_encoder")

            if hasattr(self.txt2img_pipe, 'feature_extractor') and self.txt2img_pipe.feature_extractor is not None:
                self.controlnet_pipe.feature_extractor = self.txt2img_pipe.feature_extractor
                logger.info("  → 共享 feature_extractor")

            # ── 3. 关键：独立装载 IP-Adapter 到 controlnet_pipe ──
            variant = getattr(self, '_ipa_variant', 'plus')
            weight_map = {
                'plus':      'ip-adapter-plus_sd15.safetensors',
                'plus-face': 'ip-adapter-plus-face_sd15.safetensors',
                'base':      'ip-adapter_sd15.safetensors',
            }
            weight_name = weight_map.get(variant, 'ip-adapter-plus_sd15.safetensors')

            # 优先用本地缓存
            local_dir = os.path.join(paths.CACHE_DIR, "ip_adapter")
            if os.path.exists(os.path.join(local_dir, "models", weight_name)):
                logger.info(f"  → 从本地装载: {weight_name}")
                self.controlnet_pipe.load_ip_adapter(
                    local_dir,
                    subfolder="models",
                    weight_name=weight_name,
                )
            else:
                logger.info(f"  → 从 HuggingFace 装载: {weight_name}")
                self.controlnet_pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name=weight_name,
                )

            # ── 4. 设置 scale ──
            scale = getattr(self, '_ipa_scale', 0.7)
            self.controlnet_pipe.set_ip_adapter_scale(scale)
            logger.info(f"  → set scale = {scale}")

            # ── 5. 验证 ──
            ok_count = 0
            err_count = 0
            for name, proc in self.controlnet_pipe.unet.attn_processors.items():
                if 'attn2' in name:
                    if 'IPAdapter' in type(proc).__name__:
                        ok_count += 1
                    else:
                        err_count += 1

            logger.warning(f"✅ ControlNet IPA 装载完成: {ok_count} 正确 / {err_count} 错误")
            return ok_count > 0 and err_count == 0

        except Exception as e:
            logger.error(f"❌ IPA 装载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _enable_memory_efficient(self, pipe, name="pipe"):
        """统一启用 VAE 分块解码 + 切片，省显存出大图"""
        if pipe is None:
            return
        try:
            pipe.enable_vae_tiling()
            logger.info(f"  ✅ [{name}] VAE Tiling 已启用（支持大图）")
        except Exception as e:
            logger.warning(f"  ⚠️ [{name}] VAE Tiling 启用失败: {e}")
        try:
            pipe.enable_vae_slicing()
            logger.info(f"  ✅ [{name}] VAE Slicing 已启用（省显存）")
        except Exception as e:
            logger.warning(f"  ⚠️ [{name}] VAE Slicing 启用失败: {e}")
        # xformers 加速（如果装了）
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info(f"  ✅ [{name}] xformers 加速已启用")
        except Exception:
            pass  # 没装就算了

    def prepare_reference_only(self):
        """加载 Reference-Only ControlNet (无需额外模型)"""
        # Reference-Only 不需要下载额外模型
        # 它通过修改 attention 实现,集成在 diffusers 的 community pipeline 里
    
        from diffusers import StableDiffusionReferencePipeline
    
        self.reference_pipe = StableDiffusionReferencePipeline(
            unet=self.txt2img_pipe.unet,
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        self.reference_pipe.to(self.device)


    def _force_fp32_vae(self, pipe):
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